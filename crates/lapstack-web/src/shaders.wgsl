// lapstack WebGPU kernels. Same maths as lapstack-core/src/pyramid.rs + fuse.rs
// (and the CUDA twins in gpu.rs): binomial [1 4 6 4 1]/16 taps, reflect-101
// borders, luma energy, binomial window, winner-take-all select.
//
// Every level's three colour planes live in ONE buffer, plane c at offset
// c * w * h (keeps the storage-buffer count per dispatch small). Kernels take a
// uniform `P` with the geometry they need; plane offsets are passed explicitly.

struct P {
    w: u32,      // input / fine width
    h: u32,      // input / fine height
    ow: u32,     // output / coarse width
    oh: u32,     // output / coarse height
    off_in: u32, // plane offset into `a` (elements)
    off_out: u32,// plane offset into `b` / `o` (elements)
    klen: u32,   // window length (odd) for win*, or misc int
    flag: u32,   // record-winner flag / energy mode
    f0: f32,     // scalar: frame index for select, edge scale for proxy
    f1: f32,
    f2: f32,
    f3: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read_write> e: array<f32>;
@group(0) @binding(5) var<storage, read> wt: array<f32>;
@group(0) @binding(6) var<storage, read> u: array<u32>;

// 1-D kernels are dispatched as (gx, gy) workgroups of 256 so they can cover
// more than 65535*256 elements; the flat index is x + y * (gx * 256).
fn gid1(g: vec3<u32>, nwg: vec3<u32>) -> u32 { return g.x + g.y * nwg.x * 256u; }

const K0: f32 = 1.0 / 16.0;
const K1: f32 = 4.0 / 16.0;
const K2: f32 = 6.0 / 16.0;
const E0: f32 = 2.0 * K0;
const E1: f32 = 2.0 * K2;
const OD: f32 = 2.0 * K1;

fn refl(i0: i32, n: i32) -> i32 {
    if (n == 1) { return 0; }
    var i = i0;
    loop {
        if (i < 0) { i = -i; } else if (i >= n) { i = 2 * (n - 1) - i; } else { break; }
    }
    return i;
}

// ---- REDUCE: a[off_in] (w x h) -> b (h x ow) -> o[off_out] (oh x ow) ----
@compute @workgroup_size(16, 16)
fn red_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let oj = g.x; let y = g.y;
    if (oj >= p.ow || y >= p.h) { return; }
    let w = i32(p.w); let c = 2 * i32(oj); let base = p.off_in + y * p.w;
    b[y * p.ow + oj] = K0 * a[base + u32(refl(c - 2, w))] + K1 * a[base + u32(refl(c - 1, w))]
        + K2 * a[base + u32(refl(c, w))] + K1 * a[base + u32(refl(c + 1, w))] + K0 * a[base + u32(refl(c + 2, w))];
}
@compute @workgroup_size(16, 16)
fn red_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let oj = g.x; let oi = g.y;
    if (oj >= p.ow || oi >= p.oh) { return; }
    let h = i32(p.h); let c = 2 * i32(oi);
    o[p.off_out + oi * p.ow + oj] = K0 * b[u32(refl(c - 2, h)) * p.ow + oj] + K1 * b[u32(refl(c - 1, h)) * p.ow + oj]
        + K2 * b[u32(refl(c, h)) * p.ow + oj] + K1 * b[u32(refl(c + 1, h)) * p.ow + oj] + K0 * b[u32(refl(c + 2, h)) * p.ow + oj];
}
// ---- EXPAND: a[off_in] coarse (ow x oh here = cw x ch) -> b (ch x w) -> o (h x w) ----
// exp_h: p.w = fine width, p.ow = cw, p.oh = ch
@compute @workgroup_size(16, 16)
fn exp_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.oh) { return; }
    let cw = i32(p.ow); let i = i32(x / 2u); let base = p.off_in + y * p.ow;
    var v: f32;
    if ((x & 1u) == 0u) {
        v = E0 * a[base + u32(refl(i - 1, cw))] + E1 * a[base + u32(refl(i, cw))] + E0 * a[base + u32(refl(i + 1, cw))];
    } else {
        v = OD * (a[base + u32(refl(i, cw))] + a[base + u32(refl(i + 1, cw))]);
    }
    b[y * p.w + x] = v;
}
// exp_v: p.w = fine width, p.h = fine height, p.oh = ch; then o[off_out] = fine (+/-) result
// flag 0: o = expanded (plain), 1: o -= expanded (Laplacian), 2: o += expanded (collapse)
@compute @workgroup_size(16, 16)
fn exp_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let ch = i32(p.oh); let i = i32(y / 2u);
    var v: f32;
    if ((y & 1u) == 0u) {
        v = E0 * b[u32(refl(i - 1, ch)) * p.w + x] + E1 * b[u32(refl(i, ch)) * p.w + x] + E0 * b[u32(refl(i + 1, ch)) * p.w + x];
    } else {
        v = OD * (b[u32(refl(i, ch)) * p.w + x] + b[u32(refl(i + 1, ch)) * p.w + x]);
    }
    let idx = p.off_out + y * p.w + x;
    if (p.flag == 1u) { o[idx] = o[idx] - v; } else if (p.flag == 2u) { o[idx] = o[idx] + v; } else { o[idx] = v; }
}
// ---- energy of a 3-plane level in `a` (planes at 0, n, 2n): e = Y^2 (flag 0) or R^2+G^2+B^2 (flag 1) ----
@compute @workgroup_size(256)
fn energy(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h;
    if (i >= n) { return; }
    let r = a[i]; let gg = a[n + i]; let bb = a[2u * n + i];
    if (p.flag == 1u) { e[i] = r * r + gg * gg + bb * bb; }
    else { let y = 0.299 * r + 0.587 * gg + 0.114 * bb; e[i] = y * y; }
}
// ---- separable window sum with weights wt[klen], reflect-101: e -> b (win_h), b -> e (win_v) ----
@compute @workgroup_size(16, 16)
fn win_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let r = i32(p.klen / 2u); let w = i32(p.w); var s = 0.0;
    for (var t = 0u; t < p.klen; t++) { s += wt[t] * e[y * p.w + u32(refl(i32(x) + i32(t) - r, w))]; }
    b[y * p.w + x] = s;
}
@compute @workgroup_size(16, 16)
fn win_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let r = i32(p.klen / 2u); let h = i32(p.h); var s = 0.0;
    for (var t = 0u; t < p.klen; t++) { s += wt[t] * b[u32(refl(i32(y) + i32(t) - r, h)) * p.w + x]; }
    e[y * p.w + x] = s;
}
// ---- winner-take-all: e = energy, b = best, o = acc planes, a = new planes, wt unused, `u`-less; win in e? no:
// select: where e[i] > b[i]: b[i] = e[i]; o[c*n+i] = a[c*n+i]; if flag: win (stored in `e`? no) ...
// We keep the winner map in `o`'s companion: use `e` for energy, `b` for best, `o` for acc, `a` for new,
// and `wt`-slot is read-only, so the winner map goes through binding 4? Simpler: a second entry point
// writes winners: `select_rec` uses `e` as energy and `b` as best like select, plus writes the frame
// index into `o` at offset 3n (acc buffers for the depth level are allocated with 4 planes).
@compute @workgroup_size(256)
fn sel(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h;
    if (i >= n) { return; }
    if (e[i] > b[i]) {
        b[i] = e[i];
        o[i] = a[i]; o[n + i] = a[n + i]; o[2u * n + i] = a[2u * n + i];
        if (p.flag == 1u) { o[3u * n + i] = p.f0; }
    }
}
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w) { return; }
    o[i] = p.f0;
}
@compute @workgroup_size(256)
fn clamp01(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h * 3u) { return; }
    o[i] = clamp(o[i], 0.0, 1.0);
}
@compute @workgroup_size(256)
fn copy_plane(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    o[p.off_out + i] = a[p.off_in + i];
}

// ---- input unpack + warp ----
// `u` holds the decoded frame as interleaved RGB u16 pairs (2 samples per u32),
// p.w x p.h. Warp: dest (x,y) -> source (sx,sy) via inverse affine f0..f3 +
// (wt[0], wt[1]) translation... we pass the 6 affine terms in f0..f3 + wt[0..1]:
//   sx = f0*x + f1*y + wt[0];  sy = f2*x + f3*y + wt[1]
// Out-of-bounds destination pixels take the unwarped source pixel (the native
// aligner's `valid` fallback). flag 1 = identity (plain unpack), scale = 1/65535 (16-bit
// samples; 8-bit inputs are widened to 16-bit on the CPU). p.klen picks the
// kernel (lapstack_core::align::Interp::id): 0 nearest, 1 bilinear, 2 bicubic,
// 3 spline4x4, 4 spline6x6, 5 lanczos3 — `ktaps` taps at 1 - taps/2 .. from the
// floor of the source point, weights `kweights` (the native warp_plane's).
fn sample_u16(c: u32, x: u32, y: u32) -> f32 {
    let k = (y * p.w + x) * 3u + c;
    let word = u[k >> 1u];
    let v = select(word >> 16u, word & 0xffffu, (k & 1u) == 0u);
    return f32(v) / 65535.0;
}
fn spl4(t: f32) -> vec4<f32> {
    return vec4<f32>(
        ((-1.0 / 3.0 * t + 0.8) * t - 0.46666667) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((1.0 / 3.0 * t - 0.2) * t - 0.13333334) * t);
}
fn ktaps(k: u32) -> i32 {
    switch k {
        case 0u, 1u: { return 2; }
        case 2u, 3u: { return 4; }
        default: { return 6; }
    }
}
// Keys' cubic (a = -0.5), Panorama Tools' spline36 and the 3-lobe Lanczos window at distance d
fn keys(d: f32) -> f32 {
    if (d < 1.0) { return (1.5 * d - 2.5) * d * d + 1.0; }
    if (d < 2.0) { return ((-0.5 * d + 2.5) * d - 4.0) * d + 2.0; }
    return 0.0;
}
fn spline36(d: f32) -> f32 {
    if (d < 1.0) { return ((13.0 / 11.0 * d - 453.0 / 209.0) * d - 3.0 / 209.0) * d + 1.0; }
    if (d < 2.0) { let u = d - 1.0; return ((-6.0 / 11.0 * u + 270.0 / 209.0) * u - 156.0 / 209.0) * u; }
    if (d < 3.0) { let u = d - 2.0; return ((1.0 / 11.0 * u - 45.0 / 209.0) * u + 26.0 / 209.0) * u; }
    return 0.0;
}
fn lanczos3(d: f32) -> f32 {
    if (d < 1e-6) { return 1.0; }
    if (d >= 3.0) { return 0.0; }
    let a = 3.14159265 * d; let b = a / 3.0;
    return sin(a) / a * (sin(b) / b);
}
// the kernel's weights at fraction t, taps at 1 - taps/2 .. from the floor; the
// distance-form kernels normalised to sum 1 (Lanczos needs it, the others are exact)
fn kweights(k: u32, t: f32) -> array<f32, 6> {
    var w = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    switch k {
        case 0u: { let r = f32(t >= 0.5); w[0] = 1.0 - r; w[1] = r; }
        case 1u: { w[0] = 1.0 - t; w[1] = t; }
        case 3u: { let s = spl4(t); w[0] = s[0]; w[1] = s[1]; w[2] = s[2]; w[3] = s[3]; }
        default: {
            let n = ktaps(k); let start = 1 - n / 2;
            var sum = 0.0;
            for (var i = 0; i < n; i++) {
                let d = abs(t - f32(i + start));
                var v = 0.0;
                if (k == 2u) { v = keys(d); } else if (k == 4u) { v = spline36(d); } else { v = lanczos3(d); }
                w[i] = v; sum += v;
            }
            for (var i = 0; i < n; i++) { w[i] /= sum; }
        }
    }
    return w;
}
@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let n = p.w * p.h; let idx = y * p.w + x;
    if (p.flag == 1u) {
        o[idx] = sample_u16(0u, x, y); o[n + idx] = sample_u16(1u, x, y); o[2u * n + idx] = sample_u16(2u, x, y);
        return;
    }
    let sx = p.f0 * f32(x) + p.f1 * f32(y) + wt[0];
    let sy = p.f2 * f32(x) + p.f3 * f32(y) + wt[1];
    let wm1 = f32(p.w - 1u); let hm1 = f32(p.h - 1u);
    if (sx < 0.0 || sx > wm1 || sy < 0.0 || sy > hm1) {
        o[idx] = sample_u16(0u, x, y); o[n + idx] = sample_u16(1u, x, y); o[2u * n + idx] = sample_u16(2u, x, y);
        return;
    }
    let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
    let nt = ktaps(p.klen); let start = 1 - nt / 2;
    var wx = kweights(p.klen, sx - f32(x0)); var wy = kweights(p.klen, sy - f32(y0));
    for (var c = 0u; c < 3u; c++) {
        var acc = 0.0;
        for (var j = 0; j < nt; j++) {
            let yy = u32(clamp(y0 + j + start, 0, i32(p.h) - 1));
            var r = 0.0;
            for (var i = 0; i < nt; i++) {
                let xx = u32(clamp(x0 + i + start, 0, i32(p.w) - 1));
                r += wx[i] * sample_u16(c, xx, yy);
            }
            acc += wy[j] * r;
        }
        o[c * n + idx] = acc;
    }
}

// ---- alignment cost: warp the target luma `a` (p.ow x p.oh) into the reference `b`
// (p.w x p.h) and reduce (sum d, sum d^2, count) over valid pixels per workgroup
// into e[wg*3 ..]. Host sums the partials. FP32 like the native CUDA kernel.
var<workgroup> sd: array<f32, 256>;
var<workgroup> sd2: array<f32, 256>;
var<workgroup> sn: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn cost(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32,
        @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let x = g.x; let y = g.y;
    var ld = 0.0; var ld2 = 0.0; var ln = 0.0;
    if (x < p.w && y < p.h) {
        let sx = p.f0 * f32(x) + p.f1 * f32(y) + wt[0];
        let sy = p.f2 * f32(x) + p.f3 * f32(y) + wt[1];
        let tw = i32(p.ow); let th = i32(p.oh);
        if (sx >= 0.0 && sx <= f32(tw - 1) && sy >= 0.0 && sy <= f32(th - 1)) {
            let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
            let wx = spl4(sx - f32(x0)); let wy = spl4(sy - f32(y0));
            var acc = 0.0;
            for (var j = 0; j < 4; j++) {
                let yy = u32(clamp(y0 + j - 1, 0, th - 1));
                var r = 0.0;
                for (var i = 0; i < 4; i++) {
                    let xx = u32(clamp(x0 + i - 1, 0, tw - 1));
                    r += wx[i] * a[yy * p.ow + xx];
                }
                acc += wy[j] * r;
            }
            let d = b[y * p.w + x] - acc;
            ld = d; ld2 = d * d; ln = 1.0;
        }
    }
    sd[t] = ld; sd2[t] = ld2; sn[t] = ln;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { sd[t] += sd[t + s]; sd2[t] += sd2[t + s]; sn[t] += sn[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) {
        let k = (wg.y * nwg.x + wg.x) * 3u;
        e[k] = sd[0]; e[k + 1u] = sd2[0]; e[k + 2u] = sn[0];
    }
}

// ---- brightness normalisation (lapstack_core::brightness) ----
// frame 0's channel means per 64x64 block: `a` = 3 planes (p.w x p.h) -> o[(by*p.ow+bx)*3 + c]
@compute @workgroup_size(16, 16)
fn blk_mean(@builtin(global_invocation_id) g: vec3<u32>) {
    let bx = g.x; let by = g.y;
    if (bx >= p.ow || by >= p.oh) { return; }
    let n = p.w * p.h;
    let x1 = min(bx * 64u + 64u, p.w); let y1 = min(by * 64u + 64u, p.h);
    var s = vec3<f32>(0.0); var cnt = 0.0;
    for (var y = by * 64u; y < y1; y++) {
        for (var x = bx * 64u; x < x1; x++) {
            let i = y * p.w + x;
            s += vec3<f32>(a[i], a[n + i], a[2u * n + i]); cnt += 1.0;
        }
    }
    let k = (by * p.ow + bx) * 3u;
    o[k] = s.x / max(cnt, 1.0); o[k + 1u] = s.y / max(cnt, 1.0); o[k + 2u] = s.z / max(cnt, 1.0);
}
// one workgroup per 64x64 block, one thread per 4x4 patch: the frame `a`'s channel sums
// over the block's pixels the warp covers (p.flag = 1: all of them; else the source point
// p.f0..f3 / wt lies inside the frame), the count, and frame 0's block mean `b` times
// the count — so both means are over the same pixels. e[block*8 ..] = [sr, sg, sb, n, ref_r*n, ref_g*n, ref_b*n, 0].
var<workgroup> br: array<f32, 256>;
var<workgroup> bg: array<f32, 256>;
var<workgroup> bb: array<f32, 256>;
var<workgroup> bn: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn bright(@builtin(local_invocation_id) l: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let n = p.w * p.h;
    var s = vec3<f32>(0.0); var cnt = 0.0;
    let wm1 = f32(p.w - 1u); let hm1 = f32(p.h - 1u);
    for (var dy = 0u; dy < 4u; dy++) {
        for (var dx = 0u; dx < 4u; dx++) {
            let x = wg.x * 64u + l.x * 4u + dx; let y = wg.y * 64u + l.y * 4u + dy;
            if (x >= p.w || y >= p.h) { continue; }
            if (p.flag == 0u) {
                let sx = p.f0 * f32(x) + p.f1 * f32(y) + wt[0];
                let sy = p.f2 * f32(x) + p.f3 * f32(y) + wt[1];
                if (sx < 0.0 || sx > wm1 || sy < 0.0 || sy > hm1) { continue; }
            }
            let i = y * p.w + x;
            s += vec3<f32>(a[i], a[n + i], a[2u * n + i]); cnt += 1.0;
        }
    }
    br[t] = s.x; bg[t] = s.y; bb[t] = s.z; bn[t] = cnt;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h = h >> 1u) {
        if (t < h) { br[t] += br[t + h]; bg[t] += bg[t + h]; bb[t] += bb[t + h]; bn[t] += bn[t + h]; }
        workgroupBarrier();
    }
    if (t == 0u) {
        let blk = wg.y * p.ow + wg.x;
        let k = blk * 8u;
        e[k] = br[0]; e[k + 1u] = bg[0]; e[k + 2u] = bb[0]; e[k + 3u] = bn[0];
        e[k + 4u] = b[blk * 3u] * bn[0]; e[k + 5u] = b[blk * 3u + 1u] * bn[0]; e[k + 6u] = b[blk * 3u + 2u] * bn[0]; e[k + 7u] = 0.0;
    }
}
// multiply the three planes of `o` (p.w elements each) by p.f0, p.f1, p.f2
@compute @workgroup_size(256)
fn gain3(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= 3u * p.w) { return; }
    let c = i / p.w;
    o[i] = o[i] * select(select(p.f2, p.f1, c == 1u), p.f0, c == 0u);
}

// ---- readbacks ----
// rgba8 (packed u32 per pixel) from 3 planes in `a` (p.w x p.h) -> o (as u32 bit pattern via bitcast)
@compute @workgroup_size(256)
fn to_rgba8(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let r = u32(clamp(a[i], 0.0, 1.0) * 255.0 + 0.5);
    let gg = u32(clamp(a[n + i], 0.0, 1.0) * 255.0 + 0.5);
    let bb = u32(clamp(a[2u * n + i], 0.0, 1.0) * 255.0 + 0.5);
    o[i] = bitcast<f32>(r | (gg << 8u) | (bb << 16u) | (255u << 24u));
}
// rgb16 interleaved (2 samples per u32) from 3 planes in `a`
@compute @workgroup_size(256)
fn to_rgb16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = gid1(g, nwg); let n = p.w * p.h; let total = n * 3u;
    let s0 = k * 2u; if (s0 >= total) { return; }
    var word = 0u;
    for (var j = 0u; j < 2u; j++) {
        let s = s0 + j; if (s >= total) { break; }
        let px = s / 3u; let c = s % 3u;
        let v = u32(clamp(a[c * n + px], 0.0, 1.0) * 65535.0 + 0.5);
        word |= v << (16u * j);
    }
    o[k] = bitcast<f32>(word);
}
// proxy: area-average the 3 planes in `a` (p.w x p.h) by integer factor klen into rgba8 (p.ow x p.oh)
@compute @workgroup_size(16, 16)
fn proxy(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y; if (x >= p.ow || y >= p.oh) { return; }
    let n = p.w * p.h; let f = p.klen;
    var s = vec3<f32>(0.0); var cnt = 0.0;
    for (var dy = 0u; dy < f; dy++) {
        let yy = y * f + dy; if (yy >= p.h) { break; }
        for (var dx = 0u; dx < f; dx++) {
            let xx = x * f + dx; if (xx >= p.w) { break; }
            let i = yy * p.w + xx;
            s += vec3<f32>(a[i], a[n + i], a[2u * n + i]); cnt += 1.0;
        }
    }
    s = clamp(s / max(cnt, 1.0), vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5;
    o[y * p.ow + x] = bitcast<f32>(u32(s.x) | (u32(s.y) << 8u) | (u32(s.z) << 16u) | (255u << 24u));
}
// down1: area-average one f32 plane `a` (p.w x p.h) by integer factor klen -> o (p.ow x p.oh).
// f0 (rounded) shifts the output right by that many input pixels, clamped at the input's
// edges (the refold's per-frame shift); 0 keeps partial blocks at the right edge as they are.
// off_in / off_out pick a plane of a multi-plane buffer.
@compute @workgroup_size(16, 16)
fn down1(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y; if (x >= p.ow || y >= p.oh) { return; }
    let f = p.klen; var s = 0.0; var cnt = 0.0;
    let sh = i32(round(p.f0));
    for (var dy = 0u; dy < f; dy++) {
        let yy = y * f + dy; if (yy >= p.h) { break; }
        for (var dx = 0u; dx < f; dx++) {
            let xs = i32(x * f + dx) - sh;
            if (sh == 0 && xs >= i32(p.w)) { break; }
            let xx = u32(clamp(xs, 0, i32(p.w) - 1));
            s += a[p.off_in + yy * p.w + xx]; cnt += 1.0;
        }
    }
    o[p.off_out + y * p.ow + x] = s / max(cnt, 1.0);
}
// luma plane (f32, p.w x p.h) from the packed u16 frame in `u`, for the aligner
@compute @workgroup_size(256)
fn luma_u16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    let x = i % p.w; let y = i / p.w;
    o[i] = 0.299 * sample_u16(0u, x, y) + 0.587 * sample_u16(1u, x, y) + 0.114 * sample_u16(2u, x, y);
}
// luma plane from 3 f32 planes in `a` -> o
@compute @workgroup_size(256)
fn luma_f32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    o[i] = 0.299 * a[i] + 0.587 * a[n + i] + 0.114 * a[2u * n + i];
}

// =====================================================================
// Depth from focus (twins of lapstack-core/src/depth.rs). Buffers of one
// working-grid plane are p.w x p.h unless noted; "2n buffers" hold two planes.
// =====================================================================

// sparse convolution, reflect-101: taps in wt as (dy, dx, weight) triplets,
// klen = number of taps; a = luma plane -> o = |sum| (flag 1) or sum (flag 0)
@compute @workgroup_size(16, 16)
fn conv_taps(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let w = i32(p.w); let h = i32(p.h); var s = 0.0;
    for (var t = 0u; t < p.klen; t++) {
        let dy = i32(wt[3u * t]); let dx = i32(wt[3u * t + 1u]);
        s += wt[3u * t + 2u] * a[u32(refl(i32(y) + dy, h)) * p.w + u32(refl(i32(x) + dx, w))];
    }
    o[y * p.w + x] = select(s, abs(s), p.flag == 1u);
}
// box filter with border clipping (mean over the in-image part of the window).
// box_h: a -> b (row sums); box_v: b -> o (column sums, normalised by count). klen = radius.
@compute @workgroup_size(16, 16)
fn box_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let r = i32(p.klen); let x0 = max(i32(x) - r, 0); let x1 = min(i32(x) + r + 1, i32(p.w));
    var s = 0.0;
    for (var i = x0; i < x1; i++) { s += a[y * p.w + u32(i)]; }
    b[y * p.w + x] = s;
}
@compute @workgroup_size(16, 16)
fn box_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let r = i32(p.klen); let y0 = max(i32(y) - r, 0); let y1 = min(i32(y) + r + 1, i32(p.h));
    var s = 0.0;
    for (var i = y0; i < y1; i++) { s += b[u32(i) * p.w + x]; }
    let cx = f32(min(i32(x) + r + 1, i32(p.w)) - max(i32(x) - r, 0));
    o[y * p.w + x] = s / (cx * f32(y1 - y0));
}
// o = a * wt (elementwise), n = w*h
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    o[i] = a[i] * wt[i];
}
// guided-filter coefficients: a = mean_p, wt = corr_Ip, e = [mean_I | var_I] (2n) -> b = A, o = B; f0 = eps
@compute @workgroup_size(256)
fn gf_ab(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let mi = e[i]; let vi = e[n + i];
    let A = (wt[i] - mi * a[i]) / (vi + p.f0);
    b[i] = A; o[i] = a[i] - A * mi;
}
// q = mean_a * I + mean_b:  a = mean_a, wt = mean_b, e = I -> o; flag 1 clamps q >= 0
@compute @workgroup_size(256)
fn gf_apply(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    let q = a[i] * e[i] + wt[i];
    o[i] = select(q, max(q, 0.0), p.flag == 1u);
}
// e = [mean_I | var_I] from a = mean_I, wt = mean_II
@compute @workgroup_size(256)
fn gf_var(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    e[i] = a[i]; e[n + i] = max(wt[i] - a[i] * a[i], 0.0);
}
// unpack u16 pairs (u) with scale f0 -> o (n floats)
@compute @workgroup_size(256)
fn unpack_u16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    let word = u[i >> 1u];
    let v = select(word >> 16u, word & 0xffffu, (i & 1u) == 0u);
    o[i] = f32(v) * p.f0;
}
// pack a (n floats) * f0, clamped to [0, 65535] -> o as u16 pairs
@compute @workgroup_size(256)
fn pack_u16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = gid1(g, nwg); let n = p.w * p.h; let i0 = k * 2u; if (i0 >= n) { return; }
    var word = 0u;
    for (var j = 0u; j < 2u; j++) {
        let i = i0 + j; if (i >= n) { break; }
        word |= u32(clamp(a[i] * p.f0 + 0.5, 0.0, 65535.0)) << (16u * j);
    }
    o[k] = bitcast<f32>(word);
}

// ---- streamed peak tracker: state in `o` as 9 planes
//   0 c1, 1 l1, 2 r1, 3 c2, 4 prev, 5 prev2, 6 sum, 7 cmin, 8 i1 (as f32)
// peak_push: a = this slice (>= 0), klen = slice index m
fn peak_register(i: u32, n: u32, val: f32, idx: f32, l: f32, r: f32) {
    if (val > o[i]) {
        if (o[i] >= 0.0) { o[3u * n + i] = o[i]; }
        o[i] = val; o[8u * n + i] = idx; o[n + i] = l; o[2u * n + i] = r;
    } else if (val > o[3u * n + i]) {
        o[3u * n + i] = val;
    }
}
@compute @workgroup_size(256)
fn peak_init(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    o[i] = -1.0; o[n + i] = -1.0; o[2u * n + i] = -1.0; o[3u * n + i] = -1.0;
    o[4u * n + i] = 0.0; o[5u * n + i] = 0.0; o[6u * n + i] = 0.0; o[7u * n + i] = 3.4e38; o[8u * n + i] = 0.0;
}
@compute @workgroup_size(256)
fn peak_push(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let m = p.klen; let v = a[i];
    let prev = o[4u * n + i]; let prev2 = o[5u * n + i];
    if (m >= 1u) {
        let is_peak = (m == 1u || prev >= prev2) && prev > v;
        if (is_peak) {
            peak_register(i, n, prev, f32(m - 1u), select(-1.0, prev2, m >= 2u), v);
        }
    }
    o[6u * n + i] += v; o[7u * n + i] = min(o[7u * n + i], v);
    o[5u * n + i] = prev; o[4u * n + i] = v;
}
// peak_finish: close the profiles (klen = frame count n_frames, f0 = noise floor, f1 = gate);
// writes depth -> b, raw confidence -> e
@compute @workgroup_size(256)
fn peak_finish(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let nf = p.klen;
    let prev = o[4u * n + i]; let prev2 = o[5u * n + i];
    if (nf == 1u || prev >= prev2) {
        peak_register(i, n, prev, f32(nf - 1u), select(-1.0, prev2, nf >= 2u), -1.0);
    }
    let c1 = o[i]; let l = o[n + i]; let r = o[2u * n + i]; let c2 = o[3u * n + i];
    var delta = 0.0;
    if (l >= 0.0 && r >= 0.0 && c1 > 0.0) {
        let ll = log(max(l, 1e-12)); let lc = log(c1); let lr = log(max(r, 1e-12));
        let den = ll - 2.0 * lc + lr;
        if (den < 0.0) { delta = clamp(0.5 * (ll - lr) / den, -0.5, 0.5); }
    }
    b[i] = o[8u * n + i] + delta;
    var conf = 0.0;
    if (c1 > 0.0) {
        let mean = o[6u * n + i] / f32(nf);
        let prom = clamp(1.0 - mean / c1, 0.0, 1.0);
        let pkr = select(1.0, clamp(1.0 - c2 / c1, 0.0, 1.0), c2 >= 0.0);
        var gt = 1.0;
        if (p.f1 > 0.0 && p.f0 > 0.0) { gt = clamp((c1 - p.f0) / (p.f1 * p.f0), 0.0, 1.0); }
        conf = prom * pkr * gt;
    }
    e[i] = conf;
}
// 3x3 median, reflect-101: a -> o
@compute @workgroup_size(16, 16)
fn median3(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    var v: array<f32, 9>; var k = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        let yy = u32(refl(i32(y) + dy, i32(p.h)));
        for (var dx = -1; dx <= 1; dx++) {
            v[k] = a[yy * p.w + u32(refl(i32(x) + dx, i32(p.w)))]; k++;
        }
    }
    // partial selection: 5 passes of bubble-min give the median at v[4]
    for (var i = 0u; i < 5u; i++) {
        for (var j = 8u; j > i; j--) {
            if (v[j] < v[j - 1u]) { let t = v[j]; v[j] = v[j - 1u]; v[j - 1u] = t; }
        }
    }
    o[y * p.w + x] = v[4];
}
// o = min(1, a * f0) + f1   (confidence normalisation / data weights)
@compute @workgroup_size(256)
fn scale_clamp(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    o[i] = min(a[i] * p.f0, 1.0) + p.f1;
}
// edge weights exp(-|dI|/sigma): a = guide -> o = [ax | ay] (2n), f0 = sigma
@compute @workgroup_size(16, 16)
fn edge_w(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let n = p.w * p.h; let i = y * p.w + x; let inv = -1.0 / max(p.f0, 1e-6);
    o[i] = select(0.0, exp(abs(a[i] - a[i + 1u]) * inv), x + 1u < p.w);
    o[n + i] = select(0.0, exp(abs(a[i] - a[i + p.w]) * inv), y + 1u < p.h);
}
// 1-D WLS along rows (Thomas): a = f (data), wt = wd, e = [ax|ay] (2n, read), b = u (out),
// o = scratch [cp | dp] (2n). f0 = lambda. One thread per row.
@compute @workgroup_size(64)
fn fgs_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    let y = g.x; if (y >= p.h) { return; }
    let w = p.w; let n = w * p.h; let lam = p.f0; let off = y * w;
    var prev_a = 0.0;
    for (var i = 0u; i < w; i++) {
        let k = off + i;
        let ai = select(0.0, e[k], i + 1u < w);
        let diag = wt[k] + lam * (prev_a + ai);
        let lower = -lam * prev_a; let upper = -lam * ai;
        var cprev = 0.0; var dprev = 0.0;
        if (i > 0u) { cprev = o[k - 1u]; dprev = o[n + k - 1u]; }
        var m = diag - lower * cprev;
        if (abs(m) < 1e-12) { m = 1e-12; }
        o[k] = upper / m;
        o[n + k] = (wt[k] * a[k] - lower * dprev) / m;
        prev_a = ai;
    }
    b[off + w - 1u] = o[n + off + w - 1u];
    for (var i = w - 1u; i > 0u; i--) {
        let k = off + i - 1u;
        b[k] = o[n + k] - o[k] * b[k + 1u];
    }
}
// same along columns (ay = second plane of e). One thread per column.
@compute @workgroup_size(64)
fn fgs_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; if (x >= p.w) { return; }
    let w = p.w; let h = p.h; let n = w * h; let lam = p.f0;
    var prev_a = 0.0;
    for (var y = 0u; y < h; y++) {
        let k = y * w + x;
        let ai = select(0.0, e[n + k], y + 1u < h);
        let diag = wt[k] + lam * (prev_a + ai);
        let lower = -lam * prev_a; let upper = -lam * ai;
        var cprev = 0.0; var dprev = 0.0;
        if (y > 0u) { cprev = o[k - w]; dprev = o[n + k - w]; }
        var m = diag - lower * cprev;
        if (abs(m) < 1e-12) { m = 1e-12; }
        o[k] = upper / m;
        o[n + k] = (wt[k] * a[k] - lower * dprev) / m;
        prev_a = ai;
    }
    let last = (h - 1u) * w + x;
    b[last] = o[n + last];
    for (var y = h - 1u; y > 0u; y--) {
        let k = (y - 1u) * w + x;
        b[k] = o[n + k] - o[k] * b[k + w];
    }
}
// CG for (W + lambda L) u = W d.  a = x, wt = wd, e = [ax|ay] -> b = A x   (f0 = lambda)
fn wls_matvec(i: u32, x: u32, y: u32) -> f32 {
    let n = p.w * p.h; let lam = p.f0;
    var v = wt[i] * a[i];
    if (x > 0u) { v += lam * e[i - 1u] * (a[i] - a[i - 1u]); }
    if (x + 1u < p.w) { v += lam * e[i] * (a[i] - a[i + 1u]); }
    if (y > 0u) { v += lam * e[n + i - p.w] * (a[i] - a[i - p.w]); }
    if (y + 1u < p.h) { v += lam * e[n + i] * (a[i] - a[i + p.w]); }
    return v;
}
@compute @workgroup_size(16, 16)
fn cg_matvec(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y; if (x >= p.w || y >= p.h) { return; }
    b[y * p.w + x] = wls_matvec(y * p.w + x, x, y);
}
// cg_init: given u in a, d in... : r = wd*d - A u, z = r*diag, p = z, and diag itself.
// Inputs: a = u, wt = wd, e = [ax|ay]; b = d (read), o = r, and diag/z/p come from a second kernel.
@compute @workgroup_size(16, 16)
fn cg_resid(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y; if (x >= p.w || y >= p.h) { return; }
    let i = y * p.w + x; let n = p.w * p.h; let lam = p.f0;
    o[i] = wt[i] * b[i] - wls_matvec(i, x, y);
    var dg = wt[i];
    if (x > 0u) { dg += lam * e[i - 1u]; }
    if (x + 1u < p.w) { dg += lam * e[i]; }
    if (y > 0u) { dg += lam * e[n + i - p.w]; }
    if (y + 1u < p.h) { dg += lam * e[n + i]; }
    o[n + i] = 1.0 / max(dg, 1e-12);   // o = [r | diag]
}
// z = r * diag, p = z:  a = [r|diag] (2n) -> b = z, o = p
@compute @workgroup_size(256)
fn cg_zp(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let z = a[i] * a[n + i]; b[i] = z; o[i] = z;
}
// workgroup-partial dot product: a . wt over n = w*h -> o[wg]
var<workgroup> sdot: array<f32, 256>;
@compute @workgroup_size(256)
fn dot_partial(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) t: u32,
               @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h;
    sdot[t] = select(0.0, a[i] * wt[i], i < n);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { sdot[t] += sdot[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { o[wg.y * nwg.x + wg.x] = sdot[0]; }
}
// sum the klen partials in a into scal (o): flag 0: rz = sum; 1: pap = sum, alpha = rz/pap;
// 2: rz_new = sum, beta = rz_new/rz, rz = rz_new.  scal = [rz, alpha, beta, pap]
@compute @workgroup_size(256)
fn reduce_scal(@builtin(local_invocation_index) t: u32) {
    var s = 0.0;
    for (var i = t; i < p.klen; i += 256u) { s += a[i]; }
    sdot[t] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (t < k) { sdot[t] += sdot[t + k]; }
        workgroupBarrier();
    }
    if (t == 0u) {
        let v = sdot[0];
        if (p.flag == 0u) { o[0] = v; }
        else if (p.flag == 1u) { o[3] = v; o[1] = select(0.0, o[0] / v, v > 0.0); }
        else { o[2] = select(0.0, v / o[0], o[0] > 0.0); o[0] = v; }
    }
}
// u += alpha p:  a = p, b = u, u(u32) = scal
@compute @workgroup_size(256)
fn cg_axpy_u(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    b[i] += bitcast<f32>(u[1]) * a[i];
}
// r -= alpha Ap; z = r * diag:  a = Ap, b = [r|diag] (2n), o = z, u = scal
@compute @workgroup_size(256)
fn cg_update_rz(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let r = b[i] - bitcast<f32>(u[1]) * a[i];
    b[i] = r; o[i] = r * b[n + i];
}
// p = z + beta p:  a = z, b = p, u = scal
@compute @workgroup_size(256)
fn cg_update_p(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    b[i] = a[i] + bitcast<f32>(u[2]) * b[i];
}
// robust reweight: o = wt * min(1, f0 / |a - e|)   (a = d, e = u, wt = conf)
@compute @workgroup_size(256)
fn robust_w(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); if (i >= p.w * p.h) { return; }
    o[i] = wt[i] * min(1.0, p.f0 / max(abs(a[i] - e[i]), 1e-12)) + p.f1;
}
// guided upsampling apply at full res: bilinear (a = mean_A, wt = mean_B on the ow x oh grid, block size klen)
// times the full-res luma e -> o, clamped to [0, f0]. p.w x p.h = full res.
@compute @workgroup_size(16, 16)
fn up_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y; if (x >= p.w || y >= p.h) { return; }
    let inv = 1.0 / f32(p.klen);
    let fy = clamp((f32(y) + 0.5) * inv - 0.5, 0.0, f32(p.oh - 1u));
    let fx = clamp((f32(x) + 0.5) * inv - 0.5, 0.0, f32(p.ow - 1u));
    let y0 = u32(fy); let y1 = min(y0 + 1u, p.oh - 1u); let ty = fy - f32(y0);
    let x0 = u32(fx); let x1 = min(x0 + 1u, p.ow - 1u); let tx = fx - f32(x0);
    let i00 = y0 * p.ow + x0; let i01 = y0 * p.ow + x1; let i10 = y1 * p.ow + x0; let i11 = y1 * p.ow + x1;
    let A = mix(mix(a[i00], a[i01], tx), mix(a[i10], a[i11], tx), ty);
    let B = mix(mix(wt[i00], wt[i01], tx), mix(wt[i10], wt[i11], tx), ty);
    let i = y * p.w + x;
    o[i] = clamp(A * e[i] + B, 0.0, p.f0);
}

// ---- depth-map rendering (DMAP): each image is blended in with weight
// 1 - dist(depth, [f0, f1]), the distance of the pixel's depth index from the
// image's frame range, clamped to [0, 1]: a frame (f0 = f1 = its index) gets the
// triangular weight 1 - |index - depth|; a slab (f0..f1 = its frames) full weight
// wherever the depth lies within it and the same one-frame fall-off outside.
// a = warped frame or collapsed slab (3 planes, p.w x p.h), wt = full-res depth,
// b = accumulator (3 planes), o = weight sum.
@compute @workgroup_size(256)
fn dmap_acc(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let d = wt[i];
    let t = clamp(1.0 - max(max(p.f0 - d, d - p.f1), 0.0), 0.0, 1.0);
    if (t > 0.0) {
        b[i] += t * a[i]; b[n + i] += t * a[n + i]; b[2u * n + i] += t * a[2u * n + i];
        o[i] += t;
    }
}
// ---- weighted average (twin of lapstack-core/src/wav.rs): the re-warped frame `a` (3
// planes) accumulates into b (3 planes) and its weight into o, weighed by the contrast
// map `wt` on the ow×oh grid of klen-pixel blocks (bilinear, samples at block centres),
// raised to f0, plus the floor f1.
@compute @workgroup_size(256)
fn wav_acc(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let x = i % p.w; let y = i / p.w;
    let k = f32(p.klen);
    let gx = clamp((f32(x) + 0.5) / k - 0.5, 0.0, f32(p.ow - 1u));
    let gy = clamp((f32(y) + 0.5) / k - 0.5, 0.0, f32(p.oh - 1u));
    let x0 = u32(floor(gx)); let y0 = u32(floor(gy));
    let x1 = min(x0 + 1u, p.ow - 1u); let y1 = min(y0 + 1u, p.oh - 1u);
    let fx = gx - f32(x0); let fy = gy - f32(y0);
    let v = mix(mix(wt[y0 * p.ow + x0], wt[y0 * p.ow + x1], fx), mix(wt[y1 * p.ow + x0], wt[y1 * p.ow + x1], fx), fy);
    let wgt = pow(max(v, 0.0), p.f0) + p.f1;
    b[i] += wgt * a[i]; b[n + i] += wgt * a[n + i]; b[2u * n + i] += wgt * a[2u * n + i];
    o[i] += wgt;
}
// b (3 planes) /= o (the weight floor keeps it above zero), clamped to [0,1]
@compute @workgroup_size(256)
fn wav_norm(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let d = 1.0 / max(o[i], 1e-30);
    b[i] = clamp(b[i] * d, 0.0, 1.0); b[n + i] = clamp(b[n + i] * d, 0.0, 1.0); b[2u * n + i] = clamp(b[2u * n + i] * d, 0.0, 1.0);
}
// b (3 planes) /= max(o, 1e-6), clamped to [0,1]
@compute @workgroup_size(256)
fn dmap_norm(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid1(g, nwg); let n = p.w * p.h; if (i >= n) { return; }
    let d = 1.0 / max(o[i], 1e-6);
    b[i] = clamp(b[i] * d, 0.0, 1.0); b[n + i] = clamp(b[n + i] * d, 0.0, 1.0); b[2u * n + i] = clamp(b[2u * n + i] * d, 0.0, 1.0);
}

// ---- In focus (twin of the formula in lib.rs source_focus): the warped frame
// `a` (3 planes) dimmed by how far the full-res depth `wt` (frame index per
// pixel) puts each pixel from frame off_in. b = row sums of the frame's luma
// over ±klen px (box_h); this kernel finishes the box mean with the column
// pass (border-clipped, like box_v) and writes packed RGBA8 to `o`.
// f0 = dim, f1 = w0, f2 = w1 (frames), f3 = tex; the soft clip caps at 0.4.
@compute @workgroup_size(16, 16)
fn focus_out(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x; let y = g.y;
    if (x >= p.w || y >= p.h) { return; }
    let n = p.w * p.h; let i = y * p.w + x;
    let r = i32(p.klen); let y0 = max(i32(y) - r, 0); let y1 = min(i32(y) + r + 1, i32(p.h));
    var s = 0.0;
    for (var j = y0; j < y1; j++) { s += b[u32(j) * p.w + x]; }
    let cx = f32(min(i32(x) + r + 1, i32(p.w)) - max(i32(x) - r, 0));
    let mean = s / (cx * f32(y1 - y0));
    let rgb = vec3<f32>(a[i], a[n + i], a[2u * n + i]);
    let luma = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let wgt = clamp((p.f2 - abs(wt[i] - f32(p.off_in))) / max(p.f2 - p.f1, 1e-3), 0.0, 1.0);
    let gain = wgt + (1.0 - wgt) * p.f0;
    let add = (1.0 - wgt) * 0.4 * (1.0 - exp(-p.f3 * abs(luma - mean) / 0.4));
    let v = vec3<u32>(clamp(rgb * gain + add, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5);
    o[i] = bitcast<f32>(v.x | (v.y << 8u) | (v.z << 16u) | (255u << 24u));
}

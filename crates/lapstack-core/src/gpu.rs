// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! CUDA fusion path (feature `gpu`). Same maths as `pyramid.rs` + `fuse.rs`
//! — the binomial taps, reflect-101 borders, luma energy, binomial window and
//! winner-take-all select are transcribed kernel for kernel — so the output
//! matches the CPU path to float rounding. Per frame only the three RGB
//! planes cross the bus up and the residual level (a few KB) comes back; the
//! accumulator pyramid, best-energy planes and winner map live on the device.
//! The residual rule (deviation + entropy over N tiny planes) stays on the CPU.
//! With halo control (`fuse.rs`) the guide's weights are made, REDUCEd and
//! folded on the device too (`wgtk`, `wacck`, `wnormk`), the residual included.
//!
//! cudarc with `dynamic-loading` (libcuda + libnvrtc found at runtime, no
//! toolkit at build time), kernels compiled by NVRTC once per run.
//!
//! `GpuAligner` / `align_gpu` put the aligner's cost search on the GPU as
//! well: the Nelder-Mead control flow, the Gaussian pyramids and the final
//! per-frame warp stay on the CPU (each is cheap next to the ~200-iteration
//! cost search); every cost evaluation is one `warp_cost` launch.

use crate::align::{Sim, gauss_pyramid, inverse, nelder_mead};
use crate::fuse::{FuseParams, HALO_FLOOR, HALO_REF, binomial, fuse_residuals, halo_guide, upsample_index};
use crate::pyramid::{Img3, auto_levels, half};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

const SRC: &str = r#"
#define K0 (1.0f/16.0f)
#define K1 (4.0f/16.0f)
#define K2 (6.0f/16.0f)
#define E0 (2.0f*K0)
#define E1 (2.0f*K2)
#define OD (2.0f*K1)
__device__ __forceinline__ int refl(int i, int n){
    if(n==1) return 0;
    while(i<0 || i>=n){ if(i<0) i=-i; else i=2*(n-1)-i; }
    return i;
}
// REDUCE, horizontal: tmp (h x ow) = 5-tap at even columns
extern "C" __global__ void redH(const float* in,float* tmp,int w,int h,int ow){
    int oj=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(oj>=ow||y>=h) return; int c=2*oj; const float* r=in+(size_t)y*w;
    tmp[(size_t)y*ow+oj]= K0*r[refl(c-2,w)]+K1*r[refl(c-1,w)]+K2*r[refl(c,w)]+K1*r[refl(c+1,w)]+K0*r[refl(c+2,w)];
}
// REDUCE, vertical: out (oh x ow)
extern "C" __global__ void redV(const float* tmp,float* out,int h,int ow,int oh){
    int oj=blockIdx.x*blockDim.x+threadIdx.x, oi=blockIdx.y*blockDim.y+threadIdx.y;
    if(oj>=ow||oi>=oh) return; int c=2*oi;
    out[(size_t)oi*ow+oj]= K0*tmp[(size_t)refl(c-2,h)*ow+oj]+K1*tmp[(size_t)refl(c-1,h)*ow+oj]
        +K2*tmp[(size_t)refl(c,h)*ow+oj]+K1*tmp[(size_t)refl(c+1,h)*ow+oj]+K0*tmp[(size_t)refl(c+2,h)*ow+oj];
}
// EXPAND, horizontal: tmp (ch x ow) from coarse (ch x cw)
extern "C" __global__ void expH(const float* c,float* tmp,int cw,int ch,int ow){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow||y>=ch) return; const float* r=c+(size_t)y*cw; int i=x/2; float v;
    if((x&1)==0) v=E0*r[refl(i-1,cw)]+E1*r[refl(i,cw)]+E0*r[refl(i+1,cw)];
    else v=OD*(r[refl(i,cw)]+r[refl(i+1,cw)]);
    tmp[(size_t)y*ow+x]=v;
}
// EXPAND, vertical: out (oh x ow) from tmp (ch x ow)
extern "C" __global__ void expV(const float* tmp,float* out,int ch,int ow,int oh){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=ow||y>=oh) return; int i=y/2; float v;
    if((y&1)==0) v=E0*tmp[(size_t)refl(i-1,ch)*ow+x]+E1*tmp[(size_t)refl(i,ch)*ow+x]+E0*tmp[(size_t)refl(i+1,ch)*ow+x];
    else v=OD*(tmp[(size_t)refl(i,ch)*ow+x]+tmp[(size_t)refl(i+1,ch)*ow+x]);
    out[(size_t)y*ow+x]=v;
}
extern "C" __global__ void subk(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]-=b[i]; }
extern "C" __global__ void addk(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
extern "C" __global__ void clampk(float* a,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=fminf(fmaxf(a[i],0.f),1.f); }
extern "C" __global__ void energy_y(const float* r,const float* g,const float* b,float* e,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float y=0.299f*r[i]+0.587f*g[i]+0.114f*b[i]; e[i]=y*y;
}
extern "C" __global__ void energy_rgb(const float* r,const float* g,const float* b,float* e,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    e[i]=r[i]*r[i]+g[i]*g[i]+b[i]*b[i];
}
// separable window sum with weights wt[klen] (klen odd), reflect-101
extern "C" __global__ void winH(const float* in,float* out,int w,int h,const float* wt,int klen){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; int r=klen/2; const float* row=in+(size_t)y*w; float a=0.f;
    for(int t=0;t<klen;t++) a+=wt[t]*row[refl(x+t-r,w)];
    out[(size_t)y*w+x]=a;
}
extern "C" __global__ void winV(const float* in,float* out,int w,int h,const float* wt,int klen){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return; int r=klen/2; float a=0.f;
    for(int t=0;t<klen;t++) a+=wt[t]*in[(size_t)refl(y+t-r,h)*w+x];
    out[(size_t)y*w+x]=a;
}
// winner-take-all: where en > best, take the new coefficients (+ record idx)
extern "C" __global__ void selk(const float* en,float* best,float* a0,float* a1,float* a2,
        const float* n0,const float* n1,const float* n2,float* win,float idx,int rec,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    if(en[i]>best[i]){ best[i]=en[i]; a0[i]=n0[i]; a1[i]=n1[i]; a2[i]=n2[i]; if(rec) win[i]=idx; }
}
// halo control: the guide's weights ((en + floor) / ref)^p, exponent clamped to +-60
extern "C" __global__ void wgtk(const float* en,float* w,float p,float floor_,float ref,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    w[i]=expf(fminf(fmaxf(p*logf((en[i]+floor_)/ref),-60.f),60.f));
}
// weighted fold: acc += w * new, wsum += w
extern "C" __global__ void wacck(const float* w,float* a0,float* a1,float* a2,
        const float* n0,const float* n1,const float* n2,float* ws,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float k=w[i]; a0[i]+=k*n0[i]; a1[i]+=k*n1[i]; a2[i]+=k*n2[i]; ws[i]+=k;
}
// the weighted mean: acc /= wsum
extern "C" __global__ void wnormk(float* a0,float* a1,float* a2,const float* ws,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float k=1.f/ws[i]; a0[i]*=k; a1[i]*=k; a2[i]*=k;
}
// ---- alignment cost (fused Spline4x4 warp + DC-removed-RMS reduction) ----
// Warp `tgt` into `ref`'s frame with the inverse affine (ia..ity), and accumulate,
// over valid (in-bounds) pixels, sum(d), sum(d^2), count of d=ref-warp. The host
// turns these into rms = sqrt((Sd2 - Sd^2/n)/n) — one kernel + one readback per
// Nelder-Mead eval. Matches align.rs warp_plane + dc_removed_rms.
// FP32 per-pixel (consumer FP64 is ~1/32 rate; FP32 coords give ~0.001 px at 8K).
__device__ __forceinline__ void spl4f(float t,float* w){
    w[0]=((-1.f/3.f*t+0.8f)*t-0.46666667f)*t;
    w[1]=((t-1.8f)*t-0.2f)*t+1.f;
    w[2]=((1.2f-t)*t+0.8f)*t;
    w[3]=((1.f/3.f*t-0.2f)*t-0.13333334f)*t;
}
extern "C" __global__ void warp_cost(const float* tgt,int tw,int th,
        const float* ref,int aw,int ah,
        float ia,float ib,float itx,float id_,float ie,float ity,float ig0,float ig1,double* out){
    // float block reduction (d in [-1,1], so sums are small & exact enough), then
    // one double atomicAdd per block keeps the global sum accurate over millions of px.
    __shared__ float sd[256], sd2[256], sn[256];
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    int t=threadIdx.y*blockDim.x+threadIdx.x;
    float ld=0.f, ld2=0.f, ln=0.f;
    if(x<aw && y<ah){
        float den=ig0*x+ig1*y+1.f;   // the homography's divisor (exactly 1 for an affine transform)
        float sx=(ia*x+ib*y+itx)/den, sy=(id_*x+ie*y+ity)/den;
        if(sx>=0.f && sx<=(float)(tw-1) && sy>=0.f && sy<=(float)(th-1)){
            int x0=(int)floorf(sx), y0=(int)floorf(sy);
            float wx[4],wy[4]; spl4f(sx-x0,wx); spl4f(sy-y0,wy);
            float acc=0.f;
            for(int j=0;j<4;j++){ int yy=min(max(y0+j-1,0),th-1); float r=0.f;
                for(int i=0;i<4;i++){ int xx=min(max(x0+i-1,0),tw-1); r+=wx[i]*tgt[yy*tw+xx]; }
                acc+=wy[j]*r; }
            float d=ref[y*aw+x]-acc;
            ld=d; ld2=d*d; ln=1.f;
        }
    }
    sd[t]=ld; sd2[t]=ld2; sn[t]=ln; __syncthreads();
    for(int s=(blockDim.x*blockDim.y)>>1; s>0; s>>=1){
        if(t<s){ sd[t]+=sd[t+s]; sd2[t]+=sd2[t+s]; sn[t]+=sn[t+s]; } __syncthreads();
    }
    if(t==0){ atomicAdd(&out[0],(double)sd[0]); atomicAdd(&out[1],(double)sd2[0]); atomicAdd(&out[2],(double)sn[0]); }
}
"#;

const KERNELS: [&str; 15] = [
    "redH", "redV", "expH", "expV", "subk", "addk", "clampk", "energy_y", "energy_rgb", "winH", "winV", "selk", "wgtk", "wacck", "wnormk",
];

fn cfg2(w: usize, h: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((w as u32).div_ceil(32), (h as u32).div_ceil(8), 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 }
}
fn cfg1(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

struct Lvl {
    p: [CudaSlice<f32>; 3],
    w: usize,
    h: usize,
}

/// Stream + loaded kernels, kept apart from the buffers so a launch can borrow
/// the kernels immutably while buffers are borrowed mutably.
struct G {
    s: Arc<cudarc::driver::CudaStream>,
    k: std::collections::HashMap<&'static str, CudaFunction>,
}

impl G {
    fn f(&self, n: &'static str) -> &CudaFunction {
        &self.k[n]
    }
}

/// Init a CUDA context, compile `SRC` with NVRTC and load `names`.
fn init_gpu(names: &[&'static str]) -> Result<(Arc<CudaContext>, G), String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed (is nvidia_uvm loaded?): {e:?}"))?;
    let s = ctx.default_stream();
    // arch compute_86 so warp_cost can use double atomicAdd.
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_86"), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).map_err(|e| format!("nvrtc: {e:?}"))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("cuda module: {e:?}"))?;
    let mut k = std::collections::HashMap::new();
    for &nm in names {
        k.insert(nm, module.load_function(nm).map_err(|e| format!("kernel {nm}: {e:?}"))?);
    }
    Ok((ctx, G { s, k }))
}

/// GPU twin of `fuse::Fuser` (same `new` / `push` / `finish` contract).
pub struct GpuFuser {
    _ctx: Arc<CudaContext>,
    g: G,
    pub w: usize,
    pub h: usize,
    pub levels: usize,
    params: FuseParams,
    depth_level: usize,
    acc: Vec<Lvl>,
    /// Working pyramid of the frame being folded (reused).
    cur: Vec<Lvl>,
    /// Best energy per band-pass level; with halo control the weight sum at
    /// the levels coarser than the guide, the residual's included.
    best: Vec<CudaSlice<f32>>,
    /// Halo control: (guide level, hardness).
    halo: Option<(usize, f32)>,
    win: CudaSlice<f32>,
    wt: CudaSlice<f32>,
    klen: i32,
    /// Scratch: half-size pass buffer, full-size expand output, two energy planes.
    tmp_half: CudaSlice<f32>,
    tmp_full: CudaSlice<f32>,
    en: CudaSlice<f32>,
    en2: CudaSlice<f32>,
    tops: Vec<Img3>,
    count: usize,
}

impl GpuFuser {
    pub fn new(w: usize, h: usize, params: FuseParams) -> Result<GpuFuser, String> {
        let t = std::time::Instant::now();
        let (ctx, g) = init_gpu(&KERNELS)?;
        let s = g.s.clone();
        let levels = params.levels.unwrap_or_else(|| auto_levels(w, h, 32)).max(1);
        let depth_level = params.depth_level.min(levels - 1);
        let halo = halo_guide(&params, levels);
        let al = |n: usize| s.alloc_zeros::<f32>(n).map_err(|e| format!("cuda alloc: {e:?}"));
        let mut dims = vec![(w, h)];
        for _ in 0..levels {
            let (cw, ch) = *dims.last().unwrap();
            dims.push((half(cw), half(ch)));
        }
        let mk = |dims: &[(usize, usize)]| -> Result<Vec<Lvl>, String> {
            dims.iter().map(|&(lw, lh)| Ok(Lvl { p: [al(lw * lh)?, al(lw * lh)?, al(lw * lh)?], w: lw, h: lh })).collect()
        };
        let acc = mk(&dims)?;
        let cur = mk(&dims)?;
        let mut best = Vec::new();
        for (l, &(lw, lh)) in dims.iter().enumerate() {
            let mut b = al(lw * lh)?;
            if halo.is_none_or(|(g, _)| l <= g) {
                // any energy (>= 0) beats -1, so the first frame fills the accumulator
                let neg = vec![-1.0f32; lw * lh];
                s.memcpy_htod(&neg, &mut b).map_err(|e| format!("{e:?}"))?;
            }
            // else a weight sum, from zero
            best.push(b);
        }
        let (dw, dh) = dims[depth_level];
        let win = al(dw * dh)?;
        let wtv = binomial(params.energy_radius);
        let klen = wtv.len() as i32;
        let wt = s.memcpy_stod(&wtv).map_err(|e| format!("{e:?}"))?;
        let tmp_half = al((half(w) * h).max(half(h) * w))?;
        let tmp_full = al(w * h)?;
        let en = al(w * h)?;
        let en2 = al(w * h)?;
        eprintln!("[gpu] context + nvrtc {:.2}s", t.elapsed().as_secs_f64());
        Ok(GpuFuser {
            _ctx: ctx, g, w, h, levels, params, depth_level, acc, cur, best, halo, win, wt, klen,
            tmp_half, tmp_full, en, en2, tops: Vec::new(), count: 0,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Build the Laplacian pyramid of the frame already uploaded into `cur[0]`.
    fn build(&mut self) {
        for li in 0..self.levels {
            let (fw, fh) = (self.cur[li].w, self.cur[li].h);
            let (cw, ch) = (self.cur[li + 1].w, self.cur[li + 1].h);
            let (fwi, fhi, cwi, chi) = (fw as i32, fh as i32, cw as i32, ch as i32);
            for c in 0..3 {
                let (fine, coarse) = self.cur.split_at_mut(li + 1);
                let (fp, cp) = (&mut fine[li].p[c], &mut coarse[0].p[c]);
                // reduce
                { let mut b = self.g.s.launch_builder(self.g.f("redH")); b.arg(&*fp).arg(&mut self.tmp_half).arg(&fwi).arg(&fhi).arg(&cwi); unsafe { b.launch(cfg2(cw, fh)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("redV")); b.arg(&self.tmp_half).arg(&mut *cp).arg(&fhi).arg(&cwi).arg(&chi); unsafe { b.launch(cfg2(cw, ch)).unwrap() }; }
                // expand + subtract in place
                { let mut b = self.g.s.launch_builder(self.g.f("expH")); b.arg(&*cp).arg(&mut self.tmp_half).arg(&cwi).arg(&chi).arg(&fwi); unsafe { b.launch(cfg2(fw, ch)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("expV")); b.arg(&self.tmp_half).arg(&mut self.tmp_full).arg(&chi).arg(&fwi).arg(&fhi); unsafe { b.launch(cfg2(fw, fh)).unwrap() }; }
                let n = (fw * fh) as i32;
                { let mut b = self.g.s.launch_builder(self.g.f("subk")); b.arg(&mut *fp).arg(&self.tmp_full).arg(&n); unsafe { b.launch(cfg1(fw * fh)).unwrap() }; }
            }
        }
    }

    pub fn push(&mut self, frame: &Img3) -> Result<(), String> {
        assert!(frame.w == self.w && frame.h == self.h, "frame size mismatch");
        for c in 0..3 {
            self.g.s.memcpy_htod(&frame.p[c], &mut self.cur[0].p[c]).map_err(|e| format!("{e:?}"))?;
        }
        self.build();
        let idx = self.count as f32;
        let nsel = self.halo.map_or(self.levels, |(g, _)| g + 1);
        for li in 0..nsel {
            let (lw, lh) = (self.cur[li].w, self.cur[li].h);
            let (n, ni, wi, hi) = (lw * lh, (lw * lh) as i32, lw as i32, lh as i32);
            let [n0, n1, n2] = &self.cur[li].p;
            let ek = if self.params.use_chroma { "energy_rgb" } else { "energy_y" };
            { let mut b = self.g.s.launch_builder(self.g.f(ek)); b.arg(n0).arg(n1).arg(n2).arg(&mut self.en).arg(&ni); unsafe { b.launch(cfg1(n)).unwrap() }; }
            if self.klen > 1 {
                { let mut b = self.g.s.launch_builder(self.g.f("winH")); b.arg(&self.en).arg(&mut self.en2).arg(&wi).arg(&hi).arg(&self.wt).arg(&self.klen); unsafe { b.launch(cfg2(lw, lh)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("winV")); b.arg(&self.en2).arg(&mut self.en).arg(&wi).arg(&hi).arg(&self.wt).arg(&self.klen); unsafe { b.launch(cfg2(lw, lh)).unwrap() }; }
            }
            let rec = (li == self.depth_level) as i32;
            let [a0, a1, a2] = &mut self.acc[li].p;
            let mut b = self.g.s.launch_builder(self.g.f("selk"));
            b.arg(&self.en).arg(&mut self.best[li]).arg(a0).arg(a1).arg(a2).arg(n0).arg(n1).arg(n2).arg(&mut self.win).arg(&idx).arg(&rec).arg(&ni);
            unsafe { b.launch(cfg1(n)).unwrap() };
        }
        if let Some((guide, p)) = self.halo {
            // the guide's weights (from its region energy, still in `en`), REDUCEd
            // level by level into `en2`, fold the coarser levels and the residual
            let (mut lw, mut lh) = (self.cur[guide].w, self.cur[guide].h);
            let ni = (lw * lh) as i32;
            { let mut b = self.g.s.launch_builder(self.g.f("wgtk")); b.arg(&self.en).arg(&mut self.en2).arg(&p).arg(&HALO_FLOOR).arg(&HALO_REF).arg(&ni); unsafe { b.launch(cfg1(lw * lh)).unwrap() }; }
            for li in guide + 1..=self.levels {
                let (cw, ch) = (self.cur[li].w, self.cur[li].h);
                let (lwi, lhi, cwi, chi) = (lw as i32, lh as i32, cw as i32, ch as i32);
                { let mut b = self.g.s.launch_builder(self.g.f("redH")); b.arg(&self.en2).arg(&mut self.tmp_half).arg(&lwi).arg(&lhi).arg(&cwi); unsafe { b.launch(cfg2(cw, lh)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("redV")); b.arg(&self.tmp_half).arg(&mut self.en2).arg(&lhi).arg(&cwi).arg(&chi); unsafe { b.launch(cfg2(cw, ch)).unwrap() }; }
                let n = cw * ch;
                let ni = n as i32;
                let [n0, n1, n2] = &self.cur[li].p;
                let [a0, a1, a2] = &mut self.acc[li].p;
                let mut b = self.g.s.launch_builder(self.g.f("wacck"));
                b.arg(&self.en2).arg(a0).arg(a1).arg(a2).arg(n0).arg(n1).arg(n2).arg(&mut self.best[li]).arg(&ni);
                unsafe { b.launch(cfg1(n)).unwrap() };
                (lw, lh) = (cw, ch);
            }
        } else {
            // residual level back to the host (tiny)
            let top = &self.cur[self.levels];
            let mut t = Img3::zeros(top.w, top.h);
            for c in 0..3 {
                t.p[c] = self.g.s.memcpy_dtov(&top.p[c]).map_err(|e| format!("{e:?}"))?;
            }
            self.tops.push(t);
        }
        self.count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<(Img3, Vec<f32>), String> {
        assert!(self.count > 0, "no frames pushed");
        match self.halo {
            Some((guide, _)) => {
                // the weighted means
                for li in guide + 1..=self.levels {
                    let n = self.acc[li].w * self.acc[li].h;
                    let ni = n as i32;
                    let [a0, a1, a2] = &mut self.acc[li].p;
                    let mut b = self.g.s.launch_builder(self.g.f("wnormk"));
                    b.arg(a0).arg(a1).arg(a2).arg(&self.best[li]).arg(&ni);
                    unsafe { b.launch(cfg1(n)).unwrap() };
                }
            }
            None => {
                let top = fuse_residuals(&self.tops, &self.params);
                for c in 0..3 {
                    self.g.s.memcpy_htod(&top.p[c], &mut self.acc[self.levels].p[c]).map_err(|e| format!("{e:?}"))?;
                }
            }
        }
        // collapse: G_l = L_l + EXPAND(G_{l+1})
        for li in (0..self.levels).rev() {
            let (fw, fh) = (self.acc[li].w, self.acc[li].h);
            let (cw, ch) = (self.acc[li + 1].w, self.acc[li + 1].h);
            let (fwi, fhi, cwi, chi) = (fw as i32, fh as i32, cw as i32, ch as i32);
            for c in 0..3 {
                let (fine, coarse) = self.acc.split_at_mut(li + 1);
                let (fp, cp) = (&mut fine[li].p[c], &coarse[0].p[c]);
                { let mut b = self.g.s.launch_builder(self.g.f("expH")); b.arg(cp).arg(&mut self.tmp_half).arg(&cwi).arg(&chi).arg(&fwi); unsafe { b.launch(cfg2(fw, ch)).unwrap() }; }
                { let mut b = self.g.s.launch_builder(self.g.f("expV")); b.arg(&self.tmp_half).arg(&mut self.tmp_full).arg(&chi).arg(&fwi).arg(&fhi); unsafe { b.launch(cfg2(fw, fh)).unwrap() }; }
                let n = (fw * fh) as i32;
                { let mut b = self.g.s.launch_builder(self.g.f("addk")); b.arg(&mut *fp).arg(&self.tmp_full).arg(&n); unsafe { b.launch(cfg1(fw * fh)).unwrap() }; }
            }
        }
        let n = self.w * self.h;
        let ni = n as i32;
        let mut img = Img3::zeros(self.w, self.h);
        for c in 0..3 {
            { let mut b = self.g.s.launch_builder(self.g.f("clampk")); b.arg(&mut self.acc[0].p[c]).arg(&ni); unsafe { b.launch(cfg1(n)).unwrap() }; }
            img.p[c] = self.g.s.memcpy_dtov(&self.acc[0].p[c]).map_err(|e| format!("{e:?}"))?;
        }
        let (dw, dh) = (self.acc[self.depth_level].w, self.acc[self.depth_level].h);
        let win = self.g.s.memcpy_dtov(&self.win).map_err(|e| format!("{e:?}"))?;
        let depth = upsample_index(&win, dw, dh, self.w, self.h, self.depth_level);
        Ok((img, depth))
    }
}

/// Reusable per-pair GPU aligner: uploads the pyramid levels of one
/// (reference, target) pair and runs the Nelder-Mead cost search with the
/// warp+RMS objective on CUDA. `coarsen` skips the N finest pyramid levels,
/// exactly like the CPU path.
pub struct GpuAligner {
    _ctx: Arc<CudaContext>,
    g: G,
    out: std::cell::RefCell<CudaSlice<f64>>,
}

impl GpuAligner {
    pub fn new() -> Result<GpuAligner, String> {
        let (ctx, g) = init_gpu(&["warp_cost"])?;
        let out = std::cell::RefCell::new(g.s.alloc_zeros::<f64>(3).map_err(|e| format!("cuda alloc: {e:?}"))?);
        Ok(GpuAligner { _ctx: ctx, g, out })
    }

    pub fn align_pair(&self, prev_ref: &[f32], y: &[f32], w: usize, h: usize, guess: Sim, free: [bool; Sim::N], coarsen: usize) -> Sim {
        let stream = self.g.s.clone();
        let free_idx: Vec<usize> = (0..Sim::N).filter(|&k| free[k]).collect();
        let span = Sim::SPAN;
        let pref = gauss_pyramid(prev_ref, w, h);
        let ptgt = gauss_pyramid(y, w, h);
        let nlv = pref.len().min(ptgt.len());
        let finest = coarsen.min(nlv.saturating_sub(1));
        // upload only the levels actually refined (finest..nlv)
        let mut refd: Vec<Option<CudaSlice<f32>>> = (0..nlv).map(|_| None).collect();
        let mut tgtd: Vec<Option<CudaSlice<f32>>> = (0..nlv).map(|_| None).collect();
        for lvl in finest..nlv {
            refd[lvl] = Some(stream.memcpy_stod(&pref[lvl].0).unwrap());
            tgtd[lvl] = Some(stream.memcpy_stod(&ptgt[lvl].0).unwrap());
        }

        let iv = guess.as_vec();
        let mut cur = iv;
        let lo_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] - span[k]).collect();
        let hi_f: Vec<f64> = free_idx.iter().map(|&k| iv[k] + span[k]).collect();

        for lvl in (finest..nlv).rev() {
            let (aw, ah) = (pref[lvl].1, pref[lvl].2);
            let (tw, th) = (ptgt[lvl].1, ptgt[lvl].2);
            let rd = refd[lvl].as_ref().unwrap();
            let td = tgtd[lvl].as_ref().unwrap();
            let cur_snap = cur;
            let cost = |xf: &[f64]| -> f64 {
                let mut v = cur_snap;
                for (k, &idx) in free_idx.iter().enumerate() {
                    v[idx] = xf[k];
                }
                let inv = inverse(Sim::from_vec(&v).matrix(tw, th));
                let (twi, thi, awi, ahi) = (tw as i32, th as i32, aw as i32, ah as i32);
                let (ia, ib, itx) = (inv[0][0] as f32, inv[0][1] as f32, inv[0][2] as f32);
                let (id_, ie, ity) = (inv[1][0] as f32, inv[1][1] as f32, inv[1][2] as f32);
                let (ig0, ig1) = (inv[2][0] as f32, inv[2][1] as f32);
                let mut ob = self.out.borrow_mut();
                stream.memset_zeros(&mut *ob).unwrap();
                let mut b = stream.launch_builder(self.g.f("warp_cost"));
                b.arg(td).arg(&twi).arg(&thi).arg(rd).arg(&awi).arg(&ahi)
                    .arg(&ia).arg(&ib).arg(&itx).arg(&id_).arg(&ie).arg(&ity).arg(&ig0).arg(&ig1).arg(&mut *ob);
                unsafe { b.launch(cfg2(aw, ah)).unwrap() };
                let r = stream.memcpy_dtov(&*ob).unwrap();
                let (sd, sd2, cnt) = (r[0], r[1], r[2]);
                if cnt < 16.0 {
                    return 1e9;
                }
                ((sd2 - sd * sd / cnt) / cnt).max(0.0).sqrt()
            };
            let x0: Vec<f64> = free_idx.iter().map(|&k| cur[k]).collect();
            let best = nelder_mead(&cost, &x0, &lo_f, &hi_f);
            for (k, &idx) in free_idx.iter().enumerate() {
                cur[idx] = best[k];
            }
        }
        Sim::from_vec(&cur)
    }
}


// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! Gaussian / Laplacian pyramid (Burt & Adelson 1983; Adelson et al. 1984).
//!
//! REDUCE:  G_l(i,j) = Σ_{m,n=-2..2} w(m,n) G_{l-1}(2i+m, 2j+n)
//! EXPAND:  G*_l(i,j) = 4 Σ_{m,n} w(m,n) G_l((i+m)/2, (j+n)/2)   (integer coords only)
//! L_l = G_l − EXPAND(G_{l+1}),  L_N = G_N;  collapse reverses it exactly.
//!
//! `w` is the separable 5×5 generating kernel of Wang & Chang eq. (2),
//! [1 4 6 4 1]/16 per axis (Burt's `a` = 0.375). Borders use reflect-101
//! (…, x[2], x[1], | x[0], x[1], x[2], …), so no renormalisation is needed and
//! the transform stays a plain linear operator with exact reconstruction for
//! any image size (odd or even at every level).

use rayon::prelude::*;

/// 3-plane float image (RGB, normalised to [0, 1]). Row-major, plane-separated.
#[derive(Clone)]
pub struct Img3 {
    pub w: usize,
    pub h: usize,
    pub p: [Vec<f32>; 3],
}

impl Img3 {
    pub fn zeros(w: usize, h: usize) -> Img3 {
        Img3 { w, h, p: [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]] }
    }
    /// The `r` window of the image (`r` must lie inside it).
    pub fn crop(&self, r: &crate::align::Rect) -> Img3 {
        Img3 { w: r.w, h: r.h, p: [crop_plane(&self.p[0], self.w, r), crop_plane(&self.p[1], self.w, r), crop_plane(&self.p[2], self.w, r)] }
    }
}

/// The `r` window of a `w`-wide plane.
pub fn crop_plane<T: Copy>(p: &[T], w: usize, r: &crate::align::Rect) -> Vec<T> {
    let mut out = Vec::with_capacity(r.w * r.h);
    for y in r.y..r.y + r.h {
        out.extend_from_slice(&p[y * w + r.x..y * w + r.x + r.w]);
    }
    out
}

/// Separable generating kernel, Wang & Chang eq. (2): binomial [1 4 6 4 1]/16.
pub const KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Reflect-101 index mapping into `0..n` (−1 → 1, n → n−2). Loops so that
/// very small `n` (1..3) is also handled.
#[inline]
pub fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let n = n as isize;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// Run `f(y, row)` over the rows of `out` (width `w`), in parallel for big
/// buffers and serially for the tiny deep-pyramid arrays.
#[inline]
pub fn for_rows<F: Fn(usize, &mut [f32]) + Sync>(out: &mut [f32], w: usize, f: F) {
    if out.len() >= 1 << 15 {
        out.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, r)| f(y, r));
    } else {
        out.chunks_mut(w.max(1)).enumerate().for_each(|(y, r)| f(y, r));
    }
}

/// Size of the next-coarser level: ceil(n/2) (samples at 0, 2, 4, …).
#[inline]
pub fn half(n: usize) -> usize {
    (n + 1) / 2
}

/// REDUCE one plane: 5-tap binomial blur + 2:1 decimation (phase 0).
pub fn reduce(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (ow, oh) = (half(w), half(h));
    // horizontal: h rows × ow
    let mut tmp = vec![0f32; h * ow];
    for_rows(&mut tmp, ow, |y, row| {
        let s = &src[y * w..y * w + w];
        for (i, o) in row.iter_mut().enumerate() {
            let c = 2 * i as isize;
            let mut a = 0.0;
            for (t, k) in KERNEL.iter().enumerate() {
                a += k * s[reflect(c + t as isize - 2, w)];
            }
            *o = a;
        }
    });
    // vertical: oh rows × ow
    let mut out = vec![0f32; oh * ow];
    let tmp = &tmp;
    for_rows(&mut out, ow, |y, row| {
        let c = 2 * y as isize;
        let rows: [&[f32]; 5] = std::array::from_fn(|t| {
            let sy = reflect(c + t as isize - 2, h);
            &tmp[sy * ow..sy * ow + ow]
        });
        for (j, o) in row.iter_mut().enumerate() {
            *o = KERNEL[0] * rows[0][j]
                + KERNEL[1] * rows[1][j]
                + KERNEL[2] * rows[2][j]
                + KERNEL[3] * rows[3][j]
                + KERNEL[4] * rows[4][j];
        }
    });
    (out, ow, oh)
}

/// 1-D EXPAND weights. Even output samples sit on a coarse sample and blend
/// its two neighbours; odd samples are the midpoint of their two neighbours.
/// (2·[1 6 1]/16 and 2·[4 4]/16 — the factor 4 of eq. (5) is 2 per axis.)
const EVEN: [f32; 3] = [2.0 * KERNEL[0], 2.0 * KERNEL[2], 2.0 * KERNEL[4]];
const ODD: f32 = 2.0 * KERNEL[1];

#[inline]
fn expand_row(coarse: &[f32], cw: usize, out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = if i % 2 == 0 {
            let c = (i / 2) as isize;
            EVEN[0] * coarse[reflect(c - 1, cw)]
                + EVEN[1] * coarse[reflect(c, cw)]
                + EVEN[2] * coarse[reflect(c + 1, cw)]
        } else {
            let c = (i / 2) as isize;
            ODD * (coarse[reflect(c, cw)] + coarse[reflect(c + 1, cw)])
        };
    }
}

/// EXPAND a `cw×ch` plane to `ow×oh` (`ow` ∈ {2cw−1, 2cw}, same for rows).
pub fn expand(coarse: &[f32], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<f32> {
    debug_assert!(half(ow) == cw && half(oh) == ch, "expand: size mismatch");
    // horizontal: ch rows × ow
    let mut tmp = vec![0f32; ch * ow];
    for_rows(&mut tmp, ow, |y, row| expand_row(&coarse[y * cw..y * cw + cw], cw, row));
    // vertical: oh rows × ow
    let mut out = vec![0f32; oh * ow];
    let tmp = &tmp;
    for_rows(&mut out, ow, |i, row| {
        let c = (i / 2) as isize;
        if i % 2 == 0 {
            let (a, b, d) = (reflect(c - 1, ch), reflect(c, ch), reflect(c + 1, ch));
            let (ra, rb, rd) = (&tmp[a * ow..a * ow + ow], &tmp[b * ow..b * ow + ow], &tmp[d * ow..d * ow + ow]);
            for j in 0..ow {
                row[j] = EVEN[0] * ra[j] + EVEN[1] * rb[j] + EVEN[2] * rd[j];
            }
        } else {
            let (a, b) = (reflect(c, ch), reflect(c + 1, ch));
            let (ra, rb) = (&tmp[a * ow..a * ow + ow], &tmp[b * ow..b * ow + ow]);
            for j in 0..ow {
                row[j] = ODD * (ra[j] + rb[j]);
            }
        }
    });
    out
}

pub fn reduce3(im: &Img3) -> Img3 {
    let (p0, ow, oh) = reduce(&im.p[0], im.w, im.h);
    let (p1, _, _) = reduce(&im.p[1], im.w, im.h);
    let (p2, _, _) = reduce(&im.p[2], im.w, im.h);
    Img3 { w: ow, h: oh, p: [p0, p1, p2] }
}

pub fn expand3(im: &Img3, ow: usize, oh: usize) -> Img3 {
    Img3 {
        w: ow,
        h: oh,
        p: [
            expand(&im.p[0], im.w, im.h, ow, oh),
            expand(&im.p[1], im.w, im.h, ow, oh),
            expand(&im.p[2], im.w, im.h, ow, oh),
        ],
    }
}

/// Largest number of band-pass levels such that the residual's short side
/// stays ≥ `min_top` pixels (at least 1).
pub fn auto_levels(w: usize, h: usize, min_top: usize) -> usize {
    let mut n = 0;
    let (mut cw, mut ch) = (w, h);
    while (cw > 1 || ch > 1) && half(cw).min(half(ch)) >= min_top.max(1) {
        cw = half(cw);
        ch = half(ch);
        n += 1;
    }
    n.max(1)
}

/// Laplacian pyramid with `levels` band-pass levels: `[L_0, …, L_{N−1}, G_N]`.
pub fn build(img: &Img3, levels: usize) -> Vec<Img3> {
    let mut out = Vec::with_capacity(levels + 1);
    let mut g = img.clone();
    for _ in 0..levels {
        let next = reduce3(&g);
        let up = expand3(&next, g.w, g.h);
        for c in 0..3 {
            g.p[c].par_iter_mut().zip(up.p[c].par_iter()).for_each(|(a, b)| *a -= *b);
        }
        out.push(std::mem::replace(&mut g, next));
    }
    out.push(g);
    out
}

/// Inverse transform: expand-and-add from the residual down to level 0.
pub fn collapse(mut pyr: Vec<Img3>) -> Img3 {
    let mut g = pyr.pop().expect("empty pyramid");
    while let Some(mut l) = pyr.pop() {
        let up = expand3(&g, l.w, l.h);
        for c in 0..3 {
            l.p[c].par_iter_mut().zip(up.p[c].par_iter()).for_each(|(a, b)| *a += *b);
        }
        g = l;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise_img(w: usize, h: usize, seed: u32) -> Img3 {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 10000) as f32 / 10000.0
        };
        let mut im = Img3::zeros(w, h);
        for c in 0..3 {
            for v in im.p[c].iter_mut() {
                *v = next();
            }
        }
        im
    }

    #[test]
    fn reflect_101() {
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-2, 5), 2);
        assert_eq!(reflect(5, 5), 3);
        assert_eq!(reflect(6, 5), 2);
        assert_eq!(reflect(3, 2), 1);
        assert_eq!(reflect(-3, 1), 0);
    }

    #[test]
    fn kernel_sums_to_one() {
        let s: f32 = KERNEL.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!((EVEN.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((2.0 * ODD - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reduce_preserves_constant() {
        let mut im = Img3::zeros(37, 22);
        im.p[0].iter_mut().for_each(|v| *v = 0.7);
        let (r, ow, oh) = reduce(&im.p[0], 37, 22);
        assert_eq!((ow, oh), (19, 11));
        assert!(r.iter().all(|&v| (v - 0.7).abs() < 1e-5));
        let e = expand(&r, ow, oh, 37, 22);
        assert!(e.iter().all(|&v| (v - 0.7).abs() < 1e-5));
    }

    #[test]
    fn perfect_reconstruction() {
        for &(w, h) in &[(64usize, 48usize), (63, 47), (1, 1), (2, 3), (129, 5)] {
            let im = noise_img(w, h, 7 + w as u32);
            let levels = auto_levels(w, h, 1);
            let pyr = build(&im, levels);
            assert_eq!(pyr.len(), levels + 1);
            let back = collapse(pyr);
            for c in 0..3 {
                for (a, b) in im.p[c].iter().zip(&back.p[c]) {
                    assert!((a - b).abs() < 1e-5, "{w}x{h}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn auto_levels_targets_top_size() {
        assert_eq!(auto_levels(8280, 5520, 32), 7); // 5520 → … → 44
        assert_eq!(auto_levels(16, 16, 32), 1);
    }
}

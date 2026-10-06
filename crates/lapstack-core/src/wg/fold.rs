// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! The fold on the GPU: a frame's Laplacian pyramid, the region energy of
//! every band-pass level and the winner-take-all select into the accumulator
//! (`fuse.rs` transcribed kernel for kernel), halo control's weighted sums,
//! and the collapse. Recorded into a `Rec` over the buffers a `FoldBufs`
//! names, so a run, a slab and a refold share it.

use super::gpu::{Gpu, P, Rec, grid1, grid2};
use crate::fuse::{FuseParams, HALO_FLOOR, HALO_REF, fuse_residuals};
use crate::pyramid::Img3;

/// The buffers a fold works in — a frame's Laplacian pyramid `cur`
/// (levels + 1 buffers), the accumulator `acc` and its best-energy planes,
/// the REDUCE/EXPAND scratch and the energy plane — at the sizes in `dims`:
/// the run's own, or a refold's at the view size.
pub struct FoldBufs<'a> {
    pub dims: &'a [(usize, usize)],
    pub levels: usize,
    /// Halo control: (guide level, hardness).
    pub halo: Option<(usize, f32)>,
    pub cur: &'a [wgpu::Buffer],
    pub acc: &'a [wgpu::Buffer],
    pub best: &'a [wgpu::Buffer],
    pub tmp_half: &'a wgpu::Buffer,
    pub en: &'a wgpu::Buffer,
}

/// Record the reset of a fold's accumulators for a new fold: the best
/// energies to -1 (any energy wins, so the first frame fills the
/// accumulator); with halo control the coarser levels' weight sums and
/// accumulators to 0.
pub fn record_reset(fb: &FoldBufs<'_>, rec: &mut Rec<'_>) {
    for l in 0..=fb.levels {
        let (lw, lh) = fb.dims[l];
        let sel = fb.halo.is_none_or(|(g, _)| l <= g);
        rec.dispatch("fill", [None, None, Some(&fb.best[l]), None, None, None], P { w: (lw * lh) as u32, f0: if sel { -1.0 } else { 0.0 }, ..Default::default() }, grid1(lw * lh));
        if !sel {
            rec.dispatch("fill", [None, None, Some(&fb.acc[l]), None, None, None], P { w: (3 * lw * lh) as u32, ..Default::default() }, grid1(3 * lw * lh));
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn record_fold_in(
    fb: &FoldBufs<'_>,
    klen: u32,
    wt: &wgpu::Buffer,
    use_chroma: bool,
    rec: &mut Rec<'_>,
    scratch: &wgpu::Buffer,
    winner: Option<(usize, usize)>,
    peak: Option<&(wgpu::Buffer, usize, usize, usize, usize)>,
) {
    // build: L_l = G_l - EXPAND(REDUCE(G_l)), per plane
    for l in 0..fb.levels {
        let (fw, fh) = fb.dims[l];
        let (cw, ch) = fb.dims[l + 1];
        for c in 0..3 {
            let pr = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * fw * fh) as u32, off_out: (c * cw * ch) as u32, ..Default::default() };
            rec.dispatch("red_h", [Some(&fb.cur[l]), Some(fb.tmp_half), None, None, None, None], pr, grid2(cw, fh));
            rec.dispatch("red_v", [None, Some(fb.tmp_half), Some(&fb.cur[l + 1]), None, None, None], pr, grid2(cw, ch));
            let pe = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * cw * ch) as u32, off_out: (c * fw * fh) as u32, flag: 1, ..Default::default() };
            rec.dispatch("exp_h", [Some(&fb.cur[l + 1]), Some(fb.tmp_half), None, None, None, None], pe, grid2(fw, ch));
            rec.dispatch("exp_v", [None, Some(fb.tmp_half), Some(&fb.cur[l]), None, None, None], pe, grid2(fw, fh));
        }
    }
    // the region energy of level `l` into `en` (and the peaking map from it)
    let energy = |rec: &mut Rec<'_>, l: usize| {
        let (lw, lh) = fb.dims[l];
        let pl = P { w: lw as u32, h: lh as u32, klen, flag: use_chroma as u32, ..Default::default() };
        rec.dispatch("energy", [Some(&fb.cur[l]), None, None, Some(fb.en), None, None], pl, grid1(lw * lh));
        if klen > 1 {
            rec.dispatch("win_h", [None, Some(scratch), None, Some(fb.en), Some(wt), None], pl, grid2(lw, lh));
            rec.dispatch("win_v", [None, Some(scratch), None, Some(fb.en), Some(wt), None], pl, grid2(lw, lh));
        }
        if let Some(pk) = peak.filter(|pk| l == pk.4) {
            let (kw, kh, kf) = (pk.1, pk.2, pk.3);
            rec.dispatch(
                "down1",
                [Some(fb.en), None, Some(&pk.0), None, None, None],
                P { w: lw as u32, h: lh as u32, ow: kw as u32, oh: kh as u32, klen: kf as u32, ..Default::default() },
                grid2(kw, kh),
            );
        }
    };
    // region energy + winner-take-all per band-pass level (up to the guide with halo control)
    let nsel = fb.halo.map_or(fb.levels, |(g, _)| g + 1);
    if let Some(pk) = peak.filter(|pk| pk.4 >= nsel) {
        energy(rec, pk.4); // the peaking level beyond the guide: its energy alone
    }
    for l in 0..nsel {
        let (lw, lh) = fb.dims[l];
        energy(rec, l);
        let ps = P { w: lw as u32, h: lh as u32, flag: winner.is_some_and(|(_, dl)| l == dl) as u32, f0: winner.map_or(0.0, |(i, _)| i as f32), ..Default::default() };
        rec.dispatch("sel", [Some(&fb.cur[l]), Some(&fb.best[l]), Some(&fb.acc[l]), Some(fb.en), None, None], ps, grid1(lw * lh));
    }
    if let Some((g, p)) = fb.halo {
        // halo control: the guide's weights in place of its energy, REDUCEd level
        // by level, fold the coarser levels and the residual as Σ w·L and Σ w
        let (mut lw, mut lh) = fb.dims[g];
        rec.dispatch("wgt", [None, None, None, Some(fb.en), None, None], P { w: lw as u32, h: lh as u32, f0: p, f1: HALO_FLOOR, f2: HALO_REF, ..Default::default() }, grid1(lw * lh));
        for l in g + 1..=fb.levels {
            let (cw, ch) = fb.dims[l];
            let pr = P { w: lw as u32, h: lh as u32, ow: cw as u32, oh: ch as u32, ..Default::default() };
            rec.dispatch("red_h", [Some(fb.en), Some(fb.tmp_half), None, None, None, None], pr, grid2(cw, lh));
            rec.dispatch("red_v", [None, Some(fb.tmp_half), Some(fb.en), None, None, None], pr, grid2(cw, ch));
            rec.dispatch("wacc", [Some(&fb.cur[l]), Some(&fb.best[l]), Some(&fb.acc[l]), Some(fb.en), None, None], P { w: cw as u32, h: ch as u32, ..Default::default() }, grid1(cw * ch));
            (lw, lh) = (cw, ch);
        }
    }
}

pub fn record_collapse_in(g: &Gpu, fb: &FoldBufs<'_>, fp: &FuseParams, rec: &mut Rec<'_>, tops: &[Img3]) {
    let (w, h) = fb.dims[0];
    let n = w * h;
    match fb.halo {
        Some((guide, _)) => {
            for l in guide + 1..=fb.levels {
                let (lw, lh) = fb.dims[l];
                rec.dispatch("wnorm", [None, Some(&fb.best[l]), Some(&fb.acc[l]), None, None, None], P { w: lw as u32, h: lh as u32, ..Default::default() }, grid1(lw * lh));
            }
        }
        None => {
            let top = fuse_residuals(tops, fp);
            let (tw, th) = fb.dims[fb.levels];
            let mut flat = Vec::with_capacity(3 * tw * th);
            for c in 0..3 {
                flat.extend_from_slice(&top.p[c]);
            }
            g.queue.write_buffer(&fb.acc[fb.levels], 0, bytemuck::cast_slice(&flat));
        }
    }
    for l in (0..fb.levels).rev() {
        let (fw, fh) = fb.dims[l];
        let (cw, ch) = fb.dims[l + 1];
        for c in 0..3 {
            let pe = P { w: fw as u32, h: fh as u32, ow: cw as u32, oh: ch as u32, off_in: (c * cw * ch) as u32, off_out: (c * fw * fh) as u32, flag: 2, ..Default::default() };
            rec.dispatch("exp_h", [Some(&fb.acc[l + 1]), Some(fb.tmp_half), None, None, None, None], pe, grid2(fw, ch));
            rec.dispatch("exp_v", [None, Some(fb.tmp_half), Some(&fb.acc[l]), None, None, None], pe, grid2(fw, fh));
        }
    }
    rec.dispatch("clamp01", [None, None, Some(&fb.acc[0]), None, None, None], P { w: w as u32, h: h as u32, ..Default::default() }, grid1(3 * n));
}


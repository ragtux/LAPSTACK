// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Animated GIF export. Frames arrive from the page as RGBA8 at the output
//! size; each is quantized to its own 256-color palette (median cut on a pixel
//! sample, nearest color through a 6-bit cube cache, Floyd–Steinberg
//! dithering) and LZW-encoded. The encoded bytes are handed back after every
//! frame, so the page assembles the file as a Blob and the whole GIF never
//! sits in wasm memory (a full-resolution stack runs to gigabytes).
use std::borrow::Cow;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[derive(Clone, Default)]
struct Sink(Rc<RefCell<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&format!("gif: {e}"))
}

#[wasm_bindgen]
pub struct GifWriter {
    w: u16,
    h: u16,
    dither: bool,
    sink: Sink,
    enc: Option<gif::Encoder<Sink>>,
}

#[wasm_bindgen]
impl GifWriter {
    /// A `w`×`h` animation; `repeat` loops it forever.
    #[wasm_bindgen(constructor)]
    pub fn new(w: u32, h: u32, repeat: bool, dither: bool) -> Result<GifWriter, JsValue> {
        if w == 0 || h == 0 || w > 65535 || h > 65535 {
            return Err(err("frames must be 1..65535 px per side"));
        }
        let sink = Sink::default();
        let mut enc = gif::Encoder::new(sink.clone(), w as u16, h as u16, &[]).map_err(err)?;
        if repeat {
            enc.set_repeat(gif::Repeat::Infinite).map_err(err)?;
        }
        Ok(GifWriter { w: w as u16, h: h as u16, dither, sink, enc: Some(enc) })
    }

    /// Quantize and encode one RGBA8 frame (w·h·4 bytes) shown for `delay_cs`
    /// hundredths of a second; returns the bytes written since the last call.
    pub fn push(&mut self, rgba: &[u8], delay_cs: u16) -> Result<js_sys::Uint8Array, JsValue> {
        let (w, h) = (self.w as usize, self.h as usize);
        if rgba.len() != w * h * 4 {
            return Err(err(format!("frame is {} bytes, expected {}", rgba.len(), w * h * 4)));
        }
        let enc = self.enc.as_mut().ok_or_else(|| err("finished"))?;
        let (palette, indices) = quantize(rgba, w, h, self.dither);
        let frame = gif::Frame {
            width: self.w,
            height: self.h,
            delay: delay_cs,
            palette: Some(palette),
            buffer: Cow::Owned(indices),
            ..Default::default()
        };
        enc.write_frame(&frame).map_err(err)?;
        Ok(self.drain())
    }

    /// Write the trailer; returns the remaining bytes.
    pub fn finish(&mut self) -> Result<js_sys::Uint8Array, JsValue> {
        if let Some(enc) = self.enc.take() {
            enc.into_inner().map_err(err)?;
        }
        Ok(self.drain())
    }

    fn drain(&self) -> js_sys::Uint8Array {
        let mut v = self.sink.0.borrow_mut();
        let a = js_sys::Uint8Array::from(&v[..]);
        v.clear();
        a
    }
}

/// Per-frame palette (flat RGB) and index map.
fn quantize(rgba: &[u8], w: usize, h: usize, dither: bool) -> (Vec<u8>, Vec<u8>) {
    let n = w * h;
    let stride = (n / (1 << 18)).max(1);
    let mut samples: Vec<[u8; 3]> = (0..n).step_by(stride).map(|i| [rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2]]).collect();
    let palette = median_cut(&mut samples, 256);
    // nearest palette entry for a color, cached on its 6-bit cube cell
    let mut cache = vec![u16::MAX; 1 << 18];
    let mut nearest = |c: [i32; 3]| -> usize {
        let key = ((c[0] as usize >> 2) << 12) | ((c[1] as usize >> 2) << 6) | (c[2] as usize >> 2);
        let v = cache[key];
        if v != u16::MAX {
            return v as usize;
        }
        let (mut best, mut bd) = (0usize, i32::MAX);
        for (i, p) in palette.iter().enumerate() {
            let d = (p[0] as i32 - c[0]).pow(2) + (p[1] as i32 - c[1]).pow(2) + (p[2] as i32 - c[2]).pow(2);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        cache[key] = best as u16;
        best
    };
    let mut out = vec![0u8; n];
    if !dither {
        for i in 0..n {
            out[i] = nearest([rgba[4 * i] as i32, rgba[4 * i + 1] as i32, rgba[4 * i + 2] as i32]) as u8;
        }
    } else {
        // Floyd–Steinberg: the error of each pixel spreads 7/16 right, 3/16 down-left, 5/16 down, 1/16 down-right
        let mut cur = vec![0i32; (w + 2) * 3];
        let mut next = vec![0i32; (w + 2) * 3];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut c = [0i32; 3];
                for k in 0..3 {
                    c[k] = (rgba[4 * i + k] as i32 + cur[3 * (x + 1) + k]).clamp(0, 255);
                }
                let pi = nearest(c);
                out[i] = pi as u8;
                let p = palette[pi];
                for k in 0..3 {
                    let e = c[k] - p[k] as i32;
                    cur[3 * (x + 2) + k] += e * 7 / 16;
                    next[3 * x + k] += e * 3 / 16;
                    next[3 * (x + 1) + k] += e * 5 / 16;
                    next[3 * (x + 2) + k] += e / 16;
                }
            }
            std::mem::swap(&mut cur, &mut next);
            next.iter_mut().for_each(|v| *v = 0);
        }
    }
    (palette.iter().flatten().copied().collect(), out)
}

/// Median cut: split the box with the largest (color span × population) at
/// the median of its widest channel until there are `k` boxes; a palette entry
/// is the mean color of a box.
fn median_cut(px: &mut [[u8; 3]], k: usize) -> Vec<[u8; 3]> {
    if px.is_empty() {
        return vec![[0, 0, 0]];
    }
    let span = |s: &[[u8; 3]]| -> ([u8; 3], [u8; 3]) {
        let (mut lo, mut hi) = ([255u8; 3], [0u8; 3]);
        for p in s {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    };
    let mut boxes: Vec<(usize, usize)> = vec![(0, px.len())];
    while boxes.len() < k {
        let mut pick: Option<(usize, usize, u64)> = None; // (box, channel, score)
        for (bi, &(lo, hi)) in boxes.iter().enumerate() {
            if hi - lo < 2 {
                continue;
            }
            let (mn, mx) = span(&px[lo..hi]);
            let (ch, sp) = (0..3).map(|c| (c, (mx[c] - mn[c]) as u64)).max_by_key(|&(_, s)| s).unwrap();
            if sp == 0 {
                continue;
            }
            let score = sp * (hi - lo) as u64;
            if pick.is_none_or(|(_, _, best)| score > best) {
                pick = Some((bi, ch, score));
            }
        }
        let Some((bi, ch, _)) = pick else { break };
        let (lo, hi) = boxes[bi];
        px[lo..hi].sort_unstable_by_key(|p| p[ch]);
        let mid = lo + (hi - lo) / 2;
        boxes[bi] = (lo, mid);
        boxes.push((mid, hi));
    }
    boxes
        .iter()
        .map(|&(lo, hi)| {
            let s = &px[lo..hi];
            let mut acc = [0u64; 3];
            for p in s {
                for k in 0..3 {
                    acc[k] += p[k] as u64;
                }
            }
            let n = s.len().max(1) as u64;
            [((acc[0] + n / 2) / n) as u8, ((acc[1] + n / 2) / n) as u8, ((acc[2] + n / 2) / n) as u8]
        })
        .collect()
}

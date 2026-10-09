// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Decode PNG/JPEG/TIFF bytes — or develop a camera raw's (`lapstack_core::raw`) —
//! to interleaved RGB u16 (8-bit inputs widened by ×257), the form the `warp`
//! kernel samples directly.

use image::ColorType;
use std::io::Cursor;

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub bits: u32,
    /// RGB interleaved, w*h*3 samples, padded to an even count.
    pub rgb: Vec<u16>,
}

/// `raw`: the bytes are a camera raw's (by the file's extension: a NEF or a DNG
/// is a TIFF container the image crate would take for its thumbnail).
pub fn decode_any(bytes: &[u8], raw: bool) -> Result<Frame, String> {
    if raw {
        return Ok(frame_of(lapstack_core::raw::develop(bytes)?));
    }
    decode(bytes)
}

/// The pixel size of a frame without decoding it (a raw is developed for it:
/// its size is only sure once turned by its orientation).
pub fn dims(bytes: &[u8], raw: bool) -> Result<(usize, usize), String> {
    if raw {
        let f = decode_any(bytes, true)?;
        return Ok((f.w, f.h));
    }
    let reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| format!("unrecognised image: {e}"))?;
    let (w, h) = reader.into_dimensions().map_err(|e| format!("decode: {e}"))?;
    Ok((w as usize, h as usize))
}

pub fn decode(bytes: &[u8]) -> Result<Frame, String> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("unrecognised image: {e}"))?;
    reader.no_limits();
    let img = reader.decode().map_err(|e| format!("decode: {e}"))?;
    Ok(frame_of(img))
}

pub fn frame_of(img: image::DynamicImage) -> Frame {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bits = match img.color() {
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => 16,
        _ => 8,
    };
    let n = w * h * 3;
    let mut rgb: Vec<u16> = if bits == 16 {
        img.into_rgb16().into_raw()
    } else {
        img.into_rgb8().into_raw().into_iter().map(|v| v as u16 * 257).collect()
    };
    if n % 2 == 1 {
        rgb.push(0);
    }
    Frame { w, h, bits, rgb }
}

/// The frame turned by `quarters` quarter turns clockwise (`--rotate`).
pub fn rotate(f: &Frame, quarters: u8) -> Frame {
    let q = quarters % 4;
    if q == 0 {
        return Frame { w: f.w, h: f.h, bits: f.bits, rgb: f.rgb.clone() };
    }
    let (w, h) = (f.w, f.h);
    let (ow, oh) = if q % 2 == 1 { (h, w) } else { (w, h) };
    let mut rgb = vec![0u16; (ow * oh * 3 + 1) & !1];
    for oy in 0..oh {
        for ox in 0..ow {
            let (sx, sy) = match q {
                1 => (oy, h - 1 - ox),
                2 => (w - 1 - ox, h - 1 - oy),
                _ => (w - 1 - oy, ox),
            };
            let (s, d) = (3 * (sy * w + sx), 3 * (oy * ow + ox));
            rgb[d..d + 3].copy_from_slice(&f.rgb[s..s + 3]);
        }
    }
    Frame { w: ow, h: oh, bits: f.bits, rgb }
}

/// The frame block-averaged by `2^levels` (a draft run's frames), the ragged
/// last row and column averaging what is there.
pub fn reduce(f: &Frame, levels: usize) -> Frame {
    if levels == 0 {
        return Frame { w: f.w, h: f.h, bits: f.bits, rgb: f.rgb.clone() };
    }
    let k = 1usize << levels;
    let (w, h) = (f.w, f.h);
    let (ow, oh) = (w.div_ceil(k).max(1), h.div_ceil(k).max(1));
    let mut rgb = vec![0u16; (ow * oh * 3 + 1) & !1];
    for oy in 0..oh {
        let (y0, y1) = (oy * k, ((oy + 1) * k).min(h));
        for ox in 0..ow {
            let (x0, x1) = (ox * k, ((ox + 1) * k).min(w));
            let mut s = [0u64; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = 3 * (y * w + x);
                    s[0] += f.rgb[p] as u64;
                    s[1] += f.rgb[p + 1] as u64;
                    s[2] += f.rgb[p + 2] as u64;
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u64;
            let d = 3 * (oy * ow + ox);
            for c in 0..3 {
                rgb[d + c] = ((s[c] + n / 2) / n) as u16;
            }
        }
    }
    Frame { w: ow, h: oh, bits: f.bits, rgb }
}

/// The frame resampled bilinearly to `w × h` (a frame of another size than
/// the stack's, brought to it; each axis on its own).
pub fn resize(f: &Frame, w: usize, h: usize) -> Frame {
    if f.w == w && f.h == h {
        return Frame { w, h, bits: f.bits, rgb: f.rgb.clone() };
    }
    let (sw, sh) = (f.w, f.h);
    let (kx, ky) = (sw as f64 / w as f64, sh as f64 / h as f64);
    let mut rgb = vec![0u16; (w * h * 3 + 1) & !1];
    for y in 0..h {
        let fy = ((y as f64 + 0.5) * ky - 0.5).clamp(0.0, (sh - 1) as f64);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(sh - 1);
        for x in 0..w {
            let fx = ((x as f64 + 0.5) * kx - 0.5).clamp(0.0, (sw - 1) as f64);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(sw - 1);
            let (a, b, c, d) = (3 * (y0 * sw + x0), 3 * (y0 * sw + x1), 3 * (y1 * sw + x0), 3 * (y1 * sw + x1));
            let o = 3 * (y * w + x);
            for ch in 0..3 {
                let top = f.rgb[a + ch] as f64 * (1.0 - tx) + f.rgb[b + ch] as f64 * tx;
                let bot = f.rgb[c + ch] as f64 * (1.0 - tx) + f.rgb[d + ch] as f64 * tx;
                rgb[o + ch] = (top * (1.0 - ty) + bot * ty + 0.5) as u16;
            }
        }
    }
    Frame { w, h, bits: f.bits, rgb }
}

/// A float image (the look space of a linear-DNG run, `lapstack_core::dng`)
/// as a 16-bit frame, clipped to [0, 1]: the browser's frames are 16-bit, so
/// a highlight past white is lost here where the native path keeps it.
pub fn of_img3(img: &lapstack_core::Img3) -> Frame {
    let n = img.w * img.h;
    let mut rgb = vec![0u16; (n * 3 + 1) & !1];
    for i in 0..n {
        for c in 0..3 {
            rgb[3 * i + c] = (img.p[c][i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    Frame { w: img.w, h: img.h, bits: 16, rgb }
}

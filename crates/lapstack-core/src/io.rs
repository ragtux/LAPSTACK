// Copyright (c) 2026 MATCHMUSEUM.COM
// INTERNAL USE ONLY
//
// Image I/O with bit-depth preservation.
//
// Frames are decoded into a normalized f32 `Img3` and all fusion runs in float,
// so the pipeline itself is depth-agnostic. This module only tracks the native
// depth of the inputs so the result can be written back at the same depth:
//   8-bit in  -> 8-bit out
//   16-bit in -> 16-bit out  (PNG/TIFF; JPEG has no 16-bit -> written 8-bit)

use crate::pyramid::Img3;
use image::{ColorType, DynamicImage, ImageBuffer, Luma, Rgb};
use rayon::prelude::*;

/// Native storage depth of an image file.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    Eight,
    Sixteen,
}

impl Depth {
    pub fn bits(self) -> u32 {
        match self {
            Depth::Eight => 8,
            Depth::Sixteen => 16,
        }
    }

    fn of(img: &DynamicImage) -> Depth {
        match img.color() {
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => Depth::Sixteen,
            _ => Depth::Eight,
        }
    }
}

/// Whether the output format (by extension) can store 16-bit samples.
fn ext_supports_16(path: &str) -> bool {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    matches!(ext.as_str(), "png" | "tif" | "tiff")
}

/// Scatter interleaved samples (R,G,B are the first 3 of `stride` channels) into
/// the planar f32 image, scaled by `inv`. Parallel per plane; `stride` lets us read
/// an RGBA buffer in place and drop alpha, skipping an RGBA->RGB repack copy.
fn fill<T: Copy + Into<f32> + Sync>(o: &mut Img3, raw: &[T], stride: usize, inv: f32) {
    for (c, plane) in o.p.iter_mut().enumerate() {
        plane.par_iter_mut().enumerate().for_each(|(i, out)| {
            let v: f32 = raw[stride * i + c].into();
            *out = v * inv;
        });
    }
}

/// Decode an image into a normalized f32 `Img3`, returning its native depth.
pub fn load_rgb(path: &str) -> Result<(Img3, Depth), String> {
    // Use a reader with limits disabled: the default decode memory cap rejects
    // full-resolution 16-bit frames (a 45 MP RGB16 image is ~270 MB decoded).
    let mut reader = image::ImageReader::open(path)
        .map_err(|e| format!("cannot open {path}: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    reader.no_limits();
    let dynimg = reader.decode().map_err(|e| format!("cannot decode {path}: {e}"))?;
    let depth = Depth::of(&dynimg);
    let cc = dynimg.color().channel_count() as usize;
    let (w, h) = (dynimg.width() as usize, dynimg.height() as usize);
    let mut o = Img3::zeros(w, h);
    // Read the decoded buffer in place. into_rgbaN/into_rgbN is a no-op when the
    // image is already that type (the common RGB/RGBA case), so we stride over the
    // native channel count instead of forcing an RGBA->RGB repack. Non-RGB inputs
    // (luma) are rare and fall back to a converting into_rgbN.
    let (inv16, inv8) = (1.0 / 65535.0, 1.0 / 255.0);
    match (depth, cc) {
        (Depth::Sixteen, 4) => fill(&mut o, &dynimg.into_rgba16().into_raw(), 4, inv16),
        (Depth::Sixteen, _) => fill(&mut o, &dynimg.into_rgb16().into_raw(), 3, inv16),
        (Depth::Eight, 4) => fill(&mut o, &dynimg.into_rgba8().into_raw(), 4, inv8),
        (Depth::Eight, _) => fill(&mut o, &dynimg.into_rgb8().into_raw(), 3, inv8),
    }
    Ok((o, depth))
}

/// Write an `Img3` at the requested depth, downgrading to 8-bit if the output
/// format cannot store 16-bit samples.
pub fn save_rgb(img: &Img3, path: &str, depth: Depth) -> Result<(), String> {
    let depth = if depth == Depth::Sixteen && ext_supports_16(path) { Depth::Sixteen } else { Depth::Eight };
    let n = img.w * img.h;
    let (w, h) = (img.w as u32, img.h as u32);
    match depth {
        Depth::Sixteen => {
            let mut raw = vec![0u16; n * 3];
            for i in 0..n {
                for c in 0..3 {
                    raw[3 * i + c] = (img.p[c][i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
            ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, raw)
                .unwrap()
                .save(path)
                .map_err(|e| format!("cannot write {path}: {e}"))?;
        }
        Depth::Eight => {
            let mut raw = vec![0u8; n * 3];
            for i in 0..n {
                for c in 0..3 {
                    raw[3 * i + c] = (img.p[c][i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
            ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, raw)
                .unwrap()
                .save(path)
                .map_err(|e| format!("cannot write {path}: {e}"))?;
        }
    }
    Ok(())
}

/// Save a plane in `[0, scale]` as a 16-bit grayscale PNG with a FIXED
/// mapping (`scale` → 65535), so the values can be read back exactly: depth
/// maps use `scale = n_frames − 1`, confidence maps `scale = 1`.
pub fn save_gray16(plane: &[f32], w: usize, h: usize, scale: f32, path: &str) -> Result<(), String> {
    let k = 65535.0 / scale.max(1e-6);
    let raw: Vec<u16> = plane.iter().map(|&v| (v.clamp(0.0, scale) * k + 0.5) as u16).collect();
    ImageBuffer::<Luma<u16>, _>::from_raw(w as u32, h as u32, raw)
        .unwrap()
        .save(path)
        .map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(())
}

/// Save a normalized depth map as an 8-bit grayscale image (a visualization,
/// so 8-bit is sufficient regardless of the frame depth).
pub fn save_gray(depth: &[f32], w: usize, h: usize, path: &str) -> Result<(), String> {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &d in depth {
        lo = lo.min(d);
        hi = hi.max(d);
    }
    let range = (hi - lo).max(1e-6);
    let raw: Vec<u8> = depth.iter().map(|&d| (((d - lo) / range) * 255.0 + 0.5) as u8).collect();
    image::GrayImage::from_raw(w as u32, h as u32, raw)
        .unwrap()
        .save(path)
        .map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(())
}

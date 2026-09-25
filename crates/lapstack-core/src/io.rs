// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: AGPL-3.0-only

// Image I/O with bit-depth preservation.
//
// Frames are decoded into a normalized f32 `Img3` and all fusion runs in float,
// so the pipeline itself is depth-agnostic. This module only tracks the native
// depth of the inputs so the result can be written back at the same depth:
//   8-bit in  -> 8-bit out
//   16-bit in -> 16-bit out  (PNG/TIFF; JPEG has no 16-bit -> written 8-bit)

use crate::dng::DngInfo;
use crate::meta::{self, Meta};
use crate::pyramid::Img3;
use image::{ColorType, DynamicImage, ImageBuffer, Luma, Rgb};
use rayon::prelude::*;
use std::io::Cursor;

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
/// A camera raw (`raw::is_raw`) is developed (`raw::develop`): 16-bit.
pub fn load_rgb(path: &str) -> Result<(Img3, Depth), String> {
    if crate::raw::is_raw(path) {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let dynimg = crate::raw::develop(&bytes).map_err(|e| format!("cannot develop {path}: {e}"))?;
        return Ok(img3_of(dynimg));
    }
    // Use a reader with limits disabled: the default decode memory cap rejects
    // full-resolution 16-bit frames (a 45 MP RGB16 image is ~270 MB decoded).
    let mut reader = image::ImageReader::open(path)
        .map_err(|e| format!("cannot open {path}: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    reader.no_limits();
    let dynimg = reader.decode().map_err(|e| format!("cannot decode {path}: {e}"))?;
    Ok(img3_of(dynimg))
}

/// A decoded image as a normalized f32 `Img3` and its native depth.
fn img3_of(dynimg: DynamicImage) -> (Img3, Depth) {
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
    (o, depth)
}

/// Decode a frame for a linear-DNG run (`dng.rs`): a camera raw is developed
/// to its linear camera space and taken to the look space with `look`'s white
/// balance and matrix (frame 0's, so every frame is developed alike); any
/// other file is loaded as it is, its sRGB values being the look already. The
/// frame's own space comes back with it, so the caller can see that a raw and
/// a non-raw were mixed.
pub fn load_look(path: &str, look: &DngInfo) -> Result<(Img3, Depth, DngInfo), String> {
    if crate::raw::is_raw(path) {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let (mut img, own) = crate::raw::develop_linear(&bytes).map_err(|e| format!("cannot develop {path}: {e}"))?;
        look.to_look(&mut img);
        return Ok((img, Depth::Sixteen, own));
    }
    let (img, depth) = load_rgb(path)?;
    Ok((img, depth, DngInfo::srgb()))
}

/// Frame 0 of a linear-DNG run: its space is the run's.
pub fn load_look_first(path: &str) -> Result<(Img3, Depth, DngInfo), String> {
    if crate::raw::is_raw(path) {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let (mut img, info) = crate::raw::develop_linear(&bytes).map_err(|e| format!("cannot develop {path}: {e}"))?;
        info.to_look(&mut img);
        return Ok((img, Depth::Sixteen, info));
    }
    let (img, depth) = load_rgb(path)?;
    Ok((img, depth, DngInfo::srgb()))
}

/// Read the metadata (EXIF, ICC profile, XMP) of an input file, see `meta`.
/// A camera raw's frame is turned by its orientation as it is decoded, so the
/// orientation the metadata carries into the output is set to 1.
pub fn load_meta(path: &str) -> Result<Meta, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut m = meta::extract(&bytes);
    if crate::raw::is_raw(path) {
        if let Some(e) = m.exif.as_deref().and_then(|e| meta::with_orientation(e, 1)) {
            m.exif = Some(e);
        }
    }
    Ok(m)
}

/// Write the look image as a linear DNG (`dng::write`); `info` maps it back
/// to the camera space and names the camera.
pub fn save_dng(img: &Img3, path: &str, info: &DngInfo, meta: Option<&Meta>) -> Result<(), String> {
    let mut out: Vec<u8> = Vec::new();
    crate::dng::write(&mut out, img, info, meta).map_err(|e| format!("cannot write {path}: {e}"))?;
    std::fs::write(path, out).map_err(|e| format!("cannot write {path}: {e}"))
}

/// Whether an output path asks for a linear DNG.
pub fn is_dng(path: &str) -> bool {
    path.rsplit('.').next().is_some_and(|e| e.eq_ignore_ascii_case("dng"))
}

/// The capture time of a frame (`Meta::capture_time`), reading as little of the
/// file as will do. The first 4 MB hold the EXIF of a JPEG, a PNG and a TIFF
/// whose IFD leads; a TIFF whose IFD was written after the pixels (a 270 MB
/// file with its IFD in the last 40 KB is the usual raw export) is read in a
/// window around that IFD, into a buffer sparse everywhere else so the
/// structure's absolute offsets hold. Only when both yield nothing is the whole
/// file read (a PNG with its chunks after the image data).
pub fn load_capture_time(path: &str) -> Result<Option<f64>, String> {
    use std::io::{Read, Seek, SeekFrom};
    // a raw: its TIFF structure, where it has one (NEF, CR2, ARW, DNG, …), else its own reader
    if crate::raw::is_raw(path) {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        return Ok(meta::extract(&bytes).capture_time().or_else(|| crate::raw::capture_time(&bytes)));
    }
    const HEAD: u64 = 4 << 20;
    const WINDOW: u64 = 4 << 20;
    let err = |e: std::io::Error| format!("cannot read {path}: {e}");
    let mut f = std::fs::File::open(path).map_err(err)?;
    let len = f.metadata().map_err(err)?.len();
    let mut head = Vec::with_capacity(len.min(HEAD) as usize);
    (&mut f).take(HEAD).read_to_end(&mut head).map_err(err)?;
    if let Some(t) = meta::extract(&head).capture_time() {
        return Ok(Some(t));
    }
    if len <= HEAD {
        return Ok(None);
    }
    if let Some(off) = meta::tiff_ifd0_offset(&head).map(u64::from).filter(|&o| o >= HEAD && o < len) {
        let (lo, hi) = (off.saturating_sub(WINDOW / 2), (off + WINDOW / 2).min(len));
        let mut sparse = vec![0u8; hi as usize];   // zeroed pages are not committed until written
        sparse[..head.len()].copy_from_slice(&head);
        f.seek(SeekFrom::Start(lo)).map_err(err)?;
        f.read_exact(&mut sparse[lo as usize..hi as usize]).map_err(err)?;
        if let Some(t) = meta::extract(&sparse).capture_time() {
            return Ok(Some(t));
        }
    }
    let mut whole = head;
    f.seek(SeekFrom::Start(HEAD)).map_err(err)?;
    f.read_to_end(&mut whole).map_err(err)?;
    Ok(meta::extract(&whole).capture_time())
}

/// The focus distance a frame's metadata records, in metres: the EXIF
/// SubjectDistance (a raw's own reader when its EXIF is not a TIFF
/// structure), read like `load_capture_time` reads the time — the head of the
/// file, or a window around a trailing IFD; failing that, what `exiftool`
/// finds when it is on the path, since most cameras write the focus distance
/// into their MakerNote alone (Nikon's is even encrypted), which exiftool
/// decodes and nothing here does. `None` when nothing records one.
pub fn load_subject_distance(path: &str) -> Option<f64> {
    use std::io::{Read, Seek, SeekFrom};
    if crate::raw::is_raw(path) {
        let bytes = std::fs::read(path).ok()?;
        return meta::subject_distance(&meta::extract(&bytes)).or_else(|| crate::raw::subject_distance(&bytes)).or_else(|| exiftool_distance(path));
    }
    const HEAD: u64 = 4 << 20;
    const WINDOW: u64 = 4 << 20;
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let mut head = Vec::with_capacity(len.min(HEAD) as usize);
    (&mut f).take(HEAD).read_to_end(&mut head).ok()?;
    if let Some(v) = meta::subject_distance(&meta::extract(&head)) {
        return Some(v);
    }
    if len <= HEAD {
        return None;
    }
    if let Some(off) = meta::tiff_ifd0_offset(&head).map(u64::from).filter(|&o| o >= HEAD && o < len) {
        let (lo, hi) = (off.saturating_sub(WINDOW / 2), (off + WINDOW / 2).min(len));
        let mut sparse = vec![0u8; hi as usize];
        sparse[..head.len()].copy_from_slice(&head);
        f.seek(SeekFrom::Start(lo)).ok()?;
        f.read_exact(&mut sparse[lo as usize..hi as usize]).ok()?;
        if let Some(v) = meta::subject_distance(&meta::extract(&sparse)) {
            return Some(v);
        }
    }
    exiftool_distance(path)
}

/// The focus distance by `exiftool -n`, the first of the tags the makers use
/// (SubjectDistance, FocusDistance, Nikon's LensData FocusDistance, Canon's
/// FocusDistanceUpper / Lower averaged, Sony's FocusDistance2) that carries a
/// finite positive number. `None` without exiftool on the path or without
/// such a tag; one probe per process decides whether exiftool is there.
pub fn exiftool_distance(path: &str) -> Option<f64> {
    use std::sync::OnceLock;
    static HAVE: OnceLock<bool> = OnceLock::new();
    if !*HAVE.get_or_init(|| std::process::Command::new("exiftool").arg("-ver").output().is_ok_and(|o| o.status.success())) {
        return None;
    }
    let out = std::process::Command::new("exiftool")
        .args(["-n", "-s", "-SubjectDistance", "-FocusDistance", "-FocusDistanceUpper", "-FocusDistanceLower", "-FocusDistance2", "-ApproximateFocusDistance", path])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut upper = None;
    let mut lower = None;
    let mut first = None;
    for line in text.lines() {
        let Some((tag, v)) = line.split_once(':') else { continue };
        let v: f64 = match v.trim().parse::<f64>() { Ok(v) if v.is_finite() && v > 0.0 && v < 1e5 => v, _ => continue };
        match tag.trim() {
            "FocusDistanceUpper" => upper = Some(v),
            "FocusDistanceLower" => lower = Some(v),
            _ => first = first.or(Some(v)),
        }
    }
    first.or(match (upper, lower) { (Some(u), Some(l)) => Some((u + l) / 2.0), (u, l) => u.or(l) })
}

/// Write an `Img3` at the requested depth, downgrading to 8-bit if the output
/// format cannot store 16-bit samples. `meta` is carried into the file: PNG
/// and JPEG through `meta::embed`, TIFF through lapstack's own writer.
pub fn save_rgb(img: &Img3, path: &str, depth: Depth, meta: Option<&Meta>) -> Result<(), String> {
    let depth = if depth == Depth::Sixteen && ext_supports_16(path) { Depth::Sixteen } else { Depth::Eight };
    let n = img.w * img.h;
    let (w, h) = (img.w as u32, img.h as u32);
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let err = |e: String| format!("cannot write {path}: {e}");
    let tiff = matches!(ext.as_str(), "tif" | "tiff");
    let format = image::ImageFormat::from_extension(&ext).ok_or_else(|| err("unknown extension".into()))?;
    let mut out: Vec<u8> = Vec::new();
    match depth {
        Depth::Sixteen => {
            let mut raw = vec![0u16; n * 3];
            for i in 0..n {
                for c in 0..3 {
                    raw[3 * i + c] = (img.p[c][i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
            if tiff {
                meta::write_tiff(&mut out, img.w, img.h, 16, bytemuck_cast(&raw), meta).map_err(|e| err(e.to_string()))?;
            } else {
                ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, raw).unwrap().write_to(&mut Cursor::new(&mut out), format).map_err(|e| err(e.to_string()))?;
            }
        }
        Depth::Eight => {
            let mut raw = vec![0u8; n * 3];
            for i in 0..n {
                for c in 0..3 {
                    raw[3 * i + c] = (img.p[c][i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
            if tiff {
                meta::write_tiff(&mut out, img.w, img.h, 8, &raw, meta).map_err(|e| err(e.to_string()))?;
            } else {
                ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, raw).unwrap().write_to(&mut Cursor::new(&mut out), format).map_err(|e| err(e.to_string()))?;
            }
        }
    }
    if let (Some(m), false) = (meta, tiff) {
        out = meta::embed(out, m);
    }
    std::fs::write(path, out).map_err(|e| err(e.to_string()))
}

/// u16 samples as bytes in host order (what `meta::write_tiff` takes).
fn bytemuck_cast(v: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding and any byte pattern is a valid u8; the slice covers the same memory
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
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

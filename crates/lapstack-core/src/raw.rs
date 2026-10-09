// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Camera raw files as input frames: decoded, black and white levels applied,
//! demosaicked, white-balanced as shot, taken through the camera's color
//! matrix to sRGB and given the sRGB curve — the 16-bit RGB every other input
//! becomes, turned the way the camera said. There is no exposure or tone
//! adjustment: the frames of a stack are shot alike, and what matters to the
//! stack is that they are developed alike; when the look matters, a stack
//! developed in a raw converter first, with its exposure and profile, is the
//! better input, and lapstack takes its TIFFs. The JPEG preview the camera
//! wrote into the file serves the thumbnails, and the file's own EXIF the
//! capture time of the batch split.
//!
//! The decoding itself is not in lapstack. It is `lapstack-raw`
//! (`crates/lapstack-raw`, LGPL-2.1: rawler, dnglab's library, behind a small
//! interface), a component lapstack loads at run time and the user may
//! replace: natively a shared library found beside the binary (`dylib`
//! below, `--features raw`), in the browser a wasm module of its own that the
//! worker imports and the engine reaches through the global object
//! (`lapstack-web`'s `raw_bridge`). Either installs itself here as the
//! [`RawBackend`]; without one, a raw is an error that says so, and every
//! other input works.

use crate::dng::DngInfo;
use crate::pyramid::Img3;
use image::DynamicImage;
use serde::Deserialize;
use std::sync::OnceLock;

/// The revision of the interface this client speaks (`lapstack-raw`'s `ABI`).
pub const ABI: u32 = 1;

/// The file extensions rawler decodes, lower case.
pub const EXTENSIONS: &[&str] = &[
    "ari", "arw", "cr2", "cr3", "crm", "crw", "dcr", "dcs", "dng", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "ori",
    "pef", "raf", "raw", "rw2", "rwl", "srw", "3fr", "fff", "x3f", "qtk",
];

/// Whether a file name is a camera raw's, by its extension.
pub fn is_raw(name: &str) -> bool {
    match name.rsplit_once('.') {
        Some((_, ext)) => EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// The pixels a backend hands over (the module's layouts, copied into our memory).
pub enum Pixels {
    /// 16-bit gray, `w*h` (a monochrome sensor).
    Gray16(Vec<u16>),
    /// 16-bit RGB interleaved, `w*h*3`.
    Rgb16(Vec<u16>),
    /// 8-bit RGB interleaved, `w*h*3` (a preview).
    Rgb8(Vec<u8>),
    /// Three f32 planes R, G, B, each `w*h` (the linear development).
    Planes3([Vec<f32>; 3]),
}

/// A developed image as a backend returns it.
pub struct RawImage {
    pub w: usize,
    pub h: usize,
    pub px: Pixels,
    /// How to turn a linear development (quarter turns clockwise after a
    /// mirror); the others come turned.
    pub turns: u8,
    pub flip: bool,
    /// A linear development's camera color.
    pub color: Option<RawColor>,
}

/// What a linear DNG needs to know about the camera (the module's `Color` JSON).
#[derive(Deserialize)]
pub struct RawColor {
    pub make: String,
    pub model: String,
    pub matrices: Vec<(u16, [f32; 9])>,
    pub wb: [f32; 3],
    pub d65: [f32; 9],
}

/// What a raw's own EXIF says (the module's `Metadata` JSON).
#[derive(Deserialize)]
pub struct RawMeta {
    pub orientation: u16,
    pub date_time_original: Option<String>,
    pub sub_sec_time_original: Option<String>,
    pub create_date: Option<String>,
    pub sub_sec_time_digitized: Option<String>,
    pub subject_distance: Option<f64>,
}

/// The raw decoder, wherever it lives.
pub trait RawBackend: Send + Sync {
    /// `develop` (turned 16-bit RGB or gray) or, `linear`, `develop_linear`
    /// (three f32 planes, not turned, with the camera's color).
    fn develop(&self, bytes: &[u8], linear: bool) -> Result<RawImage, String>;
    /// The camera's own JPEG preview, turned, 8-bit RGB.
    fn preview(&self, bytes: &[u8]) -> Option<RawImage>;
    fn metadata(&self, bytes: &[u8]) -> Option<RawMeta>;
    /// Where the decoder is, for the log.
    fn describe(&self) -> String;
}

static BACKEND: OnceLock<Result<Box<dyn RawBackend>, String>> = OnceLock::new();

/// Install the decoder (the browser engine does, at start-up). False when one
/// is installed already.
pub fn set_backend(b: Box<dyn RawBackend>) -> bool {
    BACKEND.set(Ok(b)).is_ok()
}

/// The decoder: the installed one, else the shared library found on first
/// use, else the error that says why there is none.
fn backend() -> Result<&'static dyn RawBackend, String> {
    match BACKEND.get_or_init(load_native) {
        Ok(b) => Ok(b.as_ref()),
        Err(e) => Err(e.clone()),
    }
}

/// Which decoder is in use, or why none is — tried, so a native build loads
/// the library to answer.
pub fn describe() -> Result<String, String> {
    backend().map(|b| b.describe())
}

#[cfg(all(feature = "raw", not(target_arch = "wasm32")))]
fn load_native() -> Result<Box<dyn RawBackend>, String> {
    dylib::load().map(|d| Box::new(d) as Box<dyn RawBackend>)
}
#[cfg(not(all(feature = "raw", not(target_arch = "wasm32"))))]
fn load_native() -> Result<Box<dyn RawBackend>, String> {
    Err("this build has no raw support (no raw decoder is installed)".to_string())
}

fn image_of(img: RawImage) -> Result<DynamicImage, String> {
    let (w, h) = (img.w as u32, img.h as u32);
    let bad = || "raw: the decoder returned an image of the wrong size".to_string();
    match img.px {
        Pixels::Gray16(v) => image::ImageBuffer::from_raw(w, h, v).map(DynamicImage::ImageLuma16).ok_or_else(bad),
        Pixels::Rgb16(v) => image::ImageBuffer::from_raw(w, h, v).map(DynamicImage::ImageRgb16).ok_or_else(bad),
        Pixels::Rgb8(v) => image::ImageBuffer::from_raw(w, h, v).map(DynamicImage::ImageRgb8).ok_or_else(bad),
        Pixels::Planes3(_) => Err("raw: the decoder returned a linear image where a developed one was asked".into()),
    }
}

/// Decode and develop a raw file: 16-bit RGB (16-bit gray for a monochrome
/// sensor), oriented.
pub fn develop(bytes: &[u8]) -> Result<DynamicImage, String> {
    image_of(backend()?.develop(bytes, false)?)
}

/// Decode a raw to the camera's own linear space — black and white levels,
/// demosaic, the sensor's crop, turned as the camera said; no white balance,
/// matrix or curve — with what the linear DNG needs to know about it
/// (`dng.rs`). Three-color sensors only: a monochrome or four-color sensor
/// has no place in an RGB DNG here.
pub fn develop_linear(bytes: &[u8]) -> Result<(Img3, DngInfo), String> {
    let r = backend()?.develop(bytes, true)?;
    let (w, h) = (r.w, r.h);
    let p = match r.px {
        Pixels::Planes3(p) if p.iter().all(|v| v.len() == w * h) => p,
        _ => return Err("raw: the decoder returned no linear planes".into()),
    };
    let c = r.color.ok_or_else(|| "raw: the decoder returned no camera color".to_string())?;
    let img = Img3 { w, h, p };
    let img = if r.turns % 4 == 0 && !r.flip { img } else { crate::prep::rotate(&img, r.turns, r.flip) };
    let info = DngInfo::of_raw(&c.make, &c.model, c.matrices, c.wb, c.d65)?;
    Ok((img, info))
}

/// The JPEG preview the camera wrote into the file (its thumbnail failing
/// that), oriented: what a filmstrip shows before the run develops the frame.
pub fn preview(bytes: &[u8]) -> Option<DynamicImage> {
    image_of(backend().ok()?.preview(bytes)?).ok()
}

/// The focus distance in meters from the file's own metadata (EXIF
/// SubjectDistance), for the files whose EXIF the TIFF reader does not reach.
pub fn subject_distance(bytes: &[u8]) -> Option<f64> {
    backend().ok()?.metadata(bytes)?.subject_distance
}

/// The capture time from the file's own metadata (EXIF DateTimeOriginal, else
/// CreateDate, with the sub-seconds), seconds since the epoch — for the files
/// whose EXIF the TIFF reader of `meta` does not reach (CR3, RAF, …).
pub fn capture_time(bytes: &[u8]) -> Option<f64> {
    let e = backend().ok()?.metadata(bytes)?;
    e.date_time_original
        .as_deref()
        .and_then(|s| crate::meta::parse_datetime(s, e.sub_sec_time_original.as_deref()))
        .or_else(|| e.create_date.as_deref().and_then(|s| crate::meta::parse_datetime(s, e.sub_sec_time_digitized.as_deref())))
}

/// The shared library `liblapstack_raw` (`crates/lapstack-raw/src/ffi.rs`),
/// loaded on first use: `LAPSTACK_RAW_LIB` names it, else it is looked for
/// beside the executable, in `../lib` from there, and on the system's library
/// path. Its `lapstack_raw_abi` must be ours.
#[cfg(all(feature = "raw", not(target_arch = "wasm32")))]
mod dylib {
    use super::*;
    use libloading::{Library, Symbol};
    use std::ffi::{CStr, c_char, c_void};
    use std::path::PathBuf;

    #[repr(C)]
    struct LrImage {
        w: u32,
        h: u32,
        channels: u32,
        bits: u32,
        turns: u8,
        flip: u8,
        _pad: [u8; 6],
        data: *const u8,
        len: usize,
        color: *const c_char,
        handle: *mut c_void,
    }

    type AbiFn = unsafe extern "C" fn() -> u32;
    type DevelopFn = unsafe extern "C" fn(*const u8, usize, u32, *mut LrImage) -> *mut c_char;
    type PreviewFn = unsafe extern "C" fn(*const u8, usize, *mut LrImage) -> u32;
    type MetadataFn = unsafe extern "C" fn(*const u8, usize) -> *mut c_char;
    type FreeImageFn = unsafe extern "C" fn(*mut LrImage);
    type FreeStrFn = unsafe extern "C" fn(*mut c_char);

    pub struct Dylib {
        path: String,
        // the library outlives the symbols taken from it: dropped last (field order)
        develop: Symbol<'static, DevelopFn>,
        preview: Symbol<'static, PreviewFn>,
        metadata: Symbol<'static, MetadataFn>,
        free_image: Symbol<'static, FreeImageFn>,
        free_str: Symbol<'static, FreeStrFn>,
        _lib: &'static Library,
    }

    const NAME: &str = if cfg!(target_os = "windows") {
        "lapstack_raw.dll"
    } else if cfg!(target_os = "macos") {
        "liblapstack_raw.dylib"
    } else {
        "liblapstack_raw.so"
    };

    fn candidates() -> Vec<PathBuf> {
        let mut c = Vec::new();
        if let Ok(p) = std::env::var("LAPSTACK_RAW_LIB") {
            if !p.is_empty() {
                c.push(PathBuf::from(p));
            }
        }
        if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
            c.push(dir.join(NAME));
            c.push(dir.join("..").join("lib").join(NAME));
        }
        c.push(PathBuf::from(NAME));
        c
    }

    pub fn load() -> Result<Dylib, String> {
        let mut tried = Vec::new();
        for p in candidates() {
            // SAFETY: the library is ours (or the user's build of it); its constructors are Rust's.
            match unsafe { Library::new(&p) } {
                Ok(lib) => {
                    let lib: &'static Library = Box::leak(Box::new(lib));
                    let s = p.display().to_string();
                    let abi: Symbol<AbiFn> = unsafe { lib.get(b"lapstack_raw_abi\0") }.map_err(|e| format!("{s}: not a lapstack raw library ({e})"))?;
                    let v = unsafe { abi() };
                    if v != ABI {
                        return Err(format!("{s}: raw library interface {v}, this lapstack speaks {ABI}"));
                    }
                    fn sym<T>(lib: &'static Library, s: &str, n: &[u8]) -> Result<Symbol<'static, T>, String> {
                        unsafe { lib.get(n) }.map_err(|e| format!("{s}: {e}"))
                    }
                    return Ok(Dylib {
                        develop: sym(lib, &s, b"lapstack_raw_develop\0")?,
                        preview: sym(lib, &s, b"lapstack_raw_preview\0")?,
                        metadata: sym(lib, &s, b"lapstack_raw_metadata\0")?,
                        free_image: sym(lib, &s, b"lapstack_raw_free_image\0")?,
                        free_str: sym(lib, &s, b"lapstack_raw_free_str\0")?,
                        path: s,
                        _lib: lib,
                    });
                }
                Err(_) => tried.push(p.display().to_string()),
            }
        }
        Err(format!(
            "no raw decoder: {NAME} (lapstack-raw, LGPL, shipped beside lapstack) was not found; looked for {}. Put it next to the binary or set LAPSTACK_RAW_LIB.",
            tried.join(", ")
        ))
    }

    impl Dylib {
        fn take(&self, o: &mut LrImage) -> Result<RawImage, String> {
            let (w, h) = (o.w as usize, o.h as usize);
            let bytes = if o.data.is_null() { &[][..] } else { unsafe { std::slice::from_raw_parts(o.data, o.len) } };
            let px = match (o.bits, o.channels) {
                (16, 1) => Pixels::Gray16(bytes.chunks_exact(2).map(|b| u16::from_ne_bytes([b[0], b[1]])).collect()),
                (16, 3) => Pixels::Rgb16(bytes.chunks_exact(2).map(|b| u16::from_ne_bytes([b[0], b[1]])).collect()),
                (8, 3) => Pixels::Rgb8(bytes.to_vec()),
                (32, 3) => {
                    let n = w * h;
                    let all: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
                    if all.len() != 3 * n {
                        return Err("raw: the decoder returned planes of the wrong size".into());
                    }
                    let (r, gb) = all.split_at(n);
                    let (g, b) = gb.split_at(n);
                    Pixels::Planes3([r.to_vec(), g.to_vec(), b.to_vec()])
                }
                (bits, ch) => return Err(format!("raw: the decoder returned {bits}-bit, {ch}-channel pixels")),
            };
            let color = if o.color.is_null() {
                None
            } else {
                let s = unsafe { CStr::from_ptr(o.color) }.to_string_lossy();
                Some(serde_json::from_str::<RawColor>(&s).map_err(|e| format!("raw: the decoder's color description does not parse: {e}"))?)
            };
            Ok(RawImage { w, h, px, turns: o.turns, flip: o.flip != 0, color })
        }

        fn empty() -> LrImage {
            LrImage { w: 0, h: 0, channels: 0, bits: 0, turns: 0, flip: 0, _pad: [0; 6], data: std::ptr::null(), len: 0, color: std::ptr::null(), handle: std::ptr::null_mut() }
        }

        fn string(&self, s: *mut c_char) -> Option<String> {
            if s.is_null() {
                return None;
            }
            let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
            unsafe { (self.free_str)(s) };
            Some(out)
        }
    }

    impl RawBackend for Dylib {
        fn develop(&self, bytes: &[u8], linear: bool) -> Result<RawImage, String> {
            let mut o = Self::empty();
            let e = unsafe { (self.develop)(bytes.as_ptr(), bytes.len(), linear as u32, &mut o) };
            if let Some(msg) = self.string(e) {
                return Err(msg);
            }
            let r = self.take(&mut o);
            unsafe { (self.free_image)(&mut o) };
            r
        }
        fn preview(&self, bytes: &[u8]) -> Option<RawImage> {
            let mut o = Self::empty();
            if unsafe { (self.preview)(bytes.as_ptr(), bytes.len(), &mut o) } == 0 {
                return None;
            }
            let r = self.take(&mut o).ok();
            unsafe { (self.free_image)(&mut o) };
            r
        }
        fn metadata(&self, bytes: &[u8]) -> Option<RawMeta> {
            let s = self.string(unsafe { (self.metadata)(bytes.as_ptr(), bytes.len()) })?;
            serde_json::from_str(&s).ok()
        }
        fn describe(&self) -> String {
            format!("raw decoder: {}", self.path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_names() {
        assert!(is_raw("DSC_0001.NEF"));
        assert!(is_raw("shoot/a.cr3"));
        assert!(is_raw("x.dng"));
        assert!(!is_raw("x.tif"));
        assert!(!is_raw("x.png"));
        assert!(!is_raw("nef"));
        assert!(!is_raw("x.jpg"));
    }
}

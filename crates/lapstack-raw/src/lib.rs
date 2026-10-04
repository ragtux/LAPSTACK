// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LGPL-2.1-only

//! lapstack-raw: camera raw files decoded and developed with rawler (dnglab's
//! library), for lapstack — as a component of its own. Natively it is a shared
//! library (`ffi.rs`: a C ABI) that the lapstack binary loads at run time; in
//! the browser it is a wasm module of its own (`wasm.rs`) that the worker
//! imports and the engine reaches through three functions on the global
//! object. lapstack itself never links this crate or rawler: the LGPL's
//! condition that the user can replace the library is met by replacing this
//! file, and this crate's source, with rawler's, ships beside every build.
//!
//! What crosses the boundary (the contract; `ABI` counts its revisions):
//!
//! - `develop`: the raw as the camera meant it to be seen — black and white
//!   levels, demosaic, the white balance as shot, the camera's matrix to sRGB,
//!   the sRGB curve, turned by the EXIF orientation. 16-bit RGB interleaved,
//!   or 16-bit gray for a monochrome sensor.
//! - `develop_linear`: the raw in the camera's own linear space — levels,
//!   demosaic, the sensor's crop; no white balance, matrix or curve — as three
//!   f32 planes (R, G, B, each `w*h`), **not turned**: `turns` (quarter turns
//!   clockwise) and `flip` (mirror first) say how the caller should turn it.
//!   With it, what a linear DNG needs to know about the camera, as JSON
//!   (`Color`): make and model, the XYZ→camera matrices with their EXIF
//!   illuminant codes, the as-shot white balance, and a D65 matrix.
//! - `preview`: the JPEG preview the camera wrote into the file (its
//!   thumbnail failing that), turned, as 8-bit RGB.
//! - `metadata`: what the batch split and the near-end cue read from a raw's
//!   own EXIF, as JSON (`Metadata`): the orientation, the date strings with
//!   their sub-seconds, the subject distance.

use image::DynamicImage;
use rawler::decoders::{Orientation, RawDecodeParams};
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::imgop::xyz::Illuminant;
use rawler::rawsource::RawSource;
use serde::Serialize;

#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// The contract's revision: bumped when a buffer layout or a JSON shape changes.
pub const ABI: u32 = 1;

/// The pixels of a developed image.
pub enum Pixels {
    /// 16-bit gray, `w*h` (a monochrome sensor, `develop`).
    Gray16(Vec<u16>),
    /// 16-bit RGB interleaved, `w*h*3` (`develop`).
    Rgb16(Vec<u16>),
    /// 8-bit RGB interleaved, `w*h*3` (`preview`).
    Rgb8(Vec<u8>),
    /// Three f32 planes R, G, B one after another, `w*h*3` (`develop_linear`).
    Planes3(Vec<f32>),
}

impl Pixels {
    pub fn channels(&self) -> u32 {
        match self {
            Pixels::Gray16(_) => 1,
            _ => 3,
        }
    }
    pub fn bits(&self) -> u32 {
        match self {
            Pixels::Rgb8(_) => 8,
            Pixels::Gray16(_) | Pixels::Rgb16(_) => 16,
            Pixels::Planes3(_) => 32,
        }
    }
    /// The buffer as bytes (for the C ABI).
    pub fn bytes(&self) -> &[u8] {
        match self {
            Pixels::Gray16(v) | Pixels::Rgb16(v) => unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) },
            Pixels::Rgb8(v) => v,
            Pixels::Planes3(v) => unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) },
        }
    }
}

pub struct Image {
    pub w: usize,
    pub h: usize,
    pub px: Pixels,
    /// How to turn a `develop_linear` image (quarter turns clockwise after a
    /// mirror); `develop` and `preview` images are turned already (0, false).
    pub turns: u8,
    pub flip: bool,
    /// `develop_linear` only.
    pub color: Option<Color>,
}

/// What a linear DNG needs to know about the camera (`develop_linear`).
#[derive(Serialize)]
pub struct Color {
    pub make: String,
    pub model: String,
    /// XYZ → camera matrices with their illuminants (EXIF LightSource codes),
    /// the cooler illuminant first as Adobe writes them; one or two.
    pub matrices: Vec<(u16, [f32; 9])>,
    /// The as-shot white balance multipliers, as the file carries them.
    pub wb: [f32; 3],
    /// The D65 matrix: the file's, or its nearest adapted to D65 (Bradford).
    pub d65: [f32; 9],
}

/// What lapstack reads from a raw's own EXIF (`metadata`).
#[derive(Serialize)]
pub struct Metadata {
    pub orientation: u16,
    pub date_time_original: Option<String>,
    pub sub_sec_time_original: Option<String>,
    pub create_date: Option<String>,
    pub sub_sec_time_digitized: Option<String>,
    /// EXIF SubjectDistance in meters, when the file has a positive one.
    pub subject_distance: Option<f64>,
}

fn err(e: rawler::RawlerError) -> String {
    format!("raw: {e}")
}

/// The image as the camera meant it to be seen (EXIF orientation).
fn orient(img: DynamicImage, o: Orientation) -> DynamicImage {
    match o {
        Orientation::HorizontalFlip => img.fliph(),
        Orientation::Rotate180 => img.rotate180(),
        Orientation::VerticalFlip => img.flipv(),
        Orientation::Transpose => img.rotate90().fliph(),
        Orientation::Rotate90 => img.rotate90(),
        Orientation::Transverse => img.rotate270().fliph(),
        Orientation::Rotate270 => img.rotate270(),
        Orientation::Normal | Orientation::Unknown => img,
    }
}

/// Orientation as quarter turns clockwise after a mirror. `flipv` is a half
/// turn of the mirrored image, `Transpose` and `Transverse` a quarter turn of it.
fn turns(o: Orientation) -> (u8, bool) {
    match o {
        Orientation::HorizontalFlip => (0, true),
        Orientation::Rotate180 => (2, false),
        Orientation::VerticalFlip => (2, true),
        Orientation::Transpose => (3, true),
        Orientation::Rotate90 => (1, false),
        Orientation::Transverse => (1, true),
        Orientation::Rotate270 => (3, false),
        Orientation::Normal | Orientation::Unknown => (0, false),
    }
}

/// Decode and develop a raw file: 16-bit RGB (16-bit gray for a monochrome
/// sensor), turned.
pub fn develop(bytes: &[u8]) -> Result<Image, String> {
    let src = RawSource::new_from_slice(bytes);
    let dec = rawler::get_decoder(&src).map_err(err)?;
    let params = RawDecodeParams::default();
    let raw = dec.raw_image(&src, &params, false).map_err(err)?;
    let orientation = raw.orientation;
    let img = RawDevelop::default()
        .develop_intermediate(&raw)
        .map_err(err)?
        .to_dynamic_image()
        .ok_or_else(|| "raw: the developed image has an odd size".to_string())?;
    let img = orient(img, orientation);
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px = match img {
        DynamicImage::ImageLuma16(b) => Pixels::Gray16(b.into_raw()),
        DynamicImage::ImageRgb16(b) => Pixels::Rgb16(b.into_raw()),
        other => Pixels::Rgb16(other.into_rgb16().into_raw()),
    };
    Ok(Image { w, h, px, turns: 0, flip: false, color: None })
}

/// Decode a raw to the camera's own linear space — black and white levels,
/// demosaic, the sensor's crop; no white balance, matrix or curve — not
/// turned, with how to turn it and what the linear DNG needs to know. Three-
/// color sensors only: a monochrome or four-color sensor has no place in an
/// RGB DNG.
pub fn develop_linear(bytes: &[u8]) -> Result<Image, String> {
    let src = RawSource::new_from_slice(bytes);
    let dec = rawler::get_decoder(&src).map_err(err)?;
    let params = RawDecodeParams::default();
    let raw = dec.raw_image(&src, &params, false).map_err(err)?;
    let steps = [ProcessingStep::Rescale, ProcessingStep::Demosaic, ProcessingStep::FujiRotate, ProcessingStep::CropActiveArea, ProcessingStep::CropDefault];
    let im = RawDevelop::new_with(&steps).develop_intermediate(&raw).map_err(err)?;
    let px = match im {
        Intermediate::ThreeColor(px) => px,
        Intermediate::Monochrome(_) => return Err("raw: a monochrome sensor cannot be written as an RGB linear DNG".into()),
        Intermediate::FourColor(_) => return Err("raw: a four-color sensor cannot be written as a linear DNG here".into()),
    };
    let (w, h) = (px.width, px.height);
    let n = w * h;
    let mut planes = vec![0f32; n * 3];
    for (i, p) in px.data.iter().enumerate() {
        planes[i] = p[0];
        planes[n + i] = p[1];
        planes[2 * n + i] = p[2];
    }
    let (turns, flip) = turns(raw.orientation);
    // the matrices as the camera holds them, the cooler illuminant first as Adobe writes them
    // (Standard Light A as ColorMatrix1, D65 as ColorMatrix2); the look's D65 matrix adapted
    // from another when there is none
    let mut matrices: Vec<(u16, [f32; 9])> = raw.color_matrix.iter().filter(|(_, m)| m.len() == 9).map(|(i, m)| (*i as u16, <[f32; 9]>::try_from(&m[..]).unwrap())).collect();
    matrices.sort_by_key(|(i, _)| kelvin(*i));
    if matrices.is_empty() {
        return Err(format!("raw: no color matrix is known for the {} {}", raw.clean_make, raw.clean_model));
    }
    let d65 = match raw.color_matrix_find_first([Illuminant::D65, Illuminant::A, Illuminant::B, Illuminant::C, Illuminant::D50, Illuminant::D55, Illuminant::D75, Illuminant::Daylight, Illuminant::Flash]) {
        Some((Illuminant::D65, m)) if m.len() == 9 => <[f32; 9]>::try_from(&m[..]).unwrap(),
        Some((illu, m)) if m.len() == 9 => {
            let m3: [[f32; 3]; 3] = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
            let a = rawler::imgop::chromatic_adaption::adapt_bradford(&illu, &Illuminant::D65, &m3);
            [a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2]]
        }
        _ => matrices[0].1,
    };
    let wb = [raw.wb_coeffs[0], raw.wb_coeffs[1], raw.wb_coeffs[2]];
    let color = Color { make: raw.clean_make.clone(), model: raw.clean_model.clone(), matrices, wb, d65 };
    Ok(Image { w, h, px: Pixels::Planes3(planes), turns, flip, color: Some(color) })
}

/// The color temperature of an EXIF LightSource, roughly, to order a
/// camera's matrices the way Adobe writes them (the cooler light first) —
/// the same table as lapstack's `dng.rs`, which reads them in this order.
fn kelvin(illuminant: u16) -> u32 {
    match illuminant {
        17 | 3 | 24 => 2856,   // A, tungsten, ISO studio tungsten
        2 | 14 | 15 => 4150,   // fluorescents
        13 => 4230,
        18 => 4874,            // B
        23 => 5003,            // D50
        20 => 5503,            // D55
        1 | 9 | 4 => 5500,     // daylight, fine weather, flash
        12 => 6430,
        19 => 6774,            // C
        21 => 6504,            // D65
        10 => 6500,            // cloudy
        22 => 7504,            // D75
        11 => 7500,            // shade
        _ => 10000,
    }
}

/// The JPEG preview the camera wrote into the file (its thumbnail failing
/// that), turned: what a filmstrip shows before the run develops the frame.
pub fn preview(bytes: &[u8]) -> Option<Image> {
    let src = RawSource::new_from_slice(bytes);
    let dec = rawler::get_decoder(&src).ok()?;
    let params = RawDecodeParams::default();
    let img = dec.preview_image(&src, &params).ok().flatten().or_else(|| dec.thumbnail_image(&src, &params).ok().flatten())?;
    let o = dec.raw_metadata(&src, &params).ok().and_then(|m| m.exif.orientation).map_or(Orientation::Normal, Orientation::from_u16);
    let img = orient(img, o);
    let (w, h) = (img.width() as usize, img.height() as usize);
    Some(Image { w, h, px: Pixels::Rgb8(img.into_rgb8().into_raw()), turns: 0, flip: false, color: None })
}

/// What lapstack reads from the file's own EXIF.
pub fn metadata(bytes: &[u8]) -> Option<Metadata> {
    let src = RawSource::new_from_slice(bytes);
    let dec = rawler::get_decoder(&src).ok()?;
    let md = dec.raw_metadata(&src, &RawDecodeParams::default()).ok()?;
    let e = &md.exif;
    let subject_distance = e.subject_distance.and_then(|r| (r.d > 0 && r.n > 0).then(|| r.n as f64 / r.d as f64));
    Some(Metadata {
        orientation: e.orientation.unwrap_or(1),
        date_time_original: e.date_time_original.clone(),
        sub_sec_time_original: e.sub_sec_time_original.clone(),
        create_date: e.create_date.clone(),
        sub_sec_time_digitized: e.sub_sec_time_digitized.clone(),
        subject_distance,
    })
}

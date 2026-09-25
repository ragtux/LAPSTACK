// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: AGPL-3.0-only

//! Camera raw files as input frames, developed with rawler (dnglab's library):
//! decoded, black and white levels applied, demosaicked, white-balanced as
//! shot, taken through the camera's colour matrix to sRGB and given the sRGB
//! curve — the 16-bit RGB every other input becomes, turned the way the camera
//! said. There is no exposure or tone adjustment: the frames of a stack are
//! shot alike, and what matters to the stack is that they are developed alike;
//! when the look matters, a stack developed in a raw converter first, with its
//! exposure and profile, is the better input, and lapstack takes its TIFFs.
//! The JPEG preview the camera wrote into the file serves the thumbnails, and
//! the file's own EXIF the capture time of the batch split.

use crate::dng::DngInfo;
use crate::pyramid::Img3;
use image::DynamicImage;

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

#[cfg(feature = "raw")]
mod with_rawler {
    use super::*;
    use rawler::decoders::{Orientation, RawDecodeParams};
    use rawler::imgop::develop::RawDevelop;
    use rawler::rawsource::RawSource;

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

    /// Decode and develop a raw file: 16-bit RGB (16-bit gray for a monochrome
    /// sensor), oriented.
    pub fn develop(bytes: &[u8]) -> Result<DynamicImage, String> {
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
        Ok(orient(img, orientation))
    }

    /// Orientation as `prep::rotate` takes it: quarter turns clockwise after a
    /// mirror. `flipv` is a half turn of the mirrored image, `Transpose` and
    /// `Transverse` a quarter turn of it.
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

    /// Decode a raw to the camera's own linear space — black and white levels,
    /// demosaic, the sensor's crop, turned as the camera said; no white balance,
    /// matrix or curve — with what the linear DNG needs to know about it
    /// (`dng.rs`). Three-colour sensors only: a monochrome or four-colour
    /// sensor has no place in an RGB DNG here.
    pub fn develop_linear(bytes: &[u8]) -> Result<(Img3, DngInfo), String> {
        use rawler::imgop::develop::{Intermediate, ProcessingStep};
        use rawler::imgop::xyz::Illuminant;
        let src = RawSource::new_from_slice(bytes);
        let dec = rawler::get_decoder(&src).map_err(err)?;
        let params = RawDecodeParams::default();
        let raw = dec.raw_image(&src, &params, false).map_err(err)?;
        let steps = [ProcessingStep::Rescale, ProcessingStep::Demosaic, ProcessingStep::FujiRotate, ProcessingStep::CropActiveArea, ProcessingStep::CropDefault];
        let im = RawDevelop::new_with(&steps).develop_intermediate(&raw).map_err(err)?;
        let px = match im {
            Intermediate::ThreeColor(px) => px,
            Intermediate::Monochrome(_) => return Err("raw: a monochrome sensor cannot be written as an RGB linear DNG".into()),
            Intermediate::FourColor(_) => return Err("raw: a four-colour sensor cannot be written as a linear DNG here".into()),
        };
        let (w, h) = (px.width, px.height);
        let mut img = Img3::zeros(w, h);
        for (i, p) in px.data.iter().enumerate() {
            img.p[0][i] = p[0];
            img.p[1][i] = p[1];
            img.p[2][i] = p[2];
        }
        let (q, flip) = turns(raw.orientation);
        let img = if q == 0 && !flip { img } else { crate::prep::rotate(&img, q, flip) };
        // the matrices as the camera holds them, the cooler illuminant first as Adobe writes them
        // (Standard Light A as ColorMatrix1, D65 as ColorMatrix2); the look's D65 matrix adapted
        // from another when there is none
        let mut matrices: Vec<(u16, [f32; 9])> = raw.color_matrix.iter().filter(|(_, m)| m.len() == 9).map(|(i, m)| (*i as u16, <[f32; 9]>::try_from(&m[..]).unwrap())).collect();
        matrices.sort_by_key(|(i, _)| crate::dng::kelvin(*i));
        if matrices.is_empty() {
            return Err(format!("raw: no colour matrix is known for the {} {}", raw.clean_make, raw.clean_model));
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
        let info = DngInfo::of_raw(&raw.clean_make, &raw.clean_model, matrices, wb, d65)?;
        Ok((img, info))
    }

    /// The JPEG preview the camera wrote into the file (its thumbnail failing
    /// that), oriented: what a filmstrip shows before the run develops the frame.
    pub fn preview(bytes: &[u8]) -> Option<DynamicImage> {
        let src = RawSource::new_from_slice(bytes);
        let dec = rawler::get_decoder(&src).ok()?;
        let params = RawDecodeParams::default();
        let img = dec.preview_image(&src, &params).ok().flatten().or_else(|| dec.thumbnail_image(&src, &params).ok().flatten())?;
        let o = dec.raw_metadata(&src, &params).ok().and_then(|m| m.exif.orientation).map_or(Orientation::Normal, Orientation::from_u16);
        Some(orient(img, o))
    }

    /// The focus distance in metres from the file's own metadata (EXIF
    /// SubjectDistance), for the files whose EXIF the TIFF reader does not reach.
    pub fn subject_distance(bytes: &[u8]) -> Option<f64> {
        let src = RawSource::new_from_slice(bytes);
        let dec = rawler::get_decoder(&src).ok()?;
        let md = dec.raw_metadata(&src, &RawDecodeParams::default()).ok()?;
        let r = md.exif.subject_distance?;
        (r.d > 0 && r.n > 0).then(|| r.n as f64 / r.d as f64)
    }

    /// The capture time from the file's own metadata (EXIF DateTimeOriginal, else
    /// CreateDate, with the sub-seconds), seconds since the epoch — for the files
    /// whose EXIF the TIFF reader of `meta` does not reach (CR3, RAF, …).
    pub fn capture_time(bytes: &[u8]) -> Option<f64> {
        let src = RawSource::new_from_slice(bytes);
        let dec = rawler::get_decoder(&src).ok()?;
        let md = dec.raw_metadata(&src, &RawDecodeParams::default()).ok()?;
        let e = &md.exif;
        e.date_time_original
            .as_deref()
            .and_then(|s| crate::meta::parse_datetime(s, e.sub_sec_time_original.as_deref()))
            .or_else(|| e.create_date.as_deref().and_then(|s| crate::meta::parse_datetime(s, e.sub_sec_time_digitized.as_deref())))
    }
}
#[cfg(feature = "raw")]
pub use with_rawler::{capture_time, develop, develop_linear, preview, subject_distance};

#[cfg(not(feature = "raw"))]
pub fn develop(_bytes: &[u8]) -> Result<DynamicImage, String> {
    Err("this build has no raw support (the `raw` feature of lapstack-core)".to_string())
}
#[cfg(not(feature = "raw"))]
pub fn develop_linear(_bytes: &[u8]) -> Result<(Img3, DngInfo), String> {
    Err("this build has no raw support (the `raw` feature of lapstack-core)".to_string())
}
#[cfg(not(feature = "raw"))]
pub fn preview(_bytes: &[u8]) -> Option<DynamicImage> {
    None
}
#[cfg(not(feature = "raw"))]
pub fn capture_time(_bytes: &[u8]) -> Option<f64> {
    None
}
#[cfg(not(feature = "raw"))]
pub fn subject_distance(_bytes: &[u8]) -> Option<f64> {
    None
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

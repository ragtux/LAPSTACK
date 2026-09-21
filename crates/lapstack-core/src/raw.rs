// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

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
pub use with_rawler::{capture_time, develop, preview};

#[cfg(not(feature = "raw"))]
pub fn develop(_bytes: &[u8]) -> Result<DynamicImage, String> {
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

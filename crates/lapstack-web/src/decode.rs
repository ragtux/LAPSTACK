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

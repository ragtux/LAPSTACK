//! Decode PNG/JPEG/TIFF bytes to interleaved RGB u16 (8-bit inputs widened by
//! ×257), the form the `warp` kernel samples directly.

use image::ColorType;
use std::io::Cursor;

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub bits: u32,
    /// RGB interleaved, w*h*3 samples, padded to an even count.
    pub rgb: Vec<u16>,
}

pub fn decode(bytes: &[u8]) -> Result<Frame, String> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("unrecognised image: {e}"))?;
    reader.no_limits();
    let img = reader.decode().map_err(|e| format!("decode: {e}"))?;
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
    Ok(Frame { w, h, bits, rgb })
}

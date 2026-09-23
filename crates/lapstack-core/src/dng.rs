// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! Linear DNG output — raw in, DNG out: the stacked image
//! written as a demosaiced, linear, camera-space DNG that a raw converter
//! develops like the raws it came from, with the exposure, white balance,
//! profile and highlight latitude of a raw instead of a baked-in rendering.
//!
//! The frames are developed to the camera's own linear space (black and
//! white levels, demosaic, the sensor's crop — no white balance, matrix or
//! curve) and the stack is fused in a *look* space made from it: the as-shot
//! white balance, the camera matrix to linear sRGB and the sRGB curve, exactly
//! the normal development except that nothing is clipped — the curve is
//! extended above 1 and mirrored below 0, so a highlight past white or a
//! colour outside sRGB keeps its value and the transform stays invertible.
//! The fusion then makes the same decisions as on a normally developed stack
//! (the luma it selects by is the ordinary one), the viewer shows an ordinary
//! image, and the DNG writer takes the fused look image back through the
//! inverse curve, inverse matrix and inverse white balance to camera space,
//! 16 bits per sample, with the camera's colour matrices and the neutral it
//! shot (`AsShotNeutral`) in the tags. The frames' own white balance is that
//! of frame 0, applied to every frame, so the stack is developed alike.
//! Non-raw frames are taken as sRGB: their camera space is linear sRGB and
//! the DNG says so (ColorMatrix1 = XYZ → sRGB at D65, a neutral of 1, 1, 1).

use crate::meta::Meta;
use crate::pyramid::Img3;
use rayon::prelude::*;
use std::io::Write;

/// XYZ (D65) → linear sRGB, IEC 61966-2-1.
pub const XYZ_TO_SRGB_D65: [[f32; 3]; 3] = [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]];
/// Linear sRGB → XYZ (D65).
pub const SRGB_TO_XYZ_D65: [[f32; 3]; 3] = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];
/// EXIF LightSource code of D65, the DNG's CalibrationIlluminant.
pub const D65: u16 = 21;

/// The colour temperature of an EXIF LightSource, roughly, to order a
/// camera's matrices the way Adobe writes them (the cooler light first).
pub fn kelvin(illuminant: u16) -> u32 {
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

/// What the DNG needs to know about the camera space the stack is in, and
/// how that space and the look space the stack was fused in map to each other.
#[derive(Clone, Debug, PartialEq)]
pub struct DngInfo {
    pub make: String,
    pub model: String,
    /// XYZ → camera matrices with their illuminants (EXIF LightSource codes,
    /// as DNG's CalibrationIlluminant), as the raw carries them; one or two.
    pub matrices: Vec<(u16, [f32; 9])>,
    /// The as-shot white balance multipliers, green = 1: the look applies
    /// them, and the neutral the DNG records is their reciprocal.
    pub wb: [f32; 3],
    /// Camera (white-balanced) → linear sRGB, and its inverse.
    pub cam2rgb: [[f64; 3]; 3],
    pub rgb2cam: [[f64; 3]; 3],
    /// Where the space came from, for the log.
    pub source: String,
}

fn inverse3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let mut inv = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv[i][j] = c(j, i) / det;
        }
    }
    Some(inv)
}

fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

impl DngInfo {
    /// The space of non-raw frames: linear sRGB as the camera space, nothing
    /// to balance, the identity as the look's matrix.
    pub fn srgb() -> DngInfo {
        let mut m = [0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                m[3 * i + j] = XYZ_TO_SRGB_D65[i][j];
            }
        }
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        DngInfo { make: String::new(), model: String::new(), matrices: vec![(D65, m)], wb: [1.0; 3], cam2rgb: id, rgb2cam: id, source: "sRGB (frames that are not camera raws)".into() }
    }

    /// The space of a camera raw: its make and model, its XYZ → camera
    /// matrices by illuminant, its as-shot white balance (any scale; green
    /// is made 1) and the D65 matrix the look is built from (the camera's own
    /// D65 matrix, or another adapted to D65 — the raw path's choice). The
    /// look's matrix is dcraw's: XYZ→cam · sRGB→XYZ, rows normalised so that
    /// white balances to white, inverted.
    pub fn of_raw(make: &str, model: &str, matrices: Vec<(u16, [f32; 9])>, wb: [f32; 3], xyz2cam_d65: [f32; 9]) -> Result<DngInfo, String> {
        let g = if wb[1].is_finite() && wb[1] > 0.0 { wb[1] } else { 1.0 };
        let wb = [wb[0], wb[1], wb[2]].map(|v| if v.is_finite() && v > 0.0 { v / g } else { 1.0 });
        let mut xyz2cam = [[0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                xyz2cam[i][j] = xyz2cam_d65[3 * i + j] as f64;
            }
        }
        let srgb2xyz = SRGB_TO_XYZ_D65.map(|r| r.map(|v| v as f64));
        let mut rgb2cam = mul3(&xyz2cam, &srgb2xyz);
        for row in &mut rgb2cam {
            let s: f64 = row.iter().sum();
            if s != 0.0 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
        let cam2rgb = inverse3(rgb2cam).ok_or("the camera's colour matrix is singular")?;
        let source = format!("{} {} ({} colour matri{})", make.trim(), model.trim(), matrices.len(), if matrices.len() == 1 { "x" } else { "ces" });
        Ok(DngInfo { make: make.trim().to_string(), model: model.trim().to_string(), matrices, wb, cam2rgb, rgb2cam, source })
    }

    /// Whether the space is a camera's (a raw's) rather than sRGB.
    pub fn is_camera(&self) -> bool {
        !self.make.is_empty() || !self.model.is_empty() || self.cam2rgb != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    /// The camera coordinates of a neutral, green = 1: DNG's AsShotNeutral.
    pub fn as_shot_neutral(&self) -> [f32; 3] {
        self.wb.map(|w| 1.0 / w)
    }

    /// A camera-linear image (planes in [0, 1], 1 = the white level) to the
    /// look space the stack is fused in, in place: white balance, matrix,
    /// extended sRGB curve.
    pub fn to_look(&self, img: &mut Img3) {
        let m = self.cam2rgb;
        let wb = self.wb.map(|v| v as f64);
        let n = img.w * img.h;
        let [r, g, b] = &mut img.p;
        r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).take(n).for_each(|((r, g), b)| {
            let c = [*r as f64 * wb[0], *g as f64 * wb[1], *b as f64 * wb[2]];
            let o = [m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2], m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2], m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2]];
            *r = encode_ext(o[0]) as f32;
            *g = encode_ext(o[1]) as f32;
            *b = encode_ext(o[2]) as f32;
        });
    }

    /// The look image back to camera space as interleaved 16-bit samples
    /// (65535 = the white level): inverse curve, inverse matrix, the white
    /// balance divided out; clipped to the DNG's range.
    pub fn from_look(&self, img: &Img3) -> Vec<u16> {
        let m = self.rgb2cam;
        let wb = self.wb.map(|v| v as f64);
        let n = img.w * img.h;
        let mut out = vec![0u16; n * 3];
        out.par_chunks_mut(3 * img.w.max(1)).enumerate().for_each(|(y, row)| {
            for x in 0..row.len() / 3 {
                let i = y * img.w + x;
                let l = [decode_ext(img.p[0][i] as f64), decode_ext(img.p[1][i] as f64), decode_ext(img.p[2][i] as f64)];
                for k in 0..3 {
                    let v = (m[k][0] * l[0] + m[k][1] * l[1] + m[k][2] * l[2]) / wb[k];
                    row[3 * x + k] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
        });
        out
    }

    /// `from_look` for an interleaved 16-bit look image (the browser's masters).
    pub fn from_look_u16(&self, rgb16: &[u16], w: usize, h: usize) -> Vec<u16> {
        let m = self.rgb2cam;
        let wb = self.wb.map(|v| v as f64);
        let mut out = vec![0u16; w * h * 3];
        out.par_chunks_mut(3 * w.max(1)).zip(rgb16.par_chunks(3 * w.max(1))).for_each(|(row, src)| {
            for x in 0..row.len() / 3 {
                let l = [0, 1, 2].map(|c| decode_ext(src[3 * x + c] as f64 / 65535.0));
                for k in 0..3 {
                    let v = (m[k][0] * l[0] + m[k][1] * l[1] + m[k][2] * l[2]) / wb[k];
                    row[3 * x + k] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
        });
        out
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        let n = self.as_shot_neutral();
        format!("linear DNG: camera space of {}; white balance {:.3} {:.3} {:.3}; neutral {:.4} {:.4} {:.4}", self.source, self.wb[0], self.wb[1], self.wb[2], n[0], n[1], n[2])
    }
}

/// The sRGB transfer curve, extended: the same formula above 1 and mirrored
/// through 0, so it is a bijection of the reals.
pub fn encode_ext(x: f64) -> f64 {
    let a = x.abs();
    let e = if a <= 0.0031308 { 12.92 * a } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
    if x < 0.0 { -e } else { e }
}

/// The inverse of `encode_ext`.
pub fn decode_ext(y: f64) -> f64 {
    let a = y.abs();
    let d = if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
    if y < 0.0 { -d } else { d }
}

/// An sRGB image (the ordinary look) to linear sRGB in place — the look's
/// inverse when the frames were not raws.
pub fn srgb_to_linear(img: &mut Img3) {
    for p in &mut img.p {
        p.par_iter_mut().for_each(|v| *v = decode_ext(*v as f64) as f32);
    }
}

/// Write the look image as a linear DNG: `info` says how it maps back to the
/// camera space and what the camera was; `meta` (the first frame's) supplies
/// the Exif IFD, XMP and the make and model when it has them. The ICC profile
/// is left out (a DNG's colour is its matrices), the orientation is 1 (the
/// frames were turned as decoded).
pub fn write<W: Write>(w: W, img: &Img3, info: &DngInfo, meta: Option<&Meta>) -> std::io::Result<()> {
    let samples = info.from_look(img);
    crate::meta::write_dng(w, img.w, img.h, &samples, info, meta)
}

/// `write` for an interleaved 16-bit look image.
pub fn write_u16<W: Write>(w: W, rgb16: &[u16], width: usize, height: usize, info: &DngInfo, meta: Option<&Meta>) -> std::io::Result<()> {
    let samples = info.from_look_u16(rgb16, width, height);
    crate::meta::write_dng(w, width, height, &samples, info, meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extended_curve_is_a_bijection() {
        for &x in &[-2.0, -0.5, -0.001, 0.0, 0.0001, 0.003, 0.02, 0.5, 1.0, 1.7, 4.0] {
            assert!((decode_ext(encode_ext(x)) - x).abs() < 1e-9, "{x}");
        }
        assert!((encode_ext(1.0) - 1.0).abs() < 1e-6);
        assert!((encode_ext(0.5) - 0.7353569).abs() < 1e-5);
    }

    #[test]
    fn look_and_back_round_trip_a_camera_image() {
        // a Nikon-like D65 matrix (XYZ → camera, ×1e-4 as the camera definitions have them)
        let m = [0.8198, -0.2239, -0.0753, -0.4638, 1.2251, 0.2635, -0.0644, 0.1436, 0.6486];
        let info = DngInfo::of_raw("NIKON", "Z 8", vec![(D65, m)], [2.1, 1.0, 1.6], m).unwrap();
        assert_eq!(info.wb, [2.1, 1.0, 1.6]);
        let n = info.as_shot_neutral();
        assert!((n[0] - 1.0 / 2.1).abs() < 1e-6 && n[1] == 1.0);
        // a neutral in camera space balances to grey and comes back
        let mut img = Img3::zeros(4, 1);
        let cam = [[1.0 / 2.1, 1.0, 1.0 / 1.6], [0.5 / 2.1, 0.5, 0.5 / 1.6], [0.9, 0.2, 0.1], [0.02, 0.7, 0.99]];
        for (i, c) in cam.iter().enumerate() {
            for k in 0..3 {
                img.p[k][i] = c[k];
            }
        }
        let cam_img = img.clone();
        info.to_look(&mut img);
        assert!((img.p[0][0] - 1.0).abs() < 1e-4 && (img.p[1][0] - 1.0).abs() < 1e-4 && (img.p[2][0] - 1.0).abs() < 1e-4, "white stays white: {:?}", [img.p[0][0], img.p[1][0], img.p[2][0]]);
        assert!((img.p[0][1] - encode_ext(0.5) as f32).abs() < 1e-4);
        let back = info.from_look(&img);
        for i in 0..4 {
            for k in 0..3 {
                let want = (cam_img.p[k][i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                assert!((back[3 * i + k] as i32 - want as i32).abs() <= 1, "pixel {i} channel {k}: {} vs {want}", back[3 * i + k]);
            }
        }
        // sRGB frames: the look is the file's own values
        let s = DngInfo::srgb();
        assert!(!s.is_camera());
        let mut g = Img3::zeros(1, 1);
        g.p[0][0] = 0.5; g.p[1][0] = 0.5; g.p[2][0] = 0.5;
        let lin = s.from_look(&g);
        assert_eq!(lin[0], (decode_ext(0.5) * 65535.0 + 0.5) as u16);
    }
}

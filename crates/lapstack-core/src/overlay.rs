// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

// Scale bar and text overlay, for the microscope.
//
// A stacked micrograph is a figure, and a figure carries a scale bar: a line
// of a round length (1, 2 or 5 × 10^n of a unit) with that length written
// over it, and often a caption — the specimen, the objective, the date. Both
// are burned into the saved images here, natively and in the browser, from
// one description (`OverlayParams`), so the two agree to the pixel:
//
// * The calibration is the size of one pixel of the frames in µm. Alignment
//   brings every frame onto frame 0's pixel grid and the crop only cuts that
//   grid, so one number serves every output at full size; an output shrunk
//   to a smaller long edge (an animation) passes the shrink as `scale`.
//   A bar asked for without a calibration is labelled in pixels of the
//   frames ("500 px"), so a bar is drawn whenever one is asked for and a
//   figure that is not calibrated still carries a scale.
// * The bar's length is the 1-2-5 value nearest a fifth of the width, or the
//   length asked for; its label picks the unit that keeps the number under a
//   thousand (500 nm, 100 µm, 2.5 mm). The bar is snapped to whole pixels.
// * Sizes follow the image: the font's em is `size` of the image height
//   (3 % by default), and the margins, the bar's thickness, the gap under
//   the label and the halo's width are fractions of the em, so the same
//   settings give the same figure at every resolution.
// * The text is set in Fira Sans, the app's own face — a 30 KB subset of the
//   Regular weight embedded below (`fonts/subset.py` made it from Mozilla's
//   TTF; SIL OFL 1.1) — and rasterised by this module: a TrueType outline
//   reader (glyf / loca / cmap / hmtx, simple and composite glyphs, no
//   hinting) and the signed-area coverage accumulation of font-rs /
//   stb_truetype v2: each edge deposits the area it sweeps into the pixels
//   it crosses, and a running sum along the row gives the exact coverage of
//   the nonzero-winding fill, anti-aliased for free.
// * The result is a few `Patch`es of coverage over the corners they occupy
//   (the ink, and behind it a halo — the ink dilated by a disc, in the
//   other colour — or a translucent box), so nothing the size of the image
//   is ever allocated, and compositing costs the patches alone. The same
//   patches serve the float image of the native path (`apply_f32`), the
//   16-bit master of the browser (`apply_u16`) and the viewer's preview
//   (`rgba8`).

use crate::pyramid::Img3;
use std::sync::OnceLock;

static FONT_BYTES: &[u8] = include_bytes!("../fonts/FiraSans-Regular.subset.ttf");

// ---------------------------------------------------------------------------
// parameters

/// A corner of the image.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    pub fn parse(s: &str) -> Option<Corner> {
        Some(match s.trim().to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
            "tl" | "topleft" => Corner::TopLeft,
            "tr" | "topright" => Corner::TopRight,
            "bl" | "bottomleft" => Corner::BottomLeft,
            "br" | "bottomright" => Corner::BottomRight,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Corner::TopLeft => "tl",
            Corner::TopRight => "tr",
            Corner::BottomLeft => "bl",
            Corner::BottomRight => "br",
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            Corner::TopLeft => "top left",
            Corner::TopRight => "top right",
            Corner::BottomLeft => "bottom left",
            Corner::BottomRight => "bottom right",
        }
    }
    fn top(self) -> bool {
        matches!(self, Corner::TopLeft | Corner::TopRight)
    }
    fn left(self) -> bool {
        matches!(self, Corner::TopLeft | Corner::BottomLeft)
    }
}

/// The ink: white with a black halo / box, or black with a white one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ink {
    White,
    Black,
}

impl Ink {
    pub fn parse(s: &str) -> Option<Ink> {
        match s.trim().to_ascii_lowercase().as_str() {
            "white" | "w" => Some(Ink::White),
            "black" | "k" | "b" => Some(Ink::Black),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Ink::White => "white",
            Ink::Black => "black",
        }
    }
}

/// What sits behind the ink so it reads on any background.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// The ink alone.
    Plain,
    /// A thin outline in the other colour (the ink dilated by a disc).
    Halo,
    /// A translucent box in the other colour behind each block.
    Box,
}

impl Style {
    pub fn parse(s: &str) -> Option<Style> {
        match s.trim().to_ascii_lowercase().as_str() {
            "plain" | "none" => Some(Style::Plain),
            "halo" | "outline" => Some(Style::Halo),
            "box" => Some(Style::Box),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Style::Plain => "plain",
            Style::Halo => "halo",
            Style::Box => "box",
        }
    }
}

/// The overlay asked for. Empty (`is_empty`) when there is neither a bar nor text.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlayParams {
    /// A scale bar wanted even without a calibration: labelled in pixels of the frames.
    pub bar: bool,
    /// The size of one pixel of the frames (frame 0's grid) in µm; 0 = not calibrated
    /// (no bar unless `bar` asks for a pixel one).
    pub um_per_px: f64,
    /// The bar's length in µm (in frame pixels without a calibration); 0 = the 1-2-5
    /// value nearest a fifth of the width.
    pub bar_um: f64,
    /// The caption; '\n' separates lines; empty = none.
    pub text: String,
    pub bar_pos: Corner,
    pub text_pos: Corner,
    /// The font size (the em) as a fraction of the image height.
    pub size: f32,
    pub color: Ink,
    pub style: Style,
}

impl Default for OverlayParams {
    fn default() -> Self {
        OverlayParams {
            bar: false,
            um_per_px: 0.0,
            bar_um: 0.0,
            text: String::new(),
            bar_pos: Corner::BottomRight,
            text_pos: Corner::BottomLeft,
            size: 0.03,
            color: Ink::White,
            style: Style::Halo,
        }
    }
}

impl OverlayParams {
    /// A calibration is given: the bar is a length.
    pub fn calibrated(&self) -> bool {
        self.um_per_px > 0.0 && self.um_per_px.is_finite()
    }
    pub fn has_bar(&self) -> bool {
        self.bar || self.calibrated()
    }
    pub fn has_text(&self) -> bool {
        !self.text.trim().is_empty()
    }
    pub fn is_empty(&self) -> bool {
        !self.has_bar() && !self.has_text()
    }
}

// ---------------------------------------------------------------------------
// lengths and units

/// Micrometres in one of a unit: nm, µm (um, μm, micron), mm, cm, m, inch, Å.
/// `None` for a unit that is not a length (ImageJ's "pixel") or unknown.
pub fn unit_um(u: &str) -> Option<f64> {
    Some(match u.trim().trim_end_matches('.').to_lowercase().as_str() {
        "nm" | "nanometer" | "nanometre" | "nanometers" | "nanometres" => 1e-3,
        "µm" | "um" | "μm" | "micron" | "microns" | "micrometer" | "micrometre" | "micrometers" | "micrometres" => 1.0,
        "mm" | "millimeter" | "millimetre" | "millimeters" | "millimetres" => 1e3,
        "cm" | "centimeter" | "centimetre" | "centimeters" | "centimetres" => 1e4,
        "m" | "meter" | "metre" | "meters" | "metres" => 1e6,
        "in" | "inch" | "inches" => 25400.0,
        "å" | "a" | "angstrom" | "angstroms" | "ångström" => 1e-4,
        _ => return None,
    })
}

/// A length with an optional unit ("100", "100um", "2.5 mm", "500 nm") in µm; the
/// unit defaults to µm. `None` when it does not parse or is not positive.
pub fn parse_length_um(s: &str) -> Option<f64> {
    let s = s.trim();
    let split = s.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == ',' || c == '-' || c == '+' || c == 'e' || c == 'E')).unwrap_or(s.len());
    // "e" starts a unit only when it is not an exponent ("1e3" is a number, "1 exa" is not a unit we know)
    let (num, unit) = s.split_at(split);
    let v: f64 = num.replace(',', ".").parse().ok()?;
    let k = if unit.trim().is_empty() { 1.0 } else { unit_um(unit)? };
    let um = v * k;
    (um > 0.0 && um.is_finite()).then_some(um)
}

/// A length in pixels ("500", "500px", "500 pixels") for a bar without a
/// calibration. `None` when it does not parse, is not positive, or carries a
/// unit of length.
pub fn parse_length_px(s: &str) -> Option<f64> {
    let s = s.trim();
    let split = s.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == ',' || c == '-' || c == '+')).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    if !matches!(unit.trim().to_ascii_lowercase().as_str(), "" | "px" | "pixel" | "pixels") {
        return None;
    }
    let v: f64 = num.replace(',', ".").parse().ok()?;
    (v > 0.0 && v.is_finite()).then_some(v)
}

/// The 1-2-5 × 10^n value nearest `target` in log scale.
pub fn nice_length_um(target: f64) -> f64 {
    if !(target > 0.0) || !target.is_finite() {
        return 1.0;
    }
    let l = target.log10();
    let e = l.floor();
    let mut best = 10f64.powi(e as i32);
    let mut dist = f64::INFINITY;
    for m in [1.0, 2.0, 5.0, 10.0] {
        let v = m * 10f64.powi(e as i32);
        let d = (v.log10() - l).abs();
        if d < dist {
            dist = d;
            best = v;
        }
    }
    best
}

/// The largest 1-2-5 × 10^n value not above `max`.
fn nice_floor_um(max: f64) -> f64 {
    if !(max > 0.0) || !max.is_finite() {
        return 1.0;
    }
    let e = max.log10().floor();
    let base = 10f64.powi(e as i32);
    let mut best = base;
    for m in [1.0, 2.0, 5.0] {
        if m * base <= max * (1.0 + 1e-9) {
            best = m * base;
        }
    }
    best
}

/// A length in µm as a label: the unit that keeps the number in [1, 1000),
/// the number with up to three significant digits and no trailing zeros —
/// "500 nm", "100 µm", "2.5 mm", "1 m".
pub fn format_length(um: f64) -> String {
    let (v, unit) = if um < 1.0 {
        (um * 1e3, "nm")
    } else if um < 1e3 {
        (um, "µm")
    } else if um < 1e6 {
        (um / 1e3, "mm")
    } else {
        (um / 1e6, "m")
    };
    let s = if (v - v.round()).abs() < 1e-6 * v.max(1.0) {
        format!("{:.0}", v.round())
    } else {
        let digits = (2 - v.log10().floor() as i32).clamp(0, 6) as usize;
        let s = format!("{v:.digits$}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    format!("{s} {unit}")
}

// ---------------------------------------------------------------------------
// the font: a TrueType outline reader for the embedded subset

fn be16(b: &[u8], o: usize) -> u16 {
    match b.get(o..o + 2) {
        Some(v) => u16::from_be_bytes([v[0], v[1]]),
        None => 0,
    }
}
fn bi16(b: &[u8], o: usize) -> i16 {
    be16(b, o) as i16
}
fn be32(b: &[u8], o: usize) -> u32 {
    match b.get(o..o + 4) {
        Some(v) => u32::from_be_bytes([v[0], v[1], v[2], v[3]]),
        None => 0,
    }
}

/// A point of an outline in font units: (x, y, on the curve).
type Pt = (f32, f32, bool);

pub struct Font {
    data: &'static [u8],
    upm: f32,
    loca: usize,
    loca_long: bool,
    glyf: usize,
    glyf_len: usize,
    hmtx: usize,
    num_h: u16,
    num_glyphs: u16,
    /// The format 4 cmap subtable.
    cmap4: usize,
    /// Metrics as fractions of the em: the ascender, the descender (positive,
    /// below the baseline) and the height of a capital (the 'H').
    pub ascent: f32,
    pub descent: f32,
    pub cap: f32,
}

impl Font {
    /// The embedded face (Fira Sans Regular, subset).
    pub fn builtin() -> &'static Font {
        static FONT: OnceLock<Font> = OnceLock::new();
        FONT.get_or_init(|| Font::parse(FONT_BYTES).expect("the embedded font parses"))
    }

    fn parse(data: &'static [u8]) -> Option<Font> {
        if data.len() < 12 || be32(data, 0) != 0x0001_0000 {
            return None;
        }
        let n = be16(data, 4) as usize;
        let table = |tag: &[u8; 4]| -> Option<(usize, usize)> {
            (0..n).map(|i| 12 + 16 * i).find(|&o| data.get(o..o + 4) == Some(&tag[..])).map(|o| (be32(data, o + 8) as usize, be32(data, o + 12) as usize))
        };
        let (head, _) = table(b"head")?;
        let (hhea, _) = table(b"hhea")?;
        let (maxp, _) = table(b"maxp")?;
        let (loca, _) = table(b"loca")?;
        let (glyf, glyf_len) = table(b"glyf")?;
        let (hmtx, _) = table(b"hmtx")?;
        let (cmap, _) = table(b"cmap")?;
        let upm = be16(data, head + 18) as f32;
        if upm <= 0.0 {
            return None;
        }
        // the first format 4 subtable
        let nsub = be16(data, cmap + 2) as usize;
        let cmap4 = (0..nsub).map(|i| cmap + be32(data, cmap + 8 + 8 * i) as usize).find(|&o| be16(data, o) == 4)?;
        let mut f = Font {
            data,
            upm,
            loca,
            loca_long: bi16(data, head + 50) != 0,
            glyf,
            glyf_len,
            hmtx,
            num_h: be16(data, hhea + 34),
            num_glyphs: be16(data, maxp + 4),
            cmap4,
            ascent: bi16(data, hhea + 4) as f32 / upm,
            descent: -(bi16(data, hhea + 6) as f32) / upm,
            cap: 0.7,
        };
        let h = f.glyph_id('H');
        let g = f.glyph(h);
        if g.len() >= 10 {
            f.cap = bi16(g, 8) as f32 / upm;
        }
        Some(f)
    }

    /// The glyph of a character; 0 (.notdef) when the face has none.
    pub fn glyph_id(&self, c: char) -> u16 {
        let c = c as u32;
        if c > 0xFFFF {
            return 0;
        }
        let d = self.data;
        let t = self.cmap4;
        let seg_x2 = be16(d, t + 6) as usize;
        let seg = seg_x2 / 2;
        let (ends, starts, deltas, ros) = (t + 14, t + 16 + seg_x2, t + 16 + 2 * seg_x2, t + 16 + 3 * seg_x2);
        for s in 0..seg {
            let end = be16(d, ends + 2 * s) as u32;
            if c > end {
                continue;
            }
            let start = be16(d, starts + 2 * s) as u32;
            if c < start {
                return 0;
            }
            let delta = be16(d, deltas + 2 * s);
            let ro = be16(d, ros + 2 * s) as usize;
            if ro == 0 {
                return (c as u16).wrapping_add(delta);
            }
            let g = be16(d, ros + 2 * s + ro + 2 * (c - start) as usize);
            return if g == 0 { 0 } else { g.wrapping_add(delta) };
        }
        0
    }

    /// The advance of a glyph in font units.
    fn advance(&self, g: u16) -> f32 {
        let i = g.min(self.num_h.saturating_sub(1)) as usize;
        be16(self.data, self.hmtx + 4 * i) as f32
    }

    /// The glyph's record in glyf (empty for a glyph without an outline, the space).
    fn glyph(&self, g: u16) -> &'static [u8] {
        if g >= self.num_glyphs {
            return &[];
        }
        let g = g as usize;
        let (a, b) = if self.loca_long {
            (be32(self.data, self.loca + 4 * g) as usize, be32(self.data, self.loca + 4 * g + 4) as usize)
        } else {
            (2 * be16(self.data, self.loca + 2 * g) as usize, 2 * be16(self.data, self.loca + 2 * g + 2) as usize)
        };
        if b <= a || b > self.glyf_len {
            return &[];
        }
        self.data.get(self.glyf + a..self.glyf + b).unwrap_or(&[])
    }

    /// The contours of a glyph in font units, composites resolved; `tf` is the
    /// affine map [a, b, c, d, dx, dy]: x' = a·x + c·y + dx, y' = b·x + d·y + dy.
    fn contours_into(&self, g: u16, tf: &[f32; 6], depth: u8, out: &mut Vec<Vec<Pt>>) {
        let d = self.glyph(g);
        if d.len() < 10 {
            return;
        }
        let nc = bi16(d, 0);
        if nc < 0 {
            // a composite: each component with its own offset and scale
            if depth > 4 {
                return;
            }
            let mut p = 10;
            loop {
                if p + 4 > d.len() {
                    break;
                }
                let flags = be16(d, p);
                let gi = be16(d, p + 2);
                p += 4;
                let (dx, dy) = if flags & 1 != 0 {
                    let v = (bi16(d, p) as f32, bi16(d, p + 2) as f32);
                    p += 4;
                    v
                } else {
                    let v = (d.get(p).map_or(0, |&b| b as i8) as f32, d.get(p + 1).map_or(0, |&b| b as i8) as f32);
                    p += 2;
                    v
                };
                // arguments that are point indices to match (not offsets) are rare; taken as no offset
                let (dx, dy) = if flags & 2 != 0 { (dx, dy) } else { (0.0, 0.0) };
                let f2 = |o: usize| bi16(d, o) as f32 / 16384.0;
                let (mut a, mut b, mut c, mut dd) = (1.0, 0.0, 0.0, 1.0);
                if flags & 8 != 0 {
                    a = f2(p);
                    dd = a;
                    p += 2;
                } else if flags & 0x40 != 0 {
                    a = f2(p);
                    dd = f2(p + 2);
                    p += 4;
                } else if flags & 0x80 != 0 {
                    a = f2(p);
                    b = f2(p + 2);
                    c = f2(p + 4);
                    dd = f2(p + 6);
                    p += 8;
                }
                // the component's map, then the parent's
                let (a1, b1, c1, d1, e1, f1) = (a, b, c, dd, dx, dy);
                let [a2, b2, c2, d2, e2, f2_] = *tf;
                let comp = [a2 * a1 + c2 * b1, b2 * a1 + d2 * b1, a2 * c1 + c2 * d1, b2 * c1 + d2 * d1, a2 * e1 + c2 * f1 + e2, b2 * e1 + d2 * f1 + f2_];
                self.contours_into(gi, &comp, depth + 1, out);
                if flags & 0x20 == 0 {
                    break;
                }
            }
            return;
        }
        let nc = nc as usize;
        let mut p = 10;
        let ends: Vec<usize> = (0..nc).map(|i| be16(d, p + 2 * i) as usize).collect();
        p += 2 * nc;
        let n = ends.last().map_or(0, |e| e + 1);
        if n == 0 || n > 4096 {
            return;
        }
        let insn = be16(d, p) as usize;
        p += 2 + insn;
        let mut flags = Vec::with_capacity(n);
        while flags.len() < n {
            let Some(&f) = d.get(p) else { return };
            p += 1;
            flags.push(f);
            if f & 8 != 0 {
                let Some(&r) = d.get(p) else { return };
                p += 1;
                for _ in 0..r {
                    flags.push(f);
                }
            }
        }
        flags.truncate(n);
        let mut read = |short: u8, same: u8| -> Option<Vec<i32>> {
            let mut v = 0i32;
            let mut out = Vec::with_capacity(n);
            for &f in &flags {
                if f & short != 0 {
                    let dv = *d.get(p)? as i32;
                    p += 1;
                    v += if f & same != 0 { dv } else { -dv };
                } else if f & same == 0 {
                    d.get(p..p + 2)?;
                    v += bi16(d, p) as i32;
                    p += 2;
                }
                out.push(v);
            }
            Some(out)
        };
        let Some(xs) = read(2, 16) else { return };
        let Some(ys) = read(4, 32) else { return };
        let [a, b, c, dd, dx, dy] = *tf;
        let mut start = 0;
        for &e in &ends {
            if e < start || e >= n {
                break;
            }
            let pts: Vec<Pt> = (start..=e)
                .map(|i| {
                    let (x, y) = (xs[i] as f32, ys[i] as f32);
                    (a * x + c * y + dx, b * x + dd * y + dy, flags[i] & 1 != 0)
                })
                .collect();
            out.push(pts);
            start = e + 1;
        }
    }

    fn contours(&self, g: u16) -> Vec<Vec<Pt>> {
        let mut out = Vec::new();
        self.contours_into(g, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0, &mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// the rasteriser: signed-area coverage accumulation (font-rs)

struct Raster {
    w: usize,
    h: usize,
    a: Vec<f32>,
}

impl Raster {
    fn new(w: usize, h: usize) -> Raster {
        Raster { w, h, a: vec![0.0; w * h + 4] }
    }

    #[inline]
    fn add(&mut self, i: isize, v: f32) {
        if i >= 0 && (i as usize) < self.w * self.h {
            self.a[i as usize] += v;
        }
    }

    /// One edge of the outline, in pixel coordinates (y down). Each pixel the
    /// edge crosses receives the signed area it sweeps there; the fill is the
    /// running sum along the row.
    fn line(&mut self, p0: (f32, f32), p1: (f32, f32)) {
        if (p0.1 - p1.1).abs() <= f32::EPSILON || !p0.0.is_finite() || !p1.0.is_finite() {
            return;
        }
        let (dir, p0, p1) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
        let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
        let mut x = p0.0;
        let y0 = p0.1.max(0.0) as usize;
        if p0.1 < 0.0 {
            x -= p0.1 * dxdy;
        }
        let y1 = p1.1.ceil().min(self.h as f32).max(0.0) as usize;
        let w = self.w as isize;
        for y in y0..y1 {
            let linestart = (y * self.w) as isize;
            let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let x0floor = x0.floor();
            let x0i = x0floor as isize;
            let x1ceil = x1.ceil();
            let x1i = x1ceil as isize;
            // an edge outside the raster's columns deposits nothing here; a line left of it
            // would owe the row its whole area, which `add` clamps to the row's first pixel
            if x0i >= w {
                x = xnext;
                continue;
            }
            if x1i <= 0 {
                self.add(linestart, d);
                x = xnext;
                continue;
            }
            if x1i <= x0i + 1 {
                let xmf = 0.5 * (x + xnext) - x0floor;
                self.add(linestart + x0i, d - d * xmf);
                self.add(linestart + x0i + 1, d * xmf);
            } else {
                let s = (x1 - x0).recip();
                let x0f = x0 - x0floor;
                let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
                let x1f = x1 - x1ceil + 1.0;
                let am = 0.5 * s * x1f * x1f;
                self.add(linestart + x0i, d * a0);
                if x1i == x0i + 2 {
                    self.add(linestart + x0i + 1, d * (1.0 - a0 - am));
                } else {
                    let a1 = s * (1.5 - x0f);
                    self.add(linestart + x0i + 1, d * (a1 - a0));
                    for xi in x0i + 2..x1i - 1 {
                        self.add(linestart + xi, d * s);
                    }
                    let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                    self.add(linestart + x1i - 1, d * (1.0 - a2 - am));
                }
                self.add(linestart + x1i, d * am);
            }
            x = xnext;
        }
    }

    /// A quadratic Bézier, flattened to as many lines as its deviation asks for.
    fn quad(&mut self, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) {
        let devx = p0.0 - 2.0 * p1.0 + p2.0;
        let devy = p0.1 - 2.0 * p1.1 + p2.1;
        let devsq = devx * devx + devy * devy;
        if devsq < 0.333 {
            self.line(p0, p2);
            return;
        }
        let n = (1.0 + (3.0 * devsq).sqrt().sqrt().floor()).min(64.0) as usize;
        let mut prev = p0;
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let mt = 1.0 - t;
            let p = (mt * mt * p0.0 + 2.0 * mt * t * p1.0 + t * t * p2.0, mt * mt * p0.1 + 2.0 * mt * t * p1.1 + t * t * p2.1);
            self.line(prev, p);
            prev = p;
        }
    }

    /// A closed TrueType contour: on-curve points joined by lines, off-curve
    /// points the control of a quadratic, two in a row implying the on-curve
    /// point between them. `map` takes font units to pixels.
    fn contour(&mut self, pts: &[Pt], map: &dyn Fn(f32, f32) -> (f32, f32)) {
        let n = pts.len();
        if n < 2 {
            return;
        }
        let at = |i: usize| -> (f32, f32) { map(pts[i % n].0, pts[i % n].1) };
        let mid = |a: (f32, f32), b: (f32, f32)| ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        let first_on = pts.iter().position(|p| p.2);
        let s0 = first_on.unwrap_or(0);
        let start = match first_on {
            Some(i) => at(i),
            None => mid(at(0), at(1)),
        };
        let mut cur = start;
        let mut ctrl: Option<(f32, f32)> = None;
        for k in 1..=n {
            let i = (s0 + k) % n;
            let q = at(i);
            if pts[i].2 {
                match ctrl {
                    Some(c) => self.quad(cur, c, q),
                    None => self.line(cur, q),
                }
                cur = q;
                ctrl = None;
            } else {
                if let Some(c) = ctrl {
                    let m = mid(c, q);
                    self.quad(cur, c, m);
                    cur = m;
                }
                ctrl = Some(q);
            }
        }
        match ctrl {
            Some(c) => self.quad(cur, c, start),
            None => {
                if cur != start {
                    self.line(cur, start);
                }
            }
        }
    }

    /// An axis-aligned rectangle (pixel coordinates), edges anti-aliased like any outline.
    fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        self.line((x0, y0), (x0, y1));
        self.line((x0, y1), (x1, y1));
        self.line((x1, y1), (x1, y0));
        self.line((x1, y0), (x0, y0));
    }

    /// The coverage in [0, 1] of the nonzero-winding fill: the running sum of
    /// the deposited areas along each row (a closed outline sums to zero over
    /// a row, so each row starts afresh).
    fn coverage(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.w * self.h];
        for y in 0..self.h {
            let mut acc = 0.0;
            for x in 0..self.w {
                acc += self.a[y * self.w + x];
                out[y * self.w + x] = acc.abs().min(1.0);
            }
        }
        out
    }
}

/// The `Ink` mask grown by a disc of radius `r` px (the halo): the maximum of
/// the coverage over the disc, so soft edges stay soft. Each row of the
/// output takes, for each row of the disc, the sliding maximum of that
/// source row over the disc's half-width there.
fn dilate(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut out = src.to_vec();
    if r == 0 {
        return out;
    }
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::with_capacity(2 * r + 2);
    let mut row_max = vec![0.0f32; w];
    for dy in -(r as isize)..=(r as isize) {
        let hw = ((r * r) as f32 - (dy * dy) as f32).max(0.0).sqrt().floor() as usize;
        for y in 0..h {
            let sy = y as isize + dy;
            if sy < 0 || sy >= h as isize {
                continue;
            }
            let row = &src[sy as usize * w..(sy as usize + 1) * w];
            // sliding maximum over [x - hw, x + hw]
            deque.clear();
            for x in 0..w + hw {
                if x < w {
                    while let Some(&back) = deque.back() {
                        if row[back] <= row[x] {
                            deque.pop_back();
                        } else {
                            break;
                        }
                    }
                    deque.push_back(x);
                }
                if x >= hw {
                    let ox = x - hw;
                    while let Some(&front) = deque.front() {
                        if front + hw < ox {
                            deque.pop_front();
                        } else {
                            break;
                        }
                    }
                    row_max[ox] = deque.front().map_or(0.0, |&i| row[i]);
                }
            }
            let o = &mut out[y * w..(y + 1) * w];
            for x in 0..w {
                if row_max[x] > o[x] {
                    o[x] = row_max[x];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// layout

/// A line of text laid out at a font size: the glyphs with their x positions, in px.
struct Line {
    glyphs: Vec<(u16, f32)>,
    width: f32,
}

fn layout(font: &Font, text: &str, em: f32) -> Line {
    let s = em / font.upm;
    let mut x = 0.0;
    let mut glyphs = Vec::with_capacity(text.chars().count());
    for c in text.chars() {
        if c == '\t' {
            x += 4.0 * font.advance(font.glyph_id(' ')) * s;
            continue;
        }
        let g = font.glyph_id(c);
        glyphs.push((g, x));
        x += font.advance(g) * s;
    }
    Line { glyphs, width: x }
}

fn draw_line(r: &mut Raster, font: &Font, line: &Line, x: f32, baseline: f32, em: f32) {
    let s = em / font.upm;
    for &(g, gx) in &line.glyphs {
        let ox = x + gx;
        let map = move |fx: f32, fy: f32| (ox + s * fx, baseline - s * fy);
        for c in font.contours(g) {
            r.contour(&c, &map);
        }
    }
}

/// Something to draw, in pixel coordinates of the image.
enum Item {
    Text { line: Line, x: f32, baseline: f32 },
    Rect { x0: f32, y0: f32, x1: f32, y1: f32 },
}

/// A block of the overlay: its bounds and its items.
struct Block {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    items: Vec<Item>,
}

// ---------------------------------------------------------------------------
// the overlay

/// The coverage of one block of the overlay over its bounding box: the ink,
/// and what lies behind it (the halo or the box; empty for `Style::Plain`).
pub struct Patch {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub ink: Vec<f32>,
    pub back: Vec<f32>,
}

/// The rendered overlay for an image of `w`×`h`: a patch per block, the
/// colours, and what the bar came to.
pub struct Overlay {
    pub w: usize,
    pub h: usize,
    pub patches: Vec<Patch>,
    pub ink_rgb: [f32; 3],
    pub back_rgb: [f32; 3],
    /// The opacity of `back` (1 for a halo, less for a box).
    pub back_alpha: f32,
    /// The bar's length in µm (frame pixels without a calibration) and in output
    /// pixels (0 without a bar), and its label.
    pub bar_um: f64,
    pub bar_px: f32,
    pub label: String,
    pub params: OverlayParams,
}

impl Overlay {
    /// Render the overlay for an output of `w`×`h` pixels. `scale` is the
    /// output's pixels per pixel of the frames: 1 at full size, 0.5 for an
    /// output shrunk to half (the calibration is scaled with it).
    pub fn render(p: &OverlayParams, w: usize, h: usize, scale: f64) -> Overlay {
        let font = Font::builtin();
        let (ink_rgb, back_rgb) = match p.color {
            Ink::White => ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
            Ink::Black => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        };
        let mut ov = Overlay {
            w,
            h,
            patches: Vec::new(),
            ink_rgb,
            back_rgb,
            back_alpha: if p.style == Style::Box { 0.6 } else { 1.0 },
            bar_um: 0.0,
            bar_px: 0.0,
            label: String::new(),
            params: p.clone(),
        };
        if w == 0 || h == 0 || p.is_empty() || !(scale > 0.0) {
            return ov;
        }
        let em = (p.size.clamp(0.002, 0.5) * h as f32).max(4.0);
        let margin = 0.8 * em;
        let cap = font.cap * em;
        let desc = font.descent * em;
        let line_h = 1.25 * em;
        let halo = ((0.05 * em).round() as usize).max(1);
        let box_pad = 0.35 * em;
        let pad = match p.style {
            Style::Plain => 1.0,
            Style::Halo => halo as f32 + 1.0,
            Style::Box => box_pad + 1.0,
        };
        let mut blocks: Vec<(Corner, Block)> = Vec::new();

        // the scale bar with its label over it: a length at the calibration, or a
        // count of frame pixels without one (a unit of 1 frame pixel = 1/scale output pixels)
        if p.has_bar() {
            let um_out = if p.calibrated() { p.um_per_px } else { 1.0 } / scale;   // units per output pixel
            let max_px = (w as f32 - 2.0 * margin - 2.0 * pad).max(2.0) as f64;
            let mut bar_um = if p.bar_um > 0.0 { p.bar_um } else { nice_length_um(0.2 * w as f64 * um_out) };
            if bar_um / um_out > max_px {
                bar_um = nice_floor_um(max_px * um_out);
            }
            let bar_px = (bar_um / um_out).round().max(1.0) as f32;
            let label = if p.calibrated() { format_length(bar_um) } else { format!("{} px", trim_float(bar_um)) };
            let line = layout(font, &label, em);
            let t = (0.1 * em).round().max(1.0);
            let gap = 0.2 * em;
            let bw = bar_px.max(line.width);
            let bh = cap + desc + gap + t;
            let (x, y) = corner_xy(p.bar_pos, w, h, margin, bw, bh);
            let (x, y) = (x.round(), y.round());
            let bx = (x + (bw - bar_px) * 0.5).round();
            let by = (y + cap + desc + gap).round();
            let items = vec![
                Item::Text { x: x + (bw - line.width) * 0.5, baseline: y + cap, line },
                Item::Rect { x0: bx, y0: by, x1: bx + bar_px, y1: by + t },
            ];
            ov.bar_um = bar_um;
            ov.bar_px = bar_px;
            ov.label = label;
            blocks.push((p.bar_pos, Block { x, y, w: bw, h: bh, items }));
        }

        // the caption
        if p.has_text() {
            let lines: Vec<Line> = p.text.trim_matches('\n').split('\n').map(|l| layout(font, l.trim_end(), em)).collect();
            let bw = lines.iter().map(|l| l.width).fold(0.0, f32::max);
            let n = lines.len();
            let bh = (n as f32 - 1.0) * line_h + cap + desc;
            let (mut x, mut y) = corner_xy(p.text_pos, w, h, margin, bw, bh);
            // in the bar's corner the text goes above the bar (below it at the top)
            if let Some((_, b)) = blocks.iter().find(|(c, _)| *c == p.text_pos) {
                let gap2 = 0.6 * em + if p.style == Style::Box { 2.0 * box_pad } else { 0.0 };
                y = if p.text_pos.top() { b.y + b.h + gap2 } else { b.y - gap2 - bh };
                x = if p.text_pos.left() { b.x } else { b.x + b.w - bw };
            }
            let (x, y) = (x.round(), y.round());
            let left = p.text_pos.left();
            let items = lines
                .into_iter()
                .enumerate()
                .map(|(i, line)| Item::Text { x: if left { x } else { x + bw - line.width }, baseline: y + cap + i as f32 * line_h, line })
                .collect();
            blocks.push((p.text_pos, Block { x, y, w: bw, h: bh, items }));
        }

        for (_, b) in blocks {
            let x0 = (b.x - pad).floor().max(0.0) as usize;
            let y0 = (b.y - pad).floor().max(0.0) as usize;
            let x1 = ((b.x + b.w + pad).ceil().max(0.0) as usize).min(w);
            let y1 = ((b.y + b.h + pad).ceil().max(0.0) as usize).min(h);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let (pw, ph) = (x1 - x0, y1 - y0);
            let mut r = Raster::new(pw, ph);
            let (ox, oy) = (x0 as f32, y0 as f32);
            for it in &b.items {
                match it {
                    Item::Text { line, x, baseline } => draw_line(&mut r, font, line, x - ox, baseline - oy, em),
                    Item::Rect { x0, y0, x1, y1 } => r.rect(x0 - ox, y0 - oy, x1 - ox, y1 - oy),
                }
            }
            let ink = r.coverage();
            let back = match p.style {
                Style::Plain => Vec::new(),
                Style::Halo => dilate(&ink, pw, ph, halo),
                Style::Box => {
                    let mut rb = Raster::new(pw, ph);
                    rb.rect(b.x - box_pad - ox, b.y - box_pad - oy, b.x + b.w + box_pad - ox, b.y + b.h + box_pad - oy);
                    rb.coverage()
                }
            };
            ov.patches.push(Patch { x: x0, y: y0, w: pw, h: ph, ink, back });
        }
        ov
    }

    /// True when there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// One line for a log: what the bar came to and where everything sits.
    pub fn describe(&self) -> String {
        let p = &self.params;
        let mut parts = Vec::new();
        if self.bar_px > 0.0 {
            let mut s = if p.calibrated() {
                format!("scale bar {} = {} px at {} µm/px, {}", self.label, self.bar_px, trim_float(p.um_per_px), p.bar_pos.describe())
            } else if self.bar_px == self.bar_um as f32 {
                format!("scale bar {} (no calibration), {}", self.label, p.bar_pos.describe())
            } else {
                format!("scale bar {} of the frames = {} px (no calibration), {}", self.label, self.bar_px, p.bar_pos.describe())
            };
            if p.bar_um > 0.0 && (p.bar_um - self.bar_um).abs() > 1e-9 * p.bar_um {
                let asked = if p.calibrated() { format_length(p.bar_um) } else { format!("{} px", trim_float(p.bar_um)) };
                s.push_str(&format!(" (the {asked} asked for does not fit)"));
            }
            parts.push(s);
        }
        if p.has_text() {
            parts.push(format!("text \"{}\", {}", p.text.replace('\n', " / "), p.text_pos.describe()));
        }
        if parts.is_empty() {
            return "nothing to draw".into();
        }
        format!("{}; {} on {}, {:.1} % of the height", parts.join("; "), p.color.name(), p.style.name(), p.size * 100.0)
    }

    /// The blend of a pixel: `v` in [0, 1], the patch's coverages at `i`.
    #[inline]
    fn blend(&self, patch: &Patch, i: usize, c: usize, v: f32) -> f32 {
        let mut v = v;
        if !patch.back.is_empty() {
            let b = patch.back[i] * self.back_alpha;
            if b > 0.0 {
                v = v * (1.0 - b) + self.back_rgb[c] * b;
            }
        }
        let k = patch.ink[i];
        if k > 0.0 {
            v = v * (1.0 - k) + self.ink_rgb[c] * k;
        }
        v
    }

    /// Burn the overlay into a float image of the overlay's size.
    pub fn apply_f32(&self, img: &mut Img3) {
        for patch in &self.patches {
            for y in 0..patch.h {
                let iy = patch.y + y;
                if iy >= img.h {
                    break;
                }
                for x in 0..patch.w {
                    let ix = patch.x + x;
                    if ix >= img.w {
                        break;
                    }
                    let i = y * patch.w + x;
                    for c in 0..3 {
                        let v = &mut img.p[c][iy * img.w + ix];
                        *v = self.blend(patch, i, c, *v);
                    }
                }
            }
        }
    }

    /// The pixels `apply_f32` will change, to put back with `restore_f32`
    /// (the native path saves the image with the overlay, then shears and
    /// textures the clean one).
    pub fn under_f32(&self, img: &Img3) -> Vec<f32> {
        let mut out = Vec::new();
        for patch in &self.patches {
            for y in 0..patch.h {
                let iy = patch.y + y;
                if iy >= img.h {
                    break;
                }
                let x1 = (patch.x + patch.w).min(img.w);
                for c in 0..3 {
                    out.extend_from_slice(&img.p[c][iy * img.w + patch.x..iy * img.w + x1]);
                }
            }
        }
        out
    }

    pub fn restore_f32(&self, img: &mut Img3, under: &[f32]) {
        let mut k = 0;
        for patch in &self.patches {
            for y in 0..patch.h {
                let iy = patch.y + y;
                if iy >= img.h {
                    break;
                }
                let x1 = (patch.x + patch.w).min(img.w);
                let n = x1 - patch.x;
                for c in 0..3 {
                    img.p[c][iy * img.w + patch.x..iy * img.w + x1].copy_from_slice(&under[k..k + n]);
                    k += n;
                }
            }
        }
    }

    /// Burn the overlay into an interleaved RGB u16 image whose rows are
    /// `stride` pixels wide, the overlay's origin at (`x0`, `y0`) — the two
    /// halves of a stereo pair are one image with the overlay drawn twice.
    pub fn apply_u16(&self, rgb: &mut [u16], stride: usize, x0: usize, y0: usize) {
        let rows = rgb.len() / (3 * stride.max(1));
        for patch in &self.patches {
            for y in 0..patch.h {
                let iy = y0 + patch.y + y;
                if iy >= rows {
                    break;
                }
                for x in 0..patch.w {
                    let ix = x0 + patch.x + x;
                    if ix >= stride {
                        break;
                    }
                    let i = y * patch.w + x;
                    let px = &mut rgb[(iy * stride + ix) * 3..(iy * stride + ix) * 3 + 3];
                    for c in 0..3 {
                        px[c] = (self.blend(patch, i, c, px[c] as f32 / 65535.0) * 65535.0 + 0.5) as u16;
                    }
                }
            }
        }
    }

    /// The patches as straight-alpha RGBA8 bitmaps, (x, y, w, h, pixels), to
    /// draw over a canvas.
    pub fn rgba8(&self) -> Vec<(usize, usize, usize, usize, Vec<u8>)> {
        self.patches
            .iter()
            .map(|p| {
                let mut out = vec![0u8; p.w * p.h * 4];
                for i in 0..p.w * p.h {
                    let b = if p.back.is_empty() { 0.0 } else { p.back[i] * self.back_alpha };
                    let k = p.ink[i];
                    let a = b + k - b * k;
                    if a <= 0.0 {
                        continue;
                    }
                    for c in 0..3 {
                        let v = (self.ink_rgb[c] * k + self.back_rgb[c] * b * (1.0 - k)) / a;
                        out[4 * i + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                    out[4 * i + 3] = (a.min(1.0) * 255.0 + 0.5) as u8;
                }
                (p.x, p.y, p.w, p.h, out)
            })
            .collect()
    }
}

/// Where a block of `bw`×`bh` sits in a corner, `margin` from the edges.
fn corner_xy(c: Corner, w: usize, h: usize, margin: f32, bw: f32, bh: f32) -> (f32, f32) {
    let x = if c.left() { margin } else { w as f32 - margin - bw };
    let y = if c.top() { margin } else { h as f32 - margin - bh };
    (x, y)
}

fn trim_float(v: f64) -> String {
    let s = format!("{v:.6}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `{date}`, `{time}`, `{frames}`, `{first}`, `{n}` in a caption: the capture
/// date and time of the first frame (seconds by the camera's clock, as
/// `Meta::capture_time` gives them), the frame count, the first frame's stem
/// and the stack's number in a batch.
pub fn expand_text(text: &str, capture: Option<f64>, frames: usize, first: &str, n: usize) -> String {
    let (date, time) = match capture {
        Some(t) => {
            let secs = t.floor() as i64;
            let days = secs.div_euclid(86400);
            let s = secs.rem_euclid(86400);
            let (y, m, d) = civil_from_days(days);
            (format!("{y:04}-{m:02}-{d:02}"), format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60))
        }
        None => (String::new(), String::new()),
    };
    text.replace("\\n", "\n")
        .replace("{date}", &date)
        .replace("{time}", &time)
        .replace("{frames}", &frames.to_string())
        .replace("{first}", first)
        .replace("{n}", &n.to_string())
}

/// The proleptic Gregorian date of a day count since 1970-01-01 (Howard Hinnant).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_parses() {
        let f = Font::builtin();
        assert_eq!(f.upm, 1000.0);
        assert_ne!(f.glyph_id('A'), 0);
        assert_ne!(f.glyph_id('µ'), 0, "the micro sign is in the subset");
        assert_ne!(f.glyph_id('μ'), 0, "and the Greek mu");
        assert_ne!(f.glyph_id('é'), 0);
        assert_eq!(f.glyph_id('\u{4e2d}'), 0, "a character outside the subset is .notdef");
        assert!(f.advance(f.glyph_id('A')) > 300.0 && f.advance(f.glyph_id('A')) < 900.0);
        assert!(f.cap > 0.6 && f.cap < 0.8, "cap height {}", f.cap);
        assert!(f.descent > 0.15 && f.descent < 0.4, "descent {}", f.descent);
        assert!(!f.contours(f.glyph_id('é')).is_empty(), "a composite glyph resolves to outlines");
    }

    /// An 'H' at 40 px: two stems and a bar of solid ink, edges anti-aliased.
    #[test]
    fn glyph_renders() {
        let f = Font::builtin();
        let line = layout(f, "H", 40.0);
        let mut r = Raster::new(40, 44);
        draw_line(&mut r, f, &line, 4.0, 36.0, 40.0);
        let cov = r.coverage();
        let sum: f32 = cov.iter().sum();
        let max = cov.iter().cloned().fold(0.0, f32::max);
        assert!(sum > 150.0 && sum < 700.0, "ink {sum}");
        assert!(max > 0.99, "solid ink inside the stems, got {max}");
        // a row through the lower half: the stems are solid and the space between them empty;
        // the row through the crossbar is ink from stem to stem
        let cap = (f.cap * 40.0) as usize;
        let row = &cov[(36 - cap / 4) * 40..(36 - cap / 4 + 1) * 40];
        let x_first = row.iter().position(|&v| v > 0.5).unwrap();
        let x_last = row.iter().rposition(|&v| v > 0.5).unwrap();
        assert!(row[(x_first + x_last) / 2] < 0.05, "the space between the stems is empty");
        assert!(row.iter().any(|&v| v > 0.05 && v < 0.95), "anti-aliased edges");
        let xm = (x_first + x_last) / 2;
        let column: f32 = (36 - cap..36).map(|y| cov[y * 40 + xm]).sum();
        assert!(column > 1.5 && column < cap as f32 * 0.5, "the crossbar alone crosses the middle column: {column} px of ink");
    }

    #[test]
    fn nice_lengths() {
        assert_eq!(nice_length_um(19.0), 20.0);
        assert_eq!(nice_length_um(34.0), 50.0);
        assert_eq!(nice_length_um(31.0), 20.0);
        assert!((nice_length_um(0.007) - 0.005).abs() < 1e-12);
        assert_eq!(nice_length_um(100.0), 100.0);
        assert_eq!(nice_floor_um(499.0), 200.0);
        assert_eq!(nice_floor_um(500.0), 500.0);
    }

    #[test]
    fn labels() {
        assert_eq!(format_length(100.0), "100 µm");
        assert_eq!(format_length(0.5), "500 nm");
        assert_eq!(format_length(2500.0), "2.5 mm");
        assert_eq!(format_length(1e6), "1 m");
        assert_eq!(format_length(0.02), "20 nm");
        assert_eq!(format_length(12.5), "12.5 µm");
    }

    #[test]
    fn lengths_parse() {
        assert_eq!(parse_length_um("100"), Some(100.0));
        assert_eq!(parse_length_um("2 mm"), Some(2000.0));
        assert_eq!(parse_length_um("500nm"), Some(0.5));
        assert_eq!(parse_length_um("1.5µm"), Some(1.5));
        assert_eq!(parse_length_um("1,5 um"), Some(1.5));
        assert_eq!(parse_length_um("abc"), None);
        assert_eq!(parse_length_um("-3"), None);
        assert_eq!(parse_length_um("3 furlongs"), None);
        assert_eq!(unit_um("micron"), Some(1.0));
        assert_eq!(unit_um("pixel"), None);
    }

    fn params(um: f64, bar: f64, text: &str) -> OverlayParams {
        OverlayParams { um_per_px: um, bar_um: bar, text: text.into(), ..Default::default() }
    }

    /// A 200 µm bar at 1 µm/px is 200 px of white on a black image, in the
    /// bottom-right corner, and nothing else changes.
    #[test]
    fn bar_composites() {
        let (w, h) = (1000, 800);
        let ov = Overlay::render(&params(1.0, 200.0, ""), w, h, 1.0);
        assert_eq!(ov.patches.len(), 1);
        assert_eq!(ov.bar_px, 200.0);
        assert_eq!(ov.label, "200 µm");
        let mut img = Img3::zeros(w, h);
        ov.apply_f32(&mut img);
        let p = &ov.patches[0];
        assert!(p.x > w / 2 && p.y > h / 2, "bottom right: patch at ({}, {})", p.x, p.y);
        // the longest run of white in any row is the bar
        let mut best = 0;
        for y in 0..h {
            let mut run = 0;
            for x in 0..w {
                if img.p[1][y * w + x] > 0.99 { run += 1; best = best.max(run); } else { run = 0; }
            }
        }
        assert!((best as i32 - 200).abs() <= 1, "bar of {best} px");
        // outside the patch nothing moved
        let outside: f32 = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).filter(|&(x, y)| x < p.x || y < p.y).map(|(x, y)| img.p[0][y * w + x]).sum();
        assert_eq!(outside, 0.0);
    }

    #[test]
    fn text_and_bar_share_a_corner() {
        let (w, h) = (1200, 900);
        let mut p = params(0.5, 0.0, "Specimen A\n40×");
        p.text_pos = Corner::BottomRight;
        let ov = Overlay::render(&p, w, h, 1.0);
        assert_eq!(ov.patches.len(), 2);
        assert!(ov.bar_um > 0.0 && ov.bar_px > 0.0);
        // the automatic bar is near a fifth of the width: 120 µm → 100 µm = 200 px
        assert_eq!(ov.label, "100 µm");
        assert_eq!(ov.bar_px, 200.0);
        // the text block sits above the bar block, both against the right margin
        let (bar, text) = (&ov.patches[0], &ov.patches[1]);
        assert!(text.y + text.h <= bar.y + 2, "text above the bar: text ends {}, bar starts {}", text.y + text.h, bar.y);
        assert!((bar.x + bar.w) as i32 - (text.x + text.w) as i32 <= 1);
        let mut img = Img3::zeros(w, h);
        ov.apply_f32(&mut img);
        assert!(img.p[0].iter().any(|&v| v > 0.99));
    }

    /// The same overlay at half the output size: the bar halves in pixels,
    /// the label stays.
    #[test]
    fn scales_with_the_output() {
        let a = Overlay::render(&params(0.325, 0.0, "x"), 4000, 3000, 1.0);
        let b = Overlay::render(&params(0.325, 0.0, "x"), 2000, 1500, 0.5);
        assert_eq!(a.label, b.label);
        assert!((a.bar_px / 2.0 - b.bar_px).abs() <= 1.0, "{} vs {}", a.bar_px, b.bar_px);
        assert!((a.patches[0].w as f32 / 2.0 - b.patches[0].w as f32).abs() <= 3.0);
    }

    /// The 16-bit path and the float path agree.
    #[test]
    fn u16_matches_f32() {
        let (w, h) = (640, 480);
        let mut p = params(2.0, 0.0, "a caption");
        p.color = Ink::Black;
        p.style = Style::Box;
        let ov = Overlay::render(&p, w, h, 1.0);
        let mut img = Img3 { w, h, p: [vec![0.3; w * h], vec![0.5; w * h], vec![0.7; w * h]] };
        let mut rgb: Vec<u16> = (0..w * h).flat_map(|_| [(0.3 * 65535.0) as u16, (0.5 * 65535.0) as u16, (0.7 * 65535.0) as u16]).collect();
        let under = ov.under_f32(&img);
        ov.apply_f32(&mut img);
        ov.apply_u16(&mut rgb, w, 0, 0);
        let mut maxd = 0.0f32;
        for i in 0..w * h {
            for c in 0..3 {
                maxd = maxd.max((img.p[c][i] - rgb[3 * i + c] as f32 / 65535.0).abs());
            }
        }
        assert!(maxd < 1.5 / 65535.0 + 1e-5, "max difference {maxd}");
        // the box darkened something behind the text, the ink is black
        assert!(img.p[0].iter().any(|&v| v < 0.01));
        assert!(img.p[0].iter().any(|&v| v > 0.3 + 0.1 && v < 1.0), "the white box shows through at 60 %");
        // restore brings the clean image back
        ov.restore_f32(&mut img, &under);
        assert!(img.p[0].iter().all(|&v| (v - 0.3).abs() < 1e-6));
        // the RGBA8 patches carry the same alpha where the ink is
        let rgba = ov.rgba8();
        assert_eq!(rgba.len(), ov.patches.len());
        assert!(rgba[0].4.chunks(4).any(|px| px[3] == 255 && px[0] == 0), "solid black ink somewhere");
    }

    #[test]
    fn stereo_halves_get_their_own_bar() {
        let (w, h) = (400, 300);
        let ov = Overlay::render(&params(1.0, 50.0, ""), w, h, 1.0);
        let mut pair = vec![0u16; 2 * w * h * 3];
        ov.apply_u16(&mut pair, 2 * w, 0, 0);
        ov.apply_u16(&mut pair, 2 * w, w, 0);
        let white = |x0: usize| (0..h).flat_map(|y| (x0..x0 + w).map(move |x| (x, y))).filter(|&(x, y)| pair[(y * 2 * w + x) * 3] > 65000).count();
        assert!(white(0) >= 50 && white(0) == white(w), "{} / {}", white(0), white(w));
    }

    /// A bar asked for without a calibration is a round count of frame pixels,
    /// labelled so; at half the output size it measures the same frame pixels.
    #[test]
    fn uncalibrated_bar_in_pixels() {
        let p = OverlayParams { bar: true, ..Default::default() };
        assert!(p.has_bar() && !p.calibrated() && !p.is_empty());
        let ov = Overlay::render(&p, 1000, 800, 1.0);
        assert_eq!(ov.patches.len(), 1);
        assert_eq!(ov.label, "200 px");
        assert_eq!(ov.bar_px, 200.0);
        assert!(ov.describe().starts_with("scale bar 200 px (no calibration), bottom right"), "{}", ov.describe());
        let half = Overlay::render(&p, 500, 400, 0.5);
        assert_eq!(half.label, "200 px");
        assert_eq!(half.bar_px, 100.0);
        assert!(half.describe().contains("200 px of the frames = 100 px"), "{}", half.describe());
        // a length asked for is pixels too, and one that does not fit is brought down
        let asked = Overlay::render(&OverlayParams { bar: true, bar_um: 250.0, ..Default::default() }, 1000, 800, 1.0);
        assert_eq!(asked.label, "250 px");
        let big = Overlay::render(&OverlayParams { bar: true, bar_um: 5000.0, ..Default::default() }, 1000, 800, 1.0);
        assert!(big.bar_px < 1000.0 && big.describe().contains("the 5000 px asked for does not fit"), "{}", big.describe());
        assert_eq!(parse_length_px("500"), Some(500.0));
        assert_eq!(parse_length_px("500 px"), Some(500.0));
        assert_eq!(parse_length_px("2 mm"), None);
        assert_eq!(parse_length_px("0"), None);
    }

    #[test]
    fn empty_and_edge_cases() {
        assert!(Overlay::render(&OverlayParams::default(), 100, 100, 1.0).is_empty());
        assert!(Overlay::render(&params(1.0, 0.0, ""), 0, 0, 1.0).is_empty());
        // a bar longer than the image is brought down to what fits
        let ov = Overlay::render(&params(1.0, 5000.0, ""), 1000, 800, 1.0);
        assert!(ov.bar_px < 1000.0 && ov.bar_um < 5000.0, "{}", ov.describe());
        assert!(ov.describe().contains("does not fit"));
        // a tiny image still renders without panicking
        let ov = Overlay::render(&params(10.0, 0.0, "tiny caption that is far wider than the image"), 40, 30, 1.0);
        let mut img = Img3::zeros(40, 30);
        ov.apply_f32(&mut img);
    }

    #[test]
    fn captions_expand() {
        // 2026-09-11 17:24:25 UTC
        let t = 1_789_147_465.0;
        assert_eq!(expand_text("{first}: {frames} frames, {date} {time}, stack {n}\\nline 2", Some(t), 25, "img_0001", 3), "img_0001: 25 frames, 2026-09-11 17:24:25, stack 3\nline 2");
        assert_eq!(expand_text("{date}", None, 1, "", 1), "");
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19723), (2024, 1, 1));
    }

    #[test]
    fn parses_names() {
        assert_eq!(Corner::parse("br"), Some(Corner::BottomRight));
        assert_eq!(Corner::parse("Top-Left"), Some(Corner::TopLeft));
        assert_eq!(Corner::parse("middle"), None);
        assert_eq!(Ink::parse("black"), Some(Ink::Black));
        assert_eq!(Style::parse("outline"), Some(Style::Halo));
    }
}

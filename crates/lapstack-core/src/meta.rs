// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY
//
// Metadata pass-through: the EXIF, ICC profile and XMP of the first frame,
// read out of its JPEG / PNG / TIFF and written into the stacked image's file,
// so the result keeps the camera, lens, exposure and colour information of the
// stack it came from (Helicon Focus and Zerene Stacker do the same).
//
// EXIF is a TIFF structure (byte-order mark, IFD0, and the Exif / GPS /
// Interoperability sub-IFDs). It is never copied verbatim: `rebuild` reads the
// source's entries and serialises a new structure with only the metadata tags
// — the image-structure tags (size, strips, compression …) describe the source
// file and not the result, the MakerNote holds absolute offsets that break when
// moved, and the IFD1 thumbnail is the source's. The Software tag is set to
// lapstack. The same rebuild serves a TIFF input (the whole file is the
// structure), a JPEG APP1 and a PNG eXIf chunk.
//
// Output: PNG gets iCCP (or cHRM from a TIFF's white point and primaries when
// there is no profile), eXIf and an XMP iTXt; JPEG gets APP1 Exif, APP1 XMP and
// APP2 ICC_PROFILE segments; TIFF (the CLI's writer in io.rs) gets the tags in
// IFD0 with the sub-IFDs appended.

use std::io::{Read, Write};

/// Metadata of one frame: what is carried into the stacked image.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Meta {
    /// A rebuilt TIFF structure (see the module comment), as in a PNG eXIf chunk.
    pub exif: Option<Vec<u8>>,
    pub icc: Option<Vec<u8>>,
    /// The XMP packet (UTF-8 XML).
    pub xmp: Option<Vec<u8>>,
    /// CIE xy of the white point and the R, G, B primaries (TIFF tags 318/319, PNG cHRM).
    pub chrm: Option<[f64; 8]>,
}

impl Meta {
    pub fn is_empty(&self) -> bool {
        self.exif.is_none() && self.icc.is_none() && self.xmp.is_none() && self.chrm.is_none()
    }
    /// One line for a log: "EXIF 1.2 KB, ICC 3.1 KB, XMP 32 KB" or "none".
    pub fn describe(&self) -> String {
        let kb = |v: &Option<Vec<u8>>, name: &str| v.as_ref().map(|b| format!("{name} {}", human(b.len())));
        let mut parts: Vec<String> = [kb(&self.exif, "EXIF"), kb(&self.icc, "ICC"), kb(&self.xmp, "XMP")].into_iter().flatten().collect();
        if self.chrm.is_some() && self.icc.is_none() {
            parts.push("white point + primaries".into());
        }
        if parts.is_empty() { "none".into() } else { parts.join(", ") }
    }
}

fn human(n: usize) -> String {
    if n < 1024 { format!("{n} B") } else if n < 1024 * 1024 { format!("{:.1} KB", n as f64 / 1024.0) } else { format!("{:.1} MB", n as f64 / (1024.0 * 1024.0)) }
}

const SOFTWARE: &str = concat!("lapstack ", env!("CARGO_PKG_VERSION"));

// ---------------------------------------------------------------------------
// reading

/// Read the metadata of a JPEG, PNG or TIFF file. Anything malformed is
/// simply absent: this must never fail a run.
pub fn extract(bytes: &[u8]) -> Meta {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        extract_jpeg(bytes)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        extract_png(bytes)
    } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        extract_tiff(bytes)
    } else {
        Meta::default()
    }
}

fn be16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn be32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const ICC_HDR: &[u8] = b"ICC_PROFILE\0";

fn extract_jpeg(b: &[u8]) -> Meta {
    let mut m = Meta::default();
    let mut icc: Vec<(u8, &[u8])> = Vec::new();
    let mut p = 2;
    while p + 4 <= b.len() && b[p] == 0xFF {
        let marker = b[p + 1];
        if marker == 0xFF { p += 1; continue; }                                   // fill byte
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) { p += 2; continue; }
        if marker == 0xDA || marker == 0xD9 { break; }                            // scan / end: no more headers
        let len = be16(b, p + 2).unwrap_or(0) as usize;
        if len < 2 || p + 2 + len > b.len() { break; }
        let seg = &b[p + 4..p + 2 + len];
        match marker {
            0xE1 if seg.starts_with(b"Exif\0\0") => m.exif = m.exif.or_else(|| rebuild(&seg[6..])),
            0xE1 if seg.starts_with(XMP_NS) => m.xmp = m.xmp.or_else(|| Some(seg[XMP_NS.len()..].to_vec())),
            0xE2 if seg.starts_with(ICC_HDR) && seg.len() > 14 => icc.push((seg[12], &seg[14..])),
            _ => {}
        }
        p += 2 + len;
    }
    if !icc.is_empty() {
        icc.sort_by_key(|(seq, _)| *seq);
        m.icc = Some(icc.into_iter().flat_map(|(_, d)| d.iter().copied()).collect());
    }
    m
}

fn inflate(z: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(z).read_to_end(&mut out).ok()?;
    Some(out)
}

fn extract_png(b: &[u8]) -> Meta {
    let mut m = Meta::default();
    let mut p = 8;
    while p + 12 <= b.len() {
        let len = be32(b, p).unwrap_or(0) as usize;
        let typ = &b[p + 4..p + 8];
        if p + 12 + len > b.len() { break; }
        let data = &b[p + 8..p + 8 + len];
        match typ {
            b"eXIf" => m.exif = m.exif.or_else(|| rebuild(data)),
            b"iCCP" => {
                if let Some(nul) = data.iter().position(|&c| c == 0) {
                    if data.get(nul + 1) == Some(&0) {
                        m.icc = m.icc.or_else(|| inflate(&data[nul + 2..]));
                    }
                }
            }
            b"iTXt" => {
                if let Some(nul) = data.iter().position(|&c| c == 0) {
                    if &data[..nul] == b"XML:com.adobe.xmp" && data.len() > nul + 3 {
                        let (flag, method) = (data[nul + 1], data[nul + 2]);
                        // language tag and translated keyword, both NUL-terminated
                        let mut q = nul + 3;
                        let mut ok = true;
                        for _ in 0..2 {
                            match data[q..].iter().position(|&c| c == 0) { Some(k) => q += k + 1, None => { ok = false; break; } }
                        }
                        if ok {
                            let text = &data[q..];
                            m.xmp = m.xmp.or_else(|| if flag == 0 { Some(text.to_vec()) } else if method == 0 { inflate(text) } else { None });
                        }
                    }
                }
            }
            // ImageMagick keeps XMP as a "Raw profile" text chunk: a name line, a length line, hex bytes
            b"zTXt" | b"tEXt" if data.starts_with(b"Raw profile type xmp\0") && m.xmp.is_none() => {
                let rest = &data[b"Raw profile type xmp\0".len()..];
                let text = if typ == b"zTXt" { rest.get(1..).and_then(inflate) } else { Some(rest.to_vec()) };
                m.xmp = text.and_then(|t| raw_profile(&t));
            }
            b"cHRM" if len == 32 => {
                let mut c = [0f64; 8];
                for (i, v) in c.iter_mut().enumerate() {
                    *v = be32(data, 4 * i).unwrap_or(0) as f64 / 100000.0;
                }
                m.chrm = Some(c);
            }
            b"IEND" => break,
            _ => {}
        }
        p += 12 + len;
    }
    m
}

/// ImageMagick's "Raw profile" text: `\n<name>\n<decimal length>\n<hex, 72 chars a line>`.
fn raw_profile(t: &[u8]) -> Option<Vec<u8>> {
    let mut lines = t.split(|&c| c == b'\n').filter(|l| !l.is_empty());
    let _name = lines.next()?;
    let len: usize = std::str::from_utf8(lines.next()?).ok()?.trim().parse().ok()?;
    let mut out = Vec::with_capacity(len);
    let mut hi: Option<u8> = None;
    for &c in lines.flatten() {
        let v = (c as char).to_digit(16)? as u8;
        match hi { None => hi = Some(v), Some(h) => { out.push(h << 4 | v); hi = None; } }
    }
    (out.len() == len).then_some(out)
}

fn extract_tiff(b: &[u8]) -> Meta {
    let mut m = Meta::default();
    let t = Tiff::new(b);
    let Some(t) = t else { return m };
    let Some((entries, _)) = t.read_ifd(t.u32(4).unwrap_or(0) as usize) else { return m };
    for e in &entries {
        match e.tag {
            700 if matches!(e.typ, 1 | 7) => m.xmp = Some(e.data.clone()),
            34675 => m.icc = Some(e.data.clone()),
            _ => {}
        }
    }
    let rat = |e: &Entry, i: usize| -> Option<f64> {
        let n = t.val32(&e.data, 8 * i)? as f64;
        let d = t.val32(&e.data, 8 * i + 4)? as f64;
        (d != 0.0).then_some(n / d)
    };
    let wp = entries.iter().find(|e| e.tag == 318 && e.typ == 5 && e.count == 2);
    let pr = entries.iter().find(|e| e.tag == 319 && e.typ == 5 && e.count == 6);
    if let (Some(wp), Some(pr)) = (wp, pr) {
        let v: Option<Vec<f64>> = (0..2).map(|i| rat(wp, i)).chain((0..6).map(|i| rat(pr, i))).collect();
        if let Some(v) = v {
            m.chrm = Some(v.try_into().unwrap());
        }
    }
    m.exif = rebuild(b);
    m
}

// ---------------------------------------------------------------------------
// TIFF structures

struct Tiff<'a> {
    b: &'a [u8],
    le: bool,
}

/// One IFD entry with its value bytes (in the structure's own byte order).
#[derive(Clone, Debug)]
struct Entry {
    tag: u16,
    typ: u16,
    count: u32,
    data: Vec<u8>,
}

/// Bytes per element of a TIFF type (classic TIFF types only).
fn type_size(typ: u16) -> Option<usize> {
    Some(match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => return None,
    })
}

/// Largest value copied through (a runaway count in a corrupt file must not allocate the world).
const MAX_VALUE: usize = 64 << 20;

impl<'a> Tiff<'a> {
    fn new(b: &'a [u8]) -> Option<Self> {
        let le = match b.get(0..4)? {
            b"II*\0" => true,
            b"MM\0*" => false,
            _ => return None,
        };
        Some(Tiff { b, le })
    }
    fn u16(&self, o: usize) -> Option<u16> {
        let v = self.b.get(o..o + 2)?.try_into().ok()?;
        Some(if self.le { u16::from_le_bytes(v) } else { u16::from_be_bytes(v) })
    }
    fn u32(&self, o: usize) -> Option<u32> {
        let v = self.b.get(o..o + 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) })
    }
    /// A u32 out of a value buffer, in this structure's byte order.
    fn val32(&self, d: &[u8], o: usize) -> Option<u32> {
        let v = d.get(o..o + 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) })
    }
    /// The entries of the IFD at `off` and the offset of the next IFD. Entries
    /// with an unknown type or a value outside the buffer are skipped.
    fn read_ifd(&self, off: usize) -> Option<(Vec<Entry>, u32)> {
        let n = self.u16(off)? as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let e = off + 2 + 12 * i;
            let (Some(tag), Some(typ), Some(count)) = (self.u16(e), self.u16(e + 2), self.u32(e + 4)) else { break };
            let Some(size) = type_size(typ) else { continue };
            let total = size.checked_mul(count as usize).filter(|&t| t <= MAX_VALUE);
            let Some(total) = total else { continue };
            let data = if total <= 4 {
                self.b.get(e + 8..e + 8 + total)
            } else {
                let o = self.u32(e + 8)? as usize;
                self.b.get(o..o.checked_add(total)?)
            };
            let Some(data) = data else { continue };
            out.push(Entry { tag, typ, count, data: data.to_vec() });
        }
        let next = self.u32(off + 2 + 12 * n).unwrap_or(0);
        Some((out, next))
    }
}

/// An IFD to serialise: its entries, and the sub-IFDs hanging off some of them.
struct Node {
    entries: Vec<Entry>,
    subs: Vec<(u16, Node)>,
}

/// Tags whose sub-IFD is followed: Exif, GPS, Interoperability.
const SUB_IFD_TAGS: [u16; 3] = [34665, 34853, 40965];

/// IFD0 tags that describe the source file's pixels, not the result's, and the
/// blobs carried separately (XMP, IPTC, Photoshop, ICC) or not at all.
fn drop_ifd0(tag: u16) -> bool {
    matches!(tag,
        254..=259 | 262..=266 | 273 | 277..=281 | 284..=293 | 297 | 301 | 317 | 320..=347 | 351
        | 400..=403 | 433..=437 | 530..=532 | 700 | 33723 | 34377 | 34675 | 37724 | 50341 | 50971
        | 50706..=50740 | 50778..=50781 | 50827..=50839 | 50936..=51125)   // DNG tags: the source's raw data
}
/// Exif IFD tags dropped: the MakerNote (absolute offsets, breaks when moved).
fn drop_exif(tag: u16) -> bool {
    tag == 37500
}

fn build(t: &Tiff, entries: Vec<Entry>, depth: usize) -> Node {
    let mut subs = Vec::new();
    let mut kept = Vec::with_capacity(entries.len());
    for e in entries {
        let drop = if depth == 0 { drop_ifd0(e.tag) } else { drop_exif(e.tag) };
        if drop {
            continue;
        }
        if SUB_IFD_TAGS.contains(&e.tag) && matches!(e.typ, 4 | 13) && e.count == 1 && depth < 3 {
            if let Some(off) = t.val32(&e.data, 0) {
                if let Some((sub, _)) = t.read_ifd(off as usize) {
                    subs.push((e.tag, build(t, sub, depth + 1)));
                    kept.push(Entry { tag: e.tag, typ: 4, count: 1, data: vec![0; 4] });   // the offset is set when written
                }
            }
            continue;
        }
        kept.push(e);
    }
    Node { entries: kept, subs }
}

/// Byte-swap the elements of a value between byte orders (no-op for 1-byte types).
fn swapped(data: &[u8], typ: u16) -> Vec<u8> {
    let mut d = data.to_vec();
    let s = match typ { 3 | 8 => 2, 4 | 9 | 11 | 13 | 5 | 10 => 4, 12 => 8, _ => return d };   // rationals swap as two u32
    for c in d.chunks_exact_mut(s) {
        c.reverse();
    }
    d
}
/// Swap every value of a tree to the other byte order (sub-IFD offsets are
/// written by `write_ifd` in the target order and are left alone).
fn swap_node(n: &mut Node) {
    for e in n.entries.iter_mut() {
        if !SUB_IFD_TAGS.contains(&e.tag) { e.data = swapped(&e.data, e.typ); }
    }
    for (_, s) in n.subs.iter_mut() { swap_node(s); }
}

struct Writer {
    out: Vec<u8>,
    le: bool,
    /// Offsets written are `base + position in out`: 0 for a standalone
    /// structure, the file offset the bytes will land at for a TIFF's tail.
    base: u32,
}

impl Writer {
    fn w16(&mut self, v: u16) { self.out.extend_from_slice(&if self.le { v.to_le_bytes() } else { v.to_be_bytes() }); }
    fn w32(&mut self, v: u32) { self.out.extend_from_slice(&if self.le { v.to_le_bytes() } else { v.to_be_bytes() }); }
    fn align(&mut self) { if (self.base as usize + self.out.len()) % 2 == 1 { self.out.push(0); } }
    /// Append the IFD (sub-IFDs first, then the table, then its values); returns its
    /// offset. Values are written as they are: the tree must be in this writer's byte order.
    fn write_ifd(&mut self, node: Node) -> u32 {
        let Node { mut entries, subs } = node;
        for (tag, sub) in subs {
            let off = self.write_ifd(sub);
            if let Some(e) = entries.iter_mut().find(|e| e.tag == tag) {
                e.data = if self.le { off.to_le_bytes().to_vec() } else { off.to_be_bytes().to_vec() };
            }
        }
        entries.sort_by_key(|e| e.tag);
        entries.dedup_by_key(|e| e.tag);
        self.align();
        let ifd_off = self.base + self.out.len() as u32;
        let n = entries.len();
        self.w16(n as u16);
        let data_start = ifd_off + 2 + 12 * n as u32 + 4;
        let mut extra: Vec<u8> = Vec::new();
        for e in &entries {
            let data = &e.data;
            self.w16(e.tag);
            self.w16(e.typ);
            self.w32(e.count);
            if data.len() <= 4 {
                let mut v = [0u8; 4];
                v[..data.len()].copy_from_slice(data);
                self.out.extend_from_slice(&v);
            } else {
                self.w32(data_start + extra.len() as u32);
                extra.extend_from_slice(data);
                if extra.len() % 2 == 1 { extra.push(0); }
            }
        }
        self.w32(0);
        self.out.extend_from_slice(&extra);
        ifd_off
    }
}

fn software_entry() -> Entry {
    let mut s = SOFTWARE.as_bytes().to_vec();
    s.push(0);
    Entry { tag: 305, typ: 2, count: s.len() as u32, data: s }
}

/// Parse a TIFF structure's IFD0 (+ Exif / GPS / Interop sub-IFDs) into a tree
/// with only the metadata tags, Software set to lapstack.
fn parse(tiff: &[u8]) -> Option<(Node, bool)> {
    let t = Tiff::new(tiff)?;
    let (entries, _) = t.read_ifd(t.u32(4)? as usize)?;
    let mut node = build(&t, entries, 0);
    node.entries.retain(|e| e.tag != 305);
    node.entries.push(software_entry());
    Some((node, t.le))
}

fn serialise(node: Node, le: bool) -> Vec<u8> {
    let mut w = Writer { out: Vec::new(), le, base: 0 };
    w.out.extend_from_slice(if le { b"II*\0" } else { b"MM\0*" });
    w.w32(0);
    let off = w.write_ifd(node);
    let ob = if le { off.to_le_bytes() } else { off.to_be_bytes() };
    w.out[4..8].copy_from_slice(&ob);
    w.out
}

/// A fresh, standalone EXIF structure from a TIFF structure (a TIFF file, a JPEG
/// APP1 payload, a PNG eXIf chunk), in the source's byte order.
pub fn rebuild(tiff: &[u8]) -> Option<Vec<u8>> {
    let (node, le) = parse(tiff)?;
    Some(serialise(node, le))
}

/// The file the EXIF goes into: the Exif spec asks for slightly different tags
/// in a JPEG (the pixel dimensions, YCbCr positioning) and a TIFF (no
/// dimensions, no ComponentsConfiguration).
#[derive(Clone, Copy, PartialEq)]
enum Container { Jpeg, Png, Tiff }

/// Set the Exif IFD's PixelX/YDimension to the result's size (the tags are
/// mandatory in a JPEG, and a source's would be stale after a crop or scale),
/// with the container-specific tags the validators look for; returns the tree.
fn adapt(tiff: &[u8], w: u32, h: u32, container: Container) -> Option<(Node, bool)> {
    let (mut node, le) = parse(tiff)?;
    let bytes32 = |v: u32| if le { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let bytes16 = |v: u16| if le { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let set = |entries: &mut Vec<Entry>, e: Entry| { entries.retain(|x| x.tag != e.tag); entries.push(e); };
    if node.subs.iter().all(|(t, _)| *t != 34665) {
        node.subs.push((34665, Node { entries: vec![Entry { tag: 36864, typ: 7, count: 4, data: b"0232".to_vec() }], subs: vec![] }));
        node.entries.push(Entry { tag: 34665, typ: 4, count: 1, data: vec![0; 4] });
    }
    let exif = &mut node.subs.iter_mut().find(|(t, _)| *t == 34665)?.1;
    match container {
        Container::Tiff => {
            exif.entries.retain(|e| !matches!(e.tag, 37121 | 40962 | 40963));
            if exif.entries.iter().all(|e| e.tag != 40960) {
                exif.entries.push(Entry { tag: 40960, typ: 7, count: 4, data: b"0100".to_vec() });
            }
        }
        Container::Jpeg | Container::Png => {
            set(&mut exif.entries, Entry { tag: 40962, typ: 4, count: 1, data: bytes32(w) });
            set(&mut exif.entries, Entry { tag: 40963, typ: 4, count: 1, data: bytes32(h) });
            if container == Container::Jpeg && node.entries.iter().all(|e| e.tag != 531) {
                node.entries.push(Entry { tag: 531, typ: 3, count: 1, data: bytes16(1) });
            }
        }
    }
    Some((node, le))
}

// ---------------------------------------------------------------------------
// writing: PNG and JPEG

/// Insert the metadata into an encoded PNG or JPEG. Unknown bytes come back untouched.
pub fn embed(file: Vec<u8>, meta: &Meta) -> Vec<u8> {
    if meta.is_empty() {
        return file;
    }
    if file.starts_with(b"\x89PNG\r\n\x1a\n") {
        embed_png(&file, meta)
    } else if file.starts_with(&[0xFF, 0xD8]) {
        embed_jpeg(&file, meta)
    } else {
        file
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

fn png_chunk(out: &mut Vec<u8>, typ: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(typ);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn deflate(data: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = e.write_all(data);
    e.finish().unwrap_or_default()
}

fn embed_png(b: &[u8], meta: &Meta) -> Vec<u8> {
    let (w, h) = (be32(b, 16).unwrap_or(0), be32(b, 20).unwrap_or(0));   // IHDR
    // the chunks we add, in the order the spec wants them: colour space, then the rest
    let mut ours: Vec<u8> = Vec::new();
    let mut replaced: Vec<&[u8]> = Vec::new();
    if let Some(icc) = &meta.icc {
        let mut d = b"ICC Profile\0\0".to_vec();
        d.extend_from_slice(&deflate(icc));
        png_chunk(&mut ours, b"iCCP", &d);
        replaced.extend([b"iCCP".as_slice(), b"sRGB".as_slice(), b"gAMA".as_slice(), b"cHRM".as_slice()]);   // iCCP stands alone
    } else if let Some(c) = &meta.chrm {
        let mut d = Vec::with_capacity(32);
        for v in c {
            d.extend_from_slice(&((v * 100000.0).round().clamp(0.0, u32::MAX as f64) as u32).to_be_bytes());
        }
        png_chunk(&mut ours, b"cHRM", &d);
        replaced.push(b"cHRM".as_slice());
    }
    if let Some(exif) = meta.exif.as_ref().and_then(|e| adapt(e, w, h, Container::Png)).map(|(n, le)| serialise(n, le)) {
        png_chunk(&mut ours, b"eXIf", &exif);
        replaced.push(b"eXIf".as_slice());
    }
    if let Some(xmp) = &meta.xmp {
        let mut d = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();   // keyword, uncompressed, method, language, translated keyword
        d.extend_from_slice(xmp);
        png_chunk(&mut ours, b"iTXt", &d);
    }
    let mut out = Vec::with_capacity(b.len() + ours.len());
    out.extend_from_slice(&b[..8]);
    let mut p = 8;
    let mut inserted = false;
    while p + 12 <= b.len() {
        let len = be32(b, p).unwrap_or(0) as usize;
        if p + 12 + len > b.len() { break; }
        let typ = &b[p + 4..p + 8];
        let chunk = &b[p..p + 12 + len];
        // ours go right after IHDR; an existing chunk of a type we wrote, or an XMP iTXt, is dropped
        let is_xmp = typ == b"iTXt" && chunk[8..].starts_with(b"XML:com.adobe.xmp\0") && meta.xmp.is_some();
        if !(replaced.contains(&typ) || is_xmp) {
            out.extend_from_slice(chunk);
        }
        if typ == b"IHDR" && !inserted {
            out.extend_from_slice(&ours);
            inserted = true;
        }
        p += 12 + len;
    }
    out.extend_from_slice(&b[p..]);
    out
}

fn jpeg_segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// Largest payload a JPEG segment holds (2-byte length included in 65535).
const SEG_MAX: usize = 65533;

/// Height and width from the first SOF segment.
fn jpeg_dims(b: &[u8]) -> (u32, u32) {
    let mut p = 2;
    while p + 4 <= b.len() && b[p] == 0xFF {
        let marker = b[p + 1];
        if marker == 0xFF { p += 1; continue; }
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) { p += 2; continue; }
        if marker == 0xDA || marker == 0xD9 { break; }
        let len = be16(b, p + 2).unwrap_or(0) as usize;
        if matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF) {
            return (be16(b, p + 7).unwrap_or(0) as u32, be16(b, p + 5).unwrap_or(0) as u32);
        }
        if len < 2 { break; }
        p += 2 + len;
    }
    (0, 0)
}

fn embed_jpeg(b: &[u8], meta: &Meta) -> Vec<u8> {
    let (w, h) = jpeg_dims(b);
    let mut ours: Vec<u8> = Vec::new();
    if let Some(exif) = meta.exif.as_ref().and_then(|e| adapt(e, w, h, Container::Jpeg)).map(|(n, le)| serialise(n, le)) {
        if exif.len() + 6 <= SEG_MAX {
            let mut d = b"Exif\0\0".to_vec();
            d.extend_from_slice(&exif);
            jpeg_segment(&mut ours, 0xE1, &d);
        }
    }
    if let Some(xmp) = &meta.xmp {
        if xmp.len() + XMP_NS.len() <= SEG_MAX {   // larger packets need ExtendedXMP; skipped
            let mut d = XMP_NS.to_vec();
            d.extend_from_slice(xmp);
            jpeg_segment(&mut ours, 0xE1, &d);
        }
    }
    if let Some(icc) = &meta.icc {
        let per = SEG_MAX - 14;
        let n = icc.len().div_ceil(per).max(1);
        if n <= 255 {
            for (i, part) in icc.chunks(per).enumerate() {
                let mut d = ICC_HDR.to_vec();
                d.push(i as u8 + 1);
                d.push(n as u8);
                d.extend_from_slice(part);
                jpeg_segment(&mut ours, 0xE2, &d);
            }
        }
    }
    // after SOI and any leading APP0 (JFIF) segments; existing Exif / XMP / ICC segments are dropped
    let mut out = Vec::with_capacity(b.len() + ours.len());
    out.extend_from_slice(&b[..2]);
    let mut p = 2;
    let mut inserted = false;
    while p + 4 <= b.len() && b[p] == 0xFF {
        let marker = b[p + 1];
        if marker == 0xDA || marker == 0xD9 { break; }
        if marker == 0xFF { out.push(0xFF); p += 1; continue; }
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) { out.extend_from_slice(&b[p..p + 2]); p += 2; continue; }
        let len = be16(b, p + 2).unwrap_or(0) as usize;
        if len < 2 || p + 2 + len > b.len() { break; }
        let seg = &b[p + 4..p + 2 + len];
        if marker != 0xE0 && !inserted {
            out.extend_from_slice(&ours);
            inserted = true;
        }
        let ours_kind = (marker == 0xE1 && (seg.starts_with(b"Exif\0\0") && meta.exif.is_some() || seg.starts_with(XMP_NS) && meta.xmp.is_some()))
            || (marker == 0xE2 && seg.starts_with(ICC_HDR) && meta.icc.is_some());
        if !ours_kind {
            out.extend_from_slice(&b[p..p + 2 + len]);
        }
        p += 2 + len;
    }
    if !inserted {
        out.extend_from_slice(&ours);
    }
    out.extend_from_slice(&b[p..]);
    out
}

// ---------------------------------------------------------------------------
// writing: TIFF

/// An uncompressed RGB TIFF (8 or 16 bits per sample, chunky, little-endian)
/// with the metadata in IFD0: the EXIF tags and sub-IFDs, the ICC profile
/// (34675), the XMP (700), the white point and primaries (318/319).
/// `samples` holds w*h*3 values of `bits` bits, as bytes in host order for 16.
pub fn write_tiff<W: Write>(mut w: W, width: usize, height: usize, bits: u16, samples: &[u8], meta: Option<&Meta>) -> std::io::Result<()> {
    let bps = bits as usize / 8;
    let row = width * 3 * bps;
    assert_eq!(samples.len(), row * height, "sample buffer size");
    // strips of about 8 MB, like most writers
    let rows_per_strip = (8 << 20) / row.max(1);
    let rows_per_strip = rows_per_strip.clamp(1, height.max(1));
    let n_strips = height.div_ceil(rows_per_strip).max(1);
    let le = |v: u32| v.to_le_bytes().to_vec();
    let shorts = |v: &[u16]| v.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>();
    let longs = |v: &[u32]| v.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>();
    let rational = |v: f64| { let d = 100000u32; longs(&[(v * d as f64).round() as u32, d]) };

    // the pixel data comes right after the 8-byte header; the IFD and its values after it
    let data_off = 8u32;
    let offsets: Vec<u32> = (0..n_strips).map(|i| data_off + (i * rows_per_strip * row) as u32).collect();
    let counts: Vec<u32> = (0..n_strips).map(|i| (rows_per_strip.min(height - i * rows_per_strip) * row) as u32).collect();

    let mut entries = vec![
        Entry { tag: 256, typ: 4, count: 1, data: le(width as u32) },
        Entry { tag: 257, typ: 4, count: 1, data: le(height as u32) },
        Entry { tag: 258, typ: 3, count: 3, data: shorts(&[bits; 3]) },
        Entry { tag: 259, typ: 3, count: 1, data: shorts(&[1]) },
        Entry { tag: 262, typ: 3, count: 1, data: shorts(&[2]) },
        Entry { tag: 273, typ: 4, count: n_strips as u32, data: longs(&offsets) },
        Entry { tag: 277, typ: 3, count: 1, data: shorts(&[3]) },
        Entry { tag: 278, typ: 4, count: 1, data: le(rows_per_strip as u32) },
        Entry { tag: 279, typ: 4, count: n_strips as u32, data: longs(&counts) },
        Entry { tag: 284, typ: 3, count: 1, data: shorts(&[1]) },
        software_entry(),
    ];
    let mut subs = Vec::new();
    if let Some(m) = meta {
        if let Some(exif) = &m.exif {
            if let Some((mut node, src_le)) = adapt(exif, width as u32, height as u32, Container::Tiff) {
                if !src_le { swap_node(&mut node); }   // the file is little-endian
                let taken: Vec<u16> = entries.iter().map(|e| e.tag).collect();
                entries.extend(node.entries.into_iter().filter(|e| !taken.contains(&e.tag)));
                subs = node.subs;
            }
        }
        if let Some(icc) = &m.icc {
            entries.push(Entry { tag: 34675, typ: 7, count: icc.len() as u32, data: icc.clone() });
        }
        if let Some(xmp) = &m.xmp {
            entries.push(Entry { tag: 700, typ: 1, count: xmp.len() as u32, data: xmp.clone() });
        }
        if let Some(c) = &m.chrm {
            if !entries.iter().any(|e| e.tag == 318) {
                entries.push(Entry { tag: 318, typ: 5, count: 2, data: c[..2].iter().flat_map(|&v| rational(v)).collect() });
                entries.push(Entry { tag: 319, typ: 5, count: 6, data: c[2..].iter().flat_map(|&v| rational(v)).collect() });
            }
        }
    }
    let tail_base = data_off + samples.len() as u32;
    let mut wr = Writer { out: Vec::new(), le: true, base: tail_base };
    let ifd0 = wr.write_ifd(Node { entries, subs });
    w.write_all(b"II*\0")?;
    w.write_all(&ifd0.to_le_bytes())?;
    w.write_all(samples)?;
    w.write_all(&wr.out)?;
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A little TIFF structure by hand: IFD0 {Make, Model, Software, ExifIFD}, Exif IFD
    /// {DateTimeOriginal, MakerNote}, in the given byte order.
    fn sample_exif(le: bool) -> Vec<u8> {
        let mut w = Writer { out: Vec::new(), le, base: 0 };
        w.out.extend_from_slice(if le { b"II*\0" } else { b"MM\0*" });
        w.w32(0);
        let ascii = |tag: u16, s: &str| { let mut d = s.as_bytes().to_vec(); d.push(0); Entry { tag, typ: 2, count: d.len() as u32, data: d } };
        let exif = Node { entries: vec![ascii(36867, "2026:09:11 17:24:25"), Entry { tag: 37500, typ: 7, count: 5, data: b"junk!".to_vec() },
                                        Entry { tag: 33434, typ: 5, count: 1, data: if le { [1u32, 250].iter().flat_map(|v| v.to_le_bytes()).collect() } else { [1u32, 250].iter().flat_map(|v| v.to_be_bytes()).collect() } }], subs: vec![] };
        let ifd0 = Node { entries: vec![ascii(271, "NIKON"), ascii(272, "Z 8"), ascii(305, "NX Studio"), Entry { tag: 34665, typ: 4, count: 1, data: vec![0; 4] },
                                        Entry { tag: 256, typ: 4, count: 1, data: vec![1, 2, 3, 4] }, Entry { tag: 274, typ: 3, count: 1, data: if le { vec![1, 0, 0, 0] } else { vec![0, 1, 0, 0] } }],
                          subs: vec![(34665, exif)] };
        let off = w.write_ifd(ifd0);
        let ob = if le { off.to_le_bytes() } else { off.to_be_bytes() };
        w.out[4..8].copy_from_slice(&ob);
        w.out
    }

    fn ascii_tag(_t: &Tiff, entries: &[Entry], tag: u16) -> Option<String> {
        let e = entries.iter().find(|e| e.tag == tag)?;
        Some(String::from_utf8_lossy(&e.data).trim_end_matches('\0').to_string())
    }

    /// PixelXDimension / PixelYDimension of the Exif IFD.
    fn dims_of(exif: &[u8]) -> (u32, u32) {
        let t = Tiff::new(exif).unwrap();
        let (ifd0, _) = t.read_ifd(t.u32(4).unwrap() as usize).unwrap();
        let sub = ifd0.iter().find(|e| e.tag == 34665).unwrap();
        let (ex, _) = t.read_ifd(t.val32(&sub.data, 0).unwrap() as usize).unwrap();
        let g = |tag| ex.iter().find(|e| e.tag == tag).map(|e| t.val32(&e.data, 0).unwrap()).unwrap_or(0);
        (g(40962), g(40963))
    }

    fn check_rebuilt(exif: &[u8], le: bool) {
        let t = Tiff::new(exif).unwrap();
        assert_eq!(t.le, le);
        let (ifd0, next) = t.read_ifd(t.u32(4).unwrap() as usize).unwrap();
        assert_eq!(next, 0);
        assert_eq!(ascii_tag(&t, &ifd0, 271).as_deref(), Some("NIKON"));
        assert_eq!(ascii_tag(&t, &ifd0, 272).as_deref(), Some("Z 8"));
        assert_eq!(ascii_tag(&t, &ifd0, 305).as_deref(), Some(SOFTWARE));
        assert!(ifd0.iter().all(|e| e.tag != 256), "image width dropped");
        let o = ifd0.iter().find(|e| e.tag == 274).unwrap();
        let v = [o.data[0], o.data[1]];
        assert_eq!(if t.le { u16::from_le_bytes(v) } else { u16::from_be_bytes(v) }, 1, "orientation kept, in {}", if le { "LE" } else { "BE" });
        let tags: Vec<u16> = ifd0.iter().map(|e| e.tag).collect();
        assert!(tags.windows(2).all(|w| w[0] < w[1]), "sorted: {tags:?}");
        let sub = ifd0.iter().find(|e| e.tag == 34665).unwrap();
        let (ex, _) = t.read_ifd(t.val32(&sub.data, 0).unwrap() as usize).unwrap();
        assert_eq!(ascii_tag(&t, &ex, 36867).as_deref(), Some("2026:09:11 17:24:25"));
        assert!(ex.iter().all(|e| e.tag != 37500), "MakerNote dropped");
        let et = ex.iter().find(|e| e.tag == 33434).unwrap();
        assert_eq!((t.val32(&et.data, 0).unwrap(), t.val32(&et.data, 4).unwrap()), (1, 250));
    }

    #[test]
    fn rebuild_keeps_metadata_drops_structure() {
        for le in [true, false] {
            let src = sample_exif(le);
            let out = rebuild(&src).unwrap();
            check_rebuilt(&out, le);
            // idempotent
            assert_eq!(rebuild(&out).unwrap(), out);
        }
    }

    fn sample_meta(le: bool) -> Meta {
        Meta { exif: rebuild(&sample_exif(le)), icc: Some((0..3000u32).map(|i| (i * 7 % 251) as u8).collect()), xmp: Some(b"<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF/></x:xmpmeta>".to_vec()), chrm: None }
    }

    fn tiny_rgb(w: u32, h: u32) -> Vec<u8> {
        (0..w * h * 3).map(|i| (i * 37 % 256) as u8).collect()
    }

    #[test]
    fn jpeg_round_trip() {
        use image::ImageEncoder;
        let (w, h) = (16, 12);
        let mut file = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, 90).write_image(&tiny_rgb(w, h), w, h, image::ExtendedColorType::Rgb8).unwrap();
        let meta = sample_meta(false);
        let out = embed(file.clone(), &meta);
        assert!(out.len() > file.len());
        let back = extract(&out);
        assert_eq!(dims_of(back.exif.as_ref().unwrap()), (w, h), "JPEG carries the pixel dimensions");
        assert_eq!(back.icc, meta.icc);
        assert_eq!(back.xmp, meta.xmp);
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!((img.width(), img.height()), (w, h));
        // embedding again replaces rather than duplicates
        let twice = embed(out.clone(), &meta);
        assert_eq!(twice.len(), out.len());
    }

    #[test]
    fn png_round_trip() {
        use image::ImageEncoder;
        let (w, h) = (16, 12);
        let mut file = Vec::new();
        image::codecs::png::PngEncoder::new(&mut file).write_image(&tiny_rgb(w, h), w, h, image::ExtendedColorType::Rgb8).unwrap();
        let mut meta = sample_meta(true);
        let out = embed(file.clone(), &meta);
        let back = extract(&out);
        check_rebuilt(back.exif.as_ref().unwrap(), true);
        assert_eq!(dims_of(back.exif.as_ref().unwrap()), (w, h));
        assert_eq!(back.icc, meta.icc);
        assert_eq!(back.xmp, meta.xmp);
        assert_eq!(back.chrm, None);
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!(img.to_rgb8().into_raw(), tiny_rgb(w, h));
        assert_eq!(embed(out.clone(), &meta).len(), out.len());
        // no profile: the chromaticities go into cHRM
        meta.icc = None;
        meta.chrm = Some([0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06]);
        let back = extract(&embed(file, &meta));
        assert_eq!(back.icc, None);
        assert_eq!(back.chrm, meta.chrm);
    }

    #[test]
    fn imagemagick_raw_profile_xmp() {
        let xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>";
        let hex: String = xmp.iter().map(|b| format!("{b:02x}")).collect();
        let text = format!("\nxmp\n{:>8}\n{}\n", xmp.len(), hex.as_bytes().chunks(72).map(|c| std::str::from_utf8(c).unwrap()).collect::<Vec<_>>().join("\n"));
        let mut file = b"\x89PNG\r\n\x1a\n".to_vec();
        png_chunk(&mut file, b"IHDR", &[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
        let mut d = b"Raw profile type xmp\0\0".to_vec();
        d.extend_from_slice(&deflate(text.as_bytes()));
        png_chunk(&mut file, b"zTXt", &d);
        png_chunk(&mut file, b"IEND", &[]);
        assert_eq!(extract(&file).xmp.as_deref(), Some(&xmp[..]));
    }

    #[test]
    fn tiff_round_trip() {
        let (w, h) = (16usize, 12usize);
        let px = tiny_rgb(w as u32, h as u32);
        let meta = Meta { chrm: Some([0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06]), ..sample_meta(false) };
        let mut file = Vec::new();
        write_tiff(&mut file, w, h, 8, &px, Some(&meta)).unwrap();
        let img = image::load_from_memory(&file).unwrap();
        assert_eq!(img.to_rgb8().into_raw(), px);
        let back = extract(&file);
        assert_eq!(back.icc, meta.icc);
        assert_eq!(back.xmp, meta.xmp);
        assert_eq!(back.chrm, meta.chrm);
        // the EXIF came in big-endian; the file is little-endian, with the same content
        check_rebuilt(back.exif.as_ref().unwrap(), true);
        // 16-bit, no metadata
        let px16: Vec<u8> = (0..w * h * 3).flat_map(|i| ((i * 977 % 65536) as u16).to_ne_bytes()).collect();
        let mut file = Vec::new();
        write_tiff(&mut file, w, h, 16, &px16, None).unwrap();
        let img = image::load_from_memory(&file).unwrap();
        assert_eq!(img.color(), image::ColorType::Rgb16);
        assert_eq!(img.into_rgb16().into_raw(), (0..w * h * 3).map(|i| (i * 977 % 65536) as u16).collect::<Vec<u16>>());
    }
}

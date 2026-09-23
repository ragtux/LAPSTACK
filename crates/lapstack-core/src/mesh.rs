// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! The stacked image as a 3D model: a textured heightfield from the depth
//! map.
//!
//! The depth map gives every pixel a position along the stack, so the result
//! is a relief — a surface over the image plane with the stacked image as its
//! texture. [`heightfield`] samples that surface on a regular vertex grid
//! (`grid` vertices along the long edge; each vertex takes the mean depth of
//! the cell of pixels around it, so the mesh is smooth at its own scale) and
//! triangulates it, each cell cut along the diagonal with the smaller depth
//! difference so a ridge or an edge is not stepped. A depth discontinuity
//! becomes a steep wall between the near surface and the far one; a
//! heightfield has no way to show what is behind it.
//!
//! Coordinates are right-handed with the image's width as the unit: x runs
//! along the width (−0.5 … 0.5), y up the height (±0.5 · h/w), z out of the
//! image towards the viewer, so the far end of the stack lies on z = 0 and
//! the near end on z = `relief` (the depth of the stack as a fraction of the
//! width, the one number the depth map cannot know). With `near_first` frame
//! 0 is the near end (the focus went front to back), as in [`crate::view`].
//!
//! Writers: [`glb`] (glTF 2.0 binary, one self-contained file with the
//! texture embedded — the format every current viewer and Blender open),
//! [`obj`] + [`mtl`] (Wavefront, the texture as a file beside it, plain
//! text most tools read) and [`stl`] (binary, geometry only, for printing).

use crate::align::Rect;
use crate::view::{self, Sample};
use rayon::prelude::*;
use std::io::Write;

/// How the surface is sampled and scaled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshParams {
    /// Vertices along the long edge of the image (the short edge gets its
    /// share; at most one per pixel).
    pub grid: usize,
    /// The depth of the stack as a fraction of the image width: the near end
    /// stands this far out of the far end's plane.
    pub relief: f32,
    /// Frame 0 is the near end of the stack.
    pub near_first: bool,
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams { grid: 1000, relief: 0.25, near_first: true }
    }
}

/// A triangle mesh over a vertex grid: `nx`×`ny` vertices, row-major from
/// the image's top-left, sharing one index for position, normal and texture
/// coordinate; triangles counter-clockwise seen from +z.
#[derive(Clone, Debug)]
pub struct Mesh {
    pub nx: usize,
    pub ny: usize,
    pub pos: Vec<[f32; 3]>,
    pub normal: Vec<[f32; 3]>,
    /// glTF convention: u along the width, v down the height from the top-left.
    pub uv: Vec<[f32; 2]>,
    pub tri: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn vertices(&self) -> usize {
        self.pos.len()
    }
    pub fn triangles(&self) -> usize {
        self.tri.len()
    }
}

/// The vertex counts along an image's axes for `grid` vertices on the long
/// edge: (nx, ny).
pub fn grid_dims(w: usize, h: usize, grid: usize) -> (usize, usize) {
    let long = w.max(h).max(2);
    let g = grid.clamp(2, long);
    let axis = |n: usize| -> usize { (((n.max(2) - 1) as f64 * (g - 1) as f64 / (long - 1) as f64).round() as usize + 1).clamp(2, n.max(2)) };
    (axis(w), axis(h))
}

/// The pixel intervals of `k` vertices spanning `n` pixels: vertex `i` sits
/// at `i · (n−1)/(k−1)` and owns the half-cell either side of it; the
/// intervals tile `0..n` without overlap.
fn cells(n: usize, k: usize) -> Vec<(usize, usize)> {
    let s = (n - 1) as f64 / (k - 1) as f64;
    (0..k)
        .map(|i| {
            let a = ((i as f64 - 0.5) * s).round().max(0.0) as usize;
            let b = (((i as f64 + 0.5) * s).round() as usize).min(n).max(a + 1);
            (a, b)
        })
        .collect()
}

/// The relief of the window `r` of a `stride`-wide depth map `z` (one sample
/// per pixel, `z · zscale` in [0, 1] = the position along the stack, 0 =
/// frame 0). The window must be at least 2×2.
pub fn heightfield<Z: Sample>(z: &[Z], stride: usize, r: &Rect, zscale: f32, p: &MeshParams) -> Mesh {
    let (w, h) = (r.w, r.h);
    assert!(w >= 2 && h >= 2, "a heightfield needs at least 2x2 pixels");
    assert!(z.len() >= (r.y + h - 1) * stride + r.x + w);
    let (nx, ny) = grid_dims(w, h, p.grid);
    let (bx, by) = (cells(w, nx), cells(h, ny));
    // the depth at each vertex: the mean over its cell, as a fraction of the stack
    let mut d = vec![0f32; nx * ny];
    d.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        let (y0, y1) = by[j];
        let mut acc = vec![0f64; nx];
        for y in y0..y1 {
            let s = &z[(r.y + y) * stride + r.x..(r.y + y) * stride + r.x + w];
            for (a, &(x0, x1)) in acc.iter_mut().zip(&bx) {
                *a += s[x0..x1].iter().map(|v| v.to_f() as f64).sum::<f64>();
            }
        }
        for (i, o) in row.iter_mut().enumerate() {
            let n = ((y1 - y0) * (bx[i].1 - bx[i].0)) as f64;
            *o = ((acc[i] / n) as f32 * zscale).clamp(0.0, 1.0);
        }
    });
    let aspect = (h - 1) as f32 / (w - 1) as f32;
    let (sx, sy) = (1.0 / (nx - 1) as f32, 1.0 / (ny - 1) as f32);
    let pos: Vec<[f32; 3]> = (0..nx * ny)
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k % nx, k / nx);
            let t = if p.near_first { 1.0 - d[k] } else { d[k] };
            [i as f32 * sx - 0.5, aspect * (0.5 - j as f32 * sy), p.relief * t]
        })
        .collect();
    let uv: Vec<[f32; 2]> = (0..nx * ny).map(|k| [(k % nx) as f32 * sx, (k / nx) as f32 * sy]).collect();
    // normals from central differences of the vertex positions (one-sided at the border)
    let normal: Vec<[f32; 3]> = (0..nx * ny)
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k % nx, k / nx);
            let (l, rr) = (pos[k - (i > 0) as usize], pos[k + (i + 1 < nx) as usize]);
            let (u, dn) = (pos[k - if j > 0 { nx } else { 0 }], pos[k + if j + 1 < ny { nx } else { 0 }]);
            let di = [rr[0] - l[0], rr[1] - l[1], rr[2] - l[2]];
            let dj = [dn[0] - u[0], dn[1] - u[1], dn[2] - u[2]];
            // dj × di points out of the image (+z) on a flat surface
            let n = [dj[1] * di[2] - dj[2] * di[1], dj[2] * di[0] - dj[0] * di[2], dj[0] * di[1] - dj[1] * di[0]];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            [n[0] / len, n[1] / len, n[2] / len]
        })
        .collect();
    // two triangles per cell, split along the diagonal with the smaller depth step
    let pos_ref = &pos;
    let tri: Vec<[u32; 3]> = (0..ny - 1)
        .into_par_iter()
        .flat_map_iter(|j| {
            (0..nx - 1).flat_map(move |i| {
                let a = (j * nx + i) as u32;
                let (b, c, dd) = (a + 1, a + nx as u32, a + nx as u32 + 1);
                let (za, zb, zc, zd) = (pos_ref[a as usize][2], pos_ref[b as usize][2], pos_ref[c as usize][2], pos_ref[dd as usize][2]);
                if (za - zd).abs() <= (zb - zc).abs() { [[a, c, dd], [a, dd, b]] } else { [[a, c, b], [b, c, dd]] }
            })
        })
        .collect();
    Mesh { nx, ny, pos, normal, uv, tri }
}

/// The texture: the window `r` of a `stride`-wide interleaved RGB image
/// (`scale` maps a sample to [0, 1]), shrunk so its long edge is at most
/// `edge` px (0 = as it is): (rgb8, w, h).
pub fn texture<T: Sample>(rgb: &[T], stride: usize, r: &Rect, scale: f32, edge: usize) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = texture_dims(r.w, r.h, edge);
    let to8 = |v: f32| (v * scale * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    if ow == r.w && oh == r.h && r.x == 0 && r.w == stride {
        return (rgb[r.y * stride * 3..(r.y + r.h) * stride * 3].iter().map(|v| to8(v.to_f())).collect(), ow, oh);
    }
    let v = view::shrink(rgb, stride, r, 3, ow, oh);
    (v.iter().map(|v| to8(v.to_f())).collect(), ow, oh)
}

/// [`texture`] for a plane-separated image in [0, 1] (the native path's `Img3`).
pub fn texture_planes(planes: [&[f32]; 3], w: usize, h: usize, edge: usize) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = texture_dims(w, h, edge);
    let r = Rect::full(w, h);
    let mut out = vec![0u8; ow * oh * 3];
    for (c, p) in planes.iter().enumerate() {
        let s: std::borrow::Cow<'_, [f32]> = if ow == w && oh == h { std::borrow::Cow::Borrowed(p) } else { std::borrow::Cow::Owned(view::shrink(p, w, &r, 1, ow, oh)) };
        for (o, v) in out.chunks_exact_mut(3).zip(s.iter()) {
            o[c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    (out, ow, oh)
}

/// The texture's size for an image `w`×`h` and a long-edge cap `edge` (0 = none).
pub fn texture_dims(w: usize, h: usize, edge: usize) -> (usize, usize) {
    if edge == 0 || w.max(h) <= edge {
        return (w, h);
    }
    let s = edge as f64 / w.max(h) as f64;
    ((w as f64 * s).round().max(1.0) as usize, (h as f64 * s).round().max(1.0) as usize)
}

/// The texture's file format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TexFormat {
    /// JPEG at this quality (1..100)
    Jpeg(u8),
    Png,
}

impl TexFormat {
    pub fn ext(self) -> &'static str {
        match self {
            TexFormat::Jpeg(_) => "jpg",
            TexFormat::Png => "png",
        }
    }
    pub fn mime(self) -> &'static str {
        match self {
            TexFormat::Jpeg(_) => "image/jpeg",
            TexFormat::Png => "image/png",
        }
    }
}

/// Encode an 8-bit interleaved RGB texture.
pub fn encode_texture(rgb8: &[u8], w: usize, h: usize, f: TexFormat) -> Result<Vec<u8>, String> {
    use image::ImageEncoder;
    let err = |e: image::ImageError| format!("texture: {e}");
    let mut out = Vec::new();
    match f {
        TexFormat::Jpeg(q) => image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q.clamp(1, 100)).write_image(rgb8, w as u32, h as u32, image::ExtendedColorType::Rgb8).map_err(err)?,
        TexFormat::Png => image::codecs::png::PngEncoder::new(&mut out).write_image(rgb8, w as u32, h as u32, image::ExtendedColorType::Rgb8).map_err(err)?,
    }
    Ok(out)
}

/// Wavefront OBJ: positions, texture coordinates (v up from the bottom, the
/// OBJ way) and normals, one index for all three, referring to the material
/// library `mtl` (a file name).
pub fn obj(m: &Mesh, mtl: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.vertices() * 80 + m.triangles() * 40);
    let _ = writeln!(out, "# lapstack 3D model: {}x{} vertices, {} triangles; x = width (1 unit), y up, z towards the viewer", m.nx, m.ny, m.triangles());
    let _ = writeln!(out, "mtllib {mtl}\no stack");
    for p in &m.pos {
        let _ = writeln!(out, "v {:.6} {:.6} {:.6}", p[0], p[1], p[2]);
    }
    for t in &m.uv {
        let _ = writeln!(out, "vt {:.6} {:.6}", t[0], 1.0 - t[1]);
    }
    for n in &m.normal {
        let _ = writeln!(out, "vn {:.4} {:.4} {:.4}", n[0], n[1], n[2]);
    }
    let _ = writeln!(out, "usemtl stack\ns 1");
    for t in &m.tri {
        let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1);
        let _ = writeln!(out, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
    }
    out
}

/// The OBJ's material library: one unlit-white diffuse material with the
/// texture `texture` (a file name beside the OBJ).
pub fn mtl(texture: &str) -> Vec<u8> {
    format!("# lapstack 3D model\nnewmtl stack\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\nd 1.0\nillum 1\nmap_Kd {texture}\n").into_bytes()
}

/// Binary STL: the triangles with their face normals, no texture.
pub fn stl(m: &Mesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + m.triangles() * 50);
    let mut header = [0u8; 80];
    let text = b"lapstack 3D model (binary STL)";
    header[..text.len()].copy_from_slice(text);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(m.triangles() as u32).to_le_bytes());
    for t in &m.tri {
        let (a, b, c) = (m.pos[t[0] as usize], m.pos[t[1] as usize], m.pos[t[2] as usize]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        for x in [n[0] / len, n[1] / len, n[2] / len] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for p in [a, b, c] {
            for x in p {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// glTF 2.0 binary (.glb): one scene, one node, one mesh, the texture
/// (`image` bytes of `mime` image/jpeg | image/png) embedded, an unlit-ish
/// material (no metal, full roughness) so the photograph is not re-lit
/// harshly, double-sided so the back of a wall is not culled.
pub fn glb(m: &Mesh, image: &[u8], mime: &str) -> Vec<u8> {
    let pad4 = |n: usize| (4 - n % 4) % 4;
    let mut bin: Vec<u8> = Vec::with_capacity(m.vertices() * 32 + m.triangles() * 12 + image.len() + 16);
    let mut views = Vec::new();
    let mut push = |bin: &mut Vec<u8>, bytes: &[u8], target: Option<u32>| -> usize {
        let off = bin.len();
        bin.extend_from_slice(bytes);
        bin.resize(bin.len() + pad4(bin.len()), 0);
        views.push(format!("{{\"buffer\":0,\"byteOffset\":{off},\"byteLength\":{}{}}}", bytes.len(), target.map_or(String::new(), |t| format!(",\"target\":{t}"))));
        views.len() - 1
    };
    let f32s = |v: &[[f32; 3]]| -> Vec<u8> { v.iter().flat_map(|p| p.iter().flat_map(|x| x.to_le_bytes())).collect() };
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &m.pos {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let v_pos = push(&mut bin, &f32s(&m.pos), Some(34962));
    let v_nrm = push(&mut bin, &f32s(&m.normal), Some(34962));
    let uv_bytes: Vec<u8> = m.uv.iter().flat_map(|t| t.iter().flat_map(|x| x.to_le_bytes())).collect();
    let v_uv = push(&mut bin, &uv_bytes, Some(34962));
    let idx: Vec<u8> = m.tri.iter().flat_map(|t| t.iter().flat_map(|i| i.to_le_bytes())).collect();
    let v_idx = push(&mut bin, &idx, Some(34963));
    let v_img = push(&mut bin, image, None);
    let nv = m.vertices();
    let json = format!(
        concat!(
            "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"lapstack\"}},\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],",
            "\"nodes\":[{{\"mesh\":0,\"name\":\"stack\"}}],",
            "\"meshes\":[{{\"name\":\"stack\",\"primitives\":[{{\"attributes\":{{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2}},\"indices\":3,\"material\":0,\"mode\":4}}]}}],",
            "\"materials\":[{{\"name\":\"stack\",\"pbrMetallicRoughness\":{{\"baseColorTexture\":{{\"index\":0}},\"metallicFactor\":0,\"roughnessFactor\":1}},\"doubleSided\":true}}],",
            "\"textures\":[{{\"sampler\":0,\"source\":0}}],\"samplers\":[{{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":33071,\"wrapT\":33071}}],",
            "\"images\":[{{\"bufferView\":{img},\"mimeType\":\"{mime}\"}}],",
            "\"accessors\":[",
            "{{\"bufferView\":{pos},\"componentType\":5126,\"count\":{nv},\"type\":\"VEC3\",\"min\":[{lx},{ly},{lz}],\"max\":[{hx},{hy},{hz}]}},",
            "{{\"bufferView\":{nrm},\"componentType\":5126,\"count\":{nv},\"type\":\"VEC3\"}},",
            "{{\"bufferView\":{uv},\"componentType\":5126,\"count\":{nv},\"type\":\"VEC2\"}},",
            "{{\"bufferView\":{idx},\"componentType\":5125,\"count\":{ni},\"type\":\"SCALAR\"}}],",
            "\"bufferViews\":[{views}],\"buffers\":[{{\"byteLength\":{blen}}}]}}"
        ),
        img = v_img, mime = mime, pos = v_pos, nrm = v_nrm, uv = v_uv, idx = v_idx, nv = nv, ni = m.triangles() * 3,
        lx = lo[0], ly = lo[1], lz = lo[2], hx = hi[0], hy = hi[1], hz = hi[2], views = views.join(","), blen = bin.len()
    );
    let mut json = json.into_bytes();
    json.resize(json.len() + pad4(json.len()), b' ');
    let mut out = Vec::with_capacity(12 + 8 + json.len() + 8 + bin.len());
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(grid: usize, near_first: bool) -> MeshParams {
        MeshParams { grid, relief: 0.2, near_first }
    }

    #[test]
    fn grid_and_cells_tile_the_image() {
        assert_eq!(grid_dims(100, 50, 11), (11, 6));
        assert_eq!(grid_dims(50, 100, 11), (6, 11));
        assert_eq!(grid_dims(4, 3, 1000), (4, 3)); // never more than one per pixel
        let c = cells(100, 11);
        assert_eq!(c[0].0, 0);
        assert_eq!(c[10].1, 100);
        for k in 1..11 {
            assert_eq!(c[k - 1].1, c[k].0, "cells tile without gaps or overlap");
        }
    }

    #[test]
    fn a_flat_scene_is_a_flat_plate_at_the_right_height() {
        // depth 0 everywhere (frame 0): the near end with near_first → z = relief; the far end → z = 0
        let (w, h) = (40, 20);
        let z = vec![0f32; w * h];
        let m = heightfield(&z, w, &Rect::full(w, h), 1.0, &params(9, true));
        assert_eq!((m.nx, m.ny), (9, 5));
        assert_eq!(m.triangles(), 2 * 8 * 4);
        assert!(m.pos.iter().all(|p| (p[2] - 0.2).abs() < 1e-6));
        assert!(m.normal.iter().all(|n| (n[2] - 1.0).abs() < 1e-6), "normals point at the viewer");
        let m = heightfield(&z, w, &Rect::full(w, h), 1.0, &params(9, false));
        assert!(m.pos.iter().all(|p| p[2].abs() < 1e-6));
        // the corners: x spans the width (1 unit), y the height in the same unit, uv from the top-left
        assert_eq!(m.pos[0][..2], [-0.5, 0.5 * 19.0 / 39.0]);
        assert_eq!(m.pos[8][..2], [0.5, 0.5 * 19.0 / 39.0]);
        assert_eq!(m.uv[0], [0.0, 0.0]);
        assert_eq!(m.uv[m.uv.len() - 1], [1.0, 1.0]);
    }

    #[test]
    fn a_ramp_is_sampled_as_cell_means_and_the_u16_scale_applies() {
        // depth grows with x, in u16 with 65535 = the last frame: the mean over each vertex's cell
        let (w, h) = (100, 10);
        let z: Vec<u16> = (0..w * h).map(|i| ((i % w) as f32 / 99.0 * 65535.0) as u16).collect();
        let m = heightfield(&z, w, &Rect::full(w, h), 1.0 / 65535.0, &params(11, false));
        for i in 0..11 {
            let (a, b) = cells(100, 11)[i];
            let want = (a..b).map(|x| x as f32 / 99.0).sum::<f32>() / (b - a) as f32 * 0.2;
            assert!((m.pos[i][2] - want).abs() < 1e-3, "vertex {i}: {} vs {want}", m.pos[i][2]);
        }
        // a slope tilts the normal against the rise: z grows with x, so the normal leans to −x
        assert!(m.normal[5][0] < -0.1 && m.normal[5][2] > 0.5);
        // the window of a larger map gives the same relief
        let big: Vec<u16> = (0..(w + 6) * (h + 4)).map(|i| { let (x, y) = (i % (w + 6), i / (w + 6)); if (3..103).contains(&x) && (2..12).contains(&y) { z[(y - 2) * w + x - 3] } else { 65535 } }).collect();
        let m2 = heightfield(&big, w + 6, &Rect { x: 3, y: 2, w, h }, 1.0 / 65535.0, &params(11, false));
        assert_eq!(m.pos, m2.pos);
    }

    #[test]
    fn cells_split_along_the_gentler_diagonal() {
        // a 2x2 vertex grid over a step: a and d high, b and c low → the a–d diagonal has no depth step
        let (w, h) = (2, 2);
        let z = vec![1f32, 0.0, 0.0, 1.0];
        let m = heightfield(&z, w, &Rect::full(w, h), 1.0, &params(2, false));
        assert_eq!(m.tri, vec![[0, 2, 3], [0, 3, 1]]);
        // a and d apart, b and c level → the b–c diagonal
        let z = vec![0f32, 0.5, 0.5, 1.0];
        let m = heightfield(&z, w, &Rect::full(w, h), 1.0, &params(2, false));
        assert_eq!(m.tri, vec![[0, 2, 1], [1, 2, 3]]);
    }

    #[test]
    fn the_writers_agree_on_the_geometry() {
        let (w, h) = (30, 20);
        let z: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / 29.0).collect();
        let m = heightfield(&z, w, &Rect::full(w, h), 1.0, &params(7, true));
        // OBJ: one v / vt / vn per vertex, one f per triangle, 1-based
        let o = String::from_utf8(obj(&m, "model.mtl")).unwrap();
        let count = |pre: &str| o.lines().filter(|l| l.starts_with(pre)).count();
        assert_eq!(count("v "), m.vertices());
        assert_eq!(count("vt "), m.vertices());
        assert_eq!(count("vn "), m.vertices());
        assert_eq!(count("f "), m.triangles());
        assert!(o.contains("mtllib model.mtl") && o.contains("usemtl stack"));
        assert!(String::from_utf8(mtl("model_texture.jpg")).unwrap().contains("map_Kd model_texture.jpg"));
        // STL: 80 + 4 + 50 per triangle, the count in the header
        let s = stl(&m);
        assert_eq!(s.len(), 84 + 50 * m.triangles());
        assert_eq!(u32::from_le_bytes(s[80..84].try_into().unwrap()) as usize, m.triangles());
        // GLB: the chunk lengths add up, the JSON is well-formed enough to find the counts, the image is at the end of BIN
        let img = b"\xff\xd8not really a jpeg\xff\xd9";
        let g = glb(&m, img, "image/jpeg");
        assert_eq!(&g[..4], b"glTF");
        assert_eq!(u32::from_le_bytes(g[8..12].try_into().unwrap()) as usize, g.len());
        let jl = u32::from_le_bytes(g[12..16].try_into().unwrap()) as usize;
        assert_eq!(&g[16..20], b"JSON");
        let json = std::str::from_utf8(&g[20..20 + jl]).unwrap();
        assert!(json.contains(&format!("\"count\":{}", m.vertices())) && json.contains(&format!("\"count\":{}", m.triangles() * 3)));
        assert!(json.contains("\"mimeType\":\"image/jpeg\""));
        let bl = u32::from_le_bytes(g[20 + jl..24 + jl].try_into().unwrap()) as usize;
        assert_eq!(&g[24 + jl..28 + jl], b"BIN\0");
        assert_eq!(28 + jl + bl, g.len());
        assert_eq!(bl % 4, 0);
        let bin = &g[28 + jl..];
        let img_off = m.vertices() * 32 + m.triangles() * 12;
        assert_eq!(&bin[img_off..img_off + img.len()], img);
        // the first position in BIN is vertex 0
        let x0 = f32::from_le_bytes(bin[0..4].try_into().unwrap());
        assert_eq!(x0, m.pos[0][0]);
    }

    #[test]
    fn textures_shrink_to_the_edge_and_encode() {
        assert_eq!(texture_dims(8280, 5520, 8192), (8192, 5461));
        assert_eq!(texture_dims(800, 600, 8192), (800, 600));
        assert_eq!(texture_dims(800, 600, 0), (800, 600));
        let (w, h) = (8, 4);
        let rgb: Vec<u16> = (0..w * h * 3).map(|i| if i % 3 == 0 { 65535 } else { 0 }).collect();
        let (t, tw, th) = texture(&rgb, w, &Rect::full(w, h), 1.0 / 65535.0, 4);
        assert_eq!((tw, th), (4, 2));
        assert_eq!(&t[..6], &[255, 0, 0, 255, 0, 0]);
        let planes = [vec![1f32; w * h], vec![0.5f32; w * h], vec![0f32; w * h]];
        let (t, tw, th) = texture_planes([&planes[0], &planes[1], &planes[2]], w, h, 0);
        assert_eq!((tw, th), (w, h));
        assert_eq!(&t[..3], &[255, 128, 0]);
        let png = encode_texture(&t, tw, th, TexFormat::Png).unwrap();
        assert_eq!(&png[1..4], b"PNG");
        let jpg = encode_texture(&t, tw, th, TexFormat::Jpeg(90)).unwrap();
        assert_eq!(&jpg[..2], &[0xff, 0xd8]);
    }
}

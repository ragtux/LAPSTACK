// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! lapstack — Laplacian-pyramid focus stacking CLI.

use lapstack_core::{DepthParams, FocusMeasure, Layout, MeshParams, Params, TexFormat, TopRule, Upsample, View, run_with};
use lapstack_core::mesh;
use lapstack_core::io;
use lapstack_core::pyramid::Img3;
use lapstack_core::view;
use std::time::Instant;

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut inputs = Vec::new();
    let mut output = "stacked.png".to_string();
    let mut save_depth = false;
    let mut save_conf = false;
    let mut metadata = true;
    let mut depth_raw: Option<String> = None;
    let mut do_align = true;
    let mut p = Params::default();
    let mut a = p.align.unwrap();
    let mut dp = DepthParams::default();
    let mut depth_mode = "dff".to_string();
    let mut slab_dir: Option<String> = None;
    let mut stereo: Option<(f32, Layout)> = None;
    let mut rocking: Option<(f32, usize)> = None;
    let mut near_first = true;
    let mut mesh_formats: Vec<String> = Vec::new();
    let mut mp = MeshParams::default();
    let mut mesh_tex = (8192usize, TexFormat::Jpeg(92));

    let mut i = 0;
    let next = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| fail(&format!("{} needs a value", args[*i - 1])))
    };
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => output = next(&mut i),
            "--levels" => p.fuse.levels = Some(next(&mut i).parse().unwrap_or_else(|_| fail("--levels: integer"))),
            "--energy-radius" => {
                p.fuse.energy_radius = next(&mut i).parse().unwrap_or_else(|_| fail("--energy-radius: integer"))
            }
            "--top" => {
                let s = next(&mut i);
                p.fuse.top_rule = TopRule::parse(&s).unwrap_or_else(|| fail(&format!("--top: unknown rule '{s}'")));
            }
            "--top-radius" => p.fuse.top_radius = next(&mut i).parse().unwrap_or_else(|_| fail("--top-radius: integer")),
            "--entropy-bins" => {
                p.fuse.entropy_bins = next(&mut i).parse().unwrap_or_else(|_| fail("--entropy-bins: integer"))
            }
            "--use-chroma" => p.fuse.use_chroma = true,
            "--depth-level" => p.fuse.depth_level = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-level: integer")),
            "--no-align" => do_align = false,
            "--no-shift" => a.shift = false,
            "--no-scale" => a.scale = false,
            "--no-rotation" => a.rotation = false,
            "--align-coarsen" => a.coarsen = next(&mut i).parse().unwrap_or_else(|_| fail("--align-coarsen: integer")),
            "--save-aligned" => p.save_aligned = Some(next(&mut i)),
            "--save-depth" => save_depth = true,
            "--save-conf" => save_conf = true,
            "--no-metadata" => metadata = false,
            "--no-crop" => p.crop = false,
            "--no-brightness" => p.brightness = false,
            "--slabs" => {
                let s = next(&mut i);
                let mut it = s.split(':');
                let size = it.next().and_then(|v| v.parse().ok()).filter(|&v| v >= 1).unwrap_or_else(|| fail("--slabs: SIZE[:OVERLAP], SIZE >= 1"));
                let overlap = match it.next() { Some(o) => o.parse().unwrap_or_else(|_| fail("--slabs: SIZE[:OVERLAP]")), None => 2 };
                p.slabs = Some((size, overlap));
            }
            "--slab-dir" => slab_dir = Some(next(&mut i)),
            "--stereo" => {
                let s = next(&mut i);
                let mut it = s.split(':');
                let pct: f32 = it.next().and_then(|v| v.parse().ok()).filter(|v: &f32| *v > 0.0).unwrap_or_else(|| fail("--stereo: PCT[:LAYOUT], PCT > 0"));
                let layout = match it.next() { Some(l) => Layout::parse(l).unwrap_or_else(|| fail("--stereo: layout sbs | cross | anaglyph")), None => Layout::SideBySide };
                stereo = Some((pct / 100.0, layout));
            }
            "--rocking" => {
                let s = next(&mut i);
                let mut it = s.split(':');
                let pct: f32 = it.next().and_then(|v| v.parse().ok()).filter(|v: &f32| *v > 0.0).unwrap_or_else(|| fail("--rocking: PCT[:N], PCT > 0"));
                let n = match it.next() { Some(n) => n.parse().ok().filter(|&n| n >= 2).unwrap_or_else(|| fail("--rocking: PCT[:N], N >= 2")), None => 24 };
                rocking = Some((pct / 100.0, n));
            }
            "--far-first" => near_first = false,
            "--mesh" => {
                for f in next(&mut i).split(',') {
                    if !matches!(f, "glb" | "obj" | "stl") {
                        fail("--mesh: glb | obj | stl, comma-separated");
                    }
                    if !mesh_formats.iter().any(|g| g == f) {
                        mesh_formats.push(f.to_string());
                    }
                }
            }
            "--mesh-relief" => mp.relief = next(&mut i).parse::<f32>().ok().filter(|v| *v > 0.0).unwrap_or_else(|| fail("--mesh-relief: PCT > 0")) / 100.0,
            "--mesh-grid" => mp.grid = next(&mut i).parse().ok().filter(|&v| v >= 2).unwrap_or_else(|| fail("--mesh-grid: N >= 2")),
            "--mesh-texture" => {
                let s = next(&mut i);
                let mut it = s.split(':');
                mesh_tex.0 = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| fail("--mesh-texture: EDGE[:jpeg[:Q] | png]"));
                mesh_tex.1 = match it.next() {
                    None | Some("jpeg") | Some("jpg") => TexFormat::Jpeg(match it.next() { Some(q) => q.parse().ok().filter(|q| (1..=100).contains(q)).unwrap_or_else(|| fail("--mesh-texture: Q 1..100")), None => 92 }),
                    Some("png") => TexFormat::Png,
                    _ => fail("--mesh-texture: EDGE[:jpeg[:Q] | png]"),
                };
            }
            "--depth-raw" => depth_raw = Some(next(&mut i)),
            "--depth" => {
                depth_mode = next(&mut i);
                if depth_mode != "dff" && depth_mode != "winner" {
                    fail("--depth: dff | winner");
                }
            }
            "--depth-scale" => dp.scale = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-scale: integer")),
            "--depth-focus" => {
                let s = next(&mut i);
                dp.focus = FocusMeasure::parse(&s).unwrap_or_else(|| fail(&format!("--depth-focus: bad spec '{s}'")));
            }
            "--depth-agg" => {
                let s = next(&mut i);
                let mut it = s.split(':');
                dp.agg_radius = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| fail("--depth-agg: R[:EPS]"));
                if let Some(e) = it.next() {
                    dp.agg_eps = e.parse().unwrap_or_else(|_| fail("--depth-agg: R[:EPS]"));
                }
            }
            "--depth-lambda" => dp.lambda = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-lambda: number")),
            "--depth-sigma" => dp.sigma_c = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-sigma: number")),
            "--depth-cg" => dp.cg_iters = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-cg: integer")),
            "--depth-no-median" => dp.median = false,
            "--depth-gate" => dp.gate = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-gate: number")),
            "--depth-robust" => dp.robust = next(&mut i).parse().unwrap_or_else(|_| fail("--depth-robust: number")),
            "--depth-upsample" => {
                let s = next(&mut i);
                dp.upsample = Upsample::parse(&s).unwrap_or_else(|| fail(&format!("--depth-upsample: bad spec '{s}'")));
            }
            "--gpu" => p.gpu = true,
            "--gpu-align" => a.gpu = true,
            "-h" | "--help" => {
                help();
                return;
            }
            s if s.starts_with('-') && s.len() > 1 => fail(&format!("unknown option '{s}'; see --help")),
            s => inputs.push(s.to_string()),
        }
        i += 1;
    }
    p.align = do_align.then_some(a);
    p.depth = (depth_mode == "dff").then_some(dp);

    if inputs.is_empty() {
        fail("no input images; use --help");
    }

    let t0 = Instant::now();
    let mut log = |s: String| eprintln!("[lapstack] {s}");
    // the first frame's EXIF, ICC profile and XMP go into the fused image (and the slabs)
    let meta = if metadata { Some(io::load_meta(&inputs[0]).unwrap_or_else(|e| fail(&e))) } else { None };
    // slabs: written as they are fused, in the output's format, to --slab-dir [<output stem>_slabs]
    let stem = match output.rfind('.') { Some(k) => &output[..k], None => &output[..] };
    let ext = match output.rfind('.') { Some(k) => &output[k..], None => ".png" };
    let slab_dir = slab_dir.unwrap_or_else(|| format!("{stem}_slabs"));
    let mut on_slab = |s: lapstack_core::Slab<'_>| -> Result<(), String> {
        if s.index == 0 {
            std::fs::create_dir_all(&slab_dir).map_err(|e| format!("cannot create {slab_dir}: {e}"))?;
        }
        let path = format!("{slab_dir}/slab_{:02}_{:03}-{:03}{ext}", s.index + 1, s.lo, s.hi);
        io::save_rgb(s.image, &path, s.bit_depth, meta.as_ref())?;
        eprintln!("[lapstack] slab {}/{} (frames {}..{}) -> {path}", s.index + 1, s.count, s.lo, s.hi);
        Ok(())
    };
    let out = run_with(&inputs, &p, &mut log, &mut on_slab).unwrap_or_else(|e| fail(&e));
    if let Some(m) = &meta {
        eprintln!("[lapstack] metadata from {}: {}", inputs[0], m.describe());
    }
    if let Err(e) = io::save_rgb(&out.image, &output, out.bit_depth, meta.as_ref()) {
        fail(&e);
    }
    eprintln!("[lapstack] fused -> {output}");
    if save_depth {
        let dp = match output.rfind('.') {
            Some(k) => format!("{}_depth.png", &output[..k]),
            None => format!("{output}_depth.png"),
        };
        if let Err(e) = io::save_gray(&out.depth, out.image.w, out.image.h, &dp) {
            fail(&e);
        }
        match &p.depth {
            Some(_) => eprintln!("[lapstack] depth map (depth from focus) -> {dp}"),
            None => eprintln!("[lapstack] depth map (level-{} winners) -> {dp}", p.fuse.depth_level.min(out.levels - 1)),
        }
    }
    let n_last = (inputs.len().max(2) - 1) as f32;
    // synthetic stereo and rocking (view.rs): the result sheared by its depth map, plane by plane
    let (iw, ih) = (out.image.w, out.image.h);
    let sheared = |v: &View| -> Img3 {
        let mut img = Img3::zeros(iw, ih);
        for c in 0..3 {
            view::render(&out.image.p[c], iw, ih, 1, &out.depth, 1.0 / n_last, v, &mut img.p[c], iw, 0);
        }
        img
    };
    if let Some((shift, layout)) = stereo {
        let (l, r) = (sheared(&View::new(-shift, near_first)), sheared(&View::new(shift, near_first)));
        let pair = match layout {
            Layout::Anaglyph => Img3 { w: iw, h: ih, p: [l.p[0].clone(), r.p[1].clone(), r.p[2].clone()] },
            _ => {
                let (a, b) = if layout == Layout::SideBySide { (&l, &r) } else { (&r, &l) };
                let mut img = Img3::zeros(2 * iw, ih);
                for c in 0..3 {
                    for y in 0..ih {
                        img.p[c][y * 2 * iw..y * 2 * iw + iw].copy_from_slice(&a.p[c][y * iw..(y + 1) * iw]);
                        img.p[c][y * 2 * iw + iw..(y + 1) * 2 * iw].copy_from_slice(&b.p[c][y * iw..(y + 1) * iw]);
                    }
                }
                img
            }
        };
        let path = format!("{stem}_stereo{ext}");
        if let Err(e) = io::save_rgb(&pair, &path, out.bit_depth, meta.as_ref()) {
            fail(&e);
        }
        eprintln!("[lapstack] stereo pair (±{:.1} %, {}) -> {path}", shift * 100.0, match layout { Layout::SideBySide => "left | right", Layout::CrossEyed => "right | left", Layout::Anaglyph => "red-cyan anaglyph" });
    }
    if let Some((amp, n)) = rocking {
        let dir = format!("{stem}_rocking");
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&format!("cannot create {dir}: {e}")));
        for (i, s) in view::rocking_shifts(amp, n).into_iter().enumerate() {
            let path = format!("{dir}/view_{i:02}{ext}");
            if let Err(e) = io::save_rgb(&sheared(&View::new(s, near_first)), &path, out.bit_depth, meta.as_ref()) {
                fail(&e);
            }
        }
        eprintln!("[lapstack] rocking: {n} views of ±{:.1} % -> {dir}/view_NN{ext}", amp * 100.0);
    }
    if !mesh_formats.is_empty() {
        // the 3D model (mesh.rs): the depth map as a relief textured with the result
        mp.near_first = near_first;
        let m = mesh::heightfield(&out.depth, iw, &lapstack_core::align::Rect::full(iw, ih), 1.0 / n_last, &mp);
        let (tex, tw, th) = mesh::texture_planes([&out.image.p[0], &out.image.p[1], &out.image.p[2]], iw, ih, mesh_tex.0);
        let write = |path: &str, bytes: &[u8]| std::fs::write(path, bytes).unwrap_or_else(|e| fail(&format!("cannot write {path}: {e}")));
        eprintln!("[lapstack] 3D model: {}x{} vertices, {} triangles, relief {:.0} % of the width, texture {tw}x{th}", m.nx, m.ny, m.triangles(), mp.relief * 100.0);
        let image = if mesh_formats.iter().any(|f| f != "stl") { mesh::encode_texture(&tex, tw, th, mesh_tex.1).unwrap_or_else(|e| fail(&e)) } else { Vec::new() };
        for f in &mesh_formats {
            match f.as_str() {
                "glb" => {
                    let path = format!("{stem}.glb");
                    write(&path, &mesh::glb(&m, &image, mesh_tex.1.mime()));
                    eprintln!("[lapstack] glTF binary -> {path}");
                }
                "obj" => {
                    let base = stem.rsplit('/').next().unwrap_or(stem);
                    let (mtl_name, tex_name) = (format!("{base}.mtl"), format!("{base}_texture.{}", mesh_tex.1.ext()));
                    write(&format!("{stem}.obj"), &mesh::obj(&m, &mtl_name));
                    write(&format!("{stem}.mtl"), &mesh::mtl(&tex_name));
                    write(&format!("{stem}_texture.{}", mesh_tex.1.ext()), &image);
                    eprintln!("[lapstack] Wavefront OBJ -> {stem}.obj + {mtl_name} + {tex_name}");
                }
                _ => {
                    let path = format!("{stem}.stl");
                    write(&path, &mesh::stl(&m));
                    eprintln!("[lapstack] binary STL -> {path}");
                }
            }
        }
    }
    if let Some(path) = &depth_raw {
        if let Err(e) = io::save_gray16(&out.depth, out.image.w, out.image.h, n_last, path) {
            fail(&e);
        }
        eprintln!("[lapstack] depth (16-bit, 65535 = frame {}) -> {path}", inputs.len() - 1);
    }
    if save_conf {
        match &out.conf {
            Some(c) => {
                let cp = match output.rfind('.') {
                    Some(k) => format!("{}_conf.png", &output[..k]),
                    None => format!("{output}_conf.png"),
                };
                if let Err(e) = io::save_gray16(c, out.image.w, out.image.h, 1.0, &cp) {
                    fail(&e);
                }
                eprintln!("[lapstack] confidence -> {cp}");
            }
            None => eprintln!("[lapstack] --save-conf: no confidence map with --depth winner"),
        }
    }
    eprintln!("[lapstack] total {:.1}s", t0.elapsed().as_secs_f64());
}

fn help() {
    eprintln!(
        "lapstack — Laplacian-pyramid focus stacking (Burt & Adelson; Wang & Chang 2011)\n\
         Usage: lapstack [options] frame1 frame2 ...\n\
         Output keeps the input bit depth (16-bit needs PNG/TIFF).\n\
           -o, --output PATH      fused image [stacked.png]\n\
           --levels N             band-pass levels [auto: residual short side >= 32 px]\n\
           --energy-radius R      region-energy window radius, binomial weights [1 = 3x3];\n\
                                  0 = per-node |L| max (Adelson et al. 1984)\n\
           --top RULE             residual rule: de (deviation+entropy, paper) | dev | avg [de]\n\
           --top-radius R         D/E window radius at the residual [2 = 5x5]\n\
           --entropy-bins N       gray levels for the entropy histogram [256]\n\
           --use-chroma           energy from R,G,B instead of luma\n\
           --no-align             frames are already registered (streams from disk)\n\
           --no-shift/scale/rotation   restrict the similarity model\n\
           --align-coarsen N      align at reduced resolution (skip N finest levels)\n\
           --save-aligned DIR     write the aligned frames\n\
           --save-depth           write the depth map next to the output (8-bit, min-max scaled)\n\
           --depth-raw PATH       write the depth map as 16-bit PNG, fixed scale (65535 = last frame)\n\
           --save-conf            write the depth confidence map (16-bit, 65535 = 1)\n\
           --no-metadata          do not copy the first frame's EXIF / ICC profile / XMP into the output\n\
           --no-crop              keep the full frame instead of cropping to the area every aligned frame covers\n\
           --no-brightness        do not equalise the frames' brightness to frame 0 (exposure flicker)\n\
           --slabs SIZE[:OVERLAP] also fuse slabs of SIZE consecutive frames, overlapping by OVERLAP [2],\n\
                                  each on its own (Zerene's slabbing): thick planes of focus to retouch from\n\
           --slab-dir DIR         where the slabs go, in the output's format [<output stem>_slabs]\n\
           --stereo PCT[:LAYOUT]  synthetic stereo pair -> <stem>_stereo.<ext>: the result sheared by its depth map,\n\
                                  the far end of the stack moved -PCT / +PCT % of the width (left / right view);\n\
                                  LAYOUT sbs (left | right, default) | cross (right | left) | anaglyph (red-cyan)\n\
           --rocking PCT[:N]      rocking animation: N [24] views, the shift sweeping +-PCT % in one sine cycle,\n\
                                  -> <stem>_rocking/view_NN.<ext> (join them with ffmpeg / ImageMagick)\n\
           --far-first            frame 0 is the far end of the stack (the focus went back to front) [near]\n\
           --mesh FORMATS         3D model (Helicon's): the depth map as a relief textured with the result, as\n\
                                  glb (glTF binary, one file) | obj (+ .mtl + texture file) | stl (geometry only),\n\
                                  comma-separated -> <stem>.glb / .obj / .stl; --far-first applies\n\
           --mesh-relief PCT      the depth of the stack as a percentage of the image width [25]\n\
           --mesh-grid N          vertices along the long edge [1000]\n\
           --mesh-texture EDGE[:jpeg[:Q] | png]   the texture's long edge, 0 = full [8192], and format [jpeg:92]\n\
           --depth MODE           dff = depth from focus (default) | winner = pyramid winner map\n\
           --depth-level L        (winner) pyramid level the map is read from [2 = 1/4 res]\n\
         Depth from focus (Jeon et al. 2019 focus measure, guided-filter aggregation,\n\
         sub-frame peaks, confidence-weighted edge-aware WLS):\n\
           --depth-scale S        work at 1/2^S resolution [1]\n\
           --depth-focus F        rdf[:RIN[:ROUT]] ring difference filter [rdf:1:3] | sml[:STEP]\n\
           --depth-agg R[:EPS]    guided-filter aggregation radius / regulariser [3:1e-4]; 0 = off\n\
           --depth-lambda L       WLS smoothness [3]; 0 = off\n\
           --depth-sigma S        WLS guide-edge sensitivity, luma units [0.04]\n\
           --depth-cg N           WLS conjugate-gradient iteration cap [200]\n\
           --depth-no-median      skip the 3x3 median before the WLS\n\
           --depth-gate G         noise gate: full confidence needs peak >= (1+G) x noise floor [1]; 0 = off\n\
           --depth-robust T       Huber reweighting of the WLS data term, T frames [1]; 0 = off\n\
           --depth-upsample U     bilinear | guided[:R[:EPS]] [guided:2:1e-2]\n\
           --gpu                  fuse on the CUDA GPU (needs the 'gpu' build feature)\n\
           --gpu-align            run the aligner's cost search on the CUDA GPU"
    );
}

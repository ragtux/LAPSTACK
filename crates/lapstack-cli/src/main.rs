// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! lapstack — Laplacian-pyramid focus stacking CLI.

use lapstack_core::{DepthParams, FocusMeasure, Params, TopRule, Upsample, run};
use lapstack_core::io;
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

    let t0 = Instant::now();
    let mut log = |s: String| eprintln!("[lapstack] {s}");
    let out = run(&inputs, &p, &mut log).unwrap_or_else(|e| fail(&e));
    // the first frame's EXIF, ICC profile and XMP go into the fused image
    let meta = if metadata { Some(io::load_meta(&inputs[0]).unwrap_or_else(|e| fail(&e))) } else { None };
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

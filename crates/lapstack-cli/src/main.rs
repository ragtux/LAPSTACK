// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! lapstack — Laplacian-pyramid focus stacking CLI.

use lapstack_core::{DepthParams, DustMode, DustParams, FocusMeasure, Layout, MeshParams, Params, Split, Stack, TexFormat, TopRule, Upsample, View, run_with};
use lapstack_core::align::{AlignModel, Interp};
use lapstack_core::batch::{self, Names};
use lapstack_core::mesh;
use lapstack_core::overlay::{self, Corner, Ink, Overlay, OverlayParams, Style};
use lapstack_core::io;
use lapstack_core::pyramid::Img3;
use lapstack_core::view;
use std::time::Instant;

/// Large blocks are recycled between frames (`pool.rs`): the fold's planes
/// would otherwise be mapped and page-faulted afresh for every frame.
#[global_allocator]
static ALLOC: lapstack_core::pool::PoolAlloc = lapstack_core::pool::PoolAlloc::new();

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn main() {
    if std::env::var_os("LAPSTACK_NO_POOL").is_some() {
        lapstack_core::pool::DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
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
    let mut video: Option<f32> = None;
    let mut near_end = NearEnd::Auto;
    let mut mesh_formats: Vec<String> = Vec::new();
    let mut mp = MeshParams::default();
    let mut mesh_tex = (8192usize, TexFormat::Jpeg(92));
    let mut split: Option<Split> = None;
    let mut dry_run = false;
    let mut skip: Option<String> = None;
    let mut reverse = false;
    let mut dust_map: Option<String> = None;
    let mut dust = DustParams::default();
    let mut save_dust: Option<String> = None;
    let mut ov = OverlayParams::default();
    let mut ov_auto = false;   // --scale-bar auto: the calibration from the first frame's TIFF

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
            "--halo-control" => p.fuse.halo = next(&mut i).parse().unwrap_or_else(|_| fail("--halo-control: number")),
            "--no-align" => do_align = false,
            "--no-shift" => a.shift = false,
            "--no-scale" => a.scale = false,
            "--no-rotation" => a.rotation = false,
            "--align-coarsen" => a.coarsen = next(&mut i).parse().unwrap_or_else(|_| fail("--align-coarsen: integer")),
            "--align-model" => {
                let s = next(&mut i);
                a.model = AlignModel::parse(&s).unwrap_or_else(|| fail(&format!("--align-model: unknown model '{s}' (similarity | affine | projective)")));
            }
            "--interpolation" => {
                let s = next(&mut i);
                a.interp = Interp::parse(&s).unwrap_or_else(|| fail(&format!("--interpolation: unknown kernel '{s}' (nearest | bilinear | bicubic | spline4x4 | spline6x6 | lanczos3)")));
            }
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
            "--wav" => p.wav = Some(p.wav.unwrap_or_default()),
            "--wav-power" => { let v = next(&mut i).parse::<f32>().ok().filter(|v| *v >= 0.0).unwrap_or_else(|| fail("--wav-power: number >= 0")); p.wav = Some(lapstack_core::wav::WavParams { power: v, ..p.wav.unwrap_or_default() }); }
            "--wav-smooth" => { let v = next(&mut i).parse::<usize>().unwrap_or_else(|_| fail("--wav-smooth: integer")); p.wav = Some(lapstack_core::wav::WavParams { smooth: v, ..p.wav.unwrap_or_default() }); }
            "--wav-gate" => { let v = next(&mut i).parse::<f32>().ok().filter(|v| *v >= 0.0).unwrap_or_else(|| fail("--wav-gate: number >= 0")); p.wav = Some(lapstack_core::wav::WavParams { gate: v, ..p.wav.unwrap_or_default() }); }
            "--split" => {
                let s = next(&mut i);
                split = Some(Split::parse(&s).unwrap_or_else(|| fail(&format!("--split: count:N | gap:SECONDS | dir, not '{s}'"))));
            }
            "--dry-run" => dry_run = true,
            "--skip" => skip = Some(next(&mut i)),
            "--reverse" => reverse = true,
            "--dust-map" => dust_map = Some(next(&mut i)),
            "--dust-threshold" => dust.threshold = next(&mut i).parse::<f32>().ok().filter(|v| *v > 0.0 && *v < 90.0).unwrap_or_else(|| fail("--dust-threshold: percent, 0 < PCT < 90")) / 100.0,
            "--dust-margin" => dust.margin = next(&mut i).parse().unwrap_or_else(|_| fail("--dust-margin: integer")),
            "--dust-mode" => { let s = next(&mut i); dust.mode = DustMode::parse(&s).unwrap_or_else(|| fail("--dust-mode: fill | flat")); }
            "--save-dust-map" => save_dust = Some(next(&mut i)),
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
            "--video" => video = Some(next(&mut i).parse().ok().filter(|v: &f32| *v > 0.0).unwrap_or_else(|| fail("--video: FPS > 0"))),
            "--far-first" => near_end = NearEnd::Last,
            "--near-end" => {
                let s = next(&mut i);
                near_end = match s.as_str() { "auto" => NearEnd::Auto, "first" => NearEnd::First, "last" => NearEnd::Last, _ => fail("--near-end: auto | first | last") };
            }
            "--rotate" => {
                let d: i32 = next(&mut i).parse().unwrap_or_else(|_| fail("--rotate: degrees, a multiple of 90"));
                p.rotate = lapstack_core::prep::quarters_of(d).unwrap_or_else(|| fail("--rotate: 90 | 180 | 270 (or -90)"));
            }
            "--draft" => p.reduce = next(&mut i).parse::<usize>().ok().filter(|&n| n <= 6).unwrap_or_else(|| fail("--draft: N levels, 0..6 (the frames reduced by 2^N)")),
            "--crop" => {
                let s = next(&mut i);
                let v: Vec<usize> = s.split(',').map(|t| t.trim().parse().unwrap_or_else(|_| fail("--crop: X,Y,W,H in pixels"))).collect();
                if v.len() != 4 || v[2] == 0 || v[3] == 0 {
                    fail("--crop: X,Y,W,H in pixels of the (turned) frame, W and H > 0");
                }
                p.crop_rect = Some(lapstack_core::align::Rect { x: v[0], y: v[1], w: v[2], h: v[3] });
            }
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
            "--scale-bar" => {
                // CAL[:LENGTH]: the size of a pixel (0.325 = µm; 325nm), auto (the first frame's TIFF) or px (no
                // calibration: the bar is labelled in pixels), and the bar's length
                let s = next(&mut i);
                let mut it = s.splitn(2, ':');
                let cal = it.next().unwrap_or("");
                ov.bar = true;
                if cal.eq_ignore_ascii_case("auto") {
                    ov_auto = true;
                } else if !matches!(cal.to_ascii_lowercase().as_str(), "px" | "pixels" | "none") {
                    ov.um_per_px = overlay::parse_length_um(cal).unwrap_or_else(|| fail("--scale-bar CAL[:LENGTH]: CAL is the size of one pixel of the frames, in µm (0.325) or with a unit (325nm), auto, or px for a bar in pixels"));
                }
                if let Some(len) = it.next() {
                    ov.bar_um = if len.eq_ignore_ascii_case("auto") {
                        0.0
                    } else if ov_auto || ov.um_per_px > 0.0 {
                        overlay::parse_length_um(len).unwrap_or_else(|| fail("--scale-bar CAL[:LENGTH]: LENGTH with a unit (100um | 2mm | 500nm), or auto"))
                    } else {
                        overlay::parse_length_px(len).unwrap_or_else(|| fail("--scale-bar px[:LENGTH]: LENGTH in pixels of the frames (500 | 500px), or auto"))
                    };
                }
            }
            "--text" => ov.text = next(&mut i),
            "--overlay-pos" => {
                let s = next(&mut i);
                let mut it = s.split(',');
                let corner = |c: &str| Corner::parse(c).unwrap_or_else(|| fail("--overlay-pos BAR[,TEXT]: corners tl | tr | bl | br"));
                ov.bar_pos = corner(it.next().unwrap_or(""));
                if let Some(t) = it.next() {
                    ov.text_pos = corner(t);
                }
            }
            "--overlay-size" => ov.size = next(&mut i).parse::<f32>().ok().filter(|v| *v > 0.0 && *v <= 50.0).unwrap_or_else(|| fail("--overlay-size: percent of the image height, 0 < PCT <= 50")) / 100.0,
            "--overlay-color" => { let s = next(&mut i); ov.color = Ink::parse(&s).unwrap_or_else(|| fail("--overlay-color: white | black")); }
            "--overlay-style" => { let s = next(&mut i); ov.style = Style::parse(&s).unwrap_or_else(|| fail("--overlay-style: halo | box | plain")); }
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
    // -o x.dng: a linear-DNG run (the raws developed to their camera space, dng.rs)
    p.dng = io::is_dng(&output);

    if inputs.is_empty() {
        fail("no input images; use --help");
    }
    if video.is_some() && rocking.is_none() {
        fail("--video joins the rocking views: it needs --rocking");
    }
    // the dust map: detected once, applied to every frame of every stack
    if let Some(path) = &dust_map {
        let t = Instant::now();
        let (img, _) = io::load_rgb(path).unwrap_or_else(|e| fail(&format!("--dust-map: {e}")));
        let map = lapstack_core::dust::detect(&lapstack_core::dust::luma(&img), img.w, img.h, &dust);
        eprintln!("[lapstack] dust map {path} ({}x{}, threshold {:.1} %, margin {} px, {}): {}  ({:.1}s)", img.w, img.h, 100.0 * dust.threshold, dust.margin, dust.mode.name(), map.describe(), t.elapsed().as_secs_f64());
        if let Some(out) = &save_dust {
            io::save_gray(&map.plane(), map.w, map.h, out).unwrap_or_else(|e| fail(&e));
            eprintln!("[lapstack] wrote the dust mask to {out} (white = dust)");
        }
        if map.is_empty() {
            eprintln!("[lapstack] no dust spots found: the frames are stacked as they are (a lower --dust-threshold finds fainter spots)");
        } else {
            p.dust = Some(map);
        }
    } else if save_dust.is_some() {
        fail("--save-dust-map writes the spots found in the dust map: it needs --dust-map");
    }
    let overlay = (ov_auto || !ov.is_empty()).then_some(ov);
    let cfg = Cfg { output, save_depth, save_conf, metadata, depth_raw, p, slab_dir, stereo, rocking, video, near_end, mesh_formats, mp, mesh_tex, overlay, overlay_auto: ov_auto };

    // a directory among the inputs stands for the image files in it; --skip counts positions in that list
    let inputs = batch::expand_dirs(&inputs).unwrap_or_else(|e| fail(&e));
    let inputs = match &skip {
        Some(spec) => {
            let out = batch::skip_list(spec, inputs.len()).unwrap_or_else(|e| fail(&e));
            for &k in &out {
                eprintln!("[lapstack] skipping frame {}: {}", k + 1, inputs[k]);
            }
            let kept = batch::without(&inputs, &out);
            if kept.is_empty() {
                fail("--skip leaves no frames");
            }
            kept
        }
        None => inputs,
    };
    let t0 = Instant::now();
    let mut stacks: Vec<Stack> = match split {
        None => vec![Stack { inputs, times: None }],
        Some(Split::Count(n)) => batch::split_count(&inputs, n),
        Some(Split::Dir) => batch::split_dir(&inputs),
        Some(Split::Gap(gap)) => {
            eprintln!("[lapstack] reading the capture times of {} frames ...", inputs.len());
            let times = batch::capture_times(&inputs, &mut |s| eprintln!("[lapstack] {s}")).unwrap_or_else(|e| fail(&e));
            batch::split_gap(&inputs, &times, gap)
        }
    };
    // each stack is reversed on its own: a rail run back to front is so in every stack of the batch
    if reverse {
        for s in &mut stacks {
            s.inputs.reverse();
            s.times = s.times.map(|(a, b)| (b, a));
        }
        eprintln!("[lapstack] frames reversed: {} is frame 0", stacks[0].inputs[0]);
    }
    let count = stacks.len();
    if let Some(rule) = split {
        eprintln!("[lapstack] {count} stack{} from {} frames ({}):", if count == 1 { "" } else { "s" }, stacks.iter().map(|s| s.inputs.len()).sum::<usize>(), describe_split(rule));
        let mut prev_end: Option<f64> = None;
        for (k, s) in stacks.iter().enumerate() {
            let names = Names { n: k + 1, count, first: s.first(), dir: &s.dir() };
            let (a, b) = (s.inputs[0].as_str(), s.inputs[s.inputs.len() - 1].as_str());
            let when = match s.times {
                Some((t0, t1)) => format!("  {} .. {}{}", batch::clock(t0), batch::clock(t1), match prev_end { Some(p) => format!(", {:+.0} s after the last", t0 - p), None => String::new() }),
                None => String::new(),
            };
            prev_end = s.times.map(|t| t.1);
            eprintln!("  {:>3}: {:>4} frames  {} .. {}  -> {}{when}", k + 1, s.inputs.len(), base(a), base(b), batch::expand(&cfg.output, &names));
        }
    }
    if dry_run {
        return;
    }

    let mut failed: Vec<(usize, String)> = Vec::new();
    for (k, s) in stacks.iter().enumerate() {
        let names = Names { n: k + 1, count, first: s.first(), dir: &s.dir() };
        let tag = if count > 1 { format!(" {}/{count}", k + 1) } else { String::new() };
        if count > 1 {
            eprintln!("[lapstack{tag}] {} frames, {} .. {}", s.inputs.len(), s.inputs[0], s.inputs[s.inputs.len() - 1]);
        }
        match run_stack(&s.inputs, &cfg, &names, &tag) {
            Ok(()) => {}
            Err(e) if count > 1 => {
                eprintln!("[lapstack{tag}] failed: {e}");
                failed.push((k + 1, e));
            }
            Err(e) => fail(&e),
        }
    }
    if count > 1 {
        let done = count - failed.len();
        eprintln!("[lapstack] batch: {count} stacks, {done} done, {} failed, total {:.1}s", failed.len(), t0.elapsed().as_secs_f64());
        for (n, e) in &failed {
            eprintln!("[lapstack]   stack {n}: {e}");
        }
        if !failed.is_empty() {
            std::process::exit(1);
        }
    }
}

fn describe_split(rule: Split) -> String {
    match rule {
        Split::Count(n) => format!("every {n} frames"),
        Split::Gap(g) => format!("a new stack at every pause over {g} s"),
        Split::Dir => "one per folder".into(),
    }
}

fn base(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// `--near-end`: frame 0 is the near end (`First`), the far end (`Last`), or
/// whatever the run's cues say, frame 0 near when nothing does (`Auto`).
#[derive(Clone, Copy, PartialEq)]
enum NearEnd { Auto, First, Last }

/// Everything the command line said, apart from the frames: applied to every stack of a batch.
struct Cfg {
    /// Output paths are templates when there is more than one stack (`batch::expand`).
    output: String,
    save_depth: bool,
    save_conf: bool,
    metadata: bool,
    depth_raw: Option<String>,
    p: Params,
    slab_dir: Option<String>,
    stereo: Option<(f32, Layout)>,
    rocking: Option<(f32, usize)>,
    /// `--video FPS`: the rocking views joined into an MP4 by ffmpeg.
    video: Option<f32>,
    /// `--near-end`: which end of the stack is near for the stereo, the rocking and the mesh.
    near_end: NearEnd,
    mesh_formats: Vec<String>,
    mp: MeshParams,
    mesh_tex: (usize, TexFormat),
    /// The scale bar and caption (`overlay.rs`) burned into the fused image, the weighted
    /// average and every stereo / rocking view; `overlay_auto` takes the calibration from the
    /// first frame's TIFF (`Meta::pixel_size_um`).
    overlay: Option<OverlayParams>,
    overlay_auto: bool,
}

/// Stack one set of frames and write everything asked for. `tag` goes into the
/// log prefix (` 3/7` in a batch).
fn run_stack(inputs: &[String], cfg: &Cfg, names: &Names<'_>, tag: &str) -> Result<(), String> {
    let t0 = Instant::now();
    let output = batch::expand(&cfg.output, names);
    let mut p = cfg.p.clone();
    p.save_aligned = cfg.p.save_aligned.as_deref().map(|d| batch::expand(d, names));
    let depth_raw = cfg.depth_raw.as_deref().map(|d| batch::expand(d, names));
    let mut log = |s: String| eprintln!("[lapstack{tag}] {s}");
    // the first frame's EXIF, ICC profile and XMP go into the fused image (and the slabs)
    let meta = if cfg.metadata { Some(io::load_meta(&inputs[0])?) } else { None };
    // slabs: written as they are fused, in the output's format, to --slab-dir [<output stem>_slabs]
    let stem = match output.rfind('.') { Some(k) => &output[..k], None => &output[..] };
    let ext = match output.rfind('.') { Some(k) => &output[k..], None => ".png" };
    let slab_dir = match &cfg.slab_dir { Some(d) => batch::expand(d, names), None => format!("{stem}_slabs") };
    let mut on_slab = |s: lapstack_core::Slab<'_>| -> Result<(), String> {
        if s.index == 0 {
            std::fs::create_dir_all(&slab_dir).map_err(|e| format!("cannot create {slab_dir}: {e}"))?;
        }
        let path = format!("{slab_dir}/slab_{:02}_{:03}-{:03}{ext}", s.index + 1, s.lo, s.hi);
        match &s.dng {
            Some(info) => io::save_dng(s.image, &path, info, meta.as_ref())?,
            None => io::save_rgb(s.image, &path, s.bit_depth, meta.as_ref())?,
        }
        eprintln!("[lapstack{tag}] slab {}/{} (frames {}..{}) -> {path}", s.index + 1, s.count, s.lo, s.hi);
        Ok(())
    };
    let mut out = run_with(inputs, &p, &mut log, &mut on_slab)?;
    if let Some(m) = &meta {
        eprintln!("[lapstack{tag}] metadata from {}: {}", inputs[0], m.describe());
    }
    if let Some(info) = &out.dng {
        eprintln!("[lapstack{tag}] {}", info.describe());
    }
    if out.reduce > 0 {
        eprintln!("[lapstack{tag}] draft: the frames reduced by {}, the outputs {}x{}", 1 << out.reduce, out.image.w, out.image.h);
    }
    // the images: the fused result, the weighted average and the stereo pair go out as a linear
    // DNG when the output asks for one (the views of a rocking as TIFFs: a video is made of them)
    let dng = out.dng.clone();
    let bit_depth = out.bit_depth;
    let save = |img: &Img3, path: &str| -> Result<(), String> {
        match &dng {
            Some(info) if io::is_dng(path) => io::save_dng(img, path, info, meta.as_ref()),
            _ => io::save_rgb(img, path, bit_depth, meta.as_ref()),
        }
    };
    let view_ext = if io::is_dng(ext) { ".tif" } else { ext };
    // the scale bar and caption: rendered once for the output's size, burned into the fused
    // image, the weighted average and each view (the clean image is kept for the shears and
    // the 3D texture — a bar sheared by the depth map would bend)
    let overlay = match &cfg.overlay {
        Some(o) => {
            let mut o = o.clone();
            let needs_meta = cfg.overlay_auto || o.text.contains("{date}") || o.text.contains("{time}");
            let own = if meta.is_none() && needs_meta { Some(io::load_meta(&inputs[0])?) } else { None };
            let m = meta.as_ref().or(own.as_ref());
            if cfg.overlay_auto {
                let (um, src) = m.and_then(|m| m.pixel_size_um()).ok_or_else(|| format!("--scale-bar auto: {} carries no pixel size (ImageJ's unit= with XResolution, OME-XML's PhysicalSizeX, or a TIFF resolution in cm or inch from a writer that is not a camera); give the calibration in µm per pixel", inputs[0]))?;
                eprintln!("[lapstack{tag}] pixel size {um} µm, from the first frame's {src}");
                o.um_per_px = um;
            }
            // a draft's pixels are 2^N frame pixels wide
            o.um_per_px *= (1u32 << out.reduce) as f64;
            o.text = overlay::expand_text(&o.text, m.and_then(|m| m.capture_time()), inputs.len(), names.first, names.n);
            let ov = Overlay::render(&o, out.image.w, out.image.h, 1.0);
            eprintln!("[lapstack{tag}] overlay: {}", ov.describe());
            Some(ov)
        }
        None => None,
    };
    // a template names a folder that may not exist yet
    if let Some(dir) = std::path::Path::new(&output).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let under = overlay.as_ref().map(|ov| ov.under_f32(&out.image));
    if let Some(ov) = &overlay {
        ov.apply_f32(&mut out.image);
    }
    save(&out.image, &output)?;
    eprintln!("[lapstack{tag}] fused -> {output}");
    if let (Some(ov), Some(u)) = (&overlay, &under) {
        ov.restore_f32(&mut out.image, u);
    }
    if let Some(wav) = &mut out.wav {
        if let Some(ov) = &overlay {
            ov.apply_f32(wav);
        }
        let path = format!("{stem}_wav{ext}");
        save(wav, &path)?;
        eprintln!("[lapstack{tag}] weighted average -> {path}");
    }
    if cfg.save_depth {
        let dp = format!("{stem}_depth.png");
        io::save_gray(&out.depth, out.image.w, out.image.h, &dp)?;
        match &p.depth {
            Some(_) => eprintln!("[lapstack{tag}] depth map (depth from focus) -> {dp}"),
            None => eprintln!("[lapstack{tag}] depth map (level-{} winners) -> {dp}", p.fuse.depth_level.min(out.levels - 1)),
        }
    }
    let n_last = (inputs.len().max(2) - 1) as f32;
    // synthetic stereo and rocking (view.rs): the result sheared by its depth map, plane by plane
    let (iw, ih) = (out.image.w, out.image.h);
    let near_first = match cfg.near_end {
        NearEnd::First => true,
        NearEnd::Last => false,
        NearEnd::Auto => out.near.as_ref().map_or(true, |(nf, _)| *nf),
    };
    if cfg.stereo.is_some() || cfg.rocking.is_some() || !cfg.mesh_formats.is_empty() {
        eprintln!("[lapstack{tag}] frame 0 is taken as the {} end{}", if near_first { "near" } else { "far" }, match (cfg.near_end, &out.near) { (NearEnd::Auto, Some(_)) => " (from the frames' focus distances)", (NearEnd::Auto, None) => " (nothing in the frames says otherwise; --near-end last if the relief looks inside out)", _ => " (--near-end)" });
    }
    let sheared = |v: &View| -> Img3 {
        let mut img = Img3::zeros(iw, ih);
        for c in 0..3 {
            view::render(&out.image.p[c], iw, ih, 1, &out.depth, 1.0 / n_last, v, &mut img.p[c], iw, 0);
        }
        if let Some(ov) = &overlay {
            ov.apply_f32(&mut img);
        }
        img
    };
    if let Some((shift, layout)) = cfg.stereo {
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
        save(&pair, &path)?;
        eprintln!("[lapstack{tag}] stereo pair (±{:.1} %, {}) -> {path}", shift * 100.0, match layout { Layout::SideBySide => "left | right", Layout::CrossEyed => "right | left", Layout::Anaglyph => "red-cyan anaglyph" });
    }
    if let Some((amp, n)) = cfg.rocking {
        let dir = format!("{stem}_rocking");
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {dir}: {e}"))?;
        for (i, s) in view::rocking_shifts(amp, n).into_iter().enumerate() {
            let path = format!("{dir}/view_{i:02}{view_ext}");
            io::save_rgb(&sheared(&View::new(s, near_first)), &path, out.bit_depth, meta.as_ref())?;
        }
        eprintln!("[lapstack{tag}] rocking: {n} views of ±{:.1} % -> {dir}/view_NN{view_ext}", amp * 100.0);
        // the views as one MP4, H.264 at crf 18, by ffmpeg when it is on the path (the size made
        // even for its 4:2:0 chroma); one sine cycle, so a player's loop is seamless
        if let Some(fps) = cfg.video {
            let out = format!("{stem}_rocking.mp4");
            let fps_s = format!("{fps}");
            let pattern = format!("{dir}/view_%02d{view_ext}");
            let args = ["-y", "-loglevel", "error", "-framerate", &fps_s, "-i", &pattern, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", &out];
            match std::process::Command::new("ffmpeg").args(args).status() {
                Ok(s) if s.success() => eprintln!("[lapstack{tag}] rocking video: {n} views at {fps} fps, H.264 -> {out}"),
                Ok(s) => eprintln!("[lapstack{tag}] ffmpeg failed ({s}); the views are in {dir}"),
                Err(e) => eprintln!("[lapstack{tag}] ffmpeg not run ({e}); the views are in {dir} — join them with: ffmpeg {}", args.join(" ")),
            }
        }
    }
    if !cfg.mesh_formats.is_empty() {
        // the 3D model (mesh.rs): the depth map as a relief textured with the result
        let mut mp = cfg.mp;
        mp.near_first = near_first;
        let (tex_edge, tex_fmt) = cfg.mesh_tex;
        let m = mesh::heightfield(&out.depth, iw, &lapstack_core::align::Rect::full(iw, ih), 1.0 / n_last, &mp);
        let (tex, tw, th) = mesh::texture_planes([&out.image.p[0], &out.image.p[1], &out.image.p[2]], iw, ih, tex_edge);
        let write = |path: &str, bytes: &[u8]| std::fs::write(path, bytes).map_err(|e| format!("cannot write {path}: {e}"));
        eprintln!("[lapstack{tag}] 3D model: {}x{} vertices, {} triangles, relief {:.0} % of the width, texture {tw}x{th}", m.nx, m.ny, m.triangles(), mp.relief * 100.0);
        let image = if cfg.mesh_formats.iter().any(|f| f != "stl") { mesh::encode_texture(&tex, tw, th, tex_fmt)? } else { Vec::new() };
        for f in &cfg.mesh_formats {
            match f.as_str() {
                "glb" => {
                    let path = format!("{stem}.glb");
                    write(&path, &mesh::glb(&m, &image, tex_fmt.mime()))?;
                    eprintln!("[lapstack{tag}] glTF binary -> {path}");
                }
                "obj" => {
                    let base = stem.rsplit('/').next().unwrap_or(stem);
                    let (mtl_name, tex_name) = (format!("{base}.mtl"), format!("{base}_texture.{}", tex_fmt.ext()));
                    write(&format!("{stem}.obj"), &mesh::obj(&m, &mtl_name))?;
                    write(&format!("{stem}.mtl"), &mesh::mtl(&tex_name))?;
                    write(&format!("{stem}_texture.{}", tex_fmt.ext()), &image)?;
                    eprintln!("[lapstack{tag}] Wavefront OBJ -> {stem}.obj + {mtl_name} + {tex_name}");
                }
                _ => {
                    let path = format!("{stem}.stl");
                    write(&path, &mesh::stl(&m))?;
                    eprintln!("[lapstack{tag}] binary STL -> {path}");
                }
            }
        }
    }
    if let Some(path) = &depth_raw {
        io::save_gray16(&out.depth, out.image.w, out.image.h, n_last, path)?;
        eprintln!("[lapstack{tag}] depth (16-bit, 65535 = frame {}) -> {path}", inputs.len() - 1);
    }
    if cfg.save_conf {
        match &out.conf {
            Some(c) => {
                let cp = format!("{stem}_conf.png");
                io::save_gray16(c, out.image.w, out.image.h, 1.0, &cp)?;
                eprintln!("[lapstack{tag}] confidence -> {cp}");
            }
            None => eprintln!("[lapstack{tag}] --save-conf: no confidence map with --depth winner"),
        }
    }
    eprintln!("[lapstack{tag}] total {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn help() {
    eprintln!(
        "lapstack — Laplacian-pyramid focus stacking (Burt & Adelson; Wang & Chang 2011)\n\
         Usage: lapstack [options] frame1 frame2 ...   (a directory stands for the image files in it)\n\
         Input: PNG, JPEG, TIFF, or a camera raw (NEF, CR2/CR3, ARW, DNG, RAF, ORF, RW2, PEF, …: developed as shot, no exposure adjustment).\n\
         Output keeps the input bit depth (16-bit needs PNG/TIFF; a raw counts as 16-bit).\n\
           -o, --output PATH      fused image [stacked.png]; a .dng writes a linear DNG (below)\n\
           --levels N             band-pass levels [auto: residual short side >= 32 px]\n\
           --energy-radius R      region-energy window radius, binomial weights [1 = 3x3];\n\
                                  0 = per-node |L| max (Adelson et al. 1984)\n\
           --top RULE             residual rule: de (deviation+entropy, paper) | dev | avg [de]\n\
           --top-radius R         D/E window radius at the residual [2 = 5x5]\n\
           --entropy-bins N       gray levels for the entropy histogram [256]\n\
           --use-chroma           energy from R,G,B instead of luma\n\
           --halo-control P       the levels coarser than --depth-level do not pick their own winners: each is the\n\
                                  mean of the frames weighed by that level's region energy to the power P, REDUCEd\n\
                                  to its size, so the coarse structure follows the frames found sharp there and a\n\
                                  defocused copy of a bright object no longer wins the coarse levels beside it\n\
                                  (its halo); 1 = weigh by the energy, higher = closer to a hard pick, 0 = off [0]\n\
           --no-align             frames are already registered (streams from disk)\n\
           --no-shift/scale/rotation   restrict the similarity model\n\
           --align-coarsen N      align at reduced resolution (skip N finest levels)\n\
           --align-model M        similarity (shift, scale, rotation) | affine (+ aspect,\n\
                                  shear) | projective (+ perspective, for a camera that tilted as it stepped) [similarity]\n\
           --interpolation K      the kernel the aligned frames are resampled with: nearest | bilinear | bicubic |\n\
                                  spline4x4 | spline6x6 | lanczos3 [spline4x4]; wider = sharper,\n\
                                  more ringing at hard edges (the registration search itself always uses spline4x4)\n\
           --save-aligned DIR     write the aligned frames\n\
           --save-depth           write the depth map next to the output (8-bit, min-max scaled)\n\
           --depth-raw PATH       write the depth map as 16-bit PNG, fixed scale (65535 = last frame)\n\
           --save-conf            write the depth confidence map (16-bit, 65535 = 1)\n\
           --no-metadata          do not copy the first frame's EXIF / ICC profile / XMP into the output\n\
           --no-crop              keep the full frame instead of cropping to the area every aligned frame covers\n\
           --no-brightness        do not equalise the frames' brightness to frame 0 (exposure flicker)\n\
           --slabs SIZE[:OVERLAP] also fuse slabs of SIZE consecutive frames, overlapping by OVERLAP [2],\n\
                                  each on its own: thick planes of focus to retouch from\n\
           --slab-dir DIR         where the slabs go, in the output's format [<output stem>_slabs]\n\
           --wav                  also the weighted average: every frame weighed by its\n\
                                  local contrast (the depth pass's focus measure) above the cell's noise floor\n\
                                  -> <stem>_wav.<ext>; no seams or halos, flat areas average (less noise),\n\
                                  softer than the pyramid\n\
           --wav-power P          the contrast above the floor raised to P before weighing [2]; 1 = plain,\n\
                                  higher = keener\n\
           --wav-smooth R         box radius the contrast, then the weights, are smoothed by on the depth\n\
                                  pass's grid [3]: a region's pick, a cross-fade along depth edges; 0 = none\n\
           --wav-gate G           a contrast counts above (1+G) x the cell's noise floor [0.5]; 0 = the floor\n\
           --split RULE           batch: cut the frames into stacks and run each — count:N (every N frames),\n\
                                  gap:SECONDS (a new stack at every pause in the capture times longer than\n\
                                  that), dir (one stack per folder); then -o, --slab-dir, --save-aligned and\n\
                                  --depth-raw are templates: {{n}} the stack's number, {{first}} its first frame's\n\
                                  stem, {{dir}} its folder; a path with no field gets _NN before its extension\n\
           --dry-run              list the stacks and their output names, then stop\n\
           --rotate DEG           turn every frame by 90, 180 or 270 degrees clockwise as decoded (a camera held\n\
                                  sideways; a raw is already turned by its EXIF orientation)\n\
           --draft N              a draft run: every frame block-averaged by 2^N as decoded, so the whole run\n\
                                  (alignment, fusion, depth, every output) is a quick check of the settings at\n\
                                  1/2^N the size; --crop and the scale bar follow the frame's own pixels [0]\n\
           --crop X,Y,W,H         cut every output to this window of the (turned) frame, in full-resolution pixels,\n\
                                  on top of the automatic crop to the area every frame covers\n\
           --skip LIST            leave frames out: 1-based positions and ranges in the frame list, after\n\
                                  directories are expanded and before any split (3,7-9,12)\n\
           --reverse              reverse the frame order (a stack shot back to front); each stack of a\n\
                                  batch on its own, so frame 0 is the near end again for --stereo / --mesh\n\
           --dust-map FILE        dust map: a frame of an evenly lit blank surface shot out of focus\n\
                                  at the stack's aperture; the dust spots found in it are taken out of every\n\
                                  frame before alignment (each interpolated from its surroundings)\n\
           --dust-threshold PCT   a pixel darker than its background by more than this is dust [3]\n\
           --dust-margin PX       pixels the spots are grown by [3]\n\
           --dust-mode MODE       fill = interpolate each spot (default) | flat = divide it by the map's attenuation\n\
                                  (keeps the detail under the spot; needs the map shot at the stack's aperture)\n\
           --save-dust-map PATH   write the spots found as an image (white = dust), to check the map\n\
           --stereo PCT[:LAYOUT]  synthetic stereo pair -> <stem>_stereo.<ext>: the result sheared by its depth map,\n\
                                  the far end of the stack moved -PCT / +PCT % of the width (left / right view);\n\
                                  LAYOUT sbs (left | right, default) | cross (right | left) | anaglyph (red-cyan)\n\
           --rocking PCT[:N]      rocking animation: N [24] views, the shift sweeping +-PCT % in one sine cycle,\n\
                                  -> <stem>_rocking/view_NN.<ext> (join them with ffmpeg / ImageMagick)\n\
           --video FPS            with --rocking: the views joined into <stem>_rocking.mp4 (H.264, crf 18) at FPS,\n\
                                  by ffmpeg if it is on the path\n\
           --near-end E           which end of the stack frame 0 is, for --stereo, --rocking and --mesh: first (near),\n\
                                  last (far; --far-first is the same), or auto = the focus distances the camera wrote\n\
                                  into the first and last frames (EXIF SubjectDistance) decide, frame 0 near when\n\
                                  there are none [auto]\n\
           --mesh FORMATS         3D model: the depth map as a relief textured with the result, as\n\
                                  glb (glTF binary, one file) | obj (+ .mtl + texture file) | stl (geometry only),\n\
                                  comma-separated -> <stem>.glb / .obj / .stl; --far-first applies\n\
           --mesh-relief PCT      the depth of the stack as a percentage of the image width [25]\n\
           --mesh-grid N          vertices along the long edge [1000]\n\
           --mesh-texture EDGE[:jpeg[:Q] | png]   the texture's long edge, 0 = full [8192], and format [jpeg:92]\n\
         Linear DNG (-o stacked.dng; raw in, DNG out): the raws are developed to the camera's own linear\n\
         space (no white balance, matrix or curve baked in), fused in the look that frame 0's white balance and\n\
         matrix give, and the result is written back in camera space with the camera's colour matrices, so a\n\
         raw converter develops the stack like a raw, highlights past white and all; frames that are not raws\n\
         are taken as sRGB and written as a linear sRGB DNG. The slabs, the weighted average and the stereo pair\n\
         are DNGs too; the rocking views are TIFFs; a stack must be all raws or none.\n\
         Scale bar and caption (microscopy), burned into the fused image, the weighted average and the views:\n\
           --scale-bar CAL[:LENGTH]   CAL = the size of one pixel of the frames in µm (0.325, or 325nm), auto = read\n\
                                  from the first frame's TIFF (ImageJ's unit=, OME-XML's PhysicalSizeX, a resolution in\n\
                                  cm or inch from a writer that is not a camera), or px = no calibration, a bar labelled\n\
                                  in pixels of the frames; LENGTH = the bar's length with a unit (100um, 2mm, 500nm; in\n\
                                  pixels after px) [auto: the 1-2-5 value nearest a fifth of the width]; the label picks\n\
                                  its unit (500 nm, 100 µm, 2.5 mm)\n\
           --text TEXT            a caption; \\n breaks a line; {{date}} {{time}} (the first frame's capture time),\n\
                                  {{frames}}, {{first}} (its stem), {{n}} (the stack in a batch)\n\
           --overlay-pos BAR[,TEXT]   the corners, tl | tr | bl | br [br,bl]; in one corner the text goes above the bar\n\
           --overlay-size PCT     the font size as % of the image height; everything else scales with it [3]\n\
           --overlay-color C      white | black [white]\n\
           --overlay-style S      halo (a thin outline in the other colour) | box (a translucent box behind) | plain [halo]\n\
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

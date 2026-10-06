// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

//! The browser app's project file (`<name>.lapstack.json`, written by
//! `web/app.js`'s `projectData`) read for the command line: its settings
//! become the flags the CLI would take for them, its frames the inputs, so a
//! stack dialed in by eye in the app runs the same way from a script
//! (`lapstack --config shoot.lapstack.json`). The translation is a table of
//! the panel's keys to flags, and the app's *Copy command* button writes the
//! same flags from the same keys (`cliCommand` in `app.js`): the two must be
//! kept in step. What the CLI has no flag for (the depth-map rendering, the
//! retouch, the animations, the content credentials) is left out and noted.

use serde_json::Value;
use std::path::Path;

/// What a project file gives the command line.
#[derive(Debug, Default)]
pub struct ProjectArgs {
    /// The flags, in the order the panel lists them; the command line's own
    /// flags come after and override.
    pub flags: Vec<String>,
    /// The frames in the run (the excluded ones left out), resolved against
    /// the project file's folder where they exist there.
    pub frames: Vec<String>,
    /// The project's name (`-o` is made from it when the command line gives none).
    pub name: Option<String>,
    /// What was left out or could not be found.
    pub notes: Vec<String>,
}

/// Read a project file into flags and frames.
pub fn read(path: &str) -> Result<ProjectArgs, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("--config {path}: {e}"))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("--config {path}: not JSON ({e})"))?;
    if v.get("lapstack_project").and_then(Value::as_i64) != Some(1) {
        return Err(format!("--config {path}: not a lapstack project file (no lapstack_project: 1)"));
    }
    let dir = Path::new(path).parent().map(Path::to_path_buf).unwrap_or_default();
    let p = v.get("params").cloned().unwrap_or(Value::Null);
    let sv = v.get("save").cloned().unwrap_or(Value::Null);
    let mut out = ProjectArgs { name: v.get("name").and_then(Value::as_str).map(str::to_string), ..Default::default() };
    let b = |k: &str| p.get(k).and_then(Value::as_bool);
    let n = |k: &str| p.get(k).and_then(Value::as_f64);
    let s = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
    let fmt = |x: f64| {
        let t = format!("{x}");
        if t.contains('.') { t.trim_end_matches('0').trim_end_matches('.').to_string() } else { t }
    };
    let mut flag = |f: &str, val: Option<String>| {
        out.flags.push(f.to_string());
        if let Some(v) = val {
            out.flags.push(v);
        }
    };
    // the output: the project's name, the Save step's format
    if let Some(name) = &out.name {
        let ext = match sv.get("sv-format").and_then(Value::as_str) {
            Some("jpeg") => "jpg",
            Some("dng") => "dng",
            _ => "png",
        };
        flag("-o", Some(format!("{name}_stacked.{ext}")));
    }
    // alignment
    if b("align") == Some(false) {
        flag("--no-align", None);
    }
    if b("shift") == Some(false) {
        flag("--no-shift", None);
    }
    if b("scale") == Some(false) {
        flag("--no-scale", None);
    }
    if b("rotation") == Some(false) {
        flag("--no-rotation", None);
    }
    if b("brightness") == Some(false) {
        flag("--no-brightness", None);
    }
    if let Some(c) = n("coarsen") {
        flag("--align-coarsen", Some(fmt(c)));
    }
    if let Some(k) = s("interp") {
        flag("--interpolation", Some(k));
    }
    if let Some(m) = s("model") {
        flag("--align-model", Some(m));
    }
    if let Some(r) = n("rotate").filter(|r| *r > 0.0) {
        flag("--rotate", Some(fmt(r)));
    }
    if let Some(d) = n("draft").filter(|d| *d > 0.0) {
        flag("--draft", Some(fmt(d)));
    }
    // fusion
    if let Some(l) = n("levels").filter(|l| *l > 0.0) {
        flag("--levels", Some(fmt(l)));
    }
    if let Some(r) = n("energy_radius") {
        flag("--energy-radius", Some(fmt(r)));
    }
    if let Some(t) = s("top") {
        flag("--top", Some(t));
    }
    if let Some(r) = n("top_radius") {
        flag("--top-radius", Some(fmt(r)));
    }
    if b("use_chroma") == Some(true) {
        flag("--use-chroma", None);
    }
    if let Some(h) = n("halo").filter(|h| *h > 0.0) {
        flag("--halo-control", Some(fmt(h)));
    }
    // depth
    if let Some(d) = n("depth_scale") {
        flag("--depth-scale", Some(fmt(d)));
    }
    if let Some(d) = n("depth_level") {
        flag("--depth-level", Some(fmt(d)));
    }
    // the weighted average
    if b("render_wav") == Some(true) {
        flag("--wav", None);
        if let Some(x) = n("wav_power") {
            flag("--wav-power", Some(fmt(x)));
        }
        if let Some(x) = n("wav_smooth") {
            flag("--wav-smooth", Some(fmt(x)));
        }
        if let Some(x) = n("wav_gate") {
            flag("--wav-gate", Some(fmt(x)));
        }
        if b("wav_edge") == Some(false) {
            flag("--wav-box", None);
        }
    }
    if b("render_dmap") == Some(true) {
        out.notes.push("the depth-map rendering (DFR) is the app's own; the command line makes the pyramid result (and --wav)".into());
    }
    // the batch split
    match s("split").as_deref() {
        Some("count") => flag("--split", Some(format!("count:{}", fmt(n("split_n").unwrap_or(30.0))))),
        Some("gap") => flag("--split", Some(format!("gap:{}", fmt(n("split_gap").unwrap_or(10.0))))),
        Some("dir") => flag("--split", Some("dir".into())),
        _ => {}
    }
    // the dust map: the file the project names, looked for beside the project file
    if let Some(name) = v.get("dust").and_then(|d| d.get("name")).and_then(Value::as_str) {
        let here = dir.join(name);
        if here.is_file() {
            flag("--dust-map", Some(here.to_string_lossy().into_owned()));
            if let Some(x) = n("dust_thr") {
                flag("--dust-threshold", Some(fmt(x)));
            }
            if let Some(x) = n("dust_margin") {
                flag("--dust-margin", Some(fmt(x)));
            }
            if let Some(m) = s("dust_mode") {
                flag("--dust-mode", Some(m));
            }
        } else {
            out.notes.push(format!("the dust map {name} is not beside the project file: give it with --dust-map"));
        }
    }
    // the scale bar and caption
    if b("ov_bar") == Some(true) {
        let um = n("ov_um").unwrap_or(0.0);
        let len = s("ov_len").unwrap_or_default();
        let cal = if um > 0.0 { fmt(um) } else { "px".to_string() };
        flag("--scale-bar", Some(if len.trim().is_empty() { cal } else { format!("{cal}:{}", len.trim()) }));
    }
    if let Some(t) = s("ov_text").filter(|t| !t.is_empty()) {
        flag("--text", Some(t));
    }
    if b("ov_bar") == Some(true) || s("ov_text").is_some_and(|t| !t.is_empty()) {
        if let (Some(bp), Some(tp)) = (s("ov_bar_pos"), s("ov_text_pos")) {
            flag("--overlay-pos", Some(format!("{bp},{tp}")));
        }
        if let Some(x) = n("ov_size") {
            flag("--overlay-size", Some(fmt(x)));
        }
        if let Some(c) = s("ov_color") {
            flag("--overlay-color", Some(c));
        }
        if let Some(c) = s("ov_style") {
            flag("--overlay-style", Some(c));
        }
    }
    // the run's crop window
    if let Some(c) = v.get("run").and_then(|r| r.get("crop")).filter(|c| c.is_object()) {
        let g = |k: &str| c.get(k).and_then(Value::as_f64).map(|x| x.round() as i64);
        if let (Some(x), Some(y), Some(w), Some(h)) = (g("x"), g("y"), g("w"), g("h")) {
            flag("--crop", Some(format!("{x},{y},{w},{h}")));
        }
    }
    // the Save step
    let sb = |k: &str| sv.get(k).and_then(Value::as_bool);
    let ss = |k: &str| sv.get(k).and_then(Value::as_str).map(str::to_string);
    let sn = |k: &str| sv.get(k).and_then(Value::as_f64).or_else(|| sv.get(k).and_then(Value::as_str).and_then(|t| t.parse().ok()));
    if sb("sv-meta") == Some(false) {
        flag("--no-metadata", None);
    }
    if sb("sv-crop") == Some(false) {
        flag("--no-crop", None);
    }
    match ss("v3-near").as_deref() {
        Some("first") => flag("--near-end", Some("first".into())),
        Some("last") => flag("--near-end", Some("last".into())),
        _ => {}
    }
    let sel: Vec<String> = sv.get("sel").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    let has = |id: &str| sel.iter().any(|s| s == id);
    if has("depth") || has("depth16") {
        flag("--save-depth", None);
    }
    if has("conf") {
        flag("--save-conf", None);
    }
    if has("stereo") {
        let shift = sn("v3-shift").unwrap_or(3.0);
        let layout = ss("v3-layout").unwrap_or_else(|| "sbs".into());
        flag("--stereo", Some(format!("{}:{layout}", fmt(shift))));
    }
    if has("anim-rock") {
        let shift = sn("v3-rock").unwrap_or(3.0);
        let views = sn("v3-views").unwrap_or(24.0);
        flag("--rocking", Some(format!("{}:{}", fmt(shift), fmt(views))));
    }
    if has("mesh") {
        flag("--mesh", Some(ss("m3-format").unwrap_or_else(|| "glb".into())));
        if let Some(x) = sn("m3-relief") {
            flag("--mesh-relief", Some(fmt(x)));
        }
        if let Some(x) = sn("m3-grid") {
            flag("--mesh-grid", Some(fmt(x)));
        }
        if let Some(x) = sn("m3-tex").filter(|x| *x > 0.0) {
            flag("--mesh-texture", Some(fmt(x)));
        }
    }
    let left: Vec<&str> = ["dfr", "winner", "anim-depth", "anim-focus", "anim-peak"].into_iter().filter(|id| has(id)).collect();
    if !left.is_empty() {
        out.notes.push(format!("the app's outputs {} have no command-line form and are left out", left.join(", ")));
    }
    // the frames: the project's list, the excluded ones left out, each looked for beside the project file
    let mut missing = 0;
    for f in v.get("frames").and_then(Value::as_array).into_iter().flatten() {
        if f.get("off").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        let rel = f.get("path").and_then(Value::as_str).unwrap_or(name);
        let cands = [dir.join(rel), dir.join(name), Path::new(rel).to_path_buf()];
        match cands.iter().find(|c| c.is_file()) {
            Some(c) => out.frames.push(c.to_string_lossy().into_owned()),
            None => {
                missing += 1;
                out.frames.push(dir.join(rel).to_string_lossy().into_owned());
            }
        }
    }
    if missing > 0 {
        out.notes.push(format!("{missing} of the project's frames are not beside the project file (looked for their paths and names there): give the frames on the command line"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_becomes_flags_and_frames() {
        let dir = std::env::temp_dir().join(format!("lapstack-project-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("shoot")).unwrap();
        std::fs::write(dir.join("shoot/a.tif"), b"x").unwrap();
        std::fs::write(dir.join("shoot/b.tif"), b"x").unwrap();
        std::fs::write(dir.join("wall.tif"), b"x").unwrap();
        let json = r#"{"lapstack_project":1,"name":"fruit","frames":[{"name":"a.tif","path":"shoot/a.tif"},{"name":"b.tif","path":"shoot/b.tif","off":true},{"name":"c.tif","path":"shoot/c.tif"}],
          "params":{"align":true,"rotation":false,"coarsen":2,"interp":"spline4x4","model":"similarity","levels":null,"energy_radius":1,"top":"de","top_radius":2,"halo":0,"draft":1,"rotate":90,
                    "depth_scale":2,"depth_level":2,"render_wav":true,"wav_power":2,"wav_smooth":3,"wav_gate":0.5,"wav_edge":false,"split":"gap","split_gap":12,
                    "dust_thr":3,"dust_margin":3,"dust_mode":"fill","ov_bar":true,"ov_um":0.325,"ov_len":"100um","ov_text":"","ov_bar_pos":"br","ov_text_pos":"bl","ov_size":3,"ov_color":"white","ov_style":"halo"},
          "save":{"sel":["lap","depth","stereo","mesh","dfr"],"sv-format":"jpeg","sv-meta":true,"sv-crop":false,"v3-near":"last","v3-shift":"2.5","v3-layout":"cross","m3-format":"obj","m3-relief":"25","m3-grid":"1000","m3-tex":"4096"},
          "run":{"crop":{"x":10,"y":20,"w":300,"h":200}},"dust":{"name":"wall.tif"}}"#;
        let path = dir.join("fruit.lapstack.json");
        std::fs::write(&path, json).unwrap();
        let pa = read(path.to_str().unwrap()).unwrap();
        let f = pa.flags.join(" ");
        for want in ["-o fruit_stacked.jpg", "--no-rotation", "--align-coarsen 2", "--rotate 90", "--draft 1", "--wav --wav-power 2 --wav-smooth 3 --wav-gate 0.5 --wav-box", "--split gap:12", "--dust-threshold 3", "--scale-bar 0.325:100um", "--overlay-pos br,bl", "--crop 10,20,300,200", "--no-crop", "--near-end last", "--save-depth", "--stereo 2.5:cross", "--mesh obj --mesh-relief 25 --mesh-grid 1000 --mesh-texture 4096"] {
            assert!(f.contains(want), "missing {want} in {f}");
        }
        assert!(!f.contains("--levels"), "{f}");
        assert!(f.contains(&format!("--dust-map {}", dir.join("wall.tif").display())), "{f}");
        assert_eq!(pa.frames.len(), 2, "{:?}", pa.frames);
        assert!(pa.frames[0].ends_with("shoot/a.tif"));
        assert!(pa.notes.iter().any(|n| n.contains("1 of the project's frames")), "{:?}", pa.notes);
        assert!(pa.notes.iter().any(|n| n.contains("dfr")), "{:?}", pa.notes);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

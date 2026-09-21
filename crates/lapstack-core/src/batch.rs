// Copyright (c) 2026 RAGTUX LLC
// INTERNAL USE ONLY

//! Batch runs: a list of frames cut into stacks — every N frames, at every
//! pause in the capture times, or by folder — and the output names each stack
//! gets. The stacking itself is `stack::run` once per stack; nothing here
//! touches pixels.

use std::cmp::Ordering;
use std::path::Path;

/// How a list of frames is cut into stacks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Split {
    /// Every `n` frames.
    Count(usize),
    /// A new stack starts where the capture time jumps by more than this many seconds.
    Gap(f64),
    /// Each folder is a stack.
    Dir,
}

impl Split {
    /// `count:N` | `gap:SECONDS` | `dir`.
    pub fn parse(s: &str) -> Option<Split> {
        let (kind, arg) = match s.split_once(':') { Some((k, a)) => (k, Some(a)), None => (s, None) };
        match (kind, arg) {
            ("count", Some(n)) => n.parse().ok().filter(|&n| n >= 1).map(Split::Count),
            ("gap", Some(g)) => g.parse().ok().filter(|&g: &f64| g > 0.0 && g.is_finite()).map(Split::Gap),
            ("dir", None) => Some(Split::Dir),
            _ => None,
        }
    }
}

/// One stack of a batch: its frames in order, and, when the split read the
/// capture times, the first and last frame's.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub inputs: Vec<String>,
    pub times: Option<(f64, f64)>,
}

impl Stack {
    /// The first frame's file stem: `DSC_0412` of `shoot/a/DSC_0412.tif`.
    pub fn first(&self) -> &str {
        stem(&self.inputs[0])
    }
    /// The first frame's folder name: `a` of `shoot/a/DSC_0412.tif` (`.` at the top).
    pub fn dir(&self) -> String {
        dir_name(&self.inputs[0])
    }
}

pub fn stem(path: &str) -> &str {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match name.rfind('.') { Some(k) if k > 0 => &name[..k], _ => name }
}

pub fn dir_name(path: &str) -> String {
    let p = Path::new(path);
    match p.parent().and_then(|d| d.file_name()) {
        Some(n) => n.to_string_lossy().into_owned(),
        None => match p.parent() { Some(d) if !d.as_os_str().is_empty() => d.to_string_lossy().into_owned(), _ => ".".into() },
    }
}

/// Is this a file lapstack can read, by extension?
pub fn is_image(name: &str) -> bool {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff")
}

/// Natural order: runs of digits compare as numbers (`f2 < f10`), the rest as bytes.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let (i0, j0) = (i, j);
            while i < a.len() && a[i].is_ascii_digit() { i += 1; }
            while j < b.len() && b[j].is_ascii_digit() { j += 1; }
            let (x, y) = (&a[i0..i], &b[j0..j]);
            let (x, y) = (trim_zeros(x), trim_zeros(y));
            let c = x.len().cmp(&y.len()).then_with(|| x.cmp(y));
            if c != Ordering::Equal { return c; }
        } else {
            let c = a[i].cmp(&b[j]);
            if c != Ordering::Equal { return c; }
            i += 1; j += 1;
        }
    }
    (a.len() - i).cmp(&(b.len() - j))
}

fn trim_zeros(d: &[u8]) -> &[u8] {
    let k = d.iter().position(|&c| c != b'0').unwrap_or(d.len());
    &d[k..]
}

/// The input list with every directory replaced by the image files in it, in
/// natural order (`f2` before `f10`); other entries stay where they are. A
/// directory without images is an error — a mistyped path should not stack
/// the wrong frames.
pub fn expand_dirs(inputs: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(inputs.len());
    for p in inputs {
        if !Path::new(p).is_dir() {
            out.push(p.clone());
            continue;
        }
        let rd = std::fs::read_dir(p).map_err(|e| format!("cannot read {p}: {e}"))?;
        let mut files: Vec<String> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| !t.is_dir()).unwrap_or(false))
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|f| is_image(f))
            .collect();
        if files.is_empty() {
            return Err(format!("{p}: no PNG, JPEG or TIFF files in this directory"));
        }
        files.sort_by(|a, b| natural_cmp(a, b));
        out.extend(files);
    }
    Ok(out)
}

/// The frames a `--skip` list leaves out of a list of `n`: 1-based positions
/// and ranges, comma-separated (`3,7-9,12`), returned as sorted 0-based
/// indices. A position past the end or a backwards range is an error: a typo
/// should not stack the wrong frames.
pub fn skip_list(spec: &str, n: usize) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim(), b.trim()),
            None => (part, part),
        };
        let pos = |s: &str| s.parse::<usize>().ok().filter(|&v| v >= 1).ok_or_else(|| format!("--skip: '{part}' is not a frame position (1-based) or range a-b"));
        let (a, b) = (pos(a)?, pos(b)?);
        if b < a {
            return Err(format!("--skip: '{part}' runs backwards"));
        }
        if b > n {
            return Err(format!("--skip: '{part}' is past the end ({n} frames)"));
        }
        out.extend(a - 1..b);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The list without the frames at `skip` (sorted 0-based indices).
pub fn without(inputs: &[String], skip: &[usize]) -> Vec<String> {
    inputs.iter().enumerate().filter(|(i, _)| skip.binary_search(i).is_err()).map(|(_, p)| p.clone()).collect()
}

/// Every `n` frames (the last stack may be shorter).
pub fn split_count(inputs: &[String], n: usize) -> Vec<Stack> {
    inputs.chunks(n.max(1)).map(|c| Stack { inputs: c.to_vec(), times: None }).collect()
}

/// A new stack starts wherever the capture time steps by more than `gap`
/// seconds (backwards too: a rewound clock is a different session). `times`
/// are the frames' capture times, one per input.
pub fn split_gap(inputs: &[String], times: &[f64], gap: f64) -> Vec<Stack> {
    assert_eq!(inputs.len(), times.len());
    let mut out: Vec<Stack> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut t0 = 0.0;
    for (i, (p, &t)) in inputs.iter().zip(times).enumerate() {
        if i > 0 && (t - times[i - 1]).abs() > gap {
            out.push(Stack { inputs: std::mem::take(&mut cur), times: Some((t0, times[i - 1])) });
        }
        if cur.is_empty() { t0 = t; }
        cur.push(p.clone());
    }
    if !cur.is_empty() {
        out.push(Stack { inputs: cur, times: Some((t0, times[times.len() - 1])) });
    }
    out
}

/// One stack per folder, in the order the folders first appear; frames keep
/// their order within a folder.
pub fn split_dir(inputs: &[String]) -> Vec<Stack> {
    let mut out: Vec<(String, Stack)> = Vec::new();
    for p in inputs {
        let dir = Path::new(p).parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
        match out.iter_mut().find(|(d, _)| *d == dir) {
            Some((_, s)) => s.inputs.push(p.clone()),
            None => out.push((dir, Stack { inputs: vec![p.clone()], times: None })),
        }
    }
    out.into_iter().map(|(_, s)| s).collect()
}

/// The capture time of every frame (`io::load_capture_time`), read in
/// parallel. A frame without one takes its file's modification time, which
/// `note` reports — a copy may have reset it, so the split it gives is worth a
/// look with a dry run.
pub fn capture_times(inputs: &[String], note: &mut dyn FnMut(String)) -> Result<Vec<f64>, String> {
    use rayon::prelude::*;
    let read: Vec<Option<f64>> = inputs.par_iter().map(|p| crate::io::load_capture_time(p)).collect::<Result<_, _>>()?;
    let mut out = Vec::with_capacity(inputs.len());
    for (p, t) in inputs.iter().zip(read) {
        out.push(match t {
            Some(t) => t,
            None => {
                let m = std::fs::metadata(p).and_then(|m| m.modified()).map_err(|e| format!("cannot read {p}: {e}"))?;
                let t = m.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
                note(format!("{p}: no capture time in its metadata; using the file's modification time"));
                t
            }
        });
    }
    Ok(out)
}

/// `HH:MM:SS` of a capture time, for a listing.
pub fn clock(t: f64) -> String {
    let s = t.rem_euclid(86400.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// What a stack's output names are made of.
pub struct Names<'a> {
    /// 1-based index of the stack and how many there are.
    pub n: usize,
    pub count: usize,
    pub first: &'a str,
    pub dir: &'a str,
}

/// An output path for one stack of a batch: `{n}` is the stack's index (zero
/// padded to the count's width), `{first}` the first frame's stem, `{dir}` its
/// folder's name. A path with no field gets `_NN` before its extension when
/// there is more than one stack, so `out.tif` becomes `out_01.tif`, and a
/// single stack keeps its name as it is.
pub fn expand(template: &str, names: &Names<'_>) -> String {
    let width = names.count.max(1).to_string().len().max(2);
    let n = format!("{:0width$}", names.n, width = width);
    let has_field = ["{n}", "{first}", "{dir}"].iter().any(|f| template.contains(f));
    if has_field {
        return template.replace("{n}", &n).replace("{first}", names.first).replace("{dir}", names.dir);
    }
    if names.count < 2 {
        return template.to_string();
    }
    let slash = template.rfind(['/', '\\']).map(|k| k + 1).unwrap_or(0);
    match template[slash..].rfind('.') {
        Some(k) if k > 0 => format!("{}_{n}{}", &template[..slash + k], &template[slash + k..]),
        _ => format!("{template}_{n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn skip_lists() {
        assert_eq!(skip_list("3,7-9,12", 12), Ok(vec![2, 6, 7, 8, 11]));
        assert_eq!(skip_list("9-7, 1", 12).is_err(), true);
        assert_eq!(skip_list("13", 12).is_err(), true);
        assert_eq!(skip_list("0", 12).is_err(), true);
        assert_eq!(skip_list("a", 12).is_err(), true);
        assert_eq!(skip_list("2,2-3", 3), Ok(vec![1, 2]));
        assert_eq!(without(&v(&["a", "b", "c", "d"]), &[0, 2]), v(&["b", "d"]));
    }

    #[test]
    fn parse_rules() {
        assert_eq!(Split::parse("count:10"), Some(Split::Count(10)));
        assert_eq!(Split::parse("gap:5"), Some(Split::Gap(5.0)));
        assert_eq!(Split::parse("gap:2.5"), Some(Split::Gap(2.5)));
        assert_eq!(Split::parse("dir"), Some(Split::Dir));
        assert_eq!(Split::parse("count:0"), None);
        assert_eq!(Split::parse("gap:-1"), None);
        assert_eq!(Split::parse("dir:x"), None);
        assert_eq!(Split::parse("time:3"), None);
    }

    #[test]
    fn names_of_a_path() {
        assert_eq!(stem("shoot/a/DSC_0412.tif"), "DSC_0412");
        assert_eq!(stem("DSC_0412"), "DSC_0412");
        assert_eq!(stem(".hidden"), ".hidden");
        assert_eq!(dir_name("shoot/a/DSC_0412.tif"), "a");
        assert_eq!(dir_name("DSC_0412.tif"), ".");
        assert_eq!(dir_name("/x.tif"), "/");
    }

    #[test]
    fn natural_order() {
        let mut names = v(&["f10.tif", "f2.tif", "f1.tif", "f02.tif", "b.tif", "f2a.tif"]);
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(names, v(&["b.tif", "f1.tif", "f2.tif", "f02.tif", "f2a.tif", "f10.tif"]));
    }

    #[test]
    fn by_count() {
        let s = split_count(&v(&["a", "b", "c", "d", "e"]), 2);
        assert_eq!(s.iter().map(|s| s.inputs.len()).collect::<Vec<_>>(), [2, 2, 1]);
        assert_eq!(s[2].inputs, v(&["e"]));
        assert_eq!(split_count(&v(&["a", "b"]), 5).len(), 1);
        assert_eq!(split_count(&[], 5).len(), 0);
    }

    #[test]
    fn by_gap() {
        let inputs = v(&["a", "b", "c", "d", "e", "f"]);
        // 1 s cadence, a 40 s pause, then a clock that went backwards
        let times = [100.0, 101.0, 102.5, 142.0, 143.0, 50.0];
        let s = split_gap(&inputs, &times, 5.0);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].inputs, v(&["a", "b", "c"]));
        assert_eq!(s[0].times, Some((100.0, 102.5)));
        assert_eq!(s[1].inputs, v(&["d", "e"]));
        assert_eq!(s[2].inputs, v(&["f"]));
        assert_eq!(s[2].times, Some((50.0, 50.0)));
        assert_eq!(split_gap(&inputs, &times, 1000.0).len(), 1);
        assert_eq!(split_gap(&[], &[], 1.0).len(), 0);
    }

    #[test]
    fn by_dir() {
        let s = split_dir(&v(&["x/1.tif", "x/2.tif", "y/1.tif", "3.tif", "x/3.tif"]));
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].inputs, v(&["x/1.tif", "x/2.tif", "x/3.tif"]));
        assert_eq!(s[1].inputs, v(&["y/1.tif"]));
        assert_eq!(s[2].inputs, v(&["3.tif"]));
        assert_eq!(s[0].dir(), "x");
        assert_eq!(s[2].dir(), ".");
        assert_eq!(s[1].first(), "1");
    }

    #[test]
    fn output_names() {
        let names = Names { n: 3, count: 12, first: "DSC_0412", dir: "beetle" };
        assert_eq!(expand("out/{dir}_{first}.tif", &names), "out/beetle_DSC_0412.tif");
        assert_eq!(expand("out/stack{n}.tif", &names), "out/stack03.tif");
        assert_eq!(expand("out.tif", &names), "out_03.tif");
        assert_eq!(expand("out.v2/stacked", &names), "out.v2/stacked_03");
        assert_eq!(expand("slabs", &names), "slabs_03");
        assert_eq!(expand("out.tif", &Names { n: 1, count: 1, first: "", dir: "" }), "out.tif");
        assert_eq!(expand("{n}.tif", &Names { n: 7, count: 250, first: "", dir: "" }), "007.tif");
    }

    #[test]
    fn expand_directories() {
        let tmp = std::env::temp_dir().join(format!("lapstack-batch-{}", std::process::id()));
        let d = tmp.join("s");
        std::fs::create_dir_all(&d).unwrap();
        for n in ["f10.png", "f2.tif", "notes.txt", "f1.JPG"] {
            std::fs::write(d.join(n), b"").unwrap();
        }
        std::fs::create_dir_all(tmp.join("empty")).unwrap();
        let ds = d.to_string_lossy().into_owned();
        let out = expand_dirs(&[ds.clone(), "other.tif".into()]).unwrap();
        let names: Vec<&str> = out.iter().map(|p| p.rsplit('/').next().unwrap()).collect();
        assert_eq!(names, ["f1.JPG", "f2.tif", "f10.png", "other.tif"]);
        assert!(expand_dirs(&[tmp.join("empty").to_string_lossy().into_owned()]).is_err());
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}

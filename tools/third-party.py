#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: MIT

"""Write THIRD-PARTY.md: every crate a lapstack build can contain, the license
lapstack takes it under, its copyright lines, and one copy of each license text
— the notices the permissive licenses ask a distribution to carry. Run from the
repository root (`just third-party`); reads `cargo metadata --all-features`, the
crates' own LICENSE files in the cargo registry, and the font license."""

import json, os, re, subprocess, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# lapstack's choice among the licenses a crate offers, most preferred first
PREFER = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib", "0BSD", "Unlicense", "BSL-1.0", "Unicode-3.0", "Unicode-DFS-2016", "CC0-1.0", "LGPL-2.1"]
FILES = {  # a license id -> the file names crates use for its text
    "MIT": ["LICENSE-MIT", "LICENSE-MIT.md", "LICENSE-MIT.txt", "LICENSE.MIT", "MIT-LICENSE", "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE"],
    "Apache-2.0": ["LICENSE-APACHE", "LICENSE-APACHE.md", "LICENSE-APACHE.txt", "LICENSE.APACHE", "LICENSE-Apache-2.0_WITH_LLVM-exception", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "BSD-3-Clause": ["LICENSE-BSD", "LICENSE-BSD-3-Clause", "LICENSE.BSD", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "BSD-2-Clause": ["LICENSE-BSD", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "ISC": ["LICENSE-ISC", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "Zlib": ["LICENSE-ZLIB", "LICENSE-Zlib", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "0BSD": ["LICENSE-0BSD", "LICENSE", "LICENSE.md"],
    "Unlicense": ["UNLICENSE", "LICENSE", "LICENSE.md"],
    "BSL-1.0": ["LICENSE-BOOST", "LICENSE-BSL", "LICENSE", "LICENSE.md"],
    "Unicode-3.0": ["LICENSE-UNICODE", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "Unicode-DFS-2016": ["LICENSE-UNICODE", "LICENSE", "LICENSE.md", "LICENSE.txt"],
    "CC0-1.0": ["LICENSE-CC0", "LICENSE", "LICENSE.md"],
    "LGPL-2.1": ["LICENSE"],
}
# license texts that name no holder (one copy stands for every crate); MIT and BSD name one, so their copy is a template
GENERIC = {"Apache-2.0", "Unicode-3.0", "Unicode-DFS-2016", "Unlicense", "BSL-1.0", "CC0-1.0", "LGPL-2.1", "0BSD"}

def choose(expr):
    """The license lapstack takes a crate under, from its SPDX expression."""
    if not expr:
        return None
    ids = [t.strip("() ") for t in re.split(r"\s+(?:OR|AND|/)\s+|/", expr)]
    ids = [i.split(" WITH ")[0] for i in ids if i]
    if " AND " in expr:  # every term applies
        return " AND ".join(sorted(set(ids), key=lambda i: PREFER.index(i) if i in PREFER else 99))
    for p in PREFER:
        if p in ids:
            return p
    return ids[0]

def notice(dirname, lic):
    """The crate's license file for `lic`, and the copyright lines in it (or in any license file)."""
    names = FILES.get(lic, ["LICENSE", "LICENSE.md", "LICENSE.txt"])
    have = {f.upper(): f for f in os.listdir(dirname)} if os.path.isdir(dirname) else {}
    text = None
    for n in names:
        if n.upper() in have:
            try:
                text = open(os.path.join(dirname, have[n.upper()]), encoding="utf-8", errors="replace").read()
            except OSError:
                text = None
            if text and (lic not in ("MIT", "Apache-2.0") or n.upper() != "LICENSE" or lic.split("-")[0].lower() in text[:400].lower()):
                break
    lines = []
    for f in sorted(have.values()):
        if not f.upper().startswith(("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "UNLICENSE", "NOTICE")):
            continue
        try:
            for ln in open(os.path.join(dirname, f), encoding="utf-8", errors="replace"):
                ln = ln.strip()
                # a holder's line ("Copyright (c) 2016 Jane Doe"), not the license text's talk of copyright
                if re.match(r"^Copyright\b", ln) and len(ln) < 200 and not re.search(r"\[yyyy\]|<year>|\{year\}|Free Software Foundation|notice|license|licence", ln, re.I):
                    if ln not in lines:
                        lines.append(ln)
        except OSError:
            pass
    return text, lines[:4]

meta = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--all-features"]))
ws = set(meta["workspace_members"])
resolved = {n["id"] for n in meta["resolve"]["nodes"]}
by_lic = defaultdict(list)
texts = {}
for p in sorted(meta["packages"], key=lambda p: (p["name"], p["version"])):
    if p["id"] in ws or p["id"] not in resolved:
        continue
    lic = choose(p.get("license"))
    d = os.path.dirname(p["manifest_path"])
    text, lines = notice(d, lic.split(" AND ")[0] if lic else None)
    for part in (lic.split(" AND ") if lic else ["(no license declared)"]):
        by_lic[part].append((p["name"], p["version"], p.get("license") or "", p.get("repository") or "", lines))
        if part not in texts and text and (part in GENERIC or part in ("MIT", "BSD-3-Clause", "BSD-2-Clause", "ISC", "Zlib")):
            texts[part] = text

out = []
out.append("# Third-party components in lapstack\n")
out.append("lapstack is free software under the MIT license (`LICENSE`). It is built from the components below, each under its own license, which lapstack keeps and whose notices this file carries. Generated by `tools/third-party.py` (`just third-party`) from `cargo metadata --all-features` and the crates' own license files, so it lists every crate any lapstack build — the CLI with and without CUDA, the browser engine, the content-credentials module — can contain; a given build contains a subset. Where a crate offers a choice of licenses, the one lapstack takes it under heads its section.\n")
out.append("## Components that are not crates\n")
out.append("- **Fira Sans and Fira Mono** (`web/fonts`, `crates/lapstack-core/fonts`): Copyright (c) 2012-2015, The Mozilla Foundation and Telefonica S.A., under the SIL Open Font License, Version 1.1 — the full text is in `web/fonts/LICENSE-fira-sans.txt`. The subsets lapstack ships are Reserved-Font-Name-free derivatives under §1 of that license.")
out.append("- **Electron** (the desktop application's shell, `desktop/`): MIT, Copyright (c) Electron contributors and GitHub Inc. electron-builder places Electron's `LICENSE` and Chromium's `LICENSES.chromium.html` in every package it makes.")
out.append("- **rawler** (`vendor/rawler`, the camera raw decoder, Copyright (c) Daniel Vogelbacher): LGPL-2.1, with a two-file change by RAGTUX LLC that is under the same license, inside **lapstack-raw** (`crates/lapstack-raw`, RAGTUX LLC, LGPL-2.1), the module built from it. lapstack does not link either: the module is loaded at run time — a shared library beside the command-line tool, a wasm module (`pkg-raw`) beside the browser app's engine — and its source ships with every build as `lapstack-raw-src.tar.gz` (under `legal/` in the app), so it can be rebuilt and replaced. `vendor/rawler/LAPSTACK-PATCH.md` has the obligations in full; rawler is listed under LGPL-2.1 below as well.\n")
out.append("## Crates, by the license lapstack takes them under\n")
order = sorted(by_lic, key=lambda l: (-len(by_lic[l]), l))
for lic in order:
    out.append(f"### {lic} ({len(by_lic[lic])})\n")
    for name, ver, expr, repo, lines in by_lic[lic]:
        offered = f" — offered as `{expr}`" if expr and expr != lic else ""
        cr = ("; ".join(lines)) if lines else "no copyright line in the package's license files"
        link = f" <{repo}>" if repo else ""
        out.append(f"- **{name} {ver}**{link}{offered}: {cr}")
    out.append("")
out.append("## License texts\n")
out.append("One copy of each license, from the first crate that carries it; for the MIT, BSD and ISC licenses the copyright holder is the one named against each crate above.\n")
for lic in order:
    if lic in texts:
        out.append(f"### {lic}\n\n```\n{texts[lic].strip()}\n```\n")
    else:
        out.append(f"### {lic}\n\nNo text found among the packages; see <https://spdx.org/licenses/{lic}.html>.\n")
open("THIRD-PARTY.md", "w").write("\n".join(out))
n = sum(len(v) for v in by_lic.values())
print(f"THIRD-PARTY.md: {n} crate entries under {len(by_lic)} licenses; texts for {sorted(texts)}")

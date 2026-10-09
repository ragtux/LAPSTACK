#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: MIT

# The source of the raw decoder module as a tarball: crates/lapstack-raw and the
# patched vendor/rawler (both LGPL-2.1), with a workspace file at the root so it
# builds as it is. The LGPL asks that the source travel with every build that
# carries the module, so web/dist.sh puts this under the app's legal/ and the
# release workflow puts it in every command-line tool archive.
#   tools/raw-src.sh OUT.tar.gz
set -euo pipefail
OUT=${1:?usage: tools/raw-src.sh OUT.tar.gz}
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
cd "$(dirname "$0")/.."
SRC=$(mktemp -d)
D="$SRC/lapstack-raw-src"
mkdir -p "$D/crates" "$D/vendor"
cp -R crates/lapstack-raw "$D/crates/"
cp -R vendor/rawler "$D/vendor/"
rm -rf "$D/crates/lapstack-raw/target" "$D/vendor/rawler/target"
cp crates/lapstack-raw/LICENSE "$D/LICENSE"
cp crates/lapstack-raw/README.md "$D/README.md"
cat > "$D/Cargo.toml" <<'TOML'
# The source of lapstack's raw decoder module (LGPL-2.1, see LICENSE and README.md).
# `cargo build --release -p lapstack-raw` here builds the shared library the
# command-line tool loads; README.md has the wasm module's two commands.
[workspace]
resolver = "3"
members = ["crates/lapstack-raw"]
exclude = ["vendor/rawler"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
TOML
tar czf "$OUT" -C "$SRC" lapstack-raw-src
rm -rf "$SRC"
echo "$OUT"

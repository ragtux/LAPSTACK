#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

# Regenerate the small test set: 8 crops (1024x768, 8-bit) from aligned frames
# + the native reference output.  Usage: web/test/make-frames.sh ALIGNED_DIR
#   ALIGNED_DIR holds aligned_NNN.png from `lapstack --save-aligned` (or any
#   pre-aligned frames named in sort order).
set -euo pipefail
cd "$(dirname "$0")/../.."
DIR=${1:-benchmark/out/lapstack/aligned}
mkdir -p web/test/frames
i=0
for k in 0 3 6 9 12 15 18 21; do
    f=$(printf "%s/aligned_%03d.png" "$DIR" $k)
    magick "$f" -crop 1024x768+3400+2700 +repage -depth 8 "$(printf "web/test/frames/f%02d.png" $i)"
    i=$((i + 1))
done
python3 -c "import json,os; json.dump(sorted(f for f in os.listdir('web/test/frames') if f.endswith('.png')), open('web/test/frames/list.json','w'))"
# --no-brightness: test.html runs with brightness off (the crops are equalised by the CLI otherwise, ~0.3 % of full scale)
target/release/lapstack --no-align --no-brightness --depth-scale 2 --depth-raw web/test/expected_dff.png --save-conf -o web/test/expected.png web/test/frames/f0*.png
# the same fusion with halo control (test.html?halo=2&expected=test/expected_halo2.png&expected_dff=&expected_conf=)
target/release/lapstack --no-align --no-brightness --depth winner --halo-control 2 -o web/test/expected_halo2.png web/test/frames/f0*.png
# raw little-endian u16 of the depth and confidence maps (the page cannot decode 16-bit PNG losslessly)
magick web/test/expected_dff.png -depth 16 -endian LSB gray:web/test/expected_dff.u16
magick web/test/expected_conf.png -depth 16 -endian LSB gray:web/test/expected_conf.u16

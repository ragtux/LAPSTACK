#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: AGPL-3.0-only

# Launch Chrome with WebGPU on the real GPU and open the app.
#
# Linux Chrome ships WebGPU behind switches, and the chrome://flags pair
# (enable-unsafe-webgpu + enable-vulkan) only yields the SwiftShader software
# adapter. The hardware adapter needs Vulkan for WebGPU as well:
# --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan. (The flags
# page's "ANGLE graphics backend = Vulkan" / --use-angle=vulkan also exposes
# it, but on NVIDIA + Wayland it makes the accelerated 2D canvas paint black,
# so the app would show nothing.) A dedicated profile dir is used so this
# never touches your main browser profile — and so the switches take effect
# even while your normal Chrome is running (Chrome ignores switches when a
# window with the same profile is already open). Chrome's Wayland backend is
# incompatible with Vulkan ("'--ozone-platform=wayland' is not compatible with
# Vulkan": the window stays black), so the WebGPU profile runs on the X11
# backend (XWayland) regardless of NIXOS_OZONE_WL.
#
#   ./web/serve.sh &      # static server on :8765
#   ./web/chrome.sh       # opens http://localhost:8765/ in the WebGPU profile
set -u
PORT=${1:-8765}
PROFILE=${LAPSTACK_CHROME_PROFILE:-$HOME/.config/lapstack-chrome}
exec "${CHROME:-google-chrome-stable}" --user-data-dir="$PROFILE" --no-first-run --ozone-platform=x11 \
    --enable-unsafe-webgpu --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --ignore-gpu-blocklist \
    "http://localhost:${PORT}/"

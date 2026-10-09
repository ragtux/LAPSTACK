# SPDX-FileCopyrightText: 2026 RAGTUX LLC
# SPDX-License-Identifier: MIT

# NixOS only: an FHS environment in which the Electron binary npm installs, and the
# AppImage / deb tools electron-builder downloads, can run (they are dynamically
# linked for generic Linux). Not needed on any other distribution.
#
#   nix-build nix-fhs.nix -o electron-fhs
#   ./electron-fhs/bin/electron-fhs -c 'npm start'          # or: npm run smoke / npm run dist:linux
#
# (nixpkgs' own electron runs the app too: nix shell nixpkgs#electron -c electron .)
with import <nixpkgs> {};
buildFHSEnv {
  name = "electron-fhs";
  targetPkgs = p: with p; [ zlib libxcrypt-legacy nspr nss gtk3 alsa-lib libdrm mesa libgbm libxkbcommon pango cairo at-spi2-atk at-spi2-core dbus expat glib cups libxshmfence vulkan-loader libglvnd udev libnotify libuuid libsecret
    xorg.libX11 xorg.libXcomposite xorg.libXdamage xorg.libXext xorg.libXfixes xorg.libXrandr xorg.libxcb xorg.libXcursor xorg.libXi xorg.libXrender xorg.libXtst xorg.libXScrnSaver xorg.libxshmfence ];
  runScript = "bash";
}

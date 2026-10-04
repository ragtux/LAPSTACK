-- SPDX-FileCopyrightText: 2026 RAGTUX LLC
-- SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

-- The plug-in's settings: one table of defaults, the persistent preferences
-- (LrPrefs, kept by Lightroom between sessions) filled from it, and the copy
-- between the preferences and an export dialog's property table, so that the
-- Export dialog and the Plug-in Extras menu item share one set of values.

local LrPrefs = import 'LrPrefs'

local M = {}

-- Every setting, with its default. The keys are prefixed so that they cannot
-- collide with Lightroom's own LR_* export settings in a property table.
M.defaults = {
    lapstack_binary = '',            -- path to the lapstack executable
    lapstack_format = 'tif',         -- output: tif | png | dng
    lapstack_stem = '{first}_stacked', -- output name, without extension: {first} the first frame's stem, {n} the frame count
    lapstack_overwrite = false,      -- overwrite an existing output (else a unique name is chosen)
    lapstack_align = true,           -- register the frames (off = --no-align)
    lapstack_coarsen = 2,            -- --align-coarsen N
    lapstack_halo = 0,               -- --halo-control P (0 = off)
    lapstack_wav = false,            -- --wav: also the weighted average
    lapstack_depth = false,          -- --save-depth: also the depth map, next to the output
    lapstack_gpu = false,            -- --gpu --gpu-align (a CUDA build)
    lapstack_extra = '',             -- appended to the command line as typed
    lapstack_keep = false,           -- keep the rendered frames next to the output
    lapstack_render = 'TIFF',        -- the menu item's render: TIFF (16-bit) | ORIGINAL (the raws as they are)
    lapstack_colorspace = 'AdobeRGB', -- the menu item's TIFF color space
}

-- The order the settings are declared in for the export presets (exportPresetFields).
function M.presetFields()
    local out = {}
    for k, v in pairs(M.defaults) do
        out[#out + 1] = { key = k, default = v }
    end
    table.sort(out, function(a, b) return a.key < b.key end)
    return out
end

-- The persistent preferences, every missing key set to its default.
function M.prefs()
    local prefs = LrPrefs.prefsForPlugin()
    for k, v in pairs(M.defaults) do
        if prefs[k] == nil then
            prefs[k] = v
        end
    end
    return prefs
end

-- A plain table of the settings out of any table that carries them (the
-- preferences, or an export dialog's property table), defaults filled in.
function M.read(t)
    local out = {}
    for k, v in pairs(M.defaults) do
        local x = t[k]
        if x == nil then x = v end
        out[k] = x
    end
    return out
end

-- Every setting of `t` set from the preferences (an export dialog opened for
-- the first time takes the plug-in's settings).
function M.copyFromPrefs(t)
    local prefs = M.prefs()
    for k, _ in pairs(M.defaults) do
        t[k] = prefs[k]
    end
end

-- Write the settings of `t` back into the preferences (an export dialog
-- closed with Export makes its settings the menu item's).
function M.storeToPrefs(t)
    local prefs = M.prefs()
    for k, _ in pairs(M.defaults) do
        if t[k] ~= nil then
            prefs[k] = t[k]
        end
    end
end

return M

-- Copyright (c) 2026 RAGTUX LLC
-- INTERNAL USE ONLY

-- The settings as dialog rows, built once for the Export dialog's section and
-- once for the Plug-in Manager's: `f` is the view factory, `bindTo` the
-- observable table the controls read and write (an export dialog's property
-- table, or the plug-in's preferences).

local LrDialogs = import 'LrDialogs'
local LrView = import 'LrView'

local M = {}

-- A file chooser for the lapstack executable; the path lands in `bindTo.lapstack_binary`.
local function browse(bindTo)
    local picked = LrDialogs.runOpenPanel({
        title = 'The lapstack executable',
        canChooseFiles = true,
        canChooseDirectories = false,
        allowsMultipleSelection = false,
    })
    if picked and picked[1] then
        bindTo.lapstack_binary = picked[1]
    end
end

function M.rows(f, bindTo)
    local bind = LrView.bind
    local label = LrView.share('lapstack_label')
    local function row(title, ...)
        return f:row {
            spacing = f:label_spacing(),
            f:static_text { title = title, alignment = 'right', width = label },
            ...
        }
    end
    return {
        row('lapstack:',
            f:edit_field { value = bind 'lapstack_binary', bind_to_object = bindTo, width_in_chars = 36, immediate = true,
                           tooltip = 'the lapstack command-line executable' },
            f:push_button { title = 'Browse…', action = function() browse(bindTo) end }),
        row('output:',
            f:popup_menu { value = bind 'lapstack_format', bind_to_object = bindTo,
                           items = { { title = 'TIFF (16-bit)', value = 'tif' }, { title = 'PNG (16-bit)', value = 'png' },
                                     { title = 'DNG (linear; from raw frames)', value = 'dng' } } },
            f:static_text { title = 'named' },
            f:edit_field { value = bind 'lapstack_stem', bind_to_object = bindTo, width_in_chars = 18, immediate = true,
                           tooltip = '{first} = the first frame\'s name, {n} = the number of frames' },
            f:checkbox { title = 'overwrite', value = bind 'lapstack_overwrite', bind_to_object = bindTo,
                         tooltip = 'else a unique name is chosen when the file exists' }),
        row('alignment:',
            f:checkbox { title = 'register the frames', value = bind 'lapstack_align', bind_to_object = bindTo },
            f:static_text { title = 'coarsen' },
            f:edit_field { value = bind 'lapstack_coarsen', bind_to_object = bindTo, width_in_chars = 3, min = 0, max = 6, precision = 0, increment = 1,
                           enabled = LrView.bind { key = 'lapstack_align', bind_to_object = bindTo },
                           tooltip = '--align-coarsen: pyramid levels the search stops short of full resolution (2 is fast and sub-pixel)' }),
        row('fusion:',
            f:static_text { title = 'halo control' },
            f:edit_field { value = bind 'lapstack_halo', bind_to_object = bindTo, width_in_chars = 3, min = 0, max = 8, precision = 0, increment = 1,
                           tooltip = '--halo-control: 0 = off, 1 = weigh the coarse levels by the guide\'s energy, up to 8 = a hard pick' },
            f:checkbox { title = 'weighted average too', value = bind 'lapstack_wav', bind_to_object = bindTo, tooltip = '--wav: the weighted average as <stem>_wav' },
            f:checkbox { title = 'depth map too', value = bind 'lapstack_depth', bind_to_object = bindTo, tooltip = '--save-depth: <stem>_depth.png' },
            f:checkbox { title = 'CUDA', value = bind 'lapstack_gpu', bind_to_object = bindTo, tooltip = '--gpu --gpu-align: a build with the gpu feature and an NVIDIA card' }),
        row('extra options:',
            f:edit_field { value = bind 'lapstack_extra', bind_to_object = bindTo, width_in_chars = 44, immediate = true,
                           tooltip = 'appended to the command line as typed, e.g. --interpolation lanczos3 --stereo 3' }),
        row('',
            f:checkbox { title = 'keep the rendered frames next to the output', value = bind 'lapstack_keep', bind_to_object = bindTo }),
    }
end

-- The menu item renders the selection itself, without the Export dialog:
-- what it renders is a preference of its own.
function M.renderRows(f, bindTo)
    local bind = LrView.bind
    local label = LrView.share('lapstack_label')
    return {
        f:row {
            spacing = f:label_spacing(),
            f:static_text { title = 'menu item renders:', alignment = 'right', width = label },
            f:popup_menu { value = bind 'lapstack_render', bind_to_object = bindTo,
                           items = { { title = '16-bit TIFF', value = 'TIFF' }, { title = 'the originals (raws as shot; for DNG output)', value = 'ORIGINAL' } } },
            f:popup_menu { value = bind 'lapstack_colorspace', bind_to_object = bindTo,
                           enabled = LrView.bind { key = 'lapstack_render', bind_to_object = bindTo, transform = function(v) return v == 'TIFF' end },
                           items = { { title = 'sRGB', value = 'sRGB' }, { title = 'Adobe RGB', value = 'AdobeRGB' }, { title = 'ProPhoto RGB', value = 'ProPhotoRGB' } } },
        },
    }
end

return M

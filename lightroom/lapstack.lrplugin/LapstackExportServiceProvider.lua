-- Copyright (c) 2026 RAGTUX LLC
-- INTERNAL USE ONLY

-- The export service: "lapstack" in the Export dialog's Export To menu. The
-- dialog keeps Lightroom's own file settings (a 16-bit TIFF is the right
-- render; Original hands lapstack the raws as shot, which is what a DNG
-- output wants), image sizing, output sharpening and metadata sections; the
-- location and naming sections are hidden, so Lightroom renders the frames
-- into a temporary folder that it removes when the export is over. The
-- plug-in's own section holds the lapstack settings; they go into the export
-- preset, and closing the dialog with Export makes them the menu item's too.
-- processRenderedPhotos waits for every frame, runs lapstack over them in the
-- order of the selection and adds the result to the catalogue.

local LrDialogs = import 'LrDialogs'
local LrView = import 'LrView'

local Prefs = require 'LapstackPrefs'
local Run = require 'LapstackRun'
local UI = require 'LapstackUI'

local provider = {}

provider.hideSections = { 'exportLocation', 'fileNaming', 'video', 'watermarking' }
provider.allowFileFormats = { 'TIFF', 'JPEG', 'DNG', 'ORIGINAL' }
provider.allowColorSpaces = { 'sRGB', 'AdobeRGB', 'ProPhotoRGB' }
provider.canExportVideo = false
provider.hidePrintResolution = true
provider.exportPresetFields = Prefs.presetFields()

-- A dialog (or preset) that never had lapstack's path takes every setting
-- from the plug-in's preferences; one that had keeps its own.
function provider.startDialog(propertyTable)
    if (propertyTable.lapstack_binary or '') == '' then
        Prefs.copyFromPrefs(propertyTable)
    end
end

function provider.endDialog(propertyTable, why)
    if why == 'ok' then
        Prefs.storeToPrefs(propertyTable)
    end
end

function provider.sectionsForTopOfDialog(f, propertyTable)
    local section = {
        title = 'lapstack',
        synopsis = LrView.bind {
            key = 'lapstack_format',
            transform = function(v)
                return 'stacked as ' .. (v == 'dng' and 'DNG' or v == 'png' and 'PNG' or 'TIFF') .. ', next to the first frame'
            end,
        },
        spacing = f:control_spacing(),
    }
    for _, r in ipairs(UI.rows(f, propertyTable)) do
        section[#section + 1] = r
    end
    section[#section + 1] = f:static_text {
        title = 'Select the frames of one stack in order (sort by capture time), render them as 16-bit TIFF — or Original for the raws, '
             .. 'which is what a DNG output wants — and Export: the stacked image lands next to the first frame and in the catalogue, stacked with it.',
        height_in_lines = 3, width_in_chars = 70,
    }
    return { section }
end

function provider.processRenderedPhotos(functionContext, exportContext)
    local session = exportContext.exportSession
    local count = session:countRenditions()
    local scope = exportContext:configureProgress {
        title = string.format('lapstack: rendering %d frame%s', count, count == 1 and '' or 's'),
        renderPortion = 0.5,
    }
    local frames, photos, failed = {}, {}, {}
    for _, rendition in exportContext:renditions { stopIfCanceled = true } do
        local ok, path = rendition:waitForRender()
        if ok then
            frames[#frames + 1] = path
            photos[#photos + 1] = rendition.photo
        else
            failed[#failed + 1] = tostring(path)
        end
    end
    if scope:isCanceled() then
        return
    end
    if #failed > 0 then
        LrDialogs.showError('lapstack: ' .. #failed .. ' frame(s) could not be rendered:\n' .. table.concat(failed, '\n'))
        return
    end
    local output, added, why = Run.stack(exportContext.propertyTable, photos, frames, scope)
    Run.report(output, added, why)
end

return provider

-- SPDX-FileCopyrightText: 2026 RAGTUX LLC
-- SPDX-License-Identifier: LicenseRef-RAGTUX-Proprietary

-- Library > Plug-in Extras > Stack with lapstack: the selected frames are
-- rendered with the plug-in's preferences (a 16-bit TIFF in the chosen color
-- space, or the originals as they are) into a folder of the system's
-- temporary directory, lapstack runs over them in the order of the selection,
-- the result is added to the catalog stacked with the first frame, and the
-- rendered files go. No dialog opens: the settings are those of the Plug-in
-- Manager's section, or of the last export through the lapstack service.

local LrApplication = import 'LrApplication'
local LrDate = import 'LrDate'
local LrDialogs = import 'LrDialogs'
local LrExportSession = import 'LrExportSession'
local LrFileUtils = import 'LrFileUtils'
local LrFunctionContext = import 'LrFunctionContext'
local LrPathUtils = import 'LrPathUtils'

local Prefs = require 'LapstackPrefs'
local Run = require 'LapstackRun'

-- The render settings of one run: what the Export dialog would have produced
-- for a Hard Drive export into `folder`.
local function exportSettings(settings, folder)
    local s = {
        LR_exportServiceProvider = 'com.adobe.ag.export.file',
        LR_exportServiceProviderTitle = 'Hard Drive',
        LR_format = settings.lapstack_render == 'ORIGINAL' and 'ORIGINAL' or 'TIFF',
        LR_export_bitDepth = 16,
        LR_export_colorSpace = settings.lapstack_colorspace or 'AdobeRGB',
        LR_tiff_compressionMethod = 'compressionMethod_None',
        LR_tiff_preserveTransparency = false,
        LR_size_doConstrain = false,
        LR_size_resolution = 300,
        LR_size_resolutionUnits = 'inch',
        LR_outputSharpeningOn = false,
        LR_minimizeEmbeddedMetadata = false,
        LR_embeddedMetadataOption = 'all',
        LR_removeLocationMetadata = false,
        LR_removeFaceMetadata = true,
        LR_metadata_keywordOptions = 'lightroomHierarchical',
        LR_useWatermark = false,
        LR_includeVideoFiles = false,
        LR_export_destinationType = 'specificFolder',
        LR_export_destinationPathPrefix = folder,
        LR_export_useSubfolder = false,
        LR_collisionHandling = 'rename',
        LR_renamingTokensOn = false,
        LR_reimportExportedPhoto = false,
    }
    return s
end

LrFunctionContext.postAsyncTaskWithContext('Stack with lapstack', function(context)
    LrDialogs.attachErrorDialogToFunctionContext(context)
    local catalog = LrApplication.activeCatalog()
    local photos = catalog:getTargetPhotos()
    if #photos < 2 then
        LrDialogs.message('Stack with lapstack', 'Select the frames of one focus stack first (at least two).', 'info')
        return
    end
    local settings = Prefs.read(Prefs.prefs())
    if settings.lapstack_binary == '' or not LrFileUtils.exists(settings.lapstack_binary) then
        LrDialogs.showError('lapstack: set the path to the lapstack executable in the Plug-in Manager (File > Plug-in Manager > lapstack).')
        return
    end
    local folder = LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), string.format('lapstack-%d', math.floor(LrDate.currentTime())))
    LrFileUtils.createAllDirectories(folder)
    local rendered = {}
    context:addCleanupHandler(function()
        Run.removeFiles(rendered)
        if LrFileUtils.exists(folder) then
            LrFileUtils.delete(folder)
        end
    end)
    local scope = Run.progress(context, string.format('Stack with lapstack: %d frames', #photos))
    local session = LrExportSession { photosToExport = photos, exportSettings = exportSettings(settings, folder) }
    session:doExportOnNewTask()
    local frames, sources, failed = {}, {}, {}
    for _, rendition in session:renditions { progressScope = scope, renderProgressPortion = 0.5, stopIfCanceled = true } do
        local ok, path = rendition:waitForRender()
        if ok then
            rendered[#rendered + 1] = path
            frames[#frames + 1] = path
            sources[#sources + 1] = rendition.photo
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
    local output, added, why = Run.stack(settings, sources, frames, scope)
    Run.report(output, added, why)
    scope:done()
end)

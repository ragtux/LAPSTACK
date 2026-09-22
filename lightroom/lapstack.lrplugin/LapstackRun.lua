-- Copyright (c) 2026 RAGTUX LLC
-- INTERNAL USE ONLY

-- What both doors share once the frames are rendered: the command line built
-- from the settings, run with its stderr in a log file next to the output, the
-- result added to the catalogue stacked with the first frame, the rendered
-- frames kept or removed. LrTasks.execute hands the line to the OS shell — on
-- Windows through cmd.exe, which drops the outermost pair of double quotes,
-- so the whole line is wrapped in one more pair there (the SDK's long-known,
-- undocumented quirk); on macOS and Linux every argument is single-quoted.

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local Prefs = require 'LapstackPrefs'

local M = {}

local function trim(s)
    return (tostring(s or ''):gsub('^%s+', ''):gsub('%s+$', ''))
end

-- One argument, quoted for the shell of this platform.
function M.quote(arg)
    arg = tostring(arg)
    if WIN_ENV then
        return '"' .. arg:gsub('"', '') .. '"'
    end
    return "'" .. arg:gsub("'", "'\\''") .. "'"
end

-- The output's extension for the format setting.
function M.extension(settings)
    local f = settings.lapstack_format
    if f == 'png' then return 'png' end
    if f == 'dng' then return 'dng' end
    return 'tif'
end

-- The lapstack options the settings ask for, each quoted for the shell, in
-- order; the extra options go last as typed (the user quotes what needs
-- quoting there), so they can override anything.
function M.options(settings)
    local o = {}
    local function add(...)
        for _, a in ipairs({ ... }) do
            o[#o + 1] = M.quote(a)
        end
    end
    if not settings.lapstack_align then
        add('--no-align')
    else
        add('--align-coarsen', tostring(math.max(0, math.floor(tonumber(settings.lapstack_coarsen) or 2))))
    end
    local halo = tonumber(settings.lapstack_halo) or 0
    if halo > 0 then
        add('--halo-control', tostring(halo))
    end
    if settings.lapstack_wav then add('--wav') end
    if settings.lapstack_depth then add('--save-depth') end
    if settings.lapstack_gpu then add('--gpu', '--gpu-align') end
    local extra = trim(settings.lapstack_extra)
    if extra ~= '' then
        o[#o + 1] = extra
    end
    return o
end

-- The command line: binary, options, -o output, the frames in order, stderr to the log.
function M.command(settings, frames, output, logPath)
    local parts = { M.quote(settings.lapstack_binary) }
    for _, o in ipairs(M.options(settings)) do
        parts[#parts + 1] = o
    end
    parts[#parts + 1] = '-o'
    parts[#parts + 1] = M.quote(output)
    for _, f in ipairs(frames) do
        parts[#parts + 1] = M.quote(f)
    end
    parts[#parts + 1] = '2>'
    parts[#parts + 1] = M.quote(logPath)
    local line = table.concat(parts, ' ')
    if WIN_ENV then
        line = '"' .. line .. '"'
    end
    return line
end

-- The last `n` lines of a text file, for an error dialog.
function M.tail(path, n)
    if not LrFileUtils.exists(path) then return '' end
    local text = LrFileUtils.readFile(path) or ''
    local lines = {}
    for l in text:gmatch('[^\r\n]+') do
        lines[#lines + 1] = l
    end
    local from = math.max(1, #lines - (n or 12) + 1)
    return table.concat(lines, '\n', from, #lines)
end

-- Where the result goes: next to the first source photo, named by the stem
-- template ({first} = the first frame's stem, {n} = the frame count), made
-- unique unless the settings say overwrite.
function M.outputPath(settings, firstPhoto, count)
    local src = firstPhoto:getRawMetadata('path')
    local dir = LrPathUtils.parent(src)
    local first = LrPathUtils.removeExtension(LrPathUtils.leafName(src))
    local stem = trim(settings.lapstack_stem)
    if stem == '' then stem = '{first}_stacked' end
    stem = stem:gsub('{first}', first):gsub('{n}', tostring(count))
    stem = stem:gsub('[\\/:*?"<>|]', '_')
    local path = LrPathUtils.child(dir, stem .. '.' .. M.extension(settings))
    if LrFileUtils.exists(path) and not settings.lapstack_overwrite then
        path = LrFileUtils.chooseUniqueFileName(path)
    end
    return path
end

-- The rendered frames copied next to the output, into <stem>_frames/.
function M.keepFrames(frames, output)
    local dir = LrPathUtils.removeExtension(output) .. '_frames'
    LrFileUtils.createAllDirectories(dir)
    for i, f in ipairs(frames) do
        local name = string.format('%03d_%s', i, LrPathUtils.leafName(f))
        LrFileUtils.copy(f, LrPathUtils.child(dir, name))
    end
    return dir
end

-- The result into the catalogue, stacked above the first frame, and selected;
-- nil and a message when Lightroom would not take the file (the file stays).
function M.import(output, firstPhoto)
    local catalog = LrApplication.activeCatalog()
    local added
    local ok, err = LrTasks.pcall(function()
        catalog:withWriteAccessDo('Stack with lapstack', function()
            added = catalog:addPhoto(output, firstPhoto, 'above')
        end, { timeout = 30 })
    end)
    if not ok or not added then
        return nil, tostring(err or 'the catalogue did not take it')
    end
    catalog:setSelectedPhotos(added, {})
    return added
end

-- Run lapstack over `frames` (rendered files, in stack order) for `photos`
-- (their LrPhoto objects, the same order), inside the task the caller runs
-- on; `scope` is a progress scope to report through. Returns the output path
-- and the photo added, or nil and a message. The CLI cannot be interrupted
-- from here: the scope's cancel is honoured before it starts and after.
function M.stack(settings, photos, frames, scope)
    settings = Prefs.read(settings)
    if #frames < 2 then
        return nil, nil, 'a focus stack needs at least two frames (' .. #frames .. ' selected)'
    end
    local binary = trim(settings.lapstack_binary)
    if binary == '' then
        return nil, nil, 'the path to the lapstack executable is not set (Plug-in Manager > lapstack, or the section in the Export dialog)'
    end
    if not LrFileUtils.exists(binary) then
        return nil, nil, 'lapstack was not found at ' .. binary
    end
    if scope and scope:isCanceled() then
        return nil, nil, 'cancelled'
    end
    local output = M.outputPath(settings, photos[1], #frames)
    local logPath = LrPathUtils.removeExtension(output) .. '.lapstack.log'
    if scope then
        scope:setCaption(string.format('stacking %d frames with lapstack …', #frames))
        scope:setPortionComplete(0.05, 1)
    end
    local line = M.command(settings, frames, output, logPath)
    local status = LrTasks.execute(line)
    if scope then
        scope:setPortionComplete(0.9, 1)
    end
    if status ~= 0 or not LrFileUtils.exists(output) then
        local why = string.format('lapstack exited with status %s and left no %s', tostring(status), LrPathUtils.leafName(output))
        local t = M.tail(logPath, 12)
        if t ~= '' then
            why = why .. '\n\n' .. t .. '\n\n(the whole log is ' .. logPath .. ')'
        else
            why = why .. '\n\nthe command line was:\n' .. line
        end
        return nil, nil, why
    end
    if settings.lapstack_keep then
        M.keepFrames(frames, output)
    end
    local added, why = M.import(output, photos[1])
    if scope then
        scope:setPortionComplete(1, 1)
    end
    return output, added, why
end

-- The rendered files removed (the frames Lightroom rendered for the run);
-- an export's temporary folder goes by itself, the menu item's is ours.
function M.removeFiles(paths)
    for _, p in ipairs(paths) do
        if LrFileUtils.exists(p) then
            LrFileUtils.delete(p)
        end
    end
end

-- A progress scope for a run, tied to the function context.
function M.progress(context, title)
    local scope = LrProgressScope({ title = title or 'lapstack', functionContext = context })
    scope:setCancelable(true)
    return scope
end

-- What became of a run, to the user: a bezel for a result in the catalogue, a
-- dialog when the file was written but not imported, an error otherwise.
function M.report(output, added, why)
    if output and added then
        LrDialogs.showBezel('lapstack: ' .. LrPathUtils.leafName(output) .. ' added to the catalogue', 3)
    elseif output then
        LrDialogs.message('lapstack wrote ' .. output, 'but it could not be added to the catalogue: ' .. tostring(why) .. '\nImport the file by hand.', 'warning')
    else
        LrDialogs.showError('lapstack: ' .. tostring(why))
    end
end

return M

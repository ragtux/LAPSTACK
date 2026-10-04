# lapstack for Lightroom Classic

`lapstack.lrplugin` sends the frames of a focus stack out of the catalog to
the `lapstack` command-line tool and brings the stacked image back into it,
stacked with the first frame — the usual focus-stacking round trip. It needs
Lightroom Classic 6 or later and a `lapstack` binary
(`cargo build --release` in this repository; `--features gpu` for the CUDA
options).

## Install

File > Plug-in Manager > Add, choose the `lapstack.lrplugin` folder (copy it
anywhere first; Lightroom reads it in place). In the plug-in's section of the
Plug-in Manager set the path to the `lapstack` executable (Browse…) and the
stacking settings: output format (16-bit TIFF, 16-bit PNG, or a linear DNG —
for frames handed over as the raws they were shot as), the output name
(`{first}` is the first frame's name, `{n}` the frame count; `{first}_stacked`
by default), alignment on or off with its `--align-coarsen`, `--halo-control`,
the weighted average and the depth map as extra files, CUDA, and a free line of
extra options appended to the command as typed (`--interpolation lanczos3
--stereo 3`, say). The result is written next to the first frame and added to
the catalog stacked above it; lapstack's log goes next to it as
`<name>.lapstack.log`.

## Use

Two doors, the same run behind them:

- **Library > Plug-in Extras > Stack with lapstack** — select the frames of
  one stack (in order: sort the grid by capture time), choose the item. The
  frames are rendered with the settings of the Plug-in Manager (a 16-bit TIFF
  in the chosen color space, or *the originals* — the raws as shot, which is
  what a DNG output wants) into a temporary folder, lapstack runs, the result
  is imported and selected, the rendered files go (*keep the rendered frames*
  copies them next to the output first). No dialog opens.
- **File > Export, Export To: lapstack** — the Export dialog with Lightroom's
  own file settings (render 16-bit TIFF, or Original for the raws), image
  sizing, output sharpening and metadata sections, and the plug-in's own
  section with the same lapstack settings; the location and naming sections
  are hidden (Lightroom renders into a temporary folder it removes afterward).
  The settings are saved in the export preset, and closing the dialog with
  Export makes them the menu item's too.

One selection is one stack: to stack several, run the CLI with `--split`
(`lapstack --help`). The run cannot be interrupted once lapstack has started;
the progress bar's cancel takes effect before and after it. A result Lightroom
will not import (an unsupported format from the extra options, say) is left
on disk and reported.

## Files

```
lapstack.lrplugin/
  Info.lua                          the plug-in's manifest
  LapstackPrefs.lua                 settings: defaults, the preferences, the copy to and from an export dialog
  LapstackRun.lua                   the command line, the run, the import, the log — shared by both doors
  LapstackUI.lua                    the settings as dialog rows
  LapstackInfoProvider.lua          the Plug-in Manager section
  LapstackExportServiceProvider.lua the export service
  LapstackMenuItem.lua              Library > Plug-in Extras > Stack with lapstack
```

On Windows the command line is wrapped in one extra pair of double quotes
(`LrTasks.execute` hands it to `cmd.exe`, which strips the outermost pair — the
SDK's long-known quirk); on macOS every argument is single-quoted for `sh`.

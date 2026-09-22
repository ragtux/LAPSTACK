-- Copyright (c) 2026 RAGTUX LLC
-- INTERNAL USE ONLY

-- lapstack for Lightroom Classic: the frames of a focus stack go out of the
-- catalogue to the lapstack command-line tool and the stacked image comes back
-- into it, stacked with the first frame — the round trip Helicon Focus and
-- Zerene Stacker offer. Two doors: an export service ("lapstack" in the Export
-- dialog's Export To menu, with Lightroom's own file, size and metadata
-- sections) and Library > Plug-in Extras > Stack with lapstack, which renders
-- the selection with the settings kept in the plug-in's preferences and asks
-- nothing.

return {
    LrSdkVersion = 6.0,
    LrSdkMinimumVersion = 6.0,
    LrToolkitIdentifier = 'com.lapstack.lightroom',
    LrPluginName = 'lapstack',

    LrPluginInfoProvider = 'LapstackInfoProvider.lua',

    LrExportServiceProvider = {
        title = 'lapstack',
        file = 'LapstackExportServiceProvider.lua',
    },

    LrLibraryMenuItems = {
        {
            title = 'Stack with lapstack',
            file = 'LapstackMenuItem.lua',
            enabledWhen = 'photosSelected',
        },
    },

    VERSION = { major = 0, minor = 1, revision = 0, build = 0 },
}

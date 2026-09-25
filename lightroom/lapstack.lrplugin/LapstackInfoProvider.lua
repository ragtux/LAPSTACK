-- SPDX-FileCopyrightText: 2026 RAGTUX LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- The plug-in's section in the Plug-in Manager: the settings bound to the
-- preferences, which the Plug-in Extras menu item uses and an Export dialog
-- starts from.

local Prefs = require 'LapstackPrefs'
local UI = require 'LapstackUI'

return {
    sectionsForTopOfDialog = function(f, _)
        local prefs = Prefs.prefs()
        local rows = UI.rows(f, prefs)
        for _, r in ipairs(UI.renderRows(f, prefs)) do
            rows[#rows + 1] = r
        end
        rows[#rows + 1] = f:static_text {
            title = 'Library > Plug-in Extras > Stack with lapstack renders the selected frames with these settings and stacks them; '
                 .. 'the lapstack export service (File > Export, Export To) does the same with the Export dialog\'s file and size settings.',
            height_in_lines = 3, width_in_chars = 70,
        }
        local section = { title = 'lapstack', synopsis = 'focus stacking with the lapstack command-line tool', spacing = f:control_spacing() }
        for _, r in ipairs(rows) do
            section[#section + 1] = r
        end
        return { section }
    end,
}

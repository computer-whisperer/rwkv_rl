--
-- Example AIOptions.lua
--
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Custom Options Definition Table format
--
-- NOTES:
-- - using an enumerated table lets you specify the options order
--
-- These keywords must be lowercase for LuaParser to read them.
--
-- key:      the string used in the script.txt
-- name:     the displayed name
-- desc:     the description (could be used as a tooltip)
-- type:     the option type (bool|number|string|list|section)
-- section:  can be used to group this option with others (optional)
-- def:      the default value
-- min:      minimum value for type=number options
-- max:      maximum value for type=number options
-- step:     quantization step, aligned to the def value
-- maxlen:   the maximum string length for type=string options
-- items:    array of item strings for type=list options
-- scope:    'all', 'player', 'team', 'allyteam'      <<< not supported yet >>>
--
-- This is an example file, show-casting all the possibilities of this format.
-- It contains one example for option types bool, string and list, and two
-- for number and section.
--
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------


local options = {

}

return options


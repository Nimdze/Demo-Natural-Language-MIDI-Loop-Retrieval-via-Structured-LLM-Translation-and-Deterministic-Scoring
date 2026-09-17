"""Colors, drum maps, and style constants for piano-roll rendering."""

import pretty_midi

BACKGROUND_COLOR = "#0E1117"
FACE_COLOR = "#1E1E1E"
GRID_COLOR = "#333333"
SPINE_COLOR = "#333333"
OUT_OF_BOUNDS_COLOR = "#FF4444"
VELOCITY_OFF_COLOR = "#00FFAA"
NOTE_EDGE_DARK = "#0E1117"
NOTE_EDGE_HIGHLIGHT = "#FFFFFF"

DRUM_ROW_EVEN = "#2A2B2E"
DRUM_ROW_ODD = "#1E1F22"
PIANO_BLACK_KEY = "#181818"
PIANO_WHITE_KEY = "#DDDDDD"
PIANO_ROW_DARK = "#1A1A1A"
PIANO_ROW_LIGHT = "#222222"
KEYBOARD_TEXT_COLOR = "#111111"

GM_DRUM_NAMES = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi-Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}

DAW_DRUM_NAMES = {
    35: "Kick 2",
    36: "Kick 1",
    37: "Rimshot",
    38: "Snare 1",
    39: "Clap",
    40: "Snare 2",
    41: "Low Tom",
    42: "Closed Hat",
    43: "High Tom",
    44: "Pedal Hat",
    45: "Low Tom",
    46: "Open Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash 1",
    50: "High Tom",
    51: "Ride 1",
    52: "Cymbal",
    53: "Ride 3",
    54: "Tambourine",
    55: "Crash 1",
    56: "Cowbell",
    57: "Crash 2",
    58: "Vibraslap",
    59: "Ride 2",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Conga",
    63: "Open Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}

DRUM_FAMILIES = {
    "kick": [35, 36],
    "snare": [37, 38, 39, 40],
    "snare_clap": [37, 38, 39, 40],
    "hats": [42, 44, 46, 51, 53, 59],
    "cymbals": [49, 52, 55, 57],
    "hats_cymbals": [42, 44, 46, 49, 51, 52, 53, 55, 57, 59],
    "toms_others": [
        41,
        43,
        45,
        47,
        48,
        50,
        54,
        56,
        58,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
    ],
    "perc": "UNMAPPED",
}

STANDARD_DRUM_FAMILIES = {
    "kick": [35, 36],
    "snare": [37, 38, 39, 40],
    "hats": [42, 44, 46, 51, 53, 59],
    "cymbals": [49, 52, 55, 57],
}

ALL_STANDARD_PITCHES = sorted({p for pitches in STANDARD_DRUM_FAMILIES.values() for p in pitches})

FAMILY_COLORS = {
    "kick": "#FF6B6B",
    "snare": "#4ECDC4",
    "snare_clap": "#4ECDC4",
    "hats": "#FFE66D",
    "cymbals": "#B5CEA8",
    "hats_cymbals": "#FFE66D",
    "toms_others": "#C792EA",
    "perc": "#C792EA",
}

DAW_FAMILY_LABEL = {
    "kick": "KICK",
    "snare": "SNRCLP",
    "snare_clap": "SNRCLP",
    "hats": "HATCYM",
    "cymbals": "HATCYM",
    "hats_cymbals": "HATCYM",
    "toms_others": "TOMFX",
    "perc": "UNMAP",
}

CATEGORY_COLORS = {
    "metadata": "#569CD6",
    "rhythm": "#CE9178",
    "harmony": "#4EC9B0",
    "melody": "#C586C0",
    "expression": "#DCDCAA",
    "tonality": "#B5CEA8",
    "drums": "#D16969",
}


def _pitch_family(pitch: int) -> str:
    for family in ("kick", "snare", "hats", "cymbals"):
        if pitch in DRUM_FAMILIES[family]:
            return family
    toms_others = DRUM_FAMILIES.get("toms_others")
    if isinstance(toms_others, list) and pitch in toms_others:
        return "toms_others"
    return "perc"


def _daw_family_label(pitch: int) -> str:
    family = _pitch_family(pitch)
    return DAW_FAMILY_LABEL.get(family, "TOMFX")


def _is_drum_track(pm: pretty_midi.PrettyMIDI) -> bool:
    """Return True if the MIDI file contains an instrument on the drum channel."""
    return any(inst.is_drum for inst in pm.instruments if inst.notes)


def _instrument_label(instrument: pretty_midi.Instrument) -> str:
    if instrument.is_drum:
        return "Drums"
    return pretty_midi.program_to_instrument_name(instrument.program) or f"Program {instrument.program}"

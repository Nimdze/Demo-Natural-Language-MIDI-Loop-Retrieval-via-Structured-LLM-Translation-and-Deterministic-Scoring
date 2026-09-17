"""Core piano roll orchestration."""

import base64
import hashlib
import io
import sys

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pretty_midi

from .config import PianoRollConfig
from .draw import (
    draw_axes,
    draw_beat_grid,
    draw_c_note_guides,
    draw_daw_background,
    draw_keyboard,
    draw_notes,
    draw_title,
    finish_spines,
)
from .styles import (
    BACKGROUND_COLOR,
    DRUM_FAMILIES,
    FACE_COLOR,
    _instrument_label,
    _is_drum_track,
)


def _midi_hash(midi: pretty_midi.PrettyMIDI) -> str:
    """Stable hash for caching based on note content."""
    parts = []
    for inst in midi.instruments:
        parts.append(f"program={inst.program},drum={inst.is_drum}")
        for note in inst.notes:
            parts.append(f"{note.pitch}-{note.start:.6f}-{note.end:.6f}-{note.velocity}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _collect_notes(midi: pretty_midi.PrettyMIDI, config: PianoRollConfig) -> list[dict]:
    ticks_per_beat = midi.resolution

    def to_beats(time: float) -> float:
        return midi.time_to_tick(time) / ticks_per_beat

    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append(
                {
                    "pitch": note.pitch,
                    "start": note.start if config.time_axis == "seconds" else to_beats(note.start),
                    "end": note.end if config.time_axis == "seconds" else to_beats(note.end),
                    "velocity": note.velocity,
                    "is_drum": instrument.is_drum,
                    "instrument_label": _instrument_label(instrument),
                }
            )
    return notes


def _resolve_range(notes: list[dict], config: PianoRollConfig) -> tuple[int, int]:
    if config.range_mode == "full":
        return 0, 127
    if config.range_mode == "custom":
        return config.y_min, config.y_max
    if config.range_mode == "auto":
        if not notes:
            return 21, 108
        pitches = {n["pitch"] for n in notes}
        y_min = max(0, min(pitches) - 2)
        y_max = min(127, max(pitches) + 2)
        return y_min, y_max
    return 21, 108


def _end_value(notes: list[dict], config: PianoRollConfig, midi: pretty_midi.PrettyMIDI) -> float:
    if config.time_axis == "seconds":
        return midi.get_end_time()
    if not notes:
        return 0.0
    return max(n["end"] for n in notes)


def _width(config: PianoRollConfig, end_value: float) -> float:
    if config.width is not None:
        return config.width
    if config.time_axis == "beats":
        return max(8.0, end_value * 0.6)
    return max(2.0, end_value * 0.8)


def _resolve_highlight(notes: list[dict], family: str | None) -> tuple[set[int], bool, str | None]:
    target = str(family).lower().strip() if family else ""
    if not target or target == "none":
        return set(), False, None

    # Family aliases
    if target == "snare_clap":
        target = "snare"

    family_pitches = DRUM_FAMILIES.get(target)
    if family_pitches == "UNMAPPED":
        return set(), True, target
    if isinstance(family_pitches, list):
        target_set = set(family_pitches)
    else:
        return set(), False, None

    pitches_in_file = sorted({n["pitch"] for n in notes})
    matched = any(n["pitch"] in target_set for n in notes if n["is_drum"])
    if not matched and pitches_in_file:
        if target == "kick" and pitches_in_file:
            target_set = {pitches_in_file[0]}
        elif target in ("snare", "snare_clap") and len(pitches_in_file) > 1:
            target_set = {pitches_in_file[1]}

    return target_set, False, target


def render(midi: pretty_midi.PrettyMIDI, config: PianoRollConfig) -> str:
    """Render a piano roll as a base64 PNG."""
    is_drum = _is_drum_track(midi)
    notes = _collect_notes(midi, config)
    y_min, y_max = _resolve_range(notes, config)
    end_value = _end_value(notes, config, midi)
    width = _width(config, end_value)

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(width, config.height))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(FACE_COLOR)

        target_set, is_perc, active_family = _resolve_highlight(notes, config.highlight_family)

        if config.daw_style and not is_drum:
            key_width = max(0.4, end_value * 0.025)
            ax.set_xlim(-key_width, end_value)
        else:
            ax.set_xlim(0, end_value)
        ax.set_ylim(y_min, y_max)

        if config.daw_style:
            draw_daw_background(ax, notes, end_value, y_min, y_max, is_drum)
            if not is_drum:
                draw_keyboard(ax, y_min, y_max, end_value)
        else:
            draw_c_note_guides(ax, y_min, y_max)

        if config.time_axis == "beats" and config.show_beat_grid:
            draw_beat_grid(ax, end_value)

        out_of_bounds = draw_notes(ax, notes, y_min, y_max, target_set, is_perc, config.show_velocity, config.daw_style)

        if config.show_controls:
            unique_pitches = {n["pitch"] for n in notes}
            draw_axes(ax, config, y_min, y_max, end_value, is_drum, unique_pitches)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        draw_title(ax, config, out_of_bounds)
        finish_spines(ax, config, config.daw_style)

        if not config.daw_style:
            fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            dpi=config.dpi,
        )
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    finally:
        if fig is not None:
            plt.close(fig)

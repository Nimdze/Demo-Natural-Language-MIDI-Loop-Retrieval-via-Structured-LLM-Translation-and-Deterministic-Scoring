"""Low-level matplotlib drawing primitives for piano rolls."""

from collections.abc import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .config import PianoRollConfig
from .styles import (
    ALL_STANDARD_PITCHES,
    DAW_DRUM_NAMES,
    DRUM_ROW_EVEN,
    DRUM_ROW_ODD,
    FACE_COLOR,
    FAMILY_COLORS,
    GM_DRUM_NAMES,
    GRID_COLOR,
    KEYBOARD_TEXT_COLOR,
    NOTE_EDGE_DARK,
    NOTE_EDGE_HIGHLIGHT,
    OUT_OF_BOUNDS_COLOR,
    PIANO_BLACK_KEY,
    PIANO_ROW_DARK,
    PIANO_ROW_LIGHT,
    PIANO_WHITE_KEY,
    SPINE_COLOR,
    VELOCITY_OFF_COLOR,
    _daw_family_label,
    _pitch_family,
)


def draw_daw_background(ax, notes: list[dict], x_max: float, y_min: int, y_max: int, is_drum: bool) -> None:
    ax.set_facecolor(FACE_COLOR)
    if not is_drum:
        for p in range(int(y_min), int(y_max) + 1):
            is_black = (p % 12) in [1, 3, 6, 8, 10]
            color = PIANO_ROW_DARK if is_black else PIANO_ROW_LIGHT
            ax.add_patch(
                patches.Rectangle(
                    (0, p - 0.5),
                    x_max,
                    1.0,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.5,
                    zorder=0,
                )
            )
        return
    unique_pitches = sorted({n["pitch"] for n in notes})
    for i, p in enumerate(unique_pitches):
        if y_min <= p <= y_max:
            color = DRUM_ROW_EVEN if i % 2 == 0 else DRUM_ROW_ODD
            ax.add_patch(
                patches.Rectangle(
                    (0, p - 0.5),
                    x_max,
                    1.0,
                    facecolor=color,
                    edgecolor="none",
                    zorder=0,
                )
            )
            ax.axhline(y=p - 0.5, color="#111111", linewidth=1, zorder=1)


def draw_keyboard(ax, y_min: int, y_max: int, x_max: float) -> None:
    key_width = max(0.4, x_max * 0.025)
    for p in range(int(y_min), int(y_max) + 1):
        is_black = (p % 12) in [1, 3, 6, 8, 10]
        face_color = PIANO_BLACK_KEY if is_black else PIANO_WHITE_KEY
        ax.add_patch(
            patches.Rectangle(
                (-key_width, p - 0.5),
                key_width,
                1.0,
                facecolor=face_color,
                edgecolor="#333333",
                linewidth=0.5,
                zorder=5,
                clip_on=False,
            )
        )
        if p % 12 == 0:
            octave = (p // 12) - 1
            ax.text(
                -key_width * 0.85,
                p,
                f"C{octave}",
                color=KEYBOARD_TEXT_COLOR,
                fontsize=8,
                fontweight="bold",
                va="center",
                zorder=6,
            )
    ax.axvline(x=0, color="#555555", linewidth=2, zorder=6)


def draw_beat_grid(ax, end_beats: float) -> None:
    grid_points = np.arange(0.0, end_beats + 0.25, 0.25)
    for b in grid_points:
        rem = round(b % 1.0, 2)
        if rem == 0.0:
            alpha = 0.5 if round(b) % 4 == 0 else 0.2
            ax.axvline(x=b, color="white", alpha=alpha, linewidth=0.5, zorder=2, linestyle="--")
        elif rem == 0.5:
            ax.axvline(x=b, color="white", alpha=0.10, linewidth=0.5, zorder=2, linestyle="-.")
        elif rem in [0.25, 0.75]:
            ax.axvline(x=b, color="white", alpha=0.05, linewidth=0.4, zorder=2, linestyle=":")


def draw_c_note_guides(ax, y_min: int, y_max: int) -> None:
    for c_note in range(0, 128, 12):
        if y_min <= c_note <= y_max:
            ax.axhline(y=c_note, color="white", alpha=0.15, linewidth=0.5, zorder=0)


def draw_notes(
    ax,
    notes: list[dict],
    y_min: int,
    y_max: int,
    target_pitches: set[int],
    is_perc_highlight: bool,
    show_velocity: bool,
    daw_style: bool,
) -> bool:
    out_of_bounds = any(n["pitch"] < y_min or n["pitch"] > y_max for n in notes)
    cmap = plt.colormaps["rainbow"]

    for note in notes:
        pitch = note["pitch"]
        is_oob = pitch < y_min or pitch > y_max
        is_drum = note["is_drum"]

        if is_perc_highlight:
            is_highlighted = is_drum and pitch not in ALL_STANDARD_PITCHES
        else:
            is_highlighted = pitch in target_pitches

        if is_oob:
            color = OUT_OF_BOUNDS_COLOR
        elif show_velocity:
            color = cmap(note["velocity"] / 127.0)
        else:
            color = VELOCITY_OFF_COLOR

        if is_highlighted:
            edgecolor = NOTE_EDGE_HIGHLIGHT
            linewidth = 1.5
            zorder = 10
        elif is_oob:
            edgecolor = "#FFFFFF"
            linewidth = 1.0
            zorder = 1
        else:
            edgecolor = NOTE_EDGE_DARK
            linewidth = 0.5
            zorder = 2

        if daw_style:
            y = pitch - 0.35
            height = 0.7
        else:
            y = pitch - 0.5
            height = 1.0

        ax.add_patch(
            patches.Rectangle(
                (note["start"], y),
                max(0.001, note["end"] - note["start"]),
                height,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
        )

    return out_of_bounds


def draw_axes(
    ax,
    config: PianoRollConfig,
    y_min: int,
    y_max: int,
    end_value: float,
    is_drum: bool,
    unique_pitches: Iterable[int],
) -> None:
    if config.time_axis == "beats":
        tick_spacing = max(1.0, round(end_value / 10.0))
        ax.set_xticks(np.arange(0.0, end_value + 0.1, tick_spacing))

    ax.set_xlabel(
        "Time (beats)" if config.time_axis == "beats" else "Time (s)",
        color="white",
        fontsize=8,
    )
    ax.set_ylabel(
        "MIDI Pitch" if not is_drum else "Drum",
        color="white",
        fontsize=8,
    )
    ax.tick_params(colors="white", labelsize=7)

    if is_drum:
        active = sorted(unique_pitches)
        if config.daw_style:
            labels = []
            for p in active:
                family = _daw_family_label(p)
                name = DAW_DRUM_NAMES.get(p, f"Perc {p}")
                labels.append(f"{family.ljust(6)} | {name}")
            ax.set_yticks(active)
            ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
            for ytick, p in zip(ax.get_yticklabels(), active):
                ytick.set_color(FAMILY_COLORS.get(_pitch_family(p), "#888888"))
        else:
            labels = [f"{GM_DRUM_NAMES.get(p, f'Pitch {p}')}  " for p in active]
            ax.set_yticks(active)
            ax.set_yticklabels(labels)
            for i, p in enumerate(active):
                ax.get_yticklabels()[i].set_color(FAMILY_COLORS.get(_pitch_family(p), "#888888"))
    else:
        if config.daw_style:
            ax.set_yticks([])
        else:
            c_pitches = [p for p in range(0, 128, 12) if y_min <= p <= y_max]
            ax.set_yticks(c_pitches)
            ax.set_yticklabels([f"C{(p // 12) - 1}" for p in c_pitches])


def draw_title(ax, config: PianoRollConfig, out_of_bounds: bool) -> None:
    title_text = config.title
    if title_text and config.title_stat_name and config.title_stat_value is not None:
        title_text = f"{title_text}\n{config.title_stat_name}: {config.title_stat_value}"
    title_color = OUT_OF_BOUNDS_COLOR if out_of_bounds else "white"
    if title_text:
        ax.set_title(title_text, fontsize=10, color=title_color, fontweight="bold", pad=8)


def finish_spines(ax, config: PianoRollConfig, daw_style: bool) -> None:
    for spine in ax.spines.values():
        if daw_style:
            spine.set_color(GRID_COLOR)
        else:
            spine.set_color(SPINE_COLOR)

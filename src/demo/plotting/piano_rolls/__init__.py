"""Piano roll rendering package."""

import pretty_midi
import streamlit as st

from .config import PianoRollConfig
from .presets import gallery_config, inspector_config, mosaic_config
from .renderer import _midi_hash, render


@st.cache_data(show_spinner=False, hash_funcs={pretty_midi.PrettyMIDI: _midi_hash})
def _render_cached(midi: pretty_midi.PrettyMIDI, config: PianoRollConfig) -> str:
    return render(midi, config)


def plot_mosaic_roll_to_base64(
    midi: pretty_midi.PrettyMIDI,
    title: str,
    stat_name: str = "",
    stat_value: str = "",
    y_min: int = 21,
    y_max: int = 108,
    highlight_family: str | None = None,
    show_velocity: bool = True,
    width: float | None = None,
    height: float = 3.5,
) -> str:
    cfg = mosaic_config(
        title=title,
        title_stat_name=stat_name or None,
        title_stat_value=stat_value or None,
        y_min=y_min,
        y_max=y_max,
        highlight_family=highlight_family,
        show_velocity=show_velocity,
        width=width,
        height=height,
    )
    return _render_cached(midi, cfg)


def plot_inspector_roll_to_base64(
    midi: pretty_midi.PrettyMIDI,
    title: str | None = None,
    y_min: int = 21,
    y_max: int = 108,
    range_mode: str = "auto",
    highlight_family: str | None = None,
    show_velocity: bool = True,
    height: float = 4.0,
) -> str:
    cfg = inspector_config(
        title=title,
        y_min=y_min,
        y_max=y_max,
        range_mode=range_mode,
        highlight_family=highlight_family,
        show_velocity=show_velocity,
        height=height,
    )
    return _render_cached(midi, cfg)


def plot_gallery_roll_to_base64(
    midi: pretty_midi.PrettyMIDI,
    title: str | None = None,
    y_min: int = 21,
    y_max: int = 108,
    range_mode: str = "auto",
    highlight_family: str | None = None,
    show_velocity: bool = True,
    height: float = 4.0,
) -> str:
    cfg = gallery_config(
        title=title,
        y_min=y_min,
        y_max=y_max,
        range_mode=range_mode,
        highlight_family=highlight_family,
        show_velocity=show_velocity,
        height=height,
    )
    return _render_cached(midi, cfg)


# Backward-compatible alias from the previous refactor.
def plot_piano_roll_to_base64(
    midi: pretty_midi.PrettyMIDI,
    y_min: int = 21,
    y_max: int = 108,
    highlight_family: str | None = None,
    show_velocity: bool = True,
    width: float | None = None,
    height: float = 3.5,
    title: str | None = None,
    show_controls: bool = True,
) -> str:
    cfg = PianoRollConfig(
        time_axis="seconds",
        range_mode="custom",
        y_min=y_min,
        y_max=y_max,
        show_velocity=show_velocity,
        show_controls=show_controls,
        daw_style=show_controls,
        show_beat_grid=show_controls,
        highlight_family=highlight_family,
        title=title,
        width=width,
        height=height,
    )
    return _render_cached(midi, cfg)

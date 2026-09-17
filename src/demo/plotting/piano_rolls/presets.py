"""Preset configurations for common piano-roll views."""

from .config import PianoRollConfig


def mosaic_config(
    y_min: int = 21,
    y_max: int = 108,
    highlight_family: str | None = None,
    show_velocity: bool = True,
    width: float | None = None,
    height: float = 3.5,
    title: str | None = None,
    title_stat_name: str | None = None,
    title_stat_value: str | None = None,
) -> PianoRollConfig:
    return PianoRollConfig(
        time_axis="seconds",
        range_mode="custom",
        y_min=y_min,
        y_max=y_max,
        show_velocity=show_velocity,
        show_controls=False,
        daw_style=False,
        show_beat_grid=False,
        highlight_family=highlight_family,
        title=title,
        title_stat_name=title_stat_name,
        title_stat_value=title_stat_value,
        width=width,
        height=height,
    )


def inspector_config(
    y_min: int = 21,
    y_max: int = 108,
    range_mode: str = "auto",
    highlight_family: str | None = None,
    show_velocity: bool = True,
    title: str | None = None,
    height: float = 4.0,
) -> PianoRollConfig:
    return PianoRollConfig(
        time_axis="beats",
        range_mode=range_mode,  # type: ignore[arg-type]
        y_min=y_min,
        y_max=y_max,
        show_velocity=show_velocity,
        show_controls=True,
        daw_style=True,
        show_beat_grid=True,
        highlight_family=highlight_family,
        title=title,
        height=height,
    )


def gallery_config(
    y_min: int = 21,
    y_max: int = 108,
    range_mode: str = "auto",
    highlight_family: str | None = None,
    show_velocity: bool = True,
    title: str | None = None,
    height: float = 4.0,
) -> PianoRollConfig:
    return PianoRollConfig(
        time_axis="beats",
        range_mode=range_mode,  # type: ignore[arg-type]
        y_min=y_min,
        y_max=y_max,
        show_velocity=show_velocity,
        show_controls=True,
        daw_style=True,
        show_beat_grid=True,
        highlight_family=highlight_family,
        title=title,
        height=height,
    )

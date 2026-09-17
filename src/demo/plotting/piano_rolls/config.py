from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PianoRollConfig:
    time_axis: Literal["seconds", "beats"] = "seconds"
    range_mode: Literal["auto", "standard", "full", "custom"] = "standard"
    y_min: int = 21
    y_max: int = 108
    show_velocity: bool = True
    show_controls: bool = True
    daw_style: bool = False
    show_beat_grid: bool = False
    highlight_family: str | None = None
    title: str | None = None
    title_stat_name: str | None = None
    title_stat_value: str | None = None
    width: float | None = None
    height: float = 3.5
    dpi: int = 100

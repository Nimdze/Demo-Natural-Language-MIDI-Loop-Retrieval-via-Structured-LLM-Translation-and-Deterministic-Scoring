"""Grid-assignment debug visualization for groove/rhythm analysis."""

import base64
import io
import sys
import traceback
from typing import Any

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import streamlit as st

ATTEMPT_WINDOW = 0.3
SUCCESS_WINDOW_TIGHT = 0.03125
SUCCESS_WINDOW_LOOSE = 0.0625


def _group_events(pm: pretty_midi.PrettyMIDI) -> list[list[pretty_midi.Note]]:
    all_notes = []
    for instrument in pm.instruments:
        all_notes.extend(instrument.notes)
    if not all_notes:
        return []
    all_notes.sort(key=lambda n: n.start)
    events = [[all_notes[0]]]
    current_start = all_notes[0].start
    for note in all_notes[1:]:
        if note.start - current_start <= 0.05:
            events[-1].append(note)
        else:
            events.append([note])
            current_start = note.start
    return events


def _estimate_bars(max_beat: float, ts_num: int) -> int:
    bars = max(1, int(np.ceil((max_beat - 0.25) / ts_num)))
    if bars == 3:
        bars = 4
    elif 4 < bars < 8:
        bars = 8
    elif 8 < bars < 16:
        bars = 16
    return bars


def calc_grid_metrics(event_beats: np.ndarray, ts_num: int) -> dict[str, float]:
    """Recompute the same groove grid metrics stored by the analyzer."""
    max_beat = float(event_beats.max()) if event_beats.size else 0.0
    bars = _estimate_bars(max_beat, ts_num)
    targets = {
        "odd1": [],
        "even1": [],
        "beat3": [],
        "beats2_4": [],
        "offbeat": [],
    }
    for bar in range(bars):
        bar_start = bar * ts_num
        if bar % 2 == 0:
            targets["odd1"].append(bar_start)
        else:
            targets["even1"].append(bar_start)
        beat3 = bar_start + 2.0
        if beat3 <= max_beat + ts_num:
            targets["beat3"].append(beat3)
        for beat in (1, 3):
            pos = bar_start + beat
            if pos <= max_beat + ts_num:
                targets["beats2_4"].append(pos)
        for beat in range(ts_num):
            pos = bar_start + beat + 0.5
            if pos <= max_beat + ts_num:
                targets["offbeat"].append(pos)

    results: dict[str, float] = {}
    for key, positions in targets.items():
        attempts = 0
        successes = 0
        total = len(positions)
        for target in positions:
            if event_beats.size == 0:
                continue
            min_dist = float(np.abs(event_beats - target).min())
            if min_dist <= ATTEMPT_WINDOW:
                attempts += 1
                success_window = SUCCESS_WINDOW_LOOSE if key == "offbeat" else SUCCESS_WINDOW_TIGHT
                if min_dist <= success_window:
                    successes += 1
        results[f"grid_attempt_pct_{key}"] = (attempts / total * 100) if total else 0.0
        results[f"grid_success_pct_{key}"] = (successes / total * 100) if total else 0.0
    return results


def recalculate_groove_fresh(midi_path: str) -> dict[str, float] | None:
    """Recompute groove stats directly from the MIDI file for comparison."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        if not pm.instruments:
            st.warning("No instruments found in MIDI")
            return None
        if len(pm.time_signature_changes) > 1:
            st.warning("Multiple time signatures found; using only the first one for this grid.")
        ts_num = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4
        events = _group_events(pm)
        if not events:
            st.warning("No notes found")
            return None
        event_beats = np.array([pm.time_to_tick(group[0].start) / pm.resolution for group in events])
        return calc_grid_metrics(event_beats, ts_num)
    except Exception as e:
        st.error(f"Error recalculating groove: {e}")
        st.code(traceback.format_exc())
        return None


def render_grid_debug(midi_path: str, rhythm_stats: dict[str, Any] | None) -> dict[str, Any] | None:
    """Render a grid-assignment debug plot and return data tables."""
    del rhythm_stats  # kept for API compatibility; the plot computes directly from MIDI
    fig = None
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        if not pm.instruments:
            st.warning("No instruments found in MIDI")
            return None

        if len(pm.time_signature_changes) > 1:
            st.warning("Multiple time signatures found; using only the first one for this grid.")
        ts_num = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4

        tempo_changes = pm.get_tempo_changes()
        bpm = 120.0
        if tempo_changes[1] is not None and len(tempo_changes[1]) > 0:
            bpm = float(tempo_changes[1][0])
            if len(tempo_changes[1]) > 1:
                st.warning("Multiple tempo changes found; using only the first BPM for this grid.")

        events = _group_events(pm)
        if not events:
            st.warning("No notes found")
            return None

        all_onsets_beats = sorted({pm.time_to_tick(group[0].start) / pm.resolution for group in events})
        max_beat = max(all_onsets_beats) if all_onsets_beats else 0.0
        bars = _estimate_bars(max_beat, ts_num)

        grid_categories = {
            "Odd-Bar Beat 1": [],
            "Even-Bar Beat 1": [],
            "Beat 3": [],
            "Beats 2 & 4": [],
            "Off-Beat (&)": [],
        }
        for bar in range(bars):
            bar_start = bar * ts_num
            if bar % 2 == 0:
                grid_categories["Odd-Bar Beat 1"].append(bar_start)
            else:
                grid_categories["Even-Bar Beat 1"].append(bar_start)
            beat3 = bar_start + 2.0
            if beat3 <= max_beat + ts_num:
                grid_categories["Beat 3"].append(beat3)
            for beat in (1, 3):
                pos = bar_start + beat
                if pos <= max_beat + ts_num:
                    grid_categories["Beats 2 & 4"].append(pos)
            for beat in range(ts_num):
                pos = bar_start + beat + 0.5
                if pos <= max_beat + ts_num:
                    grid_categories["Off-Beat (&)"].append(pos)

        grid_assignments = []
        assigned_events = set()
        for category, positions in grid_categories.items():
            success_window = SUCCESS_WINDOW_LOOSE if category == "Off-Beat (&)" else SUCCESS_WINDOW_TIGHT
            for target in positions:
                distances = [(abs(event - target), event) for event in all_onsets_beats]
                distances.sort()
                if distances and distances[0][0] <= ATTEMPT_WINDOW:
                    min_dist, closest_event = distances[0]
                    is_success = min_dist <= success_window
                    grid_assignments.append((target, category, closest_event, min_dist, is_success))
                    assigned_events.add(closest_event)

        category_colors = {
            "Odd-Bar Beat 1": "#FF6B6B",
            "Even-Bar Beat 1": "#FF9F43",
            "Beat 3": "#54A0FF",
            "Beats 2 & 4": "#5F27CD",
            "Off-Beat (&)": "#10AC84",
        }

        fig, ax = plt.subplots(figsize=(16, 10))
        event_y_positions = {event: i * 0.3 for i, event in enumerate(sorted(all_onsets_beats))}
        grid_y_base = -2

        for category_index, category in enumerate(grid_categories.keys()):
            y_pos = grid_y_base + category_index * (-0.8)
            color = category_colors[category]
            cat_assignments = [a for a in grid_assignments if a[1] == category]
            for grid_pos, _, event, _, is_success in cat_assignments:
                ax.axvline(x=grid_pos, color=color, alpha=0.3, linestyle="--", linewidth=1)
                event_y = event_y_positions.get(event, 0)
                line_style = "-" if is_success else ":"
                line_width = 2 if is_success else 1
                alpha = 0.8 if is_success else 0.4
                ax.plot(
                    [event, grid_pos],
                    [event_y, y_pos],
                    color=color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=alpha,
                )
                ax.scatter(
                    [grid_pos],
                    [y_pos],
                    color=color,
                    s=100 if is_success else 60,
                    marker="*" if is_success else "o",
                    zorder=5,
                    edgecolors="black",
                    linewidths=1,
                )

        for event in all_onsets_beats:
            y = event_y_positions[event]
            is_assigned = event in assigned_events
            ax.scatter(
                [event],
                [y],
                color="white" if is_assigned else "gray",
                s=80 if is_assigned else 40,
                zorder=4 if is_assigned else 3,
                edgecolors="black" if is_assigned else "none",
                linewidths=1.5 if is_assigned else 0,
                alpha=1.0 if is_assigned else 0.5,
            )

        ax.set_xlabel("Beat Position", fontsize=12)
        ax.set_ylabel("Events / Grid Categories", fontsize=12)
        ax.set_title(
            f"Grid Assignment Debug\nBars: {bars}, BPM: {bpm:.1f}, Events: {len(all_onsets_beats)}",
            fontsize=14,
            fontweight="bold",
        )

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
                label="Event (assigned)",
            ),
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Event (unassigned)"
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="black",
                markeredgecolor="gold",
                markersize=12,
                label="Grid (SUCCESS)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
                label="Grid (ATTEMPT ONLY)",
            ),
        ]
        for cat, color in category_colors.items():
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, label=cat))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_yticks([i * 0.3 for i in range(len(all_onsets_beats))])
        ax.set_yticklabels([f"{b:.2f}" for b in all_onsets_beats], fontsize=8)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0E1117", edgecolor="none")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()

        summary_data = []
        for category in grid_categories.keys():
            cat_assignments = [a for a in grid_assignments if a[1] == category]
            total = len(cat_assignments)
            successes = sum(1 for a in cat_assignments if a[4])
            pct = (successes / total * 100) if total else 0
            summary_data.append(
                {
                    "category": category,
                    "grid_lines": len(grid_categories[category]),
                    "attempts": total,
                    "successes": successes,
                    "success_pct": f"{pct:.1f}%",
                }
            )

        detail_data = []
        for grid_pos, category, event, dist, is_success in sorted(grid_assignments):
            detail_data.append(
                {
                    "grid_pos": f"{grid_pos:.2f}",
                    "category": category,
                    "event": f"{event:.3f}",
                    "distance": f"{dist:.4f} ({dist * 60000 / bpm:.1f}ms @ {bpm:.1f}BPM)",
                    "result": "✅ SUCCESS" if is_success else "⚠️ ATTEMPT",
                }
            )

        return {
            "base64": b64,
            "bars": bars,
            "bpm": bpm,
            "events": len(all_onsets_beats),
            "summary": summary_data,
            "detail": detail_data,
        }
    except Exception as e:
        st.error(f"Error creating grid debug visualization: {e}")
        st.code(traceback.format_exc())
        return None
    finally:
        if fig is not None:
            plt.close(fig)

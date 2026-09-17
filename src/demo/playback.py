"""On-demand MIDI playback.

Primary: FluidSynth (libfluidsynth + soundfont) for high-quality samples.
Fallback: pretty_midi's built-in synthesizer (no soundfont required).

Degrades gracefully (returns None) if neither is available.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

_DEMO_ROOT = Path(__file__).resolve().parent.parent.parent

_SF = None
_SYNTH = None
_SFID = None
_SAMPLERATE = 22050


def _find_soundfont() -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("MIDI_SOUNDFONT")
    if env:
        candidates.append(Path(env))
    candidates.append(_DEMO_ROOT / "soundfonts")
    candidates.append(Path.cwd() / "soundfonts")
    for folder in candidates:
        if folder.is_dir():
            for sf in sorted(folder.glob("*.sf3")) + sorted(folder.glob("*.sf2")):
                return sf
    return None


def _ensure_fluidsynth_lib() -> None:
    if os.environ.get("HOMEBREW_PREFIX"):
        return
    for prefix in ("/opt/miniconda3/envs/fluidsynth_env", "/opt/homebrew"):
        if os.path.exists(os.path.join(prefix, "lib", "libfluidsynth.dylib")):
            os.environ["HOMEBREW_PREFIX"] = prefix
            return


def _load_fluidsynth_engine() -> None:
    global _SF, _SYNTH, _SFID
    _ensure_fluidsynth_lib()
    if _SF is None:
        import fluidsynth
        _SF = fluidsynth
    if _SYNTH is None:
        _SYNTH = _SF.Synth(samplerate=_SAMPLERATE)
    if _SFID is None:
        path = _find_soundfont()
        if path is None:
            raise FileNotFoundError("No .sf2 soundfont found (set MIDI_SOUNDFONT).")
        _SFID = _SYNTH.sfload(str(path))
        for chan in range(16):
            _SYNTH.sfont_select(chan, _SFID)


def _fluidsynth_render(midi_path: str | Path) -> bytes | None:
    """Render a MIDI file to WAV using the fluidsynth CLI (Rosetta).

    Returns WAV bytes, or None if the CLI/soundfont is unavailable or fails.
    """
    sf = _find_soundfont()
    if sf is None:
        return None
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        cmd = [
            "fluidsynth",
            "-ni",
            "-g", "1.0",
            "-F", str(tmp),
            str(sf),
            str(midi_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            return None
        data = tmp.read_bytes()
        return data if data[:4] == b"RIFF" else None
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def _pretty_midi_render(midi_path: str | Path) -> bytes | None:
    """Fallback render via pretty_midi's built-in synthesizer."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.synthesize()
        if audio is None or len(audio) == 0:
            return None
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        sample_rate = 44100
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()
    except Exception:
        return None


def midi_to_wav_bytes(midi_path: str | Path) -> bytes | None:
    """Synthesize a MIDI file to WAV bytes, or None if unavailable."""
    result = _fluidsynth_render(midi_path)
    if result is not None:
        return result
    return _pretty_midi_render(midi_path)

"""Standalone debug tool: test playback with instrument remapping.

Run: streamlit run src/demo/debug_playback.py
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path

import pretty_midi
import streamlit as st

_DEMO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASETS_DIR = _DEMO_ROOT / "datasets"

_GM_INSTRUMENTS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavi",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass+lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]
_DRUM_KITS = {
    0: "Standard Kit", 8: "Room Kit", 16: "Power Kit",
    24: "Electronic Kit", 32: "TR-808 Kit", 40: "Brush Kit",
    48: "Orchestra Kit", 56: "SoundFX Kit",
}


def _find_soundfont() -> Path | None:
    sf_dir = _DEMO_ROOT / "soundfonts"
    if sf_dir.is_dir():
        for sf in sorted(sf_dir.glob("*.sf3")) + sorted(sf_dir.glob("*.sf2")):
            return sf
    return None


def _list_midi_files() -> list[tuple[str, Path, bool, bool]]:
    results = []
    for ds_dir in sorted(_DATASETS_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        clean = ds_dir / "clean"
        if not clean.exists():
            continue
        for p in sorted(clean.rglob("*.mid")):
            try:
                pm = pretty_midi.PrettyMIDI(str(p))
                has_pitched = any(not inst.is_drum and inst.notes for inst in pm.instruments)
                has_drums = any(inst.is_drum and inst.notes for inst in pm.instruments)
            except Exception:
                has_pitched = False
                has_drums = False
            label = str(p.relative_to(_DEMO_ROOT))
            results.append((label, p, has_pitched, has_drums))
    results.sort(key=lambda x: (0 if x[2] else 1, x[0]))
    return results


def _remap_midi(midi_path: Path, pitched_prog: int = -1, drum_prog: int | None = None) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    for inst in pm.instruments:
        if inst.is_drum:
            if drum_prog is not None:
                inst.program = drum_prog
        else:
            if pitched_prog >= 0:
                inst.program = pitched_prog
    tmp = Path(tempfile.mktemp(suffix=".mid"))
    pm.write(str(tmp))
    return tmp


def _render_wav(midi_path: Path) -> bytes | None:
    sf = _find_soundfont()
    if sf is None:
        return None
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        result = subprocess.run(
            ["fluidsynth", "-ni", "-g", "1.0", "-F", str(tmp), str(sf), str(midi_path)],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            st.error(f"FluidSynth failed (rc={result.returncode}): {result.stderr.decode()[:300]}")
            return None
        data = tmp.read_bytes()
        if data[:4] != b"RIFF":
            st.error(f"FluidSynth output not RIFF WAV: {data[:20]}")
            return None
        return data
    except Exception as e:
        st.error(f"Render exception: {e}")
        return None
    finally:
        tmp.unlink(missing_ok=True)


def _kit_name(prog: int) -> str:
    return _DRUM_KITS.get(prog, f"Kit {prog}")


st.set_page_config(page_title="Debug MIDI Playback", layout="wide")
st.title("Debug MIDI Playback")

sf = _find_soundfont()
if sf:
    st.success(f"Soundfont: **{sf.name}**")
else:
    st.error("No soundfont found in soundfonts/")
    st.stop()

all_entries = _list_midi_files()
if not all_entries:
    st.error("No MIDI files found in datasets/")
    st.stop()

file_labels = [e[0] for e in all_entries]
file_paths = [e[1] for e in all_entries]

selected_file = st.selectbox("MIDI file", file_labels, index=0)
midi_path = file_paths[file_labels.index(selected_file)]

pm = pretty_midi.PrettyMIDI(str(midi_path))

st.subheader("Tracks in file")
has_pitched = False
has_drums = False
for inst in pm.instruments:
    pitched = not inst.is_drum and len(inst.notes) > 0
    drum = inst.is_drum and len(inst.notes) > 0
    has_pitched = has_pitched or pitched
    has_drums = has_drums or drum
    if inst.is_drum:
        name = _kit_name(inst.program)
        label = f"🔴 Drums ({name})"
    else:
        name = _GM_INSTRUMENTS[inst.program] if 0 <= inst.program < 128 else f"prog_{inst.program}"
        label = f"🟢 {name}"
    if inst.notes:
        pitches = [n.pitch for n in inst.notes]
        vel = [n.velocity for n in inst.notes]
        st.text(f"  {label} — program={inst.program}, {len(inst.notes)} notes, pitch [{min(pitches)}-{max(pitches)}], vel [{min(vel)}-{max(vel)}]")

st.subheader("Instrument remapping")
st.caption("Selectors appear based on what the file contains. Each defaults to \"Original\" (no change).")

pitched_changed = False
drum_changed = False
pitched_prog = -1
drum_prog = None

if has_pitched:
    pitched_opts = [("Original (keep file's instrument)", -1)] + [(f"{i}: {name}", i) for i, name in enumerate(_GM_INSTRUMENTS)]
    chosen_pitched = st.selectbox("Pitched instrument", [l for l, _ in pitched_opts], index=0, key="pitched")
    pitched_prog = [v for _, v in pitched_opts][[l for l, _ in pitched_opts].index(chosen_pitched)]
    pitched_changed = pitched_prog != -1

if has_drums:
    drum_opt_list = [("Original (keep file's kit)", -1)] + sorted(
        [(f"Kit {prog}: {name}", prog) for prog, name in _DRUM_KITS.items()],
        key=lambda x: x[1],
    )
    chosen_drum = st.selectbox("Drum kit", [l for l, _ in drum_opt_list], index=0, key="drum")
    drum_prog_v = [v for _, v in drum_opt_list][[l for l, _ in drum_opt_list].index(chosen_drum)]
    if drum_prog_v != -1:
        drum_changed = True
        drum_prog = drum_prog_v

if not pitched_changed and not drum_changed:
    render_path = midi_path
    st.info("Playing **original file** — no remapping applied")
else:
    parts = []
    if pitched_changed:
        parts.append(f"pitched → **{_GM_INSTRUMENTS[pitched_prog]}** (prog {pitched_prog})")
    if drum_changed:
        parts.append(f"drums → **{_kit_name(drum_prog)}** (prog {drum_prog})")
    st.info("Remapping: " + ", ".join(parts))
    with st.spinner("Remapping MIDI..."):
        render_path = _remap_midi(midi_path, pitched_prog if pitched_changed else -1, drum_prog if drum_changed else None)
    pm2 = pretty_midi.PrettyMIDI(str(render_path))
    st.subheader("Remapped tracks")
    for inst in pm2.instruments:
        if inst.is_drum:
            name = _kit_name(inst.program)
            label = f"🔴 Drums ({name})"
        else:
            name = _GM_INSTRUMENTS[inst.program] if 0 <= inst.program < 128 else f"prog_{inst.program}"
            label = f"🟢 {name}"
        st.text(f"  {label} — program={inst.program}, {len(inst.notes)} notes")

with st.spinner("Rendering WAV via FluidSynth..."):
    wav = _render_wav(render_path)
    if render_path != midi_path:
        render_path.unlink(missing_ok=True)

if wav:
    h = hashlib.md5(wav).hexdigest()
    st.success(f"Rendered **{len(wav)} bytes** (MD5: {h})")
    st.audio(wav, format="audio/wav")

    if pitched_changed or drum_changed:
        wav_orig = _render_wav(midi_path)
        if wav_orig:
            h_orig = hashlib.md5(wav_orig).hexdigest()
            st.write(f"Original MD5: **{h_orig}**")
            if h == h_orig:
                st.error("❌ Remapped audio is IDENTICAL to original!")
            else:
                st.success("✅ Remapped audio is DIFFERENT from original!")

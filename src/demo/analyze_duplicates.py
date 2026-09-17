"""Analyze duplicate stems across datasets.

A "duplicate" is a MIDI file with the same note events (pitch, start, end,
velocity) as another file, differing only in metadata (program, tempo, filename,
etc.). This identifies stems that are the same across datasets or within a dataset.

Run: .venv/bin/python src/demo/analyze_duplicates.py
"""

import hashlib
from collections import defaultdict
from pathlib import Path

import pretty_midi

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


def _note_hash(midi_path: Path) -> str:
    """Hash of note events only (pitch, start, end, velocity, is_drum).

    Ignores: program number, tempo, time signature, track names, etc.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    parts = []
    for inst in sorted(pm.instruments, key=lambda x: (x.is_drum, x.program)):
        for note in sorted(inst.notes, key=lambda n: (n.start, n.pitch)):
            parts.append(f"{note.pitch},{note.start:.4f},{note.end:.4f},{note.velocity},{int(inst.is_drum)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _metadata(midi_path: Path) -> dict:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    insts = []
    for inst in pm.instruments:
        name = _GM_INSTRUMENTS[inst.program] if 0 <= inst.program < 128 else f"prog_{inst.program}"
        label = f"Drums({name})" if inst.is_drum else name
        insts.append(f"{label}({inst.program})")
    try:
        tempo = pm.estimate_tempo()
    except Exception:
        tempo = None
    return {
        "instruments": ", ".join(insts),
        "tempo": round(tempo, 1) if tempo else "?",
        "num_notes": sum(len(inst.notes) for inst in pm.instruments),
    }


def main():
    all_files = sorted(_DATASETS_DIR.rglob("*.mid"))
    print(f"Scanning {len(all_files)} MIDI files...\n")

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in all_files:
        try:
            h = _note_hash(p)
            groups[h].append(p)
        except Exception as e:
            print(f"  SKIP {p.relative_to(_DATASETS_DIR.parent)}: {e}")

    # Filter to groups with >1 file
    dups = {h: paths for h, paths in groups.items() if len(paths) > 1}

    total_dup_files = sum(len(paths) for paths in dups.values())
    unique_groups = len(dups)

    print(f"{'='*70}")
    print(f"DUPLICATE STEM ANALYSIS")
    print(f"{'='*70}")
    print(f"Total files:            {len(all_files)}")
    print(f"Files in dup groups:    {total_dup_files} ({total_dup_files - unique_groups} excess)")
    print(f"Unique content groups:  {len(groups)}")
    print(f"Duplicate groups:       {unique_groups}")

    if not dups:
        print("\nNo duplicates found.")
        return

    print(f"\n{'='*70}")
    print(f"DUPLICATE GROUPS (sorted by size, largest first)")
    print(f"{'='*70}")

    sorted_dups = sorted(dups.items(), key=lambda x: -len(x[1]))
    for h, paths in sorted_dups:
        meta = _metadata(paths[0])
        print(f"\n  {len(paths)} copies — {meta['num_notes']} notes, tempo={meta['tempo']}, {meta['instruments']}")
        for p in paths:
            rel = p.relative_to(_DATASETS_DIR)
            print(f"    {rel}")

    sum_excess = sum(len(paths) - 1 for _, paths in sorted_dups)
    print(f"\n{'='*70}")
    print(f"Summary: {total_dup_files} files in {unique_groups} groups, {sum_excess} excess files")


if __name__ == "__main__":
    main()

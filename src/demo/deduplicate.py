"""Remove duplicate MIDI files from datasets.

Keeps the first file per content-hash group. Prefers non-bpm120 files over
_bpm120.0.mid files. Deletes excess copies in place.

Run: .venv/bin/python src/demo/deduplicate.py
"""

import hashlib
from collections import defaultdict
from pathlib import Path

import pretty_midi

_DEMO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASETS_DIR = _DEMO_ROOT / "datasets"


def _note_hash(midi_path: Path) -> str:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    parts = []
    for inst in sorted(pm.instruments, key=lambda x: (x.is_drum, x.program)):
        for note in sorted(inst.notes, key=lambda n: (n.start, n.pitch)):
            parts.append(f"{note.pitch},{note.start:.4f},{note.end:.4f},{note.velocity},{int(inst.is_drum)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def main():
    all_files = sorted(_DATASETS_DIR.rglob("*.mid"))
    print(f"Scanning {len(all_files)} MIDI files...")

    groups: dict[str, list[Path]] = defaultdict(list)
    errors = 0
    for p in all_files:
        try:
            h = _note_hash(p)
            groups[h].append(p)
        except Exception as e:
            errors += 1
            # Only print first few errors
            if errors <= 5:
                print(f"  Error reading {p.relative_to(_DATASETS_DIR)}: {e}")

    dups = {h: paths for h, paths in groups.items() if len(paths) > 1}
    total_excess = sum(len(paths) - 1 for paths in dups.values())

    if errors:
        print(f"  ({errors} files had errors, skipped)")

    if not dups:
        print("No duplicates found.")
        return

    deleted = 0
    for h, paths in dups.items():
        # Sort: prefer non-bpm120, shorter paths first as tiebreaker
        sorted_paths = sorted(paths, key=lambda p: (
            1 if "_bpm120.0.mid" in p.name else 0,
            len(str(p)),
        ))
        keep = sorted_paths[0]
        for p in sorted_paths[1:]:
            p.unlink()
            deleted += 1
            print(f"  DEL {p.relative_to(_DATASETS_DIR)}  (kept {keep.relative_to(_DATASETS_DIR)})")

    deleted_dirs = 0
    for ds_dir in sorted(_DATASETS_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for clean_dir in [ds_dir / "clean", ds_dir / "raw"]:
            if not clean_dir.exists():
                continue
            for sub in sorted(clean_dir.rglob("*")):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
                    deleted_dirs += 1

    print(f"\nDone. Deleted {deleted} duplicate files, {deleted_dirs} empty dirs.")
    print(f"Files remaining: {len(all_files) - deleted}")


if __name__ == "__main__":
    main()

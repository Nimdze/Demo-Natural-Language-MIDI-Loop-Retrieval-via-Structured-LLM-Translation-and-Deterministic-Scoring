"""Remove DB entries for MIDI files that no longer exist on disk."""
import sqlite3
from pathlib import Path

_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

for db_path in sorted(_DATASETS_DIR.rglob("output/analysis.db")):
    ds = db_path.parent.parent.name
    clean = db_path.parent.parent / "clean"
    conn = sqlite3.connect(str(db_path))
    conn.execute("BEGIN")
    cursor = conn.execute("SELECT id, path FROM files")
    deleted = 0
    for row in cursor.fetchall():
        fid, path = row
        full = clean / path
        if not full.exists():
            conn.execute("DELETE FROM tags WHERE file_id = ?", (fid,))
            conn.execute("DELETE FROM files WHERE id = ?", (fid,))
            deleted += 1
    conn.commit()
    if deleted:
        conn.execute("VACUUM")
        print(f"{ds}: removed {deleted} orphan entries")
    conn.close()

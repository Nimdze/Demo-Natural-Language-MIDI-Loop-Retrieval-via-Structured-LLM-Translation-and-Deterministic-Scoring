"""Load the analyzer taxonomy and database into a search index."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class SearchIndex:
    """In-memory search index built from a taxonomy JSON and a SQLite database."""

    def __init__(
        self,
        db_path: str | Path,
        taxonomy_path: str | Path,
        restrict_to: set[int] | None = None,
    ):
        self.db_path = Path(db_path)
        self.taxonomy_path = Path(taxonomy_path)
        # Optional subset of file_ids this index is allowed to rank.
        # None = all files.
        self.restrict_to = restrict_to

        self.taxonomy: dict[str, dict[str, Any]] = {}
        self._level_position: dict[tuple[str, str], int] = {}
        self._level_global_index: dict[tuple[str, str], int] = {}
        self._level_weight: dict[tuple[str, str], int] = {}

        self.file_tags: dict[int, dict[str, str]] = {}
        self.file_paths: dict[int, str] = {}
        self.file_families: dict[int, str] = {}
        self.inverted_index: dict[tuple[str, str], set[int]] = {}

        self._load_taxonomy()
        self._load_db()

    def _load_taxonomy(self) -> None:
        with open(self.taxonomy_path) as f:
            self.taxonomy = json.load(f)

        for concept_name, data in self.taxonomy.items():
            llm = data.get("llm", {})
            levels = llm.get("levels", [])
            for position, level_entry in enumerate(levels):
                # Each level entry is either [index, name, weight] or a dict.
                if isinstance(level_entry, list) and len(level_entry) >= 3:
                    global_index, level_name, weight = level_entry
                elif isinstance(level_entry, dict):
                    global_index = level_entry.get("index", level_entry.get("global_index", position))
                    level_name = level_entry.get("name", level_entry.get("level_name", ""))
                    weight = level_entry.get("weight", level_entry.get("level_weight", 0))
                else:
                    continue
                key = (concept_name, level_name)
                self._level_position[key] = position
                self._level_global_index[key] = global_index
                self._level_weight[key] = weight

    def _load_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, path, family FROM files"
            )
            for row in cursor.fetchall():
                file_id = row["id"]
                if self.restrict_to is not None and file_id not in self.restrict_to:
                    continue
                self.file_paths[file_id] = row["path"]
                self.file_families[file_id] = row["family"] or "pitched"
                self.file_tags[file_id] = {}

            cursor.execute(
                "SELECT file_id, concept_name, level_name, level_index, level_weight FROM tags"
            )
            for row in cursor.fetchall():
                file_id = row["file_id"]
                if self.restrict_to is not None and file_id not in self.restrict_to:
                    continue
                concept_name = row["concept_name"]
                level_name = row["level_name"]
                if level_name or concept_name:
                    self.file_tags[file_id][concept_name] = level_name
                    key = self.inverted_index.setdefault((concept_name, level_name), set())
                    key.add(file_id)

    def concept_names(self) -> list[str]:
        return list(self.taxonomy.keys())

    def concept_family(self, concept_name: str) -> str:
        data = self.taxonomy.get(concept_name, {})
        llm = data.get("llm", {})
        families = llm.get("instrument_family", [])
        if "drums" in families:
            return "drums"
        return "pitched"

    def concept_families(self, concept_name: str) -> list[str]:
        """Return the raw instrument_family list for a concept."""
        data = self.taxonomy.get(concept_name, {})
        llm = data.get("llm", {})
        families = llm.get("instrument_family", [])
        if isinstance(families, (list, tuple)):
            return list(families)
        return [families] if families else []

    def concept_names_for_family(self, family: str, search_relevant_only: bool = True) -> list[str]:
        """Return concept names whose instrument_family includes the given family.

        Common concepts (e.g., pitched + drums) are returned for both families.
        The reserved ``llm_categories`` entry is always excluded. By default only
        concepts with a non-zero base_weight are returned, matching the
        search-relevant taxonomy used in the LLM prompt.
        """
        if search_relevant_only:
            return self.search_concepts(family)
        return [
            name for name in self.concept_names()
            if name not in ("llm_categories", "llm_subcategories")
            and family in self.concept_families(name)
        ]

    def level_position(self, concept_name: str, level_name: str) -> int | None:
        return self._level_position.get((concept_name, level_name))

    def level_global_index(self, concept_name: str, level_name: str) -> int | None:
        return self._level_global_index.get((concept_name, level_name))

    def level_name_at_index(self, concept_name: str, index: int) -> str | None:
        """Return the level name at the given position index for a concept."""
        names = self.level_names(concept_name)
        if 0 <= index < len(names):
            return names[index]
        return None

    def level_names(self, concept_name: str) -> list[str]:
        data = self.taxonomy.get(concept_name, {})
        levels = data.get("llm", {}).get("levels", [])
        result = []
        for entry in levels:
            if isinstance(entry, list) and len(entry) >= 2:
                result.append(entry[1])
            elif isinstance(entry, dict):
                result.append(entry.get("name", entry.get("level_name", "")))
        return result

    def base_weight(self, concept_name: str) -> float:
        data = self.taxonomy.get(concept_name, {})
        return float(data.get("base_weight", data.get("default_weight", 1.0)))

    def pair_with(self, concept_name: str) -> str | None:
        """Return the concept this one is paired with (e.g. grid attempt/success)."""
        data = self.taxonomy.get(concept_name, {})
        return (data.get("llm") or {}).get("pair_with")

    def files_for_family(self, family: str) -> set[int]:
        return {
            fid for fid, fam in self.file_families.items() if not family or fam == family
        }

    def search_concepts(self, family: str | None = None) -> list[str]:
        """Return concept names suitable for search (excludes llm_categories).

        If ``family`` is provided, only concepts whose instrument_family includes
        that family are returned. Concepts with zero base_weight are excluded.
        """
        names = []
        for cname in self.concept_names():
            if cname in ("llm_categories", "llm_subcategories"):
                continue
            if self.base_weight(cname) <= 0:
                continue
            if family is not None and family not in self.concept_families(cname):
                continue
            names.append(cname)
        return names

    def files_for_tag(self, concept_name: str, level_name: str) -> set[int]:
        return self.inverted_index.get((concept_name, level_name), set())

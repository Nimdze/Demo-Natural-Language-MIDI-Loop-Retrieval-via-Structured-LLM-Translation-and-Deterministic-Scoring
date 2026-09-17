"""Build the LLM system prompt from the taxonomy and feature descriptions."""

import re
from typing import Any

# Standalone aggregate concepts that belong immediately after their histogram
# block (e.g. texture_polyphonic_pct sums the 3+ note bins of texture_pct), so
# they read as an aggregation of it rather than appearing later alphabetically.
_HISTOGRAM_AGGREGATES_FIRST = frozenset({"texture_polyphonic_pct"})

from midi_llm_search_engine.index_loader import SearchIndex

# Presentation order for categories in the pitched section. Common (pitched+drums)
# categories come first, then pitched-only, roughly by how central they are.
CATEGORY_ORDER = [
    "rhythm",
    "dynamics",
    "metadata",
    "voicing",
    "tonality",
    "melody",
]

# Optional per-category subcategory ordering. Subcategories not listed fall back
# to alphabetical order after any listed ones.
SUBCATEGORY_ORDER = {
    "rhythm": ["duration", "spacing", "grid", "rhythmic density"],
}


GLOBAL_INSTRUCTIONS = """You are a musical query analyzer for a MIDI file search system.

Your task is to convert ANY natural language query — even one that does not explicitly mention music — into a structured list of concept targets for tag-based search.

Use the ACTIVE SCHEMA and EXACT concept/level names below. The schema is divided into a {PITCHED} section and a {DRUMS} section. The {DRUMS} section is a router: only a subset of the subcategories apply to drums, and each is applied per kit piece (kick, snare_clap, hats_cymbals, toms_others). The {DRUMS} header below explains exactly which subcategories apply, the naming pattern, and how to restrict to a named piece.

FAMILY CLASSIFICATION FIRST:
Before you select any concept, you MUST classify the query as exactly one of:
- "pitched" — for melodies, chords, bass, harmony, tonality, register, or any non-drum sound.
- "drums" — for percussion, drum kits, kicks, snares, claps, hi-hats, cymbals, or any kit piece.

Write the selected family classification at the start of the interpretation field. Then select targets ONLY from the matching taxonomy section:
- If the family classification is "pitched", use ONLY the {PITCHED} section.
- If the family classification is "drums", use ONLY the {DRUMS} section.

Do not return a pitched-only concept for a drums query, and do not return a drum-specific concept for a pitched query. If the query sounds like it could refer to either family, pick the one that is most dominant based on the musical vocabulary — or use your own interpretation in vague cases — and return only concepts from that family.

If an instrument family hint is provided, use it as a strong default for your classification, but still document the classification in the interpretation field.

HOLISTIC INTERPRETATION: do not translate word-by-word. Musical phrases have contextual meanings:
- "tight groove" = a cohesive feel (tight timing plus locked grid), not "tight" + "groove" separately.
- "busy bass" = rhythmic density in a low register (pitched).
- "sustained chords" = chordal voicings plus long durations.

EXTENSIVE INTERPRETATION: be expansive, not literal. From every word and phrase, derive the full set of relevant concepts it implies in a musical context — including the associated instrument/register, texture, articulation, feel, and playing style. A single word can legitimately trigger several tags; infer and surface them rather than stopping at the most obvious keyword match.
- "funk bass": "bass" implies a bass instrument, the low/bass register, and usually a monophonic line. "funk" implies human/played feel, a loose-but-tight groove, syncopation, off-beats, and rhythmic, percussive playing. Together they should surface instrument, register, texture (mostly monophonic), groove/timing looseness, syncopation/off-beat grid, and rhythmic-density concepts.
- "warm pad": "pad" implies a sustained, chordal texture; "warm" implies soft dynamics, consonant harmony, and a mid register.
Only add concepts that the query reasonably implies; do not invent tags with no musical basis. If several related concepts are suggested, return them, and use IMPORTANCE to rank the most strongly implied ones above the weaker associations.

CONCEPT SELECTION (map query vocabulary to the matching subcategory):
- Busyness/sparsity, how active → rhythmic density
- Gap size between attacks (motor/drive) → spacing
- How long notes ring (sustain/staccato) → duration
- Swing/feel/timing looseness → grid
- Which beat positions are hit/accurate → grid
- How many notes sound at once, voicing width → texture
- Pitch range / where the part sits → register
- Chord interval quality (fifths/octaves/dissonance) → harmonic intervals
- Melody motion/leaps/direction → melodic intervals
- Loudness/accents/volume trend → dynamics
- Pitch vocabulary / chromaticism → tonality
- Instrument / key / time signature / length → metadata
- Which kit pieces are present → prevalence (use the {DRUMS} section)

SCOPE (SUMMARY vs DETAIL):
Each concept has a scope of "summary" or "detail".
- Summary concepts aggregate a dimension (e.g. an overall "short spacing" profile).
- Detail concepts measure a specific value (e.g. the "8th note spacing" share).
A summary concept does NOT stand in for its detail concepts. When the query calls
for a specific detail value, use the detail concept — do not fall back to the
broader summary just because it is related. Prefer detail concepts when the
query names a precise value, and use summary concepts only for broad or
aggregate statements.

IMPORTANCE (relative weight of each clause against the others in the query):
Importance is NOT the level. The LEVEL says how much (Defining → Occasional) or its
absence ("No X" / "Not Present"); IMPORTANCE says how strongly this clause should
drive the ranking relative to the other clauses. "No X" could be a strong
constraint, depending on the context. "some/light/a bit" adjust the LEVEL, not
the importance.

- 5 = the defining/only/exclusive constraint — the query is fundamentally about this
- 4 = "must have X" / "requires X" / "has to be X" — a hard requirement
- 3 = default — a normal inclusion
- 2 = "ideally X" / "preferably X" / "optionally X" — a nice-to-have; boosts if present, no penalty if absent
- 1 = minor / passing hint

LEVEL INDEX ANCHORING:
Levels are ordered from most intense (index 0) to least intense (index N-1).
The level name is a semantic descriptor; the INDEX is the precise strength anchor.
So a HIGHER / more intense level has a LOWER index, and a LOWER / less intense level
has a HIGHER index.

FALLBACK STRATEGIES (when the query doesn't specify an exact level, express the
direction relative to the chosen anchor index):
- "at least X" / "X or more intense" → "down" (toward the lower index = more intense)
- "at most X" / "X or less intense" → "up" (toward the higher index = less intense)
- "exactly X" / "closest to X" → "nearest"

INDEX MAPPING:
- Index 0 = maximum / extreme / "the most" — use ONLY for the strongest possible language (e.g., "relentless", "extreme", "never", "constant", "always", "defining characteristic").
- Index 1 = strong / high — use for clear but not absolute language (e.g., "very", "heavy", "highly", "strong").
- Index 2 / N-3 = medium / neutral — use for balanced, moderate, or mid-strength language (e.g., "moderate", "average", "balanced", "neither weak nor strong").
- Index N-2 = low / weak — use for mild presence (e.g., "slight", "somewhat", "occasionally").
- Index N-1 = minimum / absent — use only for "none", "zero", "never", "no X".

LANGUAGE RULES (infer the user's intent, do not map keywords mechanically):
Interpret how the user is describing the intensity, and select the level index that
best matches their intent. Be neither too conservative nor too extreme: if the
language is strong or absolute, commit to the strong end (index 0 or N-1); if it
is mild or qualified, commit to the mild end (index N-2 or N-1); if it is balanced
or neutral, use the middle. Do not water down strong language to a middle index,
and do not inflate weak or vague language to an extreme index.
Check the current level's index range before selecting.

OUTPUT FORMAT: Return only a JSON object matching the required schema. Do not include any other text.
"""

# General music reference: maps semitone counts to interval names. Applies to BOTH
# the melodic (horizontal) and harmonic (vertical) interval concepts in the schema,
# so the model can convert musical interval names to semitones exactly.
INTERVAL_REFERENCE = (
    "SEMITONE ↔ INTERVAL REFERENCE (applies to BOTH the melodic and harmonic interval concepts):\n"
    "  00=unison  01=minor second  02=major second  03=minor third  04=major third  05=perfect fourth\n"
    "  06=tritone  07=perfect fifth  08=minor sixth  09=major sixth  10=minor seventh  11=major seventh\n"
    "  12=octave  13=minor ninth  14=major ninth  15=minor tenth  16=major tenth  17=eleventh\n"
    "  18=augmented eleventh  19=twelfth (octave + perfect fifth)  20=minor thirteenth  21=major thirteenth\n"
    "  22=minor fourteenth  23=major fourteenth  24=double octave\n"
    "Convert musical interval names to semitones exactly using this reference "
    "(e.g. perfect fourth=05, perfect fifth=07, octave=12, major third=04, major thirteenth=21). "
    "Map the EXACT semitone for the interval named — do not shift by one. Pay special attention to "
    "compound intervals beyond the octave: twelfth=19, minor thirteenth=20, major thirteenth=21, "
    "minor fourteenth=22, major fourteenth=23, double octave=24."
)


def _output_schema(include_semantic_analysis: bool) -> str:
    """Build the OUTPUT SCHEMA block, conditionally including optional fields."""
    lines = ["OUTPUT SCHEMA (JSON):", "{"]
    lines.append('  "family_classification": "pitched|drums"')
    if include_semantic_analysis:
        lines.append(
            '  "semantic_analysis": ['
            '{"concept_name": "EXACT CONCEPT NAME", "level_name": "EXACT LEVEL NAME", "why": "why this matches the query"}'
            "]"
        )
    lines.extend([
        '  "targets": [',
        '    {',
        '      "concept_name": "EXACT CONCEPT NAME",',
        '      "level_index": 0,                    // position index (0=most intense, N-1=least) — PREFERRED',
        '      "level_name": "EXACT LEVEL NAME",   // fallback: use only for categorical levels (e.g. Major/Minor, instrument family)',
        '      "importance": 1,',
        '      "fallback": "down|up|nearest"',
        "    }",
        "  ],",
        "}",
        "INSTRUCTIONS: Return only a JSON object matching the required schema above. Do not include any other text. "
        "Never return an empty \"targets\" list — always include at least one concept target.",
    ])
    return "\n".join(lines)


OUTPUT_SCHEMA = _output_schema(True)


class SystemPromptBuilder:
    """Build a dynamic system prompt from the analyzer taxonomy."""

    def __init__(
        self,
        index: SearchIndex,
        include_drum_names: bool = True,
        group_bins: bool = True,
        use_generic_bin_levels: bool = False,
    ):
        self.index = index
        self.include_drum_names = include_drum_names
        self.group_bins = group_bins
        self.use_generic_bin_levels = use_generic_bin_levels

    def build(
        self,
        first_family: str | None = None,
        include_semantic_analysis: bool = False,
    ) -> str:
        """Return the system prompt grouped by family, then category, then subcategory.

        If ``first_family`` is provided, that family's section is rendered first
        (immediately after the instructions) so relevant concepts are seen before
        the irrelevant ones.
        """
        sections = [
            GLOBAL_INSTRUCTIONS,
            "\n" + INTERVAL_REFERENCE,
            "\nACTIVE SCHEMA (by family, then category, then subcategory):\n",
        ]

        llm_categories = self.index.taxonomy.get("llm_categories", {})
        pitched = self._group_concepts(family="pitched")
        drums = self._group_concepts(family="drums")

        families = {"pitched": pitched, "drums": drums}
        family_order = ["pitched", "drums"]
        if first_family in families:
            family_order = [first_family, "drums" if first_family == "pitched" else "pitched"]

        for fname in family_order:
            fdata = families[fname]
            sections.append(f"\n{{{fname.upper()}}}")
            if not fdata:
                sections.append("(no concepts available)")
                continue
            if fname == "drums":
                sections.append(self._format_drums())
            else:
                ordered = [c for c in CATEGORY_ORDER if c in fdata]
                ordered += sorted(c for c in fdata if c not in CATEGORY_ORDER)
                for category in ordered:
                    cat_data = llm_categories.get(category, {})
                    sections.append(self._format_category(category, fdata[category], cat_data))

        sections.append("\n" + _output_schema(include_semantic_analysis))
        return "\n".join(sections)

    def _group_concepts(
        self, family: str | None = None
    ) -> dict[str, dict[str, list[str]]]:
        """Group concepts by category, then subcategory.

        If ``family`` is provided, only concepts whose ``instrument_family``
        includes that family are included. All concepts with a non-zero
        base_weight are considered. The top-level ``llm_categories`` taxonomy key
        is skipped. Concepts with a ``pair_role`` other than "attempt" are
        skipped — they will be rendered alongside their paired counterpart.
        """
        grouped: dict[str, dict[str, list[str]]] = {}
        for concept_name, data in self.index.taxonomy.items():
            if concept_name in ("llm_categories", "llm_subcategories"):
                continue

            # Paired concepts render independently — neither is secondary
            weight = data.get(
                "base_weight", data.get("default_weight", 1.0)
            )
            if not weight:
                continue

            if family is not None and family not in self.index.concept_families(concept_name):
                continue

            llm_meta = data.get("llm") or {}
            category = llm_meta.get("category", "other")
            subcategory = llm_meta.get("subcategory") or "General"
            grouped.setdefault(category, {}).setdefault(subcategory, []).append(concept_name)

        return grouped

    def _format_category(
        self, category_name: str, subcategories: dict[str, list[str]], cat_data: dict[str, Any]
    ) -> str:
        """Render a category header and all its subcategories."""
        lines = [f"\n[{category_name.upper()}]"]

        # When a category has exactly one subcategory, the category's prose would
        # duplicate the subcategory's prose. Skip the category docs and let the
        # single subcategory carry the content. With multiple subcategories, the
        # category docs serve as a genuine overview and are kept.
        if len(subcategories) != 1:
            if cat_data.get("description"):
                lines.append(cat_data["description"])
            if cat_data.get("interpretation"):
                lines.append(f"How to use: {cat_data['interpretation']}")
            if cat_data.get("examples"):
                lines.append(f"Examples: {', '.join(cat_data['examples'])}")

        llm_subcategories = self.index.taxonomy.get("llm_subcategories", {})
        order = SUBCATEGORY_ORDER.get(category_name, [])
        ordered_subcats = [s for s in order if s in subcategories]
        ordered_subcats += sorted(s for s in subcategories if s not in order)
        for subcategory in ordered_subcats:
            lines.append(f"\n### {subcategory}")
            sub_data = llm_subcategories.get(subcategory, {})
            if sub_data.get("description"):
                lines.append(sub_data["description"])
            if sub_data.get("interpretation"):
                lines.append(f"How to use: {sub_data['interpretation']}")
            if sub_data.get("disambiguation"):
                lines.append(f"Disambiguation: {sub_data['disambiguation']}")
            if sub_data.get("examples"):
                lines.append(f"Examples: {', '.join(sub_data['examples'])}")
            self._append_subcategory_concepts(lines, sorted(subcategories[subcategory]))

        return "\n".join(lines)

    def _format_drums(self) -> str:
        """Render the {DRUMS} section as a compact reference to the base concepts.

        Drum concepts are the base concepts (from rhythm/dynamics) applied to each
        kit piece. Instead of repeating each base concept's text, we render a
        reference header + the naming pattern, then the drum-only prevalence
        subcategory in full.
        """
        lines: list[str] = []
        kit_pieces = ["kick", "snare_clap", "hats_cymbals", "toms_others"]

        common_subcats, pitched_only_subcats, drum_only_subcats = [], [], []
        llm_subcategories = self.index.taxonomy.get("llm_subcategories", {})
        for sub_name, sub_data in sorted(llm_subcategories.items()):
            fams = sub_data.get("families", [])
            if "drums" in fams and "pitched" in fams:
                common_subcats.append(sub_name)
            elif "drums" in fams:
                drum_only_subcats.append(sub_name)
            elif "pitched" in fams:
                pitched_only_subcats.append(sub_name)

        lines.append(
            "Drum concepts are the base concepts from the dynamics and rhythm (excluding duration) sections, "
            "applied independently to each kit piece. "
            "The description, levels, and interpretation are identical to the base concept; "
            "only the kit piece changes."
        )
        lines.append(f"Kit pieces: {', '.join(kit_pieces)}")
        lines.append(f"Subcategories that apply to drums: {', '.join(common_subcats)}")
        if pitched_only_subcats:
            lines.append(f"Subcategories that do NOT apply to drums: {', '.join(pitched_only_subcats)}")
        lines.append(
            "Naming pattern: drum_{piece}_{base_concept} — e.g. drum_kick_spacing_8th_share "
            "is spacing_8th_share applied to the kick."
        )
        lines.append(
            "When the query names a specific kit piece (kick, snare, hi-hat, toms), "
            "emit only that piece's variants."
        )
        lines.append(
            "When the query mentions a specific drum piece (hi-hats, kick, snare, toms, "
            "crash, ride), include both its activity concepts AND its prevalence concept "
            "when appropriate."
        )
        lines.append(
            "Follow explicit per-piece and per-value instructions exactly — if the query specifies a "
            "particular kit piece or value, tag it. For example, if the query states both the kick and the "
            "hi-hats use the same spacing value, emit the specified concept for each piece rather than "
            "assuming the pieces share or differ in a value."
        )
        lines.append(
            "ANALYZE THE KIT HOLISTICALLY: a drum pattern is defined by how the kit pieces "
            "inter-relate, not by isolated parts. If the query names a specific piece, "
            "provide that piece's traits directly. But when the query expresses full-kit "
            "requirements — instead of, or in addition to, piece-specific ones — combine the "
            "individual piece traits so the desired outcome is achieved: weight the "
            "relationships between pieces and answer from an understanding of the whole kit "
            "(e.g. a kick that lands in the spaces the hats leave open, a snare backbeat "
            "that locks with the kick, or busy hats pushing against a sparse kick). Select "
            "concept targets whose combination reflects the pattern as an interlocking whole, "
            "not independent per-piece answers."
        )

        # Render drum-only subcategories (prevalence) in full, before the name list.
        drum_subcats = self._group_concepts(family="drums")
        for category in sorted(drum_subcats):
            for sub in sorted(drum_subcats[category]):
                if sub not in drum_only_subcats:
                    continue
                lines.append(f"\n### {sub}")
                sub_data = llm_subcategories.get(sub, {})
                if sub_data.get("description"):
                    lines.append(sub_data["description"])
                if sub_data.get("interpretation"):
                    lines.append(f"How to use: {sub_data['interpretation']}")
                if sub_data.get("disambiguation"):
                    lines.append(f"Disambiguation: {sub_data['disambiguation']}")
                if sub_data.get("examples"):
                    lines.append(f"Examples: {', '.join(sub_data['examples'])}")
                concepts = sorted(drum_subcats[category][sub])
                if sub == "prevalence":
                    self._render_prevalence(lines, concepts, kit_pieces)
                else:
                    self._append_subcategory_concepts(lines, concepts)

        if self.include_drum_names:
            lines.append("\nDrum concept names (by subcategory):")
            grouped: dict[str, list[str]] = {}
            prevalence_names: dict[str, list[str]] = {}
            for concept_name, data in self.index.taxonomy.items():
                if concept_name in ("llm_categories", "llm_subcategories"):
                    continue
                if not data.get("base_weight", data.get("default_weight", 1.0)):
                    continue
                if concept_name.startswith("drum_prevalence_"):
                    if self._is_family_prevalence(concept_name, kit_pieces):
                        continue  # whole-family aggregates are noted in one line
                    fam = self._prevalence_family(concept_name, kit_pieces)
                    prevalence_names.setdefault(fam, []).append(concept_name)
                    continue
                base_concept = data.get("base_concept")
                if not base_concept:
                    continue
                sub = data.get("llm", {}).get("subcategory", "other")
                if base_concept not in grouped.setdefault(sub, []):
                    grouped[sub].append(base_concept)
            for sub in sorted(grouped):
                for base in sorted(grouped[sub]):
                    instantiations = ", ".join(f"drum_{p}_{base}" for p in kit_pieces)
                    lines.append(f"  {sub} / {base}: {instantiations}")
            for fam in sorted(prevalence_names):
                lines.append(f"  prevalence / {fam}: {', '.join(sorted(prevalence_names[fam]))}")

        return "\n".join(lines)

    def _render_prevalence(self, lines: list[str], concepts: list[str], kit_pieces: list[str]) -> None:
        """Render the drum prevalence subcategory compactly.

        Every prevalence concept shares the same interpretation and level
        structure, differing only in the kit piece and its family. So show the
        directive and the level scale once (the exact concept names are listed
        under "Drum concept names (by subcategory)").
        """
        concepts = sorted(concepts)
        first_llm = self.index.taxonomy[concepts[0]].get("llm", {})

        interp = first_llm.get("interpretation", "")
        if interp:
            lines.append(f"When to use: {interp}")

        levels = self.index.level_names(concepts[0])
        lvl_desc = first_llm.get("level_descriptions", {})
        piece_label = levels[0].split(" ", 1)[1] if levels and " " in levels[0] else ""
        fam0 = self._prevalence_family(concepts[0], kit_pieces)
        lines.append("Levels (shared across all kit pieces):")
        for idx, level_name in enumerate(levels):
            name = self._replace_label(level_name, piece_label)
            d = self._replace_label(lvl_desc.get(level_name, ""), piece_label)
            d = d.replace(f"the {fam0} family", "its family")
            lines.append(f'  - "[{idx}] {name}": {d}')

        family_aggs = sorted(c for c in concepts if self._is_family_prevalence(c, kit_pieces))
        if family_aggs:
            lines.append(
                "Whole-family kit-piece prevalence aggregates exist: "
                + ", ".join(family_aggs)
                + " — the share of the whole kit from that family, using the same levels as above."
            )

    def _prevalence_family(self, concept_name: str, kit_pieces: list[str]) -> str:
        """Return the kit family a prevalence concept belongs to."""
        rest = concept_name[len("drum_prevalence_"):]
        return next((p for p in kit_pieces if rest == p or rest.startswith(p + "_")), rest)

    def _is_family_prevalence(self, concept_name: str, kit_pieces: list[str]) -> bool:
        """True for a whole-family aggregate (drum_prevalence_<family>), not a
        specific instrument within the family."""
        rest = concept_name[len("drum_prevalence_"):]
        return rest in kit_pieces

    def _append_subcategory_concepts(
        self, lines: list[str], concept_names: list[str]
    ) -> None:
        """Render concepts, grouping numeric siblings under shared headers."""
        rendered = set()
        used_in_group: set[str] = set()

        # First pass: detect numeric-suffix sibling groups
        groups: list[tuple[str, list[str]]] = []
        group_members: set[str] = set()

        for i, name in enumerate(concept_names):
            if name in rendered:
                continue
            members = [name]
            # Check if this concept has numeric siblings sharing the same prefix
            prefix = self._numeric_prefix(name)
            if prefix:
                siblings = self._find_numeric_siblings(name, prefix, concept_names)
                if len(siblings) >= 4:  # Only group if 4+ siblings
                    groups.append((prefix, siblings))
                    group_members.update(siblings)
                    rendered.update(siblings)
                    continue
            # Check if this concept is a histogram bin (word or numeric)
            bin_family = self.index.taxonomy.get(name, {}).get("llm", {}).get("bin_family")
            if bin_family and self.group_bins:
                bin_siblings = self._find_bin_siblings(bin_family, concept_names)
                if len(bin_siblings) >= 2:
                    groups.append((f"{bin_family}:", bin_siblings))
                    group_members.update(bin_siblings)
                    rendered.update(bin_siblings)
                    continue
            rendered.add(name)

        # Merge share-family + profile-family groups into a single histogram block
        # (e.g. "duration" shares + "duration_profile" profiles), so levels are shown once.
        # Grid attempt + success groups are also merged into one block.
        histogram: dict[str, dict[str, list[str]]] = {}
        for prefix, siblings in groups:
            base = prefix[:-1]
            is_profile = base.endswith("_profile")
            is_grid = base.startswith("grid_")
            if is_grid:
                # Merge grid_attempt_pct + grid_success_pct into a single grid block.
                key = "grid"
                histogram.setdefault(key, {"shares": [], "profiles": []})
                if "success" in base:
                    histogram[key]["profiles"] = siblings
                else:
                    histogram[key]["shares"] = siblings
                continue
            if base == "metadata_time_sig":
                # Merge time_sig_num + time_sig_den into a single block.
                key = "time_sig"
                histogram.setdefault(key, {"shares": [], "profiles": []})
                for s in siblings:
                    if s.endswith("_den"):
                        histogram[key]["profiles"].append(s)
                    else:
                        histogram[key]["shares"].append(s)
                continue
            # Merge the ascending and descending melodic interval profile families
            # into a single block: shares = ascending bins, profiles = descending bins.
            if base.startswith("profile_melodic_intervals_pct_asc"):
                histogram.setdefault("melodic_profile", {"shares": [], "profiles": []})
                histogram["melodic_profile"]["shares"] = siblings
                continue
            if base.startswith("profile_melodic_intervals_pct_desc"):
                histogram.setdefault("melodic_profile", {"shares": [], "profiles": []})
                histogram["melodic_profile"]["profiles"] = siblings
                continue
            key = base[:-len("_profile")] if is_profile else base
            histogram.setdefault(key, {"shares": [], "profiles": []})
            if is_profile:
                histogram[key]["profiles"] = siblings
            else:
                histogram[key]["shares"] = siblings

        # Include the static (unison/repeated-note) bin in the melodic profile block.
        if "melodic_profile" in histogram and "profile_melodic_intervals_pct_static" in concept_names:
            histogram["melodic_profile"]["profiles"].append("profile_melodic_intervals_pct_static")
            group_members.add("profile_melodic_intervals_pct_static")

        final_groups: list[tuple[str, list[str], list[str]]] = []
        for base, parts in histogram.items():
            shares = parts["shares"]
            profiles = parts["profiles"]
            if shares and profiles:
                final_groups.append((f"{base}::", shares, profiles))
            elif shares:
                final_groups.append((f"{base}:", shares, []))
            elif profiles:
                final_groups.append((f"{base}_profile:", [], profiles))

        # Second pass: render concepts (grouped or individual)
        # First render groups
        for prefix, shares, profiles in final_groups:
            self._render_concept_group(lines, prefix, shares, profiles)
        # Then render remaining ungrouped concepts. Histogram aggregates render
        # right after their histogram so they read as an aggregation of it.
        remaining = [n for n in concept_names if n not in group_members]
        remaining.sort(key=lambda n: (0 if n in _HISTOGRAM_AGGREGATES_FIRST else 1, n))
        for name in remaining:
            lines.append(self._format_concept(name))

    def _numeric_prefix(self, concept_name: str) -> str | None:
        """Extract the non-numeric prefix before a numeric suffix.

        E.g. ``profile_harmonic_intervals_pct_01_semitones -> profile_harmonic_intervals_pct_``
        Returns None if the concept name has no clear numeric suffix pattern.
        """
        parts = concept_name.split("_")
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                return "_".join(parts[:i]) + "_"
        return None

    def _find_numeric_siblings(
        self, concept_name: str, prefix: str, all_names: list[str]
    ) -> list[str]:
        """Find all concepts sharing a numeric-suffix pattern with the given concept."""
        suffix_rest = concept_name[len(prefix):]
        # Check if the suffix is purely numeric or has a trailing label
        parts = suffix_rest.split("_")
        if parts and parts[0].isdigit():
            return sorted(
                n for n in all_names
                if n.startswith(prefix) and n[len(prefix):].split("_")[0].isdigit()
            )
        return [concept_name]

    def _find_bin_siblings(self, bin_family: str, all_names: list[str]) -> list[str]:
        """Find all concepts in the same histogram bin family."""
        return sorted(
            n for n in all_names
            if self.index.taxonomy.get(n, {}).get("llm", {}).get("bin_family") == bin_family
        )

    def _render_concept_group(
        self,
        lines: list[str],
        prefix: str,
        shares: list[str],
        profiles: list[str] | None = None,
    ) -> None:
        """Render a group of sibling concepts (numeric or histogram) as one block.

        For histograms, ``prefix`` may end in ``::`` (merged shares+profiles) or
        ``:`` (shares only) or ``_profile:`` (profiles only). ``shares`` are the
        detail bin concepts; ``profiles`` are the aggregate profile concepts.
        """
        profiles = profiles or []
        siblings = shares if shares else profiles
        if not siblings:
            return

        is_bin = prefix.endswith(":") and siblings[0] in self.index.taxonomy
        is_merged = prefix.endswith("::")

        # Use the first sibling for level info (all share the same levels)
        first = siblings[0]
        data = self.index.taxonomy[first]
        llm = data.get("llm", {})
        levels = self.index.level_names(first)
        level_descriptions = llm.get("level_descriptions", {})

        if is_bin:
            bin_family = prefix[:-1] if not is_merged else prefix[:-2]
            # Collect bin labels from the share (detail) concepts
            labels = []
            for s in (shares if shares else profiles):
                labels.append(self.index.taxonomy[s].get("llm", {}).get("bin_label", s))
            labels = sorted(set(labels), key=lambda x: (len(x), x))

            if bin_family == "time_sig":
                # Time signature: num + den are merged. shares = numerator concepts,
                # profiles = denominator concepts. Render as one block.
                num_concepts = shares
                den_concepts = profiles
                lines.append("- time signature (number of beats per measure + beat unit):")
                lines.append("  When to use: Use for 'time signature', 'meter', or '3/4' queries.")
                # Combined interpretation covering both num and den.
                num_interp = self.index.taxonomy[num_concepts[0]].get("llm", {}).get("interpretation", "") if num_concepts else ""
                den_interp = self.index.taxonomy[den_concepts[0]].get("llm", {}).get("interpretation", "") if den_concepts else ""
                lines.append(f"  {num_interp} {den_interp}".rstrip())
                if num_concepts:
                    lines.append(f"  Numerator concepts: {', '.join(num_concepts)}")
                    lines.append("  Numerator levels:")
                    for idx, level_name in enumerate(self.index.level_names(num_concepts[0])):
                        lines.append(f'    - "[{idx}] {level_name}"')
                if den_concepts:
                    lines.append(f"  Denominator concepts: {', '.join(den_concepts)}")
                    lines.append("  Denominator levels:")
                    for idx, level_name in enumerate(self.index.level_names(den_concepts[0])):
                        lines.append(f'    - "[{idx}] {level_name}"')
                return

            if bin_family == "melodic_profile":
                # Ascending + static + descending melodic interval bins, shown together.
                labels = sorted(
                    {self.index.taxonomy[s].get("llm", {}).get("bin_label", s) for s in (shares or profiles) if s != "profile_melodic_intervals_pct_static"},
                    key=lambda x: (len(x), x),
                )
                lines.append(f"- melodic intervals profile (ascending, static & descending): percentage/share of {', '.join(labels)} (asc/desc) and repeated notes (0 semitones)")
                if shares:
                    lines.append(f"  Ascending bins (moving UP): {', '.join(shares)}")
                    asc_interp = self.index.taxonomy[shares[0]].get("llm", {}).get("interpretation", "")
                    if asc_interp:
                        lines.append(f"    How to use: {asc_interp}")
                desc_bins = [s for s in (profiles or []) if s.startswith("profile_melodic_intervals_pct_desc")]
                static_bins = [s for s in (profiles or []) if s == "profile_melodic_intervals_pct_static"]
                if desc_bins:
                    lines.append(f"  Descending bins (moving DOWN): {', '.join(desc_bins)}")
                    desc_interp = self.index.taxonomy[desc_bins[0]].get("llm", {}).get("interpretation", "")
                    if desc_interp:
                        lines.append(f"    How to use: {desc_interp}")
                if static_bins:
                    lines.append(f"  Static bins (repeated notes, 0 semitones): {', '.join(static_bins)}")
                    static_interp = self.index.taxonomy[static_bins[0]].get("llm", {}).get("interpretation", "")
                    if static_interp:
                        lines.append(f"    How to use: {static_interp}")
                first = (shares or desc_bins or static_bins or [])[0]
                levels = self.index.level_names(first)
                lvl_desc = self.index.taxonomy[first].get("llm", {}).get("level_descriptions", {})
                lines.append("  Levels (shared across all bins):")
                for idx, level_name in enumerate(levels):
                    lines.append(f'    - "[{idx}] {level_name}": {lvl_desc.get(level_name, "")}')
                return

            if bin_family == "grid" or bin_family.startswith("grid_"):
                # Grid: attempt + success are merged into one block. shares = attempt
                # concepts, profiles = success concepts. Render positions once (each with
                # both attempt and success examples), then both level scales.
                attempt_concepts = shares
                success_concepts = profiles
                labels = []
                for s in (attempt_concepts or success_concepts):
                    labels.append(self.index.taxonomy[s].get("llm", {}).get("bin_label", s))
                labels = sorted(set(labels), key=lambda x: (len(x), x))
                lines.append(f"- grid by position: {', '.join(labels)}")
                lines.append(
                    "  Grid analysis is holistic: a pattern is defined by the right combination of "
                    "attempts and successes across positions. E.g. 'four-on-the-floor' needs high attempt "
                    "AND high success on the odd-1, even-1, beat-3, and 2-and-4 positions. In many cases "
                    "the correct mix of these metrics is required, not a single value."
                )
                # Attempt concepts (with examples per position)
                lines.append(f"  Attempt concepts (how often each position is PLAYED): {', '.join(attempt_concepts)}")
                for s in attempt_concepts:
                    ex = self.index.taxonomy[s].get("llm", {}).get("examples", [])
                    pos = self.index.taxonomy[s].get("llm", {}).get("bin_label", s)
                    ex_str = ", ".join(ex) if ex else ""
                    if ex_str:
                        lines.append(f"    - {pos}: e.g. {ex_str}")
                # Success concepts (with examples per position)
                lines.append(f"  Success concepts (how accurately attempts land): {', '.join(success_concepts)}")
                for s in success_concepts:
                    ex = self.index.taxonomy[s].get("llm", {}).get("examples", [])
                    pos = self.index.taxonomy[s].get("llm", {}).get("bin_label", s)
                    ex_str = ", ".join(ex) if ex else ""
                    if ex_str:
                        lines.append(f"    - {pos}: e.g. {ex_str}")
                # Attempt levels (from first attempt concept)
                a_levels = self.index.level_names(attempt_concepts[0]) if attempt_concepts else []
                a_desc = self.index.taxonomy[attempt_concepts[0]].get("llm", {}).get("level_descriptions", {}) if attempt_concepts else {}
                lines.append("  Attempt levels (shared across all positions):")
                for idx, level_name in enumerate(a_levels):
                    lvl_desc = a_desc.get(level_name, "")
                    lines.append(f'    - "[{idx}] {level_name}": {lvl_desc}')
                # Success levels (from first success concept)
                s_levels = self.index.level_names(success_concepts[0]) if success_concepts else []
                s_desc = self.index.taxonomy[success_concepts[0]].get("llm", {}).get("level_descriptions", {}) if success_concepts else {}
                lines.append("  Success levels (shared across all positions):")
                for idx, level_name in enumerate(s_levels):
                    lvl_desc = s_desc.get(level_name, "")
                    lines.append(f'    - "[{idx}] {level_name}": {lvl_desc}')
                return

            # Aggregation families (e.g. the macro harmonic consonance classes):
            # every member is a summary concept. List each with its description +
            # "Use for" directive, then show the shared level structure once
            # (generalizing the member's label to X).
            members = shares if shares else profiles
            if not is_merged and members and all(
                self.index.taxonomy[s].get("llm", {}).get("scope") == "summary"
                for s in members
            ):
                agg_labels = [
                    self.index.taxonomy[s].get("llm", {}).get("bin_label", s)
                    for s in members
                ]
                readable = bin_family.replace("_pct", "").replace("_", " ")
                lines.append(f"- {readable} ({', '.join(agg_labels)})")
                for s in members:
                    llm = self.index.taxonomy[s].get("llm", {})
                    desc = llm.get("description", "")
                    interp = llm.get("interpretation", "")
                    examples = llm.get("examples", [])
                    line = f"    - {s}: {desc}"
                    if interp:
                        line += f" {interp}"
                    if examples:
                        line += f" Examples: {', '.join(examples)}"
                    lines.append(line)
                agg_first = self.index.taxonomy[members[0]].get("llm", {})
                agg_label = agg_first.get("bin_label", "")
                agg_levels = self.index.level_names(members[0])
                agg_desc = agg_first.get("level_descriptions", {})
                lines.append("  Levels (shared across all aggregates):")
                for idx, level_name in enumerate(agg_levels):
                    marker = f"[{idx}]"
                    name = self._replace_label(level_name, agg_label)
                    lvl_desc = self._replace_label(agg_desc.get(level_name, ""), agg_label)
                    if lvl_desc:
                        lines.append(f'    - "{marker} {name}": {lvl_desc}')
                    else:
                        lines.append(f'    - "{marker} {name}"')
                return

            if is_merged:
                # Histogram block: list detail concepts, then aggregate concepts,
                # then the shared level list once.
                base = bin_family
                header = f"- {base} histogram"
                if labels:
                    header += f" (bins: {', '.join(labels)})"
                lines.append(header)
                lines.append(f"  Detail concepts: {', '.join(shares)}")
                if profiles:
                    lines.append(f"  Aggregate concepts: {', '.join(profiles)}")
                    for p in profiles:
                        interp = self.index.taxonomy[p].get("llm", {}).get("interpretation", "")
                        lines.append(f"    - {p}: {interp}")                # Shared levels (from a share bin, which has label descriptions)
                bin_label = self.index.taxonomy[first].get("llm", {}).get("bin_label", "")
                lines.append("  Levels (shared across all concepts):")
                for idx, level_name in enumerate(levels):
                    marker = f"[{idx}]"
                    name = level_name.replace(bin_label, "X") if bin_label else level_name
                    lvl_desc = level_descriptions.get(level_name, "")
                    if lvl_desc and bin_label:
                        lvl_desc = lvl_desc.replace(bin_label, "X")
                    lines.append(f'    - "{marker} {name}": {lvl_desc}')
                return

            is_profile = bin_family.endswith("_profile")
            if is_profile:
                # Profiles only (no share bins): render each with its "use for",
                # then the shared level structure once.
                base = bin_family.replace("_profile", "")
                lines.append(f"- {base} profiles ({', '.join(labels)}): sums of the {base} bins")
                for s in profiles:
                    interp = self.index.taxonomy[s].get("llm", {}).get("interpretation", "")
                    lines.append(f"  - {s}: {interp}")
                lines.append(f"  Aggregate concepts: {', '.join(profiles)}")
                lines.append("  Levels (shared across all profiles):")
                for idx, level_name in enumerate(levels):
                    name = level_name.replace("Long", "X").replace("Medium", "X").replace("Short", "X")
                    lines.append(f'    - "[{idx}] {name}"')
                return

            lines.append(f"- {bin_family} bins: percentage/share of {', '.join(labels)}")
            lines.append(f"  Detail concepts: {', '.join(shares)}")
            first_interp = self.index.taxonomy[first].get("llm", {}).get("interpretation", "")
            if first_interp:
                lines.append(f"  How to use: {first_interp}")
            # The level names/descriptions come from the first bin, which embeds that
            # bin's label (e.g. "16th note duration"). For a grouped block this is
            # incorrect, so generalize the label to "X" in both name and description.
            bin_label = self.index.taxonomy[first].get("llm", {}).get("bin_label", "")
            if self.use_generic_bin_levels:
                lines.append("  Levels (shared structure across all bins):")
                for idx, level_name in enumerate(levels):
                    generic = self._generic_level_name(level_name)
                    lines.append(f'    - "{idx} {generic}"')
            else:
                lines.append("  Levels (shared across all bins):")
                for idx, level_name in enumerate(levels):
                    marker = f"[{idx}]"
                    name = level_name.replace(bin_label, "X") if bin_label else level_name
                    lvl_desc = level_descriptions.get(level_name, "")
                    if lvl_desc and bin_label:
                        lvl_desc = lvl_desc.replace(bin_label, "X")
                    lines.append(f'    - "{marker} {name}": {lvl_desc}')
            return

        # Numeric-sibling grouping (existing behavior)
        nums = []
        for s in siblings:
            rest = s[len(prefix):]
            num_str = rest.split("_")[0]
            try:
                nums.append(int(num_str))
            except ValueError:
                nums.append(0)

        if not nums:
            return

        base = prefix.strip("_")
        num_min, num_max = min(nums), max(nums)
        num_range = f"{num_min:02d}–{num_max:02d}" if num_max > 99 else f"{num_min}–{num_max}"
        rest_parts = siblings[0][len(prefix):].split("_")
        label = "_".join(rest_parts[1:]) if len(rest_parts) > 1 else ""

        if "intervals_pct" in base and label == "semitones":
            direction = ""
            if "asc" in base:
                direction = "ascending "
            elif "desc" in base:
                direction = "descending "
            desc = (
                f"Specific {direction}interval percentages by semitone distance ({num_min}–{num_max}). "
                f"The concept name suffix indicates the semitone value "
                f"(e.g., 07_semitones = perfect fifth, 12_semitones = octave). "
                f"Use these when the query names a specific interval."
            )
        else:
            desc = llm.get("description") or f"Shared values {num_min}–{num_max} with the same level structure."
        lines.append(f"- {base}_{num_range}_{label}: {desc}")
        lines.append(f"  Applicable concepts: {', '.join(siblings[:6])}..." if len(siblings) > 6 else f"  Applicable concepts: {', '.join(siblings)}")

        interp = llm.get("interpretation")
        if interp:
            lines.append(f"  When to use: {interp}")

        lines.append("  Levels (shared across all values):")
        for idx, level_name in enumerate(levels):
            marker = f"[{idx}]"
            lvl_desc = level_descriptions.get(level_name, "")
            if lvl_desc:
                lines.append(f'    - "{marker} {level_name}": {lvl_desc}')
            else:
                lines.append(f'    - "{marker} {level_name}"')

    def _generic_level_name(self, level_name: str) -> str:
        """Strip the bin label from a bin-specific level name (e.g. 'Defining 16th Note Spacing' -> 'Defining')."""
        # Level names like "Defining 16th Note Spacing" / "Defining Kick 1".
        # Keep only the leading intensity word.
        return level_name.split()[0] if level_name else level_name

    @staticmethod
    def _replace_label(text: str, label: str) -> str:
        """Case-insensitively replace a bin label with 'X' (for shared level blocks)."""
        if not text or not label:
            return text
        return re.sub(re.escape(label), "X", text, flags=re.IGNORECASE)

    def _format_concept(self, concept_name: str) -> str:
        """Render a single concept with its LLM-facing metadata and levels."""
        data = self.index.taxonomy[concept_name]
        llm = data.get("llm") or {}

        description = llm.get("description") or ""

        interpretation = llm.get("interpretation")
        examples = llm.get("examples", [])
        level_descriptions = llm.get("level_descriptions", {})
        levels = self.index.level_names(concept_name)
        paired_dims = llm.get("paired_dimensions")

        lines = [f"- {concept_name}: {description or '(no description)'}"]

        if interpretation:
            lines.append(f"  When to use: {interpretation}")
        if examples:
            lines.append(f"  Examples: {', '.join(examples)}")

        if paired_dims:
            pair_with_name = llm.get("pair_with", "")
            lines.append(f"  Paired with: {pair_with_name}")
            lines.append(f"  NOTE: {concept_name} and {pair_with_name} are a pair — both must always be returned together.")

        if levels:
            lines.append("  Levels:")
            for idx, level_name in enumerate(levels):
                marker = f"[{idx}]"
                if level_name in level_descriptions:
                    lines.append(f'    - "{marker} {level_name}": {level_descriptions[level_name]}')
                else:
                    lines.append(f'    - "{marker} {level_name}"')

        return "\n".join(lines)

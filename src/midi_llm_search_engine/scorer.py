"""Rank files against LLM translation targets.

The translator produces a list of ``Target`` objects (concept, level,
importance, fallback). This module converts those targets into a ranked list of
files by scoring how close each file's tag levels are to the targets, honoring
the fallback direction and weighting each target by its importance.

Scoring model
-------------
For each target and file, the file earns ``importance * (1 - steps / max_steps)``
where ``steps`` is the ordinal distance between the target level and the file's
level, measured asymmetrically when the target has a directional fallback. The
fallback never saturates: a file closer to the target always beats a farther
file on either side; it only biases which side wins at equal distance.

- ``down``  -> prefer denser (at least as intense): denser files lose credit by
  their distance, sparser files are penalised more than any denser file.
- ``up``    -> prefer sparser (at most as intense): sparser files lose credit by
  their distance, denser files are penalised more than any sparser file.
- ``nearest`` -> credit falls off symmetrically with absolute distance.

A file that lacks a concept entirely earns zero credit for that target (soft
absence). If the target has importance 5 (a hard requirement), the absence is
also recorded in ``missing_must_have``.

A file's raw score is the sum of earned credit across targets; the normalized
score is ``raw_score / max_possible`` in [0, 1]. Files are ranked by raw score
descending.
"""

import time

from midi_llm_search_engine.index_loader import SearchIndex
from midi_llm_search_engine.models import (
    QueryTranslation,
    SearchResponse,
    SearchResultItem,
    Target,
)

MUST_HAVE_IMPORTANCE = 5


class FileScorer:
    """Deterministic, LLM-free ranker over a search index and query targets."""

    def __init__(self, index: SearchIndex, use_base_weight: bool = False):
        self.index = index
        self._distance_cache: dict[tuple[str, str, str], int | None] = {}
        # Dormant option: when True, scale each target's contribution by its
        # concept base_weight. Off by default (all active weights are 1.0).
        self.use_base_weight = use_base_weight

    def _effective_importance(self, target: Target) -> float:
        if not self.use_base_weight:
            return float(target.importance)
        return target.importance * self.index.base_weight(target.concept_name)

    def _group_targets(self, targets: list[Target]) -> list[list[Target]]:
        """Group mutually-paired targets (e.g. grid attempt/success) into units.

        A target belongs to a pair group only when BOTH members are present in
        the query. All other targets are singleton groups scored independently.
        """
        by_name = {t.concept_name: t for t in targets}
        partners = {t.concept_name: self.index.pair_with(t.concept_name) for t in targets}
        used: set[str] = set()
        groups: list[list[Target]] = []
        for target in targets:
            if target.concept_name in used:
                continue
            partner_name = partners.get(target.concept_name)
            partner = by_name.get(partner_name) if partner_name else None
            if (
                partner is not None
                and partners.get(partner_name) == target.concept_name
            ):
                groups.append([target, partner])
                used.update((target.concept_name, partner_name))
            else:
                groups.append([target])
                used.add(target.concept_name)
        return groups

    def _reason(self, target: Target, file_level: str, steps: int) -> str:
        return (
            f"{target.concept_name}: file='{file_level}' vs "
            f"target='{target.level_name}' ({steps} step(s), fallback "
            f"{target.fallback}, importance {target.importance})"
        )

    def _score_singleton(
        self,
        target: Target,
        tags: dict[str, str],
        matched: list[str],
        missing: list[str],
        reasons: list[str],
    ) -> float:
        """Score one target and fill the per-file bookkeeping lists.

        Returns the raw credit earned (before base_weight scaling).
        """
        file_level = tags.get(target.concept_name)
        if file_level is None:
            if target.importance >= MUST_HAVE_IMPORTANCE:
                missing.append(target.concept_name)
            return 0.0
        steps = self._directed_steps(target, file_level)
        if steps is None:
            if target.importance >= MUST_HAVE_IMPORTANCE:
                missing.append(target.concept_name)
            return 0.0
        matched.append(target.concept_name)
        reasons.append(self._reason(target, file_level, steps))
        return self._credit_for(target, file_level)

    def level_steps(self, concept_name: str, level_a: str, level_b: str) -> int | None:
        """Return the absolute ordinal distance between two levels of a concept.

        ``None`` if either level does not exist for the concept.
        """
        key = (concept_name, level_a, level_b)
        cached = self._distance_cache.get(key)
        if cached is not None or key in self._distance_cache:
            return cached
        pa = self.index.level_position(concept_name, level_a)
        pb = self.index.level_position(concept_name, level_b)
        result: int | None = abs(pa - pb) if (pa is not None and pb is not None) else None
        self._distance_cache[key] = result
        return result

    def _directed_steps(self, target: Target, file_level: str) -> int | None:
        """Ordinal steps in the direction permitted by the target's fallback.

        Distance-preserving (never saturating): a file closer to the target
        always earns more credit than a farther one, on both sides. The fallback
        only biases *which side wins at equal distance*, and any file on the
        preferred side beats any file on the disfavoured side.

        Positions: lower = denser/more intense, higher = sparser/less intense.

        - ``nearest`` -> symmetric: ``abs(diff)``.
        - ``down`` (at least / prefer denser): denser files step by distance;
          sparser files step by ``pb`` (equivalently distance to the top).
        - ``up`` (at most / prefer sparser): sparser files step by distance;
          denser files step by ``N - pb`` (distance to the bottom).

        ``None`` if the file level is unknown for the concept.
        """
        pa = self.index.level_position(target.concept_name, target.level_name)
        pb = self.index.level_position(target.concept_name, file_level)
        if pa is None or pb is None:
            return None
        diff = pb - pa
        if target.fallback == "down":
            if diff <= 0:
                return -diff
            return pb
        if target.fallback == "up":
            if diff >= 0:
                return diff
            return self._max_steps(target.concept_name) - pb
        return abs(diff)

    def _max_steps(self, concept_name: str) -> int:
        levels = self.index.level_names(concept_name)
        return max(0, len(levels) - 1)

    def _credit_for(self, target: Target, file_level: str | None) -> float:
        """Earned credit (0..importance) for one target against a file's level.

        Absent or unknown concept levels earn zero; a must-have absence is still
        recorded separately by the caller.
        """
        if file_level is None:
            return 0.0
        max_steps = self._max_steps(target.concept_name)
        steps = self._directed_steps(target, file_level)
        if steps is None or max_steps == 0:
            return 0.0
        return target.importance * (1.0 - steps / max_steps)

    def score(
        self,
        targets: list[Target],
        limit: int = 20,
    ) -> list[SearchResultItem]:
        """Return files ranked against the given targets, best first.

        Files are matched purely by the returned concepts (no family default or
        family filtering). A file is a candidate only if it carries at least one
        targeted concept; concepts are family-scoped in the taxonomy, so this
        naturally separates drums and pitched without assuming a family.
        """
        if not targets:
            return []

        wanted: set[str] = set()
        for target in targets:
            wanted.add(target.concept_name)
            partner = self.index.pair_with(target.concept_name)
            if partner:
                wanted.add(partner)

        candidates = [
            file_id
            for file_id, tags in self.index.file_tags.items()
            if wanted & set(tags.keys())
        ]

        results: list[SearchResultItem] = []
        for file_id in candidates:
            tags = self.index.file_tags.get(file_id, {})
            raw = 0.0
            max_possible = 0.0
            reasons: list[str] = []
            matched: list[str] = []
            missing: list[str] = []

            for group in self._group_targets(targets):
                if len(group) == 1:
                    target = group[0]
                    credit = self._score_singleton(target, tags, matched, missing, reasons)
                    max_possible += self._effective_importance(target)
                    raw += credit * (
                        self.index.base_weight(target.concept_name)
                        if self.use_base_weight else 1.0
                    )
                    continue

                # Paired group: conjunction credit = min of member closeness,
                # scaled by the combined importance of the pair.
                group_importance = 0.0
                min_norm = 1.0
                for target in group:
                    group_importance += self._effective_importance(target)
                    file_level = tags.get(target.concept_name)
                    credit = self._credit_for(target, file_level)
                    norm = credit / target.importance
                    min_norm = min(min_norm, norm)
                    if file_level is not None:
                        steps = self._directed_steps(target, file_level)
                        if steps is not None:
                            matched.append(target.concept_name)
                            reasons.append(self._reason(target, file_level, steps))
                    elif target.importance >= MUST_HAVE_IMPORTANCE:
                        missing.append(target.concept_name)
                max_possible += group_importance
                raw += min_norm * group_importance

            if max_possible == 0:
                continue
            score_val = max(0.0, min(1.0, raw / max_possible))
            results.append(
                SearchResultItem(
                    file_path=self.index.file_paths.get(file_id, str(file_id)),
                    file_id=file_id,
                    score=score_val,
                    raw_score=raw,
                    max_possible=max_possible,
                    all_tags={k: v for k, v in tags.items() if v is not None},
                    match_reasons=reasons,
                    matched_tags=matched,
                    missing_must_have=missing,
                )
            )

        results.sort(key=lambda item: item.raw_score, reverse=True)
        return results[:limit]

    def score_translation(
        self,
        translation: QueryTranslation,
        query: str = "",
        llm_call_record=None,
        limit: int = 20,
    ) -> SearchResponse:
        """Score a full translation and wrap the ranked files in a response."""
        start = time.perf_counter()
        results = self.score(translation.targets, limit=limit)
        duration_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(
            query=query,
            instrument_family=translation.family_classification or "",
            targets=translation.targets,
            results=results,
            duration_ms=duration_ms,
            llm_call_record=llm_call_record,
        )

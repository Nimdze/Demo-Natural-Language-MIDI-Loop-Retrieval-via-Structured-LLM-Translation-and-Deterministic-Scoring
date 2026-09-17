"""End-to-end search orchestration: translate a query, then rank files."""

from midi_llm_search_engine.index_loader import SearchIndex
from midi_llm_search_engine.llm_client import LLMTranslator
from midi_llm_search_engine.models import SearchResponse
from midi_llm_search_engine.scorer import FileScorer


class SearchError(Exception):
    """Raised when a search cannot be completed (e.g. LLM translation fails).

    Carries the original cause via ``__cause__`` so callers can surface a
    clean message while retaining the underlying error for debugging.
    """


class SearchEngine:
    """High-level entry point that wires the translator to the scorer.

    ``search()`` sends a query to the LLM translator, then feeds the resulting
    ``QueryTranslation`` to the scorer to produce a ranked ``SearchResponse``.
    """

    def __init__(
        self,
        index: SearchIndex,
        translator: LLMTranslator | None = None,
        scorer: FileScorer | None = None,
    ):
        self.index = index
        self.translator = translator or LLMTranslator(index)
        self.scorer = scorer or FileScorer(index)

    def search(
        self,
        query: str,
        instrument_family: str = "pitched",
        limit: int = 20,
    ) -> SearchResponse:
        """Translate ``query`` and return ranked results.

        Raises :class:`SearchError` if the LLM translation fails, wrapping the
        underlying error. ``instrument_family`` is accepted for backward
        compatibility but is no longer sent to the model; family is determined
        by the concepts the model returns.
        """
        try:
            translation, llm_record = self.translator.translate(query, instrument_family)
        except Exception as exc:
            raise SearchError(
                f"Failed to translate query {query!r}: {exc}"
            ) from exc
        return self.scorer.score_translation(
            translation,
            query=query,
            llm_call_record=llm_record,
            limit=limit,
        )

    def warm_up(self) -> None:
        """Warm the provider prompt cache with a throwaway translation.

        Mirrors the translator_evaluation harness, which issues a cache-warming call so
        the first real search doesn't pay the cold prompt-processing cost. A
        failure here is non-fatal (the cache simply isn't warmed).
        """
        try:
            self.translator.translate("warm up the prompt cache", "pitched")
        except Exception:
            pass

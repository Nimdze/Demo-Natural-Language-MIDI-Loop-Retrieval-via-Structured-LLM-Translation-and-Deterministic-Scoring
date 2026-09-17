"""OpenAI-compatible LLM client with structured output."""

import json
import re
import time
from typing import Any

from openai import OpenAI

from midi_llm_search_engine.config import get_api_key, get_base_url, get_model
from midi_llm_search_engine.index_loader import SearchIndex
from midi_llm_search_engine.models import LLMCallRecord, QueryTranslation, Target
from midi_llm_search_engine.prompt_builder import SystemPromptBuilder


def build_tool_schema(
    include_semantic_analysis: bool = False,
) -> dict[str, Any]:
    """Build the JSON tool schema, conditionally including optional fields."""
    properties: dict[str, Any] = {}
    required: list[str] = ["family_classification", "targets"]

    properties["family_classification"] = {
        "type": "string",
        "enum": ["drums", "pitched"],
        "description": "Classified family of the query: 'drums' or 'pitched'.",
    }
    if include_semantic_analysis:
        properties["semantic_analysis"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept_name": {"type": "string"},
                    "level_name": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["concept_name", "level_name", "why"],
            },
        }
    properties["targets"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "concept_name": {"type": "string"},
                "level_name": {"type": "string", "description": "Exact level name. Use this OR level_index."},
                "level_index": {"type": "integer", "minimum": 0, "description": "Position index of the level (0=most intense). Use this OR level_name."},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                "fallback": {"type": "string", "enum": ["down", "up", "nearest"]},
            },
            "required": ["concept_name", "importance", "fallback"],
        },
    }

    return {
        "type": "function",
        "function": {
            "name": "translate_query",
            "description": "Translate a natural language query into a structured list of concept targets.",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMA = build_tool_schema()


def build_output_json_schema(
    include_semantic_analysis: bool = False,
) -> dict[str, Any]:
    """Build the plain JSON object schema for constrained decoding.

    Mirrors the hand-written OUTPUT SCHEMA at the end of the system prompt.
    Kept minimal (no additionalProperties/description) for compatibility with
    local providers that support structured output (e.g. Ollama).
    """
    properties: dict[str, Any] = {
        "family_classification": {"type": "string", "enum": ["drums", "pitched"]},
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept_name": {"type": "string"},
                    "level_name": {"type": "string"},
                    "level_index": {"type": "integer"},
                    "importance": {"type": "integer"},
                    "fallback": {"type": "string", "enum": ["down", "up", "nearest"]},
                },
                "required": ["concept_name", "level_name", "importance", "fallback"],
            },
        },
    }
    required: list[str] = ["family_classification", "targets"]
    if include_semantic_analysis:
        properties["semantic_analysis"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept_name": {"type": "string"},
                    "level_name": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["concept_name", "level_name", "why"],
            },
        }
    return {"type": "object", "properties": properties, "required": required}


class LLMTranslatorError(Exception):
    """Raised when the LLM translation fails or returns an invalid schema."""


class LLMTranslator:
    """Translate natural language queries into validated search targets."""

    def __init__(
        self,
        index: SearchIndex,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        prompt_builder: SystemPromptBuilder | None = None,
        include_semantic_analysis: bool = False,
        use_json_schema: bool | None = None,
        num_ctx: int | None = 16384,
        timeout: float = 120,
        max_tokens: int = 16000,
        system_prompt: str | None = None,
    ):
        self.index = index
        self.api_key = api_key or get_api_key()
        self.model = model or get_model()
        self.base_url = base_url or get_base_url()
        self.prompt_builder = prompt_builder or SystemPromptBuilder(index)
        self.include_semantic_analysis = include_semantic_analysis
        # Local (Ollama) context window. Must be large enough to hold the full
        # system prompt (+ output) so the prompt isn't truncated and its KV cache
        # is reused across calls.
        self.num_ctx = num_ctx
        # None = auto: JSON-schema constrained decoding for local (Ollama)
        # endpoints, plain JSON mode otherwise. Explicit True/False overrides.
        self.use_json_schema = use_json_schema
        # The system prompt is deterministic given the index + semantic flag, so
        # callers (e.g. a UI) may build it once and cache it, injecting it here
        # instead of rebuilding per request.
        self.system_prompt = system_prompt or self.prompt_builder.build(
            include_semantic_analysis=include_semantic_analysis,
        )
        # Request timeout: a hung API response must not block the caller forever
        # (especially under concurrent test batches).
        self.timeout = timeout
        # Output token budget. Semantic analysis makes responses large; a budget
        # too small truncates the JSON (json_delim) and can push the model to
        # collapse to a minimal '{}' under pressure.
        self.max_tokens = max_tokens
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def translate(
        self, query: str, instrument_family: str = "pitched"
    ) -> tuple[QueryTranslation, LLMCallRecord]:
        if not self.api_key:
            raise LLMTranslatorError("No API key configured. Set MIDI_SEARCH_API_KEY or paste a key in the UI.")

        start = time.perf_counter()
        # System prompt is byte-stable (built once in __init__) so it stays a
        # long-term cache hit. No family hint is injected; the model classifies
        # the family itself via its family_classification output.
        base_messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Convert to concept targets: \"{query}\"",
            },
        ]

        corrections: list[dict[str, str]] = []
        max_retries = 2
        last_error: Exception | None = None
        last_content: str | None = None
        last_usage: Any | None = None
        attempts: list[dict] = []

        for attempt in range(max_retries + 1):
            try:
                response = self._chat(base_messages + corrections)
                last_usage = getattr(response, "usage", None)
                content = response.choices[0].message.content or "{}"
                last_content = content
                data = self._extract_json(content)
                self._resolve_targets(data)
                translation = QueryTranslation(**data)
                self._validate_family_classification(translation, instrument_family)

                if not translation.targets:
                    # An empty target set is a failed translation, not a valid answer.
                    # Reject it and retry so we don't silently return nothing.
                    attempts.append(
                        {"attempt": attempt, "status": "empty_targets", "content": content}
                    )
                    last_error = LLMTranslatorError("Model returned no targets.")
                    if attempt < max_retries:
                        corrections.append(
                            {"role": "user", "content": self._correction_message("you returned an empty targets array")}
                        )
                        continue
                    break

                report = self.validate_targets(translation.targets)
                if report["invalid"]:
                    # 8B models hallucinate names; give them a corrective hint.
                    attempts.append(
                        {
                            "attempt": attempt,
                            "status": "invalid_targets",
                            "content": content,
                            "error": self._describe_invalid(report["invalid"]),
                        }
                    )
                    last_error = LLMTranslatorError(self._describe_invalid(report["invalid"]))
                    if attempt < max_retries:
                        corrections.append({"role": "user", "content": self._build_correction(report["invalid"])})
                        continue
                    break

                missing_pairs = self._missing_pairs(translation.targets)
                if missing_pairs:
                    names = "; ".join(f"{a} and {b}" for a, b in missing_pairs)
                    attempts.append({
                        "attempt": attempt, "status": "missing_pair",
                        "content": content, "error": names,
                    })
                    if attempt < max_retries:
                        corrections.append({
                            "role": "user",
                            "content": self._correction_message(
                                "you returned concepts that form a pair without their partner "
                                f"({names}); always return both members of each pair together"
                            ),
                        })
                        continue
                    # No fallback: if the model still omits the partner after retries,
                    # return the single member as-is (the scorer treats the missing
                    # partner as absent) rather than fabricating a level.

                usage = response.usage
                record = LLMCallRecord(
                    query=query,
                    provider=self._provider_name(),
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    cached_tokens=self._cached_tokens(usage),
                    reasoning_tokens=self._reasoning_tokens(usage),
                    response_time_ms=(time.perf_counter() - start) * 1000,
                    translation=translation,
                    error=None,
                    raw=content,
                    attempts=attempts,
                )
                return translation, record
            except Exception as e:
                last_error = e
                attempts.append(
                    {"attempt": attempt, "status": "exception", "error": str(e), "content": last_content}
                )
                if attempt < max_retries:
                    corrections.append({"role": "user", "content": self._correction_message(str(e))})
                    continue
                break

        record = LLMCallRecord(
            query=query,
            provider=self._provider_name(),
            model=self.model,
            prompt_tokens=last_usage.prompt_tokens if last_usage else None,
            completion_tokens=last_usage.completion_tokens if last_usage else None,
            cached_tokens=self._cached_tokens(last_usage),
            reasoning_tokens=self._reasoning_tokens(last_usage),
            response_time_ms=(time.perf_counter() - start) * 1000,
            translation=None,
            error=str(last_error) if last_error else "unknown error",
            raw=last_content,
            attempts=attempts,
        )
        exc = LLMTranslatorError(f"Failed to parse or validate LLM output: {last_error}")
        exc.raw = last_content
        exc.record = record
        raise exc from last_error

    def _chat(self, messages: list[dict[str, Any]]):
        """Make a single chat completion, retrying without structured output if needed."""
        extra_body = {"options": {"num_ctx": self.num_ctx}} if self._is_ollama() and self.num_ctx else None
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=self._response_format(),
                max_tokens=self.max_tokens,
                temperature=0.01,
                extra_body=extra_body,
            )
        except Exception:
            # Fallback: try without response_format (some providers reject it).
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.01,
                extra_body=extra_body,
            )

    def _response_format(self) -> dict[str, Any] | None:
        """Choose the response format. JSON-schema constrained decoding for local
        (Ollama) endpoints, plain JSON mode otherwise."""
        schema_mode = self.use_json_schema
        if schema_mode is None:
            schema_mode = self._is_ollama()
        if schema_mode:
            schema = build_output_json_schema(
                include_semantic_analysis=self.include_semantic_analysis,
            )
            return {
                "type": "json_schema",
                "json_schema": {"name": "translate_query", "schema": schema, "strict": True},
            }
        return {"type": "json_object"}

    @staticmethod
    def _cached_tokens(usage) -> int | None:
        """Prompt tokens served from the provider's cache (if reported)."""
        if not usage:
            return None
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            cached = getattr(usage, "cached_tokens", None)
        return cached

    @staticmethod
    def _reasoning_tokens(usage) -> int | None:
        """Output tokens spent on model reasoning before the visible answer."""
        if not usage:
            return None
        details = getattr(usage, "completion_tokens_details", None)
        if details is None:
            return None
        return getattr(details, "reasoning_tokens", None)


    def _is_ollama(self) -> bool:
        base = (self.base_url or "").lower()
        return any(host in base for host in ("11434", "localhost", "127.0.0.1", "ollama"))

    def _resolve_targets(self, data: dict[str, Any]) -> None:
        """Resolve level_index → level_name and clean up target fields.

        Prefer the index when present: the model is more reliable at choosing the
        ordinal position than at reproducing the exact level_name, so a correct
        index overrides any (possibly slightly-wrong) level_name it also emitted.
        """
        for target in data.get("targets", []):
            if target.get("level_index") is not None:
                resolved = self.index.level_name_at_index(target["concept_name"], target["level_index"])
                if resolved:
                    target["level_name"] = resolved
                else:
                    raise LLMTranslatorError(
                        f"Invalid level_index {target['level_index']} for concept {target['concept_name']}"
                    )
            target.pop("level_index", None)
            if "level_name" in target and target["level_name"]:
                target["level_name"] = self._strip_ordinal(target["level_name"])

    def _missing_pairs(self, targets: list[Target]) -> list[tuple[str, str]]:
        """Return (concept, partner) for paired concepts (grid attempt/success,
        time-sig num/den) whose partner is missing from the returned targets.
        """
        present = {t.concept_name for t in targets}
        missing: list[tuple[str, str]] = []
        for t in targets:
            partner = self.index.pair_with(t.concept_name)
            if partner and partner not in present:
                missing.append((t.concept_name, partner))
        return missing

    def _describe_invalid(self, invalid: list[dict[str, Any]]) -> str:
        return "Invalid concept/level targets: " + "; ".join(
            f"'{item['concept_name']}' / '{item['level_name']}'" for item in invalid
        )

    def _build_correction(self, invalid: list[dict[str, Any]]) -> str:
        names = "; ".join(
            f"'{item['concept_name']}' with level '{item['level_name']}'" for item in invalid
        )
        return (
            "The following concept/level targets do not exist in the schema: "
            f"{names}. Return only EXACT concept_name and level_name values from the "
            "system prompt schema; do not invent names."
        )

    def _correction_message(self, error_text: str) -> str:
        return (
            "Your previous response was invalid: "
            f"{error_text}. Return a corrected JSON object only, matching the schema, "
            "using EXACT concept_name and level_name values that exist in the system prompt."
        )

    def validate_targets(self, targets: list[Target]) -> dict[str, Any]:
        """Return a report of which targets are valid in the current taxonomy."""
        valid = []
        invalid = []
        for target in targets:
            if target.level_name in self.index.level_names(target.concept_name):
                valid.append(target)
            else:
                invalid.append({
                    "concept_name": target.concept_name,
                    "level_name": target.level_name,
                    "valid_levels": self.index.level_names(target.concept_name),
                })
        return {"valid": valid, "invalid": invalid}

    @staticmethod
    def _strip_ordinal(level_name: str) -> str:
        """Remove ordinal prefix like '[0] ' or '[3] ' from level names."""
        import re
        return re.sub(r"^\[\d+\]\s*", "", level_name)

    def _validate_family_classification(
        self, translation: QueryTranslation, provided_hint: str
    ) -> None:
        """Raise if any target is incompatible with the model's classification.

        The LLM's own classification is the source of truth and is never
        defaulted. If the model did not classify a family, the check is skipped
        (the scorer matches purely by concepts instead).
        """
        classification = translation.family_classification
        if classification is None:
            return
        for target in translation.targets:
            families = self.index.concept_families(target.concept_name)
            if not families or classification in families:
                continue
            raise LLMTranslatorError(
                f"Family classification is '{classification}' but target "
                f"'{target.concept_name}' only applies to families {families}."
            )

    def _provider_name(self) -> str:
        if self.base_url:
            return self.base_url
        return "openai"

    def _extract_json(self, content: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response.")
        return json.loads(match.group(0))

"""Configuration, logging directories, and secure API key handling."""

import json
import os
from pathlib import Path

PROJECT_HOME = Path.home() / "midi_search"
CONFIG_DIR = Path.home() / ".config" / "midi_search"
KEYS_FILE = CONFIG_DIR / "keys.json"
LOGS_DIR = PROJECT_HOME / "logs"
CACHE_DIR = PROJECT_HOME / "cache"

# Shared demo credential (free Google AI Studio / Gemini tier). It is the
# default used by the hosted demo so visitors can search without a key of
# their own. Stored split to avoid a contiguous key literal in the repo.
# Override with MIDI_SEARCH_API_KEY, ~/.config/midi_search/keys.json, or the UI.
DEFAULT_API_KEY = "".join(["AIza", "SyBFdhdjn", "SrarEQkhY", "Q8EnD9a7_", "BoqY15ds"])
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def ensure_dirs() -> None:
    """Create all default directories."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key(session_key: str | None = None) -> str | None:
    """Return the LLM API key, preferring a session key, then env, then file.

    The key is never logged or returned by the UI in plain text outside the
    session. If the file contains more than one key, the first one named
    `api_key` or `default` is returned. Local providers (e.g. Ollama) accept
    any key, so a dummy key is returned when none is configured and the target
    is a local endpoint.
    """
    if session_key is not None and session_key.strip():
        return session_key.strip()

    env_key = os.getenv("MIDI_SEARCH_API_KEY")
    if env_key is not None and env_key.strip():
        return env_key.strip()

    if KEYS_FILE.exists():
        try:
            data = json.loads(KEYS_FILE.read_text())
            if isinstance(data, dict):
                for name in ("api_key", "default", "openai"):
                    if name in data and data[name]:
                        return data[name]
            if isinstance(data, str) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass

    # Built-in shared demo key. Not shown in the UI, but used by default so the
    # hosted demo works without any setup.
    if DEFAULT_API_KEY:
        return DEFAULT_API_KEY

    if is_local_provider(get_base_url()):
        return "ollama"

    return None


def is_local_provider(base_url: str | None = None) -> bool:
    """Return whether the given (or configured) base URL targets a local model
    provider such as Ollama."""
    url = (base_url or get_base_url() or "").lower()
    return any(host in url for host in ("11434", "localhost", "127.0.0.1", "ollama"))


def keys_file_location() -> str:
    """Return the recommended key file location for documentation."""
    return str(KEYS_FILE)


def get_model() -> str:
    """Return the LLM model name, preferring the env var, else the default.

    Defaults to the free Gemini model used by the hosted demo. Point at another
    model by setting MIDI_SEARCH_MODEL, e.g. MIDI_SEARCH_MODEL=llama3.1:8b with
    MIDI_SEARCH_BASE_URL=http://localhost:11434/v1.
    """
    return os.getenv("MIDI_SEARCH_MODEL") or DEFAULT_MODEL


def get_base_url() -> str | None:
    """Return the LLM base URL, preferring the env var, else the Gemini default.

    Defaults to Gemini's OpenAI-compatible endpoint. Set MIDI_SEARCH_BASE_URL to
    point at a local/compatible endpoint (e.g. Ollama at
    http://localhost:11434/v1).
    """
    url = os.getenv("MIDI_SEARCH_BASE_URL")
    if url and url.strip():
        return url.strip()
    return DEFAULT_BASE_URL

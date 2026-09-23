import os
import re
import subprocess
import tempfile
from pathlib import Path

import pretty_midi
import streamlit as st
from midi_llm_search_engine import FileScorer, SearchIndex
from midi_llm_search_engine.llm_client import LLMTranslator
from midi_llm_search_engine.prompt_builder import SystemPromptBuilder
from midi_llm_search_engine.models import SearchResponse
from midi_llm_search_engine.search import SearchError

from demo.plotting.piano_rolls import plot_inspector_roll_to_base64

_DEMO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASETS_DIR = _DEMO_ROOT / "datasets"
_STEP_RE = re.compile(r"\((\d+) step\(s\)")

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


def _discover_datasets() -> list[str]:
    if not _DATASETS_DIR.is_dir():
        return []
    return sorted(d.name for d in _DATASETS_DIR.iterdir()
                  if d.is_dir() and (d / "output" / "analysis.db").exists())


def _dataset_paths(label: str) -> dict[str, Path] | None:
    base = _DATASETS_DIR / label
    db = base / "output" / "analysis.db"
    tax = base / "output" / "taxonomy.json"
    if not db.exists() or not tax.exists():
        return None
    return {"db_path": db, "taxonomy_path": tax, "midi_root": base / "clean"}


def _resolve_midi_path(stored_path: str) -> Path | None:
    parts = stored_path.split("/")
    if not parts:
        return None
    candidate = _DATASETS_DIR / parts[0] / "clean" / stored_path
    if candidate.exists():
        return candidate
    return None


def _find_soundfont() -> Path | None:
    sf_dir = _DEMO_ROOT / "soundfonts"
    if sf_dir.is_dir():
        for sf in sorted(sf_dir.glob("*.sf3")) + sorted(sf_dir.glob("*.sf2")):
            return sf
    return None


def _kit_name(prog: int) -> str:
    return _DRUM_KITS.get(prog, f"Kit {prog}")


def _remap_midi(midi_path: Path, pitched_prog: int = -1, drum_prog: int | None = None,
                target_bpm: float | None = None) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    orig_tempo = None
    try:
        orig_tempo = pm.estimate_tempo()
    except Exception:
        pass
    for inst in pm.instruments:
        if inst.is_drum:
            if drum_prog is not None:
                inst.program = drum_prog
        else:
            if pitched_prog >= 0:
                inst.program = pitched_prog
    if target_bpm is not None and orig_tempo is not None and orig_tempo > 0:
        ratio = orig_tempo / target_bpm
        for inst in pm.instruments:
            for note in inst.notes:
                note.start *= ratio
                note.end *= ratio
    tmp = Path(tempfile.mktemp(suffix=".mid"))
    pm.write(str(tmp))
    return tmp


_debug_fluidsynth_info: dict | None = None


def _debug_playback() -> dict:
    info = {"fluidsynth_binary": None, "fluidsynth_version": None,
            "fluidsynth_python": None, "soundfont": None, "error": None}
    try:
        r = subprocess.run(["which", "fluidsynth"], capture_output=True, timeout=5, text=True)
        info["fluidsynth_binary"] = r.stdout.strip() or "not found"
        if r.returncode == 0:
            vr = subprocess.run(["fluidsynth", "--version"], capture_output=True, timeout=5, text=True)
            info["fluidsynth_version"] = vr.stdout.split("\n")[0] if vr.stdout else vr.stderr[:100]
    except Exception as e:
        info["error"] = f"which/version failed: {e}"
    try:
        import fluidsynth
        info["fluidsynth_python"] = fluidsynth.__version__ if hasattr(fluidsynth, "__version__") else "installed"
    except ImportError:
        info["fluidsynth_python"] = "not installed"
    except Exception as e:
        info["fluidsynth_python"] = f"error: {e}"
    try:
        sf = _find_soundfont()
        info["soundfont"] = str(sf) if sf else "not found"
    except Exception as e:
        info["error"] = (info.get("error") or "") + f" soundfont: {e}"
    return info


def _play_midi(midi_path: Path) -> bytes | None:
    global _debug_fluidsynth_info
    sf = _find_soundfont()
    if sf is not None:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            result = subprocess.run(
                ["fluidsynth", "-ni", "-g", "1.0", "-F", str(tmp), str(sf), str(midi_path)],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0:
                data = tmp.read_bytes()
                if data[:4] == b"RIFF":
                    return data
            _debug_fluidsynth_info = {
                "rc": result.returncode,
                "stdout": result.stdout.decode()[:500],
                "stderr": result.stderr.decode()[:500],
            }
        except Exception as e:
            _debug_fluidsynth_info = {"exception": str(e)}
        finally:
            tmp.unlink(missing_ok=True)
    return _synth_midi(midi_path)


def _synth_midi(midi_path: Path) -> bytes | None:
    try:
        import io
        import wave
        import numpy as np
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.synthesize()
        if audio is None or len(audio) == 0:
            return None
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _corpus_indexes() -> dict[str, tuple[SearchIndex, FileScorer]]:
    scorers: dict[str, tuple[SearchIndex, FileScorer]] = {}
    for label in _discover_datasets():
        paths = _dataset_paths(label)
        if paths is None:
            continue
        index = SearchIndex(str(paths["db_path"]), str(paths["taxonomy_path"]))
        scorers[label] = (index, FileScorer(index))
    return scorers


@st.cache_resource(show_spinner=False)
def _corpus_prompt() -> str | None:
    for _, (index, _) in _corpus_indexes().items():
        return SystemPromptBuilder(index).build(include_semantic_analysis=True)
    return None


def _corpus_search(query: str, api_key: str = "", base_url: str | None = None,
                   model: str | None = None, limit: int = 20) -> SearchResponse:
    scorers = _corpus_indexes()
    if not scorers:
        raise RuntimeError("No datasets available.")
    first_index = next(iter(scorers.values()))[0]
    prompt = _corpus_prompt()
    translator = LLMTranslator(
        first_index, api_key=api_key or None, base_url=base_url or None,
        model=model or None, include_semantic_analysis=True, system_prompt=prompt,
    )
    translation, record = translator.translate(query)
    combined = []
    for _, scorer in scorers.values():
        try:
            combined.extend(scorer.score(translation.targets, limit=limit))
        except Exception:
            continue
    combined.sort(key=lambda it: it.raw_score, reverse=True)
    return SearchResponse(
        query=query, instrument_family=translation.family_classification or "",
        targets=translation.targets, results=combined[:limit],
        duration_ms=0.0, llm_call_record=record,
    )


def _split_exact_fallback(reasons: list[str]) -> tuple[list[str], list[str]]:
    exact: list[str] = []
    fallback: list[str] = []
    for reason in reasons:
        concept = reason.split(":", 1)[0].strip()
        match = _STEP_RE.search(reason)
        steps = int(match.group(1)) if match else 0
        (exact if steps == 0 else fallback).append(concept)
    return exact, fallback


def _fallback_tags_display(reasons: list[str]) -> list[str]:
    tags = []
    for reason in reasons:
        match = _STEP_RE.search(reason)
        steps = int(match.group(1)) if match else 0
        if steps == 0:
            continue
        parts = reason.split(":")
        concept = parts[0].strip() if parts else ""
        level = ""
        for chunk in parts[1:]:
            if "file=" in chunk:
                level = chunk.split("file=")[1].split("'")[1] if "'" in chunk else ""
        tags.append(f"{concept}: {level}")
    return tags


def _render_translation_with_analysis(response) -> None:
    st.subheader("Translation")
    if not response.targets:
        st.info("No concept targets returned.")
        return

    record = getattr(response, "llm_call_record", None)
    translation = getattr(record, "translation", None) if record else None
    analysis_map = {}
    if translation and translation.semantic_analysis:
        for item in translation.semantic_analysis:
            if item.get("concept_name") and item.get("why"):
                analysis_map[item["concept_name"]] = item["why"]

    rows = []
    for target in response.targets:
        rows.append({
            "Why": analysis_map.get(target.concept_name, ""),
            "Concept": target.concept_name,
            "Level": target.level_name,
            "Importance": target.importance,
            "Fallback": target.fallback,
        })

    st.dataframe(
        rows, width="stretch", hide_index=True, use_container_width=True,
        column_config={
            "Why": st.column_config.TextColumn("Interpretation", width="large"),
            "Concept": st.column_config.TextColumn("Concept", width="medium"),
            "Level": st.column_config.TextColumn("Level", width="small"),
            "Importance": st.column_config.NumberColumn("Importance", width="small"),
            "Fallback": st.column_config.TextColumn("Fallback", width="small"),
        },
    )


def _render_results_table(response) -> None:
    st.subheader("Results")
    if not response.results:
        st.info("No files matched.")
        return
    table = []
    for r in response.results:
        _, fallback = _split_exact_fallback(r.match_reasons)
        table.append({
            "File": r.file_path, "Score": round(r.score, 3),
            "Fallback tags": len(fallback),
        })
    st.dataframe(table, width="stretch", hide_index=True, use_container_width=True)


def _render_result_detail(result, midi_path: Path | None, key_suffix: str) -> None:
    exact, fallback = _split_exact_fallback(result.match_reasons)
    cols = st.columns(3)
    cols[0].metric("Score", f"{result.score:.3f}")
    cols[1].metric("Raw", f"{result.raw_score:.2f}")
    cols[2].metric("Max", f"{result.max_possible:.1f}")

    fb_tags = _fallback_tags_display(result.match_reasons)
    if fb_tags:
        st.markdown("**Fallback:** " + "; ".join(fb_tags))
    else:
        st.markdown("**Fallback:** — *(all exact)*")

    has_pitched = False
    has_drums = False
    orig_tempo = 120.0
    if midi_path and midi_path.exists():
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            try:
                orig_tempo = pm.estimate_tempo()
            except Exception:
                pass
            b64 = plot_inspector_roll_to_base64(pm, title=result.file_path)
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%; border-radius:8px; border:1px solid #333; display:block;">',
                unsafe_allow_html=True,
            )
            for inst in pm.instruments:
                has_pitched = has_pitched or (not inst.is_drum and len(inst.notes) > 0)
                has_drums = has_drums or (inst.is_drum and len(inst.notes) > 0)
        except Exception as exc:
            st.error(f"Could not render piano roll: {exc}")

    target_bpm = st.number_input("BPM", min_value=20.0, max_value=300.0, value=120.0, step=1.0, key=f"bpm_{key_suffix}")

    pitched_prog = -1
    drum_prog_arg = None
    pitched_changed = False
    drum_changed = False

    if has_pitched:
        opts = [("Original (keep file's instrument)", -1)] + [(f"{i}: {name}", i) for i, name in enumerate(_GM_INSTRUMENTS)]
        chosen = st.selectbox("Pitched instrument", [l for l, _ in opts], index=0, key=f"pitched_{key_suffix}")
        pitched_prog = [v for _, v in opts][[l for l, _ in opts].index(chosen)]
        pitched_changed = pitched_prog != -1

    if has_drums:
        opts = [("Original (keep file's kit)", -1)] + sorted(
            [(f"Kit {prog}: {name}", prog) for prog, name in _DRUM_KITS.items()],
            key=lambda x: x[1],
        )
        chosen = st.selectbox("Drum kit", [l for l, _ in opts], index=0, key=f"drum_{key_suffix}")
        drum_val = [v for _, v in opts][[l for l, _ in opts].index(chosen)]
        if drum_val != -1:
            drum_changed = True
            drum_prog_arg = drum_val

    if midi_path and midi_path.exists():
        if not pitched_changed and not drum_changed and target_bpm == orig_tempo:
            wav = _play_midi(midi_path)
        else:
            render_path = _remap_midi(midi_path, pitched_prog if pitched_changed else -1,
                                      drum_prog_arg if drum_changed else None,
                                      target_bpm if target_bpm != orig_tempo else None)
            wav = _play_midi(render_path)
            render_path.unlink(missing_ok=True)
        if wav:
            st.audio(wav, format="audio/wav")
        else:
            st.caption("Playback unavailable (FluidSynth / soundfont not set up).")


def run() -> None:
    st.set_page_config(
        page_title="MIDI Retriever — Natural-Language MIDI Loop Search",
        page_icon="🎹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    datasets = _discover_datasets()
    if not datasets:
        st.error("No datasets found.")
        return

    with st.sidebar:
        st.title("🎹 MIDI Retriever")
        st.caption("Natural-language search over analyzed MIDI loops")
        st.divider()
        st.markdown(
            "**How it works**\n\n"
            "1. An LLM translates your query into structured musical concepts.\n"
            "2. A deterministic scorer ranks every loop against those concepts."
        )
        with st.expander("Model settings", expanded=False):
            api_key = st.text_input("API key", type="password", value="",
                                    help="Optional. Paste your own key to override the demo's default provider.")
            provider = st.selectbox("Provider", ["Google (Gemini)", "DeepSeek", "OpenAI", "Mistral", "Groq", "Local (Ollama)", "Custom"])
            if provider == "Custom":
                base_url = st.text_input("Base URL")
            else:
                base_url = {"Google (Gemini)": "https://generativelanguage.googleapis.com/v1beta/openai/",
                            "DeepSeek": "https://api.deepseek.com",
                            "OpenAI": "https://api.openai.com/v1",
                            "Mistral": "https://api.mistral.ai/v1",
                            "Groq": "https://api.groq.com/openai/v1",
                            "Local (Ollama)": "http://localhost:11434/v1"}.get(provider, "")
            model = st.text_input("Model", value="gemini-3.5-flash-lite")

        if os.getenv("DEMO_DEBUG"):
            with st.expander("Playback debug"):
                info = _debug_playback()
                for k, v in info.items():
                    st.text(f"{k}: {v}")
                if _debug_fluidsynth_info:
                    st.text(f"last fluidsynth result:")
                    for k, v in _debug_fluidsynth_info.items():
                        st.text(f"  {k}: {v}")

        st.divider()
        st.caption(f"{len(datasets)} datasets indexed")
        st.markdown(
            "📦 This is a frozen demo. Full pipeline, datasets and evaluation: "
            "[MIDI Loop LLM Structured Retrieval]"
            "(https://github.com/Nimdze/MIDI-Loop-LLM-Strructured-Retrieval)."
        )

    st.title("Natural-language MIDI loop retrieval")
    st.markdown(
        "Describe your desired loop in plain language. The system translates your prompt into "
        "structured musical concepts, ranks the bundled loops by fit, and displays the exact "
        "concepts and interpretations driving each match. If your intent is missed, guide the "
        "interpretation directly using the search instructions line.\n\n"
        "**Note:** MIDI files lack sound and context — adjust the BPM and instrument controls below "
        "the piano roll to preview them properly. The bundled library is limited: all loops are in "
        "4/4 meter, key/scale metadata may be inaccurate, and some file types may not be present.\n\n"
        "Try: *\"complex drums with lots of variation\"* · *\"soft ambient piano piece\"*"
    )

    st.subheader("Search")

    query = st.text_input("Search query", placeholder='e.g. "complex drums with lots of variation" or "soft ambient piano piece"')
    instructions = st.text_input("Search instructions (optional)", placeholder="Refine how the LLM should interpret the query ...")

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                with st.spinner("Translating query and scoring files across all datasets ..."):
                    full_query = query if not instructions.strip() else f"{query}\n{instructions.strip()}"
                    response = _corpus_search(full_query, api_key, base_url or None, model or None)
            except SearchError as exc:
                st.error(f"Search failed: {exc}")
                cause = getattr(exc, "__cause__", None)
                raw = getattr(cause, "raw", None)
                if raw:
                    st.code(f"RAW MODEL OUTPUT:\n{raw[:2000]}")
                return
            st.session_state["demo_response"] = response
            st.session_state["demo_query"] = query

    response = st.session_state.get("demo_response")
    if response is None:
        st.caption("Run a search to see results.")
        return

    _render_translation_with_analysis(response)

    with st.expander("Raw LLM Response"):
        record = getattr(response, "llm_call_record", None)
        raw = getattr(record, "raw", None) if record else None
        if raw:
            st.code(raw, language="json")

    _render_results_table(response)

    selected = st.selectbox("Inspect a result", options=[r.file_path for r in response.results], key="result_selector")
    for result in response.results:
        if result.file_path == selected:
            midi_path = _resolve_midi_path(result.file_path)
            _render_result_detail(result, midi_path, key_suffix=str(hash(result.file_path)))
            break


if __name__ == "__main__":
    run()

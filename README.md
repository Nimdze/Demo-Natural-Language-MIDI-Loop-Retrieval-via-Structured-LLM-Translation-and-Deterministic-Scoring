# MIDI Retriever — Natural-Language MIDI Loop Search

**Structured LLM query translation + deterministic scoring for semantic retrieval over analyzed MIDI libraries.**

Describe a loop in plain language — *"fast complex drums with lots of variation"*, *"soft ambient
piano piece"*, *"syncopated funk bass with a loose feel"* — and the system returns the closest
matches from a library of analyzed MIDI loops, with the exact musical concepts that drove every
ranking.

> **Note:** The results are limited to the fixed MIDI loop library attached to the demo (MUSICO's
> datasets ASF-4, HP-10, Cinematic and Dubstep) which highly influences search results. Note also
> that MIDI loops are extremely malleable, and results very often have to have their tempo or
> rendering synth changed in order to be more satisfactory. Additionally, this is an experiment
> exploring a debuggable and customizable search approach, utilize the "search instructions" bar
> to try to steer the system to interpret your queries in a more exact way, and use the interpretation
> column in the result to see how the LLM interprets each query.

> [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://midi-loop-llm-retrieval.streamlit.app/)
>
> **Live demo:** <https://midi-loop-llm-retrieval.streamlit.app/>

---

## About this repository

This is the **self-contained demo** released as a companion to the paper: a frozen snapshot of the
retrieval application, bundled with four pre-analyzed libraries so it runs and deploys on its own.

The full project lives in the main repository — MIDI preprocessing, feature extraction and tagging,
taxonomy generation, and the search-engine package:

**➡️ [Nimdze/MIDI-Loop-LLM-Strructured-Retrieval](https://github.com/Nimdze/MIDI-Loop-LLM-Strructured-Retrieval)**

See that repository for how the datasets are produced, how the taxonomy is generated, and the
retrieval experiments. Only the subset of the search engine needed to run the demo is vendored here.

## Why this approach

Many retrieval systems approximate relevance with a learned model, which is opaque and needs
training data. This demo takes a different route:

1. **An LLM translates the query** into a structured set of musical-concept targets drawn from a
   fixed taxonomy.
2. **A deterministic scorer** matches those targets against precomputed per-file feature tags and
   ranks the library.

Because the whole pipeline is explicit, every result can be explained: the UI shows the concept
targets the model selected and which of them matched each file exactly or via a tolerance step.
There is nothing to train and no black-box similarity score.

## Features

- **Natural-language search** across multiple MIDI libraries at once.
- **Explainable results** — inspect the translated concept targets and the per-file match reasons.
- **In-browser auditioning** — every hit renders a piano roll, plays through FluidSynth, and lets
  you change BPM, pitched instrument, and drum kit to hear the loop as you intended it.

## Bundled datasets

The demo ships with four pre-analyzed libraries — **ASF-4**, **Cinematic**, **Dubstep**, and
**HP-10** (~470 MB total). Each lives in `datasets/<name>/`, with the MIDI loops in `clean/` and a
per-file tag database plus taxonomy in `output/`; any dataset following that layout is discovered
automatically at startup. See the main repository for the pipeline that produces them.

**Dataset terms.** The bundled MIDI loops are governed by their own terms and are **not** covered
by this repository. The CONLON datasets (**ASF-4**, **HP-10**) are made available **exclusively for
academic research purposes and cannot be used in any commercial project** — see the
[dataset terms](https://paolo-f.github.io/CONLON/datasets.html) and cite Angioloni et al., *CONLON*
(2020) if you use them. Other bundled libraries retain their original terms.

## Running locally

Requires Python 3.14 and [FluidSynth](https://www.fluidsynth.org/) for playback (on Debian/Ubuntu:
`apt-get install fluidsynth fluid-soundfont-gm`; on macOS: `brew install fluid-synth`).

```bash
python3 -m venv .venv
.venv/bin/pip install .
.venv/bin/streamlit run src/demo/app.py
```

Or use the convenience script:

```bash
./run.sh
```

To use your own key or a different provider, expand **Model settings** in the sidebar (any
OpenAI-compatible endpoint works, including a local Ollama server).

## Deployment

The app is a standard Streamlit app and can be hosted on
[Streamlit Community Cloud](https://share.streamlit.io) (main file path `src/demo/app.py`).
FluidSynth is installed via `packages.txt`.

## License

The application code is released under the [MIT License](LICENSE).

The bundled MIDI loops are **not** covered by that license and are governed by their own terms —
see **Dataset terms** above. In particular, the CONLON datasets (ASF-4, HP-10) are provided for
academic research only and may not be used commercially.

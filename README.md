# Local XTTS-v2 Text-to-Speech Starter

This repository contains a minimal, GPU-ready Python framework for converting
text files into speech using [Coqui TTS](https://github.com/coqui-ai/TTS) and
the multilingual **XTTS-v2** model. It is tailored for users with NVIDIA GPUs
(such as the RTX 5070 Ti) who want a local workflow that is fast, configurable,
and easy to extend.

## Features

- 🔊 Convert any UTF-8 text file into a `.wav` audio file.
- ⚡ Automatically takes advantage of CUDA GPUs when available.
- 🗣️ Optional voice cloning by providing your own reference speaker WAV file.
- 🧰 Simple CLI written in Python so you can integrate it into larger projects.

## Prerequisites

- **Python**: Coqui TTS currently supports CPython 3.10–3.12. If you are running
  Python 3.13, create a 3.12 virtual environment (e.g., via
  [pyenv](https://github.com/pyenv/pyenv) or
  [uv](https://docs.astral.sh/uv/)) so that PyTorch and TTS can be installed.
- **GPU**: An NVIDIA GPU with recent drivers. CUDA 12.4 wheels are referenced in
  `requirements.txt`.
- **OS**: Windows instructions are shown below, but the same steps apply to
  Linux/macOS with minor shell changes.

## Environment Setup (Windows PowerShell)

```powershell
# Clone or download this repository
cd path\to\TTS_local

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip (recommended)
pip install --upgrade pip

# Install dependencies (CUDA wheels pulled from download.pytorch.org)
pip install -r requirements.txt
```

> **Note:** If you need CPU-only inference, remove the extra index line and
> install the matching CPU wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

## Environment Setup (Linux/macOS Bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

The CLI entry point lives in `src/generate_audio.py`. It accepts a text file,
produces a WAV file, and exposes knobs for language, model selection, device,
and optional voice cloning.

```bash
python src/generate_audio.py \
  --input-file samples/example.txt \
  --output-file outputs/example.wav \
  --language en \
  --device auto \
  --overwrite
```

Key options:

- `--model-name`: Defaults to `tts_models/multilingual/multi-dataset/xtts_v2`.
  Replace this with any other Coqui model ID you have available locally.
- `--speaker-wav`: Path to a short (≈3–10 s) WAV file if you want to clone a
  specific speaker. Leave unset to use the built-in XTTS voice.
- `--device`: `auto` tries CUDA first; `cpu` forces CPU inference.
- `--overwrite`: Allows you to regenerate a file without manually deleting it.

All generated files are standard 16-bit PCM WAV files that can be used anywhere.

## Adding Your Own Text

Place your text inside any UTF-8 file (for example `input/story.txt`) and pass
its path through `--input-file`. The script concatenates the contents and feeds
it directly into XTTS-v2, so paragraph spacing will be honored in the generated
speech.

## Next Steps

- Experiment with other Coqui multilingual checkpoints by changing
  `--model-name`.
- Adjust pronunciation or pacing by splitting long passages into multiple files
  and generating several clips.
- Wrap `generate_audio.py` with a FastAPI/Flask service if you need to serve TTS
  over HTTP.

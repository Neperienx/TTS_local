# Local XTTS-v2 Text-to-Speech Starter

This repository contains a minimal, GPU-ready Python framework for converting
text files into speech using either [Coqui TTS](https://github.com/coqui-ai/TTS)
with the multilingual **XTTS-v2** model or the
[Suno Bark](https://github.com/suno-ai/bark) neural codec TTS system. It is
tailored for users with NVIDIA GPUs (such as the RTX 5070 Ti) who want a local
workflow that is fast, configurable, and easy to extend.

## Features

- 🔊 Convert any UTF-8 text file into a `.wav` audio file.
- ⚡ Automatically takes advantage of CUDA GPUs when available.
- 🗣️ Optional voice cloning when using XTTS or Bark history prompts for Bark.
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

# Optional: install Bark if you plan to use --engine bark
pip install "git+https://github.com/suno-ai/bark.git"
```

> **Note:** If you need CPU-only inference, remove the extra index line and
> install the matching CPU wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

## Environment Setup (Linux/macOS Bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional Bark backend
pip install "git+https://github.com/suno-ai/bark.git"
```

## Usage

The CLI entry point lives in `src/generate_audio.py`. It accepts a text file,
produces a WAV file, and exposes knobs for engine selection, language, model,
device, and optional voice cloning/history prompts.

```bash
python src/generate_audio.py --input-file samples/example.txt --output-file outputs/example.wav --language en --device auto --engine xtts --overwrite
```

For a quick multilingual comparison using the same voice, try the bundled sample:

```bash
python src/generate_audio.py --input-file samples/multilingual_voice_test.txt --output-file outputs/multilingual_voice_test.wav --language fr --device auto --engine xtts --speaker-wav samples/voice_clone_sample.wav --overwrite
```

Key options:

- `--model-name`: Defaults to `tts_models/multilingual/multi-dataset/xtts_v2`.
  Replace this with any other Coqui model ID you have available locally.
- `--speaker-wav`: Path to a short (≈3–10 s) WAV file if you want to clone a
  specific speaker. Leave unset to use the built-in XTTS voice.
- `--device`: `auto` tries CUDA first; `cpu` forces CPU inference.
- `--overwrite`: Allows you to regenerate a file without manually deleting it.
- `--engine`: Choose `xtts` (default) for Coqui models or `bark` for Suno Bark.
- `--history-prompt`: When using Bark, provide a preset (e.g., `v2/en_speaker_6`)
  or a custom history file path.

> **XTTS vs Bark.** XTTS supports explicit `--language` and `--speaker-wav`
> cloning. Bark instead relies on the `--history-prompt` parameter to control
> voice/tone, and currently only supports languages bundled with Bark.

### Using a sample WAV for voice cloning

To clone a voice with XTTS, supply a clean reference clip through `--speaker-wav`.
The sample WAV should:

- Be 3–10 seconds long.
- Contain a single speaker, with no background music or heavy noise.
- Use a standard WAV format (16-bit PCM is ideal).

Place your sample WAV in the `samples/` folder (for example,
`samples/voice_clone_sample.wav`) and run:

```bash
python src/generate_audio.py --input-file samples/example.txt --output-file outputs/voice_clone_example.wav --language en --device auto --engine xtts --speaker-wav samples/voice_clone_sample.wav --overwrite
```

If you already have a sample WAV elsewhere, you can pass its full path instead of
placing it in `samples/`. When the reference clip is clear and consistent, XTTS
should reproduce that voice in the generated speech.

All generated files are standard 16-bit PCM WAV files that can be used anywhere.

## Building a Narrated Slideshow MP4

Use `src/generate_story_video.py` to turn a JSON story (like
`Projects/example/text.json`) and a folder of images into a narrated MP4
slideshow. Each page includes a 2-second pre-roll before the narration and a
5-second buffer after the narration finishes so viewers can linger on the
image. This script also emits a combined narration WAV file for reuse.

```bash
python src/generate_story_video.py --story-file Projects/example/text.json --output-video outputs/example_story.mp4 --output-audio outputs/example_story.wav --output-dir outputs/example_story_assets --engine xtts --language en --device auto --overwrite
```

> **Requirements:** The script uses `ffmpeg` to assemble the MP4. Install it via
> your package manager (or download it from [ffmpeg.org](https://ffmpeg.org/))
> and ensure it is available on your `PATH`.

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

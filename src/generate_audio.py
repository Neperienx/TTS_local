"""CLI entry point for generating speech audio files from text files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from TTS.api import TTS
import soundfile as sf

DEFAULT_ENGINE = "xtts"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an audio file from a UTF-8 text file using the Coqui XTTS-v2 model. "
            "The script automatically detects CUDA GPUs unless a device is provided."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the UTF-8 encoded text file that contains the narration.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/xtts_output.wav"),
        help="Destination path for the generated WAV file (default: outputs/xtts_output.wav).",
    )
    parser.add_argument(
        "--engine",
        choices=["xtts", "bark"],
        default=DEFAULT_ENGINE,
        help=(
            "Inference backend to use. XTTS relies on Coqui models while Bark uses"
            " the Suno Bark neural codec architecture (default: xtts)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"Coqui model identifier to load (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Target language code supported by the model (default: en).",
    )
    parser.add_argument(
        "--speaker-wav",
        type=Path,
        default=None,
        help=(
            "Path to a reference speaker WAV file for voice cloning. "
            "Required for XTTS unless a valid --speaker-id is provided."
        ),
    )
    parser.add_argument(
        "--speaker-id",
        default=None,
        help=(
            "Optional speaker ID for XTTS models that ship with speaker embeddings. "
            "If omitted and the model includes speakers, the first available speaker"
            " is used automatically."
        ),
    )
    parser.add_argument(
        "--history-prompt",
        default=None,
        help=(
            "Optional Bark history prompt to control the speaker/tone. This is only"
            " used when --engine bark is selected."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device to run inference on. 'auto' selects CUDA when available, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def resolve_device(arg_value: str) -> str:
    if arg_value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_value


def read_text_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {path} already exists. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def synthesize_with_xtts(
    text: str,
    output_file: Path,
    model_name: str,
    language: str,
    speaker_wav: Optional[Path],
    speaker_id: Optional[str],
    device: str,
) -> None:
    speaker_wav_path = str(speaker_wav) if speaker_wav else None
    tts = TTS(model_name=model_name, progress_bar=True)
    speaker_id_to_use = speaker_id
    if not speaker_wav_path and not speaker_id:
        available_speakers = getattr(tts, "speakers", None) or []
        if available_speakers:
            speaker_id_to_use = (
                "default"
                if "default" in available_speakers
                else available_speakers[0]
            )
            print(
                "No reference speaker provided; using built-in speaker"
                f" '{speaker_id_to_use}'."
            )
        else:
            raise ValueError(
                "XTTS requires a reference voice. Provide --speaker-wav or "
                "--speaker-id (if the model supplies speaker embeddings). "
                "Alternatively, try --engine bark for a non-cloned voice."
            )
    tts.tts_to_file(
        text=text,
        file_path=str(output_file),
        language=language,
        speaker_wav=speaker_wav_path,
        speaker=speaker_id_to_use,
    )


def synthesize_with_bark(
    text: str,
    output_file: Path,
    history_prompt: Optional[str],
    device: str,
) -> None:
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
    except ImportError as exc:  # pragma: no cover - guidance for missing optional dep
        raise RuntimeError(
            "Bark support requires the optional 'bark' dependency. Install it via "
            "`pip install git+https://github.com/suno-ai/bark.git` or another Bark "
            "distribution to enable --engine bark."
        ) from exc

    use_gpu = device == "cuda"
    preload_models(
        text_use_gpu=use_gpu,
        coarse_use_gpu=use_gpu,
        fine_use_gpu=use_gpu,
        codec_use_gpu=use_gpu,
    )
    audio_array = generate_audio(text, history_prompt=history_prompt)
    sf.write(output_file, audio_array, SAMPLE_RATE)


def synthesize_speech(
    text: str,
    output_file: Path,
    model_name: str,
    language: str,
    speaker_wav: Optional[Path],
    speaker_id: Optional[str],
    device: str,
    engine: str,
    history_prompt: Optional[str],
) -> None:
    if engine == "bark":
        synthesize_with_bark(
            text=text,
            output_file=output_file,
            history_prompt=history_prompt,
            device=device,
        )
        return

    synthesize_with_xtts(
        text=text,
        output_file=output_file,
        model_name=model_name,
        language=language,
        speaker_wav=speaker_wav,
        speaker_id=speaker_id,
        device=device,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(
        "Preparing to generate audio using the"
        f" {device.upper()} device (requested: {args.device})."
    )

    text = read_text_file(args.input_file)
    ensure_output_path(args.output_file, args.overwrite)

    synthesize_speech(
        text=text,
        output_file=args.output_file,
        model_name=args.model_name,
        language=args.language,
        speaker_wav=args.speaker_wav,
        speaker_id=args.speaker_id,
        device=device,
        engine=args.engine,
        history_prompt=args.history_prompt,
    )

    print(
        "Audio successfully generated at"
        f" {args.output_file.resolve()} using {args.engine.upper()} on {device}."
    )


if __name__ == "__main__":
    main()

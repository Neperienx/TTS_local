"""CLI entry point for generating speech audio files from text files using Coqui XTTS-v2."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from TTS.api import TTS

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
            "Optional path to a reference speaker WAV file for voice cloning. "
            "If omitted, the default XTTS reference voice will be used."
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


def synthesize_speech(
    text: str,
    output_file: Path,
    model_name: str,
    language: str,
    speaker_wav: Optional[Path],
    device: str,
) -> None:
    speaker_wav_path = str(speaker_wav) if speaker_wav else None
    use_gpu = device == "cuda"

    tts = TTS(model_name=model_name, progress_bar=True)
    tts.tts_to_file(
        text=text,
        file_path=str(output_file),
        language=language,
        speaker_wav=speaker_wav_path,
        gpu=use_gpu,
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
        device=device,
    )

    print(
        f"Audio successfully generated at {args.output_file.resolve()} using {args.model_name} on {device}."
    )


if __name__ == "__main__":
    main()

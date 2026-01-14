"""Generate a narrated slideshow MP4 from a JSON story description."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from generate_audio import (
    DEFAULT_MODEL,
    resolve_device,
    synthesize_speech,
)

PRE_VOICE_BUFFER_SECONDS = 2.0
POST_VOICE_BUFFER_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an MP4 slideshow with per-page narration from a JSON story."
        )
    )
    parser.add_argument(
        "--story-file",
        type=Path,
        required=True,
        help="Path to the JSON story file (see Projects/example/text.json).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/story.mp4"),
        help="Destination MP4 file for the narrated slideshow.",
    )
    parser.add_argument(
        "--output-audio",
        type=Path,
        default=Path("outputs/story_narration.wav"),
        help="Destination WAV file containing the full narration with buffers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/story_assets"),
        help="Directory for intermediate audio/video assets.",
    )
    parser.add_argument(
        "--engine",
        choices=["xtts", "bark"],
        default="xtts",
        help="Inference backend to use for narration (default: xtts).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"Coqui model identifier to load (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Override the language for narration (default: value in JSON).",
    )
    parser.add_argument(
        "--speaker-wav",
        type=Path,
        default=None,
        help="Optional reference speaker WAV file for XTTS voice cloning.",
    )
    parser.add_argument(
        "--speaker-id",
        default=None,
        help="Optional XTTS speaker ID to use if available.",
    )
    parser.add_argument(
        "--history-prompt",
        default=None,
        help="Optional Bark history prompt (used only with --engine bark).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference on. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def load_story(story_file: Path) -> Dict[str, Any]:
    if not story_file.exists():
        raise FileNotFoundError(f"Story file not found: {story_file}")
    with story_file.open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pad_audio(
    audio: np.ndarray, sample_rate: int
) -> Tuple[np.ndarray, float]:
    pre_samples = int(PRE_VOICE_BUFFER_SECONDS * sample_rate)
    post_samples = int(POST_VOICE_BUFFER_SECONDS * sample_rate)
    silence_pre = np.zeros((pre_samples, *audio.shape[1:]), dtype=audio.dtype)
    silence_post = np.zeros((post_samples, *audio.shape[1:]), dtype=audio.dtype)
    padded_audio = np.concatenate([silence_pre, audio, silence_post], axis=0)
    duration = padded_audio.shape[0] / sample_rate
    return padded_audio, duration


def synthesize_page_audio(
    text: str,
    output_file: Path,
    model_name: str,
    language: str,
    speaker_wav: Optional[Path],
    speaker_id: Optional[str],
    device: str,
    engine: str,
    history_prompt: Optional[str],
) -> Tuple[np.ndarray, int]:
    ensure_parent(output_file)
    synthesize_speech(
        text=text,
        output_file=output_file,
        model_name=model_name,
        language=language,
        speaker_wav=speaker_wav,
        speaker_id=speaker_id,
        device=device,
        engine=engine,
        history_prompt=history_prompt,
    )
    audio, sample_rate = sf.read(output_file)
    return audio, sample_rate


def run_ffmpeg(command: List[str]) -> None:
    subprocess.run(command, check=True)


def build_video_segment(
    image_path: Path,
    audio_path: Path,
    duration: float,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    )


def concat_segments(segments: List[Path], output_video: Path, list_path: Path) -> None:
    ensure_parent(output_video)
    list_path.write_text(
        "\n".join([f"file '{segment.as_posix()}'" for segment in segments])
        + "\n",
        encoding="utf-8",
    )
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_video),
        ]
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    story = load_story(args.story_file)

    pages = story.get("pages", [])
    if not pages:
        raise ValueError("Story JSON must include a non-empty 'pages' list.")

    language = args.language or story.get("language", "en")
    base_dir = args.story_file.parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    narration_segments: List[np.ndarray] = []
    segment_paths: List[Path] = []
    sample_rate: Optional[int] = None

    for idx, page in enumerate(pages, start=1):
        image_path = base_dir / page["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        text = page["text"]

        audio_path = output_dir / f"page_{idx:02d}.wav"
        audio, page_sample_rate = synthesize_page_audio(
            text=text,
            output_file=audio_path,
            model_name=args.model_name,
            language=language,
            speaker_wav=args.speaker_wav,
            speaker_id=args.speaker_id,
            device=device,
            engine=args.engine,
            history_prompt=args.history_prompt,
        )
        if sample_rate is None:
            sample_rate = page_sample_rate
        elif sample_rate != page_sample_rate:
            raise ValueError("All generated audio must use the same sample rate.")

        padded_audio, duration = pad_audio(audio, page_sample_rate)
        sf.write(audio_path, padded_audio, page_sample_rate)
        narration_segments.append(padded_audio)

        segment_path = output_dir / f"segment_{idx:02d}.mp4"
        build_video_segment(
            image_path=image_path,
            audio_path=audio_path,
            duration=duration,
            output_path=segment_path,
        )
        segment_paths.append(segment_path)

    if sample_rate is None:
        raise ValueError("Failed to generate narration audio.")

    combined_narration = np.concatenate(narration_segments, axis=0)
    ensure_parent(args.output_audio)
    sf.write(args.output_audio, combined_narration, sample_rate)

    list_file = output_dir / "segments.txt"
    concat_segments(segment_paths, args.output_video, list_file)

    print(f"Saved narration to {args.output_audio.resolve()}")
    print(f"Saved video to {args.output_video.resolve()}")


if __name__ == "__main__":
    main()

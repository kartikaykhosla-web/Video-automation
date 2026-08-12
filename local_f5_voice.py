import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything


SMALL_HINDI_MODEL_CONFIG = {
    "dim": 768,
    "depth": 18,
    "heads": 12,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
}


def ffmpeg_path() -> str:
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    homebrew = Path("/opt/homebrew/bin/ffmpeg")
    if homebrew.exists():
        return str(homebrew)
    raise FileNotFoundError("FFmpeg is required to prepare the reference segment.")


def delivery_units(script: str) -> list[tuple[str, float]]:
    """Render complete sentences so the reference prompt is used only once per thought."""
    cleaned = re.sub(r"\s+", " ", script).strip()
    pieces = re.findall(r".*?(?:[।.!?]+|$)", cleaned)
    units: list[tuple[str, float]] = []
    for raw_piece in pieces:
        piece = raw_piece.strip()
        if not piece:
            continue
        if re.search(r"\?+$", piece):
            pause = 0.58
        else:
            pause = 0.42
        units.append((piece, pause))
    if units:
        units[-1] = (units[-1][0], 0.0)
    return units


def soften_boundaries(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prevent clicks when independently rendered sentences are joined."""
    fade_samples = min(int(sample_rate * 0.015), wave.size // 2)
    if fade_samples <= 1:
        return wave
    softened = wave.copy()
    softened[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
    softened[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return softened


def delivery_style(text: str) -> str:
    bare = text.strip()
    if bare.endswith("?") or re.search(r"\b(क्या|क्यों|कैसे|कब|कहाँ|कौन)\b", bare):
        return "question"
    serious_markers = (
        "शर्म",
        "लीक",
        "आत्महत्या",
        "भविष्य",
        "अंधेरे",
        "असंवेदनशील",
        "विरोध",
        "प्रदर्शन",
        "फेलियर",
        "माफिया",
        "कार्रवाई",
        "आरोप",
        "गलत",
        "नहीं",
    )
    if any(marker in bare for marker in serious_markers):
        return "serious"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hindi speech with the fine-tuned Deepika F5 pilot."
    )
    parser.add_argument("--script-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--reference-source", required=True, type=Path)
    parser.add_argument("--reference-start", required=True, type=float)
    parser.add_argument("--reference-end", required=True, type=float)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--serious-reference-start", type=float)
    parser.add_argument("--serious-reference-end", type=float)
    parser.add_argument("--serious-reference-text", default="")
    parser.add_argument("--question-reference-start", type=float)
    parser.add_argument("--question-reference-end", type=float)
    parser.add_argument("--question-reference-text", default="")
    parser.add_argument("--nfe-steps", type=int, default=16)
    parser.add_argument("--speed", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    for required in (
        args.script_file,
        args.checkpoint,
        args.vocab,
        args.reference_source,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    script = args.script_file.read_text(encoding="utf-8").strip()
    if not script:
        raise ValueError("The voice script is empty.")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    seed_everything(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="deepika-f5-reference-") as temp_dir:
        temp_root = Path(temp_dir)

        def extract_reference(
            name: str,
            start: float,
            end: float,
            text: str,
        ) -> tuple[Path, str]:
            reference_wav = temp_root / f"{name}.wav"
            subprocess.run(
                [
                    ffmpeg_path(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-to",
                    f"{end:.3f}",
                    "-i",
                    str(args.reference_source),
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    str(reference_wav),
                ],
                check=True,
            )
            reference_audio, reference_rate = sf.read(str(reference_wav))
            if reference_audio.ndim > 1:
                reference_audio = reference_audio.mean(axis=1)
            reference_audio, _ = librosa.effects.trim(
                reference_audio,
                top_db=40,
                frame_length=1024,
                hop_length=256,
            )
            reference_audio = np.concatenate(
                [
                    reference_audio,
                    np.zeros(int(reference_rate * 0.05), dtype=reference_audio.dtype),
                ]
            )
            sf.write(str(reference_wav), reference_audio, reference_rate)
            normalized_text = re.sub(r"[।.!?\s]+$", "", text.strip()) + "। "
            return reference_wav, normalized_text

        references = {
            "neutral": extract_reference(
                "neutral",
                args.reference_start,
                args.reference_end,
                args.reference_text,
            )
        }
        if (
            args.serious_reference_start is not None
            and args.serious_reference_end is not None
            and args.serious_reference_text.strip()
        ):
            references["serious"] = extract_reference(
                "serious",
                args.serious_reference_start,
                args.serious_reference_end,
                args.serious_reference_text,
            )
        if (
            args.question_reference_start is not None
            and args.question_reference_end is not None
            and args.question_reference_text.strip()
        ):
            references["question"] = extract_reference(
                "question",
                args.question_reference_start,
                args.question_reference_end,
                args.question_reference_text,
            )

        model = load_model(
            DiT,
            SMALL_HINDI_MODEL_CONFIG,
            str(args.checkpoint),
            vocab_file=str(args.vocab),
            device=device,
        )
        vocoder = load_vocoder(device=device)
        rendered_units: list[np.ndarray] = []
        sample_rate = 24000
        units = delivery_units(script)
        for index, (unit, pause_seconds) in enumerate(units, 1):
            style = delivery_style(unit)
            reference_wav, reference_text = references.get(
                style,
                references["neutral"],
            )
            print(
                f"delivery_unit={index}/{len(units)} "
                f"style={style} pause={pause_seconds:.2f}"
            )
            cfg_strength = 2.15 if style != "neutral" else 2.0
            generated, sample_rate, _ = infer_process(
                str(reference_wav),
                reference_text,
                unit,
                model,
                vocoder,
                nfe_step=args.nfe_steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=-1.0,
                speed=args.speed,
                cross_fade_duration=0.0,
                device=device,
                indic=True,
            )
            generated = soften_boundaries(
                generated,
                sample_rate,
            )
            rendered_units.append(generated)
            if pause_seconds:
                rendered_units.append(
                    np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)
                )
        generated = np.concatenate(rendered_units)
        sf.write(str(args.output), generated, sample_rate)

    print(f"engine=deepika-f5-pilot device={device} output={args.output}")


if __name__ == "__main__":
    main()

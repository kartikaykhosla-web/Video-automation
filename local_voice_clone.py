import argparse
import os
import re
import time
import types
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
import torchaudio
from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
    Conditionals,
    drop_invalid_tokens,
    punc_norm,
)
from chatterbox.models.s3gen import s3gen as s3gen_module


class MpsSafeResampler:
    """Run torchaudio resampling on CPU to avoid the MPS channel limit."""

    def __init__(self, source_rate: int, target_rate: int):
        self.resampler = torchaudio.transforms.Resample(source_rate, target_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.resampler(waveform.detach().cpu()).to(waveform.device)


_original_get_resampler = s3gen_module.get_resampler


def get_mps_safe_resampler(source_rate: int, target_rate: int, device):
    if str(device) == "mps":
        return MpsSafeResampler(source_rate, target_rate)
    return _original_get_resampler(source_rate, target_rate, device)


s3gen_module.get_resampler = get_mps_safe_resampler


def make_hifigan_mps_safe(model) -> None:
    """Keep generation on MPS but run the unsupported HiFi-GAN decoder on CPU."""
    if str(model.device) != "mps":
        return

    model.s3gen.mel2wav.to("cpu")
    model.s3gen.trim_fade = model.s3gen.trim_fade.cpu()

    def cpu_hift_inference(self, speech_feat, cache_source=None):
        speech_feat = speech_feat.detach().cpu()
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0, dtype=speech_feat.dtype)
        else:
            cache_source = cache_source.detach().cpu()
        return self.mel2wav.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )

    model.s3gen.hift_inference = types.MethodType(cpu_hift_inference, model.s3gen)


def generate_fast(
    model,
    text: str,
    language_id: str,
    temperature: float = 0.65,
) -> torch.Tensor:
    """Use Chatterbox's single-pass decoder instead of the 2x CFG decoder."""
    normalized = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(
        normalized,
        language_id=language_id.lower() if language_id else None,
    ).to(device=model.device, dtype=torch.long)
    text_tokens = F.pad(text_tokens, (1, 0), value=model.t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=model.t3.hp.stop_text_token)
    speech_tokens = model.t3.inference_turbo(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        temperature=temperature,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.5,
        max_gen_len=1000,
    )
    speech_tokens = drop_invalid_tokens(speech_tokens).to(model.device)
    waveform, _ = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=2,
    )
    waveform = waveform.squeeze(0).detach().cpu().numpy()
    watermarked = model.watermarker.apply_watermark(waveform, sample_rate=model.sr)
    return torch.from_numpy(watermarked).unsqueeze(0)


def split_script(text: str, max_chars: int = 360) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    sentences = re.split(r"(?<=[।.!?])\s*", clean)
    units: list[str] = []
    for sentence in sentences:
        words = sentence.strip().split()
        current_words: list[str] = []
        for word in words:
            candidate = " ".join([*current_words, word])
            if current_words and len(candidate) > max_chars:
                units.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            units.append(" ".join(current_words))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-file", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--conditionals-cache")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--pause-seconds", type=float, default=0.28)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--exaggeration", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.65)
    args = parser.parse_args()

    script = Path(args.script_file).read_text(encoding="utf-8")
    chunks = split_script(script, max_chars=args.max_chars)
    if not chunks:
        raise SystemExit("The script is empty.")

    device = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    started = time.monotonic()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    make_hifigan_mps_safe(model)
    conditionals_cache = Path(args.conditionals_cache) if args.conditionals_cache else None
    if conditionals_cache and conditionals_cache.exists():
        model.conds = Conditionals.load(conditionals_cache, map_location="cpu").to(device)
    else:
        model.prepare_conditionals(
            args.reference_audio,
            exaggeration=args.exaggeration,
        )
        if conditionals_cache:
            conditionals_cache.parent.mkdir(parents=True, exist_ok=True)
            model.conds.save(conditionals_cache)

    generated = []
    for index, chunk in enumerate(chunks):
        audio = generate_fast(
            model,
            chunk,
            args.language,
            temperature=args.temperature,
        )
        generated.append(audio.cpu())
        if index < len(chunks) - 1:
            if re.search(r"[?!]$", chunk):
                pause_seconds = args.pause_seconds * 1.25
            elif re.search(r"[,;:]$", chunk):
                pause_seconds = args.pause_seconds * 0.65
            else:
                pause_seconds = args.pause_seconds
            generated.append(
                torch.zeros((1, int(model.sr * max(0.08, pause_seconds))))
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), torch.cat(generated, dim=-1), model.sr)
    print(f"{output}\t{time.monotonic() - started:.2f}")


if __name__ == "__main__":
    main()

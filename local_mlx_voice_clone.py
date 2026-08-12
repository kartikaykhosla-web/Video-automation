import argparse
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import write as write_audio
from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals, T3Cond
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


MODEL_ID = "mlx-community/chatterbox-fp16"
SAMPLE_RATE = 24_000


def split_script(text: str, max_chars: int = 230) -> list[str]:
    """Keep synthesis units bounded even when the input has no punctuation."""
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
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--conditionals-cache")
    parser.add_argument("--pause-seconds", type=float, default=0.28)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--exaggeration", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--cfg-weight", type=float, default=0.50)
    args = parser.parse_args()

    script = Path(args.script_file).read_text(encoding="utf-8")
    chunks = split_script(script, max_chars=args.max_chars)
    if not chunks:
        raise SystemExit("The script is empty.")

    started = time.monotonic()
    model = load_model(args.model_id)

    # Chatterbox only needs ten seconds for decoder conditioning. Supplying a
    # short, normalized clip also prevents long producer samples from making
    # the voice encoder unnecessarily expensive.
    cache_path = Path(args.conditionals_cache) if args.conditionals_cache else None
    if cache_path and cache_path.exists():
        cached = mx.load(str(cache_path))
        conditionals = Conditionals(
            T3Cond(
                speaker_emb=cached["t3.speaker_emb"],
                cond_prompt_speech_tokens=cached["t3.cond_prompt_speech_tokens"],
                emotion_adv=cached["t3.emotion_adv"],
            ),
            {
                key.removeprefix("gen."): value
                for key, value in cached.items()
                if key.startswith("gen.")
            },
        )
        mx.eval(
            conditionals.t3.speaker_emb,
            conditionals.t3.cond_prompt_speech_tokens,
            conditionals.t3.emotion_adv,
            *conditionals.gen.values(),
        )
    else:
        reference = load_audio(
            args.reference_audio,
            sample_rate=SAMPLE_RATE,
            length=SAMPLE_RATE * 12,
            volume_normalize=True,
        )
        conditionals = model.prepare_conditionals(
            reference,
            SAMPLE_RATE,
            exaggeration=args.exaggeration,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached = {
                "t3.speaker_emb": conditionals.t3.speaker_emb,
                "t3.cond_prompt_speech_tokens": conditionals.t3.cond_prompt_speech_tokens,
                "t3.emotion_adv": conditionals.t3.emotion_adv,
                **{f"gen.{key}": value for key, value in conditionals.gen.items()},
            }
            mx.save_safetensors(str(cache_path), cached)

    generated: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        mx.random.seed(20_260_721 + index)
        max_speech_tokens = max(220, min(900, int(len(chunk) * 2.4)))
        result = next(
            model.generate(
                text=chunk,
                conds=conditionals,
                lang_code=args.language.lower(),
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
                repetition_penalty=1.2,
                top_p=0.9,
                max_new_tokens=max_speech_tokens,
                verbose=False,
            )
        )
        mx.eval(result.audio)
        generated.append(np.asarray(result.audio.tolist(), dtype=np.float32))
        if index < len(chunks) - 1:
            if re.search(r"[?!]$", chunk):
                pause_seconds = args.pause_seconds * 1.25
            elif re.search(r"[,;:]$", chunk):
                pause_seconds = args.pause_seconds * 0.65
            else:
                pause_seconds = args.pause_seconds
            generated.append(
                np.zeros(int(SAMPLE_RATE * max(0.08, pause_seconds)), dtype=np.float32)
            )
        mx.clear_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_audio(output, np.concatenate(generated), SAMPLE_RATE, format="wav")
    print(f"{output}\t{time.monotonic() - started:.2f}")


if __name__ == "__main__":
    main()

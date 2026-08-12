import argparse
import json
from functools import lru_cache
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


MODEL_REPO = "sulabhkatiyar/indicconformer-120m-onnx"


@lru_cache(maxsize=1)
def load_runtime():
    model_path = hf_hub_download(MODEL_REPO, "hi/model.onnx")
    vocab_path = hf_hub_download(MODEL_REPO, "hi/vocab.json")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return session, vocab, len(vocab)


def transcribe_audio(audio_path: str) -> dict:
    started = time.monotonic()
    session, vocab, blank_id = load_runtime()

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio_pe = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]])
    mel = librosa.feature.melspectrogram(
        y=audio_pe,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=8000,
        norm="slaney",
        power=2.0,
    )
    log_mel = np.log(mel + 2**-24).astype(np.float32)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, ddof=1, keepdims=True) + 1e-5
    log_mel = (log_mel - mean) / std
    mel_batch = log_mel[np.newaxis, :, :].astype(np.float32)
    mel_length = np.array([mel_batch.shape[2]], dtype=np.int64)

    logits = session.run(
        None,
        {"audio_signal": mel_batch, "length": mel_length},
    )[0]
    ids = np.argmax(logits[0], axis=-1)
    previous = -1
    tokens = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id != previous and token_id != blank_id and token_id < len(vocab):
            tokens.append(vocab[token_id])
        previous = token_id
    transcript = "".join(tokens).replace("▁", " ").strip()
    if not transcript:
        raise RuntimeError("IndicConformer returned an empty transcript.")

    duration = len(audio) / 16000.0
    return {
        "language": "hi",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "segments": [{"start": 0.0, "end": duration, "text": transcript}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = transcribe_audio(args.audio)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

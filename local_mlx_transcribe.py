import argparse
import json
from pathlib import Path


MODEL_ID = "mlx-community/whisper-turbo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--language", default="")
    args = parser.parse_args()

    import mlx_whisper

    language = args.language or None
    result = mlx_whisper.transcribe(
        args.audio,
        path_or_hf_repo=MODEL_ID,
        language=language,
        task="transcribe",
        temperature=(0.0, 0.2, 0.4, 0.6),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.2,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        verbose=None,
    )

    payload = {
        "language": result.get("language") or language or "",
        "segments": [
            {
                "start": float(segment.get("start") or 0),
                "end": float(segment.get("end") or segment.get("start") or 0),
                "text": str(segment.get("text") or "").strip(),
            }
            for segment in result.get("segments") or []
            if str(segment.get("text") or "").strip()
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

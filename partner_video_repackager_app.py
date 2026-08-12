import base64
import json
import hashlib
from io import BytesIO
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import wave
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account


APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "partner_video_repackager_work"
UPLOAD_DIR = WORK_DIR / "uploads"
AUDIO_DIR = WORK_DIR / "audio"
TRANSCRIPT_DIR = WORK_DIR / "transcripts"
EXPORT_DIR = WORK_DIR / "exports"
PREVIEW_DIR = WORK_DIR / "previews"
SCRIPT_DIR = WORK_DIR / "scripts"
REFERENCE_DIR = WORK_DIR / "reference"
OVERLAY_DIR = WORK_DIR / "overlays"
OVERLAY_EDITOR_DIR = APP_DIR / "partner_overlay_editor"
PROPERTY_LOGO_DIR = APP_DIR / "partner_property_logos"
VERTEX_SERVICE_ACCOUNT_CANDIDATES = (
    APP_DIR.parent / "service_account_vertex.json",
    APP_DIR / "service_account_vertex.json",
)
VERTEX_LOCATION = "us-central1"
VERTEX_GEMINI_MODEL = "gemini-2.5-flash"
VOICE_CACHE_DIR = AUDIO_DIR / "voice_cache"
REFERENCE_VIDEO = REFERENCE_DIR / "RPcXnvzkH3I.mp4"
REFERENCE_CONTACT_SHEET = REFERENCE_DIR / "contact_sheet.jpg"
PRIYA_VOICE_REFERENCE = REFERENCE_DIR / "priya_voice_reference.mp3"
DIAHA_VOICE_REFERENCE = REFERENCE_DIR / "diaha_voice_reference.mp3"
DEEPIKA_VOICE_REFERENCE = REFERENCE_DIR / "deepika_newsroom_reference.wav"
DEEPIKA_HUMAN_TAKE = REFERENCE_DIR / "deepika_human_take_same_script.wav"
PRIYA_HUMAN_TAKE = REFERENCE_DIR / "priya_human_take_same_script.wav"
DISHA_HUMAN_TAKE = REFERENCE_DIR / "disha_human_take_same_script.wav"
CANONICAL_HUMAN_SCRIPT = REFERENCE_DIR / "deepika_human_take_script.txt"
DEEPIKA_F5_LABEL = "Deepika – Fine-tuned F5 Pilot"
DEEPIKA_F5_HELPER = APP_DIR / "local_f5_voice.py"
DEEPIKA_F5_CHECKPOINT = (
    WORK_DIR
    / "deepika_training/checkpoints/deepika_pilot_epoch1.safetensors"
)
DEEPIKA_F5_VOCAB = WORK_DIR / "deepika_training/base_model/vocab.txt"
DEEPIKA_F5_SOURCE = Path("/Users/kartikaykhosla/Downloads/Deepika VO (1).mp3")
DEEPIKA_F5_REFERENCE_START = 414.1
DEEPIKA_F5_REFERENCE_END = 417.8
DEEPIKA_F5_REFERENCE_TEXT = (
    "उन्होंने हर एक युवा खिलाड़ी का हौसला बढ़ाया है।"
)
DEEPIKA_F5_SERIOUS_START = 1119.0
DEEPIKA_F5_SERIOUS_END = 1121.0
DEEPIKA_F5_SERIOUS_TEXT = (
    "पेपर लीक कोई मामूली बात नहीं है।"
)
DEEPIKA_F5_QUESTION_START = 1312.5
DEEPIKA_F5_QUESTION_END = 1314.45
DEEPIKA_F5_QUESTION_TEXT = (
    "क्या किसी ने हैक कर लिया था?"
)
ELEVENLABS_VOICE_LABEL = "ElevenLabs pre-trained voice"
ELEVENLABS_VOICES = {
    "ElevenLabs Hindi Voice 1": "d0grukerEzs069eKIauC",
    "ElevenLabs Hindi Voice 2": "vzov6y10x6nsGNFg883S",
}
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
PROPERTY_LOGO_FILES = {
    "Jagran": "jagran.png",
    "Jagran Josh": "jagranjosh.png",
    "The Daily Jagran": "thedailyjagran.png",
    "OnlyMyHealth": "onlymyhealth.png",
    "HerZindagi": "herzindagi.png",
}
TRANSCRIPTION_PIPELINE_VERSION = "indicconformer-hi-ctc-v1"

PRODUCER_VOICE_PROFILES: Dict[str, Dict[str, object]] = {
    "Priya": {
        "reference": PRIYA_HUMAN_TAKE,
        "human_take": PRIYA_HUMAN_TAKE,
        "words_per_minute": 173,
        "pause_seconds": 0.28,
        "max_chars": 260,
        "exaggeration": 0.50,
        "temperature": 0.75,
        "cfg_weight": 0.50,
        # Chatterbox renders this reference about 0.7 semitone above Priya's
        # measured speaking register. Correct the clone without changing pace.
        "pitch_semitones": -0.70,
    },
    "Disha": {
        "reference": DISHA_HUMAN_TAKE,
        "human_take": DISHA_HUMAN_TAKE,
        "words_per_minute": 157,
        "pause_seconds": 0.32,
        "max_chars": 225,
        "exaggeration": 0.36,
        "temperature": 0.64,
        "cfg_weight": 0.52,
        "pitch_semitones": 0.0,
    },
    "Deepika": {
        "reference": DEEPIKA_VOICE_REFERENCE,
        "human_take": DEEPIKA_HUMAN_TAKE,
        "words_per_minute": 162,
        "pause_seconds": 0.28,
        "max_chars": 240,
        "exaggeration": 0.35,
        "temperature": 0.65,
        "cfg_weight": 0.50,
        "pitch_semitones": 0.0,
    },
}

BUILTIN_PRODUCER_VOICES = {
    name: Path(str(profile["reference"]))
    for name, profile in PRODUCER_VOICE_PROFILES.items()
}

DEFAULT_DELIVERY_PROFILE: Dict[str, object] = {
    "words_per_minute": 164,
    "pause_seconds": 0.28,
    "max_chars": 240,
    "exaggeration": 0.35,
    "temperature": 0.65,
    "cfg_weight": 0.50,
    "pitch_semitones": 0.0,
}

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

SLUG_STYLE_PRESETS: Dict[str, Dict[str, str]] = {
    "Jagran Red": {
        "background": "#79070D",
        "background_end": "#B3131C",
        "accent": "#F4C542",
        "text": "#FFFFFF",
        "highlight_text": "#17120A",
        "label_text": "#17120A",
    },
    "Breaking News": {
        "background": "#C9151E",
        "background_end": "#780006",
        "accent": "#FFFFFF",
        "text": "#FFFFFF",
        "highlight_text": "#A00810",
        "label_text": "#A00810",
    },
    "Clean Light": {
        "background": "#F2EFE8",
        "background_end": "#FFFFFF",
        "accent": "#A80B16",
        "text": "#17191D",
        "highlight_text": "#FFFFFF",
        "label_text": "#FFFFFF",
    },
    "Midnight Blue": {
        "background": "#111824",
        "background_end": "#27364B",
        "accent": "#4AC7FF",
        "text": "#FFFFFF",
        "highlight_text": "#081019",
        "label_text": "#081019",
    },
    "Custom": {
        "background": "#79070D",
        "background_end": "#A80D16",
        "accent": "#F4C542",
        "text": "#FFFFFF",
        "highlight_text": "#17120A",
        "label_text": "#17120A",
    },
}

overlay_layout_editor = components.declare_component(
    "partner_overlay_timeline_editor",
    path=str(OVERLAY_EDITOR_DIR),
)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def ensure_dirs() -> None:
    for directory in [UPLOAD_DIR, AUDIO_DIR, VOICE_CACHE_DIR, TRANSCRIPT_DIR, EXPORT_DIR, PREVIEW_DIR, SCRIPT_DIR, REFERENCE_DIR, OVERLAY_DIR, PROPERTY_LOGO_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def tool_path(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in [Path("/opt/homebrew/bin"), Path("/usr/local/bin")]:
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return None


def run_command(
    args: List[str],
    timeout_seconds: Optional[float] = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args, 124, "", f"Command timed out after {timeout_seconds:g} seconds."
        )


def media_tools_healthy() -> bool:
    """Check that the media tools are executable without launching subprocesses.

    FFmpeg startup can briefly exceed the old two-second probe while macOS is
    under load. Caching that timeout incorrectly disabled export for the rest
    of the Streamlit session even though both binaries were installed.
    """
    for name in ("ffmpeg", "ffprobe"):
        executable = tool_path(name)
        if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
            return False
    return True


def safe_name(value: str, fallback: str = "file") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or fallback


def save_upload(uploaded_file) -> Path:
    ensure_dirs()
    target = UPLOAD_DIR / safe_name(uploaded_file.name, "partner_video.mp4")
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while target.exists():
        target = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    target.write_bytes(uploaded_file.getbuffer())
    return target


def newsroom_video_api_config(provider: str) -> Dict[str, str]:
    """Return account-specific Reuters/ANI API settings without exposing keys."""
    prefix = provider.upper()
    config = {
        "base_url": os.environ.get(f"{prefix}_VIDEO_API_BASE_URL", "").strip(),
        "api_key": os.environ.get(f"{prefix}_VIDEO_API_KEY", "").strip(),
        "search_path": os.environ.get(f"{prefix}_VIDEO_SEARCH_PATH", "/videos").strip(),
        "auth_header": os.environ.get(f"{prefix}_VIDEO_AUTH_HEADER", "Authorization").strip(),
        "auth_scheme": os.environ.get(f"{prefix}_VIDEO_AUTH_SCHEME", "Bearer").strip(),
    }
    try:
        secret_names = {
            "base_url": f"{prefix}_VIDEO_API_BASE_URL",
            "api_key": f"{prefix}_VIDEO_API_KEY",
            "search_path": f"{prefix}_VIDEO_SEARCH_PATH",
            "auth_header": f"{prefix}_VIDEO_AUTH_HEADER",
            "auth_scheme": f"{prefix}_VIDEO_AUTH_SCHEME",
        }
        for key, secret_name in secret_names.items():
            if not config[key] and secret_name in st.secrets:
                config[key] = str(st.secrets[secret_name]).strip()
    except Exception:
        pass
    return config


def _first_text(mapping: Dict[str, object], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def normalise_newsroom_video_items(payload: object, provider: str) -> List[Dict[str, str]]:
    """Normalise common licensed-feed response shapes into picker cards."""
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = next(
            (
                value
                for key in ("items", "results", "videos", "content", "data")
                if isinstance((value := payload.get(key)), list)
            ),
            [],
        )
    else:
        raw_items = []
    items: List[Dict[str, str]] = []
    for position, raw_item in enumerate(raw_items[:50]):
        if not isinstance(raw_item, dict):
            continue
        assets = raw_item.get("assets") if isinstance(raw_item.get("assets"), dict) else {}
        video_url = _first_text(
            raw_item,
            ("download_url", "downloadUrl", "video_url", "videoUrl", "url", "href"),
        ) or _first_text(assets, ("video", "download", "original", "mp4"))
        preview_url = _first_text(
            raw_item,
            ("preview_url", "previewUrl", "thumbnail_url", "thumbnailUrl", "thumbnail"),
        ) or _first_text(assets, ("preview", "thumbnail", "poster"))
        identifier = _first_text(raw_item, ("id", "uuid", "usn", "guid")) or str(position)
        items.append(
            {
                "id": identifier,
                "provider": provider,
                "title": _first_text(raw_item, ("title", "headline", "name", "slug")) or f"{provider} video {position + 1}",
                "description": _first_text(raw_item, ("description", "summary", "caption", "shotlist")),
                "duration": _first_text(raw_item, ("duration", "length", "runtime")),
                "preview_url": preview_url,
                "video_url": video_url,
            }
        )
    return items


def search_newsroom_videos(provider: str, query: str) -> Tuple[List[Dict[str, str]], str]:
    config = newsroom_video_api_config(provider)
    if not config["base_url"] or not config["api_key"]:
        return [], f"{provider} API access is not configured on this server."
    endpoint = f"{config['base_url'].rstrip('/')}/{config['search_path'].lstrip('/')}"
    auth_value = f"{config['auth_scheme']} {config['api_key']}".strip()
    headers = {config["auth_header"]: auth_value, "Accept": "application/json"}
    try:
        response = requests.get(
            endpoint,
            params={"q": query, "query": query, "type": "video", "limit": 24},
            headers=headers,
            timeout=25,
        )
        response.raise_for_status()
        items = normalise_newsroom_video_items(response.json(), provider)
        return items, f"Found {len(items)} entitled {provider} video(s)."
    except Exception as exc:
        return [], f"{provider} search failed: {exc}"


def download_newsroom_video(item: Dict[str, str]) -> Tuple[Optional[Path], str]:
    provider = str(item.get("provider") or "Newsroom")
    video_url = str(item.get("video_url") or "").strip()
    parsed = urlparse(video_url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None, "The selected feed item does not include a valid HTTPS download URL."
    config = newsroom_video_api_config(provider)
    auth_value = f"{config['auth_scheme']} {config['api_key']}".strip()
    headers = {config["auth_header"]: auth_value} if config.get("api_key") else {}
    target = UPLOAD_DIR / f"{safe_name(provider.lower())}_{safe_name(str(item.get('id') or 'video'))}.mp4"
    try:
        with requests.get(video_url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "json" in content_type or "text/html" in content_type:
                return None, "The provider returned metadata instead of a downloadable video asset."
            with target.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if target.stat().st_size < 1024:
            target.unlink(missing_ok=True)
            return None, "The downloaded provider video was empty."
        return target, f"Imported {item.get('title') or target.name} from {provider}."
    except Exception as exc:
        target.unlink(missing_ok=True)
        return None, f"Could not import the {provider} video: {exc}"


def save_voiceover_upload(uploaded_file) -> Path:
    ensure_dirs()
    target = AUDIO_DIR / safe_name(uploaded_file.name, "approved_voiceover.wav")
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while target.exists():
        target = AUDIO_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    target.write_bytes(uploaded_file.getbuffer())
    return target


def save_pause_audio_upload(uploaded_file, pause_id: str) -> Path:
    """Save one pause-window audio insert at a stable local path."""
    ensure_dirs()
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    target = AUDIO_DIR / f"pause_insert_{safe_name(pause_id, 'pause')}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def save_reference_voice_upload(uploaded_file) -> Path:
    ensure_dirs()
    target = AUDIO_DIR / safe_name(uploaded_file.name, "producer_voice_reference.wav")
    target.write_bytes(uploaded_file.getbuffer())
    return target


def save_overlay_upload(uploaded_file, overlay_id: int) -> Path:
    """Save one uploaded image or video overlay and return a stable local path."""
    ensure_dirs()
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    media_type = (
        "video"
        if suffix in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
        else "image"
    )
    target = OVERLAY_DIR / f"{media_type}_{overlay_id}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


@st.cache_data(show_spinner=False)
def image_preview_data_url(path_value: str, modified_ns: int) -> str:
    del modified_ns
    from PIL import Image, ImageOps

    with Image.open(path_value) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA")
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def property_logo_path(property_name: str) -> Optional[Path]:
    filename = PROPERTY_LOGO_FILES.get(property_name)
    if not filename:
        return None
    candidate = PROPERTY_LOGO_DIR / filename
    if candidate.is_file():
        return candidate
    legacy_jagran_logo = APP_DIR / "shorts_automation_work/templates/jagran-shorts-logo.png"
    if property_name == "Jagran" and legacy_jagran_logo.is_file():
        return legacy_jagran_logo
    return None


@st.cache_data(show_spinner=False)
def video_preview_data_url(path_value: str, modified_ns: int) -> str:
    del modified_ns
    try:
        import av
        from PIL import Image, ImageOps

        with av.open(path_value) as container:
            frame = next(container.decode(video=0))
            still = frame.to_image().convert("RGB")
        canvas = Image.new("RGB", (960, 540), "black")
        fitted = ImageOps.contain(still, canvas.size, Image.Resampling.LANCZOS)
        canvas.paste(fitted, ((960 - fitted.width) // 2, (540 - fitted.height) // 2))
        buffer = BytesIO()
        canvas.save(buffer, format="JPEG", quality=72, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def media_has_audio(path_value: str, modified_ns: int) -> bool:
    del modified_ns
    try:
        import av

        with av.open(path_value) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False


def seconds_to_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def compact_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def estimate_voiceover_duration(
    script: str,
    speed: float = 1.0,
    language_code: Optional[str] = None,
) -> Tuple[float, int]:
    """Estimate spoken duration from script length, punctuation, and speed."""
    clean_script = re.sub(r"\s+", " ", script).strip()
    words = re.findall(r"[\w\u0900-\u097f]+", clean_script, flags=re.UNICODE)
    word_count = len(words)
    if not word_count:
        return 0.0, 0
    is_hindi = language_code == "hi" or bool(re.search(r"[\u0900-\u097f]", clean_script))
    natural_wpm = 145.0 if is_hindi else 160.0
    spoken_seconds = word_count / natural_wpm * 60.0
    sentence_pauses = len(re.findall(r"[.!?।]+", clean_script)) * 0.34
    phrase_pauses = len(re.findall(r"[,;:—–]+", clean_script)) * 0.16
    estimated_seconds = (spoken_seconds + sentence_pauses + phrase_pauses) / max(0.1, speed)
    return max(1.0, estimated_seconds), word_count


def vertex_service_account_path() -> Optional[Path]:
    return next((path for path in VERTEX_SERVICE_ACCOUNT_CANDIDATES if path.is_file()), None)


def visual_context_parts(
    source_path: Path,
    header_path: Optional[Path],
    floating_items: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build compact image parts representing visuals used in the composition."""
    candidates: List[Tuple[Path, str]] = [(source_path, "video")]
    if header_path and header_path.is_file():
        candidates.append((header_path, "image"))
    for item in floating_items[:6]:
        path = Path(str(item.get("path") or ""))
        if path.is_file():
            candidates.append((path, str(item.get("media_type") or "image")))
    parts: List[Dict[str, object]] = []
    for path, media_type in candidates[:8]:
        try:
            if media_type == "video":
                data_url = video_preview_data_url(str(path), path.stat().st_mtime_ns)
            else:
                data_url = image_preview_data_url(str(path), path.stat().st_mtime_ns)
            if not data_url or "," not in data_url:
                continue
            header, encoded = data_url.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            parts.append({"inlineData": {"mimeType": mime_type, "data": encoded}})
        except Exception:
            continue
    return parts


def parse_gemini_metadata(text: str) -> Dict[str, object]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover complete fields from a response truncated near the end of the
        # meta-tag array. JSON-decode each captured string so escapes remain safe.
        def captured_string(field: str) -> str:
            match = re.search(
                rf'"{re.escape(field)}"\s*:\s*("(?:\\.|[^"\\])*")',
                cleaned,
                flags=re.S,
            )
            return str(json.loads(match.group(1))) if match else ""

        title_value = captured_string("title")
        headline_value = captured_string("headline")
        description_value = captured_string("description")
        tags_match = re.search(r'"(?:meta_tags|tags)"\s*:\s*\[([\s\S]*)', cleaned)
        recovered_tags: List[str] = []
        if tags_match:
            for encoded_tag in re.findall(r'"(?:\\.|[^"\\])*"', tags_match.group(1)):
                try:
                    recovered_tags.append(str(json.loads(encoded_tag)))
                except Exception:
                    continue
        payload = {
            "title": title_value,
            "headline": headline_value,
            "description": description_value,
            "meta_tags": recovered_tags,
        }
    title = re.sub(r"\s+", " ", str(payload.get("title") or "")).strip()
    headline = re.sub(r"\s+", " ", str(payload.get("headline") or "")).strip()
    description = str(payload.get("description") or "").strip()
    raw_tags = payload.get("meta_tags") or payload.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = re.split(r"[,\n]", raw_tags)
    tags: List[str] = []
    for value in raw_tags if isinstance(raw_tags, list) else []:
        tag = re.sub(r"^[#\s]+|\s+$", "", str(value))
        if tag and tag.casefold() not in {existing.casefold() for existing in tags}:
            tags.append(tag)
        if len(tags) == 20:
            break
    if not title or not headline or not description:
        raise ValueError("Gemini did not return a title, headline and description.")
    return {
        "title": title,
        "headline": headline,
        "description": description,
        "meta_tags": tags,
    }


def generate_video_metadata_with_gemini(
    transcript: str,
    output_language: str,
    source_path: Path,
    header_path: Optional[Path],
    floating_items: List[Dict[str, object]],
) -> Tuple[Optional[Dict[str, object]], str]:
    credential_path = vertex_service_account_path()
    if not credential_path:
        return None, "Vertex service-account file was not found."
    try:
        account_info = json.loads(credential_path.read_text(encoding="utf-8"))
        project_id = str(account_info.get("project_id") or "").strip()
        credentials = service_account.Credentials.from_service_account_info(
            account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        credentials.refresh(GoogleAuthRequest())
        if not project_id:
            return None, "The Vertex service account does not contain a project_id."
        prompt = f"""
You are a newsroom SEO editor. Analyse the transcript and the attached representative
frames from the raw video and floating visuals. Return only valid JSON with this schema:
{{"title":"...","headline":"...","description":"...","meta_tags":["tag 1","tag 2"]}}

Requirements:
- Output language mode: {output_language}.
- If the mode is Hindi + English, format title, headline and description as
  "Hindi: ...\nEnglish: ..." and include both Hindi and English meta tags.
- Title: accurate, compelling, 55-75 characters when practical.
- Headline: factual newsroom headline, distinct from title, no clickbait.
- Description: a factual publishing summary of 80-160 words.
- meta_tags: 10-20 highly relevant search phrases, maximum 20, no # symbols.
- Use only facts supported by the transcript or visible frames.
- Do not invent identities, locations, dates, allegations, or outcomes.

TRANSCRIPT:
{transcript[:18000]}
""".strip()
        parts: List[Dict[str, object]] = [{"text": prompt}]
        parts.extend(visual_context_parts(source_path, header_path, floating_items))
        endpoint = (
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/{VERTEX_LOCATION}/publishers/google/models/"
            f"{VERTEX_GEMINI_MODEL}:generateContent"
        )
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "required": ["title", "headline", "description", "meta_tags"],
                        "properties": {
                            "title": {"type": "STRING"},
                            "headline": {"type": "STRING"},
                            "description": {"type": "STRING"},
                            "meta_tags": {
                                "type": "ARRAY",
                                "minItems": 10,
                                "maxItems": 20,
                                "items": {"type": "STRING"},
                            },
                        },
                    },
                },
            },
            timeout=90,
        )
        response.raise_for_status()
        response_payload = response.json()
        response_text = "".join(
            str(part.get("text") or "")
            for part in response_payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        return parse_gemini_metadata(response_text), "Generated from transcript and selected visual frames."
    except Exception as exc:
        return None, f"Gemini metadata generation failed: {exc}"


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalise_cut_ranges(
    ranges: List[Dict[str, object]],
    source_duration: float,
) -> List[Tuple[float, float]]:
    """Clamp, sort and merge raw-video ranges selected for removal."""
    cleaned: List[Tuple[float, float]] = []
    for item in ranges:
        start = clamp_float(float(item.get("start") or 0.0), 0.0, source_duration)
        end = clamp_float(float(item.get("end") or 0.0), 0.0, source_duration)
        if end - start >= 0.05:
            cleaned.append((start, end))
    cleaned.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def kept_source_ranges(
    cuts: List[Tuple[float, float]],
    source_duration: float,
) -> List[Tuple[float, float]]:
    """Return source intervals that remain after applying merged cuts."""
    kept: List[Tuple[float, float]] = []
    cursor = 0.0
    for start, end in cuts:
        if start > cursor + 0.01:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < source_duration - 0.01:
        kept.append((cursor, source_duration))
    return kept


def normalise_voice_pauses(
    pauses: List[Dict[str, object]],
    voiceover_start: float,
) -> List[Tuple[float, float]]:
    """Sort and merge output-timeline intervals where narration should pause."""
    cleaned: List[Tuple[float, float]] = []
    for item in pauses:
        start = max(voiceover_start, float(item.get("start") or voiceover_start))
        duration = max(0.1, float(item.get("duration") or 0.1))
        cleaned.append((start, start + duration))
    cleaned.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(start, end - start) for start, end in merged]


def default_grid_layout(item_ids: List[int], columns: int = 2) -> List[Dict[str, object]]:
    if not item_ids:
        return []
    columns = max(1, min(int(columns), 4))
    rows = (len(item_ids) + columns - 1) // columns
    gap = 0.018
    width = (1.0 - gap * (columns + 1)) / columns
    height = (1.0 - gap * (rows + 1)) / rows
    return [
        {
            "id": str(item_id),
            "x": gap + (index % columns) * (width + gap),
            "y": gap + (index // columns) * (height + gap),
            "w": width,
            "h": height,
            "z": index + 1,
        }
        for index, item_id in enumerate(item_ids)
    ]


def probe_video(path: Path) -> Dict[str, float]:
    if not media_tools_healthy():
        return probe_video_native(path)
    ffprobe = tool_path("ffprobe")
    if not ffprobe:
        return probe_video_native(path)
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout_seconds=2,
    )
    if result.returncode != 0:
        return probe_video_native(path)
    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
        duration = float((payload.get("format") or {}).get("duration") or 0)
        fps_raw = str(stream.get("r_frame_rate") or "0/1")
        numerator, denominator = fps_raw.split("/")
        fps = float(numerator) / max(float(denominator), 1.0)
        return {
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "duration": duration,
            "fps": fps,
        }
    except Exception:
        return probe_video_native(path)


@st.cache_data(show_spinner=False)
def probe_video_native(path: Path) -> Dict[str, float]:
    """Read basic metadata with PyAV when the system FFprobe is unavailable."""
    try:
        import av

        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return {}
            duration = 0.0
            if container.duration:
                duration = float(container.duration) / 1_000_000
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            return {
                "width": int(stream.width or 0),
                "height": int(stream.height or 0),
                "duration": max(0.0, duration),
                "fps": fps,
            }
    except Exception:
        return {}


def probe_media_duration(path: Path) -> float:
    if not media_tools_healthy():
        return 0.0
    ffprobe = tool_path("ffprobe")
    if not ffprobe:
        return 0.0
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout_seconds=2,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def extract_audio(source: Path) -> Tuple[Optional[Path], str]:
    ensure_dirs()
    audio_path = AUDIO_DIR / f"{source.stem}_pyav_speech_v1.wav"
    if audio_path.exists() and audio_path.stat().st_size > 1024:
        return audio_path, "Using previously extracted audio."

    try:
        import av

        container = av.open(str(source))
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            container.close()
            return None, "The uploaded video does not contain an audio track."

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    for converted in resampler.resample(frame):
                        samples = converted.to_ndarray().reshape(-1)
                        output.writeframes(samples.tobytes())
            for converted in resampler.resample(None):
                samples = converted.to_ndarray().reshape(-1)
                output.writeframes(samples.tobytes())
        container.close()
    except Exception as exc:
        if audio_path.exists():
            audio_path.unlink()
        return None, f"Native audio extraction failed: {exc}"

    if not audio_path.exists() or audio_path.stat().st_size <= 1024:
        return None, "Native audio extraction produced an empty file."
    return audio_path, "Audio extracted with the native PyAV decoder."


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str):
    """Load each Whisper model once per Streamlit server process."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, min(8, os.cpu_count() or 4)),
    )


def transcribe_video(
    source: Path,
    model_name: str = "base",
    language_code: Optional[str] = None,
) -> Tuple[Optional[str], List[TranscriptSegment], str]:
    audio_path, audio_message = extract_audio(source)
    if not audio_path:
        return None, [], audio_message
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        return None, [], f"Install faster-whisper with `./.venv-shorts/bin/pip install -U faster-whisper`. Details: {exc}"

    try:
        model = load_whisper_model(model_name)
        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "task": "transcribe",
        }
        if language_code:
            transcribe_kwargs["language"] = language_code
        if language_code == "hi":
            transcribe_kwargs["initial_prompt"] = (
                "यह हिंदी समाचार का प्रतिलेख है। पूरा प्रतिलेख केवल देवनागरी लिपि में लिखें।"
            )

        segments: List[TranscriptSegment] = []
        blocks: List[str] = []
        whisper_segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
        detected_language = str(getattr(info, "language", "") or "")
        for segment in whisper_segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            start = float(segment.start or 0)
            end = float(segment.end or segment.start or 0)
            segments.append(TranscriptSegment(start=start, end=end, text=text))
            blocks.append(
                f"{len(segments)}\n{seconds_to_srt(start)} --> {seconds_to_srt(end)}\n{text}\n"
            )

        if not blocks:
            return None, [], "No transcript segments were returned."
        srt_text = "\n".join(blocks)
        transcript_path = TRANSCRIPT_DIR / f"{source.stem}_{model_name}.srt"
        transcript_path.write_text(srt_text, encoding="utf-8")
        language = f" ({detected_language})" if detected_language else ""
        return srt_text, segments, f"Transcript generated{language}: {transcript_path.name}"
    except Exception as exc:
        return None, [], str(exc)[-2200:] or "Transcription failed."


def transcribe_video_mlx(
    source: Path,
    language_code: Optional[str] = None,
) -> Tuple[Optional[str], List[TranscriptSegment], str]:
    audio_path, audio_message = extract_audio(source)
    if not audio_path:
        return None, [], audio_message

    ffmpeg = tool_path("ffmpeg")
    helper = APP_DIR / "local_mlx_transcribe.py"
    mlx_python = APP_DIR / ".venv-shorts" / "bin" / "python"
    result_path = TRANSCRIPT_DIR / f".{source.stem}_mlx_large_v3.json"
    if not ffmpeg or not helper.exists() or not mlx_python.exists():
        return None, [], "The Apple MLX transcription runtime is unavailable."

    command = [
        str(mlx_python), str(helper), "--audio", str(audio_path),
        "--output-json", str(result_path),
    ]
    if language_code:
        command.extend(["--language", language_code])
    child_environment = os.environ.copy()
    child_environment["PATH"] = os.pathsep.join(
        [str(Path(ffmpeg).parent), child_environment.get("PATH", "")]
    )
    process = subprocess.run(
        command, capture_output=True, text=True, env=child_environment
    )
    if process.returncode != 0:
        details = process.stderr[-2000:] or process.stdout[-1200:]
        return None, [], f"Apple MLX transcription failed: {details}"
    if not result_path.exists():
        return None, [], "Apple MLX transcription returned no result."

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        result_path.unlink(missing_ok=True)
    segments: List[TranscriptSegment] = []
    blocks: List[str] = []
    for raw_segment in result.get("segments") or []:
        text = str(raw_segment.get("text") or "").strip()
        if not text:
            continue
        start = float(raw_segment.get("start") or 0)
        end = float(raw_segment.get("end") or start)
        segments.append(TranscriptSegment(start=start, end=end, text=text))
        blocks.append(
            f"{len(segments)}\n{seconds_to_srt(start)} --> {seconds_to_srt(end)}\n{text}\n"
        )
    if not blocks:
        return None, [], "Apple MLX returned no transcript segments."
    srt_text = "\n".join(blocks)
    transcript_path = TRANSCRIPT_DIR / f"{source.stem}_mlx_large_v3.srt"
    transcript_path.write_text(srt_text, encoding="utf-8")
    detected_language = str(result.get("language") or language_code or "")
    return srt_text, segments, f"Transcript generated ({detected_language}): {transcript_path.name}"


def normalize_hindi_news_transcript(text: str) -> str:
    """Fix deterministic CTC artifacts without asking a language model to invent text."""
    clean = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "साथसाथ": "साथ-साथ",
        "माननी्य": "माननीय",
        "मान्य न्यायालय": "माननीय न्यायालय",
        "सलाईनामा": "सुलहनामा",
    }
    for source, target in replacements.items():
        clean = clean.replace(source, target)
    clean = re.sub(
        r"दो\s+[^\s]+\s+बाईस\s+तेईस(?=\s+अगस्त)",
        "इक्कीस, बाईस, तेईस",
        clean,
    )
    sentence_starts = [
        "जिसमें ",
        "इसके साथ-साथ ",
        "तो इस कारण से ",
        "पच्चीस अगस्त को इस मामले में ",
        "कोर्ट का मानना है ",
        "जहां एक बार ",
        "यहां एक बार ",
        "तो इसको देखते हुए ",
        "अब इस मामले की ",
    ]
    for marker in sentence_starts:
        clean = clean.replace(f" {marker}", f"।\n{marker}")
    clean = re.sub(r"।(?:\s*।)+", "।", clean).strip()
    if clean and not clean.endswith(("।", ".", "?", "!")):
        clean += "।"
    return clean


def transcribe_video_indic(
    source: Path,
) -> Tuple[Optional[str], List[TranscriptSegment], str]:
    audio_path, audio_message = extract_audio(source)
    if not audio_path:
        return None, [], audio_message

    digest = hashlib.sha256()
    with source.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_path = TRANSCRIPT_DIR / f"indic_cache_{digest.hexdigest()[:20]}_v2.json"

    cache_hit = cache_path.exists()
    if cache_hit:
        result = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        try:
            from local_indic_transcribe import transcribe_audio

            result = transcribe_audio(str(audio_path))
        except Exception as exc:
            return None, [], f"Local Hindi transcription failed: {exc}"
        cache_path.write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )

    raw_text = " ".join(
        str(segment.get("text") or "").strip()
        for segment in result.get("segments") or []
        if str(segment.get("text") or "").strip()
    )
    transcript_text = normalize_hindi_news_transcript(raw_text)
    if not transcript_text:
        return None, [], "The local Hindi model returned an empty transcript."
    duration = float((result.get("segments") or [{}])[-1].get("end") or 0)
    segments = [TranscriptSegment(start=0.0, end=duration, text=transcript_text)]
    srt_text = (
        f"1\n{seconds_to_srt(0)} --> {seconds_to_srt(duration)}\n"
        f"{transcript_text}\n"
    )
    transcript_path = TRANSCRIPT_DIR / f"{source.stem}_indic_hi.srt"
    transcript_path.write_text(srt_text, encoding="utf-8")
    if cache_hit:
        return srt_text, segments, "Complete Hindi transcript loaded from cache."
    elapsed = float(result.get("elapsed_seconds") or 0)
    timing = f" in {elapsed:.1f} seconds" if elapsed else ""
    return srt_text, segments, f"Complete Hindi transcript generated locally{timing}."


def plain_transcript(segments: List[TranscriptSegment]) -> str:
    return "\n".join(segment.text for segment in segments if segment.text.strip())


def generate_anchor_script(transcript_text: str, partner: str, tone: str, target_seconds: int) -> str:
    clean = re.sub(r"\s+", " ", transcript_text).strip()
    if not clean:
        clean = (
            "This is a sample partner feed. The newsroom will verify the facts, add context, "
            "and produce a clean anchor-led video package."
        )
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    facts = [sentence.strip() for sentence in sentences if sentence.strip()]
    key_points = facts[:5] if facts else [clean[:220]]
    word_budget = max(80, int(target_seconds * 2.25))
    body = " ".join(key_points)
    body_words = body.split()[:word_budget]
    body = " ".join(body_words)
    partner_label = partner.strip() or "partner"
    opener = f"Here is the latest update based on video inputs from {partner_label}."
    if tone == "Explainer":
        opener = f"Here is what this development means, based on video inputs from {partner_label}."
    elif tone == "Breaking":
        opener = f"Breaking update. We are tracking visuals and details from {partner_label}."
    elif tone == "Bulletin":
        opener = f"Top update from the newsroom, based on footage shared by {partner_label}."
    closing = "We will continue to track this story and bring verified updates as they come in."
    return f"{opener}\n\n{body}\n\n{closing}"


def create_voiceover(script: str, voice_name: str = "Samantha") -> Tuple[Optional[Path], str]:
    ensure_dirs()
    output_path = AUDIO_DIR / "anchor_voiceover.aiff"
    script_path = SCRIPT_DIR / "anchor_script.txt"
    script_path.write_text(script, encoding="utf-8")
    if platform.system() != "Darwin" or not tool_path("say"):
        return None, "Local test voiceover currently uses macOS `say`. Add a cloud TTS provider in the next version."
    result = run_command(["say", "-v", voice_name, "-o", str(output_path), "-f", str(script_path)])
    if result.returncode != 0:
        return None, result.stderr[-1200:] or "Voiceover generation failed."
    return output_path, f"Voiceover generated: {output_path.name}"


def elevenlabs_api_key() -> str:
    """Read the ElevenLabs credential without exposing or persisting it."""
    environment_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        secrets_key = str(st.secrets.get("ELEVENLABS_API_KEY", "")).strip()
    except Exception:
        secrets_key = ""
    if secrets_key:
        return secrets_key
    return str(st.session_state.get("partner_elevenlabs_api_key", "")).strip()


def remember_elevenlabs_api_key() -> None:
    """Copy the password widget value to a non-widget session key."""
    entered_key = str(
        st.session_state.get("partner_elevenlabs_api_key_input", "")
    ).strip()
    if entered_key:
        st.session_state["partner_elevenlabs_api_key"] = entered_key


def create_elevenlabs_voiceover(
    script: str,
    voice_id: str,
    speed: float = 1.0,
) -> Tuple[Optional[Path], str]:
    """Generate one continuous Hindi reading and cache that exact approved audio."""
    ensure_dirs()
    clean_script = re.sub(r"\s+", " ", script).strip()
    if not clean_script:
        return None, "Enter a script before generating the ElevenLabs voiceover."
    speed = clamp_float(float(speed), 0.7, 1.2)
    if len(clean_script) > 9_500:
        return (
            None,
            "This script is longer than the safe single-generation limit. "
            "Shorten it below 9,500 characters to preserve one continuous delivery.",
        )

    api_key = elevenlabs_api_key()
    if not api_key:
        return (
            None,
            "ElevenLabs is selected, but ELEVENLABS_API_KEY is not configured "
            "in the server environment.",
        )

    cache_key = hashlib.sha256(
        (
            f"elevenlabs-v2:{ELEVENLABS_MODEL_ID}:{voice_id}:"
            f"speed={speed:.2f}:{clean_script}"
        ).encode("utf-8")
    ).hexdigest()
    output_path = VOICE_CACHE_DIR / f"elevenlabs_{cache_key}.mp3"
    if output_path.exists() and output_path.stat().st_size > 1_000:
        return output_path, "Using the approved ElevenLabs audio from cache."

    try:
        import requests

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            params={"output_format": "mp3_44100_128"},
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": clean_script,
                "model_id": ELEVENLABS_MODEL_ID,
                "voice_settings": {"speed": speed},
                "apply_text_normalization": "auto",
            },
            timeout=180,
        )
    except Exception as exc:
        return None, f"Could not reach ElevenLabs: {exc}"

    if not response.ok:
        detail = response.text.strip()
        try:
            payload = response.json()
            detail = str(payload.get("detail") or payload)
        except Exception:
            pass
        return None, f"ElevenLabs generation failed ({response.status_code}): {detail[:700]}"
    if len(response.content) < 1_000:
        return None, "ElevenLabs returned an incomplete audio response."

    output_path.write_bytes(response.content)
    return (
        output_path,
        f"Voiceover generated with ElevenLabs Multilingual v2 at {speed:.2f}× speed. "
        "This exact audio will be reused for video export.",
    )


def producer_delivery_profile(reference_audio: Path) -> Dict[str, object]:
    """Resolve per-producer cadence controls, with newsroom defaults for new voices."""
    resolved_reference = reference_audio.resolve()
    for producer_name, configured in PRODUCER_VOICE_PROFILES.items():
        profile_reference = Path(str(configured["reference"]))
        if profile_reference.exists() and profile_reference.resolve() == resolved_reference:
            return {**DEFAULT_DELIVERY_PROFILE, **configured, "name": producer_name}
    return {**DEFAULT_DELIVERY_PROFILE, "name": "uploaded producer"}


def normalized_script_identity(value: str) -> str:
    return re.sub(r"[\s।,.!?;:'\"()\-]+", "", value).lower()


def prepare_script_for_delivery(script: str, language_code: str) -> str:
    """Add conservative speech-only phrasing without changing spoken words."""
    clean = re.sub(r"\s+", " ", script).strip()
    if not clean or language_code.lower() != "hi":
        return clean

    connectives = {
        "लेकिन", "मगर", "क्योंकि", "इसलिए", "जबकि", "हालांकि",
        "अगर", "फिर", "इसके", "दूसरी", "वहीं", "दरअसल",
    }
    sentence_enders = {
        "है", "हैं", "था", "थी", "थे", "हुआ", "हुई", "हुए",
        "गया", "गई", "गए", "चाहिए", "सकता", "सकती", "सकते",
        "होगा", "होगी", "होंगे", "करेंगे", "करेंगी", "किया", "दिया",
    }
    question_openers = {"क्या", "क्यों", "कैसे", "कब", "कहाँ", "कौन"}

    # Give connective words a light pause while retaining the original text.
    words = clean.split()
    since_pause = 0
    for index, word in enumerate(words):
        bare = word.strip("।,.!?;:\"'()")
        if bare in connectives and index > 0 and since_pause >= 7:
            previous = words[index - 1]
            if not re.search(r"[।,.!?;:]$", previous):
                words[index - 1] = previous + ","
            since_pause = 0
        since_pause = 0 if re.search(r"[।,.!?;:]$", word) else since_pause + 1
    clean = " ".join(words)

    prepared: List[str] = []
    raw_sentences = re.findall(r".*?(?:[।.!?]+|$)", clean)
    for raw_sentence in raw_sentences:
        raw_sentence = raw_sentence.strip()
        if not raw_sentence:
            continue
        terminal_match = re.search(r"([।.!?]+)$", raw_sentence)
        terminal = terminal_match.group(1) if terminal_match else "।"
        body = re.sub(r"[।.!?]+$", "", raw_sentence).strip()
        remaining = body.split()
        while len(remaining) > 28:
            candidates = []
            for candidate in range(10, min(28, len(remaining)) + 1):
                bare = remaining[candidate - 1].strip("।,.!?;:\"'()")
                if bare in sentence_enders:
                    candidates.append(candidate)

            question_position = next(
                (
                    index
                    for index, word in enumerate(remaining[:28])
                    if word.strip("।,.!?;:\"'()") in question_openers
                ),
                None,
            )
            question_break = False
            if question_position is not None:
                question_candidates = [
                    value for value in candidates if value >= question_position + 4
                ]
                if question_candidates:
                    split_at = question_candidates[0]
                    question_break = True
                else:
                    before_question = [
                        value for value in candidates if value <= question_position
                    ]
                    split_at = before_question[-1] if before_question else None
            else:
                split_at = min(candidates, key=lambda value: abs(value - 22)) if candidates else None
            split_at = split_at or min(22, len(remaining))
            while (
                split_at < min(28, len(remaining))
                and remaining[split_at].strip("।,.!?;:\"'()")
                in {"है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे"}
            ):
                split_at += 1
            phrase = " ".join(remaining[:split_at]).rstrip("।,.!?;:")
            if question_break:
                phrase_terminal = "?"
            elif split_at in candidates:
                # This is a grammatical clause ending (for example, है/था/किया),
                # so give the voice model a real sentence boundary. A comma here
                # made long Hindi scripts sound like one word-by-word utterance.
                phrase_terminal = "।"
            else:
                # With no grammatical ending available, use only a light breath.
                phrase_terminal = ","
            prepared.append(phrase + phrase_terminal)
            remaining = remaining[split_at:]
        if remaining:
            phrase = " ".join(remaining).rstrip("।.!?")
            if any(
                word.strip("।,.!?;:\"'()") in question_openers
                for word in remaining
            ):
                terminal = "?"
            prepared.append(phrase + terminal)
    return " ".join(prepared)


def apply_producer_pace(
    audio_path: Path,
    script: str,
    delivery_profile: Dict[str, object],
) -> bool:
    """Gently align generated speech to a producer's observed newsroom pace."""
    ffmpeg = tool_path("ffmpeg")
    if not ffmpeg:
        return False
    word_count = len(re.findall(r"\S+", script))
    words_per_minute = float(delivery_profile.get("words_per_minute") or 164)
    current_duration = probe_media_duration(audio_path)
    if word_count < 5 or current_duration <= 0 or words_per_minute <= 0:
        return False
    target_duration = word_count * 60.0 / words_per_minute
    raw_tempo = current_duration / max(target_duration, 0.1)
    tempo = max(0.90, min(1.10, raw_tempo))
    if abs(tempo - 1.0) < 0.025:
        return False
    adjusted_path = audio_path.with_name(f"{audio_path.stem}_paced.wav")
    result = run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-af",
            f"atempo={tempo:.5f}",
            "-ac",
            "1",
            "-ar",
            "24000",
            str(adjusted_path),
        ],
        timeout_seconds=120,
    )
    if result.returncode != 0 or not adjusted_path.exists():
        if adjusted_path.exists():
            adjusted_path.unlink()
        return False
    adjusted_path.replace(audio_path)
    return True


def apply_producer_pitch(
    audio_path: Path,
    delivery_profile: Dict[str, object],
) -> bool:
    """Match a clone to the producer's measured register without changing duration."""
    ffmpeg = tool_path("ffmpeg")
    semitones = float(delivery_profile.get("pitch_semitones") or 0.0)
    if not ffmpeg or abs(semitones) < 0.05:
        return False

    # Changing the interpreted sample rate shifts pitch and duration together;
    # the inverse atempo restores the original duration. Chatterbox emits 24 kHz.
    pitch_factor = 2.0 ** (semitones / 12.0)
    tempo = 1.0 / pitch_factor
    adjusted_path = audio_path.with_name(f"{audio_path.stem}_pitch.wav")
    result = run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-af",
            (
                f"asetrate=24000*{pitch_factor:.8f},"
                f"aresample=24000,atempo={tempo:.8f}"
            ),
            "-ac",
            "1",
            "-ar",
            "24000",
            str(adjusted_path),
        ],
        timeout_seconds=120,
    )
    if result.returncode != 0 or not adjusted_path.exists():
        adjusted_path.unlink(missing_ok=True)
        return False
    adjusted_path.replace(audio_path)
    return True


def voiceover_duration_is_plausible(
    audio_path: Path,
    script: str,
    delivery_profile: Dict[str, object],
) -> bool:
    """Reject obviously truncated synthesis before it reaches video export."""
    word_count = len(re.findall(r"\S+", script))
    if word_count < 12:
        return True
    words_per_minute = float(delivery_profile.get("words_per_minute") or 164)
    expected_duration = word_count * 60.0 / max(words_per_minute, 1.0)
    actual_duration = probe_media_duration(audio_path)
    return actual_duration >= expected_duration * 0.55


def create_local_cloned_voiceover(
    script: str,
    reference_audio: Path,
    language_code: str = "hi",
    voice_model_mode: str = "fast",
) -> Tuple[Optional[Path], str]:
    mlx_python = APP_DIR / ".venv-mlx-voice" / "bin" / "python"
    mlx_helper = APP_DIR / "local_mlx_voice_clone.py"
    torch_python = APP_DIR / ".venv-voice" / "bin" / "python"
    torch_helper = APP_DIR / "local_voice_clone.py"
    mlx_available = mlx_python.exists() and mlx_helper.exists()
    torch_available = torch_python.exists() and torch_helper.exists()
    if not mlx_available and not torch_available:
        return None, "The local voice-cloning environment is not installed."
    if not reference_audio.exists():
        return None, "The producer reference audio is missing."

    clean_script = script.strip()
    delivery_script = prepare_script_for_delivery(clean_script, language_code)
    delivery_profile = producer_delivery_profile(reference_audio)
    human_take_value = delivery_profile.get("human_take")
    human_take = Path(str(human_take_value)) if human_take_value else None
    if human_take and human_take.exists() and CANONICAL_HUMAN_SCRIPT.exists():
        canonical_script = CANONICAL_HUMAN_SCRIPT.read_text(encoding="utf-8")
        match_ratio = SequenceMatcher(
            None,
            normalized_script_identity(clean_script),
            normalized_script_identity(canonical_script),
        ).ratio()
        if match_ratio >= 0.97:
            return (
                human_take,
                f"Using {delivery_profile['name']}’s verified human recording for this matching script.",
            )

    reference_stat = reference_audio.stat()
    use_fast_mlx = mlx_available and voice_model_mode == "fast"
    mlx_model_id = (
        "mlx-community/chatterbox-4bit"
        if use_fast_mlx
        else "mlx-community/chatterbox-fp16"
    )
    effective_max_chars = (
        max(360, int(delivery_profile["max_chars"]))
        if use_fast_mlx
        else int(delivery_profile["max_chars"])
    )
    effective_cfg_weight = (
        # Zero guidance was marginally faster in an early benchmark, but made
        # longer scripts sound word-by-word and weakened punctuation prosody.
        # 0.30 restores phrase delivery without a measurable warm-run penalty.
        0.30 if use_fast_mlx else float(delivery_profile["cfg_weight"])
    )
    if use_fast_mlx:
        engine_identity = "chatterbox-mlx-4bit-fast-natural-v3-fluency-guided"
    elif mlx_available:
        engine_identity = "chatterbox-mlx-fp16-quality-v4-natural-delivery"
    else:
        engine_identity = "chatterbox-pytorch-fast-v3-natural-delivery"
    delivery_identity = json.dumps(
        {
            key: delivery_profile[key]
            for key in (
                "words_per_minute",
                "pause_seconds",
                "exaggeration",
                "temperature",
                "cfg_weight",
                "pitch_semitones",
            )
        }
        | {
            "max_chars": effective_max_chars,
            "cfg_weight": effective_cfg_weight,
        },
        sort_keys=True,
    )
    reference_identity = (
        f"{reference_audio.resolve()}:{reference_stat.st_size}:"
        f"{reference_stat.st_mtime_ns}:{engine_identity}:{delivery_identity}"
    )
    voice_key = hashlib.sha256(reference_identity.encode("utf-8")).hexdigest()[:16]
    output_key = hashlib.sha256(
        f"{voice_key}:{language_code}:{delivery_script}".encode("utf-8")
    ).hexdigest()[:20]
    script_path = SCRIPT_DIR / f"voice_script_{output_key}.txt"
    output_path = VOICE_CACHE_DIR / f"voiceover_{output_key}.wav"
    conditionals_suffix = "safetensors" if mlx_available else "pt"
    conditionals_path = VOICE_CACHE_DIR / f"voice_conditionals_{voice_key}.{conditionals_suffix}"
    torch_conditionals_path = VOICE_CACHE_DIR / f"voice_conditionals_{voice_key}.pt"
    if output_path.exists() and output_path.stat().st_size > 1024:
        if voiceover_duration_is_plausible(output_path, clean_script, delivery_profile):
            return output_path, "Using the cached producer voiceover."
        output_path.unlink()
    script_path.write_text(delivery_script, encoding="utf-8")
    if mlx_available:
        command = [
            str(mlx_python),
            str(mlx_helper),
            "--script-file",
            str(script_path),
            "--reference-audio",
            str(reference_audio),
            "--output",
            str(output_path),
            "--language",
            language_code,
            "--model-id",
            mlx_model_id,
            "--conditionals-cache",
            str(conditionals_path),
            "--pause-seconds",
            str(delivery_profile["pause_seconds"]),
            "--max-chars",
            str(effective_max_chars),
            "--exaggeration",
            str(delivery_profile["exaggeration"]),
            "--temperature",
            str(delivery_profile["temperature"]),
            "--cfg-weight",
            str(effective_cfg_weight),
        ]
    else:
        command = [
            str(torch_python),
            str(torch_helper),
            "--script-file",
            str(script_path),
            "--reference-audio",
            str(reference_audio),
            "--output",
            str(output_path),
            "--language",
            language_code,
            "--conditionals-cache",
            str(torch_conditionals_path),
            "--device",
            "cpu",
            "--pause-seconds",
            str(delivery_profile["pause_seconds"]),
            "--max-chars",
            str(delivery_profile["max_chars"]),
            "--exaggeration",
            str(delivery_profile["exaggeration"]),
            "--temperature",
            str(delivery_profile["temperature"]),
        ]

    lock_path = output_path.with_suffix(".lock")
    lock_fd: Optional[int] = None
    for attempt in range(2):
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
            break
        except FileExistsError:
            lock_age = time.time() - lock_path.stat().st_mtime
            if attempt == 0 and lock_age > 30 * 60:
                lock_path.unlink(missing_ok=True)
                continue
            return None, (
                "This producer voiceover is already being generated. "
                "Please wait for the active run instead of clicking Generate again."
            )

    used_mlx = mlx_available
    try:
        result = run_command(command)
        if result.returncode != 0 and mlx_available and torch_available:
            mlx_output_complete = (
                output_path.exists()
                and output_path.stat().st_size > 1024
                and voiceover_duration_is_plausible(
                    output_path, clean_script, delivery_profile
                )
            )
            if mlx_output_complete:
                result = subprocess.CompletedProcess(
                    command,
                    0,
                    result.stdout,
                    result.stderr,
                )
            else:
                fallback_command = [
                    str(torch_python),
                    str(torch_helper),
                    "--script-file",
                    str(script_path),
                    "--reference-audio",
                    str(reference_audio),
                    "--output",
                    str(output_path),
                    "--language",
                    language_code,
                    "--conditionals-cache",
                    str(torch_conditionals_path),
                    "--device",
                    "cpu",
                    "--pause-seconds",
                    str(delivery_profile["pause_seconds"]),
                    "--max-chars",
                    str(delivery_profile["max_chars"]),
                    "--exaggeration",
                    str(delivery_profile["exaggeration"]),
                    "--temperature",
                    str(delivery_profile["temperature"]),
                ]
                result = run_command(fallback_command)
                used_mlx = False
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return None, result.stderr[-2200:] or result.stdout[-1200:] or "Local voice cloning failed."
    if not output_path.exists():
        return None, "The local voice-cloning model did not produce an audio file."
    if not voiceover_duration_is_plausible(output_path, clean_script, delivery_profile):
        output_path.unlink()
        return None, (
            "Voice generation stopped because the model produced an incomplete reading. "
            "Please generate again; the partial audio was not used in the video."
        )
    pace_adjusted = apply_producer_pace(output_path, clean_script, delivery_profile)
    pitch_adjusted = apply_producer_pitch(output_path, delivery_profile)
    elapsed_match = re.search(r"\t([0-9]+(?:\.[0-9]+)?)\s*$", result.stdout)
    timing = f" in {float(elapsed_match.group(1)):.0f} seconds" if elapsed_match else ""
    if used_mlx:
        engine_label = "Apple MLX 4-bit" if use_fast_mlx else "Apple MLX FP16"
    else:
        engine_label = "local CPU"
    pace_label = " with matched newsroom pacing" if pace_adjusted else ""
    pitch_label = " and calibrated vocal register" if pitch_adjusted else ""
    return (
        output_path,
        f"Voiceover generated with {engine_label}{timing} using "
        f"{delivery_profile['name']}’s punctuation-aware delivery profile"
        f"{pace_label}{pitch_label}.",
    )


def create_local_cloned_voiceover_preview(
    script: str,
    reference_audio: Path,
    language_code: str = "hi",
    voice_model_mode: str = "fast",
) -> Tuple[Optional[bytes], str]:
    """Generate an audio-only preview without retaining a WAV on disk."""
    mlx_python = APP_DIR / ".venv-mlx-voice" / "bin" / "python"
    mlx_helper = APP_DIR / "local_mlx_voice_clone.py"
    if not mlx_python.exists() or not mlx_helper.exists():
        return None, "The local Apple MLX voice environment is not installed."
    if not reference_audio.exists():
        return None, "The selected producer reference audio is missing."

    clean_script = script.strip()
    if not clean_script:
        return None, "Enter or generate a transcript first."
    delivery_script = prepare_script_for_delivery(clean_script, language_code)
    delivery_profile = producer_delivery_profile(reference_audio)

    # Preserve the verified real take for its exact matching script, but return
    # bytes so Streamlit does not create another media artifact.
    human_take_value = delivery_profile.get("human_take")
    human_take = Path(str(human_take_value)) if human_take_value else None
    if human_take and human_take.exists() and CANONICAL_HUMAN_SCRIPT.exists():
        canonical_script = CANONICAL_HUMAN_SCRIPT.read_text(encoding="utf-8")
        match_ratio = SequenceMatcher(
            None,
            normalized_script_identity(clean_script),
            normalized_script_identity(canonical_script),
        ).ratio()
        if match_ratio >= 0.97:
            return (
                human_take.read_bytes(),
                f"Previewing {delivery_profile['name']}’s verified human recording.",
            )

    fast_mode = voice_model_mode == "fast"
    model_id = (
        "mlx-community/chatterbox-4bit"
        if fast_mode
        else "mlx-community/chatterbox-fp16"
    )
    max_chars = (
        max(360, int(delivery_profile["max_chars"]))
        if fast_mode
        else int(delivery_profile["max_chars"])
    )
    cfg_weight = 0.30 if fast_mode else float(delivery_profile["cfg_weight"])
    reference_stat = reference_audio.stat()
    conditionals_identity = (
        f"{reference_audio.resolve()}:{reference_stat.st_size}:"
        f"{reference_stat.st_mtime_ns}:{model_id}:"
        f"{delivery_profile['exaggeration']}"
    )
    conditionals_key = hashlib.sha256(
        conditionals_identity.encode("utf-8")
    ).hexdigest()[:16]
    conditionals_path = (
        VOICE_CACHE_DIR / f"preview_conditionals_{conditionals_key}.safetensors"
    )

    ensure_dirs()
    with tempfile.TemporaryDirectory(prefix="partner-voice-preview-") as temp_dir:
        temp_root = Path(temp_dir)
        script_path = temp_root / "delivery_script.txt"
        output_path = temp_root / "voice_preview.wav"
        script_path.write_text(delivery_script, encoding="utf-8")
        command = [
            str(mlx_python),
            str(mlx_helper),
            "--script-file",
            str(script_path),
            "--reference-audio",
            str(reference_audio),
            "--output",
            str(output_path),
            "--language",
            language_code,
            "--model-id",
            model_id,
            "--conditionals-cache",
            str(conditionals_path),
            "--pause-seconds",
            str(delivery_profile["pause_seconds"]),
            "--max-chars",
            str(max_chars),
            "--exaggeration",
            str(delivery_profile["exaggeration"]),
            "--temperature",
            str(delivery_profile["temperature"]),
            "--cfg-weight",
            str(cfg_weight),
        ]
        result = run_command(command)
        if result.returncode != 0:
            return (
                None,
                result.stderr[-2200:]
                or result.stdout[-1200:]
                or "Local voice preview generation failed.",
            )
        if not output_path.exists() or output_path.stat().st_size <= 1024:
            return None, "The voice model did not produce an audio preview."
        if not voiceover_duration_is_plausible(
            output_path, clean_script, delivery_profile
        ):
            return None, "The voice model produced an incomplete preview."

        pace_adjusted = apply_producer_pace(
            output_path, clean_script, delivery_profile
        )
        pitch_adjusted = apply_producer_pitch(output_path, delivery_profile)
        preview_bytes = output_path.read_bytes()

    quality_label = "Fast natural" if fast_mode else "Maximum fidelity"
    adjustments = []
    if pace_adjusted:
        adjustments.append("matched pacing")
    if pitch_adjusted:
        adjustments.append("calibrated register")
    adjustment_label = f" with {' and '.join(adjustments)}" if adjustments else ""
    return (
        preview_bytes,
        f"{quality_label} audio preview generated in session memory"
        f"{adjustment_label}. No audio or video export was retained.",
    )


def deepika_f5_is_ready() -> bool:
    return all(
        path.exists()
        for path in (
            APP_DIR / ".venv-voice/bin/python",
            DEEPIKA_F5_HELPER,
            DEEPIKA_F5_CHECKPOINT,
            DEEPIKA_F5_VOCAB,
            DEEPIKA_F5_SOURCE,
        )
    )


def deepika_f5_command(script_path: Path, output_path: Path) -> List[str]:
    return [
        str(APP_DIR / ".venv-voice/bin/python"),
        str(DEEPIKA_F5_HELPER),
        "--script-file",
        str(script_path),
        "--output",
        str(output_path),
        "--checkpoint",
        str(DEEPIKA_F5_CHECKPOINT),
        "--vocab",
        str(DEEPIKA_F5_VOCAB),
        "--reference-source",
        str(DEEPIKA_F5_SOURCE),
        "--reference-start",
        str(DEEPIKA_F5_REFERENCE_START),
        "--reference-end",
        str(DEEPIKA_F5_REFERENCE_END),
        "--reference-text",
        DEEPIKA_F5_REFERENCE_TEXT,
        "--serious-reference-start",
        str(DEEPIKA_F5_SERIOUS_START),
        "--serious-reference-end",
        str(DEEPIKA_F5_SERIOUS_END),
        "--serious-reference-text",
        DEEPIKA_F5_SERIOUS_TEXT,
        "--question-reference-start",
        str(DEEPIKA_F5_QUESTION_START),
        "--question-reference-end",
        str(DEEPIKA_F5_QUESTION_END),
        "--question-reference-text",
        DEEPIKA_F5_QUESTION_TEXT,
        "--nfe-steps",
        "16",
        "--speed",
        "0.88",
    ]


def create_deepika_f5_preview(script: str) -> Tuple[Optional[bytes], str]:
    """Generate a trained-model preview while retaining no derived audio."""
    if not deepika_f5_is_ready():
        return None, "The Deepika fine-tuned F5 pilot files are incomplete."
    delivery_script = prepare_script_for_delivery(script, "hi")
    if not delivery_script:
        return None, "Enter or generate a transcript first."

    with tempfile.TemporaryDirectory(prefix="deepika-f5-preview-") as temp_dir:
        temp_root = Path(temp_dir)
        script_path = temp_root / "script.txt"
        output_path = temp_root / "preview.wav"
        script_path.write_text(delivery_script, encoding="utf-8")
        result = run_command(
            deepika_f5_command(script_path, output_path),
            timeout_seconds=30 * 60,
        )
        if result.returncode != 0:
            return (
                None,
                result.stderr[-2600:]
                or result.stdout[-1600:]
                or "Deepika F5 preview generation failed.",
            )
        if not output_path.exists() or output_path.stat().st_size <= 1024:
            return None, "The Deepika F5 pilot did not produce an audio preview."
        if not voiceover_duration_is_plausible(
            output_path,
            script,
            PRODUCER_VOICE_PROFILES["Deepika"],
        ):
            return None, "The Deepika F5 pilot produced an incomplete reading."
        preview_bytes = output_path.read_bytes()

    return (
        preview_bytes,
        "Generated with Deepika’s fine-tuned Hindi F5 pilot. "
        "The temporary WAV was deleted after loading the preview into this session.",
    )


def create_deepika_f5_voiceover(script: str) -> Tuple[Optional[Path], str]:
    """Generate the trained Deepika voice used by the final video renderer."""
    if not deepika_f5_is_ready():
        return None, "The Deepika fine-tuned F5 pilot files are incomplete."
    ensure_dirs()
    delivery_script = prepare_script_for_delivery(script, "hi")
    model_stat = DEEPIKA_F5_CHECKPOINT.stat()
    output_key = hashlib.sha256(
        (
            f"deepika-f5-pilot-standard-reference-v10:{model_stat.st_size}:{model_stat.st_mtime_ns}:"
            f"{delivery_script}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    output_path = VOICE_CACHE_DIR / f"deepika_f5_{output_key}.wav"
    if output_path.exists() and output_path.stat().st_size > 1024:
        return output_path, "Using the cached Deepika fine-tuned F5 voiceover."

    with tempfile.TemporaryDirectory(prefix="deepika-f5-export-") as temp_dir:
        script_path = Path(temp_dir) / "script.txt"
        script_path.write_text(delivery_script, encoding="utf-8")
        result = run_command(
            deepika_f5_command(script_path, output_path),
            timeout_seconds=30 * 60,
        )
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        return (
            None,
            result.stderr[-2600:]
            or result.stdout[-1600:]
            or "Deepika F5 voice generation failed.",
        )
    if not output_path.exists() or output_path.stat().st_size <= 1024:
        return None, "The Deepika F5 pilot did not produce a voiceover."
    if not voiceover_duration_is_plausible(
        output_path,
        script,
        PRODUCER_VOICE_PROFILES["Deepika"],
    ):
        output_path.unlink(missing_ok=True)
        return None, "The Deepika F5 pilot produced an incomplete reading."
    return output_path, "Voiceover generated with Deepika’s fine-tuned Hindi F5 pilot."


def build_script_slate(script: str, source: Path) -> Path:
    ensure_dirs()
    output = EXPORT_DIR / f"{source.stem}_script_slate.png"
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), "#10131a")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
    except Exception:
        title_font = body_font = small_font = ImageFont.load_default()
    draw.rectangle((0, 0, OUTPUT_WIDTH, 150), fill="#8f0711")
    draw.text((70, 46), "AI Anchor Script Preview", font=title_font, fill="#ffffff")
    draw.text((70, 178), "Generated newsroom script", font=small_font, fill="#bfc7d5")
    wrapped = textwrap.wrap(re.sub(r"\s+", " ", script).strip(), width=72)
    y = 230
    for line in wrapped[:18]:
        draw.text((70, y), line, font=body_font, fill="#f4f6fb")
        y += 45
    draw.rectangle((70, OUTPUT_HEIGHT - 130, OUTPUT_WIDTH - 70, OUTPUT_HEIGHT - 70), outline="#e43b45", width=3)
    draw.text((90, OUTPUT_HEIGHT - 114), "Prototype placeholder for anchor clone / presenter frame", font=small_font, fill="#f6d35d")
    image.save(output)
    return output


def _overlay_font(text: str, size: int):
    from PIL import ImageFont

    devanagari = bool(re.search(r"[\u0900-\u097f]", text))
    candidates = (
        [
            "/System/Library/Fonts/Kohinoor.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        if devanagari
        else [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def build_image_overlay_asset(item: Dict[str, object], source: Path) -> Path:
    """Create a transparent 1920x1080 layer for one uploaded still image."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    ensure_dirs()
    image_path = Path(str(item["path"]))
    placement = str(item.get("placement") or "Full frame")
    width_percent = int(item.get("width_percent") or 45)
    fit_mode = str(item.get("fit_mode") or "panel")
    custom_layout = all(key in item for key in ("x", "y", "w", "h"))
    layout_values = ":".join(
        f"{float(item.get(key) or 0):.5f}" for key in ("x", "y", "w", "h")
    ) if custom_layout else "legacy"
    cache_payload = (
        f"visual-editor-v1:{image_path}:{image_path.stat().st_mtime_ns}:"
        f"{placement}:{width_percent}:{layout_values}:{fit_mode}"
    )
    digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:16]
    output = OVERLAY_DIR / f"{source.stem}_image_layer_{digest}.png"
    if output.exists():
        return output

    with Image.open(image_path) as original:
        still = ImageOps.exif_transpose(original).convert("RGBA")
        canvas = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
        if custom_layout:
            x = int(clamp_float(float(item.get("x") or 0), 0.0, 0.96) * OUTPUT_WIDTH)
            y = int(clamp_float(float(item.get("y") or 0), 0.0, 0.96) * OUTPUT_HEIGHT)
            box_width = max(80, int(clamp_float(float(item.get("w") or 0.45), 0.08, 1.0) * OUTPUT_WIDTH))
            box_height = max(80, int(clamp_float(float(item.get("h") or 0.45), 0.08, 1.0) * OUTPUT_HEIGHT))
            box_width = min(box_width, OUTPUT_WIDTH - x)
            box_height = min(box_height, OUTPUT_HEIGHT - y)
            tile = Image.new(
                "RGBA",
                (box_width, box_height),
                (0, 0, 0, 0 if fit_mode == "contain_transparent" else 255),
            )
            if fit_mode != "contain_transparent":
                background = ImageOps.fit(
                    still.convert("RGB"),
                    (box_width, box_height),
                    method=Image.Resampling.LANCZOS,
                )
                blur_radius = max(10, int(min(box_width, box_height) * 0.035))
                background = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                background = ImageEnhance.Brightness(background).enhance(0.70).convert("RGBA")
                tile.alpha_composite(background)
            fitted = ImageOps.contain(still, (box_width, box_height), Image.Resampling.LANCZOS)
            tile.alpha_composite(
                fitted,
                ((box_width - fitted.width) // 2, (box_height - fitted.height) // 2),
            )
            canvas.alpha_composite(tile, (x, y))
        elif placement == "Full frame":
            # Fill the full 16:9 canvas with a softened version of the image so
            # portrait/square images never reveal the underlying source video.
            background = ImageOps.fit(
                still.convert("RGB"),
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                method=Image.Resampling.LANCZOS,
            )
            background = background.filter(ImageFilter.GaussianBlur(radius=34))
            background = ImageEnhance.Brightness(background).enhance(0.72).convert("RGBA")
            canvas.alpha_composite(background)
            fitted = ImageOps.contain(still, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            x = (OUTPUT_WIDTH - fitted.width) // 2
            y = (OUTPUT_HEIGHT - fitted.height) // 2
            canvas.alpha_composite(fitted, (x, y))
        else:
            target_width = max(240, int(OUTPUT_WIDTH * width_percent / 100))
            target_height = min(OUTPUT_HEIGHT - 120, int(target_width * still.height / max(still.width, 1)))
            fitted = ImageOps.contain(still, (target_width, target_height))
            margin = 56
            positions = {
                "Center": ((OUTPUT_WIDTH - fitted.width) // 2, (OUTPUT_HEIGHT - fitted.height) // 2),
                "Top left": (margin, margin),
                "Top right": (OUTPUT_WIDTH - fitted.width - margin, margin),
                "Bottom left": (margin, OUTPUT_HEIGHT - fitted.height - margin),
                "Bottom right": (OUTPUT_WIDTH - fitted.width - margin, OUTPUT_HEIGHT - fitted.height - margin),
            }
            x, y = positions.get(placement, positions["Center"])
            canvas.alpha_composite(fitted, (x, y))
        canvas.save(output)
    return output


def build_slug_overlay_asset(slug: Dict[str, object], source: Path) -> Path:
    """Render a polished newsroom lower-third with adaptive typography."""
    from PIL import Image, ImageColor, ImageDraw, ImageFilter

    ensure_dirs()
    text = re.sub(r"\s+", " ", str(slug.get("text") or "")).strip()
    highlight = re.sub(r"\s+", " ", str(slug.get("highlight_text") or "")).strip()
    label = re.sub(r"\s+", " ", str(slug.get("label") or "")).strip()
    style_name = str(slug.get("style") or "Jagran Red")
    preset = SLUG_STYLE_PRESETS.get(style_name, SLUG_STYLE_PRESETS["Jagran Red"])
    background = str(slug.get("background_color") or preset["background"])
    background_end = str(
        slug.get("background_end_color") or preset["background_end"]
    )
    highlight_color = str(slug.get("highlight_color") or preset["accent"])
    region = str(slug.get("region") or "lower_third")
    geometry = slug.get("geometry") if isinstance(slug.get("geometry"), dict) else {}
    text_color = str(preset["text"])
    highlight_text_color = str(preset["highlight_text"])
    label_text_color = str(preset["label_text"])
    cache_payload = json.dumps(
        {
            "text": text,
            "highlight": highlight,
            "label": label,
            "style": style_name,
            "background": background,
            "background_end": background_end,
            "highlight_color": highlight_color,
            "region": region,
            "geometry": geometry,
            "design_version": 5,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:16]
    output = OVERLAY_DIR / f"{source.stem}_slug_layer_{digest}.png"
    if output.exists():
        return output

    canvas = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
    if region == "template_header":
        geometry_x = clamp_float(float(geometry.get("x") or 0.01), 0.0, 0.95)
        geometry_y = clamp_float(float(geometry.get("y") or 0.01), 0.0, 0.95)
        geometry_w = clamp_float(float(geometry.get("w") or 0.98), 0.12, 1.0)
        geometry_h = clamp_float(float(geometry.get("h") or 0.18), 0.10, 1.0)
        left = int(geometry_x * OUTPUT_WIDTH)
        top = int(geometry_y * OUTPUT_HEIGHT)
        right = min(OUTPUT_WIDTH, left + int(geometry_w * OUTPUT_WIDTH))
        bottom = min(OUTPUT_HEIGHT, top + int(geometry_h * OUTPUT_HEIGHT))
        radius = 18
    else:
        left, top = 84, OUTPUT_HEIGHT - 254
        right, bottom = OUTPUT_WIDTH - 84, OUTPUT_HEIGHT - 92
        radius = 24

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (left + 5, top + 10, right + 5, bottom + 10),
        radius=radius,
        fill=(0, 0, 0, 150),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))

    panel_width = right - left
    panel_height = bottom - top
    start_rgb = ImageColor.getrgb(background)
    end_rgb = ImageColor.getrgb(background_end)
    gradient = Image.new("RGBA", (panel_width, panel_height))
    gradient_pixels = gradient.load()
    for y_position in range(panel_height):
        ratio = y_position / max(1, panel_height - 1)
        row_colour = tuple(
            int(start_rgb[channel] * (1 - ratio) + end_rgb[channel] * ratio)
            for channel in range(3)
        ) + (246,)
        for x_position in range(panel_width):
            gradient_pixels[x_position, y_position] = row_colour
    mask = Image.new("L", (panel_width, panel_height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, panel_width - 1, panel_height - 1), radius=radius, fill=255
    )
    canvas.paste(gradient, (left, top), mask)
    draw = ImageDraw.Draw(canvas)

    accent_rgba = ImageColor.getcolor(highlight_color, "RGBA")
    accent_left = left
    draw.rounded_rectangle(
        (accent_left, top, accent_left + 18, bottom),
        radius=9,
        fill=accent_rgba,
    )
    draw.line(
        (accent_left + 42, top + 19, right - 34, top + 19),
        fill=(*ImageColor.getrgb(highlight_color), 105),
        width=2,
    )

    text_left = accent_left + 58
    text_right = right - 42
    text_top = top + 33
    if label:
        label_font = _overlay_font(label, 27)
        label_bbox = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_bbox[2] - label_bbox[0] + 40
        label_top = top - 27
        label_bottom = top + 22
        draw.rounded_rectangle(
            (text_left, label_top, text_left + label_width, label_bottom),
            radius=14,
            fill=accent_rgba,
        )
        label_y = label_top + (label_bottom - label_top - (label_bbox[3] - label_bbox[1])) / 2 - label_bbox[1]
        draw.text(
            (text_left + 20, label_y),
            label,
            font=label_font,
            fill=label_text_color,
        )
        text_top += 13

    max_text_width = text_right - text_left
    highlight_match = (
        re.search(re.escape(highlight), text, flags=re.IGNORECASE)
        if highlight
        else None
    )
    word_matches = list(re.finditer(r"\S+", text))

    def wrapped_words(font_size: int) -> Tuple[object, List[List[Tuple[str, bool]]]]:
        font_value = _overlay_font(text, font_size)
        lines: List[List[Tuple[str, bool]]] = [[]]
        current_width = 0.0
        space_width = draw.textlength(" ", font=font_value)
        for word_match in word_matches:
            word = word_match.group(0)
            word_width = draw.textlength(word, font=font_value)
            addition = word_width + (space_width if lines[-1] else 0)
            if lines[-1] and current_width + addition > max_text_width:
                lines.append([])
                current_width = 0.0
                addition = word_width
            is_highlighted = bool(
                highlight_match
                and word_match.start() < highlight_match.end()
                and word_match.end() > highlight_match.start()
            )
            lines[-1].append((word, is_highlighted))
            current_width += addition
        return font_value, lines

    font_size = 52
    font, lines = wrapped_words(font_size)
    sample_bbox = draw.textbbox((0, 0), text or "Ag", font=font)
    line_height = sample_bbox[3] - sample_bbox[1]
    line_gap = 13
    block_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    available_text_height = max(40, bottom - text_top - 12)
    while font_size > 30 and (
        len(lines) > 2 or block_height > available_text_height
    ):
        font_size -= 2
        font, lines = wrapped_words(font_size)
        sample_bbox = draw.textbbox((0, 0), text or "Ag", font=font)
        line_height = sample_bbox[3] - sample_bbox[1]
        block_height = (
            len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        )
    if len(lines) > 2:
        lines = lines[:2]
        last_word, last_highlighted = lines[-1][-1]
        lines[-1][-1] = (last_word.rstrip("…") + "…", last_highlighted)

    block_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    y = text_top + max(0, (bottom - text_top - block_height) / 2) - sample_bbox[1]
    space_width = draw.textlength(" ", font=font)
    for line in lines:
        x = text_left
        for word_index, (word, highlighted_word) in enumerate(line):
            if word_index:
                x += space_width
            word_width = draw.textlength(word, font=font)
            if highlighted_word:
                draw.rounded_rectangle(
                    (
                        x - 8,
                        y + sample_bbox[1] - 6,
                        x + word_width + 8,
                        y + sample_bbox[3] + 6,
                    ),
                    radius=8,
                    fill=accent_rgba,
                )
                draw.text(
                    (x, y),
                    word,
                    font=font,
                    fill=highlight_text_color,
                )
            else:
                draw.text((x, y), word, font=font, fill=text_color)
            x += word_width
        y += line_height + line_gap
    canvas.save(output)
    return output


def build_source_cut_cache(
    source: Path,
    cuts: List[Tuple[float, float]],
) -> Tuple[Optional[Path], str]:
    """Create or reuse an accurately cut, video-only source for final rendering."""
    if not cuts:
        return source, "No source sections selected for removal."
    ffmpeg = tool_path("ffmpeg")
    if not ffmpeg:
        return None, "ffmpeg is required to remove source-video sections."
    source_duration = probe_media_duration(source)
    keep_ranges = kept_source_ranges(cuts, source_duration)
    if not keep_ranges:
        return None, "The selected removals delete the entire source video."
    source_stat = source.stat()
    cache_payload = json.dumps(
        {
            "source": str(source.resolve()),
            "size": source_stat.st_size,
            "modified": source_stat.st_mtime_ns,
            "cuts": cuts,
            "version": 1,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:18]
    cut_cache_dir = WORK_DIR / "cut_cache"
    cut_cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cut_cache_dir / f"{source.stem}_cuts_{digest}.mp4"
    if output_path.exists() and output_path.stat().st_size > 1_024:
        return output_path, "Using the cached source-video cuts."

    filter_parts: List[str] = []
    source_labels = [f"cutsource{index}" for index in range(len(keep_ranges))]
    if len(source_labels) > 1:
        filter_parts.append(
            f"[0:v]split={len(source_labels)}"
            + "".join(f"[{label}]" for label in source_labels)
        )
    kept_labels: List[str] = []
    for index, (keep_start, keep_end) in enumerate(keep_ranges):
        input_label = source_labels[index] if len(source_labels) > 1 else "0:v"
        output_label = f"cutkeep{index}"
        filter_parts.append(
            f"[{input_label}]trim=start={keep_start:.3f}:end={keep_end:.3f},"
            f"setpts=PTS-STARTPTS[{output_label}]"
        )
        kept_labels.append(f"[{output_label}]")
    if len(kept_labels) == 1:
        filter_parts.append(f"{kept_labels[0]}null[cutout]")
    else:
        filter_parts.append(
            "".join(kept_labels)
            + f"concat=n={len(kept_labels)}:v=1:a=0[cutout]"
        )
    args = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[cutout]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = run_command(args)
    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        return None, result.stderr[-1800:] or "Source-video cutting failed."
    return output_path, "Selected raw-video sections removed."


def build_browser_preview(source: Path) -> Tuple[Optional[Path], str]:
    """Create a small, seekable proxy that Streamlit can serve reliably."""
    ffmpeg = tool_path("ffmpeg")
    if not ffmpeg:
        return None, "FFmpeg is required to create the browser preview."
    source_stat = source.stat()
    cache_payload = json.dumps(
        {
            "source": str(source.resolve()),
            "size": source_stat.st_size,
            "modified": source_stat.st_mtime_ns,
            "version": 1,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:18]
    preview_path = PREVIEW_DIR / f"{source.stem}_preview_{digest}.mp4"
    if preview_path.exists() and preview_path.stat().st_size > 1_024:
        return preview_path, "Using the cached browser preview."
    args = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "scale=960:-2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-maxrate",
        "2M",
        "-bufsize",
        "4M",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "60",
        "-keyint_min",
        "60",
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    result = run_command(args)
    if result.returncode != 0:
        if preview_path.exists():
            preview_path.unlink()
        return None, result.stderr[-1800:] or "Browser preview creation failed."
    return preview_path, "Browser preview created."


def export_horizontal_video(
    source: Path,
    script: str,
    voiceover: Optional[Path],
    keep_original_audio: bool,
    add_intro_slate: bool,
    image_overlays: Optional[List[Dict[str, object]]] = None,
    slug_overlays: Optional[List[Dict[str, object]]] = None,
    source_cuts: Optional[List[Dict[str, object]]] = None,
    voice_timing: Optional[Dict[str, object]] = None,
    output_stem: Optional[str] = None,
    tail_mode: str = "end",
    pause_audio_overlays: Optional[List[Dict[str, object]]] = None,
    template_layout: str = "classic",
    template_geometry: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Optional[Path], str]:
    ffmpeg = tool_path("ffmpeg")
    if not ffmpeg:
        return None, "ffmpeg is required. Install it first."
    ensure_dirs()
    original_source = source
    original_source_duration = probe_media_duration(original_source)
    requested_source_cuts = normalise_cut_ranges(
        source_cuts or [], original_source_duration
    )
    if requested_source_cuts:
        cut_source, cut_message = build_source_cut_cache(
            original_source, requested_source_cuts
        )
        if not cut_source:
            return None, cut_message
        source = cut_source
    export_stem = output_stem or original_source.stem
    output_path = EXPORT_DIR / f"{export_stem}_ai_anchor_horizontal.mp4"
    counter = 1
    while output_path.exists():
        output_path = EXPORT_DIR / f"{export_stem}_ai_anchor_horizontal_{counter}.mp4"
        counter += 1

    source_duration = probe_media_duration(source)
    merged_source_cuts: List[Tuple[float, float]] = []
    source_keep_ranges = kept_source_ranges(
        merged_source_cuts, source_duration
    )
    if not source_keep_ranges:
        return None, "The selected removals delete the entire source video."
    edited_source_duration = sum(
        end - start for start, end in source_keep_ranges
    )

    source_geometry = (template_geometry or {}).get("source", {})
    if template_layout == "two_column":
        video_x = int(clamp_float(float(source_geometry.get("x", 0.0)), 0.0, 0.95) * OUTPUT_WIDTH)
        video_y = int(clamp_float(float(source_geometry.get("y", 0.20)), 0.0, 0.95) * OUTPUT_HEIGHT)
        video_width = int(clamp_float(float(source_geometry.get("w", 0.50)), 0.08, 1.0) * OUTPUT_WIDTH)
        video_height = int(clamp_float(float(source_geometry.get("h", 0.80)), 0.08, 1.0) * OUTPUT_HEIGHT)
        video_width = min(video_width, OUTPUT_WIDTH - video_x)
        video_height = min(video_height, OUTPUT_HEIGHT - video_y)
        video_width -= video_width % 2
        video_height -= video_height % 2
    elif template_layout == "three_column":
        video_width, video_height = OUTPUT_WIDTH // 3, int(OUTPUT_HEIGHT * 0.80)
        video_x, video_y = 0, OUTPUT_HEIGHT - video_height
    else:
        video_width, video_height = OUTPUT_WIDTH, OUTPUT_HEIGHT
        video_x, video_y = 0, 0
    if template_layout in {"two_column", "three_column"}:
        video_transform = (
            f"scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
            f"crop={video_width}:{video_height},"
            f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:{video_x}:{video_y}:black,setsar=1"
        )
    else:
        video_transform = (
            f"scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
            f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:{video_x}:{video_y}:black,setsar=1"
        )
    voiceover_duration = probe_media_duration(voiceover) if voiceover else 0.0
    voiceover_start = max(
        0.0, float((voice_timing or {}).get("start") or 0.0)
    )
    voice_pauses = normalise_voice_pauses(
        list((voice_timing or {}).get("pauses") or []),
        voiceover_start,
    )
    voice_segments: List[Tuple[float, float, float]] = []
    if voiceover and voiceover_duration > 0.01:
        audio_cursor = 0.0
        video_cursor = voiceover_start
        for pause_start, pause_duration in voice_pauses:
            playable = max(0.0, pause_start - video_cursor)
            audio_end = min(voiceover_duration, audio_cursor + playable)
            if audio_end > audio_cursor + 0.001:
                voice_segments.append((audio_cursor, audio_end, video_cursor))
                audio_cursor = audio_end
            if audio_cursor >= voiceover_duration - 0.001:
                break
            video_cursor = max(video_cursor, pause_start) + pause_duration
        if audio_cursor < voiceover_duration - 0.001:
            voice_segments.append(
                (audio_cursor, voiceover_duration, video_cursor)
            )
    scheduled_voice_end = max(
        (
            video_start + audio_end - audio_start
            for audio_start, audio_end, video_start in voice_segments
        ),
        default=voiceover_start + voiceover_duration,
    )
    intro_duration = 4.0 if add_intro_slate else 0.0
    extension = max(
        0.0,
        scheduled_voice_end - (edited_source_duration + intro_duration),
    )
    desired_main_duration = edited_source_duration + extension
    normalised_tail_mode = (
        tail_mode if tail_mode in {"end", "black", "loop"} else "end"
    )
    source_repeat_count = 1
    if normalised_tail_mode == "loop" and extension > 0.01:
        # Repeat only the already-cut source. Removed ranges cannot return
        # because they are absent from every repeated input.
        source_repeat_count = max(
            1,
            int(
                math.ceil(
                    desired_main_duration / max(0.1, edited_source_duration)
                )
            ),
        )
        source_repeat_count = min(source_repeat_count, 8)

    if add_intro_slate:
        slate = build_script_slate(script, source)
        args = [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-t",
            "4",
            "-i",
            str(slate),
            "-i",
            str(source),
        ]
        input_index = 1
    else:
        args = [ffmpeg, "-y", "-i", str(source)]
        input_index = 0

    source_input_indices = [input_index]
    next_input_index = input_index + 1
    for _ in range(source_repeat_count - 1):
        args.extend(["-i", str(source)])
        source_input_indices.append(next_input_index)
        next_input_index += 1
    voiceover_input_index: Optional[int] = None
    if voiceover:
        args.extend(["-i", str(voiceover)])
        voiceover_input_index = next_input_index
        next_input_index += 1

    pause_audio_inputs: List[Dict[str, object]] = []
    for pause_audio in pause_audio_overlays or []:
        pause_audio_path = Path(str(pause_audio.get("path") or ""))
        if not pause_audio_path.exists():
            continue
        pause_start = max(0.0, float(pause_audio.get("start") or 0.0))
        pause_duration = max(0.1, float(pause_audio.get("duration") or 0.1))
        args.extend(["-i", str(pause_audio_path)])
        pause_audio_inputs.append(
            {
                "input": next_input_index,
                "start": pause_start,
                "duration": pause_duration,
            }
        )
        next_input_index += 1

    visual_inputs: List[Dict[str, object]] = []
    byte_audio_inputs: List[Dict[str, object]] = []
    for item in sorted(image_overlays or [], key=lambda value: int(value.get("z") or 0)):
        if not item.get("path"):
            continue
        media_type = str(item.get("media_type") or "image")
        start = max(0.0, float(item.get("start") or 0.0))
        duration = max(0.1, float(item.get("duration") or 0.1))
        end = start + duration
        if media_type == "video":
            video_path = Path(str(item["path"]))
            trim_start = 0.0
            args.extend(["-i", str(video_path)])
            visual_inputs.append(
                {
                    "kind": "video",
                    "input": next_input_index,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "trim_start": trim_start,
                    "x": float(item.get("x") or 0.0),
                    "y": float(item.get("y") or 0.0),
                    "w": float(item.get("w") or 1.0),
                    "h": float(item.get("h") or 1.0),
                }
            )
            if bool(item.get("use_clip_audio")) and media_has_audio(
                str(video_path), video_path.stat().st_mtime_ns
            ):
                byte_audio_inputs.append(
                    {
                        "input": next_input_index,
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "trim_start": trim_start,
                    }
                )
        else:
            layer = build_image_overlay_asset(item, source)
            args.extend(["-framerate", "30", "-i", str(layer)])
            visual_inputs.append(
                {
                    "kind": "image",
                    "input": next_input_index,
                    "start": start,
                    "end": end,
                }
            )
        next_input_index += 1

    for slug in slug_overlays or []:
        if not slug.get("text"):
            continue
        layer = build_slug_overlay_asset(slug, source)
        start = max(0.0, float(slug.get("start") or 0.0))
        duration = slug.get("duration")
        end = 86400.0 if duration is None else start + max(0.1, float(duration))
        args.extend(["-framerate", "30", "-i", str(layer)])
        visual_inputs.append(
            {
                "kind": "image",
                "input": next_input_index,
                "start": start,
                "end": end,
            }
        )
        next_input_index += 1

    filter_parts: List[str] = []
    full_source_kept = (
        len(source_keep_ranges) == 1
        and source_keep_ranges[0][0] <= 0.01
        and source_keep_ranges[0][1] >= source_duration - 0.01
    )
    if full_source_kept:
        filter_parts.append(
            f"[{input_index}:v]{video_transform}[editedmainv]"
        )
    else:
        source_video_labels = [
            f"sourcekeepv{keep_number}"
            for keep_number in range(len(source_keep_ranges))
        ]
        filter_parts.append(
            f"[{input_index}:v]split={len(source_video_labels)}"
            + "".join(f"[{label}]" for label in source_video_labels)
        )
        kept_video_labels: List[str] = []
        for keep_number, (keep_start, keep_end) in enumerate(source_keep_ranges):
            keep_label = f"keptv{keep_number}"
            filter_parts.append(
                f"[{source_video_labels[keep_number]}]"
                f"trim=start={keep_start:.3f}:"
                f"end={keep_end:.3f},setpts=PTS-STARTPTS,"
                f"{video_transform}[{keep_label}]"
            )
            kept_video_labels.append(f"[{keep_label}]")
        filter_parts.append(
            "".join(kept_video_labels)
            + f"concat=n={len(kept_video_labels)}:v=1:a=0[editedmainv]"
        )
    if len(source_input_indices) > 1:
        repeated_source_labels = ["[editedmainv]"]
        for repeat_number, repeat_input_index in enumerate(source_input_indices[1:], 1):
            repeat_label = f"repeatedmainv{repeat_number}"
            filter_parts.append(
                f"[{repeat_input_index}:v]{video_transform},"
                f"setpts=PTS-STARTPTS[{repeat_label}]"
            )
            repeated_source_labels.append(f"[{repeat_label}]")
        filter_parts.append(
            "".join(repeated_source_labels)
            + f"concat=n={len(repeated_source_labels)}:v=1:a=0,"
            f"trim=duration={desired_main_duration:.3f},"
            "setpts=PTS-STARTPTS[repeatedmainv]"
        )
        repeated_duration = edited_source_duration * len(source_input_indices)
        if repeated_duration + 0.01 < desired_main_duration:
            filter_parts.append(
                f"[repeatedmainv]tpad=stop_mode=add:color=black:"
                f"stop_duration={desired_main_duration - repeated_duration:.3f}"
                "[mainv]"
            )
        else:
            filter_parts.append("[repeatedmainv]null[mainv]")
    elif extension > 0.01 and normalised_tail_mode == "black":
        filter_parts.append(
            f"[editedmainv]tpad=stop_mode=add:color=black:"
            f"stop_duration={extension:.3f}[mainv]"
        )
    else:
        filter_parts.append("[editedmainv]null[mainv]")
    if add_intro_slate:
        filter_parts.append(f"[0:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1[slatev]")
        filter_parts.append("[slatev][mainv]concat=n=2:v=1:a=0[basev]")
    else:
        filter_parts.append("[mainv]null[basev]")

    current_video_label = "basev"
    for overlay_number, visual in enumerate(visual_inputs):
        overlay_input = int(visual["input"])
        start = float(visual["start"])
        end = float(visual["end"])
        next_video_label = f"overlayv{overlay_number}"
        if visual["kind"] == "video":
            trim_start = float(visual["trim_start"])
            duration = float(visual["duration"])
            x = int(
                clamp_float(float(visual["x"]), 0.0, 0.96) * OUTPUT_WIDTH
            )
            y = int(
                clamp_float(float(visual["y"]), 0.0, 0.96) * OUTPUT_HEIGHT
            )
            box_width = max(
                80,
                int(clamp_float(float(visual["w"]), 0.08, 1.0) * OUTPUT_WIDTH),
            )
            box_height = max(
                80,
                int(clamp_float(float(visual["h"]), 0.08, 1.0) * OUTPUT_HEIGHT),
            )
            box_width = min(box_width, OUTPUT_WIDTH - x)
            box_height = min(box_height, OUTPUT_HEIGHT - y)
            # H.264 works most reliably with even frame dimensions.
            box_width = max(80, box_width - box_width % 2)
            box_height = max(80, box_height - box_height % 2)
            filter_parts.extend(
                [
                    (
                        f"[{overlay_input}:v]"
                        f"trim=start={trim_start:.3f}:duration={duration:.3f},"
                        "setpts=PTS-STARTPTS,split=2"
                        f"[bytebg{overlay_number}][bytefg{overlay_number}]"
                    ),
                    (
                        f"[bytebg{overlay_number}]"
                        f"scale={box_width}:{box_height}:"
                        "force_original_aspect_ratio=increase,"
                        f"crop={box_width}:{box_height},boxblur=20:2"
                        f"[bytebgfit{overlay_number}]"
                    ),
                    (
                        f"[bytefg{overlay_number}]"
                        f"scale={box_width}:{box_height}:"
                        "force_original_aspect_ratio=decrease"
                        f"[bytefgfit{overlay_number}]"
                    ),
                    (
                        f"[bytebgfit{overlay_number}][bytefgfit{overlay_number}]"
                        "overlay=(W-w)/2:(H-h)/2:shortest=1,"
                        f"setpts=PTS+{start:.3f}/TB[bytev{overlay_number}]"
                    ),
                ]
            )
            overlay_source = f"bytev{overlay_number}"
            overlay_position = f"{x}:{y}"
        else:
            static_duration = max(0.1, end - start)
            filter_parts.append(
                f"[{overlay_input}:v]"
                "loop=loop=-1:size=1:start=0,"
                f"trim=duration={static_duration:.3f},"
                f"setpts=PTS-STARTPTS+{start:.3f}/TB"
                f"[staticv{overlay_number}]"
            )
            overlay_source = f"staticv{overlay_number}"
            overlay_position = "0:0"
        enable_option = (
            f":enable='between(t,{start:.3f},{end:.3f})'"
            if visual["kind"] == "video"
            else ""
        )
        filter_parts.append(
            f"[{current_video_label}][{overlay_source}]"
            f"overlay={overlay_position}{enable_option}:"
            f"eof_action=pass:repeatlast=0[{next_video_label}]"
        )
        current_video_label = next_video_label
    filter_parts.append(f"[{current_video_label}]null[vout]")

    base_audio_label: Optional[str] = None
    if voiceover_input_index is not None:
        if voice_segments:
            voice_source_labels = [
                f"voicesource{segment_number}"
                for segment_number in range(len(voice_segments))
            ]
            if len(voice_source_labels) > 1:
                filter_parts.append(
                    f"[{voiceover_input_index}:a]"
                    f"asplit={len(voice_source_labels)}"
                    + "".join(f"[{label}]" for label in voice_source_labels)
                )
            scheduled_labels: List[str] = []
            scheduled_cursor = 0.0
            for segment_number, (
                audio_start,
                audio_end,
                video_start,
            ) in enumerate(voice_segments):
                silence_duration = max(0.0, video_start - scheduled_cursor)
                if silence_duration > 0.001:
                    silence_label = f"voicesilence{segment_number}"
                    filter_parts.append(
                        "anullsrc=r=44100:cl=mono,"
                        f"atrim=duration={silence_duration:.3f}"
                        f"[{silence_label}]"
                    )
                    scheduled_labels.append(f"[{silence_label}]")
                segment_label = f"voicesegment{segment_number}"
                voice_source = (
                    voice_source_labels[segment_number]
                    if len(voice_source_labels) > 1
                    else f"{voiceover_input_index}:a"
                )
                filter_parts.append(
                    f"[{voice_source}]"
                    f"atrim=start={audio_start:.3f}:end={audio_end:.3f},"
                    "asetpts=PTS-STARTPTS,aresample=44100,"
                    "aformat=sample_fmts=fltp:channel_layouts=mono"
                    f"[{segment_label}]"
                )
                scheduled_labels.append(f"[{segment_label}]")
                scheduled_cursor = video_start + audio_end - audio_start
            if len(scheduled_labels) == 1:
                filter_parts.append(
                    f"{scheduled_labels[0]}anull[scheduledvoice]"
                )
            else:
                filter_parts.append(
                    "".join(scheduled_labels)
                    + f"concat=n={len(scheduled_labels)}:"
                    "v=0:a=1[scheduledvoice]"
                )
            filter_parts.append("[scheduledvoice]apad[baseaudio]")
        else:
            if voiceover_start > 0.001:
                filter_parts.append(
                    "anullsrc=r=44100:cl=mono,"
                    f"atrim=duration={voiceover_start:.3f}[voiceleadingsilence]"
                )
                filter_parts.append(
                    f"[{voiceover_input_index}:a]aresample=44100,"
                    "aformat=sample_fmts=fltp:channel_layouts=mono"
                    "[voiceunpaused]"
                )
                filter_parts.append(
                    "[voiceleadingsilence][voiceunpaused]"
                    "concat=n=2:v=0:a=1,apad[baseaudio]"
                )
            else:
                filter_parts.append(
                    f"[{voiceover_input_index}:a]apad[baseaudio]"
                )
        base_audio_label = "baseaudio"
    elif keep_original_audio and media_has_audio(
        str(source), source.stat().st_mtime_ns
    ):
        if full_source_kept:
            filter_parts.append(f"[{input_index}:a]apad[baseaudio]")
        else:
            source_audio_labels = [
                f"sourcekeepa{keep_number}"
                for keep_number in range(len(source_keep_ranges))
            ]
            if len(source_audio_labels) > 1:
                filter_parts.append(
                    f"[{input_index}:a]asplit={len(source_audio_labels)}"
                    + "".join(f"[{label}]" for label in source_audio_labels)
                )
            kept_audio_labels: List[str] = []
            for keep_number, (keep_start, keep_end) in enumerate(source_keep_ranges):
                keep_audio_label = f"kepta{keep_number}"
                source_audio = (
                    source_audio_labels[keep_number]
                    if len(source_audio_labels) > 1
                    else f"{input_index}:a"
                )
                filter_parts.append(
                    f"[{source_audio}]atrim=start={keep_start:.3f}:"
                    f"end={keep_end:.3f},asetpts=PTS-STARTPTS"
                    f"[{keep_audio_label}]"
                )
                kept_audio_labels.append(f"[{keep_audio_label}]")
            filter_parts.append(
                "".join(kept_audio_labels)
                + f"concat=n={len(kept_audio_labels)}:v=0:a=1,"
                "apad[baseaudio]"
            )
        base_audio_label = "baseaudio"

    if pause_audio_inputs:
        pause_audio_labels: List[str] = []
        for pause_audio_number, item in enumerate(pause_audio_inputs):
            delay_ms = max(0, int(round(float(item["start"]) * 1000)))
            pause_audio_label = f"pauseinserta{pause_audio_number}"
            filter_parts.append(
                f"[{int(item['input'])}:a]"
                f"atrim=duration={float(item['duration']):.3f},"
                "asetpts=PTS-STARTPTS,aresample=44100,"
                "aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"adelay={delay_ms}:all=1[{pause_audio_label}]"
            )
            pause_audio_labels.append(f"[{pause_audio_label}]")
        if base_audio_label is not None:
            filter_parts.append(
                f"[{base_audio_label}]"
                + "".join(pause_audio_labels)
                + f"amix=inputs={len(pause_audio_labels) + 1}:"
                "duration=longest:dropout_transition=0:normalize=0"
                "[basewithpauseaudio]"
            )
            base_audio_label = "basewithpauseaudio"
        elif len(pause_audio_labels) == 1:
            filter_parts.append(
                f"{pause_audio_labels[0]}apad[basewithpauseaudio]"
            )
            base_audio_label = "basewithpauseaudio"
        else:
            filter_parts.append(
                "".join(pause_audio_labels)
                + f"amix=inputs={len(pause_audio_labels)}:"
                "duration=longest:dropout_transition=0:normalize=0,"
                "apad[basewithpauseaudio]"
            )
            base_audio_label = "basewithpauseaudio"

    output_audio_label: Optional[str] = base_audio_label
    if byte_audio_inputs and base_audio_label is not None:
        mute_filters = ",".join(
            (
                "volume=0:"
                f"enable='between(t,{float(item['start']):.3f},"
                f"{float(item['end']):.3f})'"
            )
            for item in byte_audio_inputs
        )
        filter_parts.append(
            f"[{base_audio_label}]{mute_filters}[mutedbaseaudio]"
        )
        audio_labels = ["[mutedbaseaudio]"]
        for audio_number, item in enumerate(byte_audio_inputs):
            delay_ms = max(0, int(round(float(item["start"]) * 1000)))
            filter_parts.append(
                f"[{int(item['input'])}:a]"
                f"atrim=start={float(item['trim_start']):.3f}:"
                f"duration={float(item['duration']):.3f},"
                "asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}:all=1[bytea{audio_number}]"
            )
            audio_labels.append(f"[bytea{audio_number}]")
        filter_parts.append(
            "".join(audio_labels)
            + f"amix=inputs={len(audio_labels)}:"
            "duration=longest:dropout_transition=0:normalize=0[aout]"
        )
        output_audio_label = "aout"
    elif byte_audio_inputs:
        audio_labels = []
        for audio_number, item in enumerate(byte_audio_inputs):
            delay_ms = max(0, int(round(float(item["start"]) * 1000)))
            filter_parts.append(
                f"[{int(item['input'])}:a]"
                f"atrim=start={float(item['trim_start']):.3f}:"
                f"duration={float(item['duration']):.3f},"
                "asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}:all=1[bytea{audio_number}]"
            )
            audio_labels.append(f"[bytea{audio_number}]")
        if len(audio_labels) == 1:
            filter_parts.append(f"{audio_labels[0]}apad[aout]")
        else:
            filter_parts.append(
                "".join(audio_labels)
                + f"amix=inputs={len(audio_labels)}:"
                "duration=longest:dropout_transition=0:normalize=0,apad[aout]"
            )
        output_audio_label = "aout"

    # Serial graph scheduling avoids FFmpeg frame-queue exhaustion when a cut
    # source and a deliberately paused audio timeline are rendered together.
    args.extend(
        [
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
        ]
    )
    if output_audio_label:
        args.extend(["-map", f"[{output_audio_label}]", "-shortest"])
    else:
        args.extend(["-an"])
    args.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "25",
            "-maxrate",
            "5M",
            "-bufsize",
            "10M",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "60",
            "-keyint_min",
            "60",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    result = run_command(args)
    if result.returncode != 0:
        return None, result.stderr[-2200:] or "Horizontal export failed."
    return output_path, f"Horizontal 1920x1080 video exported: {output_path.name}"


def export_reference_style_video(
    source: Path,
    voiceover: Optional[Path],
    opener_seconds: int = 8,
    opener_start: int = 0,
) -> Tuple[Optional[Path], str]:
    ffmpeg = tool_path("ffmpeg")
    if not ffmpeg:
        return None, "ffmpeg is required. Install it first."
    if not REFERENCE_VIDEO.exists():
        return None, "Reference anchor sample is not available locally."
    ensure_dirs()
    output_path = EXPORT_DIR / f"{source.stem}_reference_anchor_style.mp4"
    counter = 1
    while output_path.exists():
        output_path = EXPORT_DIR / f"{source.stem}_reference_anchor_style_{counter}.mp4"
        counter += 1

    scale_pad = (
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    args = [
        ffmpeg,
        "-y",
        "-i",
        str(REFERENCE_VIDEO),
        "-i",
        str(source),
    ]
    if voiceover:
        args.extend(["-i", str(voiceover)])
    filter_graph = (
        f"[0:v]trim=start={opener_start}:end={opener_start + opener_seconds},setpts=PTS-STARTPTS,{scale_pad}[opener];"
        f"[1:v]{scale_pad}[main];"
        "[opener][main]concat=n=2:v=1:a=0[vout]"
    )
    args.extend(["-filter_complex", filter_graph, "-map", "[vout]"])
    if voiceover:
        args.extend(["-map", "2:a", "-af", "apad", "-shortest"])
    else:
        args.extend(["-map", "1:a?", "-shortest"])
    args.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    result = run_command(args)
    if result.returncode != 0:
        return None, result.stderr[-2200:] or "Reference-style export failed."
    return output_path, f"Reference-style horizontal video exported: {output_path.name}"


def render_stage_header(number: int, title: str, description: str) -> None:
    """Render a compact studio-style workflow heading."""
    st.markdown(
        f"""
        <div class="studio-stage-header">
            <div class="studio-stage-number">{number:02d}</div>
            <div>
                <div class="studio-stage-kicker">WORKFLOW STAGE</div>
                <h2>{title}</h2>
                <p>{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_header() -> None:
    """Render a compact production header and session-aware workflow status."""
    source_ready = bool(st.session_state.get("partner_video_path"))
    script_ready = bool(
        str(st.session_state.get("partner_editable_transcript") or "").strip()
        or str(st.session_state.get("partner_manual_script") or "").strip()
    )
    voice_ready = bool(
        st.session_state.get("partner_voiceover")
        or st.session_state.get("partner_eleven_preview_bytes")
    )
    timeline_ready = bool(
        st.session_state.get("partner_image_overlays")
        or st.session_state.get("partner_slug_overlays")
        or st.session_state.get("partner_source_cuts")
    )
    export_ready = bool(st.session_state.get("partner_latest_export"))
    statuses = [
        ("01", "Upload", source_ready),
        ("02", "Script", script_ready),
        ("03", "Edit", script_ready),
        ("04", "Voice", voice_ready),
        ("05", "Timeline", timeline_ready),
        ("06", "Export", export_ready),
    ]
    completed = sum(1 for _, _, ready in statuses if ready)
    active_index = min(completed, len(statuses) - 1)
    steps = []
    for index, (number, label, ready) in enumerate(statuses):
        state = "is-complete" if ready else "is-active" if index == active_index else ""
        marker = "&#10003;" if ready else number
        steps.append(
            f'<div class="studio-progress-step {state}">'
            f'<span class="studio-progress-marker">{marker}</span>'
            f'<span class="studio-progress-label">{label}</span>'
            "</div>"
        )

    st.markdown(
        f"""
        <section class="studio-app-header">
            <div class="studio-brand-row">
                <div>
                    <div class="studio-eyebrow">JAGRAN NEWSROOM TOOLS</div>
                    <h1>Partner Video Studio</h1>
                    <p>Build a broadcast-ready 1920x1080 story from partner footage.</p>
                </div>
                <div class="studio-session-status">
                    <span class="studio-status-dot"></span>
                    Local workspace
                </div>
            </div>
            <div class="studio-progress" aria-label="Production progress">
                {''.join(steps)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Partner Video Repackager", layout="wide")
    ensure_dirs()
    ffmpeg_ok = media_tools_healthy()
    st.markdown(
        """
        <style>
        :root {
            --studio-bg: #f3efe8;
            --studio-panel: rgba(255, 255, 255, 0.94);
            --studio-panel-2: #fbfaf7;
            --studio-border: rgba(40, 47, 58, 0.11);
            --studio-ink: #202734;
            --studio-muted: #697386;
            --studio-red: #d63b45;
            --studio-orange: #e87343;
            --studio-blue: #3977b9;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 10% 0%, rgba(214, 59, 69, 0.10), transparent 28rem),
                radial-gradient(circle at 94% 8%, rgba(232, 115, 67, 0.09), transparent 30rem),
                var(--studio-bg);
            color: var(--studio-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1.25rem; }
        .stMainBlockContainer {
            max-width: 1240px;
            padding: 2.25rem 2.2rem 5rem;
        }
        .studio-hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.25rem;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 24px;
            background:
                linear-gradient(125deg, rgba(214,59,69,.20), transparent 42%),
                linear-gradient(145deg, #252c38, #171d27);
            box-shadow: 0 24px 65px rgba(50,38,31,.16);
            margin-bottom: 1rem;
        }
        .studio-hero::after {
            content: "";
            position: absolute;
            width: 260px; height: 260px;
            right: -65px; top: -105px;
            border: 42px solid rgba(255,255,255,.035);
            border-radius: 50%;
        }
        .studio-eyebrow {
            color: #ffb0b4;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .18em;
            margin-bottom: .8rem;
        }
        .studio-hero h1 {
            max-width: 800px;
            margin: 0;
            color: #f8fafc;
            font-size: clamp(2rem, 3.4vw, 3.15rem);
            line-height: 1.02;
            letter-spacing: -.055em;
        }
        .studio-hero p {
            max-width: 680px;
            margin: 1rem 0 1.4rem;
            color: #b2bac8;
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .studio-pills { display: flex; flex-wrap: wrap; gap: .55rem; }
        .studio-pill {
            padding: .42rem .72rem;
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 999px;
            color: #dbe2ee;
            background: rgba(255,255,255,.045);
            font-size: .78rem;
            font-weight: 650;
        }
        .studio-stage-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 2rem 0 1.1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--studio-border);
        }
        .studio-stage-number {
            display: grid;
            place-items: center;
            width: 52px; height: 52px;
            flex: 0 0 52px;
            border-radius: 16px;
            color: white;
            background: linear-gradient(145deg, var(--studio-red), #c92e45);
            box-shadow: 0 12px 32px rgba(255,77,87,.2);
            font-size: .88rem;
            font-weight: 850;
            letter-spacing: .08em;
        }
        .studio-stage-kicker {
            color: var(--studio-red);
            font-size: .66rem;
            font-weight: 850;
            letter-spacing: .16em;
        }
        .studio-stage-header h2 {
            margin: .12rem 0 .12rem;
            color: var(--studio-ink);
            font-size: 1.55rem;
            letter-spacing: -.025em;
        }
        .studio-stage-header p {
            margin: 0;
            color: var(--studio-muted);
            font-size: .87rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            padding: 1.45rem;
            border: 1px dashed rgba(255,77,87,.42);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(214,59,69,.06), rgba(255,255,255,.82));
        }
        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid var(--studio-border);
            border-radius: 16px;
            background: var(--studio-panel);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--studio-border) !important;
            border-radius: 16px !important;
            background: var(--studio-panel);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {
            border-radius: 12px !important;
            background: #ffffff !important;
            border-color: rgba(40,47,58,.12) !important;
            color: var(--studio-ink) !important;
        }
        div[data-testid="stTextArea"] textarea { min-height: 250px; }
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            min-height: 2.75rem;
            border-radius: 12px;
            font-weight: 750;
            transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
        }
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            transform: translateY(-1px);
            border-color: rgba(255,77,87,.55);
            box-shadow: 0 10px 26px rgba(0,0,0,.22);
        }
        [data-testid="stButton"] button[kind="primary"] {
            border: 0;
            color: white;
            background: linear-gradient(95deg, var(--studio-red), var(--studio-orange));
            box-shadow: 0 14px 35px rgba(255,77,87,.23);
        }
        [data-testid="stAlert"] {
            border-radius: 14px;
            border-color: rgba(40,47,58,.09);
            box-shadow: 0 8px 22px rgba(50,38,31,.05);
        }
        [data-testid="stVideo"] {
            overflow: hidden;
            border: 1px solid var(--studio-border);
            border-radius: 18px;
            background: #020305;
            box-shadow: 0 18px 50px rgba(0,0,0,.28);
        }
        [data-testid="stVideo"] video {
            max-height: 540px;
            object-fit: contain;
            background: #020305;
        }
        [data-testid="stAudio"] { border-radius: 14px; overflow: hidden; }
        [role="tablist"] {
            position: sticky;
            top: .65rem;
            z-index: 50;
            display: grid !important;
            grid-template-columns: repeat(4, 1fr);
            gap: .55rem;
            padding: .5rem !important;
            margin: 1.15rem 0 .4rem;
            border: 1px solid var(--studio-border);
            border-radius: 18px;
            background: rgba(255,255,255,.88);
            box-shadow: 0 16px 40px rgba(50,38,31,.09);
            backdrop-filter: blur(16px);
        }
        [data-testid="stTab"] {
            justify-content: center;
            min-height: 3rem;
            padding: .65rem .8rem !important;
            border-radius: 12px !important;
            color: #697386 !important;
            font-weight: 760 !important;
        }
        [data-testid="stTab"][aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(95deg, var(--studio-red), var(--studio-orange)) !important;
            box-shadow: 0 10px 24px rgba(214,59,69,.20);
        }
        [data-testid="stTab"][aria-selected="true"] p { color: #ffffff !important; }
        [role="tablist"] [data-testid="stElementToolbar"] { display: none !important; }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4 {
            color: var(--studio-ink);
        }
        .studio-hero p, .studio-hero h1 { color: #f8fafc !important; }
        hr { border-color: var(--studio-border) !important; margin: 2.4rem 0 !important; }
        small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--studio-muted) !important; }
        @media (max-width: 850px) {
            .stMainBlockContainer { padding: 1.2rem 1rem 3rem; }
            .studio-hero { padding: 1.7rem 1.4rem; border-radius: 20px; }
            [role="tablist"] { grid-template-columns: repeat(2, 1fr); position: static; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        :root {
            --studio-bg: #f5f7fa;
            --studio-panel: #ffffff;
            --studio-panel-2: #f9fafb;
            --studio-border: #e3e7ee;
            --studio-border-strong: #cfd5df;
            --studio-ink: #18202c;
            --studio-muted: #687386;
            --studio-red: #d9273e;
            --studio-red-dark: #a8142a;
            --studio-green: #188967;
            --studio-focus: rgba(217, 39, 62, .16);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--studio-bg);
        }
        .stMainBlockContainer {
            max-width: 1320px;
            padding: 1.35rem 2rem 5rem;
        }
        .studio-app-header {
            overflow: hidden;
            margin-bottom: 1rem;
            border: 1px solid var(--studio-border);
            border-top: 4px solid var(--studio-red);
            border-radius: 14px;
            background: var(--studio-panel);
            box-shadow: 0 8px 28px rgba(24,32,44,.07);
        }
        .studio-brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.35rem 1.5rem 1.15rem;
        }
        .studio-app-header .studio-eyebrow {
            margin-bottom: .3rem;
            color: var(--studio-red);
            font-size: .68rem;
            font-weight: 850;
            letter-spacing: .14em;
        }
        .studio-app-header h1 {
            margin: 0;
            color: var(--studio-ink) !important;
            font-size: clamp(1.65rem, 2.3vw, 2.15rem);
            line-height: 1.1;
            letter-spacing: 0;
        }
        .studio-app-header p {
            margin: .35rem 0 0;
            color: var(--studio-muted) !important;
            font-size: .9rem;
        }
        .studio-session-status {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            flex: 0 0 auto;
            padding: .52rem .75rem;
            border: 1px solid #d9eee7;
            border-radius: 8px;
            color: #12664f;
            background: #f0faf6;
            font-size: .76rem;
            font-weight: 750;
        }
        .studio-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #19a576;
            box-shadow: 0 0 0 4px rgba(25,165,118,.12);
        }
        .studio-progress {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            border-top: 1px solid var(--studio-border);
            background: #fbfcfd;
        }
        .studio-progress-step {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: .48rem;
            min-width: 0;
            padding: .72rem .55rem;
            color: #8791a2;
            font-size: .75rem;
            font-weight: 700;
        }
        .studio-progress-step + .studio-progress-step {
            border-left: 1px solid var(--studio-border);
        }
        .studio-progress-marker {
            display: grid;
            place-items: center;
            width: 24px;
            height: 24px;
            flex: 0 0 24px;
            border: 1px solid #d8dde5;
            border-radius: 50%;
            background: #fff;
            font-size: .62rem;
        }
        .studio-progress-step.is-active {
            color: var(--studio-red-dark);
            background: #fff7f8;
        }
        .studio-progress-step.is-active::after {
            content: "";
            position: absolute;
            right: 20%;
            bottom: 0;
            left: 20%;
            height: 3px;
            border-radius: 3px 3px 0 0;
            background: var(--studio-red);
        }
        .studio-progress-step.is-active .studio-progress-marker {
            border-color: var(--studio-red);
            color: #fff;
            background: var(--studio-red);
        }
        .studio-progress-step.is-complete { color: #12664f; }
        .studio-progress-step.is-complete .studio-progress-marker {
            border-color: var(--studio-green);
            color: #fff;
            background: var(--studio-green);
        }
        .studio-stage-header {
            margin: 1.25rem 0 .85rem;
            padding: .9rem 1rem;
            border: 1px solid var(--studio-border);
            border-radius: 12px;
            background: var(--studio-panel);
            box-shadow: 0 4px 16px rgba(24,32,44,.04);
        }
        .studio-stage-number {
            width: 42px;
            height: 42px;
            flex-basis: 42px;
            border-radius: 10px;
            background: var(--studio-red);
            box-shadow: 0 8px 20px rgba(217,39,62,.18);
            font-size: .78rem;
            letter-spacing: 0;
        }
        .studio-stage-kicker { font-size: .61rem; letter-spacing: .12em; }
        .studio-stage-header h2 { font-size: 1.2rem; letter-spacing: 0; }
        [data-testid="stFileUploaderDropzone"] {
            padding: 1.2rem;
            border: 1px dashed #c8ced8;
            border-radius: 12px;
            background: #ffffff;
            transition: border-color .16s ease, background .16s ease, box-shadow .16s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--studio-red);
            background: #fffafb;
            box-shadow: 0 0 0 4px var(--studio-focus);
        }
        [data-testid="stFileUploaderDropzone"] button {
            border: 1px solid var(--studio-border-strong) !important;
            color: var(--studio-ink) !important;
            background: #ffffff !important;
            box-shadow: 0 2px 7px rgba(24,32,44,.06) !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            border-color: var(--studio-red) !important;
            color: var(--studio-red-dark) !important;
            background: #fff7f8 !important;
        }
        [data-testid="stFileUploaderDropzone"] button *,
        [data-testid="stFileUploaderDropzone"] button svg {
            color: inherit !important;
            fill: currentColor !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > div > span,
        [data-testid="stFileUploaderDropzoneInstructions"] > div > small,
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stFileUploaderDropzone"] small {
            color: var(--studio-muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderFileName"] {
            color: var(--studio-ink) !important;
        }
        [data-testid="stExpander"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            background: var(--studio-panel);
            box-shadow: 0 4px 14px rgba(24,32,44,.035);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {
            border-radius: 9px !important;
            border-color: var(--studio-border-strong) !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--studio-red) !important;
            box-shadow: 0 0 0 3px var(--studio-focus) !important;
        }
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            border-radius: 9px;
        }
        [data-testid="stButton"] button[kind="primary"] {
            background: var(--studio-red);
            box-shadow: 0 10px 24px rgba(217,39,62,.2);
        }
        [data-testid="stAlert"] { border-radius: 10px; }
        [data-testid="stVideo"] { border-radius: 12px; }
        [data-testid="stAudio"] { border-radius: 10px; }
        [role="tablist"] {
            gap: .3rem;
            padding: .35rem !important;
            margin: .8rem 0 .4rem;
            border-radius: 11px;
            background: rgba(255,255,255,.94);
            box-shadow: 0 8px 24px rgba(24,32,44,.07);
            backdrop-filter: blur(12px);
        }
        [data-testid="stTab"] {
            min-height: 2.65rem;
            padding: .55rem .75rem !important;
            border-radius: 8px !important;
        }
        [data-testid="stTab"][aria-selected="true"] {
            background: var(--studio-ink) !important;
            box-shadow: 0 6px 16px rgba(24,32,44,.18);
        }
        [data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid var(--studio-border);
            border-radius: 10px;
            background: var(--studio-panel);
        }
        @media (prefers-reduced-motion: no-preference) {
            .studio-app-header, .studio-stage-header {
                animation: studio-enter .28s ease-out both;
            }
            @keyframes studio-enter {
                from { opacity: 0; transform: translateY(5px); }
                to { opacity: 1; transform: translateY(0); }
            }
        }
        @media (max-width: 850px) {
            .stMainBlockContainer { padding: .85rem .75rem 3rem; }
            .studio-brand-row { align-items: flex-start; padding: 1rem; }
            .studio-session-status { display: none; }
            .studio-progress { grid-template-columns: repeat(3, 1fr); }
            .studio-progress-step:nth-child(4) { border-left: 0; }
            .studio-progress-step:nth-child(n+4) { border-top: 1px solid var(--studio-border); }
        }
        @media (max-width: 520px) {
            .studio-app-header h1 { font-size: 1.45rem; }
            .studio-progress-step {
                gap: .3rem;
                padding: .62rem .25rem;
                font-size: .66rem;
            }
            .studio-progress-marker {
                width: 21px;
                height: 21px;
                flex-basis: 21px;
            }
            .studio-stage-header { align-items: flex-start; }
            .studio-stage-header p { line-height: 1.45; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_workspace_header()

    if not ffmpeg_ok:
        st.warning("FFmpeg is currently unavailable. Upload, preview, and transcription will still work, but final video export requires repairing FFmpeg.")

    workspace_tabs = st.tabs(
        [
            "Story setup",
            "Voice studio",
            "Visual timeline",
            "Review & export",
        ]
    )

    with workspace_tabs[0]:
        render_stage_header(
            1,
            "Bring in the source",
            "Upload a local file or choose licensed Reuters/ANI footage.",
        )
        source_method = st.radio(
            "Video source",
            ["Upload raw video", "Reuters library", "ANI library"],
            horizontal=True,
            key="partner_source_method",
        )
        uploaded = None
        if source_method == "Upload raw video":
            uploaded = st.file_uploader(
                "Video file",
                type=["mp4", "mov", "m4v", "webm", "mkv"],
                label_visibility="collapsed",
            )
        else:
            provider = "Reuters" if source_method.startswith("Reuters") else "ANI"
            provider_config = newsroom_video_api_config(provider)
            if not provider_config["base_url"] or not provider_config["api_key"]:
                st.info(
                    f"{provider} library access requires your licensed API base URL "
                    f"and key. Configure `{provider.upper()}_VIDEO_API_BASE_URL` and "
                    f"`{provider.upper()}_VIDEO_API_KEY` in the server environment or "
                    ".streamlit/secrets.toml."
                )
            search_columns = st.columns([0.78, 0.22], vertical_alignment="bottom")
            provider_query = search_columns[0].text_input(
                f"Search {provider} videos",
                key=f"partner_{provider.lower()}_query",
                placeholder="Search by topic, location or slug",
            )
            if search_columns[1].button(
                "Search library",
                use_container_width=True,
                key=f"partner_{provider.lower()}_search",
                disabled=not (
                    provider_query.strip()
                    and provider_config["base_url"]
                    and provider_config["api_key"]
                ),
            ):
                with st.spinner(f"Searching your {provider} entitlement..."):
                    results, search_message = search_newsroom_videos(
                        provider, provider_query.strip()
                    )
                st.session_state[f"partner_{provider.lower()}_results"] = results
                st.session_state[f"partner_{provider.lower()}_message"] = search_message
            provider_results = st.session_state.get(
                f"partner_{provider.lower()}_results", []
            )
            provider_message = st.session_state.get(
                f"partner_{provider.lower()}_message"
            )
            if provider_message:
                st.caption(provider_message)
            for result_index, result in enumerate(provider_results):
                with st.container(border=True):
                    card_columns = st.columns([0.22, 0.58, 0.20], vertical_alignment="center")
                    if result.get("preview_url"):
                        card_columns[0].image(result["preview_url"], use_container_width=True)
                    else:
                        card_columns[0].caption(f"{provider} VIDEO")
                    card_columns[1].markdown(f"**{result['title']}**")
                    if result.get("description"):
                        card_columns[1].caption(result["description"][:320])
                    if result.get("duration"):
                        card_columns[1].caption(f"Duration: {result['duration']}")
                    if card_columns[2].button(
                        "Use this video",
                        use_container_width=True,
                        key=f"partner_use_{provider.lower()}_{result_index}_{result['id']}",
                        disabled=not bool(result.get("video_url")),
                    ):
                        with st.spinner(f"Importing the licensed {provider} video..."):
                            imported_path, import_message = download_newsroom_video(result)
                        if imported_path:
                            st.session_state["partner_video_path"] = str(imported_path)
                            st.session_state["partner_video_signature"] = (
                                f"{provider}:{result['id']}:{imported_path.stat().st_size}"
                            )
                            st.session_state["partner_video_source_label"] = provider
                            for dependent_key in (
                                "partner_transcript",
                                "partner_segments",
                                "partner_editable_transcript",
                                "partner_manual_script",
                                "partner_voiceover",
                                "partner_eleven_preview_bytes",
                                "partner_audio_preview_bytes",
                                "partner_latest_export",
                                "partner_latest_preview",
                            ):
                                st.session_state.pop(dependent_key, None)
                            st.success(import_message)
                            st.rerun()
                        st.error(import_message)
    
        source_path: Optional[Path] = None
        if uploaded:
            signature = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("partner_video_signature") != signature:
                source_path = save_upload(uploaded)
                st.session_state["partner_video_path"] = str(source_path)
                st.session_state["partner_video_signature"] = signature
                st.session_state["partner_video_source_label"] = "Local upload"
                st.session_state.pop("partner_transcript", None)
                st.session_state.pop("partner_segments", None)
                st.session_state.pop("partner_editable_transcript", None)
                st.session_state.pop("partner_manual_script", None)
                st.session_state.pop("partner_voiceover", None)
                st.session_state.pop("partner_voiceover_signature", None)
                st.session_state.pop("partner_image_overlays", None)
                st.session_state.pop("partner_removed_overlay_signatures", None)
                st.session_state.pop("partner_slug_enabled", None)
                st.session_state.pop("partner_slug_overlays", None)
                st.session_state.pop("partner_source_cuts", None)
                st.session_state.pop("partner_voice_pauses", None)
                st.session_state.pop("partner_voiceover_start", None)
                st.session_state.pop("partner_video_tail_mode", None)
                st.session_state.pop("partner_latest_export", None)
                st.session_state.pop("partner_latest_preview", None)
                st.session_state.pop("partner_latest_export_message", None)
                st.success(f"Uploaded: {source_path.name}")
            elif st.session_state.get("partner_video_path"):
                source_path = Path(st.session_state["partner_video_path"])
                st.success(f"Ready: {source_path.name}")
        elif st.session_state.get("partner_video_path"):
            existing_source = Path(str(st.session_state["partner_video_path"]))
            if existing_source.exists():
                source_path = existing_source
                source_label = st.session_state.get("partner_video_source_label") or "Selected source"
                st.success(f"{source_label}: {source_path.name}")
    
        if not source_path:
            st.info("Upload a partner video here to unlock the production workspace.")
            workspace_tabs[1].info(
                "Voice Studio unlocks after a source video and script are ready."
            )
            workspace_tabs[2].info(
                "Visual Timeline unlocks after the story setup is complete."
            )
            workspace_tabs[3].info(
                "Review & Export unlocks when the production has a source and script."
            )
            return
    
        if st.session_state.get("partner_transcription_pipeline") != TRANSCRIPTION_PIPELINE_VERSION:
            st.session_state["partner_transcription_pipeline"] = TRANSCRIPTION_PIPELINE_VERSION
            st.session_state.pop("partner_transcript", None)
            st.session_state.pop("partner_segments", None)
            st.session_state.pop("partner_editable_transcript", None)
    
        meta = probe_video(source_path)
        raw_video_duration = max(0.1, float(meta.get("duration") or 60.0))
        editor_video_duration = raw_video_duration
        if meta:
            st.caption(f"{int(meta.get('width', 0))}x{int(meta.get('height', 0))} · {compact_time(meta.get('duration', 0))} · {meta.get('fps', 0):.2f} fps")
        st.video(str(source_path))
    
        render_stage_header(
            2,
            "Prepare the story",
            "Generate a transcript from the footage or start with a newsroom-ready script.",
        )
        script_method = st.radio(
            "Choose method",
            ["Generate transcript from video", "Write script manually"],
            horizontal=True,
            label_visibility="collapsed",
        )
        transcript_controls = st.columns([0.7, 0.3], vertical_alignment="bottom")
        language_label = transcript_controls[0].selectbox(
            "Language",
            ["Hindi", "English", "Auto detect"],
            index=0,
            help="Hindi is the default so Hindi speech is written in Devanagari instead of Urdu script.",
        )
        generate_clicked = False
        if script_method == "Generate transcript from video":
            generate_clicked = transcript_controls[1].button(
                "Generate transcript",
                type="primary",
                use_container_width=True,
            )
        else:
            transcript_controls[1].caption("Type or paste the script below.")
    
        if generate_clicked:
            language_code = {
                "Hindi": "hi",
                "English": "en",
            }.get(language_label)
            spinner_text = (
                "Transcribing the complete audio with the local Hindi model..."
                if language_code == "hi"
                else "Transcribing the complete audio locally..."
            )
            with st.spinner(spinner_text):
                if language_code == "hi":
                    transcript, segments, message = transcribe_video_indic(source_path)
                elif platform.system() == "Darwin" and platform.machine() == "arm64":
                    transcript, segments, message = transcribe_video_mlx(source_path, language_code)
                else:
                    transcript, segments, message = transcribe_video(source_path, "base", language_code)
            if transcript:
                st.session_state["partner_transcript"] = transcript
                st.session_state["partner_segments"] = [segment.__dict__ for segment in segments]
                st.session_state["partner_editable_transcript"] = plain_transcript(segments)
                st.success(message)
            else:
                st.error(message)
    
        if script_method == "Write script manually" and "partner_manual_script" not in st.session_state:
            st.session_state["partner_manual_script"] = ""
    
        if script_method == "Generate transcript from video" and not st.session_state.get("partner_editable_transcript"):
            workspace_tabs[1].info(
                "Generate the transcript in Story Setup to continue to Voice Studio."
            )
            workspace_tabs[2].info(
                "The timeline becomes available after the transcript is generated."
            )
            workspace_tabs[3].info(
                "Complete the transcript before reviewing or exporting the video."
            )
            return
    
        render_stage_header(
            3,
            "Refine the transcript"
            if script_method == "Generate transcript from video"
            else "Write the narration",
            "Polish the language, punctuation and delivery before creating audio.",
        )
        editable_key = "partner_editable_transcript" if script_method == "Generate transcript from video" else "partner_manual_script"
        edited_transcript = st.text_area(
            "Transcript",
            key=editable_key,
            height=280,
            label_visibility="collapsed",
        )
        if edited_transcript.strip():
            if st.button(
                "Continue to Voice Studio →",
                type="primary",
                use_container_width=True,
                key="partner_continue_to_voice",
            ):
                components.html(
                    """
                    <script>
                    const tabs = window.parent.document.querySelectorAll('[role="tab"]');
                    const voiceTab = Array.from(tabs).find(
                        (tab) => tab.textContent.trim() === 'Voice studio'
                    );
                    if (voiceTab) voiceTab.click();
                    </script>
                    """,
                    height=0,
                )
    
    with workspace_tabs[1]:
        render_stage_header(
            4,
            "Direct the voice",
            "Choose a voice, set its pace and approve the audio before video rendering.",
        )
        voice_options = [
            "No voiceover — use original video audio",
            ELEVENLABS_VOICE_LABEL,
            "Clone producer voice from sample",
            "Upload completed voiceover",
            "Hindi test voice (Veena)",
            "English test voice (Samantha)",
        ]
        voice_choice = st.selectbox(
            "Voiceover",
            voice_options,
            index=0,
            label_visibility="collapsed",
        )
    
        voiceover_upload = None
        producer_consent = False
        reference_audio: Optional[Path] = None
        selected_producer_label: Optional[str] = None
        voice_script_for_generation = edited_transcript
        delivery_words_match = True
        voice_model_mode = "fast"
        voice_speed = 1.0
        selected_eleven_voice_label = next(iter(ELEVENLABS_VOICES))
        selected_eleven_voice_id = ELEVENLABS_VOICES[selected_eleven_voice_label]
        if voice_choice == "No voiceover — use original video audio":
            st.info(
                "Voiceover is optional. The original audio from the raw video "
                "will be retained unless an unmuted floating video is playing."
            )
        elif voice_choice == ELEVENLABS_VOICE_LABEL:
            selected_eleven_voice_label = st.selectbox(
                "ElevenLabs voice",
                list(ELEVENLABS_VOICES),
                key="partner_selected_elevenlabs_voice",
            )
            selected_eleven_voice_id = ELEVENLABS_VOICES[
                selected_eleven_voice_label
            ]
            voice_speed = st.slider(
                "Voice speed",
                min_value=0.7,
                max_value=1.2,
                value=1.0,
                step=0.05,
                key="partner_elevenlabs_voice_speed",
                format="%.2f×",
                help=(
                    "1.00× is the natural voice speed. Lower values speak more "
                    "slowly; higher values speak faster."
                ),
            )
            speed_description = "Natural"
            if voice_speed < 0.95:
                speed_description = "Slower"
            elif voice_speed > 1.05:
                speed_description = "Faster"
            selected_producer_label = selected_eleven_voice_label
            voice_script_for_generation = edited_transcript
            estimated_voice_seconds, estimated_word_count = estimate_voiceover_duration(
                voice_script_for_generation,
                voice_speed,
                "hi" if language_label == "Hindi" else "en",
            )
            duration_columns = st.columns(3)
            duration_columns[0].metric("Estimated length", compact_time(estimated_voice_seconds))
            duration_columns[1].metric("Script words", f"{estimated_word_count:,}")
            duration_columns[2].metric("Voice speed", f"{voice_speed:.2f}×")
            st.caption(
                f"{speed_description} delivery · estimate includes sentence and "
                "punctuation pauses. Generated audio may vary slightly."
            )
            st.info(
                "Engine: **ElevenLabs Multilingual v2** · pre-trained voice · "
                "one continuous generation for natural Hindi delivery."
            )
            if elevenlabs_api_key():
                st.success("ElevenLabs API key is configured on this server.")
            else:
                st.warning(
                    "Enter an ElevenLabs API key below to enable audio preview "
                    "and video generation. It is kept only for this app session."
                )
                st.text_input(
                    "ElevenLabs API key",
                    type="password",
                    key="partner_elevenlabs_api_key_input",
                    placeholder="Paste your ElevenLabs API key",
                    on_change=remember_elevenlabs_api_key,
                    help=(
                        "The key is masked and held only in Streamlit session "
                        "memory. It is not written to the project files."
                    ),
                )
            eleven_preview_signature = hashlib.sha256(
                (
                    f"{ELEVENLABS_MODEL_ID}:{selected_eleven_voice_id}:"
                    f"speed={voice_speed:.2f}:{voice_script_for_generation}"
                ).encode("utf-8")
            ).hexdigest()
            if (
                st.session_state.get("partner_eleven_preview_signature")
                != eleven_preview_signature
            ):
                st.session_state["partner_eleven_preview_signature"] = (
                    eleven_preview_signature
                )
                st.session_state.pop("partner_eleven_preview_bytes", None)
                st.session_state.pop("partner_eleven_preview_message", None)
                st.session_state.pop("partner_eleven_preview_duration", None)
    
            st.markdown("#### Test voice before video")
            st.caption(
                "The complete script is generated as one continuous reading. "
                "The same cached audio is then used in the final video."
            )
            if st.button(
                f"Generate audio preview with {selected_eleven_voice_label}",
                use_container_width=True,
                disabled=not (
                    elevenlabs_api_key()
                    and voice_script_for_generation.strip()
                    and len(re.sub(r"\s+", " ", voice_script_for_generation).strip())
                    <= 9_500
                ),
                key="partner_generate_elevenlabs_preview",
            ):
                with st.spinner("Generating the ElevenLabs audio preview..."):
                    preview_path, preview_message = create_elevenlabs_voiceover(
                        voice_script_for_generation,
                        selected_eleven_voice_id,
                        voice_speed,
                    )
                if preview_path:
                    st.session_state["partner_eleven_preview_bytes"] = (
                        preview_path.read_bytes()
                    )
                    st.session_state["partner_eleven_preview_message"] = preview_message
                    st.session_state["partner_eleven_preview_duration"] = (
                        probe_media_duration(preview_path)
                    )
                else:
                    st.error(preview_message)
    
            eleven_preview = st.session_state.get("partner_eleven_preview_bytes")
            if eleven_preview:
                st.audio(eleven_preview, format="audio/mpeg")
                exact_preview_duration = float(
                    st.session_state.get("partner_eleven_preview_duration") or 0.0
                )
                if exact_preview_duration > 0:
                    st.caption(
                        f"Exact generated length: **{compact_time(exact_preview_duration)}** "
                        f"({exact_preview_duration:.1f} seconds)"
                    )
                st.success(
                    st.session_state.get("partner_eleven_preview_message")
                    or "ElevenLabs audio preview generated."
                )
        elif voice_choice == "Clone producer voice from sample":
            reference_choice = st.selectbox(
                "Producer voice",
                [
                    *BUILTIN_PRODUCER_VOICES,
                    DEEPIKA_F5_LABEL,
                    "Upload another producer sample",
                ],
                key="partner_selected_producer_voice",
            )
            selected_producer_label = reference_choice
            reference_upload = None
            is_deepika_f5 = reference_choice == DEEPIKA_F5_LABEL
            if is_deepika_f5:
                voice_model_mode = "deepika-f5-standard-reference-v10"
                st.info(
                    "Engine: **Deepika fine-tuned Hindi F5 pilot** · one-epoch "
                    "experimental checkpoint · separate from the older Chatterbox clone."
                )
                st.caption(
                    "Generate an audio preview first. This pilot will not be used for "
                    "the final video unless you keep this voice selected."
                )
            else:
                generation_mode_label = st.radio(
                    "Voice generation mode",
                    ["Fast natural (recommended)", "Maximum fidelity"],
                    horizontal=True,
                    key="partner_voice_generation_mode",
                    help=(
                        "Fast natural uses the official 4-bit Apple MLX model. "
                        "Maximum fidelity uses FP16 and takes longer for a new script."
                    ),
                )
                voice_model_mode = (
                    "fast" if generation_mode_label.startswith("Fast") else "maximum"
                )
            if not is_deepika_f5 and voice_model_mode == "fast":
                st.caption(
                    "Uses the optimized 4-bit Apple MLX path. A new long script still "
                    "requires synthesis once; repeating the same script is instant."
                )
            elif not is_deepika_f5:
                st.caption(
                    "Uses FP16 for maximum voice fidelity. New long scripts take "
                    "considerably longer to synthesize locally."
                )
            if is_deepika_f5:
                reference_audio = (
                    DEEPIKA_F5_SOURCE if DEEPIKA_F5_SOURCE.exists() else None
                )
                selected_profile = PRODUCER_VOICE_PROFILES["Deepika"]
                st.caption(
                    "Delivery profile: the trained Deepika checkpoint with sentence-level "
                    "neutral, serious/assertive and interrogative Deepika references."
                )
                reference_signature = "deepika-f5-pilot-epoch1-v1"
                if st.session_state.get("producer_reference_signature") != reference_signature:
                    st.session_state["producer_reference_signature"] = reference_signature
                    st.session_state["producer_reference_path"] = str(DEEPIKA_F5_SOURCE)
                    st.session_state.pop("producer_clone_voice_id", None)
            elif reference_choice in BUILTIN_PRODUCER_VOICES:
                selected_reference = BUILTIN_PRODUCER_VOICES[reference_choice]
                reference_audio = selected_reference if selected_reference.exists() else None
                selected_profile = PRODUCER_VOICE_PROFILES[reference_choice]
                st.caption(
                    f"Delivery profile: approximately {selected_profile['words_per_minute']} words/minute "
                    f"with {float(selected_profile['pause_seconds']):.2f}-second phrase pauses."
                )
                reference_signature = f"builtin-{reference_choice.lower()}-delivery-v2"
                if st.session_state.get("producer_reference_signature") != reference_signature:
                    st.session_state["producer_reference_signature"] = reference_signature
                    st.session_state["producer_reference_path"] = str(selected_reference)
                    st.session_state.pop("producer_clone_voice_id", None)
            else:
                selected_producer_label = "Uploaded producer sample"
                reference_upload = st.file_uploader(
                    "Producer reference audio",
                    type=["wav", "mp3", "m4a", "aac", "aiff", "aif"],
                    help="Use a clean, single-speaker recording. A longer sample generally produces a more faithful voice.",
                )
            producer_consent = st.checkbox(
                "I confirm that the producer has authorized this voice to be cloned and used for this script."
            )
            if reference_upload:
                reference_signature = f"{reference_upload.name}:{reference_upload.size}"
                if st.session_state.get("producer_reference_signature") != reference_signature:
                    reference_audio = save_reference_voice_upload(reference_upload)
                    st.session_state["producer_reference_path"] = str(reference_audio)
                    st.session_state["producer_reference_signature"] = reference_signature
                    st.session_state.pop("producer_clone_voice_id", None)
                elif st.session_state.get("producer_reference_path"):
                    reference_audio = Path(st.session_state["producer_reference_path"])
            if reference_audio and reference_audio.exists():
                st.info(f"Selected voice for generation: **{selected_producer_label}**")
                if not is_deepika_f5:
                    st.audio(str(reference_audio))
            voice_language_code = "hi" if language_label == "Hindi" else "en"
            delivery_source_signature = hashlib.sha256(
                f"{voice_language_code}:{edited_transcript}".encode("utf-8")
            ).hexdigest()
            if st.session_state.get("partner_delivery_source_signature") != delivery_source_signature:
                st.session_state["partner_delivery_source_signature"] = delivery_source_signature
                st.session_state["partner_voice_delivery_script"] = prepare_script_for_delivery(
                    edited_transcript,
                    voice_language_code,
                )
            with st.expander("Review voice delivery punctuation", expanded=False):
                st.caption(
                    "Punctuation controls breathing, questions and emphasis. "
                    "You may adjust punctuation, but keep the spoken words unchanged."
                )
                voice_script_for_generation = st.text_area(
                    "Voice delivery script",
                    key="partner_voice_delivery_script",
                    height=220,
                    label_visibility="collapsed",
                )
                delivery_words_match = (
                    normalized_script_identity(voice_script_for_generation)
                    == normalized_script_identity(edited_transcript)
                )
                if not delivery_words_match:
                    st.error(
                        "The delivery version must contain the same words as the transcript. "
                        "Change punctuation only."
                    )
    
            preview_ready = bool(
                reference_audio
                and reference_audio.exists()
                and producer_consent
                and delivery_words_match
                and voice_script_for_generation.strip()
                and (not is_deepika_f5 or deepika_f5_is_ready())
            )
            preview_reference_identity = ""
            if reference_audio and reference_audio.exists():
                preview_reference_stat = reference_audio.stat()
                preview_reference_identity = (
                    f"{reference_audio.resolve()}:{preview_reference_stat.st_size}:"
                    f"{preview_reference_stat.st_mtime_ns}"
                )
            preview_signature = hashlib.sha256(
                (
                    f"{preview_reference_identity}:{voice_model_mode}:"
                    f"{voice_language_code}:{voice_script_for_generation}"
                ).encode("utf-8")
            ).hexdigest()
            if st.session_state.get("partner_audio_preview_signature") != preview_signature:
                st.session_state.pop("partner_audio_preview_bytes", None)
                st.session_state.pop("partner_audio_preview_message", None)
                st.session_state["partner_audio_preview_signature"] = preview_signature
    
            st.markdown("#### Test voice before video")
            st.caption(
                "Generate and review only the selected producer voice. The preview is "
                "kept in this app session and its temporary WAV is deleted immediately."
            )
            if st.button(
                f"Generate audio preview with {selected_producer_label or 'selected producer'}",
                use_container_width=True,
                disabled=not preview_ready,
                key="partner_generate_audio_preview",
            ):
                with st.spinner(
                    f"Generating an audio-only preview with "
                    f"{selected_producer_label or 'the selected producer'}..."
                ):
                    if is_deepika_f5:
                        preview_bytes, preview_message = create_deepika_f5_preview(
                            voice_script_for_generation,
                        )
                    else:
                        preview_bytes, preview_message = create_local_cloned_voiceover_preview(
                            voice_script_for_generation,
                            reference_audio,
                            voice_language_code,
                            voice_model_mode,
                        )
                if preview_bytes:
                    st.session_state["partner_audio_preview_bytes"] = preview_bytes
                    st.session_state["partner_audio_preview_message"] = preview_message
                else:
                    st.error(preview_message)
    
            session_preview = st.session_state.get("partner_audio_preview_bytes")
            if session_preview:
                st.audio(session_preview, format="audio/wav")
                st.success(
                    st.session_state.get("partner_audio_preview_message")
                    or "Audio preview generated."
                )
        elif voice_choice == "Upload completed voiceover":
            voiceover_upload = st.file_uploader(
                "Voiceover audio",
                type=["wav", "mp3", "m4a", "aac", "aiff", "aif"],
                label_visibility="collapsed",
            )
            if voiceover_upload:
                voiceover_signature = f"{voiceover_upload.name}:{voiceover_upload.size}"
                if st.session_state.get("partner_voiceover_signature") != voiceover_signature:
                    saved_voiceover = save_voiceover_upload(voiceover_upload)
                    st.session_state["partner_voiceover"] = str(saved_voiceover)
                    st.session_state["partner_voiceover_signature"] = voiceover_signature
    
        uploaded_voiceover = Path(st.session_state["partner_voiceover"]) if st.session_state.get("partner_voiceover") else None
        if voice_choice == "Upload completed voiceover" and uploaded_voiceover and uploaded_voiceover.exists():
            st.audio(str(uploaded_voiceover))

        voice_workspace_ready = bool(
            voice_choice == "No voiceover — use original video audio"
            or
            st.session_state.get("partner_eleven_preview_bytes")
            or st.session_state.get("partner_audio_preview_bytes")
            or (uploaded_voiceover and uploaded_voiceover.exists())
            or voice_choice
            in {"Hindi test voice (Veena)", "English test voice (Samantha)"}
        )
        st.divider()
        if st.button(
            "Continue to Visual Timeline →",
            type="primary",
            use_container_width=True,
            disabled=not voice_workspace_ready,
            key="partner_continue_to_timeline",
            help=(
                None
                if voice_workspace_ready
                else "Generate, upload, or skip the voiceover before continuing."
            ),
        ):
            components.html(
                """
                <script>
                const tabs = window.parent.document.querySelectorAll('[role="tab"]');
                const timelineTab = Array.from(tabs).find(
                    (tab) => tab.textContent.trim() === 'Visual timeline'
                );
                if (timelineTab) timelineTab.click();
                </script>
                """,
                height=0,
            )

    with workspace_tabs[2]:
        render_stage_header(
            5,
            "Build the rendered-video layout",
            "Arrange the PNG, slug, logo, raw video and looping-image panel.",
        )
        template_layout = "two_column"
        st.caption(
            "The top PNG and slug banner are separate canvas layers. Move and "
            "resize either one independently."
        )
        selected_property = st.selectbox(
            "Property logo",
            list(PROPERTY_LOGO_FILES),
            key="partner_selected_property",
            help=f"Logo files are loaded locally from {PROPERTY_LOGO_DIR}.",
        )
        selected_logo_path = property_logo_path(selected_property)
        if selected_logo_path:
            st.caption(f"Using local {selected_property} logo: {selected_logo_path.name}")
        else:
            expected_logo = PROPERTY_LOGO_DIR / PROPERTY_LOGO_FILES[selected_property]
            st.warning(
                f"Add the {selected_property} logo at `{expected_logo}`. The layout "
                "will work without it until that file is available."
            )
        template_columns = st.columns([0.42, 0.58], vertical_alignment="top")
        template_header_upload = template_columns[0].file_uploader(
            "Top PNG",
            type=["png"],
            key="partner_template_header_upload",
            help="This PNG occupies the left 20% of the top strip.",
        )
        template_loop_uploads = template_columns[1].file_uploader(
            "Floating images and videos",
            type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v", "webm", "mkv"],
            accept_multiple_files=True,
            key="partner_template_loop_uploads",
            help="Images and videos play in order and repeat for the complete output.",
        )
        template_photo_seconds = st.number_input(
            "Default seconds per floating item",
            min_value=0.5,
            max_value=60.0,
            value=5.0,
            step=0.5,
            key="partner_template_photo_seconds",
        )
        if template_header_upload:
            header_signature = (
                f"{template_header_upload.name}:{template_header_upload.size}"
            )
            if st.session_state.get("partner_template_header_signature") != header_signature:
                header_path = save_overlay_upload(
                    template_header_upload, time.time_ns()
                )
                st.session_state["partner_template_header_signature"] = header_signature
                st.session_state["partner_template_header_path"] = str(header_path)
        loop_signature = tuple(
            f"{item.name}:{item.size}" for item in (template_loop_uploads or [])
        )
        if loop_signature and (
            st.session_state.get("partner_template_loop_signature") != loop_signature
            or not st.session_state.get("partner_template_loop_items")
        ):
            loop_items = []
            for position, uploaded_media in enumerate(template_loop_uploads or []):
                saved_media = save_overlay_upload(uploaded_media, time.time_ns() + position)
                media_type = "video" if saved_media.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm", ".mkv"} else "image"
                clip_duration = probe_media_duration(saved_media) if media_type == "video" else 0.0
                loop_items.append({
                    "path": str(saved_media), "name": uploaded_media.name,
                    "media_type": media_type, "clip_duration": clip_duration,
                    "use_clip_audio": False,
                })
            st.session_state["partner_template_loop_signature"] = loop_signature
            st.session_state["partner_template_loop_items"] = loop_items

        template_loop_items = [
            dict(item) for item in st.session_state.get("partner_template_loop_items", [])
            if Path(str(item.get("path") or "")).is_file()
        ]
        for media_index, media_item in enumerate(template_loop_items):
            if media_item.get("media_type") != "video":
                continue
            media_path = Path(str(media_item["path"]))
            has_audio = media_has_audio(str(media_path), media_path.stat().st_mtime_ns)
            media_item["use_clip_audio"] = st.toggle(
                f"Play audio from floating video: {media_item.get('name') or media_path.name}",
                value=bool(media_item.get("use_clip_audio")) and has_audio,
                disabled=not has_audio,
                key=f"partner_template_media_audio_{media_index}_{media_path.name}",
                help="While this audio plays, raw-video or voiceover audio is muted automatically.",
            )
        st.session_state["partner_template_loop_items"] = template_loop_items

        template_header_path_for_editor = Path(
            str(st.session_state.get("partner_template_header_path") or "")
        )
        template_loop_paths_for_editor = [Path(str(item["path"])) for item in template_loop_items]
        default_canvas_layout = [
            {"id": "header_image", "x": 0.01, "y": 0.01, "w": 0.20, "h": 0.18, "z": 3, "start": 0.0, "duration": editor_video_duration},
            {"id": "slug", "x": 0.22, "y": 0.01, "w": 0.77, "h": 0.18, "z": 3, "start": 0.0, "duration": editor_video_duration},
            {"id": "source", "x": 0.0, "y": 0.20, "w": 0.50, "h": 0.80, "z": 1, "start": 0.0, "duration": editor_video_duration},
            {"id": "images", "x": 0.50, "y": 0.20, "w": 0.50, "h": 0.80, "z": 1, "start": 0.0, "duration": editor_video_duration},
        ]
        if selected_logo_path:
            default_canvas_layout.append(
                {"id": "logo", "x": 0.84, "y": 0.035, "w": 0.13, "h": 0.11, "z": 5, "start": 0.0, "duration": editor_video_duration}
            )
        current_canvas_layout = st.session_state.get(
            "partner_template_canvas_layout", default_canvas_layout
        )
        current_canvas_ids = {str(item.get("id")) for item in current_canvas_layout}
        for default_item in default_canvas_layout:
            if default_item["id"] not in current_canvas_ids:
                current_canvas_layout.append(default_item)
        preview_header_slug = {
            "text": "Slug / headline preview",
            "style": "Jagran Red",
            "region": "template_header",
            "geometry": next(
                (item for item in current_canvas_layout if item.get("id") == "slug"),
                default_canvas_layout[1],
            ),
        }
        preview_header_asset = build_slug_overlay_asset(preview_header_slug, source_path)
        canvas_images = [
            {"id": "slug", "name": "Slug banner", "kind": "image", "src": image_preview_data_url(str(preview_header_asset), preview_header_asset.stat().st_mtime_ns), "start": 0.0, "duration": editor_video_duration},
            {"id": "source", "name": "Raw video", "kind": "video", "src": video_preview_data_url(str(source_path), source_path.stat().st_mtime_ns), "start": 0.0, "duration": editor_video_duration},
        ]
        if template_header_path_for_editor.is_file():
            canvas_images.insert(
                0,
                {"id": "header_image", "name": "Top PNG", "kind": "image", "src": image_preview_data_url(str(template_header_path_for_editor), template_header_path_for_editor.stat().st_mtime_ns), "start": 0.0, "duration": editor_video_duration},
            )
        if template_loop_paths_for_editor:
            loop_preview_path = template_loop_paths_for_editor[0]
            first_floating_type = str(template_loop_items[0].get("media_type") or "image")
            loop_preview_src = (
                video_preview_data_url(str(loop_preview_path), loop_preview_path.stat().st_mtime_ns)
                if first_floating_type == "video"
                else image_preview_data_url(str(loop_preview_path), loop_preview_path.stat().st_mtime_ns)
            )
        else:
            first_floating_type = "image"
            loop_preview_src = ""
        canvas_images.append(
            {"id": "images", "name": "Floating media", "kind": first_floating_type, "src": loop_preview_src, "start": 0.0, "duration": editor_video_duration}
        )
        if selected_logo_path:
            canvas_images.append(
                {"id": "logo", "name": f"{selected_property} logo", "kind": "image", "src": image_preview_data_url(str(selected_logo_path), selected_logo_path.stat().st_mtime_ns), "start": 0.0, "duration": editor_video_duration}
            )
        st.markdown("**Arrange the video canvas**")
        st.caption(
            "Drag any component to reposition it. Resize it from the lower-right "
            "corner; these exact proportions are used in the final render."
        )
        template_canvas_result = overlay_layout_editor(
            images=canvas_images,
            layout=current_canvas_layout,
            background="",
            video_duration=editor_video_duration,
            spatial_only=True,
            default={"items": current_canvas_layout},
            key=f"partner_template_canvas_{st.session_state.get('partner_video_signature', source_path.name)}_{selected_property}",
        )
        if isinstance(template_canvas_result, dict) and template_canvas_result.get("items"):
            st.session_state["partner_template_canvas_layout"] = template_canvas_result["items"]

        with st.expander("Advanced timing and source edits", expanded=False):
            st.caption(
                "Optional controls for removing source sections and fine-tuning "
                "narration timing remain available below."
            )
        st.markdown("**A. Remove sections from the uploaded raw video**")
        st.caption(
            f"The timestamps in this section refer only to the original upload "
            f"({source_path.name}) shown in Step 1—not to the generated video. "
            "Cuts are completed before voiceover, images, video bytes or slugs are composed."
        )
        if "partner_source_cuts" not in st.session_state:
            st.session_state["partner_source_cuts"] = []
        source_cut_items: List[Dict[str, object]] = st.session_state[
            "partner_source_cuts"
        ]
        cut_header_columns = st.columns([0.72, 0.28], vertical_alignment="center")
        cut_header_columns[0].caption(
            f"Raw video length: {compact_time(raw_video_duration)} "
            f"({raw_video_duration:.2f} seconds)"
        )
        if cut_header_columns[1].button(
            "＋ Remove section",
            use_container_width=True,
            key="partner_add_source_cut",
        ):
            previous_end = max(
                (float(item.get("end") or 0.0) for item in source_cut_items),
                default=0.0,
            )
            cut_start = min(previous_end, max(0.0, raw_video_duration - 1.0))
            source_cut_items.append(
                {
                    "id": time.time_ns(),
                    "start": cut_start,
                    "end": min(raw_video_duration, cut_start + 1.0),
                    "to_end": False,
                }
            )
            st.session_state["partner_source_cuts"] = source_cut_items
            st.rerun()
    
        remove_cut_id: Optional[str] = None
        for cut_index, cut in enumerate(source_cut_items):
            cut_id = str(cut["id"])
            cut_start_value = clamp_float(
                float(cut.get("start") or 0.0), 0.0, max(0.0, raw_video_duration - 0.1)
            )
            cut_end_value = clamp_float(
                float(cut.get("end") or cut_start_value + 1.0),
                cut_start_value + 0.1,
                raw_video_duration,
            )
            with st.expander(
                f"Remove {compact_time(cut_start_value)}–{compact_time(cut_end_value)}",
                expanded=len(source_cut_items) == 1,
            ):
                remove_through_end = st.checkbox(
                    "Continue this removal through the end of the uploaded video",
                    value=bool(cut.get("to_end") or False),
                    key=f"partner_cut_to_end_{cut_id}",
                    help=(
                        "Use this when the unwanted section is at the end. It prevents "
                        "a fractional final second from remaining in the edited base."
                    ),
                )
                cut_columns = st.columns([1, 1, 0.32], vertical_alignment="bottom")
                cut_start = cut_columns[0].number_input(
                    "From raw video (seconds)",
                    min_value=0.0,
                    max_value=max(0.0, raw_video_duration - 0.1),
                    value=cut_start_value,
                    step=0.1,
                    key=f"partner_cut_start_{cut_id}",
                )
                if remove_through_end:
                    cut_columns[1].text_input(
                        "To raw video (seconds)",
                        value=f"{raw_video_duration:.2f} · End of upload",
                        disabled=True,
                        key=f"partner_cut_end_display_{cut_id}",
                    )
                    cut_end = raw_video_duration
                else:
                    cut_end = cut_columns[1].number_input(
                        "To raw video (seconds)",
                        min_value=min(raw_video_duration, float(cut_start) + 0.1),
                        max_value=raw_video_duration,
                        value=max(
                            cut_end_value,
                            min(raw_video_duration, float(cut_start) + 0.1),
                        ),
                        step=0.1,
                        key=f"partner_cut_end_{cut_id}",
                    )
                if cut_columns[2].button(
                    "Remove",
                    key=f"partner_delete_cut_{cut_id}",
                    use_container_width=True,
                ):
                    remove_cut_id = cut_id
                cut["start"] = float(cut_start)
                cut["end"] = float(cut_end)
                cut["to_end"] = bool(remove_through_end)
    
        if remove_cut_id is not None:
            st.session_state["partner_source_cuts"] = [
                item for item in source_cut_items if str(item["id"]) != remove_cut_id
            ]
            st.rerun()
        st.session_state["partner_source_cuts"] = source_cut_items
        source_cuts_for_export = normalise_cut_ranges(
            source_cut_items, raw_video_duration
        )
        kept_ranges = kept_source_ranges(source_cuts_for_export, raw_video_duration)
        edited_source_duration = sum(end - start for start, end in kept_ranges)
        if not kept_ranges:
            st.error("The selected removals delete the entire raw video. Keep at least 0.1 seconds.")
            return
        removed_duration = raw_video_duration - edited_source_duration
        st.success(
            f"Raw-video base after cuts: {compact_time(edited_source_duration)}"
            + (
                f" · {removed_duration:.1f} seconds removed"
                if removed_duration > 0.01
                else " · no sections removed"
            )
        )
    
        st.markdown("**When narration is longer than the edited raw video**")
        tail_mode_labels = {
            "End with the edited raw video (no loop)": "end",
            "Continue on a black background (no loop)": "black",
            "Loop the edited raw video": "loop",
        }
        selected_tail_mode_label = st.selectbox(
            "End-of-video behaviour",
            list(tail_mode_labels),
            index=0,
            key="partner_video_tail_mode",
            help=(
                "Looping is never automatic. Select it only when you intentionally "
                "want the edited raw footage to repeat."
            ),
        )
        video_tail_mode = tail_mode_labels[selected_tail_mode_label]
        if video_tail_mode == "end":
            st.caption(
                "The final video ends with the edited raw footage. Any remaining "
                "voiceover after that point is not included."
            )
        elif video_tail_mode == "black":
            st.caption(
                "The complete voiceover is retained, with a black background after "
                "the edited raw footage ends."
            )
        else:
            st.caption(
                "Only the edited raw-video base repeats; removed sections remain excluded."
            )
    
        st.markdown("**Editor audio timeline**")
        st.caption(
            "Set narration start and pause windows directly alongside the canvas. "
            "After every pause, voiceover resumes from the exact point where it stopped."
        )
        voiceover_enabled = voice_choice != "No voiceover — use original video audio"
        if not voiceover_enabled:
            st.info("Voiceover is off, so the raw video's original audio remains active.")
        voiceover_start = st.number_input(
            "Voiceover starts at final-video timestamp (seconds)",
            min_value=0.0,
            max_value=86400.0,
            value=float(st.session_state.get("partner_voiceover_start") or 0.0),
            step=0.1,
            key="partner_voiceover_start",
            help="Use 0 to start narration immediately, or enter a delay.",
        )
        if "partner_voice_pauses" not in st.session_state:
            st.session_state["partner_voice_pauses"] = []
        voice_pause_items: List[Dict[str, object]] = st.session_state[
            "partner_voice_pauses"
        ]
        existing_pause_ends = [
            float(item.get("start") or voiceover_start)
            + max(0.1, float(item.get("duration") or 0.1))
            for item in voice_pause_items
        ]
        pause_timeline_end = max(
            float(voiceover_start) + 1.0,
            float(voiceover_start) + float(edited_source_duration),
            *existing_pause_ends,
        )
        pause_header_columns = st.columns([0.72, 0.28], vertical_alignment="center")
        pause_header_columns[0].caption(
            "Add a pause window only when needed. Choose where narration pauses and resumes."
        )
        with pause_header_columns[1].popover(
            "＋ Add pause window",
            use_container_width=True,
            disabled=not voiceover_enabled,
        ):
            st.caption("Drag both ends to define the silent window.")
            suggested_pause_start = min(
                max(existing_pause_ends, default=float(voiceover_start)),
                pause_timeline_end - 0.1,
            )
            suggested_pause_end = min(
                pause_timeline_end,
                suggested_pause_start + 1.0,
            )
            new_pause_window = st.slider(
                "Pause from → Resume at",
                min_value=float(voiceover_start),
                max_value=float(pause_timeline_end),
                value=(float(suggested_pause_start), float(suggested_pause_end)),
                step=0.1,
                key="partner_new_voice_pause_window",
                format="%.1f sec",
            )
            st.caption(
                f"Pause: {compact_time(new_pause_window[0])} → "
                f"Resume: {compact_time(new_pause_window[1])}"
            )
            if st.button(
                "Add this pause window",
                type="primary",
                use_container_width=True,
                key="partner_confirm_voice_pause",
                disabled=new_pause_window[1] <= new_pause_window[0],
            ):
                voice_pause_items.append(
                    {
                        "id": time.time_ns(),
                        "start": float(new_pause_window[0]),
                        "duration": float(new_pause_window[1] - new_pause_window[0]),
                        "insert_mode": "silent",
                        "insert_script": "",
                    }
                )
                st.session_state["partner_voice_pauses"] = voice_pause_items
                st.session_state.pop("partner_new_voice_pause_window", None)
                st.rerun()
    
        remove_pause_id: Optional[str] = None
        for pause_index, pause in enumerate(voice_pause_items):
            pause_id = str(pause["id"])
            pause_start = max(
                float(voiceover_start),
                float(pause.get("start") or voiceover_start),
            )
            pause_end = pause_start + max(0.1, float(pause.get("duration") or 0.1))
            pause_columns = st.columns([1, 0.2], vertical_alignment="bottom")
            pause_window = pause_columns[0].slider(
                f"Pause window {pause_index + 1}: Pause from → Resume at",
                min_value=float(voiceover_start),
                max_value=float(max(pause_timeline_end, pause_end)),
                value=(float(pause_start), float(pause_end)),
                step=0.1,
                key=f"partner_voice_pause_window_{pause_id}",
                format="%.1f sec",
            )
            if pause_columns[1].button(
                "Remove",
                key=f"partner_delete_voice_pause_{pause_id}",
                use_container_width=True,
            ):
                remove_pause_id = pause_id
            pause_columns[0].caption(
                f"Narration pauses at {compact_time(pause_window[0])} and resumes at "
                f"{compact_time(pause_window[1])}."
            )
            pause["start"] = float(pause_window[0])
            pause["duration"] = float(pause_window[1] - pause_window[0])
            with st.container(border=True):
                st.markdown(f"**Optional script and audio during pause {pause_index + 1}**")
                pause_insert_labels = {"Keep this pause silent": "silent"}
                if voice_choice != "Upload completed voiceover":
                    pause_insert_labels["Generate audio from a separate script"] = (
                        "generated"
                    )
                pause_insert_labels["Upload prepared audio"] = "upload"
                current_insert_mode = str(pause.get("insert_mode") or "silent")
                current_insert_label = next(
                    (
                        label
                        for label, mode in pause_insert_labels.items()
                        if mode == current_insert_mode
                    ),
                    "Keep this pause silent",
                )
                insert_mode_label = st.selectbox(
                    "What should play while the main voiceover is paused?",
                    list(pause_insert_labels),
                    index=list(pause_insert_labels).index(current_insert_label),
                    key=f"partner_pause_insert_mode_{pause_id}",
                )
                insert_mode = pause_insert_labels[insert_mode_label]
                insert_script = ""
                if insert_mode in {"generated", "upload"}:
                    insert_script = st.text_area(
                        (
                            "Separate script to generate"
                            if insert_mode == "generated"
                            else "Script/reference text (optional)"
                        ),
                        value=str(pause.get("insert_script") or ""),
                        key=f"partner_pause_insert_script_{pause_id}",
                        placeholder="Enter the additional narration for this pause window",
                        height=100,
                    )
                if insert_mode == "generated":
                    st.caption(
                        "This script will use the currently selected voice. The main "
                        "voiceover resumes when the pause window ends."
                    )
                    if not insert_script.strip():
                        st.warning("Enter the separate script for this pause.")
                elif insert_mode == "upload":
                    pause_audio_upload = st.file_uploader(
                        "Upload audio for this pause",
                        type=["wav", "mp3", "m4a", "aac", "aiff", "aif"],
                        key=f"partner_pause_insert_upload_{pause_id}",
                    )
                    if pause_audio_upload:
                        audio_signature = (
                            f"{pause_audio_upload.name}:{pause_audio_upload.size}"
                        )
                        if pause.get("insert_audio_signature") != audio_signature:
                            saved_pause_audio = save_pause_audio_upload(
                                pause_audio_upload,
                                pause_id,
                            )
                            pause["insert_audio_path"] = str(saved_pause_audio)
                            pause["insert_audio_signature"] = audio_signature
                    saved_audio_value = pause.get("insert_audio_path")
                    if saved_audio_value and Path(str(saved_audio_value)).exists():
                        st.audio(str(saved_audio_value))
                    else:
                        st.warning("Upload the audio that should play during this pause.")
                pause["insert_mode"] = insert_mode
                pause["insert_script"] = insert_script.strip()
    
        if remove_pause_id is not None:
            st.session_state["partner_voice_pauses"] = [
                item for item in voice_pause_items if str(item["id"]) != remove_pause_id
            ]
            st.rerun()
        st.session_state["partner_voice_pauses"] = voice_pause_items
        voice_pauses_for_export = (
            normalise_voice_pauses(voice_pause_items, float(voiceover_start))
            if voiceover_enabled
            else []
        )
        voice_timing_for_export: Dict[str, object] = {
            "start": float(voiceover_start),
            "pauses": [
                {"start": start, "duration": duration}
                for start, duration in voice_pauses_for_export
            ],
        }
        if voice_pauses_for_export:
            st.info(
                "Narration pauses: "
                + " · ".join(
                    f"{compact_time(start)} → {compact_time(start + duration)}"
                    for start, duration in voice_pauses_for_export
                )
            )
    
        editor_video_duration = max(
            0.1,
            edited_source_duration
            + float(voiceover_start)
            + sum(duration for _, duration in voice_pauses_for_export),
        )
        # Floating media is managed by the primary editor above. Keep the legacy
        # collection empty so the old duplicate uploader/editor is not rendered.
        if "partner_image_overlays" not in st.session_state:
            st.session_state["partner_image_overlays"] = []
        st.session_state["partner_image_overlays"] = []
        overlay_items = []
        bulk_uploads = []
        current_bulk_signatures = {
            f"{overlay_upload.name}:{overlay_upload.size}"
            for overlay_upload in (bulk_uploads or [])
        }
        removed_signatures = set(st.session_state.get("partner_removed_overlay_signatures", []))
        removed_signatures.intersection_update(current_bulk_signatures)
        st.session_state["partner_removed_overlay_signatures"] = list(removed_signatures)
        existing_signatures = {
            str(item.get("upload_signature") or "") for item in overlay_items
        }
        next_available_start = max(
            (
                float(item.get("start") or 0.0)
                + float(item.get("duration") or 5.0)
                for item in overlay_items
                if item.get("path")
            ),
            default=0.0,
        )
        added_bulk_image = False
        for bulk_position, overlay_upload in enumerate(bulk_uploads or []):
            overlay_signature = f"{overlay_upload.name}:{overlay_upload.size}"
            if overlay_signature in existing_signatures or overlay_signature in removed_signatures:
                continue
            overlay_id = time.time_ns() + bulk_position
            saved_overlay = save_overlay_upload(overlay_upload, overlay_id)
            media_type = (
                "video"
                if saved_overlay.suffix.lower()
                in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
                else "image"
            )
            clip_duration = (
                max(0.1, probe_media_duration(saved_overlay))
                if media_type == "video"
                else 0.0
            )
            default_duration = min(
                5.0,
                editor_video_duration,
                clip_duration if media_type == "video" else editor_video_duration,
            )
            default_start = min(
                next_available_start,
                max(0.0, editor_video_duration - default_duration),
            )
            overlay_items.append(
                {
                    "id": overlay_id,
                    "path": str(saved_overlay),
                    "name": overlay_upload.name,
                    "upload_signature": overlay_signature,
                    "media_type": media_type,
                    "clip_duration": clip_duration,
                    "trim_start": 0.0,
                    "use_clip_audio": False,
                    "start": default_start,
                    "duration": default_duration,
                }
            )
            next_available_start = default_start + default_duration
            existing_signatures.add(overlay_signature)
            added_bulk_image = True
        if added_bulk_image:
            st.session_state["partner_image_overlays"] = overlay_items
            st.rerun()
    
        # Remove legacy empty placeholders from the previous long-form image UI.
        overlay_items = [
            item
            for item in overlay_items
            if item.get("path") and Path(str(item["path"])).exists()
        ]
        valid_overlay_items: List[Dict[str, object]] = list(overlay_items)
    
        st.session_state["partner_image_overlays"] = overlay_items
    
        image_overlays_for_export: List[Dict[str, object]] = []
        if valid_overlay_items:
            video_items = [
                item
                for item in valid_overlay_items
                if str(item.get("media_type") or "image") == "video"
            ]
            if video_items:
                st.markdown("**Video byte options**")
                st.caption(
                    "Placement and duration are controlled in the visual timeline. "
                    "Each uploaded clip plays from its beginning."
                )
                for item in video_items:
                    clip_path = Path(str(item["path"]))
                    clip_duration = max(
                        0.1,
                        float(item.get("clip_duration") or probe_media_duration(clip_path)),
                    )
                    item["clip_duration"] = clip_duration
                    with st.expander(
                        f"🎬 {item.get('name') or clip_path.name}",
                        expanded=len(video_items) == 1,
                    ):
                        has_clip_audio = media_has_audio(
                            str(clip_path), clip_path.stat().st_mtime_ns
                        )
                        audio_mode = st.selectbox(
                            "Audio while byte is visible",
                            [
                                "Keep main voiceover",
                                "Play original byte audio",
                            ],
                            index=(
                                1
                                if bool(item.get("use_clip_audio"))
                                and has_clip_audio
                                else 0
                            ),
                            disabled=not has_clip_audio,
                            key=f"partner_byte_audio_{item['id']}",
                        )
                        item["trim_start"] = 0.0
                        item["use_clip_audio"] = bool(
                            has_clip_audio
                            and audio_mode == "Play original byte audio"
                        )
                        available_clip_duration = clip_duration
                        item["duration"] = min(
                            float(item.get("duration") or 5.0),
                            available_clip_duration,
                            max(
                                0.1,
                                editor_video_duration
                                - float(item.get("start") or 0.0),
                            ),
                        )
                        st.caption(
                            f"Uploaded byte length: {compact_time(clip_duration)} · "
                            "starts at 0:00 inside the clip"
                        )
    
            current_layout = []
            missing_layout_ids = []
            for item in valid_overlay_items:
                if all(key in item for key in ("x", "y", "w", "h")):
                    current_layout.append(
                        {
                            "id": str(item["id"]),
                            "x": float(item["x"]),
                            "y": float(item["y"]),
                            "w": float(item["w"]),
                            "h": float(item["h"]),
                            "z": int(item.get("z") or 1),
                            "start": float(item.get("start") or 0.0),
                            "duration": float(item.get("duration") or 5.0),
                        }
                    )
                else:
                    missing_layout_ids.append(int(item["id"]))
            if missing_layout_ids:
                defaults = default_grid_layout(
                    [int(item["id"]) for item in valid_overlay_items],
                    columns=min(2, len(valid_overlay_items)),
                )
                default_by_id = {entry["id"]: entry for entry in defaults}
                for item in valid_overlay_items:
                    if int(item["id"]) not in missing_layout_ids:
                        continue
                    default = default_by_id[str(item["id"])]
                    for key in ("x", "y", "w", "h", "z"):
                        item[key] = default[key]
                current_layout = [
                    {
                        "id": str(item["id"]),
                        "x": float(item["x"]),
                        "y": float(item["y"]),
                        "w": float(item["w"]),
                        "h": float(item["h"]),
                        "z": int(item.get("z") or 1),
                        "start": float(item.get("start") or 0.0),
                        "duration": float(item.get("duration") or 5.0),
                    }
                    for item in valid_overlay_items
                ]
    
            editor_images = []
            for item in valid_overlay_items:
                item_path = Path(str(item["path"]))
                media_type = str(item.get("media_type") or "image")
                preview_src = (
                    video_preview_data_url(
                        str(item_path), item_path.stat().st_mtime_ns
                    )
                    if media_type == "video"
                    else image_preview_data_url(
                        str(item_path), item_path.stat().st_mtime_ns
                    )
                )
                editor_images.append(
                    {
                        "id": str(item["id"]),
                        "name": str(item.get("name") or item_path.name),
                        "kind": media_type,
                        "start": float(item.get("start") or 0.0),
                        "duration": float(item.get("duration") or 5.0),
                        "src": preview_src,
                    }
                )
            source_preview = video_preview_data_url(
                str(source_path), source_path.stat().st_mtime_ns
            )
            st.markdown("**Visual timeline and layout editor**")
            st.caption(
                "Select a visual, drag or resize it on the canvas, then set its "
                "start and duration on the timeline."
            )
            layout_result = overlay_layout_editor(
                images=editor_images,
                layout=current_layout,
                background=source_preview,
                video_duration=editor_video_duration,
                default={"items": current_layout},
                key=f"partner_overlay_editor_{st.session_state.get('partner_video_signature', source_path.name)}",
            )
            if isinstance(layout_result, dict):
                deleted_ids = {
                    str(value) for value in layout_result.get("deleted_ids", [])
                }
                present_ids = {str(item["id"]) for item in valid_overlay_items}
                actual_deleted_ids = deleted_ids.intersection(present_ids)
                if actual_deleted_ids:
                    removed_signatures.update(
                        str(item.get("upload_signature") or "")
                        for item in valid_overlay_items
                        if str(item["id"]) in actual_deleted_ids
                    )
                    st.session_state["partner_removed_overlay_signatures"] = [
                        signature for signature in removed_signatures if signature
                    ]
                    st.session_state["partner_image_overlays"] = [
                        item
                        for item in overlay_items
                        if str(item["id"]) not in actual_deleted_ids
                    ]
                    st.rerun()
                layout_by_id = {
                    str(entry.get("id")): entry
                    for entry in layout_result.get("items", [])
                    if isinstance(entry, dict)
                }
                for item in valid_overlay_items:
                    updated = layout_by_id.get(str(item["id"]))
                    if not updated:
                        continue
                    item["x"] = clamp_float(float(updated.get("x") or 0), 0.0, 0.96)
                    item["y"] = clamp_float(float(updated.get("y") or 0), 0.0, 0.96)
                    item["w"] = clamp_float(float(updated.get("w") or 0.45), 0.08, 1.0)
                    item["h"] = clamp_float(float(updated.get("h") or 0.45), 0.08, 1.0)
                    item["z"] = int(updated.get("z") or 1)
                    item["start"] = clamp_float(
                        float(updated.get("start") or 0.0),
                        0.0,
                        editor_video_duration,
                    )
                    maximum_duration = max(
                        0.1,
                        editor_video_duration - float(item["start"]),
                    )
                    if str(item.get("media_type") or "image") == "video":
                        maximum_duration = min(
                            maximum_duration,
                            max(0.1, float(item.get("clip_duration") or 0.1)),
                        )
                    item["duration"] = clamp_float(
                        float(updated.get("duration") or 5.0),
                        0.1,
                        maximum_duration,
                    )
            st.session_state["partner_image_overlays"] = overlay_items
            image_overlays_for_export = [dict(item) for item in valid_overlay_items]

        template_duration = max(0.1, editor_video_duration)
        template_header_path = Path(
            str(st.session_state.get("partner_template_header_path") or "")
        )
        stored_canvas_layout = st.session_state.get(
            "partner_template_canvas_layout", default_canvas_layout
        )
        template_geometry = {
            str(item.get("id")): {
                key: float(item.get(key) or 0.0) for key in ("x", "y", "w", "h")
            }
            for item in stored_canvas_layout
            if isinstance(item, dict) and item.get("id")
        }
        if template_header_path.is_file():
            header_image_geometry = template_geometry.get(
                "header_image", {"x": 0.01, "y": 0.01, "w": 0.20, "h": 0.18}
            )
            image_overlays_for_export.append(
                {
                    "id": "top-png",
                    "path": str(template_header_path),
                    "media_type": "image",
                    "start": 0.0,
                    "duration": template_duration,
                    "x": float(header_image_geometry.get("x", 0.01)),
                    "y": float(header_image_geometry.get("y", 0.01)),
                    "w": float(header_image_geometry.get("w", 0.20)),
                    "h": float(header_image_geometry.get("h", 0.18)),
                    "z": 900,
                }
            )
        template_loop_items_for_export = [
            dict(item) for item in st.session_state.get("partner_template_loop_items", [])
            if Path(str(item.get("path") or "")).is_file()
        ]
        if template_loop_items_for_export:
            panel_count = 1
            image_geometry = template_geometry.get(
                "images", {"x": 0.50, "y": 0.20, "w": 0.50, "h": 0.80}
            )
            seconds_per_image = float(template_photo_seconds)
            for panel_index in range(panel_count):
                panel_time = 0.0
                image_index = panel_index % len(template_loop_items_for_export)
                while panel_time < template_duration - 0.01:
                    floating_item = template_loop_items_for_export[
                        image_index % len(template_loop_items_for_export)
                    ]
                    image_path = Path(str(floating_item["path"]))
                    media_type = str(floating_item.get("media_type") or "image")
                    item_duration = seconds_per_image
                    if media_type == "video":
                        item_duration = min(
                            item_duration,
                            max(0.1, float(floating_item.get("clip_duration") or item_duration)),
                        )
                    visible_duration = min(item_duration, template_duration - panel_time)
                    image_overlays_for_export.append(
                        {
                            "id": f"template-loop-{panel_index}-{panel_time:.2f}",
                            "path": str(image_path),
                            "media_type": media_type,
                            "clip_duration": float(floating_item.get("clip_duration") or 0.0),
                            "trim_start": 0.0,
                            "use_clip_audio": bool(floating_item.get("use_clip_audio")),
                            "start": panel_time,
                            "duration": visible_duration,
                            "x": float(image_geometry.get("x", 0.50)),
                            "y": float(image_geometry.get("y", 0.20)),
                            "w": float(image_geometry.get("w", 0.50)),
                            "h": float(image_geometry.get("h", 0.80)),
                            "z": 800 + panel_index,
                        }
                    )
                    panel_time += visible_duration
                    image_index += panel_count
        if selected_logo_path and selected_logo_path.is_file():
            logo_geometry = template_geometry.get(
                "logo", {"x": 0.84, "y": 0.035, "w": 0.13, "h": 0.11}
            )
            image_overlays_for_export.append(
                {
                    "id": "property-logo",
                    "path": str(selected_logo_path),
                    "media_type": "image",
                    "start": 0.0,
                    "duration": template_duration,
                    "x": float(logo_geometry.get("x", 0.84)),
                    "y": float(logo_geometry.get("y", 0.035)),
                    "w": float(logo_geometry.get("w", 0.13)),
                    "h": float(logo_geometry.get("h", 0.11)),
                    "z": 950,
                    "fit_mode": "contain_transparent",
                }
            )
    
        st.divider()
        st.markdown("**Top-strip slugs**")
        st.caption(
            "Add multiple text bars to the right side of the top strip. Each slug has its own wording, "
            "highlight, colours, start time and duration."
        )
        if "partner_slug_overlays" not in st.session_state:
            legacy_text = str(st.session_state.get("partner_slug_text") or "").strip()
            st.session_state["partner_slug_overlays"] = (
                [
                    {
                        "id": time.time_ns(),
                        "text": legacy_text,
                        "highlight_text": str(
                            st.session_state.get("partner_slug_highlight") or ""
                        ),
                        "label": "",
                        "style": "Jagran Red",
                        "background_color": str(
                            st.session_state.get("partner_slug_background") or "#8F0711"
                        ),
                        "highlight_color": str(
                            st.session_state.get("partner_slug_highlight_colour")
                            or "#F6D35D"
                        ),
                        "start": float(
                            st.session_state.get("partner_slug_start") or 0.0
                        ),
                        "duration": float(
                            st.session_state.get("partner_slug_duration") or 5.0
                        ),
                    }
                ]
                if legacy_text
                else []
            )
    
        slug_items: List[Dict[str, object]] = st.session_state[
            "partner_slug_overlays"
        ]
        add_slug_columns = st.columns([0.72, 0.28], vertical_alignment="center")
        add_slug_columns[0].caption(
            f"{len(slug_items)} slug{'s' if len(slug_items) != 1 else ''} added"
        )
        if add_slug_columns[1].button(
            "＋ Add slug",
            use_container_width=True,
            key="partner_add_slug",
        ):
            last_end = max(
                (
                    float(item.get("start") or 0.0)
                    + float(item.get("duration") or 5.0)
                    for item in slug_items
                ),
                default=0.0,
            )
            default_duration = min(5.0, editor_video_duration)
            default_start = min(
                last_end,
                max(0.0, editor_video_duration - default_duration),
            )
            slug_items.append(
                {
                    "id": time.time_ns(),
                    "text": "",
                    "highlight_text": "",
                    "label": "",
                    "style": "Jagran Red",
                    "background_color": SLUG_STYLE_PRESETS["Jagran Red"]["background"],
                    "background_end_color": SLUG_STYLE_PRESETS["Jagran Red"]["background_end"],
                    "highlight_color": SLUG_STYLE_PRESETS["Jagran Red"]["accent"],
                    "start": default_start,
                    "duration": default_duration,
                }
            )
            st.session_state["partner_slug_overlays"] = slug_items
            st.rerun()
    
        slugs_for_export: List[Dict[str, object]] = []
        remove_slug_id: Optional[str] = None
        from PIL import Image
    
        for slug_index, slug in enumerate(slug_items):
            slug_id = str(slug["id"])
            slug_start_value = clamp_float(
                float(slug.get("start") or 0.0),
                0.0,
                max(0.0, editor_video_duration - 0.1),
            )
            slug_duration_value = clamp_float(
                float(slug.get("duration") or 5.0),
                0.1,
                max(0.1, editor_video_duration - slug_start_value),
            )
            slug_label = (
                str(slug.get("text") or "").strip()
                or f"Untitled slug {slug_index + 1}"
            )
            timing_label = (
                f"{compact_time(slug_start_value)}–"
                f"{compact_time(slug_start_value + slug_duration_value)}"
            )
            with st.expander(
                f"Slug {slug_index + 1} · {timing_label} · {slug_label[:48]}",
                expanded=not str(slug.get("text") or "").strip(),
            ):
                style_columns = st.columns([0.56, 0.44])
                current_style = str(slug.get("style") or "Jagran Red")
                if current_style not in SLUG_STYLE_PRESETS:
                    current_style = "Jagran Red"
                slug_style = style_columns[0].selectbox(
                    "Lower-third design",
                    list(SLUG_STYLE_PRESETS),
                    index=list(SLUG_STYLE_PRESETS).index(current_style),
                    key=f"partner_slug_style_{slug_id}",
                )
                slug_label_text = style_columns[1].text_input(
                    "Category label (optional)",
                    value=str(slug.get("label") or ""),
                    key=f"partner_slug_label_{slug_id}",
                    placeholder="NEWS UPDATE",
                )
                slug_text = st.text_input(
                    "Slug text",
                    value=str(slug.get("text") or ""),
                    key=f"partner_slug_text_{slug_id}",
                    placeholder="Enter the headline or label to display",
                )
                slug_highlight = st.text_input(
                    "Text to highlight (optional)",
                    value=str(slug.get("highlight_text") or ""),
                    key=f"partner_slug_highlight_{slug_id}",
                    placeholder="Exact word or phrase from this slug",
                    help=(
                        "The first matching word or phrase uses the selected "
                        "highlight colour."
                    ),
                )
                selected_preset = SLUG_STYLE_PRESETS[slug_style]
                if slug_style == "Custom":
                    colour_columns = st.columns(2)
                    slug_background = colour_columns[0].color_picker(
                        "Panel colour",
                        value=str(
                            slug.get("background_color")
                            or selected_preset["background"]
                        ),
                        key=f"partner_slug_background_{slug_id}",
                    )
                    slug_highlight_colour = colour_columns[1].color_picker(
                        "Accent colour",
                        value=str(
                            slug.get("highlight_color") or selected_preset["accent"]
                        ),
                        key=f"partner_slug_highlight_colour_{slug_id}",
                    )
                    slug_background_end = slug_background
                else:
                    slug_background = selected_preset["background"]
                    slug_background_end = selected_preset["background_end"]
                    slug_highlight_colour = selected_preset["accent"]
                    st.caption(
                        "Preset colours are applied automatically. Choose Custom "
                        "to select your own panel and accent colours."
                    )
                timing_columns = st.columns(2)
                slug_start = timing_columns[0].number_input(
                    "Start timestamp (seconds)",
                    min_value=0.0,
                    max_value=max(0.0, editor_video_duration - 0.1),
                    value=slug_start_value,
                    step=0.1,
                    key=f"partner_slug_start_{slug_id}",
                )
                maximum_slug_duration = max(
                    0.1, editor_video_duration - float(slug_start)
                )
                slug_duration = timing_columns[1].number_input(
                    "Duration (seconds)",
                    min_value=0.1,
                    max_value=maximum_slug_duration,
                    value=min(slug_duration_value, maximum_slug_duration),
                    step=0.1,
                    key=f"partner_slug_duration_{slug_id}",
                )
    
                slug.update(
                    {
                        "text": slug_text.strip(),
                        "highlight_text": slug_highlight.strip(),
                        "label": slug_label_text.strip(),
                        "style": slug_style,
                        "background_color": slug_background,
                        "background_end_color": slug_background_end,
                        "highlight_color": slug_highlight_colour,
                        "start": float(slug_start),
                        "duration": float(slug_duration),
                    }
                )
                if slug_text.strip():
                    slug["region"] = "template_header"
                    slug["geometry"] = template_geometry.get(
                        "slug", {"x": 0.22, "y": 0.01, "w": 0.77, "h": 0.18}
                    )
                    slugs_for_export.append(dict(slug))
                    slug_preview_path = build_slug_overlay_asset(slug, source_path)
                    with Image.open(slug_preview_path) as slug_preview:
                        st.image(
                            slug_preview.crop(
                                (0, 0, OUTPUT_WIDTH, int(OUTPUT_HEIGHT * 0.20))
                            ),
                            caption=f"Preview · {timing_label}",
                            use_container_width=True,
                        )
                else:
                    st.warning("Enter text for this slug or remove it.")
                if st.button(
                    "Remove this slug",
                    key=f"partner_remove_slug_{slug_id}",
                ):
                    remove_slug_id = slug_id
    
        if remove_slug_id is not None:
            st.session_state["partner_slug_overlays"] = [
                item for item in slug_items if str(item["id"]) != remove_slug_id
            ]
            st.rerun()
        st.session_state["partner_slug_overlays"] = slug_items
    
        ordered_slugs = sorted(
            slugs_for_export,
            key=lambda item: (float(item.get("start") or 0.0), int(item["id"])),
        )
        for first, second in zip(ordered_slugs, ordered_slugs[1:]):
            first_end = float(first["start"]) + float(first["duration"])
            if float(second["start"]) < first_end:
                st.info(
                    "Some slug timings overlap. During the overlap, the later slug "
                    "in the list will appear on top."
                )
                break
    
    with workspace_tabs[3]:
        render_stage_header(
            6,
            "Review and export",
            "Render the finished story and prepare the full-resolution master for download.",
        )
        st.markdown("### Gemini title and SEO metadata")
        st.caption(
            "Gemini analyses the approved transcript plus representative frames "
            "from the raw video, top PNG and floating media. Review every field before publishing."
        )
        metadata_language = st.radio(
            "Generate metadata in",
            ["Hindi", "English", "Hindi + English"],
            horizontal=True,
            key="partner_metadata_language",
        )
        metadata_action_columns = st.columns([0.72, 0.28], vertical_alignment="center")
        metadata_action_columns[0].caption(
            f"Model: {VERTEX_GEMINI_MODEL} on Vertex AI · maximum 20 meta tags"
        )
        if metadata_action_columns[1].button(
            "Generate with Gemini",
            type="primary",
            use_container_width=True,
            key="partner_generate_video_metadata",
            disabled=not bool(edited_transcript.strip()),
        ):
            with st.spinner("Analysing the transcript and video visuals..."):
                generated_metadata, metadata_message = generate_video_metadata_with_gemini(
                    edited_transcript,
                    metadata_language,
                    source_path,
                    template_header_path if template_header_path.is_file() else None,
                    template_loop_items,
                )
            if generated_metadata:
                st.session_state["partner_generated_title"] = generated_metadata["title"]
                st.session_state["partner_generated_headline"] = generated_metadata["headline"]
                st.session_state["partner_generated_description"] = generated_metadata["description"]
                st.session_state["partner_generated_meta_tags"] = ", ".join(
                    generated_metadata["meta_tags"]
                )
                st.session_state["partner_metadata_message"] = metadata_message
                st.rerun()
            st.error(metadata_message)
        if st.session_state.get("partner_metadata_message"):
            st.success(st.session_state["partner_metadata_message"])
        generated_title = st.text_input(
            "Video title",
            key="partner_generated_title",
            placeholder="Generate or enter the publishing title",
        )
        generated_headline = st.text_area(
            "Headline",
            key="partner_generated_headline",
            height=90,
            placeholder="Generate or enter the newsroom headline",
        )
        generated_description = st.text_area(
            "Description",
            key="partner_generated_description",
            height=180,
            placeholder="Generate or enter the publishing description",
        )
        generated_meta_tags = st.text_area(
            "Keywords / meta tags — comma separated",
            key="partner_generated_meta_tags",
            height=120,
            placeholder="tag one, tag two, tag three",
        )
        parsed_meta_tags = [
            tag.strip().lstrip("#")
            for tag in re.split(r"[,\n]", generated_meta_tags)
            if tag.strip()
        ][:20]
        st.caption(f"{len(parsed_meta_tags)}/20 meta tags · all fields remain editable")
        st.divider()
        if voice_choice == ELEVENLABS_VOICE_LABEL:
            voice_ready = bool(
                elevenlabs_api_key()
                and edited_transcript.strip()
                and len(re.sub(r"\s+", " ", edited_transcript).strip()) <= 9_500
            )
        elif voice_choice == "Clone producer voice from sample":
            voice_ready = bool(
                reference_audio
                and reference_audio.exists()
                and producer_consent
                and delivery_words_match
                and (not is_deepika_f5 or deepika_f5_is_ready())
            )
        elif voice_choice == "Upload completed voiceover":
            voice_ready = bool(uploaded_voiceover and uploaded_voiceover.exists())
        elif voice_choice == "No voiceover — use original video audio":
            voice_ready = True
        else:
            voice_ready = True
        slug_ready = len(slugs_for_export) == len(slug_items)
        pause_inserts_ready = (
            voice_choice == "No voiceover — use original video audio"
            or all(
            (
                str(item.get("insert_mode") or "silent") == "silent"
                or (
                    str(item.get("insert_mode")) == "generated"
                    and bool(str(item.get("insert_script") or "").strip())
                )
                or (
                    str(item.get("insert_mode")) == "upload"
                    and bool(item.get("insert_audio_path"))
                    and Path(str(item.get("insert_audio_path"))).exists()
                )
            )
            for item in voice_pause_items
            )
        )
        can_generate = (
            bool(edited_transcript.strip())
            and voice_ready
            and slug_ready
            and pause_inserts_ready
        )
        generate_button_label = "Generate 1920x1080 video"
        if voice_choice == ELEVENLABS_VOICE_LABEL:
            generate_button_label += f" with {selected_eleven_voice_label}"
        elif voice_choice == "Clone producer voice from sample" and selected_producer_label:
            generate_button_label += f" with {selected_producer_label}"
        if st.button(
            generate_button_label,
            type="primary",
            use_container_width=True,
            disabled=not can_generate,
        ):
            voiceover_path = uploaded_voiceover if voice_choice == "Upload completed voiceover" else None
            if voice_choice == ELEVENLABS_VOICE_LABEL:
                with st.spinner("Generating the ElevenLabs voiceover..."):
                    voiceover_path, voice_message = create_elevenlabs_voiceover(
                        edited_transcript,
                        selected_eleven_voice_id,
                        voice_speed,
                    )
                if not voiceover_path:
                    st.error(voice_message)
                    return
                st.success(voice_message)
            elif voice_choice == "Clone producer voice from sample":
                with st.spinner(
                    f"Generating the voiceover with {selected_producer_label or 'the selected producer'}..."
                ):
                    if is_deepika_f5:
                        voiceover_path, voice_message = create_deepika_f5_voiceover(
                            voice_script_for_generation,
                        )
                    else:
                        voiceover_path, voice_message = create_local_cloned_voiceover(
                            voice_script_for_generation,
                            reference_audio,
                            "hi" if language_label == "Hindi" else "en",
                            voice_model_mode,
                        )
                if not voiceover_path:
                    st.error(voice_message)
                    return
                st.success(voice_message)
            elif voice_choice not in [
                "Upload completed voiceover",
                "No voiceover — use original video audio",
            ]:
                voice_name = "Veena" if voice_choice.startswith("Hindi") else "Samantha"
                with st.spinner("Generating voiceover..."):
                    voiceover_path, voice_message = create_voiceover(edited_transcript, voice_name)
                if not voiceover_path:
                    st.error(voice_message)
                    return
    
            pause_audio_overlays_for_export: List[Dict[str, object]] = []
            for pause_index, pause in enumerate(voice_pause_items):
                insert_mode = str(pause.get("insert_mode") or "silent")
                if insert_mode == "silent":
                    continue
                pause_audio_path: Optional[Path] = None
                if insert_mode == "upload":
                    uploaded_pause_audio = Path(
                        str(pause.get("insert_audio_path") or "")
                    )
                    if uploaded_pause_audio.exists():
                        pause_audio_path = uploaded_pause_audio
                elif insert_mode == "generated":
                    pause_script = str(pause.get("insert_script") or "").strip()
                    with st.spinner(
                        f"Generating audio for pause {pause_index + 1}..."
                    ):
                        if voice_choice == ELEVENLABS_VOICE_LABEL:
                            pause_audio_path, pause_audio_message = (
                                create_elevenlabs_voiceover(
                                    pause_script,
                                    selected_eleven_voice_id,
                                    voice_speed,
                                )
                            )
                        elif voice_choice == "Clone producer voice from sample":
                            if is_deepika_f5:
                                pause_audio_path, pause_audio_message = (
                                    create_deepika_f5_voiceover(pause_script)
                                )
                            else:
                                pause_audio_path, pause_audio_message = (
                                    create_local_cloned_voiceover(
                                        pause_script,
                                        reference_audio,
                                        "hi" if language_label == "Hindi" else "en",
                                        voice_model_mode,
                                    )
                                )
                        else:
                            pause_voice_name = (
                                "Veena"
                                if language_label == "Hindi"
                                else "Samantha"
                            )
                            pause_audio_path, pause_audio_message = create_voiceover(
                                pause_script,
                                pause_voice_name,
                            )
                    if not pause_audio_path:
                        st.error(
                            f"Pause {pause_index + 1} audio failed: "
                            f"{pause_audio_message}"
                        )
                        return
                if not pause_audio_path:
                    st.error(f"Pause {pause_index + 1} does not have usable audio.")
                    return
                pause_duration = max(0.1, float(pause.get("duration") or 0.1))
                pause_audio_duration = probe_media_duration(pause_audio_path)
                if pause_audio_duration > pause_duration + 0.05:
                    st.warning(
                        f"Pause {pause_index + 1} audio is "
                        f"{pause_audio_duration:.1f}s, but the pause window is "
                        f"{pause_duration:.1f}s. The insert will be trimmed to fit."
                    )
                pause_audio_overlays_for_export.append(
                    {
                        "path": str(pause_audio_path),
                        "start": float(pause.get("start") or voiceover_start),
                        "duration": pause_duration,
                    }
                )
    
            render_source_path = source_path
            if source_cuts_for_export:
                with st.spinner("Removing sections from the uploaded raw video..."):
                    render_source_path, cut_message = build_source_cut_cache(
                        source_path,
                        source_cuts_for_export,
                    )
                if not render_source_path:
                    st.error(cut_message)
                    return
                st.success(cut_message)
    
            with st.spinner("Composing voiceover and overlays on the edited raw video..."):
                output, message = export_horizontal_video(
                    render_source_path,
                    edited_transcript,
                    voiceover_path,
                    keep_original_audio=(
                        voice_choice == "No voiceover — use original video audio"
                    ),
                    add_intro_slate=False,
                    image_overlays=image_overlays_for_export,
                    slug_overlays=slugs_for_export,
                    source_cuts=None,
                    voice_timing=voice_timing_for_export,
                    output_stem=source_path.stem,
                    tail_mode=video_tail_mode,
                    pause_audio_overlays=pause_audio_overlays_for_export,
                    template_layout=template_layout,
                    template_geometry=template_geometry,
                )
            if output:
                metadata_sidecar = output.with_suffix(".metadata.json")
                metadata_sidecar.write_text(
                    json.dumps(
                        {
                            "title": generated_title.strip(),
                            "headline": generated_headline.strip(),
                            "description": generated_description.strip(),
                            "meta_tags": parsed_meta_tags,
                            "language_mode": metadata_language,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                st.session_state["partner_latest_export"] = str(output)
                st.session_state["partner_latest_preview"] = str(output)
                st.session_state["partner_latest_export_message"] = message
                with st.spinner("Preparing the browser preview..."):
                    preview_path, preview_message = build_browser_preview(output)
                st.session_state["partner_latest_preview"] = str(
                    preview_path or output
                )
                if not preview_path:
                    st.warning(
                        "The full video was created, but its lightweight preview "
                        f"could not be prepared: {preview_message}"
                    )
                st.rerun()
            else:
                st.error(message)
    
        if not st.session_state.get("partner_latest_export"):
            previous_exports = sorted(
                EXPORT_DIR.glob(f"{source_path.stem}_ai_anchor_horizontal*.mp4"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if previous_exports:
                recovered_export = previous_exports[0]
                recovered_preview, _ = build_browser_preview(recovered_export)
                st.session_state["partner_latest_export"] = str(recovered_export)
                st.session_state["partner_latest_preview"] = str(
                    recovered_preview or recovered_export
                )
                st.session_state["partner_latest_export_message"] = (
                    f"Latest generated video: {recovered_export.name}"
                )
    
        latest_export_value = st.session_state.get("partner_latest_export")
        latest_preview_value = st.session_state.get("partner_latest_preview")
        if latest_export_value:
            latest_export = Path(str(latest_export_value))
            latest_preview = Path(str(latest_preview_value or latest_export_value))
            if latest_export.exists():
                st.success(
                    st.session_state.get("partner_latest_export_message")
                    or f"Video exported: {latest_export.name}"
                )
                if latest_preview.exists() and latest_preview != latest_export:
                    st.caption(
                        "Playback preview · The downloaded file remains full-resolution 1920×1080."
                    )
                st.video(str(latest_preview if latest_preview.exists() else latest_export))
                download_ready_key = f"partner_download_ready_{latest_export.name}"
                if not st.session_state.get(download_ready_key):
                    st.caption(
                        f"Full-resolution file: {latest_export.stat().st_size / (1024 * 1024):.1f} MB"
                    )
                    if st.button(
                        "Prepare full-resolution download",
                        key=f"partner_prepare_download_{latest_export.name}",
                    ):
                        st.session_state[download_ready_key] = True
                        st.rerun()
                else:
                    with latest_export.open("rb") as file_obj:
                        st.download_button(
                            "Download full-resolution video",
                            file_obj,
                            file_name=latest_export.name,
                            mime="video/mp4",
                            key=f"partner_download_{latest_export.name}",
                            on_click="ignore",
                        )
            else:
                st.session_state.pop("partner_latest_export", None)
                st.session_state.pop("partner_latest_preview", None)
                st.session_state.pop("partner_latest_export_message", None)
    
    
if __name__ == "__main__":
    main()

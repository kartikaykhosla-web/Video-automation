# Video Automation Studio

A Streamlit newsroom video editor for turning uploaded footage into branded horizontal videos. It supports transcription, editable scripts, optional voiceover, freehand canvas overlays, timed image/video layers, property logos, slug graphics, and Gemini-generated publishing metadata.

## Core features

- Upload raw footage or select a configured partner-video source.
- Generate and edit Hindi transcripts.
- Use original audio or optional ElevenLabs/local voiceover.
- Add pauses and preview estimated voiceover duration.
- Position and resize overlays directly on the video canvas.
- Add timed images or video overlays with audio-priority controls.
- Select property branding from `partner_property_logos/`.
- Generate Hindi, English, or bilingual title, headline, description, and up to 20 keywords using Vertex AI Gemini.
- Export the rendered video and a companion metadata JSON file.

## Setup

1. Install Python 3.12 and FFmpeg (`ffmpeg` and `ffprobe` must be on `PATH`).
2. Create a virtual environment and install the base dependencies:

   ```bash
   python3 -m venv .venv-shorts
   .venv-shorts/bin/pip install -r requirements.txt
   ```

3. Add property logo PNGs to `partner_property_logos/` using the filenames documented there.
4. Configure secrets through environment variables or Streamlit Secrets. Use
   `.streamlit/secrets.example.toml` only as a field-name reference. Never commit
   API keys or service-account JSON files.
5. Start the editor:

   ```bash
   .venv-shorts/bin/streamlit run partner_video_repackager_app.py --server.port 8503
   ```

## Credentials

- ElevenLabs: set `ELEVENLABS_API_KEY` in the server environment, or enter it for the current app session.
- Vertex AI Gemini: configure the `[vertex_service_account]` table in Streamlit
  Secrets. For local use, `service_account_vertex.json` can instead be placed
  beside the app or in its parent directory; it is ignored by Git.
- Reuters/ANI: configure the applicable API credentials when those provider adapters are enabled.

## Optional local voice engines

The included helper scripts support MLX, Chatterbox, and F5-TTS configurations. Those engines require platform-specific environments and model assets beyond the base requirements. The ElevenLabs/original-audio workflow does not require them.

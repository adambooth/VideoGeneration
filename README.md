# Automated Video Creator MVP

Windows desktop MVP for turning article or webpage URLs into short-form vertical videos with a staged approval workflow.

## What it does

- Accepts one or more URLs
- Extracts page text
- Uses the Google Gemini API to create a short-form script and metadata
- Uses Microsoft Edge neural TTS by default to generate narration
- Keeps ElevenLabs available as an optional narration provider
- Fetches stock footage or images from Pexels and Pixabay when configured
- Falls back to hybrid or local generated scenes when stock assets are unavailable
- Adds style-based generated background music
- Renders a vertical MP4 with FFmpeg
- Includes an Accounts / Connections panel with saved local credentials and connection tests
- Includes a first-launch setup wizard
- Pauses after major stages for approval:
  - Script
  - Voiceover
  - Visual plan
  - Final render

## Stack

- Python 3.11+
- PySide6 desktop GUI
- Requests for API access
- edge-tts for free Microsoft Edge neural narration
- Trafilatura for content extraction
- Pillow for visual asset generation
- FFmpeg for media rendering

## Install

1. Install Python 3.11 or newer.
2. Install FFmpeg and ensure `ffmpeg` and `ffprobe` are available on your `PATH`.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

## API setup

### Gemini (Google AI)

1. Create an API key in Google AI Studio.
2. Paste it into the `Gemini API Key` field in `Accounts / Connections`.
3. The app defaults to `gemini-2.5-flash`.
4. The Accounts panel also includes a model selector and connection test.

### Narration

#### Edge TTS (Default)

1. No API key is required.
2. In `Accounts / Connections`, leave `Narration Engine` set to `Edge TTS`.
3. Pick a voice, speed, and volume.
4. Use `Test Voice` to generate a preview clip.

#### ElevenLabs (Optional)

1. Create an ElevenLabs API key.
2. Switch `Narration Engine` to `ElevenLabs`.
3. Paste it into the `ElevenLabs API Key` field.
4. Paste your target ElevenLabs `Voice ID` into the `Voice ID` field.
5. Use `Test Voice Connection` to confirm the selected voice is reachable.

### Optional Stock Media APIs

#### Pexels

1. Create a free Pexels API key.
2. Paste it into the `Pexels API Key` field.
3. Best used with `Stock Footage` or `Hybrid` visual modes.

Official docs: [Pexels API documentation](https://www.pexels.com/api/documentation/)

#### Pixabay

1. Create a free Pixabay API key.
2. Paste it into the `Pixabay API Key` field.
3. Used as a backup stock source when Pexels is unavailable or returns weak matches.

Official docs: [Pixabay API documentation](https://pixabay.com/api/docs/)

## Run

```powershell
python main.py
```

On first launch, the app will open a setup wizard if Gemini is not connected or narration is not ready.

## FFmpeg requirement

Rendering depends on FFmpeg. If the app reports FFmpeg is missing:

1. Install FFmpeg for Windows.
2. Add its `bin` folder to your system `PATH`.
3. Restart the app.

## Project output structure

Each project is saved under your chosen export folder:

```text
Projects/
  project-name/
    project.json
    source.txt
    script.txt
    metadata.txt
    voice.mp3
    music.wav
    final.mp4
    assets/
      scene_01.png
      scene_02.png
      ...
```

## Notes

- The app supports resuming saved projects from the GUI.
- No upload modules are included yet, but the code is organized so uploader services can be added later.
- Saved credentials are stored locally in `.avc_settings/AutomatedVideoCreator/settings.json` and sensitive fields are obfuscated/encrypted before being written.
- The default narrator is `Edge TTS` with `en-US-GuyNeural`, `+5%` speed, and `0%` volume.
- Visual provider options are:
  - `Stock Footage`
  - `Hybrid`
  - `AI Images`
  - `Local Fallback`
- Music style options are:
  - `Suspense`
  - `Ambient`
  - `Corporate`
  - `None`
- `Stock Footage` and `Hybrid` work best when at least one stock API key is configured.
- If no stock keys are present, the app falls back to locally generated visuals rather than failing.

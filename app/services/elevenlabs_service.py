from __future__ import annotations

import time
from pathlib import Path

import requests


class ElevenLabsService:
    API_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    SPEECH_TO_SPEECH_URL_TEMPLATE = "https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}"
    VOICES_URL = "https://api.elevenlabs.io/v2/voices"

    def generate_voiceover(
        self,
        api_key: str,
        voice_id: str,
        text: str,
        output_path: str,
        retries: int = 3,
    ) -> str:
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")
        if not voice_id:
            raise ValueError("ElevenLabs Voice ID is required.")

        url = self.API_URL_TEMPLATE.format(voice_id=voice_id)
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.58,
                "similarity_boost": 0.82,
                "style": 0.32,
                "use_speaker_boost": True,
            },
        }
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    url,
                    timeout=120,
                    headers={
                        "xi-api-key": api_key,
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        self._extract_error_message(
                            response,
                            "ElevenLabs rejected the API key for voice generation. Check that the key is correct and belongs to the current workspace.",
                        )
                    )
                if response.status_code == 402:
                    raise RuntimeError(self._extract_error_message(response, "ElevenLabs account has no available credits or billing access for voice generation."))
                response.raise_for_status()
                path = Path(output_path)
                path.write_bytes(response.content)
                self._validate_audio_file(path)
                return str(path)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(4, attempt * 1.2))
        raise RuntimeError(f"ElevenLabs voice generation failed after retries: {last_error}")

    def convert_speech(
        self,
        *,
        api_key: str,
        voice_id: str,
        input_audio_path: str,
        output_path: str,
        retries: int = 3,
    ) -> str:
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")
        if not voice_id:
            raise ValueError("ElevenLabs Voice ID is required.")
        input_path = Path(input_audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Missing source audio for voice conversion: {input_audio_path}")

        url = self.SPEECH_TO_SPEECH_URL_TEMPLATE.format(voice_id=voice_id)
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with input_path.open("rb") as audio_file:
                    response = requests.post(
                        url,
                        timeout=180,
                        headers={
                            "xi-api-key": api_key,
                            "Accept": "audio/mpeg",
                        },
                        data={
                            "model_id": "eleven_multilingual_sts_v2",
                            "stability": "0.45",
                            "similarity_boost": "0.88",
                            "style": "0.20",
                            "use_speaker_boost": "true",
                        },
                        files={
                            "audio": (input_path.name, audio_file, "audio/mpeg"),
                        },
                    )
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        self._extract_error_message(
                            response,
                            "ElevenLabs rejected the API key for voice changing. Check that the key is correct and belongs to the current workspace.",
                        )
                    )
                if response.status_code == 402:
                    raise RuntimeError(
                        self._extract_error_message(
                            response,
                            "ElevenLabs account has no available credits or billing access for voice changing.",
                        )
                    )
                response.raise_for_status()
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                self._validate_audio_file(path)
                return str(path)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(5, attempt * 1.4))
        raise RuntimeError(f"ElevenLabs voice conversion failed after retries: {last_error}")

    def _validate_audio_file(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError("ElevenLabs did not produce an audio file.")
        if path.stat().st_size < 2048:
            raise RuntimeError("Generated ElevenLabs audio file is too small to be valid.")

        header = path.read_bytes()[:8]
        is_mp3 = header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
        is_wav = header.startswith(b"RIFF")
        if not (is_mp3 or is_wav):
            raise RuntimeError("Generated audio file is not a recognized MP3 or WAV output.")

    def test_connection(self, api_key: str, voice_id: str) -> tuple[bool, str]:
        if not api_key:
            return False, "Missing API key"
        if not voice_id:
            return False, "Missing Voice ID"
        try:
            response = requests.get(
                self.VOICES_URL,
                timeout=20,
                headers={"xi-api-key": api_key},
            )
            if not response.ok:
                if response.status_code == 402:
                    return False, "API key is valid, but ElevenLabs billing or credits are not available"
                if response.status_code in {401, 403}:
                    return False, "Invalid Key"
                return False, self._extract_error_message(response, f"Error {response.status_code}")
            voices = response.json().get("voices", [])
            matched_voice = next((item for item in voices if item.get("voice_id") == voice_id), None)
            if matched_voice:
                voice_warning = self._voice_usage_warning(matched_voice)
                if voice_warning:
                    return False, voice_warning
                return True, "Connected"
            return False, "API key OK, but Voice ID is not in this ElevenLabs workspace"
        except Exception as exc:  # noqa: BLE001
            return False, f"Error: {exc}"

    def _extract_error_message(self, response: requests.Response, fallback: str) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, dict):
                    message = detail.get("message") or detail.get("status")
                    if message:
                        return str(message)
                if isinstance(detail, str) and detail.strip():
                    return detail.strip()
                message = payload.get("message") or payload.get("error")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        except Exception:
            pass
        return fallback

    def _voice_usage_warning(self, voice: dict) -> str:
        category = str(voice.get("category") or "").strip().lower()
        labels = voice.get("labels") or {}
        access_level = ""
        if isinstance(labels, dict):
            access_level = str(labels.get("access_level") or labels.get("use_case") or "").strip().lower()

        if category == "library" or "library" in access_level:
            return "Voice found, but this appears to be a library voice. ElevenLabs free plans usually cannot use library voices through the API."
        return ""

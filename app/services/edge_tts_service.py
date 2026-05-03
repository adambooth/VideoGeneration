from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class EdgeTTSService:
    POPULAR_VOICES = [
        "en-US-GuyNeural",
        "en-US-ChristopherNeural",
        "en-US-DavisNeural",
        "en-US-AriaNeural",
        "en-GB-RyanNeural",
        "en-GB-ThomasNeural",
        "en-GB-SoniaNeural",
    ]

    def generate_voiceover(
        self,
        text: str,
        output_path: str,
        voice: str = "en-US-GuyNeural",
        rate: int = 5,
        volume: int = 0,
        retries: int = 3,
    ) -> str:
        if not text.strip():
            raise ValueError("Narration text is empty.")
        if not voice.strip():
            raise ValueError("Edge TTS voice is required.")

        last_error: Exception | None = None
        path = Path(output_path)
        for attempt in range(1, retries + 1):
            try:
                self._save_audio(
                    text=self._prepare_text(text),
                    output_path=path,
                    voice=voice.strip(),
                    rate=rate,
                    volume=volume,
                )
                self._validate_audio_file(path)
                self._normalize_audio(path, volume)
                self._validate_audio_file(path)
                return str(path)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(4, attempt * 1.1))
        raise RuntimeError(f"Edge TTS voice generation failed after retries: {last_error}")

    def test_voice(self, voice: str, rate: int, volume: int) -> tuple[bool, str, str | None]:
        if not voice.strip():
            return False, "Missing voice selection", None
        try:
            temp_dir = Path(tempfile.gettempdir()) / "avc_edge_tts_preview"
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = temp_dir / "edge_tts_sample.mp3"
            self.generate_voiceover(
                text="This is a quick voice test for your automated video creator.",
                output_path=str(output_path),
                voice=voice,
                rate=rate,
                volume=volume,
                retries=1,
            )
            return True, "Connected", str(output_path)
        except Exception as exc:  # noqa: BLE001
            return False, f"Error: {exc}", None

    def check_available(self) -> tuple[bool, str]:
        try:
            self._import_edge_tts()
        except Exception as exc:  # noqa: BLE001
            return False, f"Edge TTS unavailable: {exc}"
        return True, "Ready"

    def _save_audio(self, text: str, output_path: Path, voice: str, rate: int, volume: int) -> None:
        edge_tts = self._import_edge_tts()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self._format_percent(rate),
            volume=self._format_percent(volume),
        )
        asyncio.run(communicate.save(str(output_path)))

    def _normalize_audio(self, path: Path, volume: int) -> None:
        if not shutil.which("ffmpeg"):
            return
        temp_path = path.with_name(f"{path.stem}_normalized.mp3")
        volume_gain = max(-2.0, min(2.0, volume / 10.0))
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                "-af",
                f"volume={volume_gain:+.2f}dB,loudnorm=I=-16:TP=-1.5:LRA=11",
                str(temp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not temp_path.exists():
            return
        path.write_bytes(temp_path.read_bytes())
        temp_path.unlink(missing_ok=True)

    def _validate_audio_file(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError("Edge TTS did not produce an audio file.")
        if path.stat().st_size < 2048:
            raise RuntimeError("Generated Edge TTS audio file is too small to be valid.")
        header = path.read_bytes()[:8]
        is_mp3 = header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
        if not is_mp3:
            raise RuntimeError("Generated Edge TTS file is not a recognized MP3 output.")

    def _format_percent(self, value: int) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value}%"

    def _prepare_text(self, text: str) -> str:
        cleaned = text.replace("\r", "\n").strip()
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        joined = []
        for line in lines:
            if joined and not re.search(r"[.!?…,:;]$", joined[-1]):
                joined[-1] = f"{joined[-1]} {line}"
            else:
                joined.append(line)
        smoothed = " ".join(joined)
        smoothed = re.sub(r"\s+", " ", smoothed).strip()
        return smoothed

    def _import_edge_tts(self):
        try:
            import edge_tts  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Install the 'edge-tts' package to use Edge narration.") from exc
        return edge_tts

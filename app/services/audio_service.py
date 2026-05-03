from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


class AudioService:
    STYLE_MAP = {
        "Suspense": {
            "base": (110.0, 146.83, 174.61),
            "pulse": 0.25,
            "color": 0.12,
        },
        "Ambient": {
            "base": (174.61, 220.0, 261.63),
            "pulse": 0.12,
            "color": 0.08,
        },
        "Corporate": {
            "base": (196.0, 246.94, 293.66),
            "pulse": 0.18,
            "color": 0.10,
        },
        "None": {
            "base": (),
            "pulse": 0.0,
            "color": 0.0,
        },
    }

    def generate_background_music(self, output_path: str, duration_seconds: float, style: str) -> str:
        path = Path(output_path)
        if style == "None":
            self._write_silence(path, duration_seconds)
            return str(path)

        sample_rate = 44100
        frames = int(sample_rate * max(duration_seconds, 1))
        settings = self.STYLE_MAP.get(style, self.STYLE_MAP["Ambient"])
        freqs = settings["base"]

        with wave.open(str(path), "w") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for index in range(frames):
                t = index / sample_rate
                left = self._sample_value(t, freqs, settings["pulse"], settings["color"], side=0)
                right = self._sample_value(t, freqs, settings["pulse"], settings["color"], side=1)
                wav_file.writeframes(struct.pack("<hh", left, right))
        return str(path)

    def _sample_value(
        self,
        t: float,
        frequencies: tuple[float, ...],
        pulse: float,
        color: float,
        side: int,
    ) -> int:
        if not frequencies:
            return 0
        envelope = min(1.0, t / 1.2) * max(0.18, 1.0 - ((t + side * 0.3) % 14) / 22)
        value = 0.0
        drift = 1.0 + math.sin(t / 7.0 + side) * 0.01
        for idx, freq in enumerate(frequencies):
            shifted = freq * drift * (1.0 + idx * 0.002)
            value += math.sin(2 * math.pi * shifted * t) * (color / (idx + 1))
            value += math.sin(2 * math.pi * (shifted / 2) * t) * ((color * 0.55) / (idx + 1))
        pulse_layer = math.sin(2 * math.pi * pulse * t) * 0.035
        noise_like = math.sin(2 * math.pi * 0.07 * t + side * 0.9) * 0.015
        sample = int(max(-1.0, min(1.0, (value + pulse_layer + noise_like) * envelope)) * 32767)
        return sample

    def _write_silence(self, path: Path, duration_seconds: float) -> None:
        sample_rate = 44100
        frames = int(sample_rate * max(duration_seconds, 1))
        with wave.open(str(path), "w") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            silent_frame = struct.pack("<hh", 0, 0)
            for _ in range(frames):
                wav_file.writeframes(silent_frame)

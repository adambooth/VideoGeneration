from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from app.config import AppSettings

WINDOWS_DPAPI_AVAILABLE = os.name == "nt"

if WINDOWS_DPAPI_AVAILABLE:
    from ctypes import POINTER, Structure, byref, cast, create_string_buffer, windll
    from ctypes.wintypes import DWORD

    class DATA_BLOB(Structure):
        _fields_ = [("cbData", DWORD), ("pbData", POINTER(type(create_string_buffer(1))._type_))]  # type: ignore[attr-defined]


class SettingsStore:
    def __init__(self) -> None:
        base_dir = Path.cwd() / ".avc_settings"
        self.settings_dir = base_dir / "AutomatedVideoCreator"
        self.settings_path = self.settings_dir / "settings.json"

    def load(self) -> AppSettings:
        if not self.settings_path.exists():
            return AppSettings()
        payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        decrypted = {}
        for key, value in payload.items():
            if isinstance(value, str) and value.startswith("enc:"):
                decrypted[key] = self._decrypt(value[4:])
            else:
                decrypted[key] = value
        return AppSettings.from_dict(decrypted)

    def save(self, settings: AppSettings) -> None:
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        raw = settings.to_dict()
        secure_keys = {
            "gemini_api_key",
            "elevenlabs_api_key",
            "elevenlabs_voice_id",
            "serpapi_api_key",
            "pexels_api_key",
            "pixabay_api_key",
            "gemini_custom_model",
            "veo_model",
        }
        encoded = {}
        for key, value in raw.items():
            if key in secure_keys and isinstance(value, str) and value:
                encoded[key] = f"enc:{self._encrypt(value)}"
            else:
                encoded[key] = value
        self.settings_path.write_text(json.dumps(encoded, indent=2), encoding="utf-8")

    def has_required_accounts(self) -> bool:
        settings = self.load()
        if not settings.gemini_api_key:
            return False
        if settings.narration_engine == "ElevenLabs":
            return bool(settings.elevenlabs_api_key and settings.elevenlabs_voice_id)
        if settings.narration_engine == "Disabled":
            return False
        return True

    def _encrypt(self, value: str) -> str:
        if not WINDOWS_DPAPI_AVAILABLE:
            return base64.b64encode(value.encode("utf-8")).decode("ascii")
        try:
            raw = value.encode("utf-8")
            blob_in = self._to_blob(raw)
            blob_out = DATA_BLOB()
            if windll.crypt32.CryptProtectData(byref(blob_in), "AutomatedVideoCreator", None, None, None, 0, byref(blob_out)):
                data = self._from_blob(blob_out)
                return base64.b64encode(data).decode("ascii")
        except Exception:
            pass
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, value: str) -> str:
        if not WINDOWS_DPAPI_AVAILABLE:
            try:
                return base64.b64decode(value).decode("utf-8")
            except Exception:
                return ""
        try:
            raw = base64.b64decode(value)
            blob_in = self._to_blob(raw)
            blob_out = DATA_BLOB()
            if windll.crypt32.CryptUnprotectData(byref(blob_in), None, None, None, None, 0, byref(blob_out)):
                data = self._from_blob(blob_out)
                return data.decode("utf-8")
        except Exception:
            pass
        try:
            return base64.b64decode(value).decode("utf-8")
        except Exception:
            return ""

    def _to_blob(self, data: bytes) -> DATA_BLOB:
        if not WINDOWS_DPAPI_AVAILABLE:
            raise RuntimeError("Windows DPAPI is not available on this platform.")
        buffer = create_string_buffer(data, len(data))
        blob = DATA_BLOB()
        blob.cbData = len(data)
        blob.pbData = cast(buffer, DATA_BLOB._fields_[1][1])  # type: ignore[index]
        blob._buffer = buffer  # type: ignore[attr-defined]
        return blob

    def _from_blob(self, blob: DATA_BLOB) -> bytes:
        if not WINDOWS_DPAPI_AVAILABLE:
            raise RuntimeError("Windows DPAPI is not available on this platform.")
        pointer_type = POINTER(type(create_string_buffer(1))._type_)  # type: ignore[attr-defined]
        buffer = cast(blob.pbData, pointer_type)
        data = bytes(buffer[: blob.cbData])
        windll.kernel32.LocalFree(blob.pbData)
        return data

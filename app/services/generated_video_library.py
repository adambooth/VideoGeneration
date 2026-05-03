from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


class GeneratedVideoLibrary:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "metadata").mkdir(exist_ok=True)

    def build_clip_key(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: str,
        reference_image_signature: str,
        duration_seconds: int,
        aspect_ratio: str,
        resolution: str,
        style_name: str,
        scene_purpose: str,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            "reference_image_signature": reference_image_signature.strip(),
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "style_name": style_name,
            "scene_purpose": scene_purpose,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return digest[:24]

    def find_existing_clip(self, clip_key: str) -> dict | None:
        metadata_path = self.root / "metadata" / f"{clip_key}.json"
        if not metadata_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        video_path = Path(payload.get("video_path", ""))
        if not video_path.exists():
            return None
        poster_path = Path(payload.get("poster_path", ""))
        if payload.get("poster_path") and not poster_path.exists():
            payload["poster_path"] = ""
        return payload

    def save_generated_clip(
        self,
        *,
        clip_key: str,
        source_video_path: str,
        clip_name: str,
        metadata: dict,
    ) -> dict:
        slug = self._slugify(clip_name)
        target_dir = self.root / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_video_path = target_dir / f"{slug}_{clip_key}.mp4"
        shutil.copy2(source_video_path, target_video_path)
        poster_source_path = Path(str(metadata.get("poster_path", "") or ""))
        target_poster_path = ""
        if poster_source_path.exists():
            target_poster = target_dir / f"{slug}_{clip_key}{poster_source_path.suffix or '.jpg'}"
            shutil.copy2(poster_source_path, target_poster)
            target_poster_path = str(target_poster)
        payload = {
            **metadata,
            "clip_key": clip_key,
            "clip_name": clip_name,
            "video_path": str(target_video_path),
            "poster_path": target_poster_path,
        }
        metadata_path = self.root / "metadata" / f"{clip_key}.json"
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def copy_clip_into_project(self, metadata: dict, destination_path: str) -> str:
        source_path = Path(metadata["video_path"])
        target_path = Path(destination_path)
        if source_path.resolve() == target_path.resolve():
            return str(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return str(target_path)

    def copy_poster_into_project(self, metadata: dict, destination_path: str) -> str:
        poster_path = metadata.get("poster_path", "")
        if not poster_path:
            return ""
        source_path = Path(poster_path)
        if not source_path.exists():
            return ""
        target_path = Path(destination_path)
        if source_path.resolve() == target_path.resolve():
            return str(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return str(target_path)

    def _slugify(self, value: str) -> str:
        lowered = value.strip().lower()
        safe = "".join(char if char.isalnum() else "-" for char in lowered)
        safe = "-".join(part for part in safe.split("-") if part)
        return safe[:80] or "generated-scene"

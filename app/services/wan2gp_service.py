from __future__ import annotations

import json
import re
from pathlib import Path


class Wan2GPService:
    FORCED_PORTRAIT_RESOLUTION = "768x1024"
    NO_TEXT_NEGATIVE_PROMPT = (
        "subtitles, captions, text overlay, on-screen text, words, letters, typography, watermark, logo, burned-in dialogue, speech bubbles, comic text, title cards, intertitles, font, readable text, written words, annotation"
    )

    def validate_paths(self, *, root_dir: str, template_path: str, image_path: str) -> None:
        root = Path(root_dir).expanduser()
        template = Path(template_path).expanduser()
        image = Path(image_path).expanduser()
        if not root.exists() or not (root / "wgp.py").exists():
            raise FileNotFoundError(f"Wan2GP root is invalid or missing wgp.py: {root}")
        if not template.exists() or not template.is_file():
            raise FileNotFoundError(f"Wan2GP template JSON not found: {template}")
        if not image.exists() or not image.is_file():
            raise FileNotFoundError(f"Wan2GP start image not found: {image}")

    def validate_image_paths(self, image_paths: list[str]) -> list[str]:
        resolved: list[str] = []
        for raw_path in image_paths:
            path = Path(str(raw_path).strip()).expanduser()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Wan2GP storyboard image not found: {path}")
            resolved.append(str(path.resolve()))
        return resolved

    def load_template(self, template_path: str) -> dict:
        return json.loads(Path(template_path).read_text(encoding="utf-8"))

    def build_flux_image_payload(
        self,
        *,
        template: dict,
        prompt: str,
        reference_image_path: str,
        output_filename: str,
    ) -> dict:
        payload = dict(template)
        payload["prompt"] = prompt.strip()
        payload["image_start"] = ""
        payload["image_end"] = ""
        payload["image_prompt_type"] = ""
        payload["image_refs"] = [str(Path(reference_image_path).expanduser().resolve()).replace("\\", "/")]
        payload["resolution"] = self.FORCED_PORTRAIT_RESOLUTION
        existing_negative = str(payload.get("negative_prompt", "") or "").strip()
        payload["negative_prompt"] = (
            f"{existing_negative}, {self.NO_TEXT_NEGATIVE_PROMPT}" if existing_negative else self.NO_TEXT_NEGATIVE_PROMPT
        )
        payload["output_filename"] = output_filename
        return payload

    def build_settings_payload(
        self,
        *,
        template: dict,
        prompt: str,
        image_path: str,
        image_end_path: str | None = None,
        output_filename: str,
        clip_length_seconds: int | None = None,
    ) -> dict:
        payload = dict(template)
        payload["prompt"] = prompt.strip()
        payload["image_start"] = str(Path(image_path).expanduser().resolve()).replace("\\", "/")
        if image_end_path:
            payload["image_end"] = str(Path(image_end_path).expanduser().resolve()).replace("\\", "/")
            payload["image_prompt_type"] = "SE"
        else:
            payload["image_end"] = ""
            payload["image_prompt_type"] = payload.get("image_prompt_type", "S") or "S"
        payload["resolution"] = self.FORCED_PORTRAIT_RESOLUTION
        existing_negative = str(payload.get("negative_prompt", "") or "").strip()
        payload["negative_prompt"] = (
            f"{existing_negative}, {self.NO_TEXT_NEGATIVE_PROMPT}" if existing_negative else self.NO_TEXT_NEGATIVE_PROMPT
        )
        if clip_length_seconds:
            payload["video_length"] = max(24, int(round(float(clip_length_seconds) * 24)))
        payload["output_filename"] = output_filename
        return payload

    def write_settings_file(self, settings_payload: dict, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings_payload, indent=2), encoding="utf-8")
        return str(path)

    def build_command(self, *, env_name: str, settings_path: str, output_dir: str) -> list[str]:
        return [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-u",
            "wgp.py",
            "--process",
            settings_path,
            "--output-dir",
            output_dir,
        ]

    def parse_output_path_from_log(self, line: str, *, root_dir: str, output_dir: str) -> str:
        match = re.search(r"(?:New video|Postprocessed video|Remuxed Video|New image|Image) saved to Path:\s*(.+)", line, re.IGNORECASE)
        if not match:
            return ""
        raw = match.group(1).strip()
        candidate = Path(raw)
        if candidate.is_absolute():
            return str(candidate)
        root_candidate = Path(root_dir) / candidate
        if root_candidate.exists():
            return str(root_candidate.resolve())
        out_candidate = Path(output_dir) / candidate.name
        if out_candidate.exists():
            return str(out_candidate.resolve())
        return str(root_candidate.resolve())

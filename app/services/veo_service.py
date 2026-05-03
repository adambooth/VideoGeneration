from __future__ import annotations

import hashlib
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class VeoService:
    DEFAULT_MODEL = "veo-3.1-lite-generate-preview"
    DEFAULT_ASPECT_RATIO = "9:16"
    DEFAULT_RESOLUTION = "720p"
    DEFAULT_DURATION_SECONDS = 8

    def generate_clip(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        negative_prompt: str,
        reference_image_path: str,
        output_path: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        duration_seconds: int = DEFAULT_DURATION_SECONDS,
        resolution: str = DEFAULT_RESOLUTION,
        retries: int = 2,
    ) -> str:
        if not api_key.strip():
            raise ValueError("Gemini API key is required for Veo generation.")
        if not prompt.strip():
            raise ValueError("Veo prompt is empty.")

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                clip_path = self._generate_via_python_sdk(
                    api_key=api_key.strip(),
                    model=model.strip() or self.DEFAULT_MODEL,
                    prompt=prompt.strip(),
                    negative_prompt=negative_prompt.strip(),
                    reference_image_path=reference_image_path.strip(),
                    output_path=output_path,
                    aspect_ratio=aspect_ratio,
                    duration_seconds=duration_seconds,
                    resolution=resolution,
                )
                self._validate_video_file(Path(clip_path))
                self._wait_for_file_release(Path(clip_path))
                return clip_path
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(5, attempt * 1.5))
        raise RuntimeError(f"Veo clip generation failed after retries: {last_error}")

    def build_prompt(
        self,
        *,
        visual_style: str,
        scene_prompt: str,
        dialogue_cue: str,
        negative_prompt: str,
        camera_style: str,
        style_notes: str,
        character_mode: str = "Auto",
        has_reference_image: bool = False,
    ) -> str:
        style_prefix = self._style_prefix(visual_style)
        avoid_clause = ""
        if negative_prompt.strip():
            avoid_clause = f"Avoid: {negative_prompt.strip()}."
        character_clause = self._character_clause(character_mode, has_reference_image)
        spoken_clause = self._spoken_clause(dialogue_cue, character_mode, has_reference_image)
        parts = [
            scene_prompt.strip(),
            spoken_clause,
            camera_style.strip(),
            style_prefix,
            style_notes.strip(),
            character_clause,
            avoid_clause,
            "No subtitles, no captions, no text overlays, no logos, narration-only visual storytelling.",
            "Vertical short-form storytelling clip",
        ]
        return ". ".join(part for part in parts if part).strip()

    def _generate_via_python_sdk(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        negative_prompt: str,
        reference_image_path: str,
        output_path: str,
        aspect_ratio: str,
        duration_seconds: int,
        resolution: str,
    ) -> str:
        sdk = self._import_genai()
        types = self._import_genai_types()
        client = sdk.Client(api_key=api_key)

        source_image = self._build_source_image(types, reference_image_path)
        config = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=resolution,
        )

        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=source_image,
            config=config,
        )

        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        if not getattr(operation, "response", None):
            raise RuntimeError("Veo operation finished without a response payload.")
        generated_videos = getattr(operation.response, "generated_videos", []) or []
        if not generated_videos:
            raise RuntimeError("Veo did not return a generated video.")
        generated_video = generated_videos[0]
        client.files.download(file=generated_video.video)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="avc-veo-") as temp_dir:
            temp_output = Path(temp_dir) / output_file.name
            generated_video.video.save(str(temp_output))
            self._validate_video_file(temp_output)
            self._wait_for_file_release(temp_output)
            if output_file.exists():
                output_file.unlink()
            shutil.copy2(temp_output, output_file)
        return output_path

    def generate_poster(self, video_path: str, output_path: str) -> str:
        if not shutil.which("ffmpeg"):
            return ""
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                "thumbnail,scale=540:-1",
                "-frames:v",
                "1",
                output_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return ""
        return output_path

    def can_generate(self) -> tuple[bool, str]:
        try:
            self._import_genai()
            self._import_genai_types()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return True, "Ready"

    def _validate_video_file(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError("Veo did not produce a video file.")
        if path.stat().st_size < 8192:
            raise RuntimeError("Generated Veo video file is too small to be valid.")

    def reference_image_signature(self, reference_image_path: str) -> str:
        candidate = Path(reference_image_path.strip())
        if not candidate.exists() or not candidate.is_file():
            return ""
        hasher = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:24]

    def _wait_for_file_release(self, path: Path, timeout_seconds: float = 12.0) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with path.open("rb"):
                    return
            except OSError as exc:  # noqa: PERF203
                last_error = exc
                time.sleep(0.35)
        if last_error:
            raise last_error

    def _style_prefix(self, visual_style: str) -> str:
        presets = {
            "Sketchbook Storytelling": "Stylized 2D sketchbook animation, hand-drawn ink lines, cinematic lighting, storybook motion",
            "2D Cartoon Storytelling": "Stylized 2D cartoon storytelling, animated illustration, expressive motion, cinematic composition",
            "Sketchbook Noir": "Moody 2D noir sketchbook animation, rain, shadows, inked outlines, dramatic cinematic framing",
            "Stylized Documentary": "Stylized illustrated documentary motion, clean graphic storytelling, cinematic b-roll composition",
            "Clean Explainer Illustration": "Clean illustrated explainer animation, polished 2D design, dynamic camera movement",
        }
        return presets.get(visual_style, presets["Sketchbook Storytelling"])

    def _character_clause(self, character_mode: str, has_reference_image: bool) -> str:
        if not has_reference_image:
            return ""
        if character_mode == "Two Character Conversation":
            return (
                "Use the uploaded image as a master two-character reference. Preserve both characters, their positions, proportions, outfit logic, and identity continuity across scenes. "
                "If one character speaks first and the other answers, show clear turn-taking, visible reaction beats, and realistic conversational eye-lines. "
                "Keep both characters on-screen through the ending of the shot. Do not let the final moment drift to an empty background or lose the characters from frame."
            )
        base = (
            "Use the provided reference character in every scene. Preserve the same face, silhouette, outfit language, and visual identity across the whole sequence."
        )
        return (
            f"{base} The character should carry the whole scene visually with expressive motion, reactions, and clear performance beats. "
            "Keep the character as the clear main subject, framed near the center of the screen for the full shot. "
            "Do not let the camera drift away from the character, do not push the character to the edge of frame, and do not cut to empty environment details. "
            "End the shot with the character still centered and clearly visible on-screen."
        )

    def _spoken_clause(self, dialogue_cue: str, character_mode: str, has_reference_image: bool) -> str:
        if character_mode == "Two Character Conversation":
            spoken = dialogue_cue.strip().replace("\n", " ")
            if not spoken:
                return ""
            return (
                f'Conversation beat: {spoken}. Use realistic lip synchronization for whichever character is speaking each quoted line, '
                "with clear pauses, reactions, and strict turn-taking between the two characters. "
                "Only one person speaks at a time, and there must be a short natural pause between every response. "
                "The final beat of the clip must still show the characters clearly on-screen."
            )
        spoken = self._normalize_spoken_dialogue(dialogue_cue)
        if not spoken:
            return ""
        if has_reference_image:
            return (
                f'The character looks into the camera and says, "{spoken}" with natural expressive delivery and realistic lip synchronization. '
                "The quoted words are the exact spoken dialogue. Keep the character centered as the dominant subject, and the final beat of the clip must still show the character clearly on-screen."
            )
        return f'Primary spoken line: "{spoken}"'

    def _normalize_spoken_dialogue(self, dialogue_cue: str) -> str:
        raw = dialogue_cue.strip()
        if not raw:
            return ""
        quoted_matches = list(re.finditer(r'"([^"]+)"', raw))
        if quoted_matches:
            spoken = " ".join(match.group(1).strip() for match in quoted_matches if match.group(1).strip())
            return spoken.replace('"', "").strip()
        if ":" in raw:
            raw = raw.split(":", 1)[1].strip()
        return raw.replace('"', "").strip()

    def _build_source_image(self, types_module, reference_image_path: str):
        candidate = Path(reference_image_path.strip())
        if not candidate.exists() or not candidate.is_file():
            return None
        mime_type, _ = mimetypes.guess_type(candidate.name)
        return types_module.Image.from_file(
            location=str(candidate),
            mime_type=mime_type or "image/png",
        )

    def _import_genai(self):
        try:
            from google import genai  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Install the google-genai package to use Veo generation.") from exc
        return genai

    def _import_genai_types(self):
        try:
            from google.genai import types  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Install the google-genai package to use Veo generation.") from exc
        return types

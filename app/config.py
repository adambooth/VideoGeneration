from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

if os.name == "nt":
    DEFAULT_EXPORT_DIR = str((Path.cwd() / "Projects").resolve())
    DEFAULT_GENERATED_VIDEO_DIR = str((Path.cwd() / "GeneratedVideos").resolve())
    DEFAULT_DEEVID_PROFILE_DIR = str((Path.cwd() / ".deevid_profile").resolve())
    DEFAULT_WAN2GP_ROOT = r"E:\wan2gp\Wan2GP"
    DEFAULT_WAN2GP_TEMPLATE = str((Path.cwd() / "wanGPJSONSettings.json").resolve())
    DEFAULT_FLUX_TEMPLATE = str((Path.cwd() / "wanGPJSONFluxKlein9BSettings.json").resolve())
else:
    DEFAULT_EXPORT_DIR = "/workspace/AutomatedVideoCreator-Transfer/FinalVideos"
    DEFAULT_GENERATED_VIDEO_DIR = "/workspace/AutomatedVideoCreator-Transfer/FinalVideos"
    DEFAULT_DEEVID_PROFILE_DIR = "/workspace/AutomatedVideoCreator-Transfer/.deevid_profile"
    DEFAULT_WAN2GP_ROOT = "/workspace/Wan2GP"
    DEFAULT_WAN2GP_TEMPLATE = "/workspace/AutomatedVideoCreator-Transfer/wanGPJSONSettings.json"
    DEFAULT_FLUX_TEMPLATE = "/workspace/AutomatedVideoCreator-Transfer/wanGPJSONFluxKlein9BSettings.json"


def _normalize_gemini_model(model: str, label: str) -> tuple[str, str]:
    normalized_model = (model or "").strip() or "gemini-2.5-flash"
    normalized_label = (label or "").strip() or "Gemini 2.5 Flash"
    if normalized_model == "gemini-2.5-pro":
        return "gemini-2.5-flash", "Gemini 2.5 Flash"
    if normalized_label == "Gemini 2.5 Pro":
        return normalized_model if normalized_model != "gemini-2.5-pro" else "gemini-2.5-flash", "Custom"
    return normalized_model, normalized_label


def _normalize_character_mode(value: str) -> str:
    normalized = (value or "").strip()
    if normalized in {"Auto", "Speaking / Presenter", "Reenacting"}:
        return "Solo"
    if normalized == "Two Character Conversation":
        return normalized
    return normalized or "Solo"


@dataclass
class VideoSettings:
    output_quality: str = "High"
    fps: int = 30
    length_target: int = 30
    platform: str = "YouTube Shorts"
    visual_provider: str = "Veo + Stock"
    music_style: str = "Suspense"

    def resolution(self) -> tuple[int, int]:
        return 1080, 1920


@dataclass
class AppSettings:
    workflow_mode: str = "Scene Automation"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_model_label: str = "Gemini 2.5 Flash"
    gemini_custom_model: str = ""
    veo_enabled: bool = True
    veo_model: str = "veo-3.1-lite-generate-preview"
    veo_planning_mode: str = "Auto"
    veo_visual_style: str = "Sketchbook Storytelling"
    veo_reference_image_path: str = ""
    veo_character_mode: str = "Solo"
    veo_continuity_mode: str = "Chain Last Frame"
    generated_video_folder: str = DEFAULT_GENERATED_VIDEO_DIR
    wan2gp_root_dir: str = DEFAULT_WAN2GP_ROOT
    wan2gp_env_name: str = "wan2gp"
    wan2gp_template_path: str = DEFAULT_WAN2GP_TEMPLATE
    flux_template_path: str = DEFAULT_FLUX_TEMPLATE
    wan2gp_continuity_mode: str = "Same Start Image"
    deevid_profile_dir: str = DEFAULT_DEEVID_PROFILE_DIR
    narration_engine: str = "Edge TTS"
    edge_tts_voice: str = "en-US-GuyNeural"
    edge_tts_rate: int = 5
    edge_tts_volume: int = 0
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_voice_id_a: str = ""
    elevenlabs_voice_id_b: str = ""
    serpapi_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    content_mode: str = "General Viral Mode"
    export_folder: str = DEFAULT_EXPORT_DIR
    video: VideoSettings = field(default_factory=VideoSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        video_data = data.get("video", {})
        gemini_model, gemini_label = _normalize_gemini_model(
            data.get("gemini_model", "gemini-2.5-flash"),
            data.get("gemini_model_label", "Gemini 2.5 Flash"),
        )
        return cls(
            gemini_api_key=data.get("gemini_api_key", ""),
            workflow_mode=data.get("workflow_mode", "Scene Automation"),
            gemini_model=gemini_model,
            gemini_model_label=gemini_label,
            gemini_custom_model=data.get("gemini_custom_model", ""),
            veo_enabled=bool(data.get("veo_enabled", True)),
            veo_model=data.get("veo_model", "veo-3.1-lite-generate-preview"),
            veo_planning_mode=data.get("veo_planning_mode", "Auto"),
            veo_visual_style=data.get("veo_visual_style", "Sketchbook Storytelling"),
            veo_reference_image_path=data.get("veo_reference_image_path", ""),
            veo_character_mode=_normalize_character_mode(data.get("veo_character_mode", "Solo")),
            veo_continuity_mode=data.get("veo_continuity_mode", "Chain Last Frame"),
            generated_video_folder=data.get("generated_video_folder", DEFAULT_GENERATED_VIDEO_DIR),
            wan2gp_root_dir=data.get("wan2gp_root_dir", DEFAULT_WAN2GP_ROOT),
            wan2gp_env_name=data.get("wan2gp_env_name", "wan2gp"),
            wan2gp_template_path=data.get("wan2gp_template_path", DEFAULT_WAN2GP_TEMPLATE),
            flux_template_path=data.get("flux_template_path", DEFAULT_FLUX_TEMPLATE),
            wan2gp_continuity_mode=data.get("wan2gp_continuity_mode", "Same Start Image"),
            deevid_profile_dir=data.get("deevid_profile_dir", DEFAULT_DEEVID_PROFILE_DIR),
            narration_engine=data.get("narration_engine", "Edge TTS"),
            edge_tts_voice=data.get("edge_tts_voice", "en-US-GuyNeural"),
            edge_tts_rate=int(data.get("edge_tts_rate", 5)),
            edge_tts_volume=int(data.get("edge_tts_volume", 0)),
            elevenlabs_api_key=data.get("elevenlabs_api_key", ""),
            elevenlabs_voice_id=data.get("elevenlabs_voice_id", ""),
            elevenlabs_voice_id_a=data.get("elevenlabs_voice_id_a", ""),
            elevenlabs_voice_id_b=data.get("elevenlabs_voice_id_b", ""),
            serpapi_api_key=data.get("serpapi_api_key", ""),
            pexels_api_key=data.get("pexels_api_key", ""),
            pixabay_api_key=data.get("pixabay_api_key", ""),
            content_mode=data.get("content_mode", "General Viral Mode"),
            export_folder=data.get("export_folder", DEFAULT_EXPORT_DIR),
            video=VideoSettings(
                output_quality=video_data.get("output_quality", "High"),
                fps=int(video_data.get("fps", 30)),
                length_target=int(video_data.get("length_target", 30)),
                platform=video_data.get("platform", "YouTube Shorts"),
                visual_provider=video_data.get("visual_provider", "Veo + Stock"),
                music_style=video_data.get("music_style", "Suspense"),
            ),
        )

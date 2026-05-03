from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class PipelineStage(str, Enum):
    IDLE = "Idle"
    EXTRACT = "Extract Content"
    SCRIPT = "Generate Script"
    VOICE = "Generate Voiceover"
    VISUALS = "Generate Visual Plan"
    RENDER = "Render Final Video"
    COMPLETE = "Complete"
    ERROR = "Error"


APPROVAL_STAGES = {
    PipelineStage.SCRIPT,
    PipelineStage.VOICE,
    PipelineStage.VISUALS,
    PipelineStage.RENDER,
}


@dataclass
class SceneSpec:
    headline: str
    supporting_text: str
    visual_keywords: list[str]
    mood: str
    narration_text: str = ""
    action_prompt: str = ""
    audio_dialogue_cue: str = ""
    duration_hint: float = 4.0
    purpose: str = ""
    negative_prompt: str = ""
    camera_style: str = ""
    style_notes: str = ""
    asset_path: str = ""
    asset_type: str = "image"
    visual_query: str = ""
    overlay_text: str = ""
    source_name: str = ""
    source_url: str = ""
    source_credit: str = ""
    poster_path: str = ""
    generated_prompt: str = ""
    generated_clip_key: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReuseEstimate:
    total_scene_count: int = 0
    reused_clip_count: int = 0
    new_clip_count: int = 0
    estimated_raw_seconds: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScriptPackage:
    title: str
    description: str
    hashtags: list[str]
    call_to_action: str
    spoken_script: str
    visual_style: str
    intro_script: str = ""
    scenes: list[SceneSpec] = field(default_factory=list)
    stock_footage_tags: list[str] = field(default_factory=list)
    planning_mode: str = "Auto"
    reuse_estimate: ReuseEstimate = field(default_factory=ReuseEstimate)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scenes"] = [scene.to_dict() for scene in self.scenes]
        payload["reuse_estimate"] = self.reuse_estimate.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptPackage":
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            hashtags=list(data.get("hashtags", [])),
            call_to_action=data.get("call_to_action", ""),
            spoken_script=data.get("spoken_script", ""),
            visual_style=data.get("visual_style", ""),
            intro_script=data.get("intro_script", ""),
            scenes=[SceneSpec(**scene) for scene in data.get("scenes", [])],
            stock_footage_tags=list(data.get("stock_footage_tags", [])),
            planning_mode=data.get("planning_mode", "Auto"),
            reuse_estimate=ReuseEstimate(**data.get("reuse_estimate", {})),
        )


@dataclass
class ProjectState:
    project_name: str
    project_dir: str
    export_dir: str
    urls: list[str]
    source_mode: str = "url"
    source_prompt: str = ""
    character_brief: str = ""
    character_a_brief: str = ""
    character_b_brief: str = ""
    story_type: str = ""
    facts_topic: str = ""
    fact_items: list[str] = field(default_factory=list)
    fact_1: str = ""
    fact_2: str = ""
    fact_3: str = ""
    stage: str = PipelineStage.IDLE.value
    source_text: str = ""
    source_summary: str = ""
    script_package: dict = field(default_factory=dict)
    visual_plan: list[dict] = field(default_factory=list)
    generated_video_library_hits: list[dict] = field(default_factory=list)
    voice_path: str = ""
    voice_segment_paths: list[str] = field(default_factory=list)
    voice_segment_durations: list[float] = field(default_factory=list)
    source_scene_audio_paths: list[str] = field(default_factory=list)
    converted_scene_audio_paths: list[str] = field(default_factory=list)
    continuity_frame_paths: list[str] = field(default_factory=list)
    music_path: str = ""
    final_video_path: str = ""
    last_error: str = ""

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectState":
        return cls(
            project_name=data["project_name"],
            project_dir=data["project_dir"],
            export_dir=data["export_dir"],
            urls=list(data.get("urls", [])),
            source_mode=data.get("source_mode", "url"),
            source_prompt=data.get("source_prompt", ""),
            character_brief=data.get("character_brief", ""),
            character_a_brief=data.get("character_a_brief", ""),
            character_b_brief=data.get("character_b_brief", ""),
            story_type=data.get("story_type", ""),
            facts_topic=data.get("facts_topic", ""),
            fact_items=list(data.get("fact_items", []))
            or [
                value
                for value in [data.get("fact_1", ""), data.get("fact_2", ""), data.get("fact_3", "")]
                if str(value).strip()
            ],
            fact_1=data.get("fact_1", ""),
            fact_2=data.get("fact_2", ""),
            fact_3=data.get("fact_3", ""),
            stage=data.get("stage", PipelineStage.IDLE.value),
            source_text=data.get("source_text", ""),
            source_summary=data.get("source_summary", ""),
            script_package=dict(data.get("script_package", {})),
            visual_plan=list(data.get("visual_plan", [])),
            generated_video_library_hits=list(data.get("generated_video_library_hits", [])),
            voice_path=data.get("voice_path", ""),
            voice_segment_paths=list(data.get("voice_segment_paths", [])),
            voice_segment_durations=[float(value) for value in data.get("voice_segment_durations", [])],
            source_scene_audio_paths=list(data.get("source_scene_audio_paths", [])),
            converted_scene_audio_paths=list(data.get("converted_scene_audio_paths", [])),
            continuity_frame_paths=list(data.get("continuity_frame_paths", [])),
            music_path=data.get("music_path", ""),
            final_video_path=data.get("final_video_path", ""),
            last_error=data.get("last_error", ""),
        )

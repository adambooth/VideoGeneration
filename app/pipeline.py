from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.config import AppSettings
from app.models import APPROVAL_STAGES, PipelineStage, ProjectState, ScriptPackage
from app.project_store import ProjectStore
from app.services.audio_service import AudioService
from app.services.content_extractor import ContentExtractor
from app.services.deevid_service import DeeVidService
from app.services.edge_tts_service import EdgeTTSService
from app.services.elevenlabs_service import ElevenLabsService
from app.services.generated_video_library import GeneratedVideoLibrary
from app.services.gemini_service import GeminiService
from app.services.render_service import RenderService
from app.services.veo_service import VeoService
from app.services.visual_service import VisualService


class PipelineController(QObject):
    log_message = Signal(str)
    stage_changed = Signal(str)
    preview_ready = Signal(str, str)
    approval_required = Signal(str)
    pipeline_finished = Signal(str)
    project_loaded = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.extractor = ContentExtractor()
        self.deevid_service = DeeVidService()
        self.gemini_service = GeminiService()
        self.edge_tts_service = EdgeTTSService()
        self.elevenlabs_service = ElevenLabsService()
        self.veo_service = VeoService()
        self.visual_service = VisualService()
        self.audio_service = AudioService()
        self.render_service = RenderService()
        self.project_store = ProjectStore()

        self._condition = threading.Condition()
        self._approval_action = "continue"
        self._cancel_requested = False
        self._pause_requested = False
        self._current_state: ProjectState | None = None
        self._current_settings: AppSettings | None = None
        self._callbacks: dict[str, list] = {
            "log": [],
            "stage": [],
            "preview": [],
            "approval": [],
            "finished": [],
            "project_loaded": [],
        }

    def register_callback(self, event_name: str, callback) -> None:
        if event_name not in self._callbacks:
            raise ValueError(f"Unsupported callback event: {event_name}")
        self._callbacks[event_name].append(callback)

    def _notify_callbacks(self, event_name: str, *args) -> None:
        for callback in self._callbacks.get(event_name, []):
            try:
                callback(*args)
            except Exception:
                continue

    def _emit_log(self, message: str) -> None:
        self.log_message.emit(message)
        self._notify_callbacks("log", message)

    def _emit_stage(self, stage: str) -> None:
        self.stage_changed.emit(stage)
        self._notify_callbacks("stage", stage)

    def _emit_preview(self, kind: str, content: str) -> None:
        self.preview_ready.emit(kind, content)
        self._notify_callbacks("preview", kind, content)

    def _emit_approval(self, stage: str) -> None:
        self.approval_required.emit(stage)
        self._notify_callbacks("approval", stage)

    def _emit_finished(self, message: str) -> None:
        self.pipeline_finished.emit(message)
        self._notify_callbacks("finished", message)

    def _emit_project_loaded(self, payload: dict) -> None:
        self.project_loaded.emit(payload)
        self._notify_callbacks("project_loaded", payload)

    @Slot(list, str, dict)
    def start_new_project(self, urls: list[str], export_dir: str, settings_data: dict) -> None:
        self._cancel_requested = False
        self._pause_requested = False
        self._approval_action = "continue"
        self._current_settings = AppSettings.from_dict(settings_data)
        self._current_state = self.project_store.create_project(export_dir, urls)
        self._current_state.source_mode = settings_data.get("source_mode", "url")
        self._current_state.source_prompt = settings_data.get("source_prompt", "")
        self._current_state.character_brief = settings_data.get("character_brief", "")
        self._current_state.character_a_brief = settings_data.get("character_a_brief", "")
        self._current_state.character_b_brief = settings_data.get("character_b_brief", "")
        self._current_state.story_type = settings_data.get("story_type", "")
        self._current_state.facts_topic = settings_data.get("facts_topic", "")
        self._current_state.fact_items = [
            str(value).strip()
            for value in settings_data.get("fact_items", [])
            if str(value).strip()
        ]
        self._current_state.fact_1 = settings_data.get("fact_1", "")
        self._current_state.fact_2 = settings_data.get("fact_2", "")
        self._current_state.fact_3 = settings_data.get("fact_3", "")
        self.project_store.save_state(self._current_state)
        self._emit_project_loaded(self._current_state.to_dict())
        self._run_pipeline(resume=False)

    @Slot(str, dict)
    def resume_project(self, project_dir: str, settings_data: dict) -> None:
        self._cancel_requested = False
        self._pause_requested = False
        self._approval_action = "continue"
        self._current_settings = AppSettings.from_dict(settings_data)
        self._current_state = self.project_store.load_project(project_dir)
        self._emit_project_loaded(self._current_state.to_dict())
        self._run_pipeline(resume=True)

    @Slot(str, dict, str)
    def rollback_project(self, project_dir: str, settings_data: dict, target_stage: str) -> None:
        self._cancel_requested = False
        self._pause_requested = False
        self._approval_action = "continue"
        self._current_settings = AppSettings.from_dict(settings_data)
        self._current_state = self.project_store.load_project(project_dir)
        target = self._stage_from_value(target_stage)
        self._rollback_state_to_stage(self._current_state, target, self._current_settings)
        self.project_store.save_state(self._current_state)
        self._emit_project_loaded(self._current_state.to_dict())
        self._run_pipeline(resume=True)

    @Slot()
    def request_pause(self) -> None:
        self._emit_log("Pause requested. The pipeline will pause at the next safe checkpoint.")
        self._pause_requested = True

    @Slot()
    def request_stop(self) -> None:
        self._emit_log("Stop requested. The current project will halt safely.")
        self._cancel_requested = True
        self._release_wait("cancel")

    @Slot()
    def continue_after_approval(self) -> None:
        self._pause_requested = False
        self._release_wait("continue")

    @Slot()
    def regenerate_current_step(self) -> None:
        self._pause_requested = False
        self._release_wait("regenerate")

    @Slot(dict)
    def reload_settings(self, settings_data: dict) -> None:
        self._current_settings = AppSettings.from_dict(settings_data)

    def _run_pipeline(self, resume: bool) -> None:
        state = self._require_state()
        settings = self._require_settings()

        try:
            ordered_stages = self._ordered_stages(settings)

            start_index = 0
            if resume and state.stage in [stage.value for stage in ordered_stages]:
                start_index = [stage.value for stage in ordered_stages].index(state.stage)
            elif resume and state.stage == PipelineStage.ERROR.value:
                start_index = self._infer_resume_index(state)
            elif resume and state.stage == PipelineStage.COMPLETE.value:
                self._emit_preview("video", state.final_video_path)
                self._emit_stage(PipelineStage.COMPLETE.value)
                self._emit_finished("Project is already complete.")
                return

            index = start_index
            while index < len(ordered_stages):
                self._checkpoint_pause()
                stage = ordered_stages[index]
                if self._cancel_requested:
                    raise RuntimeError("Pipeline stopped by user.")

                self._set_stage(stage)
                if stage == PipelineStage.EXTRACT:
                    self._run_extract(state)
                elif stage == PipelineStage.SCRIPT:
                    self._run_script(state, settings)
                elif stage == PipelineStage.VOICE:
                    self._run_voice(state, settings)
                elif stage == PipelineStage.VISUALS:
                    self._run_visuals(state)
                elif stage == PipelineStage.RENDER:
                    self._run_render(state, settings)

                if stage in APPROVAL_STAGES and not self._stage_handles_inline_approvals(stage, settings) and not (stage == PipelineStage.VOICE and state.source_mode == "facts"):
                    action = self._await_approval(stage)
                    if action == "regenerate":
                        continue
                    if action == "cancel":
                        raise RuntimeError("Pipeline stopped by user.")
                index += 1

            self._set_stage(PipelineStage.COMPLETE)
            self._emit_finished("Project completed successfully.")
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)

    def _run_extract(self, state: ProjectState) -> None:
        if state.source_mode == "scene":
            self._emit_log("Preparing scene prompt package from manual concept input.")
            settings = self._require_settings()
            character_image = settings.veo_reference_image_path.strip()
            two_character_mode = settings.veo_character_mode == "Two Character Conversation"
            prompt_parts = [
                f"Story Type: {state.story_type}".strip(),
                "Character Image: uploaded and should be used as the recurring on-screen character.".strip() if character_image else "",
                f"Character A: {state.character_a_brief}".strip() if two_character_mode and state.character_a_brief.strip() else "",
                f"Character B: {state.character_b_brief}".strip() if two_character_mode and state.character_b_brief.strip() else "",
                f"Character Notes: {state.character_brief}".strip() if not two_character_mode and state.character_brief.strip() else "",
                f"Concept: {state.source_prompt}".strip(),
            ]
            state.source_text = "\n".join(part for part in prompt_parts if part and not part.endswith(":"))
            state.source_summary = state.source_text
            self.project_store.write_text_file(state, "source.txt", state.source_text)
            self.project_store.save_state(state)
            self._emit_preview("text", state.source_text)
            return
        if state.source_mode == "facts":
            self._emit_log("Preparing fact-mode prompt package from manual fact input.")
            settings = self._require_settings()
            fact_items = [fact.strip() for fact in state.fact_items if fact.strip()]
            if not fact_items:
                fact_items = [
                    fact.strip()
                    for fact in [state.fact_1, state.fact_2, state.fact_3]
                    if str(fact).strip()
                ]
            prompt_parts = [
                "FACT MODE INPUT",
                f"Topic: {state.facts_topic}".strip(),
                f"Story Type: {state.story_type}".strip(),
                f"Character Notes: {state.character_brief}".strip() if state.character_brief.strip() else "",
                "Character Image: uploaded and should be used as the recurring on-screen fact host." if settings.veo_reference_image_path.strip() else "",
                f"Create exactly {len(fact_items) + 1} scenes total.",
                "Scene 1 is the intro hook clip.",
                *[f"Scene {index + 2} covers Fact {index + 1} only." for index in range(len(fact_items))],
                "One fact per scene after the intro. Do not merge facts.",
                *[f"Fact {index + 1}: {fact}".strip() for index, fact in enumerate(fact_items)],
            ]
            state.source_text = "\n".join(part for part in prompt_parts if part and not part.endswith(":"))
            state.source_summary = state.source_text
            self.project_store.write_text_file(state, "source.txt", state.source_text)
            self.project_store.save_state(state)
            self._emit_preview("text", state.source_text)
            return
        self._emit_log("Extracting readable text from supplied URLs.")
        extracted = self.extractor.extract(state.urls)
        state.source_text = extracted.combined_text
        state.source_summary = extracted.summary
        self.project_store.write_text_file(state, "source.txt", state.source_text)
        self.project_store.save_state(state)
        self._emit_preview("text", extracted.summary + "\n\n" + extracted.combined_text[:5000])

    def _run_script(self, state: ProjectState, settings: AppSettings) -> None:
        self._emit_log("Generating short-form script package with Gemini.")
        content_mode = "General Viral Mode"
        planning_mode = "Auto"
        character_mode = settings.veo_character_mode
        if state.source_mode == "facts":
            content_mode = "Facts / Listicle Mode"
            planning_mode = "List / Topic Mode"
            character_mode = "Solo"
        script_package = self.gemini_service.generate_script(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            content_mode=content_mode,
            length_target=settings.video.length_target,
            source_text=state.source_text,
            planning_mode=planning_mode,
            visual_style="Reference-Driven",
            character_mode=character_mode,
        )
        library = GeneratedVideoLibrary(settings.generated_video_folder)
        script_package.reuse_estimate = self._build_reuse_estimate(script_package, settings, library)
        state.script_package = script_package.to_dict()
        self.project_store.write_text_file(state, "script.txt", script_package.spoken_script)
        metadata_text = self._format_metadata(script_package)
        self.project_store.write_text_file(state, "metadata.txt", metadata_text)
        self.project_store.save_state(state)
        self._emit_preview("script", metadata_text + "\n\nSCRIPT\n" + script_package.spoken_script)

    def _run_voice(self, state: ProjectState, settings: AppSettings) -> None:
        if settings.video.visual_provider == "Veo + Stock":
            if state.source_mode == "facts":
                self._emit_log("Extracting Veo fact clip audio and converting it with one ElevenLabs voice.")
                if not settings.elevenlabs_api_key.strip():
                    raise ValueError("ElevenLabs API key is required for Facts mode voice changing.")
                if not settings.elevenlabs_voice_id.strip():
                    raise ValueError("ElevenLabs Voice ID is required for Facts mode voice changing.")
                source_dir = Path(state.project_dir) / "scene_audio_source"
                converted_dir = Path(state.project_dir) / "scene_audio_converted"
                source_dir.mkdir(parents=True, exist_ok=True)
                converted_dir.mkdir(parents=True, exist_ok=True)
                state.source_scene_audio_paths = []
                state.converted_scene_audio_paths = []
                veo_scenes = [scene for scene in state.visual_plan if str(scene.get("segment_kind", "veo")) == "veo"]
                for index, scene in enumerate(veo_scenes, start=1):
                    asset_path = str(scene.get("asset_path", "")).strip()
                    if not asset_path:
                        raise ValueError(f"Missing Veo video for fact scene {index}.")
                    source_audio = str(source_dir / f"scene_{index:02d}.mp3")
                    converted_audio = str(converted_dir / f"scene_{index:02d}.mp3")
                    self.render_service.extract_audio(asset_path, source_audio)
                    state.source_scene_audio_paths.append(source_audio)
                    state.converted_scene_audio_paths.append(
                        self.elevenlabs_service.convert_speech(
                            api_key=settings.elevenlabs_api_key,
                            voice_id=settings.elevenlabs_voice_id,
                            input_audio_path=source_audio,
                            output_path=converted_audio,
                        )
                    )
                state.voice_segment_paths = list(state.converted_scene_audio_paths)
                state.voice_segment_durations = [
                    self.render_service.get_audio_duration(segment_path)
                    for segment_path in state.converted_scene_audio_paths
                    if Path(segment_path).exists()
                ]
                state.voice_path = self._concat_voice_segments(
                    state.converted_scene_audio_paths,
                    str(Path(state.project_dir) / "voice.mp3"),
                )
                self.project_store.save_state(state)
                self._emit_preview("audio", state.voice_path)
                return
            self._emit_log("Extracting Veo scene audio and converting it with ElevenLabs Voice Changer.")
            if not settings.elevenlabs_api_key.strip():
                raise ValueError("ElevenLabs API key is required for Veo voice changing.")
            if settings.veo_character_mode == "Two Character Conversation":
                if not settings.elevenlabs_voice_id_a.strip():
                    raise ValueError("ElevenLabs Voice ID A is required for two-character Veo voice changing.")
                if not settings.elevenlabs_voice_id_b.strip():
                    raise ValueError("ElevenLabs Voice ID B is required for two-character Veo voice changing.")
            elif not settings.elevenlabs_voice_id.strip():
                raise ValueError("ElevenLabs Voice ID is required for Veo voice changing.")
            source_dir = Path(state.project_dir) / "scene_audio_source"
            converted_dir = Path(state.project_dir) / "scene_audio_converted"
            turn_source_dir = Path(state.project_dir) / "scene_audio_turns" / "source"
            turn_converted_dir = Path(state.project_dir) / "scene_audio_turns" / "converted"
            source_dir.mkdir(parents=True, exist_ok=True)
            converted_dir.mkdir(parents=True, exist_ok=True)
            turn_source_dir.mkdir(parents=True, exist_ok=True)
            turn_converted_dir.mkdir(parents=True, exist_ok=True)
            state.source_scene_audio_paths = []
            state.converted_scene_audio_paths = []
            for index, scene in enumerate(state.visual_plan, start=1):
                asset_path = str(scene.get("asset_path", "")).strip()
                if not asset_path:
                    raise ValueError(f"Missing Veo video for scene {index}.")
                source_audio = str(source_dir / f"scene_{index:02d}.mp3")
                converted_audio = str(converted_dir / f"scene_{index:02d}.mp3")
                self.render_service.extract_audio(asset_path, source_audio)
                state.source_scene_audio_paths.append(source_audio)
                if settings.veo_character_mode == "Two Character Conversation":
                    turns = self._parse_dialogue_turns(
                        str(scene.get("audio_dialogue_cue") or scene.get("narration_text") or ""),
                        state,
                    )
                    state.converted_scene_audio_paths.append(
                        self._convert_two_character_scene_audio(
                            source_audio_path=source_audio,
                            output_path=converted_audio,
                            scene_prefix=f"scene_{index:02d}",
                            turns=turns,
                            voice_id_a=settings.elevenlabs_voice_id_a.strip(),
                            voice_id_b=settings.elevenlabs_voice_id_b.strip(),
                            api_key=settings.elevenlabs_api_key,
                            source_turn_dir=turn_source_dir,
                            converted_turn_dir=turn_converted_dir,
                        )
                    )
                else:
                    state.converted_scene_audio_paths.append(
                        self.elevenlabs_service.convert_speech(
                            api_key=settings.elevenlabs_api_key,
                            voice_id=settings.elevenlabs_voice_id,
                            input_audio_path=source_audio,
                            output_path=converted_audio,
                        )
                    )
            state.voice_segment_paths = list(state.converted_scene_audio_paths)
            state.voice_segment_durations = [
                self.render_service.get_audio_duration(segment_path)
                for segment_path in state.converted_scene_audio_paths
                if Path(segment_path).exists()
            ]
            state.voice_path = self._concat_voice_segments(
                state.converted_scene_audio_paths,
                str(Path(state.project_dir) / "voice.mp3"),
            )
            self.project_store.save_state(state)
            self._emit_preview("audio", state.voice_path)
            return
        script_package = ScriptPackage.from_dict(state.script_package)
        voice_path = str(Path(state.project_dir) / "voice.mp3")
        segment_dir = Path(state.project_dir) / "voice_segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        engine = (settings.narration_engine or "Edge TTS").strip()
        if engine == "Disabled":
            raise ValueError("Narration engine is disabled. Select Edge TTS or ElevenLabs before starting a project.")
        segments = self._build_narration_segments(script_package)
        if engine == "ElevenLabs":
            self._emit_log("Generating narration with ElevenLabs.")
            try:
                state.voice_segment_paths = self._generate_elevenlabs_segments(
                    segments=segments,
                    output_dir=segment_dir,
                    api_key=settings.elevenlabs_api_key,
                    voice_id=settings.elevenlabs_voice_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._emit_log(f"ElevenLabs unavailable. Switched to Edge TTS. Reason: {exc}")
                state.voice_segment_paths = self._generate_edge_segments(
                    segments=segments,
                    output_dir=segment_dir,
                    voice=settings.edge_tts_voice,
                    rate=settings.edge_tts_rate,
                    volume=settings.edge_tts_volume,
                )
        else:
            self._emit_log("Generating narration with Edge TTS.")
            state.voice_segment_paths = self._generate_edge_segments(
                segments=segments,
                output_dir=segment_dir,
                voice=settings.edge_tts_voice,
                rate=settings.edge_tts_rate,
                volume=settings.edge_tts_volume,
            )
        state.voice_path = self._concat_voice_segments(state.voice_segment_paths, voice_path)
        state.voice_segment_durations = [
            self.render_service.get_audio_duration(segment_path)
            for segment_path in state.voice_segment_paths
            if Path(segment_path).exists()
        ]
        self.project_store.save_state(state)
        self._emit_preview("audio", state.voice_path)

    def _run_visuals(self, state: ProjectState) -> None:
        settings = self._require_settings()
        self._emit_log(f"Building visual assets with {settings.video.visual_provider}.")
        script_package = ScriptPackage.from_dict(state.script_package)
        if settings.video.visual_provider == "Veo + Stock":
            state.visual_plan = self._build_veo_visual_assets(state, script_package, settings)
        elif settings.video.visual_provider == "DeeVid Automation":
            state.visual_plan = self._build_deevid_visual_assets(state, script_package, settings)
        else:
            state.visual_plan = self.visual_service.build_scene_assets(state.project_dir, script_package.scenes, settings)
        for index, scene in enumerate(state.visual_plan, start=1):
            poster_path_value = scene.get("poster_path", "")
            local_poster_exists = bool(poster_path_value) and Path(poster_path_value).exists()
            if scene.get("asset_type") == "video" and not local_poster_exists:
                poster_path = str(Path(state.project_dir) / "assets" / f"scene_{index:02d}_poster.jpg")
                scene["poster_path"] = self.visual_service.generate_video_poster(scene["asset_path"], poster_path)
        self.project_store.save_state(state)
        self._emit_preview("visuals", json.dumps(state.visual_plan, indent=2))

    def _run_render(self, state: ProjectState, settings: AppSettings) -> None:
        self._emit_log("Rendering final vertical MP4 with FFmpeg.")
        final_path = str(Path(state.project_dir) / "final.mp4")
        if settings.video.visual_provider == "Veo + Stock":
            if state.source_mode == "facts":
                state.music_path = ""
                state.final_video_path = self.render_service.render_scene_audio_video(
                    project_dir=state.project_dir,
                    scene_assets=state.visual_plan,
                    scene_audio_paths=state.converted_scene_audio_paths,
                    quality=settings.video.output_quality,
                    output_path=final_path,
                )
                self.project_store.save_state(state)
                self._emit_preview("video", state.final_video_path)
                return
            state.music_path = ""
            state.final_video_path = self.render_service.render_scene_audio_video(
                project_dir=state.project_dir,
                scene_assets=state.visual_plan,
                scene_audio_paths=state.converted_scene_audio_paths,
                output_path=final_path,
                quality=settings.video.output_quality,
            )
            self.project_store.save_state(state)
            self._emit_preview("video", state.final_video_path)
            return
        if state.voice_path:
            music_path = str(Path(state.project_dir) / "music.wav")
            voice_duration = self.render_service.get_audio_duration(state.voice_path)
            state.music_path = self.audio_service.generate_background_music(
                music_path,
                voice_duration,
                settings.video.music_style,
            )
        else:
            state.music_path = ""
        state.final_video_path = self.render_service.render_video(
            project_dir=state.project_dir,
            scene_assets=state.visual_plan,
            voice_path=state.voice_path,
            music_path=state.music_path,
            output_path=final_path,
            fps=settings.video.fps,
            quality=settings.video.output_quality,
            voice_segment_durations=state.voice_segment_durations,
        )
        self.project_store.save_state(state)
        self._emit_preview("video", state.final_video_path)

    def _stage_from_value(self, stage_value: str) -> PipelineStage:
        normalized = (stage_value or "").strip()
        for stage in PipelineStage:
            if stage.value == normalized:
                return stage
        raise ValueError(f"Unsupported rollback stage: {stage_value}")

    def _rollback_state_to_stage(
        self,
        state: ProjectState,
        target_stage: PipelineStage,
        settings: AppSettings,
    ) -> None:
        ordered = self._ordered_stages(settings)
        if target_stage not in ordered:
            raise ValueError(f"{target_stage.value} is not available for the current workflow.")

        stage_index = {stage: index for index, stage in enumerate(ordered)}

        def should_reset(stage: PipelineStage) -> bool:
            return stage_index.get(stage, -1) >= stage_index[target_stage]

        if should_reset(PipelineStage.EXTRACT):
            state.source_text = ""
            state.source_summary = ""

        if should_reset(PipelineStage.SCRIPT):
            state.script_package = {}
            state.visual_plan = []
            state.generated_video_library_hits = []

        if should_reset(PipelineStage.VISUALS):
            state.visual_plan = []
            state.generated_video_library_hits = []
            state.continuity_frame_paths = []

        if should_reset(PipelineStage.VOICE):
            state.voice_path = ""
            state.voice_segment_paths = []
            state.voice_segment_durations = []
            state.source_scene_audio_paths = []
            state.converted_scene_audio_paths = []

        if should_reset(PipelineStage.RENDER):
            state.music_path = ""
            state.final_video_path = ""

        state.stage = target_stage.value
        state.last_error = ""

    def _ordered_stages(self, settings: AppSettings) -> list[PipelineStage]:
        if settings.video.visual_provider == "Veo + Stock":
            return [
                PipelineStage.EXTRACT,
                PipelineStage.SCRIPT,
                PipelineStage.VISUALS,
                PipelineStage.VOICE,
                PipelineStage.RENDER,
            ]
        return [
            PipelineStage.EXTRACT,
            PipelineStage.SCRIPT,
            PipelineStage.VOICE,
            PipelineStage.VISUALS,
            PipelineStage.RENDER,
        ]

    def _await_approval(self, stage: PipelineStage) -> str:
        self._emit_log(f"{stage.value} finished and is waiting for approval.")
        self._emit_approval(stage.value)
        return self._wait_for_action()

    def _checkpoint_pause(self) -> None:
        if not self._pause_requested:
            return
        self._emit_log("Pipeline paused. Press Continue to resume.")
        self._emit_approval("Manual Pause")
        action = self._wait_for_action()
        if action == "cancel":
            raise RuntimeError("Pipeline stopped by user.")

    def _wait_for_action(self) -> str:
        with self._condition:
            self._approval_action = ""
            while not self._approval_action:
                self._condition.wait()
            return self._approval_action

    def _release_wait(self, action: str) -> None:
        with self._condition:
            self._approval_action = action
            self._condition.notify_all()

    def _set_stage(self, stage: PipelineStage) -> None:
        state = self._require_state()
        state.stage = stage.value
        self.project_store.save_state(state)
        self._emit_stage(stage.value)

    def _handle_error(self, exc: Exception) -> None:
        if self._current_state is not None:
            self._current_state.stage = PipelineStage.ERROR.value
            self._current_state.last_error = str(exc)
            self.project_store.save_state(self._current_state)
        self._emit_stage(PipelineStage.ERROR.value)
        self._emit_log(f"Error: {exc}")
        self._emit_finished(str(exc))

    def _stage_handles_inline_approvals(self, stage: PipelineStage, settings: AppSettings) -> bool:
        return stage == PipelineStage.VISUALS and settings.video.visual_provider in {"Veo + Stock", "DeeVid Automation"}

    def _format_metadata(self, script_package: ScriptPackage) -> str:
        scene_blocks = []
        for index, scene in enumerate(script_package.scenes, start=1):
            scene_blocks.append(
                "\n".join(
                    [
                        f"{index}. {scene.headline}",
                        f"Action Prompt (Veo 3.1): {scene.action_prompt or scene.supporting_text or scene.visual_query}",
                        f"Audio / Dialogue Cue: {scene.audio_dialogue_cue or scene.narration_text}",
                    ]
                )
            )
        scene_lines = "\n\n".join(scene_blocks)
        return (
            f"Title: {script_package.title}\n\n"
            f"Description:\n{script_package.description}\n\n"
            f"Hashtags:\n{' '.join(script_package.hashtags)}\n\n"
            f"CTA:\n{script_package.call_to_action}\n\n"
            f"Scene Prompt Package:\n{scene_lines}\n\n"
            f"Stock Footage Tags:\n{' | '.join(script_package.stock_footage_tags)}\n\n"
            f"Clip Estimate:\nReuse {script_package.reuse_estimate.reused_clip_count} | New {script_package.reuse_estimate.new_clip_count} | Raw {script_package.reuse_estimate.estimated_raw_seconds}s\n"
        )

    def _build_narration_segments(self, script_package: ScriptPackage) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        for index, scene in enumerate(script_package.scenes, start=1):
            narration = scene.narration_text.strip() or scene.supporting_text.strip() or scene.headline.strip()
            segments.append((f"scene_{index:02d}", narration))
        return segments

    def _generate_edge_segments(
        self,
        *,
        segments: list[tuple[str, str]],
        output_dir: Path,
        voice: str,
        rate: int,
        volume: int,
    ) -> list[str]:
        paths: list[str] = []
        for name, text in segments:
            output_path = output_dir / f"{name}.mp3"
            paths.append(
                self.edge_tts_service.generate_voiceover(
                    text=text,
                    output_path=str(output_path),
                    voice=voice,
                    rate=rate,
                    volume=volume,
                )
            )
        return paths

    def _generate_elevenlabs_segments(
        self,
        *,
        segments: list[tuple[str, str]],
        output_dir: Path,
        api_key: str,
        voice_id: str,
    ) -> list[str]:
        paths: list[str] = []
        for name, text in segments:
            output_path = output_dir / f"{name}.mp3"
            paths.append(
                self.elevenlabs_service.generate_voiceover(
                    api_key=api_key,
                    voice_id=voice_id,
                    text=text,
                    output_path=str(output_path),
                )
            )
        return paths

    def _concat_voice_segments(self, segment_paths: list[str], output_path: str) -> str:
        return self.render_service.concat_audio_segments(segment_paths, output_path)

    def _require_state(self) -> ProjectState:
        if self._current_state is None:
            raise RuntimeError("No active project state.")
        return self._current_state

    def _require_settings(self) -> AppSettings:
        if self._current_settings is None:
            raise RuntimeError("No active settings.")
        return self._current_settings

    def _infer_resume_index(self, state: ProjectState) -> int:
        if state.final_video_path and Path(state.final_video_path).exists():
            return 4
        settings = self._current_settings
        if settings and settings.video.visual_provider == "Veo + Stock":
            if state.converted_scene_audio_paths and all(Path(path).exists() for path in state.converted_scene_audio_paths):
                return 4
            if state.visual_plan:
                return 3
        if state.visual_plan:
            return 4
        if state.voice_path and Path(state.voice_path).exists():
            return 3
        if state.script_package:
            return 2
        if state.source_text:
            return 1
        return 0

    def _build_reuse_estimate(
        self,
        script_package: ScriptPackage,
        settings: AppSettings,
        library: GeneratedVideoLibrary,
    ):
        reused = 0
        new = 0
        raw_seconds = 0
        image_path = settings.veo_reference_image_path.strip()
        image_signature = self.veo_service.reference_image_signature(image_path)
        duration_seconds = 8 if image_path else 6 if settings.video.length_target <= 20 else 8
        for scene in script_package.scenes:
            clip_key = library.build_clip_key(
                model=settings.veo_model,
                prompt=self.veo_service.build_prompt(
                    visual_style="Reference-Driven",
                    scene_prompt=scene.action_prompt or scene.visual_query or scene.supporting_text or scene.headline,
                    dialogue_cue=scene.audio_dialogue_cue or scene.narration_text,
                    negative_prompt=scene.negative_prompt,
                    camera_style=scene.camera_style,
                    style_notes=scene.style_notes,
                    character_mode=settings.veo_character_mode,
                    has_reference_image=bool(image_path),
                ),
                negative_prompt=scene.negative_prompt,
                reference_image_signature=image_signature,
                duration_seconds=duration_seconds,
                aspect_ratio="9:16",
                resolution="720p",
                style_name="Reference-Driven",
                scene_purpose=scene.purpose or scene.headline,
            )
            scene.generated_clip_key = clip_key
            if library.find_existing_clip(clip_key):
                reused += 1
            else:
                new += 1
            raw_seconds += duration_seconds
        script_package.reuse_estimate.total_scene_count = len(script_package.scenes)
        script_package.reuse_estimate.reused_clip_count = reused
        script_package.reuse_estimate.new_clip_count = new
        script_package.reuse_estimate.estimated_raw_seconds = raw_seconds
        return script_package.reuse_estimate

    def _build_veo_visual_assets(self, state: ProjectState, script_package: ScriptPackage, settings: AppSettings) -> list[dict]:
        library = GeneratedVideoLibrary(settings.generated_video_folder)
        asset_dir = Path(state.project_dir) / "assets"
        continuity_dir = Path(state.project_dir) / "continuity_frames"
        asset_dir.mkdir(parents=True, exist_ok=True)
        continuity_dir.mkdir(parents=True, exist_ok=True)
        plan: list[dict] = []
        state.generated_video_library_hits = []
        state.continuity_frame_paths = []
        base_reference_image_path = settings.veo_reference_image_path.strip()
        current_image_path = base_reference_image_path
        use_chain_last_frame = settings.veo_continuity_mode == "Chain Last Frame" and state.source_mode != "facts"
        use_exact_last_frame = settings.veo_character_mode != "Two Character Conversation"
        if settings.veo_character_mode == "Two Character Conversation" and use_chain_last_frame:
            self._emit_log(
                "Two-character mode is using best-end-frame continuity: the app samples several near-end frames and picks the best carry-forward frame instead of the literal last frame."
            )
        elif use_chain_last_frame:
            self._emit_log(
                "Solo continuity is using the literal last frame. Veo prompts are being tightened to keep the character centered and visible through the end of each shot."
            )
        if state.source_mode == "facts":
            self._emit_log("Facts mode uses the same uploaded host image for every Veo clip to keep the presenter consistent.")
        duration_seconds = 8 if current_image_path else 6 if settings.video.length_target <= 20 else 8

        for index, scene in enumerate(script_package.scenes, start=1):
            scene_input_image_path = base_reference_image_path if state.source_mode == "facts" else current_image_path
            current_image_signature = self.veo_service.reference_image_signature(scene_input_image_path)
            prompt = self.veo_service.build_prompt(
                visual_style="Reference-Driven",
                scene_prompt=scene.action_prompt or scene.visual_query or scene.supporting_text or scene.headline,
                dialogue_cue=scene.audio_dialogue_cue or scene.narration_text,
                negative_prompt=scene.negative_prompt,
                camera_style=scene.camera_style,
                style_notes=scene.style_notes,
                character_mode=settings.veo_character_mode,
                has_reference_image=bool(scene_input_image_path),
            )
            clip_key = library.build_clip_key(
                model=settings.veo_model,
                prompt=prompt,
                negative_prompt=scene.negative_prompt,
                reference_image_signature=current_image_signature,
                duration_seconds=duration_seconds,
                aspect_ratio="9:16",
                resolution="720p",
                style_name="Reference-Driven",
                scene_purpose=scene.purpose or scene.headline,
            )
            scene.generated_clip_key = clip_key
            clip_path = asset_dir / f"scene_{index:02d}.mp4"
            poster_path = asset_dir / f"scene_{index:02d}_poster.jpg"
            existing = library.find_existing_clip(clip_key)
            if existing:
                self._emit_log(f"Reusing existing Veo clip for scene {index}: {scene.headline}")
                scene.asset_path = library.copy_clip_into_project(existing, str(clip_path))
                scene.poster_path = library.copy_poster_into_project(existing, str(poster_path))
                scene.asset_type = "video"
                scene.source_name = "Generated Video Library"
                scene.source_credit = "Exact clip reuse"
                scene.generated_prompt = prompt
                state.generated_video_library_hits.append(existing)
                plan_preview = scene.to_dict()
                self._emit_preview("video", scene.asset_path)
                self._emit_log(f"Approve Veo scene {index} before continuing.")
                self._emit_approval(f"Approve Scene {index}")
                action = self._wait_for_action()
                if action == "cancel":
                    raise RuntimeError("Pipeline stopped by user.")
                if action == "regenerate":
                    self._emit_log(f"Regenerating Veo scene {index} instead of reusing cached clip.")
                    existing = None
                else:
                    if use_chain_last_frame:
                        continuity_path = str(continuity_dir / f"scene_{index:02d}_last.jpg")
                        current_image_path = self.render_service.extract_last_frame(
                            scene.asset_path,
                            continuity_path,
                            prefer_two_subjects=settings.veo_character_mode == "Two Character Conversation",
                            exact_last_frame=use_exact_last_frame,
                        )
                        state.continuity_frame_paths.append(current_image_path)
                    plan.append(plan_preview)
                    continue
            if not existing:
                while True:
                    self._emit_log(f"Generating Veo clip {index}/{len(script_package.scenes)} for {scene.headline}.")
                    generated_path = self.veo_service.generate_clip(
                        api_key=settings.gemini_api_key,
                        model=settings.veo_model,
                        prompt=prompt,
                        negative_prompt=scene.negative_prompt,
                        reference_image_path=scene_input_image_path,
                        output_path=str(clip_path),
                        aspect_ratio="9:16",
                        duration_seconds=duration_seconds,
                        resolution="720p",
                    )
                    generated_poster = self.veo_service.generate_poster(generated_path, str(poster_path))
                    clip_name = f"Reference-Driven {scene.purpose or scene.headline}"
                    metadata = library.save_generated_clip(
                        clip_key=clip_key,
                        source_video_path=generated_path,
                        clip_name=clip_name,
                        metadata={
                            "model": settings.veo_model,
                            "prompt": prompt,
                            "negative_prompt": scene.negative_prompt,
                            "reference_image_signature": current_image_signature,
                            "reference_image_path": scene_input_image_path,
                            "character_mode": settings.veo_character_mode,
                            "duration_seconds": duration_seconds,
                            "aspect_ratio": "9:16",
                            "resolution": "720p",
                            "style_name": "Reference-Driven",
                            "scene_purpose": scene.purpose or scene.headline,
                            "poster_path": generated_poster,
                        },
                    )
                    scene.asset_path = library.copy_clip_into_project(metadata, str(clip_path))
                    scene.poster_path = library.copy_poster_into_project(metadata, str(poster_path))
                    scene.asset_type = "video"
                    scene.source_name = "Google Veo"
                    scene.source_credit = settings.veo_model
                    scene.generated_prompt = prompt
                    plan_preview = scene.to_dict()
                    self._emit_preview("video", scene.asset_path)
                    self._emit_log(f"Approve Veo scene {index} before continuing.")
                    self._emit_approval(f"Approve Scene {index}")
                    action = self._wait_for_action()
                    if action == "continue":
                        if use_chain_last_frame:
                            continuity_path = str(continuity_dir / f"scene_{index:02d}_last.jpg")
                            current_image_path = self.render_service.extract_last_frame(
                                scene.asset_path,
                                continuity_path,
                                prefer_two_subjects=settings.veo_character_mode == "Two Character Conversation",
                                exact_last_frame=use_exact_last_frame,
                            )
                            state.continuity_frame_paths.append(current_image_path)
                        plan.append(plan_preview)
                        break
                    if action == "cancel":
                        raise RuntimeError("Pipeline stopped by user.")
                    self._emit_log(f"Regenerating Veo scene {index}.")

        return plan

    def _build_deevid_visual_assets(self, state: ProjectState, script_package: ScriptPackage, settings: AppSettings) -> list[dict]:
        asset_dir = Path(state.project_dir) / "assets"
        input_dir = Path(state.project_dir) / "scene_inputs"
        asset_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        plan: list[dict] = []

        for index, scene in enumerate(script_package.scenes, start=1):
            image_path = self._find_scene_input_image(input_dir, index)
            if not image_path:
                raise FileNotFoundError(
                    f"Missing scene start image for scene {index}. Add a file like scene_{index:02d}.png or .jpg inside {input_dir}."
                )
            output_path = asset_dir / f"scene_{index:02d}.mp4"
            prompt = (
                f"{scene.visual_query or scene.supporting_text or scene.headline}. "
                f"Dialogue / Audio Cue: {scene.narration_text or scene.supporting_text or scene.headline}. "
                f"Character direction: {settings.veo_character_mode}. "
                "Style: Reference-driven animated storytelling. "
                "Vertical short-form video."
            )
            while True:
                self._emit_log(f"Generating DeeVid scene {index}/{len(script_package.scenes)} for {scene.headline}.")
                generated_path = self.deevid_service.generate_scene(
                    profile_dir=settings.deevid_profile_dir,
                    image_path=str(image_path),
                    prompt=prompt,
                    output_path=str(output_path),
                )
                scene.asset_path = generated_path
                scene.asset_type = "video"
                scene.poster_path = self.visual_service.generate_video_poster(
                    generated_path,
                    str(asset_dir / f"scene_{index:02d}_poster.jpg"),
                )
                scene.source_name = "DeeVid"
                scene.source_credit = "Dedicated automation browser"
                scene.generated_prompt = prompt
                plan_preview = scene.to_dict()
                self._emit_preview("video", generated_path)
                self._emit_log(f"Approve DeeVid scene {index} before continuing.")
                self._emit_approval(f"Approve Scene {index}")
                action = self._wait_for_action()
                if action == "continue":
                    plan.append(plan_preview)
                    break
                if action == "cancel":
                    raise RuntimeError("Pipeline stopped by user.")
                self._emit_log(f"Regenerating DeeVid scene {index}.")
        return plan

    def _find_scene_input_image(self, input_dir: Path, index: int) -> Path | None:
        candidates = []
        for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            candidates.append(input_dir / f"scene_{index:02d}{suffix}")
            candidates.append(input_dir / f"scene_{index}{suffix}")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _parse_dialogue_turns(self, dialogue_cue: str, state: ProjectState) -> list[tuple[str, str]]:
        normalized = dialogue_cue.replace("“", '"').replace("”", '"').replace("’", "'").strip()
        if not normalized:
            return [("A", "")]

        name_a = self._character_alias(state.character_a_brief)
        name_b = self._character_alias(state.character_b_brief)
        matches = list(re.finditer(r'"([^"]+)"', normalized))
        turns: list[tuple[str, str]] = []
        next_default = "A"
        for match in matches:
            spoken = match.group(1).strip()
            if not spoken:
                continue
            lead_in = normalized[max(0, match.start() - 90):match.start()].lower()
            if "character a" in lead_in or (name_a and name_a in lead_in):
                speaker = "A"
            elif "character b" in lead_in or (name_b and name_b in lead_in):
                speaker = "B"
            else:
                speaker = next_default
            turns.append((speaker, spoken))
            next_default = "B" if speaker == "A" else "A"

        if turns:
            return turns

        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]
        if not sentences:
            return [("A", normalized)]
        inferred: list[tuple[str, str]] = []
        next_default = "A"
        for sentence in sentences:
            inferred.append((next_default, sentence))
            next_default = "B" if next_default == "A" else "A"
        return inferred

    def _character_alias(self, brief: str) -> str:
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", (brief or "").lower()).strip()
        if not cleaned:
            return ""
        token = cleaned.split()[0]
        return token

    def _convert_two_character_scene_audio(
        self,
        *,
        source_audio_path: str,
        output_path: str,
        scene_prefix: str,
        turns: list[tuple[str, str]],
        voice_id_a: str,
        voice_id_b: str,
        api_key: str,
        source_turn_dir: Path,
        converted_turn_dir: Path,
    ) -> str:
        if not turns:
            raise ValueError("Two-character conversation mode requires dialogue turns.")
        source_segments = self.render_service.split_audio_for_turns(
            source_audio_path,
            expected_count=len(turns),
            output_dir=str(source_turn_dir),
            prefix=scene_prefix,
        )
        converted_segments: list[str] = []
        for turn_index, ((speaker, _spoken), source_segment_path) in enumerate(zip(turns, source_segments, strict=True), start=1):
            voice_id = voice_id_a if speaker == "A" else voice_id_b
            converted_segment = str(converted_turn_dir / f"{scene_prefix}_turn_{turn_index:02d}_{speaker.lower()}.mp3")
            converted_segments.append(
                self.elevenlabs_service.convert_speech(
                    api_key=api_key,
                    voice_id=voice_id,
                    input_audio_path=source_segment_path,
                    output_path=converted_segment,
                )
            )
        return self.render_service.concat_audio_segments(converted_segments, output_path)

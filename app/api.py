from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from app.config import AppSettings
from app.models import PipelineStage
from app.pipeline import PipelineController
from app.services.render_service import RenderService
from app.services.wan2gp_service import Wan2GPService
from app.services.gemini_service import GeminiService
from app.settings_store import SettingsStore


APP_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML_PATH = APP_ROOT / "web" / "index.html"


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


SECRET_SETTING_KEYS = {
    "gemini_api_key",
    "elevenlabs_api_key",
    "elevenlabs_voice_id",
    "elevenlabs_voice_id_a",
    "elevenlabs_voice_id_b",
    "serpapi_api_key",
    "pexels_api_key",
    "pixabay_api_key",
    "gemini_custom_model",
}


def _preserve_existing_secrets(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(updates)
    for key in SECRET_SETTING_KEYS:
        if key in cleaned and isinstance(cleaned[key], str) and not cleaned[key].strip():
            cleaned[key] = base.get(key, "")
    return cleaned



def _pick_path(*, mode: str, initial_path: str = "", title: str = "", filetypes: list[tuple[str, str]] | None = None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Tkinter is required for native file dialogs on the web UI host.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        start = Path(initial_path).expanduser() if initial_path else Path.cwd()
        if mode == "file":
            selected = filedialog.askopenfilename(
                title=title or "Choose File",
                initialdir=str(start.parent if start.is_file() else start),
                filetypes=filetypes or [("All Files", "*.*")],
            )
        else:
            selected = filedialog.askdirectory(
                title=title or "Choose Folder",
                initialdir=str(start if start.exists() else Path.cwd()),
                mustexist=mode == "project",
            )
        return str(selected or "")
    finally:
        root.destroy()


class WebSocketHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for client in list(self._clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    def broadcast(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)


class BackendState:
    def __init__(self) -> None:
        self.settings_store = SettingsStore()
        self.pipeline = PipelineController()
        self.wan2gp_service = Wan2GPService()
        self.render_service = RenderService()
        self.hub = WebSocketHub()
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._wan_process: subprocess.Popen | None = None
        self._wan_cancel_requested = False
        self.settings = self.settings_store.load()
        self.current_project: dict[str, Any] | None = None
        self.current_stage = PipelineStage.IDLE.value
        self.run_state = "idle"
        self.approval_stage = ""
        self.logs: list[str] = []
        self.preview: dict[str, Any] = {"kind": "text", "content": "Preview will appear here."}
        self._local_condition = threading.Condition()
        self._local_approval_action = ""
        self._local_wait_active = False

        self.pipeline.register_callback("log", self._on_log_message)
        self.pipeline.register_callback("stage", self._on_stage_changed)
        self.pipeline.register_callback("preview", self._on_preview_ready)
        self.pipeline.register_callback("approval", self._on_approval_required)
        self.pipeline.register_callback("finished", self._on_pipeline_finished)
        self.pipeline.register_callback("project_loaded", self._on_project_loaded)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.hub.attach_loop(loop)

    def _on_log_message(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-500:]
        self.hub.broadcast({"type": "log", "message": message})

    def _on_stage_changed(self, stage: str) -> None:
        self.current_stage = stage
        if stage == PipelineStage.ERROR.value:
            self.run_state = "error"
        elif stage == PipelineStage.COMPLETE.value:
            self.run_state = "complete"
        elif stage != PipelineStage.IDLE.value:
            self.run_state = "running"
        self.hub.broadcast({"type": "stage_changed", "stage": stage, "runState": self.run_state})

    def _on_preview_ready(self, kind: str, content: str) -> None:
        self.preview = {"kind": kind, "content": content}
        self.hub.broadcast({"type": "preview_ready", "kind": kind, "content": content})

    def _on_approval_required(self, stage: str) -> None:
        self.run_state = "waiting"
        self.approval_stage = stage
        self.hub.broadcast({"type": "approval_required", "stage": stage})

    def _on_pipeline_finished(self, message: str) -> None:
        if self.current_stage == PipelineStage.COMPLETE.value:
            self.run_state = "complete"
        elif self.current_stage == PipelineStage.ERROR.value:
            self.run_state = "error"
        else:
            self.run_state = "idle"
        self.approval_stage = ""
        self.hub.broadcast({"type": "finished", "message": message, "runState": self.run_state})

    def _on_project_loaded(self, payload: dict[str, Any]) -> None:
        self.current_project = payload
        self.hub.broadcast({"type": "project_loaded", "project": payload})

    def _ensure_idle(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            raise HTTPException(status_code=409, detail="A project is already running.")

    def _emit_local_approval(self, stage: str) -> None:
        self.run_state = "waiting"
        self.approval_stage = stage
        self.hub.broadcast({"type": "approval_required", "stage": stage})

    def _wait_for_local_action(self) -> str:
        with self._local_condition:
            self._local_wait_active = True
            self._local_approval_action = ""
            while not self._local_approval_action:
                self._local_condition.wait()
            action = self._local_approval_action
            self._local_wait_active = False
            self._local_approval_action = ""
            return action

    def _release_local_wait(self, action: str) -> bool:
        with self._local_condition:
            if not self._local_wait_active:
                return False
            self._local_approval_action = action
            self._local_condition.notify_all()
            return True

    def _start_worker(self, target, *args) -> None:
        self._ensure_idle()
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._worker_thread = thread
        thread.start()

    def refresh_settings(self) -> AppSettings:
        self.settings = self.settings_store.load()
        return self.settings

    def save_settings(self, updates: dict[str, Any]) -> AppSettings:
        merged = _merge_dict(self.settings.to_dict(), _preserve_existing_secrets(self.settings.to_dict(), updates))
        self.settings = AppSettings.from_dict(merged)
        self.settings_store.save(self.settings)
        return self.settings

    def reload_pipeline_settings(self) -> None:
        self.refresh_settings()
        self.pipeline.reload_settings(self.settings.to_dict())

    def start_project(self, payload: dict[str, Any]) -> None:
        settings = self.save_settings(payload.get("settings", {}))
        urls = [url.strip() for url in payload.get("urls", []) if str(url).strip()]
        export_dir = payload.get("export_folder") or settings.export_folder
        settings_data = settings.to_dict()
        settings_data["source_mode"] = payload.get("source_mode", "scene")
        settings_data["source_prompt"] = payload.get("source_prompt", "")
        settings_data["character_brief"] = payload.get("character_brief", "")
        settings_data["character_a_brief"] = payload.get("character_a_brief", "")
        settings_data["character_b_brief"] = payload.get("character_b_brief", "")
        settings_data["story_type"] = payload.get("story_type", "")
        settings_data["facts_topic"] = payload.get("facts_topic", "")
        settings_data["fact_items"] = payload.get("fact_items", [])
        settings_data["fact_1"] = payload.get("fact_1", "")
        settings_data["fact_2"] = payload.get("fact_2", "")
        settings_data["fact_3"] = payload.get("fact_3", "")
        self.logs = []
        self.preview = {"kind": "text", "content": "Starting project..."}
        self.current_stage = PipelineStage.IDLE.value
        self.run_state = "running"
        self.approval_stage = ""
        self._start_worker(self.pipeline.start_new_project, urls, export_dir, settings_data)

    def start_wan2gp(self, payload: dict[str, Any]) -> None:
        settings = self.save_settings(payload.get("settings", {}))
        image_path = str(payload.get("image_path", "")).strip()
        image_paths = [str(path).strip() for path in payload.get("image_paths", []) if str(path).strip()]
        prompts = [line.strip() for line in payload.get("prompts", []) if str(line).strip() and not str(line).strip().startswith("#")]
        character_count = str(payload.get("character_count", "2") or "2").strip()
        character_mode = "Two Character Conversation" if character_count == "2" else "Solo"
        character_a = str(payload.get("character_a", "")).strip()
        character_b = str(payload.get("character_b", "")).strip()
        export_dir = str(payload.get("export_folder") or settings.export_folder).strip()
        clip_length_seconds = int(payload.get("clip_length_seconds", 6) or 6)
        raw_clip_lengths = payload.get("clip_length_seconds_items", [])
        clip_length_seconds_items = [
            max(1, int(float(value)))
            for value in raw_clip_lengths
            if str(value).strip()
        ]
        if not image_path and not image_paths:
            raise HTTPException(status_code=400, detail="image_path is required")
        if not prompts:
            raise HTTPException(status_code=400, detail="At least one prompt line is required")
        if image_paths and len(prompts) != len(image_paths):
            raise HTTPException(status_code=400, detail="Storyboard image count must exactly match the WAN prompt count.")
        self.logs = []
        self.preview = {"kind": "text", "content": "Starting WAN2GP batch..."}
        self.current_stage = "WAN2GP Generation"
        self.run_state = "running"
        self.approval_stage = ""
        self._wan_cancel_requested = False
        self._start_worker(
            self._run_wan2gp_batch,
            settings,
            image_path,
            image_paths,
            prompts,
            character_mode,
            character_count,
            character_a,
            character_b,
            export_dir,
            clip_length_seconds,
            clip_length_seconds_items,
        )

    def start_flux_storyboard(self, payload: dict[str, Any]) -> None:
        settings = self.save_settings(payload.get("settings", {}))
        reference_image_path = str(payload.get("reference_image_path", "")).strip()
        concept = str(payload.get("concept", "")).strip()
        export_dir = str(payload.get("export_folder") or settings.export_folder).strip()
        if not reference_image_path:
            raise HTTPException(status_code=400, detail="reference_image_path is required")
        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")
        self.logs = []
        self.preview = {"kind": "text", "content": "Starting Flux storyboard generation..."}
        self.current_stage = "Flux Storyboard"
        self.run_state = "running"
        self.approval_stage = ""
        self._wan_cancel_requested = False
        self._start_worker(self._run_flux_storyboard_batch, settings, reference_image_path, concept, export_dir)

    def start_flux_image(self, payload: dict[str, Any]) -> None:
        settings = self.save_settings(payload.get("settings", {}))
        reference_image_path = str(payload.get("reference_image_path", "")).strip()
        prompts = [str(item).strip() for item in payload.get("prompts", []) if str(item).strip()]
        prompt = str(payload.get("prompt", "")).strip()
        if not prompts and prompt:
            prompts = [prompt]
        export_dir = str(payload.get("export_folder") or settings.export_folder).strip()
        if not reference_image_path:
            raise HTTPException(status_code=400, detail="reference_image_path is required")
        if not prompts:
            raise HTTPException(status_code=400, detail="At least one prompt is required")
        self.logs = []
        self.preview = {"kind": "text", "content": "Starting Flux image generation..."}
        self.current_stage = "Flux Image"
        self.run_state = "running"
        self.approval_stage = ""
        self._wan_cancel_requested = False
        self._start_worker(self._run_flux_image_batch, settings, reference_image_path, prompts, export_dir)

    def stop_wan2gp(self) -> None:
        self._wan_cancel_requested = True
        self._release_local_wait("cancel")
        process = self._wan_process
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    def resume_project(self, project_dir: str, settings_updates: dict[str, Any] | None = None) -> None:
        settings = self.save_settings(settings_updates or {})
        self.logs = []
        self.preview = {"kind": "text", "content": "Loading project..."}
        self.current_stage = PipelineStage.IDLE.value
        self.run_state = "running"
        self.approval_stage = ""
        self._start_worker(self.pipeline.resume_project, project_dir, settings.to_dict())

    def load_project(self, project_dir: str, settings_updates: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = self.save_settings(settings_updates or {})
        project = self.pipeline.project_store.load_project(project_dir)
        self.current_project = project.to_dict()
        self.current_stage = project.stage
        self.approval_stage = ""
        self.logs = [f"Loaded project: {project.project_name}"]
        if project.stage == PipelineStage.COMPLETE.value:
            self.run_state = "complete"
        elif project.stage == PipelineStage.ERROR.value:
            self.run_state = "error"
        else:
            self.run_state = "idle"

        preview = {"kind": "text", "content": "Project loaded."}
        if project.final_video_path and Path(project.final_video_path).exists():
            preview = {"kind": "video", "content": project.final_video_path}
        elif project.visual_plan:
            preview = {"kind": "visuals", "content": json.dumps(project.visual_plan, indent=2)}
        elif project.voice_path and Path(project.voice_path).exists():
            preview = {"kind": "audio", "content": project.voice_path}
        elif project.script_package:
            script_package = project.script_package
            preview = {
                "kind": "script",
                "content": f"{script_package.get('title', '')}\n\n{script_package.get('spoken_script', '')}".strip(),
            }
        elif project.source_text:
            preview = {"kind": "text", "content": project.source_text}

        self.preview = preview
        self.settings = settings
        return self.current_state_payload()

    def rollback_project(self, project_dir: str, target_stage: str, settings_updates: dict[str, Any] | None = None) -> None:
        settings = self.save_settings(settings_updates or {})
        self.logs = [f"Rolling project back to {target_stage}..."]
        self.preview = {"kind": "text", "content": f"Rolling back to {target_stage}..."}
        self.current_stage = PipelineStage.IDLE.value
        self.run_state = "running"
        self.approval_stage = ""
        self._start_worker(self.pipeline.rollback_project, project_dir, settings.to_dict(), target_stage)

    def current_state_payload(self) -> dict[str, Any]:
        return {
            "stage": self.current_stage,
            "runState": self.run_state,
            "approvalStage": self.approval_stage,
            "logs": self.logs,
            "preview": self.preview,
            "project": self.current_project,
            "settings": self.settings.to_dict(),
        }

    def handle_local_continue(self) -> bool:
        self.run_state = "running"
        self.approval_stage = ""
        return self._release_local_wait("continue")

    def handle_local_regenerate(self) -> bool:
        self.run_state = "running"
        self.approval_stage = ""
        return self._release_local_wait("regenerate")

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
        return slug[:80] or "clip"

    def _find_latest_mp4(self, output_dir: Path, started_at: float) -> str:
        candidates = [path for path in output_dir.rglob("*.mp4") if path.is_file() and path.stat().st_mtime >= started_at - 1]
        if not candidates:
            return ""
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

    def _find_latest_image(self, output_dir: Path, started_at: float) -> str:
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(
                path for path in output_dir.rglob(pattern) if path.is_file() and path.stat().st_mtime >= started_at - 1
            )
        if not candidates:
            return ""
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

    def _extract_wan_progress(self, line: str) -> str:
        match = re.search(r"(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[", line)
        if match:
            percent, current_step, total_steps = match.groups()
            return f"{percent}% • {current_step}/{total_steps} steps"
        percent_only = re.search(r"\b(\d+)%\b", line)
        if percent_only:
            return f"{percent_only.group(1)}%"
        return ""

    def _sanitize_wan_dialogue(self, prompt: str) -> str:
        sanitized = re.sub(
            r"\[\s*TECHNICAL CONSTRAINTS\s*:[^\]]*\]",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        return sanitized.strip()

    def _build_wan_prompt(self, prompt: str, continuity_mode: str, character_mode: str = "Solo") -> str:
        cleaned = self._sanitize_wan_dialogue(prompt.strip())
        multi_character_suffix = (
            " Keep both reference characters visually consistent, clearly readable, and distinct from each other. Do not introduce any new characters. Preserve the same two-character relationship from the reference image."
            if character_mode == "Two Character Conversation"
            else ""
        )
        solo_same_frame_suffix = (
            " Keep the main character clearly visible for the full shot. Maintain the same overall composition by the final frame. "
            "Do not let the character leave the frame. Do not introduce any new characters."
        )
        if continuity_mode != "Chain Last Frame":
            if continuity_mode == "Same Start + End Image":
                if character_mode == "Two Character Conversation":
                    return (
                        f"{cleaned} Keep both visible characters on screen and readable for the full shot. "
                        "Maintain the same overall composition by the final frame. End on a stable composition that closely matches the original image. "
                        f"Do not let the characters leave the frame. Do not introduce any new characters.{multi_character_suffix}"
                    )
                return (
                    f"{cleaned} {solo_same_frame_suffix} End on a stable composition that closely matches the original image."
                )
            return f"{cleaned}{multi_character_suffix}"
        continuity_suffix = (
            " Keep the main speaking character centered and clearly visible for the full shot. "
            "Do not let the character leave the frame. End the shot with the character still on screen, "
            "fully readable, in a stable medium shot or medium close-up. Do not end on empty background."
        )
        return f"{cleaned}{continuity_suffix}{multi_character_suffix}"

    def _run_flux_storyboard_batch(self, settings: AppSettings, reference_image_path: str, concept: str, export_dir: str) -> None:
        try:
            self.wan2gp_service.validate_paths(
                root_dir=settings.wan2gp_root_dir,
                template_path=settings.flux_template_path,
                image_path=reference_image_path,
            )
            template = self.wan2gp_service.load_template(settings.flux_template_path)
            self._on_log_message(f"Flux root verified: {settings.wan2gp_root_dir}")
            self._on_log_message(f"Flux template verified: {settings.flux_template_path}")
            self._on_log_message(f"Flux reference image verified: {reference_image_path}")
            while True:
                self._on_log_message("Generating 4 storyboard prompts with Gemini...")
                prompts = self.pipeline.gemini_service.generate_flux_storyboard_prompts(
                    settings.gemini_api_key,
                    settings.gemini_model,
                    concept,
                    prompt_count=4,
                    character_mode=settings.veo_character_mode,
                )
                self.preview = {"kind": "text", "content": "\n\n".join(prompts)}
                self.hub.broadcast({"type": "preview_ready", "kind": "text", "content": "\n\n".join(prompts)})
                self._on_log_message("Flux storyboard prompts are ready and waiting for approval.")
                self._emit_local_approval("Approve Flux Storyboard Prompts")
                action = self._wait_for_local_action()
                if action == "continue":
                    break
                if action == "cancel":
                    raise RuntimeError("Flux storyboard generation stopped by user.")
                self._on_log_message("Regenerating Flux storyboard prompts.")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            batch_started_at = datetime.now()
            project_name = f"flux-storyboard-{timestamp}"
            project_dir = Path(export_dir).expanduser().resolve() / project_name
            settings_dir = project_dir / "settings"
            output_dir = project_dir / "storyboard_images"
            settings_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.current_project = {
                "project_name": project_name,
                "project_dir": str(project_dir),
                "export_dir": str(Path(export_dir).expanduser().resolve()),
                "source_mode": "flux_storyboard",
                "image_path": str(Path(reference_image_path).expanduser().resolve()),
                "image_paths": [],
                "flux_concept": concept,
                "flux_prompts": prompts,
                "stage": "Flux Storyboard",
                "wan_started_at": batch_started_at.isoformat(),
                "wan_completed_at": "",
                "wan_elapsed_seconds": 0,
            }
            self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
            generated_images: list[str] = []

            for index, prompt in enumerate(prompts, start=1):
                while True:
                    if self._wan_cancel_requested:
                        raise RuntimeError("Flux storyboard generation stopped by user.")
                    self.current_stage = f"Flux Storyboard Image {index}/4"
                    self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                    self._on_log_message(f"Generating Flux storyboard image {index}/4.")
                    self._on_log_message(f"Flux prompt {index}: {prompt}")
                    output_filename = f"{index:02d}_{self._slugify(prompt)}"
                    settings_payload = self.wan2gp_service.build_flux_image_payload(
                        template=template,
                        prompt=prompt,
                        reference_image_path=reference_image_path,
                        output_filename=output_filename,
                    )
                    settings_path = self.wan2gp_service.write_settings_file(
                        settings_payload,
                        str(settings_dir / f"storyboard_{index:02d}.json"),
                    )
                    command = self.wan2gp_service.build_command(
                        env_name=settings.wan2gp_env_name,
                        settings_path=settings_path,
                        output_dir=str(output_dir),
                    )
                    self._on_log_message(f"Flux settings file: {settings_path}")
                    self._on_log_message(f"Flux output filename base: {output_filename}")
                    self._on_log_message("Launching Flux image generation...")
                    started_at = datetime.now().timestamp()
                    self._wan_process = subprocess.Popen(
                        command,
                        cwd=settings.wan2gp_root_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    self._on_log_message(f"Flux process started (pid {self._wan_process.pid}).")
                    detected_output = ""
                    line_queue: queue.Queue[str | None] = queue.Queue()
                    assert self._wan_process.stdout is not None

                    def _read_stdout(stream, sink: queue.Queue[str | None]) -> None:
                        try:
                            for raw_line in stream:
                                sink.put(raw_line)
                        finally:
                            sink.put(None)

                    reader = threading.Thread(target=_read_stdout, args=(self._wan_process.stdout, line_queue), daemon=True)
                    reader.start()
                    last_heartbeat = time.monotonic()
                    stdout_closed = False
                    while True:
                        try:
                            item = line_queue.get(timeout=1.0)
                        except queue.Empty:
                            item = None
                            if time.monotonic() - last_heartbeat >= 10:
                                self._on_log_message("Flux still running... waiting for generator output.")
                                self.hub.broadcast({"type": "stage_changed", "stage": f"Flux Storyboard Image {index}/4 • Running", "runState": self.run_state})
                                last_heartbeat = time.monotonic()
                        else:
                            if item is None:
                                stdout_closed = True
                            else:
                                cleaned = item.rstrip()
                                if cleaned:
                                    self._on_log_message(cleaned)
                                    progress_suffix = self._extract_wan_progress(cleaned)
                                    if progress_suffix:
                                        self.current_stage = f"Flux Storyboard Image {index}/4 • {progress_suffix}"
                                        self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                                    parsed_path = self.wan2gp_service.parse_output_path_from_log(
                                        cleaned,
                                        root_dir=settings.wan2gp_root_dir,
                                        output_dir=str(output_dir),
                                    )
                                    if parsed_path:
                                        detected_output = parsed_path
                                        self.current_stage = f"Flux Storyboard Image {index}/4 • Saved"
                                        self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                                    last_heartbeat = time.monotonic()
                        if self._wan_cancel_requested and self._wan_process.poll() is None:
                            self._wan_process.terminate()
                        if stdout_closed and self._wan_process.poll() is not None:
                            break
                    return_code = self._wan_process.wait()
                    self._wan_process = None
                    if self._wan_cancel_requested:
                        raise RuntimeError("Flux storyboard generation stopped by user.")
                    if return_code != 0:
                        raise RuntimeError(f"Flux storyboard generation failed for image {index} (exit code {return_code}).")
                    final_output = detected_output or self._find_latest_image(output_dir, started_at)
                    if not final_output:
                        raise RuntimeError(f"Flux finished image {index} but no output image was detected.")
                    self._on_log_message(f"Flux storyboard image {index}/4 ready: {final_output}")
                    self.preview = {"kind": "image", "content": final_output}
                    self.hub.broadcast({"type": "preview_ready", "kind": "image", "content": final_output})
                    self._on_log_message(f"Flux storyboard image {index} is waiting for approval.")
                    self._emit_local_approval(f"Approve Flux Storyboard Image {index}")
                    action = self._wait_for_local_action()
                    if action == "continue":
                        generated_images.append(final_output)
                        self.current_project["image_paths"] = generated_images
                        self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
                        break
                    if action == "cancel":
                        raise RuntimeError("Flux storyboard generation stopped by user.")
                    self._on_log_message(f"Regenerating Flux storyboard image {index}.")

            self.current_project["stage"] = "Complete"
            completed_at = datetime.now()
            self.current_project["wan_completed_at"] = completed_at.isoformat()
            self.current_project["wan_elapsed_seconds"] = max(0, int((completed_at - batch_started_at).total_seconds()))
            self.current_stage = "Complete"
            self.run_state = "complete"
            self.preview = {"kind": "image", "content": generated_images[-1] if generated_images else ""}
            self._on_log_message("Flux storyboard generation completed successfully with 4 images.")
            self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
            self.hub.broadcast({"type": "finished", "message": "Flux storyboard generation completed successfully.", "runState": self.run_state})
        except Exception as exc:  # noqa: BLE001
            self.run_state = "error"
            self.current_stage = "Error"
            if isinstance(self.current_project, dict):
                completed_at = datetime.now()
                started_raw = self.current_project.get("wan_started_at", "")
                elapsed_seconds = 0
                try:
                    if started_raw:
                        elapsed_seconds = max(0, int((completed_at - datetime.fromisoformat(started_raw)).total_seconds()))
                except Exception:
                    elapsed_seconds = 0
                self.current_project["wan_completed_at"] = completed_at.isoformat()
                self.current_project["wan_elapsed_seconds"] = elapsed_seconds
            self._on_log_message(f"Error: {exc}")
            self.hub.broadcast({"type": "finished", "message": str(exc), "runState": self.run_state})

    def _run_flux_image_batch(self, settings: AppSettings, reference_image_path: str, prompts: list[str], export_dir: str) -> None:
        try:
            self.wan2gp_service.validate_paths(
                root_dir=settings.wan2gp_root_dir,
                template_path=settings.flux_template_path,
                image_path=reference_image_path,
            )
            template = self.wan2gp_service.load_template(settings.flux_template_path)
            self._on_log_message(f"Flux root verified: {settings.wan2gp_root_dir}")
            self._on_log_message(f"Flux template verified: {settings.flux_template_path}")
            self._on_log_message(f"Flux reference image verified: {reference_image_path}")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            batch_started_at = datetime.now()
            project_name = f"flux-image-{timestamp}"
            project_dir = Path(export_dir).expanduser().resolve() / project_name
            settings_dir = project_dir / "settings"
            output_dir = project_dir / "images"
            settings_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.current_project = {
                "project_name": project_name,
                "project_dir": str(project_dir),
                "export_dir": str(Path(export_dir).expanduser().resolve()),
                "source_mode": "flux_image",
                "image_path": str(Path(reference_image_path).expanduser().resolve()),
                "prompt": prompts[0] if prompts else "",
                "prompts": prompts,
                "prompt_count": len(prompts),
                "stage": "Flux Image",
                "wan_started_at": batch_started_at.isoformat(),
                "wan_completed_at": "",
                "wan_elapsed_seconds": 0,
            }
            self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
            generated_images: list[str] = []
            total_images = len(prompts)
            for index, prompt in enumerate(prompts, start=1):
                if self._wan_cancel_requested:
                    raise RuntimeError("Flux image generation stopped by user.")
                self.current_stage = f"Flux Image {index}/{total_images}"
                self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                self._on_log_message(f"Generating Flux image {index}/{total_images}.")
                self._on_log_message(f"Flux prompt {index}: {prompt}")
                output_filename = f"{index:02d}_{self._slugify(prompt)}"
                settings_payload = self.wan2gp_service.build_flux_image_payload(
                    template=template,
                    prompt=prompt,
                    reference_image_path=reference_image_path,
                    output_filename=output_filename,
                )
                settings_path = self.wan2gp_service.write_settings_file(
                    settings_payload,
                    str(settings_dir / f"flux_image_{index:02d}.json"),
                )
                command = self.wan2gp_service.build_command(
                    env_name=settings.wan2gp_env_name,
                    settings_path=settings_path,
                    output_dir=str(output_dir),
                )
                self._on_log_message(f"Flux settings file: {settings_path}")
                self._on_log_message(f"Flux output filename base: {output_filename}")
                self._on_log_message("Launching Flux image generation...")
                started_at = datetime.now().timestamp()
                self._wan_process = subprocess.Popen(
                    command,
                    cwd=settings.wan2gp_root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self._on_log_message(f"Flux process started (pid {self._wan_process.pid}).")
                detected_output = ""
                line_queue: queue.Queue[str | None] = queue.Queue()
                assert self._wan_process.stdout is not None

                def _read_stdout(stream, sink: queue.Queue[str | None]) -> None:
                    try:
                        for raw_line in stream:
                            sink.put(raw_line)
                    finally:
                        sink.put(None)

                reader = threading.Thread(target=_read_stdout, args=(self._wan_process.stdout, line_queue), daemon=True)
                reader.start()
                last_heartbeat = time.monotonic()
                stdout_closed = False
                while True:
                    try:
                        item = line_queue.get(timeout=1.0)
                    except queue.Empty:
                        item = None
                        if time.monotonic() - last_heartbeat >= 10:
                            self._on_log_message("Flux still running... waiting for generator output.")
                            self.hub.broadcast({"type": "stage_changed", "stage": f"Flux Image {index}/{total_images} ? Running", "runState": self.run_state})
                            last_heartbeat = time.monotonic()
                    else:
                        if item is None:
                            stdout_closed = True
                        else:
                            cleaned = item.rstrip()
                            if cleaned:
                                self._on_log_message(cleaned)
                                progress_suffix = self._extract_wan_progress(cleaned)
                                if progress_suffix:
                                    self.current_stage = f"Flux Image {index}/{total_images} ? {progress_suffix}"
                                    self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                                parsed_path = self.wan2gp_service.parse_output_path_from_log(
                                    cleaned,
                                    root_dir=settings.wan2gp_root_dir,
                                    output_dir=str(output_dir),
                                )
                                if parsed_path:
                                    detected_output = parsed_path
                                    self.current_stage = f"Flux Image {index}/{total_images} ? Saved"
                                    self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                                last_heartbeat = time.monotonic()
                    if self._wan_cancel_requested and self._wan_process.poll() is None:
                        self._wan_process.terminate()
                    if stdout_closed and self._wan_process.poll() is not None:
                        break
                return_code = self._wan_process.wait()
                self._wan_process = None
                if self._wan_cancel_requested:
                    raise RuntimeError("Flux image generation stopped by user.")
                if return_code != 0:
                    raise RuntimeError(f"Flux image generation failed (exit code {return_code}).")
                final_output = detected_output or self._find_latest_image(output_dir, started_at)
                if not final_output:
                    raise RuntimeError("Flux finished but no output image was detected.")
                generated_images.append(final_output)
                self.current_project["generated_images"] = generated_images
                self.preview = {"kind": "image", "content": final_output}
                self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
                self.hub.broadcast({"type": "preview_ready", "kind": "image", "content": final_output})

            self.current_project["stage"] = "Complete"
            completed_at = datetime.now()
            self.current_project["wan_completed_at"] = completed_at.isoformat()
            self.current_project["wan_elapsed_seconds"] = max(0, int((completed_at - batch_started_at).total_seconds()))
            self.current_stage = "Complete"
            self.run_state = "complete"
            self.preview = {"kind": "image", "content": generated_images[-1] if generated_images else ""}
            self._on_log_message(f"Flux batch ready: {len(generated_images)} image(s)")
            self._on_log_message("Flux image generation completed successfully.")
            self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
            if generated_images:
                self.hub.broadcast({"type": "preview_ready", "kind": "image", "content": generated_images[-1]})
            self.hub.broadcast({"type": "finished", "message": "Flux image generation completed successfully.", "runState": self.run_state})
        except Exception as exc:  # noqa: BLE001
            self.run_state = "error"
            self.current_stage = "Error"
            if isinstance(self.current_project, dict):
                completed_at = datetime.now()
                started_raw = self.current_project.get("wan_started_at", "")
                elapsed_seconds = 0
                try:
                    if started_raw:
                        elapsed_seconds = max(0, int((completed_at - datetime.fromisoformat(started_raw)).total_seconds()))
                except Exception:
                    elapsed_seconds = 0
                self.current_project["wan_completed_at"] = completed_at.isoformat()
                self.current_project["wan_elapsed_seconds"] = elapsed_seconds
            self._on_log_message(f"Error: {exc}")
            self.hub.broadcast({"type": "finished", "message": str(exc), "runState": self.run_state})

    def _run_wan2gp_batch(
        self,
        settings: AppSettings,
        image_path: str,
        image_paths: list[str],
        prompts: list[str],
        character_mode: str,
        character_count: str,
        character_a: str,
        character_b: str,
        export_dir: str,
        clip_length_seconds: int,
        clip_length_seconds_items: list[int] | None = None,
    ) -> None:
        try:
            storyboard_paths = self.wan2gp_service.validate_image_paths(image_paths) if image_paths else []
            primary_image_path = storyboard_paths[0] if storyboard_paths else image_path
            self.wan2gp_service.validate_paths(
                root_dir=settings.wan2gp_root_dir,
                template_path=settings.wan2gp_template_path,
                image_path=primary_image_path,
            )
            template = self.wan2gp_service.load_template(settings.wan2gp_template_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            batch_started_at = datetime.now()
            project_name = f"wan2gp-{timestamp}"
            project_dir = Path(export_dir).expanduser().resolve() / project_name
            settings_dir = project_dir / "settings"
            output_dir = project_dir / "outputs"
            settings_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            normalized_clip_lengths = [
                max(1, int(value))
                for value in (clip_length_seconds_items or [])
            ]
            if len(normalized_clip_lengths) < len(prompts):
                normalized_clip_lengths.extend([clip_length_seconds] * (len(prompts) - len(normalized_clip_lengths)))
            normalized_clip_lengths = normalized_clip_lengths[: len(prompts)]
            self.current_project = {
                "project_name": project_name,
                "project_dir": str(project_dir),
                "export_dir": str(Path(export_dir).expanduser().resolve()),
                "source_mode": "wan2gp",
                "image_path": primary_image_path,
                "image_paths": storyboard_paths,
                "prompt_count": len(prompts),
                "prompts": prompts,
                "character_mode": character_mode,
                "character_count": character_count,
                "character_a": character_a,
                "character_b": character_b,
                "clip_length_seconds": clip_length_seconds,
                "clip_length_seconds_items": normalized_clip_lengths,
                "stage": "WAN2GP Generation",
                "wan2gp_continuity_mode": settings.wan2gp_continuity_mode,
                "wan_started_at": batch_started_at.isoformat(),
                "wan_completed_at": "",
                "wan_elapsed_seconds": 0,
            }
            self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
            self._on_log_message(f"WAN2GP root verified: {settings.wan2gp_root_dir}")
            self._on_log_message(f"WAN2GP template verified: {settings.wan2gp_template_path}")
            self._on_log_message(f"WAN2GP start image verified: {primary_image_path}")
            if storyboard_paths:
                self._on_log_message(f"WAN2GP storyboard images verified: {len(storyboard_paths)}")
            self._on_log_message(f"WAN2GP export folder: {project_dir}")

            generated_paths: list[str] = []
            continuity_dir = project_dir / "continuity_frames"
            continuity_dir.mkdir(parents=True, exist_ok=True)
            current_image_path = primary_image_path
            for index, prompt in enumerate(prompts, start=1):
                if self._wan_cancel_requested:
                    raise RuntimeError("WAN2GP batch stopped by user.")
                self.current_stage = f"WAN2GP Clip {index}/{len(prompts)}"
                self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                self._on_log_message(f"Generating WAN2GP clip {index}/{len(prompts)}.")
                clip_start_image = current_image_path
                clip_end_image = image_path if settings.wan2gp_continuity_mode == "Same Start + End Image" else None
                if storyboard_paths and index <= len(storyboard_paths):
                    clip_start_image = storyboard_paths[index - 1]
                    clip_end_image = None
                    self._on_log_message(f"WAN2GP storyboard scene for clip {index}: start={clip_start_image}")
                if settings.wan2gp_continuity_mode == "Chain Last Frame":
                    self._on_log_message(f"WAN2GP continuity image for clip {index}: {current_image_path}")
                output_filename = f"{index:02d}_{self._slugify(prompt)}"
                current_clip_length = normalized_clip_lengths[index - 1] if index - 1 < len(normalized_clip_lengths) else clip_length_seconds
                settings_payload = self.wan2gp_service.build_settings_payload(
                    template=template,
                    prompt=self._build_wan_prompt(prompt, settings.wan2gp_continuity_mode, character_mode),
                    image_path=clip_start_image,
                    image_end_path=clip_end_image,
                    output_filename=output_filename,
                    clip_length_seconds=current_clip_length,
                )
                settings_path = self.wan2gp_service.write_settings_file(
                    settings_payload,
                    str(settings_dir / f"clip_{index:02d}.json"),
                )
                command = self.wan2gp_service.build_command(
                    env_name=settings.wan2gp_env_name,
                    settings_path=settings_path,
                    output_dir=str(output_dir),
                )
                self._on_log_message(f"WAN2GP settings file: {settings_path}")
                self._on_log_message(f"WAN2GP output filename base: {output_filename}")
                self._on_log_message(f"WAN2GP clip length for clip {index}: {current_clip_length}s")
                self._on_log_message("Launching WAN2GP process...")
                started_at = datetime.now().timestamp()
                self._wan_process = subprocess.Popen(
                    command,
                    cwd=settings.wan2gp_root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self._on_log_message(f"WAN2GP process started (pid {self._wan_process.pid}).")
                detected_output = ""
                line_queue: queue.Queue[str | None] = queue.Queue()
                assert self._wan_process.stdout is not None

                def _read_stdout(stream, sink: queue.Queue[str | None]) -> None:
                    try:
                        for raw_line in stream:
                            sink.put(raw_line)
                    finally:
                        sink.put(None)

                reader = threading.Thread(target=_read_stdout, args=(self._wan_process.stdout, line_queue), daemon=True)
                reader.start()
                last_heartbeat = time.monotonic()
                stdout_closed = False
                while True:
                    try:
                        item = line_queue.get(timeout=1.0)
                    except queue.Empty:
                        item = None
                        if time.monotonic() - last_heartbeat >= 10:
                            elapsed = int(time.monotonic() - last_heartbeat)
                            self._on_log_message("WAN2GP still running... waiting for generator output.")
                            self.hub.broadcast({"type": "stage_changed", "stage": f"WAN2GP Clip {index}/{len(prompts)} • Running", "runState": self.run_state})
                            last_heartbeat = time.monotonic()
                    else:
                        if item is None:
                            stdout_closed = True
                        else:
                            cleaned = item.rstrip()
                            if cleaned:
                                self._on_log_message(cleaned)
                                progress_suffix = self._extract_wan_progress(cleaned)
                                if progress_suffix:
                                    self.current_stage = f"WAN2GP Clip {index}/{len(prompts)} • {progress_suffix}"
                                    self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                                parsed_path = self.wan2gp_service.parse_output_path_from_log(
                                    cleaned,
                                    root_dir=settings.wan2gp_root_dir,
                                    output_dir=str(output_dir),
                                )
                                if parsed_path:
                                    detected_output = parsed_path
                                    self.current_stage = f"WAN2GP Clip {index}/{len(prompts)} • Saved"
                                    self.hub.broadcast({"type": "stage_changed", "stage": self.current_stage, "runState": self.run_state})
                                last_heartbeat = time.monotonic()
                    if self._wan_cancel_requested and self._wan_process.poll() is None:
                        self._wan_process.terminate()
                    if stdout_closed and self._wan_process.poll() is not None:
                        break
                return_code = self._wan_process.wait()
                self._wan_process = None
                if self._wan_cancel_requested:
                    raise RuntimeError("WAN2GP batch stopped by user.")
                if return_code != 0:
                    raise RuntimeError(f"WAN2GP generation failed for clip {index} (exit code {return_code}).")
                final_output = detected_output or self._find_latest_mp4(output_dir, started_at)
                if not final_output:
                    raise RuntimeError(f"WAN2GP finished clip {index} but no output video was detected.")
                self._on_log_message(f"WAN2GP clip {index}/{len(prompts)} ready: {final_output}")
                self.preview = {"kind": "video", "content": final_output}
                self.hub.broadcast({"type": "preview_ready", "kind": "video", "content": final_output})
                self._on_log_message(f"WAN2GP clip {index} is waiting for approval.")
                self._emit_local_approval(f"Approve WAN2GP Clip {index}")
                action = self._wait_for_local_action()
                if action == "cancel":
                    raise RuntimeError("WAN2GP batch stopped by user.")
                if action == "regenerate":
                    self._on_log_message(f"Regenerating WAN2GP clip {index}.")
                    continue
                generated_paths.append(final_output)
                if settings.wan2gp_continuity_mode == "Chain Last Frame" and not storyboard_paths and index < len(prompts):
                    next_frame_path = continuity_dir / f"scene_{index:02d}_last.jpg"
                    current_image_path = self.render_service.extract_last_frame(
                        final_output,
                        str(next_frame_path),
                        exact_last_frame=True,
                    )
                    self._on_log_message(f"WAN2GP carry-forward frame saved: {current_image_path}")

            self.current_project["generated_videos"] = generated_paths
            self.current_project["stage"] = "Complete"
            completed_at = datetime.now()
            self.current_project["wan_completed_at"] = completed_at.isoformat()
            self.current_project["wan_elapsed_seconds"] = max(0, int((completed_at - batch_started_at).total_seconds()))
            self.current_stage = "Complete"
            self.run_state = "complete"
            self.preview = {"kind": "video", "content": generated_paths[-1] if generated_paths else ""}
            self._on_log_message(f"WAN2GP batch completed successfully with {len(generated_paths)} clips.")
            self.hub.broadcast({"type": "project_loaded", "project": self.current_project})
            self.hub.broadcast({"type": "finished", "message": "WAN2GP batch completed successfully.", "runState": self.run_state})
        except Exception as exc:  # noqa: BLE001
            self.run_state = "error"
            self.current_stage = "Error"
            if isinstance(self.current_project, dict) and self.current_project.get("source_mode") == "wan2gp":
                completed_at = datetime.now()
                started_raw = self.current_project.get("wan_started_at", "")
                elapsed_seconds = 0
                try:
                    if started_raw:
                        elapsed_seconds = max(0, int((completed_at - datetime.fromisoformat(started_raw)).total_seconds()))
                except Exception:
                    elapsed_seconds = 0
                self.current_project["wan_completed_at"] = completed_at.isoformat()
                self.current_project["wan_elapsed_seconds"] = elapsed_seconds
            self._on_log_message(f"Error: {exc}")
            self.hub.broadcast({"type": "finished", "message": str(exc), "runState": self.run_state})


state = BackendState()
app = FastAPI(title="Automated Video Creator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    state.attach_loop(asyncio.get_running_loop())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(INDEX_HTML_PATH), media_type="text/html; charset=utf-8")


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = state.refresh_settings()
    edge_ok, edge_message = state.pipeline.edge_tts_service.check_available()
    deevid_ok, deevid_message = state.pipeline.deevid_service.check_available()
    ffmpeg_ok = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    visual_ok = bool(settings.serpapi_api_key or settings.pexels_api_key or settings.pixabay_api_key or settings.video.visual_provider in {"Veo + Stock", "DeeVid Automation"})
    return {
        "ok": True,
        "backend": "online",
        "edge_tts": {"ok": edge_ok, "message": edge_message},
        "visual_apis": {"ok": visual_ok, "message": "Connected" if visual_ok else "Optional"},
        "ffmpeg": {"ok": ffmpeg_ok, "message": "Detected" if ffmpeg_ok else "Missing from PATH"},
        "deevid": {"ok": deevid_ok, "message": deevid_message},
        "stage": state.current_stage,
        "runState": state.run_state,
    }


@app.get("/settings")
async def get_settings() -> dict[str, Any]:
    return state.refresh_settings().to_dict()


@app.post("/settings")
async def post_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = state.save_settings(payload)
    return {"success": True, "settings": settings.to_dict()}


@app.get("/project/state")
async def get_project_state() -> dict[str, Any]:
    state.refresh_settings()
    return state.current_state_payload()


@app.post("/project/start")
async def start_project(payload: dict[str, Any]) -> dict[str, Any]:
    state.start_project(payload)
    return {"success": True, "message": "Project started."}


@app.post("/project/resume")
async def resume_project(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir", "") or (state.current_project or {}).get("project_dir", "")).strip()
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required")
    state.resume_project(project_dir, payload.get("settings"))
    return {"success": True, "message": "Project resumed."}


@app.post("/project/load")
async def load_project(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir", "")).strip()
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required")
    snapshot = state.load_project(project_dir, payload.get("settings"))
    return {"success": True, "message": "Project loaded.", "state": snapshot}


@app.post("/project/rollback")
async def rollback_project(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir", "") or (state.current_project or {}).get("project_dir", "")).strip()
    target_stage = str(payload.get("target_stage", "")).strip()
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required")
    if not target_stage:
        raise HTTPException(status_code=400, detail="target_stage is required")
    state.rollback_project(project_dir, target_stage, payload.get("settings"))
    return {"success": True, "message": f"Project rolled back to {target_stage}."}


@app.post("/project/stop")
async def stop_project() -> dict[str, Any]:
    state.pipeline.request_stop()
    state.stop_wan2gp()
    return {"success": True}


@app.post("/project/pause")
async def pause_project() -> dict[str, Any]:
    state.pipeline.request_pause()
    return {"success": True}


@app.post("/project/continue")
async def continue_project() -> dict[str, Any]:
    state.reload_pipeline_settings()
    if not state.handle_local_continue():
        state.run_state = "running"
        state.approval_stage = ""
        state.pipeline.continue_after_approval()
    return {"success": True}


@app.post("/project/regenerate")
async def regenerate_project() -> dict[str, Any]:
    state.reload_pipeline_settings()
    if not state.handle_local_regenerate():
        state.run_state = "running"
        state.approval_stage = ""
        state.pipeline.regenerate_current_step()
    return {"success": True}


@app.post("/project/open-folder")
async def open_project_folder(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ""
    if payload and payload.get("path"):
        path = str(payload["path"])
    elif state.current_project and state.current_project.get("project_dir"):
        path = str(state.current_project["project_dir"])
    elif state.settings.export_folder:
        path = state.settings.export_folder
    if not path:
        raise HTTPException(status_code=404, detail="No project folder available.")
    os.startfile(path)  # type: ignore[attr-defined]
    return {"success": True}


@app.post("/wan2gp/start")
async def start_wan2gp(payload: dict[str, Any]) -> dict[str, Any]:
    state.start_wan2gp(payload)
    return {"success": True, "message": "WAN2GP batch started."}


@app.post("/flux/storyboard/start")
async def start_flux_storyboard(payload: dict[str, Any]) -> dict[str, Any]:
    state.start_flux_storyboard(payload)
    return {"success": True, "message": "Flux storyboard generation started."}

@app.post("/flux/image/start")
async def start_flux_image(payload: dict[str, Any]) -> dict[str, Any]:
    state.start_flux_image(payload)
    return {"success": True, "message": "Flux image generation started."}


@app.post("/prompting/generate")
async def generate_prompting_pack(payload: dict[str, Any]) -> JSONResponse:
    settings = payload.get("settings", {}) or {}
    selected_model = (
        settings.get("gemini_custom_model")
        or settings.get("gemini_model")
        or GeminiService.DEFAULT_MODEL
    )
    api_key = settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return JSONResponse({"success": False, "error": "Gemini API key not configured."}, status_code=400)
    try:
        state._on_log_message(
            f"Prompting generate requested: videos={int(payload.get('num_videos', 1))}, length={payload.get('video_length', '15')}, model={selected_model}"
        )
        result = state.pipeline.gemini_service.generate_wan_production_pack_structured(
            api_key=api_key,
            model=str(selected_model),
            num_videos=int(payload.get("num_videos", 1)),
            video_length=str(payload.get("video_length", "15")),
            character_count=str(payload.get("character_count", "2") or "2"),
            context_vibe=str(payload.get("context_vibe", "")),
            character_a=str(payload.get("character_a", "")),
            character_b=str(payload.get("character_b", "")),
            reference_image_path=str(payload.get("reference_image_path", "")),
        )
        return JSONResponse({"success": True, **result})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"Prompting generate failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/prompting/generate-context-vibe")
async def generate_prompting_context_vibe(payload: dict[str, Any]) -> JSONResponse:
    settings = payload.get("settings", {}) or {}
    selected_model = (
        settings.get("gemini_custom_model")
        or settings.get("gemini_model")
        or GeminiService.DEFAULT_MODEL
    )
    api_key = settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    reference_image_path = str(payload.get("reference_image_path", "")).strip()
    if not api_key:
        return JSONResponse({"success": False, "error": "Gemini API key not configured."}, status_code=400)
    if not reference_image_path:
        return JSONResponse({"success": False, "error": "Reference image is required."}, status_code=400)
    try:
        context_vibe = state.pipeline.gemini_service.generate_context_vibe(
            api_key=api_key,
            model=str(selected_model),
            character_count=str(payload.get("character_count", "2") or "2"),
            character_a=str(payload.get("character_a", "")),
            character_b=str(payload.get("character_b", "")),
            reference_image_path=reference_image_path,
        )
        return JSONResponse({"success": True, "context_vibe": context_vibe})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"Context/vibe generate failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/prompting/describe-characters")
async def describe_prompting_characters(payload: dict[str, Any]) -> JSONResponse:
    settings = payload.get("settings", {}) or {}
    selected_model = (
        settings.get("gemini_custom_model")
        or settings.get("gemini_model")
        or GeminiService.DEFAULT_MODEL
    )
    api_key = settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    reference_image_path = str(payload.get("reference_image_path", "")).strip()
    character_count = str(payload.get("character_count", "2") or "2")
    if not api_key:
        return JSONResponse({"success": False, "error": "Gemini API key not configured."}, status_code=400)
    if not reference_image_path:
        return JSONResponse({"success": False, "error": "Reference image is required."}, status_code=400)
    try:
        state._on_log_message(f"Character description generate requested: count={character_count}, model={selected_model}")
        result = state.pipeline.gemini_service.generate_character_descriptions(
            api_key=api_key,
            model=str(selected_model),
            character_count=character_count,
            reference_image_path=reference_image_path,
        )
        return JSONResponse({"success": True, **result})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"Character description generate failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/flux/image/prompts")
async def generate_flux_image_prompts(payload: dict[str, Any]) -> JSONResponse:
    settings = payload.get("settings", {}) or {}
    selected_model = (
        settings.get("gemini_custom_model")
        or settings.get("gemini_model")
        or GeminiService.DEFAULT_MODEL
    )
    api_key = settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    reference_image_path = str(payload.get("reference_image_path", "")).strip()
    image_count = max(1, min(4, int(payload.get("image_count", 1) or 1)))
    raw_prompt_ideas = payload.get("prompt_ideas", [])
    prompt_ideas = [str(item).strip() for item in raw_prompt_ideas[:image_count]] if isinstance(raw_prompt_ideas, list) else []
    if len(prompt_ideas) < image_count:
        prompt_ideas.extend([""] * (image_count - len(prompt_ideas)))
    character_mode = str(settings.get("veo_character_mode", "Solo") or "Solo")
    if not api_key:
        return JSONResponse({"success": False, "error": "Gemini API key not configured."}, status_code=400)
    if not reference_image_path:
        return JSONResponse({"success": False, "error": "Reference image is required."}, status_code=400)
    try:
        state._on_log_message(f"Flux image prompt generate requested: count={len(prompt_ideas)}, model={selected_model}")
        prompts = state.pipeline.gemini_service.generate_flux_image_prompts(
            api_key=api_key,
            model=str(selected_model),
            prompt_ideas=prompt_ideas,
            reference_image_path=reference_image_path,
            character_mode=character_mode,
        )
        return JSONResponse({"success": True, "prompts": prompts})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"Flux image prompt generate failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/wan/rewrite-prompts")
async def rewrite_wan_prompts(payload: dict[str, Any]) -> JSONResponse:
    settings = payload.get("settings", {}) or {}
    selected_model = (
        settings.get("gemini_custom_model")
        or settings.get("gemini_model")
        or GeminiService.DEFAULT_MODEL
    )
    api_key = settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    prompts = [str(item).strip() for item in payload.get("prompts", []) if str(item).strip()]
    reference_image_paths = [str(item).strip() for item in payload.get("reference_image_paths", []) if str(item).strip()]
    character_count = str(payload.get("character_count", "") or "").strip()
    character_mode = "Two Character Conversation" if character_count == "2" else str(settings.get("veo_character_mode", "Solo") or "Solo")
    if character_count == "1":
        character_mode = "Solo"
    character_a = str(payload.get("character_a", "")).strip()
    character_b = str(payload.get("character_b", "")).strip()
    if not api_key:
        return JSONResponse({"success": False, "error": "Gemini API key not configured."}, status_code=400)
    if not prompts:
        return JSONResponse({"success": False, "error": "At least one prompt is required."}, status_code=400)
    if not reference_image_paths:
        return JSONResponse({"success": False, "error": "At least one reference image is required."}, status_code=400)
    try:
        state._on_log_message(f"WAN prompt rewrite requested: count={len(prompts)}, model={selected_model}")
        rewritten = state.pipeline.gemini_service.rewrite_wan_prompts_with_character_labels(
            api_key=api_key,
            model=str(selected_model),
            prompts=prompts,
            reference_image_paths=reference_image_paths,
            character_mode=character_mode,
            character_a=character_a,
            character_b=character_b,
        )
        return JSONResponse({"success": True, "prompts": rewritten})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"WAN prompt rewrite failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/dialogs/open-file")
async def open_file_dialog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    selected = _pick_path(
        mode="file",
        initial_path=str(payload.get("initial_path", "") or ""),
        title=str(payload.get("title", "") or "Choose File"),
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All Files", "*.*")],
    )
    return {"success": True, "path": selected}


@app.post("/dialogs/open-folder")
async def open_folder_dialog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    selected = _pick_path(
        mode="folder",
        initial_path=str(payload.get("initial_path", "") or ""),
        title=str(payload.get("title", "") or "Choose Folder"),
    )
    return {"success": True, "path": selected}


@app.post("/dialogs/open-project")
async def open_project_dialog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    selected = _pick_path(
        mode="project",
        initial_path=str(payload.get("initial_path", "") or state.settings.export_folder),
        title=str(payload.get("title", "") or "Choose Existing Project Folder"),
    )
    return {"success": True, "path": selected}


@app.get("/file")
async def serve_file(path: str) -> FileResponse:
    target = Path(path).expanduser().resolve()
    allowed_roots = {
        APP_ROOT.resolve(),
        Path(state.settings.export_folder).expanduser().resolve(),
        Path(state.settings.generated_video_folder).expanduser().resolve(),
    }
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    if not any(root == target or root in target.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="File path not allowed.")
    return FileResponse(target)


@app.post("/connections/test-gemini")
async def test_gemini(payload: dict[str, Any]) -> dict[str, Any]:
    success, message = state.pipeline.gemini_service.test_connection(
        str(payload.get("api_key", "")).strip(),
        str(payload.get("model", "")).strip() or None,
    )
    return {"success": success, "message": message}


@app.post("/connections/test-edge")
async def test_edge(payload: dict[str, Any]) -> dict[str, Any]:
    success, message, preview_path = state.pipeline.edge_tts_service.test_voice(
        str(payload.get("voice", "")).strip(),
        int(payload.get("rate", 5)),
        int(payload.get("volume", 0)),
    )
    return {"success": success, "message": message, "preview_path": preview_path}


@app.post("/connections/test-elevenlabs")
async def test_elevenlabs(payload: dict[str, Any]) -> dict[str, Any]:
    success, message = state.pipeline.elevenlabs_service.test_connection(
        str(payload.get("api_key", "")).strip(),
        str(payload.get("voice_id", "")).strip(),
    )
    return {"success": success, "message": message}


@app.post("/connections/test-pexels")
async def test_pexels(payload: dict[str, Any]) -> dict[str, Any]:
    success, message = state.pipeline.visual_service.test_pexels_connection(str(payload.get("api_key", "")).strip())
    return {"success": success, "message": message}


@app.post("/connections/test-serpapi")
async def test_serpapi(payload: dict[str, Any]) -> dict[str, Any]:
    success, message = state.pipeline.visual_service.test_serpapi_connection(str(payload.get("api_key", "")).strip())
    return {"success": success, "message": message}


@app.post("/connections/test-pixabay")
async def test_pixabay(payload: dict[str, Any]) -> dict[str, Any]:
    success, message = state.pipeline.visual_service.test_pixabay_connection(str(payload.get("api_key", "")).strip())
    return {"success": success, "message": message}


@app.post("/image-finder/search")
async def image_finder_search(payload: dict[str, Any]) -> JSONResponse:
    settings = state.save_settings(payload.get("settings", {}))
    query = str(payload.get("query", "")).strip()
    if not query:
        return JSONResponse({"success": False, "error": "Search query is required."}, status_code=400)
    if not settings.serpapi_api_key and not settings.pexels_api_key and not settings.pixabay_api_key:
        return JSONResponse({"success": False, "error": "Add a SerpAPI, Pexels, or Pixabay API key in Connections first."}, status_code=400)
    try:
        results = state.pipeline.visual_service.search_reference_images(query=query, settings=settings, limit=12)
        return JSONResponse({"success": True, "results": results})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"Image finder search failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/image-finder/use-in-klein")
async def image_finder_use_in_klein(payload: dict[str, Any]) -> JSONResponse:
    image_urls = [str(item).strip() for item in payload.get("image_urls", []) if str(item).strip()]
    if not image_urls:
        image_url = str(payload.get("image_url", "")).strip()
        if image_url:
            image_urls = [image_url]
    filename_hint = str(payload.get("filename_hint", "reference.jpg")).strip() or "reference.jpg"
    output_dir = Path(r"C:\Users\adam_\Desktop\AutomatedVideoCreator\OutputImages") / "image-finder"
    try:
        imported_path = state.pipeline.visual_service.import_reference_image(
            image_urls=image_urls,
            filename_hint=filename_hint,
            output_dir=str(output_dir),
        )
        return JSONResponse({"success": True, "path": imported_path})
    except Exception as exc:  # noqa: BLE001
        state._on_log_message(f"Image finder import failed: {exc}")
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/connections/open-deevid-browser")
async def open_deevid_browser() -> dict[str, Any]:
    profile_dir = state.settings.deevid_profile_dir
    thread = threading.Thread(
        target=state.pipeline.deevid_service.open_login_browser,
        args=(profile_dir,),
        daemon=True,
    )
    thread.start()
    return {"success": True, "message": "DeeVid browser opened."}


@app.websocket("/ws/pipeline")
async def pipeline_ws(websocket: WebSocket) -> None:
    await state.hub.connect(websocket)
    try:
        await websocket.send_json({"type": "snapshot", **state.current_state_payload()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.hub.disconnect(websocket)
    except Exception:
        state.hub.disconnect(websocket)

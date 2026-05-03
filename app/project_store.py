from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from app.models import ProjectState


class ProjectStore:
    PROJECT_FILE = "project.json"

    def create_project(self, export_root: str, urls: list[str]) -> ProjectState:
        root = Path(export_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        project_name = self._build_project_name(urls)
        project_dir = root / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "assets").mkdir(exist_ok=True)
        state = ProjectState(
            project_name=project_name,
            project_dir=str(project_dir),
            export_dir=str(root),
            urls=urls,
        )
        self.save_state(state)
        return state

    def load_project(self, project_dir: str) -> ProjectState:
        project_path = Path(project_dir).expanduser().resolve()
        payload = json.loads((project_path / self.PROJECT_FILE).read_text(encoding="utf-8"))
        return ProjectState.from_dict(payload)

    def save_state(self, state: ProjectState) -> None:
        project_path = Path(state.project_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / self.PROJECT_FILE).write_text(
            json.dumps(state.to_dict(), indent=2),
            encoding="utf-8",
        )

    def write_text_file(self, state: ProjectState, filename: str, text: str) -> str:
        path = Path(state.project_dir) / filename
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _build_project_name(self, urls: list[str]) -> str:
        host = "project"
        if urls:
            parsed = urlparse(urls[0])
            host = parsed.netloc or "project"
        host = re.sub(r"[^a-zA-Z0-9]+", "-", host).strip("-").lower() or "project"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{host}-{timestamp}"

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import QThread, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config import AppSettings, DEFAULT_DEEVID_PROFILE_DIR, DEFAULT_EXPORT_DIR, VideoSettings
from app.models import PipelineStage
from app.pipeline import PipelineController
from app.services.deevid_service import DeeVidService
from app.services.edge_tts_service import EdgeTTSService
from app.services.elevenlabs_service import ElevenLabsService
from app.services.gemini_service import GeminiService
from app.services.visual_service import VisualService
from app.settings_store import SettingsStore
from app.ui.setup_wizard import SetupWizard


class MainWindow(QMainWindow):
    start_project_requested = Signal(list, str, dict)
    resume_project_requested = Signal(str, dict)
    pause_requested = Signal()
    stop_requested = Signal()
    continue_requested = Signal()
    regenerate_requested = Signal()

    STAGE_ORDER = [
        PipelineStage.EXTRACT.value,
        PipelineStage.SCRIPT.value,
        PipelineStage.VOICE.value,
        PipelineStage.VISUALS.value,
        PipelineStage.RENDER.value,
        PipelineStage.COMPLETE.value,
    ]
    GEMINI_OPTIONS = {
        "Gemini 2.5 Flash": GeminiService.DEFAULT_MODEL,
        "Gemini 2.5 Flash Lite": GeminiService.FLASH_LITE_FALLBACK,
        "Custom": "",
    }
    WORKFLOW_OPTIONS = ["Scene Automation"]
    NARRATION_OPTIONS = ["Edge TTS", "ElevenLabs", "Disabled"]
    VEO_PLANNING_OPTIONS = ["Auto"]
    VEO_CHARACTER_OPTIONS = ["Auto", "Speaking / Presenter", "Reenacting"]
    VEO_STYLE_OPTIONS = ["Reference-Driven"]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Automated Video Creator")
        self.resize(1680, 1020)
        self._current_preview_kind = "text"
        self._last_project_dir = ""
        self._connection_state = {
            "gemini": "Not Connected",
            "narration": "Not Connected",
            "visuals": "Optional",
            "ffmpeg": "Checking",
        }

        self.settings_store = SettingsStore()
        self.current_settings = self.settings_store.load()

        self.gemini_service = GeminiService()
        self.deevid_service = DeeVidService()
        self.edge_tts_service = EdgeTTSService()
        self.elevenlabs_service = ElevenLabsService()
        self.visual_service = VisualService()

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)

        self.pipeline_thread = QThread(self)
        self.pipeline = PipelineController()
        self.pipeline.moveToThread(self.pipeline_thread)
        self.pipeline_thread.start()

        self.start_project_requested.connect(self.pipeline.start_new_project, Qt.QueuedConnection)
        self.resume_project_requested.connect(self.pipeline.resume_project, Qt.QueuedConnection)
        self.pipeline.log_message.connect(self.append_log)
        self.pipeline.stage_changed.connect(self.on_stage_changed)
        self.pipeline.preview_ready.connect(self.on_preview_ready)
        self.pipeline.approval_required.connect(self.on_approval_required)
        self.pipeline.pipeline_finished.connect(self.on_pipeline_finished)
        self.pipeline.project_loaded.connect(self.on_project_loaded)

        self._build_ui()
        self._apply_style()
        self._load_settings_into_ui(self.current_settings)
        self._refresh_ffmpeg_status()
        self._refresh_connection_badges()
        self._set_buttons_for_idle()
        self._maybe_show_setup_wizard()
        QTimer.singleShot(150, self.auto_test_saved_connections)

        if self.current_settings.gemini_api_key and not self.gemini_key_edit.text():
            self.gemini_key_edit.setText(self.current_settings.gemini_api_key)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_top_shell_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_nav_rail())
        self.project_settings_page = self._build_project_settings_tab()
        self.accounts_page = self._build_accounts_tab()
        self.main_stack = QStackedWidget()
        self.main_stack.addWidget(self._build_project_shell())
        self.main_stack.addWidget(self._build_settings_shell())
        self.main_stack.addWidget(self._build_connect_shell())
        body.addWidget(self.main_stack, 1)
        outer.addLayout(body)

        open_folder = QAction("Open Current Project Folder", self)
        open_folder.triggered.connect(self.open_current_project_folder)
        self.menuBar().addAction(open_folder)

    def _build_top_shell_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        self.app_title_label = QLabel("Automated Video Creator")
        self.app_title_label.setProperty("shellTitle", True)
        layout.addWidget(self.app_title_label)
        divider = QLabel(" ")
        divider.setFixedWidth(1)
        divider.setProperty("shellDivider", True)
        layout.addWidget(divider)
        self.gemini_status_badge = QLabel()
        self.eleven_status_badge = QLabel()
        self.visual_status_badge = QLabel()
        self.ffmpeg_status_badge = QLabel()
        for badge in (
            self.gemini_status_badge,
            self.eleven_status_badge,
            self.visual_status_badge,
            self.ffmpeg_status_badge,
        ):
            badge.setProperty("statusBadge", True)
            layout.addWidget(badge)
        layout.addStretch(1)
        return widget

    def _build_nav_rail(self) -> QWidget:
        rail = QWidget()
        rail.setProperty("navRail", True)
        rail.setFixedWidth(74)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(10, 16, 10, 16)
        layout.setSpacing(12)

        self.nav_project_button = QPushButton("Project")
        self.nav_settings_button = QPushButton("Settings")
        self.nav_connect_button = QPushButton("Connect")
        for button in (self.nav_project_button, self.nav_settings_button, self.nav_connect_button):
            button.setProperty("navButton", True)
            button.setCheckable(True)
            button.setMinimumHeight(56)
        self.nav_project_button.setChecked(True)
        self.nav_project_button.clicked.connect(lambda: self._set_settings_tab_from_nav("project"))
        self.nav_settings_button.clicked.connect(lambda: self._set_settings_tab_from_nav("settings"))
        self.nav_connect_button.clicked.connect(lambda: self._set_settings_tab_from_nav("connect"))

        layout.addWidget(self.nav_project_button)
        layout.addWidget(self.nav_settings_button)
        layout.addWidget(self.nav_connect_button)
        layout.addStretch(1)

        avatar = QLabel("A")
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setProperty("navAvatar", True)
        avatar.setFixedSize(36, 36)
        layout.addWidget(avatar, alignment=Qt.AlignHCenter)
        return rail

    def _build_work_area(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        input_group = QGroupBox("Project Input")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(10)
        self.story_type_edit = QLineEdit()
        self.story_type_edit.setPlaceholderText("Story type, e.g. cartoon beef, brainrot facts, anime rivalry")
        self.character_brief_edit = QLineEdit()
        self.character_brief_edit.setPlaceholderText("Optional character notes, e.g. smug chemist, detective, presenter")
        self.character_brief_edit.setToolTip("Optional. If you upload a Veo character image, that image becomes the main character source for every scene.")
        self.story_type_edit.setMinimumHeight(40)
        self.character_brief_edit.setMinimumHeight(40)
        input_layout.addWidget(self.story_type_edit)
        input_layout.addWidget(self.character_brief_edit)
        self.url_input = QPlainTextEdit()
        self.url_input.setPlaceholderText("Describe the story or skit concept here.")
        self.url_input.setMinimumHeight(84)
        self.url_input.setMaximumHeight(120)
        input_layout.addWidget(self.url_input)

        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        self.export_folder_edit = QLineEdit(DEFAULT_EXPORT_DIR)
        self.export_folder_edit.setMinimumHeight(40)
        browse_button = QPushButton("Browse Export Folder")
        browse_button.setMinimumHeight(40)
        browse_button.clicked.connect(self.choose_export_folder)
        export_row.addWidget(self.export_folder_edit)
        export_row.addWidget(browse_button)
        input_layout.addLayout(export_row)

        self.start_button = QPushButton("Start Project")
        self.resume_button = QPushButton("Open Existing Project")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.regenerate_button = QPushButton("Regenerate Current Step")
        self.continue_button = QPushButton("Continue")
        self.start_button.setProperty("buttonRole", "primary")
        self.resume_button.setProperty("buttonRole", "ghost")
        self.pause_button.setProperty("buttonRole", "ghost")
        self.stop_button.setProperty("buttonRole", "danger")
        self.regenerate_button.setProperty("buttonRole", "ghost")
        self.continue_button.setProperty("buttonRole", "success")
        for button in (
            self.start_button,
            self.resume_button,
            self.pause_button,
            self.stop_button,
            self.regenerate_button,
            self.continue_button,
        ):
            button.setMinimumHeight(34)
        self.start_button.clicked.connect(self.start_project)
        self.resume_button.clicked.connect(self.resume_project)
        self.pause_button.clicked.connect(self._request_pause)
        self.stop_button.clicked.connect(self._request_stop)
        self.regenerate_button.clicked.connect(self._request_regenerate)
        self.continue_button.clicked.connect(self._request_continue)
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)
        primary_row.addWidget(self.start_button)
        primary_row.addWidget(self.resume_button)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)
        secondary_row.addWidget(self.pause_button)
        secondary_row.addWidget(self.stop_button)
        tertiary_row = QHBoxLayout()
        tertiary_row.setSpacing(8)
        tertiary_row.addWidget(self.regenerate_button)
        tertiary_row.addWidget(self.continue_button)
        input_layout.addLayout(primary_row)
        input_layout.addLayout(secondary_row)
        input_layout.addLayout(tertiary_row)
        layout.addWidget(input_group)

        progress_group = QGroupBox("Pipeline Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.current_stage_label = QLabel("Current stage: Idle")
        self.progress_list = QListWidget()
        for stage in self.STAGE_ORDER:
            QListWidgetItem(stage, self.progress_list)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.STAGE_ORDER) - 1)
        progress_layout.addWidget(self.current_stage_label)
        progress_layout.addWidget(self.progress_list)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group, 1)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_summary = QLabel("Preview results will appear here.")
        self.preview_summary.setWordWrap(True)
        self.preview_summary.setStyleSheet("font-weight: 600;")
        preview_layout.addWidget(self.preview_summary)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 10))
        preview_layout.addWidget(self.preview_text, 2)
        self.preview_image = QLabel()
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setMinimumHeight(320)
        self.preview_image.setStyleSheet("background: #0f172a; border-radius: 12px;")
        self.preview_image.hide()
        preview_layout.addWidget(self.preview_image, 2)
        self.video_widget = QVideoWidget()
        self.video_widget.hide()
        self.video_widget.setMinimumHeight(320)
        self.media_player.setVideoOutput(self.video_widget)
        preview_layout.addWidget(self.video_widget, 2)

        media_controls = QHBoxLayout()
        self.play_media_button = QPushButton("Play Preview")
        self.play_media_button.clicked.connect(self.toggle_media)
        self.play_media_button.setEnabled(False)
        self.open_file_button = QPushButton("Open Preview File")
        self.open_file_button.clicked.connect(self.open_preview_file)
        self.open_file_button.setEnabled(False)
        media_controls.addWidget(self.play_media_button)
        media_controls.addWidget(self.open_file_button)
        preview_layout.addLayout(media_controls)
        layout.addWidget(preview_group, 2)

        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(1000)
        logs_layout.addWidget(self.logs)
        layout.addWidget(logs_group, 1)
        return panel

    def _build_project_shell(self) -> QWidget:
        shell = QWidget()
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(20, 20, 20, 20)
        shell_layout.setSpacing(16)
        shell_layout.addWidget(self._build_work_area(), 1)
        quick_panel = self._build_quick_settings_panel()
        quick_panel.setMinimumWidth(330)
        quick_panel.setMaximumWidth(360)
        shell_layout.addWidget(quick_panel)
        return shell

    def _build_settings_shell(self) -> QWidget:
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(20, 20, 20, 20)
        shell_layout.setSpacing(16)
        title = QLabel("Project Settings")
        title.setProperty("shellTitle", True)
        subtitle = QLabel("Adjust output defaults, style, pacing, character behavior, and generation options.")
        subtitle.setProperty("shellSubtitle", True)
        shell_layout.addWidget(title)
        shell_layout.addWidget(subtitle)
        shell_layout.addWidget(self.project_settings_page, 1)
        return shell

    def _build_connect_shell(self) -> QWidget:
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(20, 20, 20, 20)
        shell_layout.setSpacing(16)
        title = QLabel("Accounts / Connections")
        title.setProperty("shellTitle", True)
        subtitle = QLabel("Connect Gemini, narration providers, stock APIs, and export destinations.")
        subtitle.setProperty("shellSubtitle", True)
        shell_layout.addWidget(title)
        shell_layout.addWidget(subtitle)
        shell_layout.addWidget(self.accounts_page, 1)
        return shell

    def _build_quick_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setProperty("quickRail", True)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        quick_group = QGroupBox("Quick Settings")
        quick_form = QFormLayout(quick_group)
        quick_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        quick_form.addRow("Quality", self.quality_combo)
        quick_form.addRow("Platform", self.platform_combo)
        quick_form.addRow("Length (s)", self.length_combo)
        quick_form.addRow("Visual Provider", self.visual_provider_combo)
        layout.addWidget(quick_group)

        info_group = QGroupBox("Output Info")
        info_layout = QVBoxLayout(info_group)
        self.quick_info_label = QLabel()
        self.quick_info_label.setWordWrap(True)
        info_layout.addWidget(self.quick_info_label)
        layout.addWidget(info_group)

        actions_group = QGroupBox("Workspace")
        actions_layout = QVBoxLayout(actions_group)
        open_accounts = QPushButton("Open Accounts / Connections")
        open_accounts.setProperty("buttonRole", "ghost")
        open_accounts.clicked.connect(lambda: self._set_settings_tab_from_nav("connect"))
        open_settings = QPushButton("Open Project Settings")
        open_settings.setProperty("buttonRole", "ghost")
        open_settings.clicked.connect(lambda: self._set_settings_tab_from_nav("settings"))
        open_folder = QPushButton("Open Current Project Folder")
        open_folder.setProperty("buttonRole", "ghost")
        open_folder.clicked.connect(self.open_current_project_folder)
        actions_layout.addWidget(open_accounts)
        actions_layout.addWidget(open_settings)
        actions_layout.addWidget(open_folder)
        layout.addWidget(actions_group)
        layout.addStretch(1)
        return panel

    def _set_settings_tab_from_nav(self, target: str) -> None:
        mapping = {
            "project": 0,
            "settings": 1,
            "connect": 2,
        }
        index = mapping.get(target, 0)
        self.main_stack.setCurrentIndex(index)
        self.nav_project_button.setChecked(target == "project")
        self.nav_settings_button.setChecked(target == "settings")
        self.nav_connect_button.setChecked(target == "connect")

    def _refresh_quick_info(self) -> None:
        narrator = self.narration_engine_combo.currentText() if hasattr(self, "narration_engine_combo") else "Edge TTS"
        model = self.gemini_model_combo.currentText() if hasattr(self, "gemini_model_combo") else "Gemini 2.5 Flash"
        provider = self.visual_provider_combo.currentText() if hasattr(self, "visual_provider_combo") else "Veo + Stock"
        self.quick_info_label.setText(
            "Resolution          1080 x 1920\n"
            f"Narrator            {narrator}\n"
            f"Model               {model}\n"
            f"Visuals             {provider}\n"
            f"Veo                 {self.veo_model_label.text().strip() if hasattr(self, 'veo_model_label') else 'veo-3.1-lite-generate-preview'}"
        )

    def _build_project_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        group = QGroupBox("Project Settings")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["High", "Medium", "Low"])
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "30", "60"])
        self.length_combo = QComboBox()
        self.length_combo.addItems(["20", "30", "45", "60"])
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(["YouTube Shorts", "TikTok", "Instagram Reels"])
        self.workflow_mode_combo = QComboBox()
        self.workflow_mode_combo.addItems(self.WORKFLOW_OPTIONS)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["True Crime Mode", "Finance Tips Mode", "Facts / Listicle Mode", "General Viral Mode"])
        self.visual_provider_combo = QComboBox()
        self.visual_provider_combo.addItems(["Veo + Stock", "DeeVid Automation", "Stock Footage", "Hybrid", "AI Images", "Local Fallback"])
        self.veo_planning_combo = QComboBox()
        self.veo_planning_combo.addItems(self.VEO_PLANNING_OPTIONS)
        self.veo_planning_combo.setToolTip("Choose how the app structures the short. Auto selects Story or List mode from the source content.")
        self.veo_character_combo = QComboBox()
        self.veo_character_combo.addItems(self.VEO_CHARACTER_OPTIONS)
        self.veo_character_combo.setToolTip(
            "Auto mixes presenter-style hooks/endings with reenactment in the middle. "
            "Speaking / Presenter keeps the reference character talking to camera. "
            "Reenacting keeps the character acting through the story without direct presenter delivery."
        )
        self.veo_visual_style_combo = QComboBox()
        self.veo_visual_style_combo.addItems(self.VEO_STYLE_OPTIONS)
        self.veo_visual_style_combo.setToolTip("Choose one consistent visual style for all generated Veo scenes in the project.")
        self.veo_reference_image_edit = QLineEdit()
        self.veo_reference_image_edit.setPlaceholderText("Optional character/reference image for Veo 3.1 scenes")
        self.veo_reference_image_edit.setToolTip("Optional. Use one character or subject image to guide all Veo scenes in this project. Veo 3.1 reference-image generations use 8-second source clips.")
        veo_reference_browse = QPushButton("Browse")
        veo_reference_browse.clicked.connect(self.choose_veo_reference_image)
        veo_reference_clear = QPushButton("Clear")
        veo_reference_clear.clicked.connect(lambda: self.veo_reference_image_edit.clear())
        veo_reference_row = QHBoxLayout()
        veo_reference_row.setContentsMargins(0, 0, 0, 0)
        veo_reference_row.addWidget(self.veo_reference_image_edit)
        veo_reference_row.addWidget(veo_reference_browse)
        veo_reference_row.addWidget(veo_reference_clear)
        veo_reference_widget = QWidget()
        veo_reference_widget.setLayout(veo_reference_row)
        self.music_style_combo = QComboBox()
        self.music_style_combo.addItems(["Suspense", "Ambient", "Corporate", "None"])
        form.addRow("Output Quality", self.quality_combo)
        form.addRow("FPS", self.fps_combo)
        form.addRow("Length Target (s)", self.length_combo)
        form.addRow("Platform", self.platform_combo)
        form.addRow("Visual Provider", self.visual_provider_combo)
        form.addRow("Character Mode", self.veo_character_combo)
        form.addRow("Veo Reference Image", veo_reference_widget)
        form.addRow("Music Style", self.music_style_combo)
        layout.addWidget(group)
        save_button = QPushButton("Save Defaults")
        save_button.setProperty("buttonRole", "primary")
        save_button.clicked.connect(self.save_all_settings)
        layout.addWidget(save_button)
        layout.addStretch(1)
        return widget

    def _build_accounts_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        scroll.setWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        gemini_group = QGroupBox("Gemini / Google AI")
        gemini_form = QFormLayout(gemini_group)
        gemini_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.gemini_key_edit = QLineEdit()
        self.gemini_key_edit.setEchoMode(QLineEdit.Password)
        self.gemini_key_edit.setMinimumWidth(280)
        self.gemini_model_combo = QComboBox()
        self.gemini_model_combo.addItems(list(self.GEMINI_OPTIONS.keys()))
        self.gemini_model_combo.setMinimumWidth(280)
        self.gemini_custom_model_edit = QLineEdit()
        self.gemini_custom_model_edit.setPlaceholderText("Enter custom Gemini model id only if needed")
        self.gemini_custom_model_edit.setMinimumWidth(280)
        self.gemini_status_label = QLabel("Status: Not Connected")
        self.gemini_status_label.setWordWrap(True)
        gemini_form.addRow("API Key", self.gemini_key_edit)
        gemini_form.addRow("Model", self.gemini_model_combo)
        gemini_form.addRow("Custom Model", self.gemini_custom_model_edit)
        gemini_form.addRow("Status", self.gemini_status_label)
        gemini_buttons = QHBoxLayout()
        gemini_test = QPushButton("Test Connection")
        gemini_test.clicked.connect(self.test_gemini_connection)
        gemini_save = QPushButton("Save")
        gemini_save.clicked.connect(self.save_gemini_settings)
        gemini_buttons.addWidget(gemini_test)
        gemini_buttons.addWidget(gemini_save)
        gemini_form.addRow("", gemini_buttons)
        layout.addWidget(gemini_group)

        narration_group = QGroupBox("Narration")
        narration_form = QFormLayout(narration_group)
        narration_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.narration_engine_combo = QComboBox()
        self.narration_engine_combo.addItems(self.NARRATION_OPTIONS)
        self.narration_engine_combo.currentTextChanged.connect(self._update_narration_engine_ui)
        narration_form.addRow("Narration Engine", self.narration_engine_combo)

        self.edge_voice_combo = QComboBox()
        self.edge_voice_combo.addItems(EdgeTTSService.POPULAR_VOICES)
        narration_form.addRow("Edge Voice", self.edge_voice_combo)

        self.edge_speed_slider = QSlider(Qt.Horizontal)
        self.edge_speed_slider.setRange(-20, 20)
        self.edge_speed_slider.setSingleStep(1)
        self.edge_speed_value_label = QLabel("+5%")
        self.edge_speed_slider.valueChanged.connect(self._update_edge_slider_labels)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.edge_speed_slider)
        speed_row.addWidget(self.edge_speed_value_label)
        narration_form.addRow("Speed", speed_row)

        self.edge_volume_slider = QSlider(Qt.Horizontal)
        self.edge_volume_slider.setRange(-10, 10)
        self.edge_volume_slider.setSingleStep(1)
        self.edge_volume_value_label = QLabel("+0%")
        self.edge_volume_slider.valueChanged.connect(self._update_edge_slider_labels)
        volume_row = QHBoxLayout()
        volume_row.addWidget(self.edge_volume_slider)
        volume_row.addWidget(self.edge_volume_value_label)
        narration_form.addRow("Volume", volume_row)

        self.edge_tts_status_label = QLabel("Status: Ready")
        self.edge_tts_status_label.setWordWrap(True)
        narration_form.addRow("Edge Status", self.edge_tts_status_label)
        edge_buttons = QHBoxLayout()
        edge_test = QPushButton("Test Voice")
        edge_test.clicked.connect(self.test_edge_tts_voice)
        edge_save = QPushButton("Save")
        edge_save.clicked.connect(self.save_narration_settings)
        edge_buttons.addWidget(edge_test)
        edge_buttons.addWidget(edge_save)
        narration_form.addRow("", edge_buttons)

        self.elevenlabs_key_edit = QLineEdit()
        self.elevenlabs_key_edit.setEchoMode(QLineEdit.Password)
        self.elevenlabs_key_edit.setMinimumWidth(280)
        self.voice_id_edit = QLineEdit()
        self.voice_id_edit.setMinimumWidth(280)
        self.elevenlabs_status_label = QLabel("Status: Optional")
        self.elevenlabs_status_label.setWordWrap(True)
        narration_form.addRow("ElevenLabs API Key", self.elevenlabs_key_edit)
        narration_form.addRow("ElevenLabs Voice ID", self.voice_id_edit)
        narration_form.addRow("ElevenLabs Status", self.elevenlabs_status_label)
        eleven_buttons = QHBoxLayout()
        eleven_test = QPushButton("Test Voice Connection")
        eleven_test.clicked.connect(self.test_elevenlabs_connection)
        eleven_save = QPushButton("Save")
        eleven_save.clicked.connect(self.save_narration_settings)
        eleven_buttons.addWidget(eleven_test)
        eleven_buttons.addWidget(eleven_save)
        narration_form.addRow("", eleven_buttons)
        layout.addWidget(narration_group)

        stock_group = QGroupBox("Stock Footage Providers")
        stock_form = QFormLayout(stock_group)
        stock_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.pexels_key_edit = QLineEdit()
        self.pexels_key_edit.setEchoMode(QLineEdit.Password)
        self.pexels_key_edit.setMinimumWidth(280)
        self.pexels_status_label = QLabel("Status: Optional")
        self.pexels_status_label.setWordWrap(True)
        self.pixabay_key_edit = QLineEdit()
        self.pixabay_key_edit.setEchoMode(QLineEdit.Password)
        self.pixabay_key_edit.setMinimumWidth(280)
        self.pixabay_status_label = QLabel("Status: Optional")
        self.pixabay_status_label.setWordWrap(True)
        stock_form.addRow("Pexels API Key", self.pexels_key_edit)
        pexels_buttons = QHBoxLayout()
        pexels_test = QPushButton("Test Connection")
        pexels_test.clicked.connect(self.test_pexels_connection)
        pexels_save = QPushButton("Save")
        pexels_save.clicked.connect(self.save_visual_settings)
        pexels_buttons.addWidget(pexels_test)
        pexels_buttons.addWidget(pexels_save)
        stock_form.addRow("Pexels", pexels_buttons)
        stock_form.addRow("Pexels Status", self.pexels_status_label)
        stock_form.addRow("Pixabay API Key", self.pixabay_key_edit)
        pixabay_buttons = QHBoxLayout()
        pixabay_test = QPushButton("Test Connection")
        pixabay_test.clicked.connect(self.test_pixabay_connection)
        pixabay_save = QPushButton("Save")
        pixabay_save.clicked.connect(self.save_visual_settings)
        pixabay_buttons.addWidget(pixabay_test)
        pixabay_buttons.addWidget(pixabay_save)
        stock_form.addRow("Pixabay", pixabay_buttons)
        stock_form.addRow("Pixabay Status", self.pixabay_status_label)
        layout.addWidget(stock_group)

        veo_group = QGroupBox("Veo / Generated Video Library")
        veo_form = QFormLayout(veo_group)
        veo_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.veo_model_label = QLabel("veo-3.1-lite-generate-preview")
        self.veo_model_label.setToolTip("Default Veo generation model used for new AI clips.")
        self.veo_summary_label = QLabel("New projects will auto-check the generated video library and reuse exact clip matches before creating new Veo clips.")
        self.veo_summary_label.setWordWrap(True)
        self.veo_estimate_label = QLabel("Estimate: no project analyzed yet.")
        self.veo_estimate_label.setWordWrap(True)
        veo_form.addRow("Veo Model", self.veo_model_label)
        veo_form.addRow("Behavior", self.veo_summary_label)
        veo_form.addRow("Pre-run Estimate", self.veo_estimate_label)
        layout.addWidget(veo_group)

        export_group = QGroupBox("Export Settings")
        export_form = QFormLayout(export_group)
        export_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.default_export_edit = QLineEdit()
        self.default_export_edit.setMinimumWidth(280)
        export_browse = QPushButton("Browse")
        export_browse.clicked.connect(self.choose_default_export_folder)
        export_row = QHBoxLayout()
        export_row.addWidget(self.default_export_edit)
        export_row.addWidget(export_browse)
        self.generated_video_folder_edit = QLineEdit()
        self.generated_video_folder_edit.setMinimumWidth(280)
        generated_video_browse = QPushButton("Browse")
        generated_video_browse.clicked.connect(self.choose_generated_video_folder)
        generated_video_row = QHBoxLayout()
        generated_video_row.addWidget(self.generated_video_folder_edit)
        generated_video_row.addWidget(generated_video_browse)
        self.deevid_profile_edit = QLineEdit(DEFAULT_DEEVID_PROFILE_DIR)
        self.deevid_profile_edit.setMinimumWidth(280)
        deevid_profile_browse = QPushButton("Browse")
        deevid_profile_browse.clicked.connect(self.choose_deevid_profile_folder)
        deevid_profile_row = QHBoxLayout()
        deevid_profile_row.addWidget(self.deevid_profile_edit)
        deevid_profile_row.addWidget(deevid_profile_browse)
        deevid_login_button = QPushButton("Open DeeVid Login Browser")
        deevid_login_button.clicked.connect(self.open_deevid_login_browser)
        self.default_resolution_label = QLabel("1080 x 1920")
        export_form.addRow("Default Save Folder", export_row)
        export_form.addRow("Generated Video Library", generated_video_row)
        export_form.addRow("DeeVid Browser Profile", deevid_profile_row)
        export_form.addRow("", deevid_login_button)
        export_form.addRow("Output Resolution", self.default_resolution_label)
        self.export_fps_combo = QComboBox()
        self.export_fps_combo.addItems(["24", "30", "60"])
        self.export_platform_combo = QComboBox()
        self.export_platform_combo.addItems(["YouTube Shorts", "TikTok", "Instagram Reels"])
        export_form.addRow("Default FPS", self.export_fps_combo)
        export_form.addRow("Default Platform", self.export_platform_combo)
        export_save = QPushButton("Save Export Settings")
        export_save.clicked.connect(self.save_export_settings)
        export_form.addRow("", export_save)
        layout.addWidget(export_group)
        layout.addStretch(1)
        return scroll

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0d1117;
                color: #e6edf3;
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QMenuBar {
                background: #0d1117;
                color: #e6edf3;
                border-bottom: 1px solid #30363d;
                padding: 4px 10px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                border-radius: 8px;
            }
            QMenuBar::item:selected {
                background: #1c2128;
            }
            QLabel[shellTitle="true"] {
                font-size: 20px;
                font-weight: 700;
                color: #f0f6fc;
                padding-right: 10px;
            }
            QLabel[shellDivider="true"] {
                background: #30363d;
                min-width: 1px;
                max-width: 1px;
                min-height: 28px;
                margin-right: 6px;
            }
            QWidget[navRail="true"] {
                background: #161b22;
                border-right: 1px solid #30363d;
            }
            QPushButton[navButton="true"] {
                background: transparent;
                color: #7d8590;
                border: 1px solid transparent;
                border-radius: 12px;
                padding: 8px 6px;
                text-align: center;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton[navButton="true"]:hover {
                background: #1c2128;
                color: #e6edf3;
                border: 1px solid #30363d;
            }
            QPushButton[navButton="true"]:checked {
                background: rgba(31,111,235,0.18);
                color: #58a6ff;
                border: 1px solid rgba(88,166,255,0.55);
            }
            QLabel[navAvatar="true"] {
                background: rgba(31,111,235,0.18);
                color: #58a6ff;
                border: 1px solid rgba(88,166,255,0.35);
                border-radius: 18px;
                font-weight: 700;
            }
            QWidget[quickRail="true"] {
                background: #161b22;
                border-left: 1px solid #30363d;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 14px;
                margin-top: 10px;
                padding: 14px;
                background: #161b22;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                color: #7d8590;
                padding: 0 8px;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.08em;
            }
            QLabel {
                background: transparent;
            }
            QPushButton {
                background: #1f6feb;
                color: white;
                border: 1px solid transparent;
                border-radius: 10px;
                padding: 10px 14px;
                min-height: 18px;
                font-weight: 600;
            }
            QPushButton:hover:!disabled {
                background: #1a5fd0;
            }
            QPushButton:pressed:!disabled {
                background: #164fae;
            }
            QPushButton:disabled {
                background: #21262d;
                color: #7d8590;
                border: 1px solid #30363d;
            }
            QPushButton[buttonRole="ghost"] {
                background: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
            }
            QPushButton[buttonRole="ghost"]:hover:!disabled {
                background: #1c2128;
                border: 1px solid #58a6ff;
            }
            QPushButton[buttonRole="danger"] {
                background: transparent;
                color: #f85149;
                border: 1px solid rgba(248,81,73,0.45);
            }
            QPushButton[buttonRole="danger"]:hover:!disabled {
                background: rgba(248,81,73,0.14);
            }
            QPushButton[buttonRole="success"] {
                background: #238636;
                color: white;
            }
            QPushButton[buttonRole="success"]:hover:!disabled {
                background: #2ea043;
            }
            QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QListWidget, QTabWidget::pane, QScrollArea {
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 8px;
                background: #1c2128;
                color: #e6edf3;
                selection-background-color: #1f6feb;
                selection-color: white;
            }
            QLineEdit, QComboBox {
                min-height: 26px;
                font-size: 14px;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QListWidget:focus {
                border: 1px solid #58a6ff;
            }
            QPlainTextEdit, QTextEdit {
                line-height: 1.5;
            }
            QComboBox::drop-down {
                border: none;
                width: 26px;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #7d8590;
                margin-right: 8px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 8px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: rgba(31,111,235,0.22);
                color: #f0f6fc;
            }
            QListWidget::item:hover {
                background: #21262d;
            }
            QTabWidget::pane {
                margin-top: 10px;
                padding: 10px;
                background: #161b22;
            }
            QTabBar::tab {
                background: #1c2128;
                color: #7d8590;
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 10px 14px;
                margin-right: 6px;
                min-width: 70px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: rgba(31,111,235,0.18);
                color: #e6edf3;
                border: 1px solid rgba(88,166,255,0.55);
            }
            QTabBar::tab:hover:!selected {
                background: #21262d;
                color: #e6edf3;
            }
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 10px;
                overflow: hidden;
                background: #1c2128;
                text-align: center;
                color: #e6edf3;
                min-height: 12px;
            }
            QProgressBar::chunk {
                background: #1f6feb;
                border-radius: 10px;
            }
            QLabel[statusBadge="true"] {
                background: #1c2128;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 999px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QScrollBar:vertical {
                background: #161b22;
                width: 8px;
                margin: 4px 0 4px 0;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #484f58;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar:horizontal, QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: transparent;
                border: none;
                height: 0;
                width: 0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #30363d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
                border: 1px solid #1f6feb;
            }
            QSlider::sub-page:horizontal {
                background: #1f6feb;
                border-radius: 3px;
            }
            """
        )

    def _load_settings_into_ui(self, settings: AppSettings) -> None:
        self.gemini_key_edit.setText(settings.gemini_api_key)
        self.narration_engine_combo.setCurrentText(settings.narration_engine or "Edge TTS")
        self.edge_voice_combo.setCurrentText(settings.edge_tts_voice or "en-US-GuyNeural")
        self.edge_speed_slider.setValue(int(settings.edge_tts_rate))
        self.edge_volume_slider.setValue(int(settings.edge_tts_volume))
        self.elevenlabs_key_edit.setText(settings.elevenlabs_api_key)
        self.voice_id_edit.setText(settings.elevenlabs_voice_id)
        self.pexels_key_edit.setText(settings.pexels_api_key)
        self.pixabay_key_edit.setText(settings.pixabay_api_key)
        self.gemini_custom_model_edit.setText(settings.gemini_custom_model)
        self.default_export_edit.setText(settings.export_folder)
        self.generated_video_folder_edit.setText(settings.generated_video_folder)
        self.deevid_profile_edit.setText(settings.deevid_profile_dir)
        self.export_folder_edit.setText(settings.export_folder)
        self.workflow_mode_combo.setCurrentText("Scene Automation")
        self.veo_planning_combo.setCurrentText("Auto")
        self.veo_character_combo.setCurrentText(settings.veo_character_mode or "Auto")
        self.veo_visual_style_combo.setCurrentText("Reference-Driven")
        self.veo_reference_image_edit.setText(settings.veo_reference_image_path or "")
        self.veo_model_label.setText(settings.veo_model or "veo-3.1-lite-generate-preview")
        label = settings.gemini_model_label or "Gemini 2.5 Flash"
        index = self.gemini_model_combo.findText(label)
        if index < 0:
            if settings.gemini_model == GeminiService.FLASH_LITE_FALLBACK:
                index = self.gemini_model_combo.findText("Gemini 2.5 Flash Lite")
            else:
                index = self.gemini_model_combo.findText("Custom")
        self.gemini_model_combo.setCurrentIndex(max(0, index))
        self.quality_combo.setCurrentText(settings.video.output_quality)
        self.fps_combo.setCurrentText(str(settings.video.fps))
        self.length_combo.setCurrentText(str(settings.video.length_target))
        self.platform_combo.setCurrentText(settings.video.platform)
        self.export_fps_combo.setCurrentText(str(settings.video.fps))
        self.export_platform_combo.setCurrentText(settings.video.platform)
        self.mode_combo.setCurrentText("General Viral Mode")
        self.visual_provider_combo.setCurrentText(settings.video.visual_provider)
        self.music_style_combo.setCurrentText(settings.video.music_style)
        self._update_edge_slider_labels()
        self._update_narration_engine_ui()
        self._refresh_quick_info()

    def collect_settings(self) -> AppSettings:
        model_label = self.gemini_model_combo.currentText()
        gemini_model = self._resolved_gemini_model()
        fps_value = int(self.fps_combo.currentText())
        platform_value = self.platform_combo.currentText()
        return AppSettings(
            workflow_mode="Scene Automation",
            gemini_api_key=self.gemini_key_edit.text().strip(),
            gemini_model=gemini_model,
            gemini_model_label=model_label,
            gemini_custom_model=self.gemini_custom_model_edit.text().strip(),
            veo_enabled=True,
            veo_model=self.veo_model_label.text().strip() or "veo-3.1-lite-generate-preview",
            veo_planning_mode="Auto",
            veo_character_mode=self.veo_character_combo.currentText(),
            veo_visual_style="Reference-Driven",
            veo_reference_image_path=self.veo_reference_image_edit.text().strip(),
            generated_video_folder=self.generated_video_folder_edit.text().strip(),
            deevid_profile_dir=self.deevid_profile_edit.text().strip(),
            narration_engine=self.narration_engine_combo.currentText(),
            edge_tts_voice=self.edge_voice_combo.currentText(),
            edge_tts_rate=int(self.edge_speed_slider.value()),
            edge_tts_volume=int(self.edge_volume_slider.value()),
            elevenlabs_api_key=self.elevenlabs_key_edit.text().strip(),
            elevenlabs_voice_id=self.voice_id_edit.text().strip(),
            pexels_api_key=self.pexels_key_edit.text().strip(),
            pixabay_api_key=self.visual_service._normalize_api_key(self.pixabay_key_edit.text()),
            content_mode="General Viral Mode",
            export_folder=self.default_export_edit.text().strip() or DEFAULT_EXPORT_DIR,
            video=VideoSettings(
                output_quality=self.quality_combo.currentText(),
                fps=fps_value,
                length_target=int(self.length_combo.currentText()),
                platform=platform_value,
                visual_provider=self.visual_provider_combo.currentText(),
                music_style=self.music_style_combo.currentText(),
            ),
        )

    def save_all_settings(self) -> None:
        self.current_settings = self.collect_settings()
        self.settings_store.save(self.current_settings)
        self.export_folder_edit.setText(self.current_settings.export_folder)
        self._normalize_connection_inputs()
        self._refresh_connection_badges()
        self._refresh_quick_info()
        self.append_log("Settings saved.")
        QTimer.singleShot(0, self.auto_test_saved_connections)

    def save_gemini_settings(self) -> None:
        self.save_all_settings()
        self.gemini_status_label.setText("Status: Saved")

    def save_narration_settings(self) -> None:
        self.save_all_settings()
        self.edge_tts_status_label.setText("Status: Saved")
        self.elevenlabs_status_label.setText("Status: Saved")

    def save_visual_settings(self) -> None:
        self.save_all_settings()
        self.pexels_status_label.setText("Status: Saved")
        self.pixabay_status_label.setText("Status: Saved")

    def save_export_settings(self) -> None:
        self.fps_combo.setCurrentText(self.export_fps_combo.currentText())
        self.platform_combo.setCurrentText(self.export_platform_combo.currentText())
        self.save_all_settings()
        self.export_folder_edit.setText(self.default_export_edit.text().strip())

    def start_project(self) -> None:
        source_mode = "scene"
        urls: list[str] = []
        if not self.url_input.toPlainText().strip():
            QMessageBox.warning(self, "Missing Scene Concept", "Please enter a story concept or prompt.")
            return
        if not self._validate_required_accounts_for_start():
            return
        self.save_all_settings()
        self.current_settings.export_folder = self.export_folder_edit.text().strip() or self.current_settings.export_folder
        self.settings_store.save(self.current_settings)
        self.logs.clear()
        self.preview_text.clear()
        self.veo_estimate_label.setText("Estimate: preparing project...")
        self._set_buttons_for_running(waiting=False)
        payload = self.current_settings.to_dict()
        payload["source_mode"] = source_mode
        payload["source_prompt"] = self.url_input.toPlainText().strip()
        payload["character_brief"] = self.character_brief_edit.text().strip()
        payload["story_type"] = self.story_type_edit.text().strip()
        self.start_project_requested.emit([], self.current_settings.export_folder, payload)

    def resume_project(self) -> None:
        if not self._validate_required_accounts_for_start():
            return
        project_dir = QFileDialog.getExistingDirectory(self, "Select Existing Project Folder", self.export_folder_edit.text())
        if not project_dir:
            return
        self.save_all_settings()
        self._set_buttons_for_running(waiting=False)
        self.resume_project_requested.emit(project_dir, self.current_settings.to_dict())

    def _request_pause(self) -> None:
        self.pipeline.request_pause()

    def _request_stop(self) -> None:
        self.pipeline.request_stop()

    def _request_continue(self) -> None:
        self.pipeline.continue_after_approval()

    def _request_regenerate(self) -> None:
        self.pipeline.regenerate_current_step()

    def test_gemini_connection(self) -> None:
        self._normalize_connection_inputs()
        _, message = self.gemini_service.test_connection(self.gemini_key_edit.text().strip(), self._resolved_gemini_model())
        self.gemini_status_label.setText(f"Status: {message}")
        self._connection_state["gemini"] = message
        self._refresh_connection_badges()

    def test_edge_tts_voice(self) -> None:
        self._normalize_connection_inputs()
        success, message, preview_path = self.edge_tts_service.test_voice(
            self.edge_voice_combo.currentText(),
            int(self.edge_speed_slider.value()),
            int(self.edge_volume_slider.value()),
        )
        self.edge_tts_status_label.setText(f"Status: {message}")
        self._connection_state["narration"] = "Ready" if success else message
        self._refresh_connection_badges()
        if success and preview_path:
            self.on_preview_ready("audio", preview_path)
            self.append_log("Edge TTS preview generated successfully.")

    def test_elevenlabs_connection(self) -> None:
        self._normalize_connection_inputs()
        _, message = self.elevenlabs_service.test_connection(
            self.elevenlabs_key_edit.text().strip(),
            self.voice_id_edit.text().strip(),
        )
        self.elevenlabs_status_label.setText(f"Status: {message}")
        if self.narration_engine_combo.currentText() == "ElevenLabs":
            self._connection_state["narration"] = message
        self._refresh_connection_badges()

    def test_pexels_connection(self) -> None:
        self._normalize_connection_inputs()
        _, message = self.visual_service.test_pexels_connection(self.pexels_key_edit.text().strip())
        self.pexels_status_label.setText(f"Status: {message}")
        self._update_visual_status_from_stock_tests()

    def test_pixabay_connection(self) -> None:
        self._normalize_connection_inputs()
        _, message = self.visual_service.test_pixabay_connection(self.pixabay_key_edit.text().strip())
        self.pixabay_status_label.setText(f"Status: {message}")
        if "Connected" in message:
            self.append_log(f"Pixabay connection test passed: {message}")
        else:
            self.append_log(f"Pixabay connection test result: {message}")
        self._update_visual_status_from_stock_tests()

    def auto_test_saved_connections(self) -> None:
        self._normalize_connection_inputs()
        if self.gemini_key_edit.text().strip():
            _, message = self.gemini_service.test_connection(self.gemini_key_edit.text().strip(), self._resolved_gemini_model())
            self.gemini_status_label.setText(f"Status: {message}")
            self._connection_state["gemini"] = message
        else:
            self.gemini_status_label.setText("Status: Not Connected")
            self._connection_state["gemini"] = "Not Connected"

        edge_ready, edge_message = self.edge_tts_service.check_available()
        self.edge_tts_status_label.setText(f"Status: {edge_message}")

        if self.elevenlabs_key_edit.text().strip() and self.voice_id_edit.text().strip():
            _, message = self.elevenlabs_service.test_connection(
                self.elevenlabs_key_edit.text().strip(),
                self.voice_id_edit.text().strip(),
            )
            self.elevenlabs_status_label.setText(f"Status: {message}")
        else:
            self.elevenlabs_status_label.setText("Status: Optional")

        engine = self.narration_engine_combo.currentText()
        if engine == "Edge TTS":
            self._connection_state["narration"] = edge_message if not edge_ready else "Ready"
        elif engine == "ElevenLabs":
            if self.elevenlabs_key_edit.text().strip() and self.voice_id_edit.text().strip():
                self._connection_state["narration"] = self.elevenlabs_status_label.text().replace("Status: ", "", 1)
            else:
                self._connection_state["narration"] = "Not Connected"
        else:
            self._connection_state["narration"] = "Disabled"

        if self.pexels_key_edit.text().strip():
            _, message = self.visual_service.test_pexels_connection(self.pexels_key_edit.text().strip())
            self.pexels_status_label.setText(f"Status: {message}")
        else:
            self.pexels_status_label.setText("Status: Optional")

        if self.pixabay_key_edit.text().strip():
            _, message = self.visual_service.test_pixabay_connection(self.pixabay_key_edit.text().strip())
            self.pixabay_status_label.setText(f"Status: {message}")
        else:
            self.pixabay_status_label.setText("Status: Optional")
        self._update_visual_status_from_stock_tests()
        self._refresh_connection_badges()

    def choose_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", self.export_folder_edit.text())
        if folder:
            self.export_folder_edit.setText(folder)
            self.default_export_edit.setText(folder)

    def choose_default_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Default Export Folder", self.default_export_edit.text())
        if folder:
            self.default_export_edit.setText(folder)
            self.export_folder_edit.setText(folder)

    def choose_generated_video_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Generated Video Library Folder", self.generated_video_folder_edit.text())
        if folder:
            self.generated_video_folder_edit.setText(folder)

    def choose_deevid_profile_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DeeVid Browser Profile Folder", self.deevid_profile_edit.text())
        if folder:
            self.deevid_profile_edit.setText(folder)

    def choose_veo_reference_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Veo Reference Image",
            str(Path(self.veo_reference_image_edit.text()).parent if self.veo_reference_image_edit.text().strip() else Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self.veo_reference_image_edit.setText(file_path)

    def open_deevid_login_browser(self) -> None:
        self._normalize_connection_inputs()
        ready, message = self.deevid_service.check_available()
        if not ready:
            QMessageBox.warning(self, "DeeVid Automation", message)
            return
        self.append_log("Opening dedicated DeeVid login browser. Log in once there, then close it when finished.")
        self.deevid_service.open_login_browser(self.deevid_profile_edit.text().strip())

    def on_stage_changed(self, stage: str) -> None:
        self.current_stage_label.setText(f"Current stage: {stage}")
        if stage in self.STAGE_ORDER:
            index = self.STAGE_ORDER.index(stage)
            self.progress_list.setCurrentRow(index)
            self.progress_bar.setValue(index)

    def on_preview_ready(self, preview_kind: str, payload: str) -> None:
        self._current_preview_kind = preview_kind
        self.media_player.stop()
        self.play_media_button.setEnabled(preview_kind in {"audio", "video"})
        self.open_file_button.setEnabled(preview_kind in {"audio", "video"})
        self.preview_text.hide()
        self.preview_image.hide()
        self.video_widget.hide()

        if preview_kind in {"text", "script", "visuals"}:
            self.preview_summary.setText(f"{preview_kind.title()} preview ready.")
            self.preview_text.setPlainText(payload)
            self.preview_text.show()
            if preview_kind == "script":
                self._update_veo_estimate_from_script_preview(payload)
        elif preview_kind == "audio":
            self.preview_summary.setText(f"Voiceover preview ready: {payload}")
            self.preview_text.setPlainText(f"Audio file ready:\n{payload}")
            self.preview_text.show()
            self.media_player.setSource(QUrl.fromLocalFile(payload))
        elif preview_kind == "video":
            self.preview_summary.setText(f"Final render preview ready: {payload}")
            self.video_widget.show()
            self.media_player.setSource(QUrl.fromLocalFile(payload))
        else:
            self.preview_summary.setText("Preview updated.")
            self.preview_text.setPlainText(payload)
            self.preview_text.show()

        if preview_kind == "visuals":
            try:
                plan = json.loads(payload)
                preview_path = ""
                if plan:
                    first = plan[0]
                    preview_path = first.get("poster_path") or first.get("asset_path", "")
                if preview_path and Path(preview_path).exists():
                    pixmap = QPixmap(preview_path).scaled(280, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.preview_image.setPixmap(pixmap)
                    self.preview_image.show()
            except Exception:
                pass

        if preview_kind in {"audio", "video"}:
            self._last_project_dir = str(Path(payload).parent)
            self.play_media_button.setText("Play Preview")

    def on_approval_required(self, stage_label: str) -> None:
        self.append_log(f"Waiting for user approval at: {stage_label}")
        self._set_buttons_for_running(waiting=True)

    def on_pipeline_finished(self, message: str) -> None:
        self.append_log(message)
        self._set_buttons_for_idle()

    def on_project_loaded(self, payload: dict) -> None:
        self._last_project_dir = payload.get("project_dir", "")
        if payload.get("source_mode") == "scene":
            self.workflow_mode_combo.setCurrentText("Scene Automation")
            self.url_input.setPlainText(payload.get("source_prompt", ""))
            self.story_type_edit.setText(payload.get("story_type", ""))
            self.character_brief_edit.setText(payload.get("character_brief", ""))
        else:
            self.workflow_mode_combo.setCurrentText("Scene Automation")
            self.url_input.setPlainText(payload.get("source_prompt", "") or "\n".join(payload.get("urls", [])))
        self.preview_summary.setText(f"Loaded project: {payload.get('project_name', '')}")
        self.preview_text.setPlainText(payload.get("source_summary", "Loaded project state."))
        self.preview_text.show()
        script_package = payload.get("script_package", {})
        if script_package:
            estimate = script_package.get("reuse_estimate", {})
            if estimate:
                self._set_veo_estimate_label(
                    reused=int(estimate.get("reused_clip_count", 0)),
                    new=int(estimate.get("new_clip_count", 0)),
                    raw_seconds=int(estimate.get("estimated_raw_seconds", 0)),
                )

    def append_log(self, message: str) -> None:
        self.logs.appendPlainText(message)

    def open_current_project_folder(self) -> None:
        if not self._last_project_dir:
            QMessageBox.information(self, "No Project", "No project folder is available yet.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_project_dir))

    def open_preview_file(self) -> None:
        media_url = self.media_player.source()
        if not media_url.isEmpty():
            QDesktopServices.openUrl(media_url)

    def toggle_media(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_media_button.setText("Play Preview")
        else:
            self.media_player.play()
            self.play_media_button.setText("Pause Preview")

    def _set_buttons_for_idle(self) -> None:
        self.start_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.regenerate_button.setEnabled(False)
        self.continue_button.setEnabled(False)

    def _set_buttons_for_running(self, waiting: bool) -> None:
        self.start_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.pause_button.setEnabled(not waiting)
        self.stop_button.setEnabled(True)
        self.regenerate_button.setEnabled(waiting)
        self.continue_button.setEnabled(waiting)

    def _is_valid_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _resolved_gemini_model(self) -> str:
        label = self.gemini_model_combo.currentText()
        if label == "Custom":
            custom = self.gemini_custom_model_edit.text().strip()
            resolved = custom or self.current_settings.gemini_model or GeminiService.DEFAULT_MODEL
            if resolved == "gemini-2.5-pro":
                return GeminiService.DEFAULT_MODEL
            return resolved
        return self.GEMINI_OPTIONS.get(label, GeminiService.DEFAULT_MODEL) or GeminiService.DEFAULT_MODEL

    def _validate_required_accounts_for_start(self) -> bool:
        missing = []
        if not self.gemini_key_edit.text().strip():
            missing.append("Gemini API key required")
        visual_provider = self.visual_provider_combo.currentText()
        narrator = self.narration_engine_combo.currentText()
        if visual_provider != "Veo + Stock":
            if narrator == "Disabled":
                missing.append("Narration engine cannot be Disabled")
            elif narrator == "ElevenLabs":
                if not self.elevenlabs_key_edit.text().strip():
                    missing.append("ElevenLabs API key required")
                if not self.voice_id_edit.text().strip():
                    missing.append("ElevenLabs Voice ID required")
            elif narrator == "Edge TTS":
                ready, message = self.edge_tts_service.check_available()
                if not ready:
                    missing.append(message)
        if visual_provider == "Veo + Stock" and not self.generated_video_folder_edit.text().strip():
            missing.append("Generated Video Library folder required")
        if visual_provider == "DeeVid Automation":
            if not self.deevid_profile_edit.text().strip():
                missing.append("DeeVid browser profile folder required")
            ready, message = self.deevid_service.check_available()
            if not ready:
                missing.append(message)
        reference_image = self.veo_reference_image_edit.text().strip()
        if reference_image and not Path(reference_image).exists():
            missing.append("Veo reference image file not found")
        if missing:
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Warning)
            dialog.setWindowTitle("Missing Connections")
            dialog.setText("\n".join(missing))
            dialog.setInformativeText("Open Accounts / Connections to finish setup.")
            dialog.setStandardButtons(QMessageBox.Ok)
            dialog.exec()
            self._set_settings_tab_from_nav("connect")
            return False
        return True

    def _refresh_ffmpeg_status(self) -> None:
        detected = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        self._connection_state["ffmpeg"] = "Detected" if detected else "Missing"
        self._refresh_connection_badges()

    def _refresh_connection_badges(self) -> None:
        gemini_text = "Gemini Connected" if self.gemini_key_edit.text().strip() else "Gemini Not Connected"
        if any(token in self._connection_state["gemini"] for token in ("Invalid", "Error")):
            gemini_text = f"Gemini {self._connection_state['gemini']}"
        self.gemini_status_badge.setText(gemini_text)

        narrator = self.narration_engine_combo.currentText()
        narration_state = self._connection_state["narration"]
        if narrator == "Edge TTS":
            narration_text = "Edge TTS Ready" if narration_state in {"Ready", "Connected"} else f"Edge TTS {narration_state}"
        elif narrator == "ElevenLabs":
            narration_text = "ElevenLabs Connected" if self.elevenlabs_key_edit.text().strip() and self.voice_id_edit.text().strip() and narration_state in {"Connected", "Ready"} else f"ElevenLabs {narration_state}"
        else:
            narration_text = "Narration Disabled"
        self.eleven_status_badge.setText(narration_text)

        visuals_configured = bool(self.pexels_key_edit.text().strip() or self.pixabay_key_edit.text().strip())
        visuals_text = "Visual APIs Connected" if visuals_configured else "Visual APIs Optional"
        if self._connection_state["visuals"] not in {"Optional", "Configured"}:
            visuals_text = f"Visual APIs {self._connection_state['visuals']}"
        self.visual_status_badge.setText(visuals_text)

        ffmpeg_text = "FFmpeg Detected" if self._connection_state["ffmpeg"] == "Detected" else "FFmpeg Missing"
        self.ffmpeg_status_badge.setText(ffmpeg_text)

    def _update_visual_status_from_stock_tests(self) -> None:
        statuses = [self.pexels_status_label.text(), self.pixabay_status_label.text()]
        if any("Connected" in item for item in statuses):
            self._connection_state["visuals"] = "Connected"
        elif any("Error" in item or "HTTP" in item or "missing expected fields" in item for item in statuses):
            self._connection_state["visuals"] = "Issue"
        elif self.pexels_key_edit.text().strip() or self.pixabay_key_edit.text().strip():
            self._connection_state["visuals"] = "Configured"
        else:
            self._connection_state["visuals"] = "Optional"
        self._refresh_connection_badges()

    def _normalize_connection_inputs(self) -> None:
        self.gemini_key_edit.setText(self.gemini_key_edit.text().strip())
        self.elevenlabs_key_edit.setText(self.elevenlabs_key_edit.text().strip())
        self.voice_id_edit.setText(self.voice_id_edit.text().strip())
        self.pexels_key_edit.setText(self.pexels_key_edit.text().strip())
        self.pixabay_key_edit.setText(self.pixabay_key_edit.text().strip())
        self.default_export_edit.setText(self.default_export_edit.text().strip())
        self.generated_video_folder_edit.setText(self.generated_video_folder_edit.text().strip())
        self.deevid_profile_edit.setText(self.deevid_profile_edit.text().strip())
        self.veo_reference_image_edit.setText(self.veo_reference_image_edit.text().strip())
        self.export_folder_edit.setText(self.export_folder_edit.text().strip())

    def _maybe_show_setup_wizard(self) -> None:
        if self.settings_store.has_required_accounts():
            return
        wizard = SetupWizard(self.current_settings, self)
        if wizard.exec():
            self.current_settings = wizard.apply_to_settings(self.current_settings)
            self._load_settings_into_ui(self.current_settings)
            self.save_all_settings()
            self._set_settings_tab_from_nav("connect")

    def _update_edge_slider_labels(self) -> None:
        self.edge_speed_value_label.setText(self._format_percent_value(self.edge_speed_slider.value()))
        self.edge_volume_value_label.setText(self._format_percent_value(self.edge_volume_slider.value()))

    def _format_percent_value(self, value: int) -> str:
        return f"{value:+d}%"

    def _update_narration_engine_ui(self) -> None:
        engine = self.narration_engine_combo.currentText()
        edge_enabled = engine == "Edge TTS"
        eleven_enabled = engine == "ElevenLabs"
        self.edge_voice_combo.setEnabled(edge_enabled)
        self.edge_speed_slider.setEnabled(edge_enabled)
        self.edge_volume_slider.setEnabled(edge_enabled)
        self.elevenlabs_key_edit.setEnabled(eleven_enabled)
        self.voice_id_edit.setEnabled(eleven_enabled)
        if engine == "Disabled":
            self._connection_state["narration"] = "Disabled"
        self._refresh_connection_badges()

    def _update_veo_estimate_from_script_preview(self, payload: str) -> None:
        reuse_match = None
        raw_match = None
        for line in payload.splitlines():
            if line.startswith("Clip Estimate:"):
                continue
            if line.startswith("Reuse "):
                reuse_match = line
        if reuse_match:
            try:
                parts = reuse_match.replace("Reuse ", "").split("|")
                reused = int(parts[0].strip())
                new = int(parts[1].replace("New ", "").strip())
                raw_seconds = int(parts[2].replace("Raw ", "").replace("s", "").strip())
                self._set_veo_estimate_label(reused, new, raw_seconds)
            except Exception:
                pass

    def _set_veo_estimate_label(self, reused: int, new: int, raw_seconds: int) -> None:
        self.veo_estimate_label.setText(
            f"Estimate: reuse {reused} clip(s), generate {new} new Veo clip(s), ~{raw_seconds}s raw video."
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.pipeline.request_stop()
        self.save_all_settings()
        self.pipeline_thread.quit()
        self.pipeline_thread.wait(2000)
        super().closeEvent(event)


def run() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

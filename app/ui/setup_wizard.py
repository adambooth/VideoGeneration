from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from app.config import AppSettings
from app.services.edge_tts_service import EdgeTTSService


class SetupWizard(QWizard):
    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Automated Video Creator Setup")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.resize(640, 420)

        self.gemini_page = _GeminiPage(settings)
        self.narration_page = _NarrationPage(settings)
        self.stock_page = _StockPage(settings)
        self.export_page = _ExportPage(settings)

        self.addPage(self.gemini_page)
        self.addPage(self.narration_page)
        self.addPage(self.stock_page)
        self.addPage(self.export_page)

    def apply_to_settings(self, settings: AppSettings) -> AppSettings:
        settings.gemini_api_key = self.gemini_page.api_key_edit.text().strip()
        settings.narration_engine = self.narration_page.engine_combo.currentText()
        settings.edge_tts_voice = self.narration_page.edge_voice_combo.currentText()
        settings.elevenlabs_api_key = self.narration_page.api_key_edit.text().strip()
        settings.elevenlabs_voice_id = self.narration_page.voice_id_edit.text().strip()
        settings.pexels_api_key = self.stock_page.pexels_key_edit.text().strip()
        settings.pixabay_api_key = self.stock_page.pixabay_key_edit.text().strip()
        settings.export_folder = self.export_page.export_folder_edit.text().strip() or settings.export_folder
        return settings


class _BasePage(QWizardPage):
    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        self.setTitle(title)
        self.setSubTitle(subtitle)
        self.layout = QVBoxLayout(self)


class _GeminiPage(_BasePage):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__("Connect Gemini", "Enter your Google AI Gemini API key to enable script generation.")
        form = QFormLayout()
        self.api_key_edit = QLineEdit(settings.gemini_api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.textChanged.connect(self.completeChanged)
        form.addRow("Gemini API Key", self.api_key_edit)
        self.layout.addLayout(form)
        self.layout.addWidget(QLabel("Gemini 2.5 Flash is used by default for low-cost script generation."))

    def isComplete(self) -> bool:  # type: ignore[override]
        return bool(self.api_key_edit.text().strip())


class _NarrationPage(_BasePage):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__("Choose Narration", "Edge TTS is free and enabled by default. ElevenLabs is optional.")
        form = QFormLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Edge TTS", "ElevenLabs", "Disabled"])
        self.engine_combo.setCurrentText(settings.narration_engine or "Edge TTS")
        self.edge_voice_combo = QComboBox()
        self.edge_voice_combo.addItems(EdgeTTSService.POPULAR_VOICES)
        self.edge_voice_combo.setCurrentText(settings.edge_tts_voice or "en-US-GuyNeural")
        self.api_key_edit = QLineEdit(settings.elevenlabs_api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.voice_id_edit = QLineEdit(settings.elevenlabs_voice_id)
        self.engine_combo.currentTextChanged.connect(self.completeChanged)
        self.api_key_edit.textChanged.connect(self.completeChanged)
        self.voice_id_edit.textChanged.connect(self.completeChanged)
        form.addRow("Narration Engine", self.engine_combo)
        form.addRow("Edge Voice", self.edge_voice_combo)
        form.addRow("ElevenLabs API Key", self.api_key_edit)
        form.addRow("ElevenLabs Voice ID", self.voice_id_edit)
        self.layout.addLayout(form)
        self.layout.addWidget(QLabel("You can switch to ElevenLabs later in Accounts / Connections if you want premium voices."))

    def isComplete(self) -> bool:  # type: ignore[override]
        if self.engine_combo.currentText() == "ElevenLabs":
            return bool(self.api_key_edit.text().strip() and self.voice_id_edit.text().strip())
        return True


class _StockPage(_BasePage):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__("Optional Visual APIs", "Add free stock footage providers now or skip and add them later.")
        form = QFormLayout()
        self.pexels_key_edit = QLineEdit(settings.pexels_api_key)
        self.pexels_key_edit.setEchoMode(QLineEdit.Password)
        self.pixabay_key_edit = QLineEdit(settings.pixabay_api_key)
        self.pixabay_key_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Pexels API Key", self.pexels_key_edit)
        form.addRow("Pixabay API Key", self.pixabay_key_edit)
        self.layout.addLayout(form)
        self.layout.addWidget(QLabel("These are optional. The app will fall back to local visuals if they are blank."))


class _ExportPage(_BasePage):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__("Choose Export Folder", "Pick the default location where projects and renders will be saved.")
        form = QFormLayout()
        self.export_folder_edit = QLineEdit(settings.export_folder)
        form.addRow("Export Folder", self.export_folder_edit)
        self.layout.addLayout(form)
        self.layout.addWidget(QLabel("You can change this later in Accounts / Connections."))

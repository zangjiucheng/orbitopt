"""Model and planner preferences for the Optimize workspace."""

from __future__ import annotations

import json
import math
import os
from urllib.parse import urlsplit

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from orbitopt import llm_env
from orbitopt.viz.optimize import backend
from orbitopt.viz.optimize.session import StudioSession
from orbitopt.viz.theme import DANGER, INK_MUTED, INK_PRIMARY, SUCCESS

_KEY = "optimize/connection"
_SOURCES = (
    ("Offline mock", "mock"),
    ("Environment (LLM_*)", "env"),
    ("Local server", "local"),
    ("OpenAI-compatible", "openai_compat"),
    ("DeepSeek", "deepseek"),
)


class OptimizationSettings(QWidget):
    """Edit a connection and apply it to subsequent Optimize jobs."""

    def __init__(self, session: StudioSession, settings):
        super().__init__()
        self.setObjectName("optimizeSettings")
        self.setAttribute(Qt.WA_StyledBackground)
        self.session = session
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.setContentsMargins(20, 26, 20, 24)
        card = QWidget()
        card.setObjectName("mathCard")
        card.setAttribute(Qt.WA_StyledBackground)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.setMaximumWidth(620)
        layout.addWidget(card)
        body = QVBoxLayout(card)
        body.setContentsMargins(20, 18, 20, 18)
        body.setSpacing(12)
        title = QLabel("Model connection")
        title.setStyleSheet(f"font-size: 17px; font-weight: 600; color: {INK_PRIMARY};")
        body.addWidget(title)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(9)
        body.addLayout(form)
        self.provider_combo = QComboBox()
        for label, key in _SOURCES:
            self.provider_combo.addItem(label, key)
        form.addRow("Provider", self.provider_combo)
        self.model = QLineEdit()
        self.endpoint = QLineEdit()
        self.timeout = QLineEdit()
        self.timeout.setPlaceholderText("Provider default")
        form.addRow("Model ID", self.model)
        form.addRow("API base URL", self.endpoint)
        form.addRow("Timeout (seconds)", self.timeout)
        self.planner_combo = QComboBox()
        self.planner_combo.addItem("Rule-based", "rule")
        self.planner_combo.addItem("LLM", "llm")
        self.planner_combo.setToolTip("Choose what selects the next solver after the problem is confirmed.")
        form.addRow("Solver planner", self.planner_combo)
        self.source_status = QLabel()
        self.source_status.setWordWrap(True)
        self.source_status.setTextFormat(Qt.PlainText)
        self.source_status.setStyleSheet(f"color: {INK_MUTED};")
        body.addWidget(self.source_status)
        self.apply_button = QPushButton("Apply settings")
        self.apply_button.setObjectName("primaryAction")
        body.addWidget(self.apply_button, alignment=Qt.AlignRight)
        self.result = QLabel()
        self.result.setWordWrap(True)
        body.addWidget(self.result)
        self.provider_combo.currentIndexChanged.connect(self._source_changed)
        self.apply_button.clicked.connect(lambda: self._apply())
        self._restore()

    def _source_changed(self) -> None:
        source = self.provider_combo.currentData()
        custom = source in llm_env.KINDS
        for field in (self.model, self.endpoint, self.timeout):
            field.setEnabled(custom)
        self.endpoint.setPlaceholderText({"local": "Default Ollama endpoint",
                                         "deepseek": "Default DeepSeek endpoint"}.get(source, "https://…/v1"))
        environ = os.environ if self.session.environ is None else self.session.environ
        status = "Bundled examples and reply files run offline."
        if source == "env":
            provider = next((c for c in self.session.choices if c.key == "env"), None)
            status = (f"Found {provider.model} in LLM_*." if provider
                      else "Set LLM_MODEL and any endpoint required by LLM_KIND.")
        elif source == "local":
            status = "Connects to Ollama or a compatible local server."
        elif source in llm_env.KEYED_KINDS:
            key_state = "available" if environ.get(llm_env.API_KEY) else "missing"
            status = f"The API key comes from LLM_API_KEY ({key_state})."
        self.source_status.setText(status)
        self.result.clear()

    def _data(self) -> dict[str, str]:
        source = self.provider_combo.currentData()
        custom = source in llm_env.KINDS
        return {
            "source": source,
            "model": self.model.text().strip() if custom else "",
            "endpoint": self.endpoint.text().strip() if custom else "",
            "timeout": self.timeout.text().strip() if custom else "",
            "planner": self.planner_combo.currentData(),
        }

    def _provider(self, data: dict[str, str]) -> backend.ProviderChoice:
        source = data["source"]
        if source == "mock":
            return backend.MOCK
        if source == "env":
            provider = next((c for c in backend.provider_choices(self.session.environ) if c.key == "env"), None)
            if provider is None:
                raise ValueError("The environment does not describe a provider. Check LLM_* and try again.")
            return provider
        if not data["model"]:
            raise ValueError("Enter a model ID.")
        endpoint = data["endpoint"]
        if endpoint:
            try:
                url = urlsplit(endpoint)
                if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                        or url.query or url.fragment or any(char.isspace() for char in endpoint)):
                    raise ValueError
                _ = url.port
            except ValueError:
                raise ValueError("Enter an HTTP(S) base URL without credentials, query parameters or fragments.") from None
        if source == "openai_compat" and not endpoint:
            raise ValueError("Enter the OpenAI-compatible API base URL.")
        timeout = None
        if data["timeout"]:
            try:
                timeout = float(data["timeout"])
            except ValueError:
                raise ValueError("Enter a positive timeout in seconds.") from None
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Enter a positive timeout in seconds.")
        return backend.ProviderChoice("custom", source, data["model"], endpoint or None,
                                      llm_env.API_KEY if source in llm_env.KEYED_KINDS else None, timeout)

    def _restore(self) -> None:
        try:
            saved = json.loads(self.settings.value(_KEY, "{}")) if self.settings is not None else {}
        except (TypeError, ValueError):
            saved = {}
        if not isinstance(saved, dict):
            saved = {}
        provider = self.session.provider
        default_source = provider.kind if provider.key == "custom" else provider.key
        self.provider_combo.setCurrentIndex(max(0, self.provider_combo.findData(saved.get("source", default_source))))
        self.planner_combo.setCurrentIndex(max(0, self.planner_combo.findData(saved.get("planner", self.session.planner))))
        for field, key in ((self.model, "model"), (self.endpoint, "endpoint"), (self.timeout, "timeout")):
            value = saved.get(key)
            field.setText(value if isinstance(value, str) else "")
        self._source_changed()
        if saved:
            self._apply(save=False)

    def _apply(self, *, save: bool = True) -> None:
        data = self._data()
        try:
            provider = self._provider(data)
            self.session.set_connection(provider, data["planner"])
        except (ValueError, RuntimeError) as exc:
            self.result.setStyleSheet(f"color: {DANGER};")
            self.result.setText(str(exc))
            return
        if save and self.settings is not None:
            self.settings.setValue(_KEY, json.dumps(data))
        self.result.setStyleSheet(f"color: {SUCCESS};")
        self.result.setText("Settings applied." if save else "")

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QListWidget, QListWidgetItem,
    QTextEdit, QMessageBox, QFileDialog, QDialog, QFormLayout,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox, QMenu
)

from llama_cpp import Llama  # llama-cpp-python

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


@dataclass
class AppSettings:
    model_path: str = ""
    temperature: float = 0.7
    max_tokens: int = 256
    n_ctx: int = 4096
    n_threads: int = max(1, (os.cpu_count() or 8) - 1)
    n_gpu_layers: int = 0  # 0=CPU-only; >0 tries GPU offload if supported
    verbose: bool = False


@dataclass
class ChatSession:
    title: str
    messages: List[Message]


class LlamaCppEngine:
    """
    Loads a GGUF model and provides chat completions.
    Reuses the loaded model between turns + chats.
    """
    def __init__(self):
        self._llm: Optional[Llama] = None
        self._signature: Optional[tuple] = None

    def is_loaded(self) -> bool:
        return self._llm is not None

    def unload(self):
        self._llm = None
        self._signature = None

    def load_if_needed(self, s: AppSettings):
        if not s.model_path or not os.path.exists(s.model_path):
            raise FileNotFoundError("GGUF model path is missing or invalid.")

        sig = (s.model_path, s.n_ctx, s.n_threads, s.n_gpu_layers, s.verbose)
        if self._llm is not None and self._signature == sig:
            return

        # (Re)load model
        self._llm = Llama(
            model_path=s.model_path,
            n_ctx=s.n_ctx,
            n_threads=s.n_threads,
            n_gpu_layers=s.n_gpu_layers,
            verbose=s.verbose,
        )
        self._signature = sig

    def chat(self, messages: List[Message], s: AppSettings) -> str:
        self.load_if_needed(s)

        # llama-cpp-python supports OpenAI-style chat completion API
        out = self._llm.create_chat_completion(
            messages=messages,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
        )
        return out["choices"][0]["message"]["content"]


class GenerateWorker(QThread):
    finished_text = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, engine: LlamaCppEngine, messages: List[Message], settings: AppSettings):
        super().__init__()
        self.engine = engine
        self.messages = messages
        self.settings = settings

    def run(self):
        try:
            text = self.engine.chat(self.messages, self.settings)
            self.finished_text.emit(text)
        except Exception as e:
            self.failed.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(self, parent, settings: AppSettings):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Model picker row
        self.model_path = QLineEdit(self.settings.model_path)
        self.btn_browse = QPushButton("Browse…")
        row = QHBoxLayout()
        row.addWidget(self.model_path)
        row.addWidget(self.btn_browse)
        row_wrap = QWidget()
        row_wrap.setLayout(row)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(self.settings.temperature)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(16, 4096)
        self.max_tokens.setValue(self.settings.max_tokens)

        self.n_ctx = QSpinBox()
        self.n_ctx.setRange(512, 32768)
        self.n_ctx.setValue(self.settings.n_ctx)

        self.n_threads = QSpinBox()
        self.n_threads.setRange(1, max(1, os.cpu_count() or 64))
        self.n_threads.setValue(self.settings.n_threads)

        self.n_gpu_layers = QSpinBox()
        self.n_gpu_layers.setRange(0, 200)
        self.n_gpu_layers.setValue(self.settings.n_gpu_layers)

        form.addRow("Model (GGUF):", row_wrap)
        form.addRow("Temperature:", self.temperature)
        form.addRow("Max tokens:", self.max_tokens)
        form.addRow("Context (n_ctx):", self.n_ctx)
        form.addRow("Threads:", self.n_threads)
        form.addRow("GPU layers:", self.n_gpu_layers)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)

        self.btn_browse.clicked.connect(self.on_browse)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF model", "", "GGUF (*.gguf);;All files (*.*)")
        if path:
            self.model_path.setText(path)

    def apply_to_settings(self) -> bool:
        mp = self.model_path.text().strip()
        if mp and not os.path.exists(mp):
            QMessageBox.warning(self, "Invalid model path", "Selected model file does not exist.")
            return False

        self.settings.model_path = mp
        self.settings.temperature = float(self.temperature.value())
        self.settings.max_tokens = int(self.max_tokens.value())
        self.settings.n_ctx = int(self.n_ctx.value())
        self.settings.n_threads = int(self.n_threads.value())
        self.settings.n_gpu_layers = int(self.n_gpu_layers.value())
        return True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Chat (PyQt) — Basic Test Build")

        self.settings = AppSettings()
        self.engine = LlamaCppEngine()
        self.worker: Optional[GenerateWorker] = None

        self.sessions: List[ChatSession] = []
        self.current_index: int = -1

        self._build_ui()
        self._build_menu()

        self.new_chat()  # create initial chat

    def _build_menu(self):
        menubar = self.menuBar()

        app_menu = menubar.addMenu("App")
        act_settings = app_menu.addAction("Settings…")
        act_unload = app_menu.addAction("Unload model")
        app_menu.addSeparator()
        act_quit = app_menu.addAction("Quit")

        act_settings.triggered.connect(self.open_settings)
        act_unload.triggered.connect(self.unload_model)
        act_quit.triggered.connect(self.close)

        help_menu = menubar.addMenu("Help")
        act_about = help_menu.addAction("About")
        act_about.triggered.connect(self.about)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # Left: chat sessions list
        left = QVBoxLayout()
        self.session_list = QListWidget()
        self.btn_new_chat = QPushButton("New chat")
        left.addWidget(self.btn_new_chat)
        left.addWidget(self.session_list)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(260)

        # Right: chat view + input
        right = QVBoxLayout()

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message…")
        self.btn_send = QPushButton("Send")
        input_row.addWidget(self.input)
        input_row.addWidget(self.btn_send)

        self.status = QLabel("Ready. (Set a GGUF model via App → Settings)")
        self.status.setStyleSheet("color: gray;")

        right.addWidget(self.chat_view)
        right.addLayout(input_row)
        right.addWidget(self.status)

        right_wrap = QWidget()
        right_wrap.setLayout(right)

        layout.addWidget(left_wrap)
        layout.addWidget(right_wrap)

        # signals
        self.btn_new_chat.clicked.connect(self.new_chat)
        self.session_list.currentRowChanged.connect(self.switch_chat)
        self.btn_send.clicked.connect(self.on_send)
        self.input.returnPressed.connect(self.on_send)

    def about(self):
        QMessageBox.information(
            self,
            "About",
            "Basic local chat UI using PyQt + llama-cpp-python.\n"
            "Next: streaming, persistence, RAG, browsing."
        )

    def set_busy(self, busy: bool):
        self.btn_send.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.btn_new_chat.setEnabled(not busy)
        self.session_list.setEnabled(not busy)
        self.status.setText("Generating…" if busy else "Ready.")

    def append_to_view(self, who: str, text: str):
        self.chat_view.append(f"<b>{who}:</b>")
        self.chat_view.append(text.replace("\n", "<br>"))
        self.chat_view.append("")  # spacing

    def new_chat(self):
        title = f"Chat {len(self.sessions) + 1}"
        sess = ChatSession(
            title=title,
            messages=[{"role": "system", "content": "You are a helpful assistant."}],
        )
        self.sessions.append(sess)
        self.session_list.addItem(title)
        self.session_list.setCurrentRow(len(self.sessions) - 1)

    def switch_chat(self, idx: int):
        if idx < 0 or idx >= len(self.sessions):
            return
        self.current_index = idx
        self.render_current_chat()

    def render_current_chat(self):
        self.chat_view.clear()
        sess = self.sessions[self.current_index]
        for m in sess.messages:
            if m["role"] == "system":
                continue
            who = "You" if m["role"] == "user" else "Assistant"
            self.append_to_view(who, m["content"])

    def open_settings(self):
        dlg = SettingsDialog(self, self.settings)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if dlg.apply_to_settings():
                # Model signature may have changed; next call reloads automatically
                self.status.setText("Settings saved. (Model will load on next message)")
            else:
                self.status.setText("Settings not applied.")

    def unload_model(self):
        self.engine.unload()
        self.status.setText("Model unloaded.")

    def on_send(self):
        if self.current_index < 0:
            return

        text = self.input.text().strip()
        if not text:
            return

        if not self.settings.model_path:
            QMessageBox.information(self, "Model required", "Set a GGUF model in App → Settings first.")
            return

        self.input.clear()
        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "user", "content": text})
        self.append_to_view("You", text)

        self.set_busy(True)
        self.worker = GenerateWorker(self.engine, list(sess.messages), self.settings)
        self.worker.finished_text.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, assistant_text: str):
        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": assistant_text})
        self.append_to_view("Assistant", assistant_text)

        # Auto-title chat from first user message (simple, non-fancy)
        if len([m for m in sess.messages if m["role"] == "user"]) == 1:
            first = next(m["content"] for m in sess.messages if m["role"] == "user")
            sess.title = (first[:28] + "…") if len(first) > 28 else first
            self.session_list.item(self.current_index).setText(sess.title)

        self.set_busy(False)
        self.worker = None

    def on_failed(self, err: str):
        QMessageBox.critical(self, "Generation failed", err)
        self.set_busy(False)
        self.worker = None


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

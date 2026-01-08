import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QListWidget, QTextEdit,
    QMessageBox, QFileDialog, QDialog, QFormLayout,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox, QCheckBox
)

from llama_cpp import Llama

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


# -----------------------------
# Settings / Session Models
# -----------------------------

@dataclass
class AppSettings:
    model_path: str = ""

    # Generation
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True

    # Runtime speed knobs (model-agnostic)
    n_ctx: int = 2048                 # smaller = faster + less RAM
    n_threads: int = max(1, (os.cpu_count() or 8) - 1)
    n_batch: int = 256                # 128/256/512; higher can help throughput
    n_gpu_layers: int = 0             # optional acceleration if supported
    use_mmap: bool = True             # mmap model file (often faster startup / lower RAM pressure)
    use_mlock: bool = False           # lock pages in RAM (can help stability/perf if enough RAM)
    flash_attn: bool = False          # keep off by default; enable only if stable on your build

    # Keep UI fast over time: cap history
    keep_last_messages: int = 16      # last N messages (user+assistant), system always kept


@dataclass
class ChatSession:
    title: str
    messages: List[Message]


# -----------------------------
# Engine (loads once, reused)
# -----------------------------

class LlamaCppEngine:
    def __init__(self):
        self._llm: Optional[Llama] = None
        self._signature: Optional[tuple] = None

    def unload(self):
        self._llm = None
        self._signature = None

    def is_loaded(self) -> bool:
        return self._llm is not None

    def load_if_needed(self, s: AppSettings):
        if not s.model_path or not os.path.exists(s.model_path):
            raise FileNotFoundError("GGUF model path is missing or invalid.")

        sig = (
            s.model_path, s.n_ctx, s.n_threads, s.n_batch,
            s.n_gpu_layers, s.use_mmap, s.use_mlock, s.flash_attn
        )
        if self._llm is not None and self._signature == sig:
            return

        # Always clean state before reload
        self.unload()

        # Build model instance
        self._llm = Llama(
            model_path=s.model_path,
            n_ctx=s.n_ctx,
            n_threads=s.n_threads,
            n_batch=s.n_batch,
            n_gpu_layers=s.n_gpu_layers,
            use_mmap=s.use_mmap,
            use_mlock=s.use_mlock,
            flash_attn=s.flash_attn,
            verbose=False,
        )
        self._signature = sig

    def _trim_history(self, messages: List[Message], keep_last: int) -> List[Message]:
        """Keep system + last N non-system messages to avoid slowdown over time."""
        if keep_last <= 0:
            return messages

        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return system + rest[-keep_last:]

    def chat_blocking(self, messages: List[Message], s: AppSettings) -> str:
        self.load_if_needed(s)
        msgs = self._trim_history(messages, s.keep_last_messages)
        out = self._llm.create_chat_completion(
            messages=msgs,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
        )
        return out["choices"][0]["message"]["content"]

    def chat_stream(self, messages: List[Message], s: AppSettings) -> Iterable[str]:
        """Yields text chunks."""
        self.load_if_needed(s)
        msgs = self._trim_history(messages, s.keep_last_messages)

        stream = self._llm.create_chat_completion(
            messages=msgs,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            stream=True,
        )

        # Stream returns incremental deltas
        for chunk in stream:
            # Safe extraction across llama-cpp-python variants
            try:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text
            except Exception:
                # If structure differs, ignore that chunk
                continue


# -----------------------------
# Workers
# -----------------------------

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
            text = self.engine.chat_blocking(self.messages, self.settings)
            self.finished_text.emit(text)
        except Exception as e:
            self.failed.emit(str(e))


class StreamWorker(QThread):
    chunk = pyqtSignal(str)
    finished_full = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, engine: LlamaCppEngine, messages: List[Message], settings: AppSettings):
        super().__init__()
        self.engine = engine
        self.messages = messages
        self.settings = settings
        self._accum: List[str] = []

    def run(self):
        try:
            for t in self.engine.chat_stream(self.messages, self.settings):
                self._accum.append(t)
                self.chunk.emit(t)
            self.finished_full.emit("".join(self._accum))
        except Exception as e:
            self.failed.emit(str(e))


# -----------------------------
# Settings UI
# -----------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent, settings: AppSettings):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Model
        self.model_path = QLineEdit(self.settings.model_path)
        self.btn_browse = QPushButton("Browse…")
        row = QHBoxLayout()
        row.addWidget(self.model_path)
        row.addWidget(self.btn_browse)
        row_wrap = QWidget()
        row_wrap.setLayout(row)

        # Generation
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(self.settings.temperature)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(16, 4096)
        self.max_tokens.setValue(self.settings.max_tokens)

        self.stream = QCheckBox("Stream responses")
        self.stream.setChecked(self.settings.stream)

        # Runtime knobs
        self.n_ctx = QSpinBox()
        self.n_ctx.setRange(512, 32768)
        self.n_ctx.setValue(self.settings.n_ctx)

        self.n_threads = QSpinBox()
        self.n_threads.setRange(1, max(1, os.cpu_count() or 64))
        self.n_threads.setValue(self.settings.n_threads)

        self.n_batch = QSpinBox()
        self.n_batch.setRange(16, 2048)
        self.n_batch.setValue(self.settings.n_batch)

        self.n_gpu_layers = QSpinBox()
        self.n_gpu_layers.setRange(0, 200)
        self.n_gpu_layers.setValue(self.settings.n_gpu_layers)

        self.use_mmap = QCheckBox("use_mmap (recommended)")
        self.use_mmap.setChecked(self.settings.use_mmap)

        self.use_mlock = QCheckBox("use_mlock (only if enough RAM)")
        self.use_mlock.setChecked(self.settings.use_mlock)

        self.flash_attn = QCheckBox("flash_attn (advanced)")
        self.flash_attn.setChecked(self.settings.flash_attn)

        self.keep_last = QSpinBox()
        self.keep_last.setRange(4, 200)
        self.keep_last.setValue(self.settings.keep_last_messages)

        form.addRow("Model (GGUF):", row_wrap)
        form.addRow("Temperature:", self.temperature)
        form.addRow("Max tokens:", self.max_tokens)
        form.addRow("", self.stream)
        form.addRow("Context (n_ctx):", self.n_ctx)
        form.addRow("Threads:", self.n_threads)
        form.addRow("Batch (n_batch):", self.n_batch)
        form.addRow("GPU layers:", self.n_gpu_layers)
        form.addRow("", self.use_mmap)
        form.addRow("", self.use_mlock)
        form.addRow("", self.flash_attn)
        form.addRow("Keep last messages:", self.keep_last)

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

    def apply(self) -> bool:
        mp = self.model_path.text().strip()
        if mp and not os.path.exists(mp):
            QMessageBox.warning(self, "Invalid model path", "Selected model file does not exist.")
            return False

        self.settings.model_path = mp
        self.settings.temperature = float(self.temperature.value())
        self.settings.max_tokens = int(self.max_tokens.value())
        self.settings.stream = bool(self.stream.isChecked())

        self.settings.n_ctx = int(self.n_ctx.value())
        self.settings.n_threads = int(self.n_threads.value())
        self.settings.n_batch = int(self.n_batch.value())
        self.settings.n_gpu_layers = int(self.n_gpu_layers.value())

        self.settings.use_mmap = bool(self.use_mmap.isChecked())
        self.settings.use_mlock = bool(self.use_mlock.isChecked())
        self.settings.flash_attn = bool(self.flash_attn.isChecked())

        self.settings.keep_last_messages = int(self.keep_last.value())
        return True


# -----------------------------
# Main UI
# -----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Koila — Local Chat (PyQt + llama.cpp)")

        self.settings = AppSettings()
        self.engine = LlamaCppEngine()

        self.sessions: List[ChatSession] = []
        self.current_index: int = -1

        # workers
        self.worker_block: Optional[GenerateWorker] = None
        self.worker_stream: Optional[StreamWorker] = None

        self._build_ui()
        self._build_menu()
        self.new_chat()

    def _build_menu(self):
        menubar = self.menuBar()
        app_menu = menubar.addMenu("App")

        act_settings = app_menu.addAction("Settings…")
        act_warm = app_menu.addAction("Warm-load model")
        act_unload = app_menu.addAction("Unload model")
        app_menu.addSeparator()
        act_quit = app_menu.addAction("Quit")

        act_settings.triggered.connect(self.open_settings)
        act_warm.triggered.connect(self.warm_load_model)
        act_unload.triggered.connect(self.unload_model)
        act_quit.triggered.connect(self.close)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # left: sessions
        left = QVBoxLayout()
        self.btn_new_chat = QPushButton("New chat")
        self.session_list = QListWidget()
        left.addWidget(self.btn_new_chat)
        left.addWidget(self.session_list)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(260)

        # right: chat
        right = QVBoxLayout()
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message…")
        self.btn_send = QPushButton("Send")
        input_row.addWidget(self.input)
        input_row.addWidget(self.btn_send)

        self.status = QLabel("Ready. (Set model in App → Settings)")
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

    def set_busy(self, busy: bool):
        self.btn_send.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.btn_new_chat.setEnabled(not busy)
        self.session_list.setEnabled(not busy)
        self.status.setText("Generating…" if busy else "Ready.")

    def append_block(self, who: str, text: str):
        self.chat_view.append(f"<b>{who}:</b>")
        self.chat_view.append(text.replace("\n", "<br>"))
        self.chat_view.append("")

    def begin_stream_assistant(self):
        self.chat_view.append("<b>Assistant: </b>")
        self._stream_cursor = self.chat_view.textCursor()
        self._stream_cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_view.setTextCursor(self._stream_cursor)

    def append_stream_chunk(self, chunk: str):
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertPlainText(chunk)
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def end_stream_assistant(self):
        self.chat_view.append("")
        self.chat_view.append("")

    def new_chat(self):
        title = f"Chat {len(self.sessions) + 1}"
        system_prompt = (
            "You are a helpful assistant. Provide high-quality answers. Be clear, detailed, and concise. "
            "Avoid using unwanted characters. Your name is Koila, if asked. "
            "Do not mention your thought process unless asked. Respond only in English unless requested otherwise."
            "CRITICAL RULES:"
            "- If you're not certain about something, say \"I'm not sure\" or \"I don't have reliable information about that\""
            "- Never make up facts, dates, or statistics"
            "- When unsure, acknowledge uncertainty rather than guessing"
            "- If asked about recent events after your knowledge cutoff, clearly state your knowledge cutoff"

Be honest about limitations."
        )
        sess = ChatSession(title=title, messages=[{"role": "system", "content": system_prompt}])
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
            self.append_block(who, m["content"])

    def open_settings(self):
        dlg = SettingsDialog(self, self.settings)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if dlg.apply():
                self.engine.unload()
                self.status.setText("Settings saved. (Use Warm-load for first-run speed)")
            else:
                self.status.setText("Settings not applied.")

    def warm_load_model(self):
        if not self.settings.model_path:
            QMessageBox.information(self, "Model required", "Set a GGUF model in App → Settings first.")
            return
        try:
            t0 = time.time()
            self.engine.load_if_needed(self.settings)
            dt = time.time() - t0
            self.status.setText(f"Model loaded in {dt:.1f}s")
        except Exception as e:
            self.engine.unload()
            QMessageBox.critical(self, "Load failed", str(e))

    def unload_model(self):
        self.engine.unload()
        self.status.setText("Model unloaded.")

    def _autotitle_if_first_user(self, sess: ChatSession):
        if len([m for m in sess.messages if m["role"] == "user"]) == 1:
            first = next(m["content"] for m in sess.messages if m["role"] == "user")
            sess.title = (first[:28] + "…") if len(first) > 28 else first
            self.session_list.item(self.current_index).setText(sess.title)

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
        self.append_block("You", text)

        self.set_busy(True)

        if self.settings.stream:
            self.begin_stream_assistant()
            self.worker_stream = StreamWorker(self.engine, list(sess.messages), self.settings)
            self.worker_stream.chunk.connect(self.append_stream_chunk)
            self.worker_stream.finished_full.connect(self.on_stream_done)
            self.worker_stream.failed.connect(self.on_failed)
            self.worker_stream.start()
        else:
            self.worker_block = GenerateWorker(self.engine, list(sess.messages), self.settings)
            self.worker_block.finished_text.connect(self.on_done)
            self.worker_block.failed.connect(self.on_failed)
            self.worker_block.start()

    def on_stream_done(self, full_text: str):
        self.end_stream_assistant()

        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": full_text})
        self._autotitle_if_first_user(sess)

        self.set_busy(False)
        self.worker_stream = None

    def on_done(self, assistant_text: str):
        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": assistant_text})
        self.append_block("Assistant", assistant_text)
        self._autotitle_if_first_user(sess)

        self.set_busy(False)
        self.worker_block = None

    def on_failed(self, err: str):
        self.engine.unload()
        QMessageBox.critical(self, "Generation failed", err)
        self.set_busy(False)
        self.worker_block = None
        self.worker_stream = None


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

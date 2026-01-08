import os
import sys
import time
import json
import re
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Tuple

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
    n_ctx: int = 2048
    n_threads: int = max(1, (os.cpu_count() or 8) - 1)
    n_batch: int = 256
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    flash_attn: bool = False

    # Keep UI fast over time
    keep_last_messages: int = 16

    # Browsing (model-agnostic hallucination reducer)
    browse_enabled_default: bool = False
    browse_num_sources: int = 3
    browse_timeout_sec: int = 12
    browse_max_chars_per_source: int = 1600  # keep evidence small for speed
    browse_cache_enabled: bool = True


@dataclass
class ChatSession:
    title: str
    messages: List[Message]


# -----------------------------
# Simple web search + fetch (DuckDuckGo Lite/HTML)
# -----------------------------

class WebCache:
    """Tiny SQLite cache for fetched pages."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _init(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache (url TEXT PRIMARY KEY, ts INTEGER, content TEXT)"
            )

    def get(self, url: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("SELECT content FROM cache WHERE url=?", (url,)).fetchone()
            return row[0] if row else None

    def put(self, url: str, content: str):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO cache(url, ts, content) VALUES(?, strftime('%s','now'), ?)",
                (url, content),
            )


class Browser:
    """
    Best-effort, no-API-key browsing:
    - search with DuckDuckGo Lite (fast HTML)
    - fallback to DDG HTML
    - fetch pages and extract rough text
    """
    def __init__(self, timeout_sec: int = 12, cache: Optional[WebCache] = None):
        self.timeout_sec = timeout_sec
        self.cache = cache
        self.ua = "Mozilla/5.0 (KoilaLocalChat/1.0; +https://example.local)"

    def _http_get(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = resp.read()
        # best-effort decode
        return data.decode("utf-8", errors="ignore")

    def search(self, query: str, k: int) -> List[str]:
        q = urllib.parse.quote_plus(query)

        # Try Lite first (simpler/faster)
        lite_url = f"https://lite.duckduckgo.com/lite/?q={q}"
        try:
            html = self._http_get(lite_url)
            urls = self._parse_ddg_lite_results(html, k)
            if urls:
                return urls
        except Exception:
            pass

        # Fallback to HTML endpoint
        html_url = f"https://html.duckduckgo.com/html/?q={q}"
        html = self._http_get(html_url)
        urls = self._parse_ddg_html_results(html, k)
        return urls

    def _parse_ddg_lite_results(self, html: str, k: int) -> List[str]:
        # Lite uses simple <a rel="nofollow" class="result-link" href="...">
        urls: List[str] = []
        for m in re.finditer(r'href="(https?://[^"]+)"', html):
            u = m.group(1)
            # skip ddg internal/nav
            if "duckduckgo.com" in urllib.parse.urlparse(u).netloc:
                continue
            urls.append(u)
            if len(urls) >= k:
                break
        return self._dedupe(urls)

    def _parse_ddg_html_results(self, html: str, k: int) -> List[str]:
        # HTML endpoint often uses redirect links with "uddg=" param
        urls: List[str] = []
        for m in re.finditer(r'href="([^"]+)"', html):
            href = m.group(1)
            if "uddg=" in href:
                try:
                    parsed = urllib.parse.urlparse(href)
                    qs = urllib.parse.parse_qs(parsed.query)
                    if "uddg" in qs:
                        u = urllib.parse.unquote(qs["uddg"][0])
                        if u.startswith("http"):
                            urls.append(u)
                except Exception:
                    continue
            elif href.startswith("http"):
                if "duckduckgo.com" in urllib.parse.urlparse(href).netloc:
                    continue
                urls.append(href)

            if len(urls) >= k:
                break
        return self._dedupe(urls)

    def _dedupe(self, urls: List[str]) -> List[str]:
        out, seen = [], set()
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

    def fetch_text(self, url: str, max_chars: int) -> str:
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                return cached[:max_chars]

        html = self._http_get(url)
        text = self._html_to_text(html)
        text = re.sub(r"\s+", " ", text).strip()

        if self.cache:
            self.cache.put(url, text)
        return text[:max_chars]

    def _html_to_text(self, html: str) -> str:
        # remove scripts/styles
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        # remove tags
        html = re.sub(r"(?s)<.*?>", " ", html)
        # decode some entities
        html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return html


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

    def load_if_needed(self, s: AppSettings):
        if not s.model_path or not os.path.exists(s.model_path):
            raise FileNotFoundError("GGUF model path is missing or invalid.")

        sig = (
            s.model_path, s.n_ctx, s.n_threads, s.n_batch,
            s.n_gpu_layers, s.use_mmap, s.use_mlock, s.flash_attn
        )
        if self._llm is not None and self._signature == sig:
            return

        self.unload()
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
        if keep_last <= 0:
            return messages
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return system + rest[-keep_last:]

    def chat_stream(self, messages: List[Message], s: AppSettings) -> Iterable[str]:
        self.load_if_needed(s)
        msgs = self._trim_history(messages, s.keep_last_messages)

        stream = self._llm.create_chat_completion(
            messages=msgs,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text
            except Exception:
                continue


# -----------------------------
# Workers
# -----------------------------

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


class BrowseAndStreamWorker(QThread):
    status = pyqtSignal(str)
    chunk = pyqtSignal(str)
    finished_full = pyqtSignal(str, str)  # (assistant_text, sources_text)
    failed = pyqtSignal(str)

    def __init__(self, engine: LlamaCppEngine, base_messages: List[Message], user_query: str, settings: AppSettings, browser: Browser):
        super().__init__()
        self.engine = engine
        self.base_messages = base_messages
        self.user_query = user_query
        self.settings = settings
        self.browser = browser
        self._accum: List[str] = []

    def run(self):
        try:
            self.status.emit("Browsing…")
            urls = self.browser.search(self.user_query, self.settings.browse_num_sources)

            snippets: List[Tuple[str, str]] = []
            for i, url in enumerate(urls, start=1):
                try:
                    txt = self.browser.fetch_text(url, self.settings.browse_max_chars_per_source)
                    if txt:
                        snippets.append((url, txt))
                except Exception:
                    continue

            if not snippets:
                # fallback: still answer, but tell model no sources found
                evidence = "No web sources could be fetched."
                sources_text = ""
            else:
                evidence_lines = ["Web evidence (use for answering; cite as [1], [2], ...):"]
                sources_lines = []
                for idx, (url, txt) in enumerate(snippets, start=1):
                    evidence_lines.append(f"[{idx}] {url}\n{txt}\n")
                    sources_lines.append(f"[{idx}] {url}")
                evidence = "\n".join(evidence_lines).strip()
                sources_text = "\n".join(sources_lines).strip()

            browse_rules = (
                "BROWSING RULES:\n"
                "- Use the provided web evidence when answering.\n"
                "- If the evidence does not contain the answer, say you couldn't confirm from sources.\n"
                "- Do not invent facts.\n"
                "- Add citations like [1] [2] next to the relevant sentences.\n"
            )

            augmented = list(self.base_messages)
            augmented.append({"role": "system", "content": browse_rules})
            augmented.append({"role": "system", "content": evidence})
            augmented.append({"role": "user", "content": self.user_query})

            self.status.emit("Answering…")
            for t in self.engine.chat_stream(augmented, self.settings):
                self._accum.append(t)
                self.chunk.emit(t)

            self.finished_full.emit("".join(self._accum), sources_text)
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

        # Browsing
        self.browse_default = QCheckBox("Enable browsing by default")
        self.browse_default.setChecked(self.settings.browse_enabled_default)

        self.browse_sources = QSpinBox()
        self.browse_sources.setRange(1, 8)
        self.browse_sources.setValue(self.settings.browse_num_sources)

        self.browse_timeout = QSpinBox()
        self.browse_timeout.setRange(3, 60)
        self.browse_timeout.setValue(self.settings.browse_timeout_sec)

        self.browse_max_chars = QSpinBox()
        self.browse_max_chars.setRange(200, 6000)
        self.browse_max_chars.setValue(self.settings.browse_max_chars_per_source)

        self.browse_cache = QCheckBox("Cache fetched pages (recommended)")
        self.browse_cache.setChecked(self.settings.browse_cache_enabled)

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

        form.addRow("Browsing:", QLabel("<b>Web browsing</b>"))
        form.addRow("", self.browse_default)
        form.addRow("Sources:", self.browse_sources)
        form.addRow("Timeout (sec):", self.browse_timeout)
        form.addRow("Max chars/source:", self.browse_max_chars)
        form.addRow("", self.browse_cache)

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

        self.settings.browse_enabled_default = bool(self.browse_default.isChecked())
        self.settings.browse_num_sources = int(self.browse_sources.value())
        self.settings.browse_timeout_sec = int(self.browse_timeout.value())
        self.settings.browse_max_chars_per_source = int(self.browse_max_chars.value())
        self.settings.browse_cache_enabled = bool(self.browse_cache.isChecked())
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

        cache = WebCache(db_path=os.path.join(os.path.expanduser("~"), ".koila", "webcache.sqlite"))
        self.browser = Browser(timeout_sec=self.settings.browse_timeout_sec, cache=cache)

        self.sessions: List[ChatSession] = []
        self.current_index: int = -1

        self.worker_stream: Optional[StreamWorker] = None
        self.worker_browse: Optional[BrowseAndStreamWorker] = None

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

        self.chk_browse = QCheckBox("Browse")
        self.chk_browse.setChecked(self.settings.browse_enabled_default)

        self.btn_send = QPushButton("Send")

        input_row.addWidget(self.input)
        input_row.addWidget(self.chk_browse)
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
        self.chk_browse.setEnabled(not busy)
        if not busy:
            self.status.setText("Ready.")

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
            "Do not mention your thought process unless asked. Respond only in English unless requested otherwise. "
            "CRITICAL RULES: If you're not certain, say you are not sure. Never make up facts."
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
                # update browser timeout/cache preference
                cache = WebCache(db_path=os.path.join(os.path.expanduser("~"), ".koila", "webcache.sqlite")) if self.settings.browse_cache_enabled else None
                self.browser = Browser(timeout_sec=self.settings.browse_timeout_sec, cache=cache)
                self.chk_browse.setChecked(self.settings.browse_enabled_default)
                self.status.setText("Settings saved.")
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

        user_text = self.input.text().strip()
        if not user_text:
            return

        if not self.settings.model_path:
            QMessageBox.information(self, "Model required", "Set a GGUF model in App → Settings first.")
            return

        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "user", "content": user_text})
        self.input.clear()
        self.append_block("You", user_text)

        self.set_busy(True)
        self.begin_stream_assistant()

        browse_this = self.chk_browse.isChecked()
        if browse_this:
            self.status.setText("Browsing…")
            # base messages exclude the just-added user turn (we add it after evidence)
            base = [m for m in sess.messages if not (m["role"] == "user" and m["content"] == user_text)]
            self.worker_browse = BrowseAndStreamWorker(self.engine, base, user_text, self.settings, self.browser)
            self.worker_browse.status.connect(self.status.setText)
            self.worker_browse.chunk.connect(self.append_stream_chunk)
            self.worker_browse.finished_full.connect(self.on_browse_done)
            self.worker_browse.failed.connect(self.on_failed)
            self.worker_browse.start()
        else:
            self.status.setText("Answering…")
            self.worker_stream = StreamWorker(self.engine, list(sess.messages), self.settings)
            self.worker_stream.chunk.connect(self.append_stream_chunk)
            self.worker_stream.finished_full.connect(self.on_stream_done)
            self.worker_stream.failed.connect(self.on_failed)
            self.worker_stream.start()

    def on_browse_done(self, full_text: str, sources_text: str):
        self.end_stream_assistant()
        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": full_text})
        if sources_text:
            self.append_block("Sources", sources_text)
        self._autotitle_if_first_user(sess)
        self.set_busy(False)
        self.worker_browse = None

    def on_stream_done(self, full_text: str):
        self.end_stream_assistant()
        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": full_text})
        self._autotitle_if_first_user(sess)
        self.set_busy(False)
        self.worker_stream = None

    def on_failed(self, err: str):
        self.engine.unload()
        QMessageBox.critical(self, "Generation failed", err)
        self.set_busy(False)
        self.worker_stream = None
        self.worker_browse = None


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 760)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

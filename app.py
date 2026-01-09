import os
import sys
import time
import re
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QMessageBox, QFileDialog,
    QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox,
    QCheckBox, QPlainTextEdit, QLineEdit, QTextBrowser, QComboBox
)

from llama_cpp import Llama

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


@dataclass
class AppSettings:
    # Model selection
    model_path: str = ""
    models_dir: str = ""  # folder containing .gguf files (optional)

    # Generation
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True

    # Runtime knobs (speed)
    n_ctx: int = 2048
    n_threads: int = max(1, (os.cpu_count() or 8) - 1)
    n_batch: int = 256
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    flash_attn: bool = False

    # Prevent "slower over time"
    keep_last_messages: int = 16

    # Browsing
    browse_enabled_default: bool = False
    browse_num_sources: int = 3
    browse_timeout_sec: int = 12
    browse_max_chars_per_source: int = 1600
    browse_cache_enabled: bool = True
    browse_min_text_len: int = 500


@dataclass
class ChatSession:
    title: str
    messages: List[Message]


def _escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def render_text_as_html(text: str) -> str:
    t = _escape_html(text)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    return t.replace("\n", "<br>")


def linkify_url(url: str) -> str:
    u = _escape_html(url)
    return f'<a href="{u}">{u}</a>'


class WebCache:
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
    def __init__(self, timeout_sec: int, max_chars_per_source: int, min_text_len: int, cache: Optional[WebCache]):
        self.timeout_sec = timeout_sec
        self.max_chars_per_source = max_chars_per_source
        self.min_text_len = min_text_len
        self.cache = cache
        self.ua = "Mozilla/5.0 (KoilaLocalChat/1.0)"
        self.deny_substrings = [
            "grammar", "spell", "checker", "editor",
            "translate", "converter", "calculator",
        ]

    def _http_get(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = resp.read()
        return data.decode("utf-8", errors="ignore")

    def search(self, query: str, k: int) -> List[str]:
        q = urllib.parse.quote_plus(query)

        lite_url = f"https://lite.duckduckgo.com/lite/?q={q}"
        try:
            html = self._http_get(lite_url)
            urls = self._parse_ddg_lite_results(html, k)
            urls = self._filter_urls(urls)
            if urls:
                return urls
        except Exception:
            pass

        html_url = f"https://html.duckduckgo.com/html/?q={q}"
        html = self._http_get(html_url)
        urls = self._parse_ddg_html_results(html, k)
        urls = self._filter_urls(urls)
        return urls

    def _parse_ddg_lite_results(self, html: str, k: int) -> List[str]:
        urls: List[str] = []
        for m in re.finditer(r'href="(https?://[^"]+)"', html):
            u = m.group(1)
            if "duckduckgo.com" in urllib.parse.urlparse(u).netloc:
                continue
            urls.append(u)
            if len(urls) >= k * 3:
                break
        return self._dedupe(urls)[:k]

    def _parse_ddg_html_results(self, html: str, k: int) -> List[str]:
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
            if len(urls) >= k * 3:
                break
        return self._dedupe(urls)[:k]

    def _dedupe(self, urls: List[str]) -> List[str]:
        out, seen = [], set()
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

    def _filter_urls(self, urls: List[str]) -> List[str]:
        good: List[str] = []
        for u in urls:
            host = urllib.parse.urlparse(u).netloc.lower()
            if any(bad in host for bad in self.deny_substrings):
                continue
            good.append(u)
        return good

    def fetch_text(self, url: str) -> str:
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                return cached[: self.max_chars_per_source]

        html = self._http_get(url)
        text = self._html_to_text(html)
        text = re.sub(r"\s+", " ", text).strip()

        host = urllib.parse.urlparse(url).netloc.lower()
        if any(bad in host for bad in self.deny_substrings):
            return ""
        if len(text) < self.min_text_len:
            return ""

        if self.cache:
            self.cache.put(url, text)
        return text[: self.max_chars_per_source]

    def _html_to_text(self, html: str) -> str:
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        html = re.sub(r"(?s)<.*?>", " ", html)
        html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return html


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
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if keep_last > 0:
            rest = rest[-keep_last:]
        return system + rest

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


class StreamWorker(QThread):
    chunk = pyqtSignal(str)
    finished_full = pyqtSignal(str)  # assistant_text
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
    finished_full = pyqtSignal(str, list)  # assistant_text, urls_used
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
            for url in urls:
                try:
                    txt = self.browser.fetch_text(url)
                    if txt:
                        snippets.append((url, txt))
                except Exception:
                    continue

            if not snippets:
                self.finished_full.emit(
                    "I couldn’t verify this from web sources (no usable sources were fetched). Please try again or refine the query.",
                    []
                )
                return

            evidence_lines = ["Web evidence (use for answering; cite as [1], [2], ...):"]
            urls_used: List[str] = []
            for idx, (url, txt) in enumerate(snippets, start=1):
                evidence_lines.append(f"[{idx}] {url}\n{txt}\n")
                urls_used.append(url)
            evidence = "\n".join(evidence_lines).strip()

            browse_rules = (
                "BROWSING RULES:\n"
                "- You DO have web evidence below. Do NOT say you can't browse or lack real-time access.\n"
                "- Use the provided web evidence when answering.\n"
                "- If the evidence does not contain the answer, say you couldn't confirm from sources.\n"
                "- Do not invent facts.\n"
                "- Add citations like [1] [2] next to relevant sentences.\n"
            )

            augmented = list(self.base_messages)
            augmented.append({"role": "system", "content": browse_rules})
            augmented.append({"role": "system", "content": evidence})
            augmented.append({"role": "user", "content": self.user_query})

            self.status.emit(f"Answering… (sources: {len(urls_used)})")
            for t in self.engine.chat_stream(augmented, self.settings):
                self._accum.append(t)
                self.chunk.emit(t)

            self.finished_full.emit("".join(self._accum), urls_used)
        except Exception as e:
            self.failed.emit(str(e))


class AppSettingsDialog(QDialog):
    def __init__(self, parent, settings: AppSettings):
        super().__init__(parent)
        self.setWindowTitle("Settings — App & Model")
        self.settings = settings

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.models_dir = QLineEdit(self.settings.models_dir)
        self.btn_models_dir = QPushButton("Browse…")
        row_dir = QHBoxLayout()
        row_dir.addWidget(self.models_dir)
        row_dir.addWidget(self.btn_models_dir)
        row_dir_wrap = QWidget()
        row_dir_wrap.setLayout(row_dir)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(self.settings.temperature)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(16, 4096)
        self.max_tokens.setValue(self.settings.max_tokens)

        self.stream = QCheckBox("Stream responses")
        self.stream.setChecked(self.settings.stream)

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

        form.addRow("Models folder:", row_dir_wrap)
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

        self.btn_models_dir.clicked.connect(self.on_pick_models_dir)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def on_pick_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select models folder", self.models_dir.text().strip() or "")
        if path:
            self.models_dir.setText(path)

    def apply(self) -> bool:
        md = self.models_dir.text().strip()
        if md and not os.path.isdir(md):
            QMessageBox.warning(self, "Invalid folder", "Models folder does not exist.")
            return False

        self.settings.models_dir = md
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


class BrowsingSettingsDialog(QDialog):
    def __init__(self, parent, settings: AppSettings):
        super().__init__(parent)
        self.setWindowTitle("Settings — Browsing")
        self.settings = settings

        layout = QVBoxLayout(self)
        form = QFormLayout()

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

        self.browse_min_len = QSpinBox()
        self.browse_min_len.setRange(100, 5000)
        self.browse_min_len.setValue(self.settings.browse_min_text_len)

        self.browse_cache = QCheckBox("Cache fetched pages (recommended)")
        self.browse_cache.setChecked(self.settings.browse_cache_enabled)

        form.addRow("", self.browse_default)
        form.addRow("Sources:", self.browse_sources)
        form.addRow("Timeout (sec):", self.browse_timeout)
        form.addRow("Max chars/source:", self.browse_max_chars)
        form.addRow("Min text length:", self.browse_min_len)
        form.addRow("", self.browse_cache)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def apply(self) -> bool:
        self.settings.browse_enabled_default = bool(self.browse_default.isChecked())
        self.settings.browse_num_sources = int(self.browse_sources.value())
        self.settings.browse_timeout_sec = int(self.browse_timeout.value())
        self.settings.browse_max_chars_per_source = int(self.browse_max_chars.value())
        self.settings.browse_min_text_len = int(self.browse_min_len.value())
        self.settings.browse_cache_enabled = bool(self.browse_cache.isChecked())
        return True


class ChatInput(QPlainTextEdit):
    sendRequested = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(e)
            else:
                self.sendRequested.emit()
        else:
            super().keyPressEvent(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Koila — Local Chat (PyQt + llama.cpp)")

        self.settings = AppSettings()
        self.engine = LlamaCppEngine()

        self._rebuild_browser()

        self.sessions: List[ChatSession] = []
        self.current_index: int = -1

        self.worker_stream: Optional[StreamWorker] = None
        self.worker_browse: Optional[BrowseAndStreamWorker] = None

        self._stream_start_pos: Optional[int] = None

        self._build_ui()
        self._build_menu()
        self.new_chat()

    def _rebuild_browser(self):
        cache = WebCache(db_path=os.path.join(os.path.expanduser("~"), ".koila", "webcache.sqlite")) if self.settings.browse_cache_enabled else None
        self.browser = Browser(
            timeout_sec=self.settings.browse_timeout_sec,
            max_chars_per_source=self.settings.browse_max_chars_per_source,
            min_text_len=self.settings.browse_min_text_len,
            cache=cache
        )

    def _build_menu(self):
        menubar = self.menuBar()
        app_menu = menubar.addMenu("App")

        act_settings_app = app_menu.addAction("Settings — App & Model…")
        act_settings_browse = app_menu.addAction("Settings — Browsing…")
        app_menu.addSeparator()
        act_warm = app_menu.addAction("Warm-load model")
        act_unload = app_menu.addAction("Unload model")
        app_menu.addSeparator()
        act_quit = app_menu.addAction("Quit")

        act_settings_app.triggered.connect(self.open_app_settings)
        act_settings_browse.triggered.connect(self.open_browsing_settings)
        act_warm.triggered.connect(self.warm_load_model)
        act_unload.triggered.connect(self.unload_model)
        act_quit.triggered.connect(self.close)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self.btn_new_chat = QPushButton("New chat")
        self.session_list = QListWidget()
        left.addWidget(self.btn_new_chat)
        left.addWidget(self.session_list)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(260)

        right = QVBoxLayout()

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )

        input_row = QHBoxLayout()

        self.input = ChatInput()
        self.input.setPlaceholderText("Type a message… (Enter to send, Shift+Enter for new line)")
        self.input.setFixedHeight(90)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(260)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)

        self.chk_browse = QCheckBox("Browse")
        self.chk_browse.setChecked(self.settings.browse_enabled_default)

        self.btn_send = QPushButton("Send")

        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.model_combo)
        input_row.addWidget(self.chk_browse)
        input_row.addWidget(self.btn_send)

        self.status = QLabel("Ready. (Select a model from dropdown, or set Models folder in settings)")
        self.status.setStyleSheet("color: gray;")

        right.addWidget(self.chat_view)
        right.addLayout(input_row)
        right.addWidget(self.status)

        right_wrap = QWidget()
        right_wrap.setLayout(right)

        layout.addWidget(left_wrap)
        layout.addWidget(right_wrap)

        self.btn_new_chat.clicked.connect(self.new_chat)
        self.session_list.currentRowChanged.connect(self.switch_chat)
        self.btn_send.clicked.connect(self.on_send)
        self.input.sendRequested.connect(self.on_send)

        self.refresh_model_list()

    def refresh_model_list(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Select model…", "")

        md = self.settings.models_dir.strip()
        if md and os.path.isdir(md):
            for fn in sorted(os.listdir(md)):
                if fn.lower().endswith(".gguf"):
                    full = os.path.join(md, fn)
                    self.model_combo.addItem(fn, full)

        if self.settings.model_path and os.path.isfile(self.settings.model_path):
            existing = [self.model_combo.itemData(i) for i in range(self.model_combo.count())]
            if self.settings.model_path not in existing:
                self.model_combo.addItem(os.path.basename(self.settings.model_path), self.settings.model_path)

        if self.settings.model_path:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == self.settings.model_path:
                    self.model_combo.setCurrentIndex(i)
                    break

        self.model_combo.blockSignals(False)

    def on_model_selected(self, idx: int):
        path = self.model_combo.itemData(idx)
        if not path:
            return
        if path != self.settings.model_path:
            self.settings.model_path = path
            self.engine.unload()
            self.status.setText(f"Selected model: {os.path.basename(path)}")

    def set_busy(self, busy: bool, status: str = ""):
        self.btn_send.setEnabled(not busy)
        self.btn_new_chat.setEnabled(not busy)
        self.session_list.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.chk_browse.setEnabled(not busy)
        self.input.setEnabled(True)  # allow typing while busy

        if status:
            self.status.setText(status)
        elif not busy:
            self.status.setText("Ready.")

    def append_message(self, who: str, content: str):
        html = f"<b>{_escape_html(who)}:</b><br>{render_text_as_html(content)}<br><br>"
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertHtml(html)
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def begin_stream_assistant(self):
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertHtml("<b>Assistant:</b><br>")
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self._stream_start_pos = self.chat_view.textCursor().position()

    def append_stream_chunk(self, chunk: str):
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertPlainText(chunk)
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def end_stream_assistant(self, full_text: Optional[str] = None):
        if full_text is not None and self._stream_start_pos is not None:
            cur = self.chat_view.textCursor()
            cur.setPosition(self._stream_start_pos)
            cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            cur.removeSelectedText()
            cur.insertHtml(render_text_as_html(full_text))
            self.chat_view.setTextCursor(cur)
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

        self.chat_view.insertHtml("<br><br>")
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self._stream_start_pos = None

    def append_sources(self, urls: List[str]):
        if not urls:
            return
        lines = ["<b>Sources:</b><br>"]
        for i, u in enumerate(urls, start=1):
            lines.append(f"[{i}] {linkify_url(u)}<br>")
        lines.append("<br>")
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertHtml("".join(lines))
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _current_session_has_user_messages(self) -> bool:
        if self.current_index < 0:
            return False
        sess = self.sessions[self.current_index]
        return any(m["role"] == "user" for m in sess.messages)

    def new_chat(self):
        if self.current_index >= 0 and not self._current_session_has_user_messages():
            QMessageBox.information(self, "New chat", "Use the current chat first (send at least one message) before creating another.")
            return

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
            self.append_message(who, m["content"])

    def open_app_settings(self):
        dlg = AppSettingsDialog(self, self.settings)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if dlg.apply():
                self.engine.unload()
                self.refresh_model_list()
                self.set_busy(False, "Settings saved.")
            else:
                self.set_busy(False, "Settings not applied.")

    def open_browsing_settings(self):
        dlg = BrowsingSettingsDialog(self, self.settings)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if dlg.apply():
                self._rebuild_browser()
                self.chk_browse.setChecked(self.settings.browse_enabled_default)
                self.set_busy(False, "Browsing settings saved.")
            else:
                self.set_busy(False, "Browsing settings not applied.")

    def warm_load_model(self):
        if not self.settings.model_path:
            QMessageBox.information(self, "Model required", "Select a GGUF model from the dropdown first.")
            return
        try:
            self.set_busy(True, "Loading model…")
            t0 = time.time()
            self.engine.load_if_needed(self.settings)
            dt = time.time() - t0
            self.set_busy(False, f"Model loaded in {dt:.1f}s")
        except Exception as e:
            self.engine.unload()
            self.set_busy(False)
            QMessageBox.critical(self, "Load failed", str(e))

    def unload_model(self):
        self.engine.unload()
        self.set_busy(False, "Model unloaded.")

    def _autotitle_if_first_user(self, sess: ChatSession):
        if len([m for m in sess.messages if m["role"] == "user"]) == 1:
            first = next(m["content"] for m in sess.messages if m["role"] == "user")
            sess.title = (first[:28] + "…") if len(first) > 28 else first
            self.session_list.item(self.current_index).setText(sess.title)

    def _filtered_history_for_browsing(self, base: List[Message]) -> List[Message]:
        out: List[Message] = []
        for m in base:
            if m["role"] != "assistant":
                out.append(m)
                continue
            c = (m.get("content") or "").lower()
            if ("real-time" in c or "real time" in c or "can't browse" in c or "cannot browse" in c
                    or "no internet access" in c or "don't have access to the internet" in c):
                continue
            out.append(m)
        return out

    def on_send(self):
        if self.current_index < 0:
            return

        user_text = self.input.toPlainText().strip()
        if not user_text:
            return

        if not self.settings.model_path or not os.path.isfile(self.settings.model_path):
            QMessageBox.information(self, "Model required", "Select a valid GGUF model from the dropdown first.")
            return

        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "user", "content": user_text})
        self.input.setPlainText("")
        self.append_message("You", user_text)

        self.set_busy(True, "Answering…")
        self.begin_stream_assistant()

        browse_this = self.chk_browse.isChecked()
        if browse_this:
            base = [m for m in sess.messages if not (m["role"] == "user" and m["content"] == user_text)]
            base = self._filtered_history_for_browsing(base)

            self.worker_browse = BrowseAndStreamWorker(self.engine, base, user_text, self.settings, self.browser)
            self.worker_browse.status.connect(lambda s: self.set_busy(True, s))
            self.worker_browse.chunk.connect(self.append_stream_chunk)
            self.worker_browse.finished_full.connect(self.on_browse_done)
            self.worker_browse.failed.connect(self.on_failed)
            self.worker_browse.start()
        else:
            self.worker_stream = StreamWorker(self.engine, list(sess.messages), self.settings)
            self.worker_stream.chunk.connect(self.append_stream_chunk)
            self.worker_stream.finished_full.connect(self.on_stream_done)
            self.worker_stream.failed.connect(self.on_failed)
            self.worker_stream.start()

    def on_browse_done(self, full_text: str, urls_used: List[str]):
        self.end_stream_assistant(full_text)

        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": full_text})
        self._autotitle_if_first_user(sess)

        if urls_used:
            self.append_sources(urls_used)

        self.set_busy(False)
        self.worker_browse = None

    def on_stream_done(self, full_text: str):
        self.end_stream_assistant(full_text)

        sess = self.sessions[self.current_index]
        sess.messages.append({"role": "assistant", "content": full_text})
        self._autotitle_if_first_user(sess)

        self.set_busy(False)
        self.worker_stream = None

    def on_failed(self, err: str):
        self.engine.unload()
        self.end_stream_assistant("")
        self.set_busy(False)
        QMessageBox.critical(self, "Generation failed", err)
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

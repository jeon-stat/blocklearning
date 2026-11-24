"""DL studio 로컬 HTTP 서버: 127.0.0.1에만 열고, 화면(static)과 JSON API(core.Studio)를 제공한다."""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from dlstudio.core import Studio, UserError, _clean, _jsonable, _path  # noqa: F401  (UserError는 예전 import 자리)
from dlstudio.data import DataError

STATIC_DIR = Path(__file__).parent / "static"


class StudioServer(Studio, ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, static_dir: Path):
        ThreadingHTTPServer.__init__(self, ("127.0.0.1", port), Handler)
        Studio.__init__(self)
        self.static_dir = Path(static_dir).resolve()


class Handler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, fmt, *args):            # 터미널을 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(_clean(obj), ensure_ascii=False, allow_nan=False, default=_jsonable)
        self._send(code, body.encode(), "application/json; charset=utf-8")

    def _same_origin(self) -> bool:
        """다른 웹사이트가 로컬 API를 부르지 못하게 (CSRF·DNS rebinding): Host와 Origin이 이 서버여야 함."""
        port = self.server.server_address[1]
        ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in ok:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in ok}

    def _error(self, code: int, message: str, hint: str = "", detail: str = "", issues=None) -> None:
        self._json(code, {"error": {"message": message, "hint": hint, "detail": detail, "issues": issues or []}})

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if not self._same_origin():
            return self._error(403, "다른 사이트에서 온 요청은 받지 않음")
        if path.startswith("/api/"):
            if path == "/api/label/file":
                return self._label_file(parse_qs(urlparse(self.path).query).get("name", [""])[0])
            code, obj = self.server.call("GET", path)
            return self._json(code, obj)
        if path in ("/", "/index.html"):
            return self._file("index.html")
        if path in ("/label", "/label.html"):
            return self._file("label.html")
        if path.startswith("/static/"):
            return self._file(path[len("/static/"):])
        self._error(404, "없는 주소")

    def _label_file(self, name: str) -> None:
        """라벨링 그림 원본 (크게 보기). 작업의 샘플 이름만, 데이터 폴더 안의 그림 파일만."""
        from dlstudio.data import IMAGE_EXTS
        from dlstudio.labeling import _inside
        srv = self.server
        with srv.label_lock:
            p = srv.label
            try:
                f = p.file_path(name) if p is not None else None
            except DataError:
                f = None
            root = p.source.guard_root() if p is not None else ""
        if f is None or f.suffix.lower() not in IMAGE_EXTS or not _inside(str(f), root) or not f.is_file():
            return self._error(404, "없는 그림")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        self._send(200, f.read_bytes(), ctype)

    def _file(self, rel: str) -> None:
        root = self.server.static_dir
        p = (root / rel).resolve()
        if root not in p.parents or not p.is_file():          # static 폴더 밖(..)은 거절
            return self._error(404, "없는 파일")
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if not self._same_origin():
            return self._error(403, "다른 사이트에서 온 요청은 받지 않음")
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._error(415, "JSON 요청만 받음")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            return self._error(400, "요청을 읽을 수 없음 (JSON 아님)")
        code, obj = self.server.call("POST", path, body)
        self._json(code, obj)


def make_server(port: int = 0, static_dir: Optional[str] = None) -> StudioServer:
    return StudioServer(port, Path(static_dir) if static_dir else STATIC_DIR)

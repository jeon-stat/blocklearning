"""BlockLearning 로컬 HTTP 서버: 127.0.0.1에만 열고, 화면(static)과 JSON API(core.Studio)를 제공한다.

접속 열쇠(token): 실행할 때마다 새로 만들어, 열쇠가 든 주소(…/?token=…)로 처음 연 브라우저만 들어온다 (Jupyter와 같은 방식).
같은 컴퓨터를 쓰는 다른 사람(공용 GPU 서버)이나 다른 웹사이트가 이 서버에 들어오지 못하게 한다.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import re
import secrets
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from blocklearning.core import Studio, UserError, _clean, _jsonable, _path  # noqa: F401  (UserError는 예전 import 자리)
from blocklearning.data import DataError
from blocklearning.myprojects import PROJECTS_DIR, Library

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path.home() / ".blocklearning" / "uploads"      # 브라우저로 고른 파일을 두는 곳 (서버에서 실행하면 서버 쪽)
MAX_UPLOAD = 4 * 1024 ** 3                                     # 파일 하나 4GB까지
TOKEN_HEADER = "X-BlockLearning-Token"
HOST_RE = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\]):\d{1,5}$")   # SSH·VS Code로 이어 온 포트 번호는 달라도 됨
DENIED_PAGE = """<!doctype html><meta charset="utf-8"><title>BlockLearning</title>
<body style="font-family:system-ui,sans-serif;max-width:560px;margin:60px auto;padding:0 16px;line-height:1.6">
<h2>접속 열쇠가 필요합니다</h2>
<p>BlockLearning을 실행한 명령창에 나온 주소(<code>…/?token=…</code>)로 여세요.</p></body>"""


class StudioServer(Studio, ThreadingHTTPServer):
    daemon_threads = True
    EXTRA_ROUTES = {"/api/projects": "api_projects"}

    def __init__(self, port: int, static_dir: Path, token: str, upload_dir: Path = UPLOAD_DIR,
                 projects_dir: Path = PROJECTS_DIR):
        ThreadingHTTPServer.__init__(self, ("127.0.0.1", port), Handler)
        Studio.__init__(self)
        self.static_dir = Path(static_dir).resolve()
        self.token = token
        self.cookie = "bl_" + hashlib.sha256(token.encode()).hexdigest()[:12]   # 서버마다 다른 이름 (여러 개를 띄워도 섞이지 않게)
        self.upload_dir = Path(upload_dir)
        self.library = Library(self, Path(projects_dir))

    def check_token(self, value: Optional[str]) -> bool:
        return bool(value) and hmac.compare_digest(str(value).encode(), self.token.encode())

    def call(self, method: str, path: str, body=None) -> tuple:
        """첫 화면 '내 프로젝트'(/api/projects/…)는 Library, 나머지는 Studio."""
        if method == "POST" and path.startswith("/api/projects/"):
            op = path[len("/api/projects/"):]
            return Studio.call(self, method, "/api/projects", dict(body or {}, _op=op))
        return Studio.call(self, method, path, body)

    def api_projects(self, b: dict) -> dict:
        b = dict(b)
        return self.library.call(b.pop("_op", ""), b)


class Handler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, fmt, *args):            # 터미널을 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str, headers: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")       # 열쇠 든 주소가 다른 곳으로 새지 않게
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(_clean(obj), ensure_ascii=False, allow_nan=False, default=_jsonable)
        self._send(code, body.encode(), "application/json; charset=utf-8")

    def _same_origin(self) -> bool:
        """다른 웹사이트가 로컬 API를 부르지 못하게 (CSRF·DNS rebinding): Host는 이 컴퓨터, Origin은 같은 출처."""
        host = self.headers.get("Host", "")
        if not HOST_RE.match(host):
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{host}"

    def _authed(self) -> bool:
        if self.server.check_token(self.headers.get(TOKEN_HEADER)):
            return True
        jar = SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except Exception:
            return False
        c = jar.get(self.server.cookie)
        return c is not None and self.server.check_token(c.value)

    def _error(self, code: int, message: str, hint: str = "", detail: str = "", issues=None) -> None:
        self._json(code, {"error": {"message": message, "hint": hint, "detail": detail, "issues": issues or []}})

    def _deny(self, path: str) -> None:
        if path.startswith("/api/"):
            return self._error(403, "접속 열쇠가 없음", "BlockLearning을 실행한 명령창에 나온 주소(…/?token=…)로 다시 여세요")
        self._send(403, DENIED_PAGE.encode(), "text/html; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        if not self._same_origin():
            return self._error(403, "다른 사이트에서 온 요청은 받지 않음")
        token = parse_qs(url.query).get("token", [""])[0]
        if token:                                                # 열쇠 든 주소로 처음 열 때: 쿠키로 기억하고 열쇠를 주소에서 지움
            if not self.server.check_token(token):
                return self._deny(path)
            cookie = f"{self.server.cookie}={self.server.token}; Path=/; HttpOnly; SameSite=Lax"
            return self._send(302, b"", "text/plain", {"Location": path or "/", "Set-Cookie": cookie})
        if not self._authed():
            return self._deny(path)
        if path.startswith("/api/"):
            if path == "/api/label/file":
                return self._label_file(parse_qs(url.query).get("name", [""])[0])
            if path == "/api/download":
                return self._download(parse_qs(url.query).get("path", [""])[0])
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
        from blocklearning.data import IMAGE_EXTS
        from blocklearning.labeling import _inside
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

    def _download(self, path: str) -> None:
        """저장·내보내기로 쓴 파일을 브라우저로 (서버에서 실행할 때 내 PC로 받기). 그 밖의 파일은 주지 않음."""
        try:
            p = str(Path(path).resolve()) if path else ""
        except (OSError, ValueError):
            p = ""
        if p not in self.server.outputs or not Path(p).is_file():
            return self._error(404, "없는 파일")
        name = Path(p).name
        plain = name if re.fullmatch(r"[A-Za-z0-9._ -]+", name) else "download" + re.sub(r"[^A-Za-z0-9.]", "", Path(name).suffix)
        disp = f"attachment; filename=\"{plain}\"; filename*=UTF-8''{quote(name, safe='')}"
        self._send(200, Path(p).read_bytes(), "application/octet-stream", {"Content-Disposition": disp})

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
        url = urlparse(self.path)
        path = unquote(url.path)
        if not self._same_origin():
            return self._error(403, "다른 사이트에서 온 요청은 받지 않음")
        if not self._authed():
            return self._deny(path)
        ctype = self.headers.get("Content-Type") or ""
        if path == "/api/upload":
            if not ctype.startswith("application/octet-stream"):
                return self._error(415, "파일 내용만 받음")
            return self._upload(parse_qs(url.query))
        if not ctype.startswith("application/json"):
            return self._error(415, "JSON 요청만 받음")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            return self._error(400, "요청을 읽을 수 없음 (JSON 아님)")
        code, obj = self.server.call("POST", path, body)
        self._json(code, obj)

    def _upload(self, q: dict) -> None:
        """브라우저에서 고른 파일 하나를 이 컴퓨터의 올린 파일 폴더(~/.blocklearning/uploads/<묶음>/<상대 경로>)에 저장."""
        batch, rel = q.get("batch", [""])[0], q.get("rel", [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", batch):
            return self._error(400, "올리기 묶음 이름이 맞지 않음")
        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
        if not parts or len(parts) > 32 or any(p == ".." or ":" in p or len(p) > 255 for p in parts):
            return self._error(400, "파일 이름이 맞지 않음")
        try:
            n = int(self.headers.get("Content-Length") or -1)
        except ValueError:
            n = -1
        if n < 0:
            return self._error(411, "파일 크기를 알 수 없음")
        if n > MAX_UPLOAD:
            return self._error(413, "파일이 너무 큼 (하나에 4GB까지)", "큰 데이터는 서버에 직접 두고 경로를 넣으세요")
        root = (self.server.upload_dir / batch).resolve()
        dest = root.joinpath(*parts).resolve()
        if root not in dest.parents:
            return self._error(400, "파일 이름이 맞지 않음")
        dest.parent.mkdir(parents=True, exist_ok=True)
        left = n
        with open(dest, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            dest.unlink(missing_ok=True)
            return self._error(400, "파일을 끝까지 받지 못함")
        self._json(200, {"path": str(dest), "top": str(root / parts[0])})


def make_server(port: int = 0, static_dir: Optional[str] = None, token: Optional[str] = None,
                upload_dir: Optional[str] = None, projects_dir: Optional[str] = None) -> StudioServer:
    return StudioServer(port, Path(static_dir) if static_dir else STATIC_DIR, token or secrets.token_urlsafe(24),
                        Path(upload_dir) if upload_dir else UPLOAD_DIR, Path(projects_dir) if projects_dir else PROJECTS_DIR)

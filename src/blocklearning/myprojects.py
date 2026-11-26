"""설치형 '내 프로젝트': 이 컴퓨터의 프로젝트 폴더(~/BlockLearning/projects)에 자동 저장하고 첫 화면에 목록으로.

프로젝트 하나 = <id>.blocklearning (프로젝트 파일, 데이터는 경로로) + <id>.json (이름·한 줄 요약·진행·저장 시각).
웹 버전의 저장소(store.js의 local)와 같은 모양으로 화면(projects.js)에 준다.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path

from blocklearning.core import UserError

PROJECTS_DIR = Path.home() / "BlockLearning" / "projects"
ID_RE = re.compile(r"p[a-z0-9]{6,40}")


class Library:
    def __init__(self, studio, root: Path = PROJECTS_DIR):
        self.studio, self.root = studio, Path(root)
        self.current = None                          # 지금 화면에서 연 프로젝트 id (새로 고쳐도 이어서 저장)

    def _paths(self, pid) -> tuple:
        if not isinstance(pid, str) or not ID_RE.fullmatch(pid):
            raise UserError("프로젝트 번호가 맞지 않음", "첫 화면 목록에서 다시 고르세요")
        return self.root / f"{pid}.json", self.root / f"{pid}.blocklearning"

    def _info(self, pid) -> dict:
        meta, _ = self._paths(pid)
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise UserError("프로젝트가 없음", "첫 화면 목록을 새로 고치세요") from None

    def _write(self, pid, info: dict) -> None:
        meta, _ = self._paths(pid)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, meta)

    def call(self, op: str, b: dict) -> dict:
        fn = {"list": self.list, "create": self.create, "save": self.save, "open": self.open,
              "rename": self.rename, "remove": self.remove, "close": self.close}.get(op)
        if fn is None:
            raise KeyError(op)
        return fn(b)

    def list(self, b: dict) -> dict:
        items = []
        for meta in self.root.glob("p*.json"):
            pid = meta.stem
            if not ID_RE.fullmatch(pid):
                continue
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            items.append({"id": pid, "name": str(info.get("name") or pid), "meta": info.get("meta") or "",
                          "steps": info.get("steps") or "", "updated": info.get("updated") or 0,
                          "file": str(self._paths(pid)[1].resolve())})
        items.sort(key=lambda p: -p["updated"])
        current = self.current if self.studio.data is not None and any(p["id"] == self.current for p in items) else None
        return {"items": items, "dir": str(self.root.resolve()), "current": current}

    def create(self, b: dict) -> dict:
        name = str(b.get("name") or "").strip() or "제목 없는 프로젝트"
        pid = "p" + format(int(time.time() * 1000), "x") + secrets.token_hex(3)
        info = {"name": name[:200], "updated": int(time.time() * 1000)}
        self._write(pid, info)
        return dict(info, id=pid)

    def save(self, b: dict) -> dict:
        pid = b.get("id")
        info = self._info(pid)
        r = self.studio.api_project_save({"path": str(self._paths(pid)[1])})
        for k in ("meta", "steps"):
            if isinstance(b.get(k), str):
                info[k] = b[k]
        info["updated"] = int(time.time() * 1000)
        self._write(pid, info)
        self.current = pid
        return {"ok": True, "files": r["files"]}

    def open(self, b: dict) -> dict:
        pid = b.get("id")
        self._info(pid)
        path = self._paths(pid)[1]
        if not path.is_file():                       # 만들었지만 아직 저장한 적 없음
            return {"empty": True}
        r = self.studio.api_project_open({"path": str(path)})
        self.studio.wrote(path)                       # 서버에서 실행할 때 [내려받기]로 받을 수 있게
        self.current = pid
        return r

    def rename(self, b: dict) -> dict:
        pid, name = b.get("id"), str(b.get("name") or "").strip()
        if not name:
            raise UserError("이름이 비어 있음", "새 이름을 넣으세요")
        info = self._info(pid)
        info["name"], info["updated"] = name[:200], int(time.time() * 1000)
        self._write(pid, info)
        return {"ok": True}

    def remove(self, b: dict) -> dict:
        for p in self._paths(b.get("id")):
            p.unlink(missing_ok=True)
        if self.current == b.get("id"):
            self.current = None
        return {"ok": True}

    def close(self, b: dict) -> dict:
        """새 프로젝트: 지금 작업(데이터·모델·학습 결과)을 비움. 저장은 화면이 먼저 함."""
        self.studio.reset_work()
        self.current = None
        return {"ok": True}

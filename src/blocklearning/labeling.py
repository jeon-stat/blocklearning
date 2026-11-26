"""라벨링 작업실: 시계열 폴더·이미지 폴더·표(CSV)에 정답(클래스 또는 숫자)을 붙인다.

작업 파일(.dllabel, JSON)에 정답·출처·확인 여부·메모를 두고, 샘플 목록과 값은 데이터 폴더에서 다시 읽는다.
데이터 파일은 절대 바꾸지 않는다. 화면·HTTP는 모른다 (server.py가 감쌈).
"""
from __future__ import annotations

import copy
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from blocklearning.data import DataError, Issue, _label_table, _num, _read_table
from blocklearning.labeling_sources import downsample, make_source

FORMAT, VERSION = "dlstudio-label", 1
COLORS = ["#1d9e75", "#c0392b", "#2f6fde", "#d85a30", "#7f77dd", "#9a6700", "#0f766e", "#be185d", "#4b5563"]
STATS = ("length", "min", "max", "mean", "std", "first", "last", "range", "argmax_frac")
MAX_CLASSES = 9            # 단축키 1–9
SCHEMAS = ("class", "number", "segments", "boxes")
REGION = ("segments", "boxes")   # 샘플 하나에 여러 정답 (구간·상자): 샘플 정답·규칙·정답 표는 안 씀
PURPOSES = ("general", "anomaly")
NORMAL = "정상"
UNDO_STEPS = 50
PAGE = 60
THUMB = 120
DETAIL_MAX = 2000          # 크게 보기: 이보다 길면 구간별 최소·최대로 줄임
BY = ("manual", "rule", "import", "model")
CLUSTER_K = (2, 20)        # 묶음 수 범위
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.isclose}


def _err(message: str, hint: str = "") -> DataError:
    return DataError([Issue(None, None, message, hint)])


def _has(it: dict) -> bool:
    """정답이 있는지 (숫자 정답 0.0도 정답, 구간 작업은 구간이 하나라도 있으면)."""
    lab = it.get("label")
    return lab is not None and lab != "" or bool(it.get("segments")) or bool(it.get("boxes"))


def _regions(it: dict) -> list:
    """구간·상자 정답의 클래스 목록 (마지막 칸이 클래스)."""
    return [x[-1] for x in (it.get("segments") or it.get("boxes") or [])]


def _finite(v) -> bool:
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(float(v)))
    except (OverflowError, TypeError, ValueError):
        return False


MAX_POLY = 200            # 다각형 꼭짓점 수 상한


def _in01(v) -> bool:
    return _finite(v) and -1e-6 <= float(v) <= 1 + 1e-6


def _clean_shape(x, classes: set):
    """도형 하나 (그림 크기에 대한 0–1 비율) → 정리한 값 또는 None.
    상자 [x, y, w, h, 클래스] · 다각형 ["poly", [[x, y], …], 클래스] (꼭짓점 3개 이상) · 점 ["pt", x, y, 클래스]
    · 원 ["circle", 가운데 x, 가운데 y, 반지름(그림 너비에 대한 비율), 클래스]."""
    if not isinstance(x, (list, tuple)) or not x or not isinstance(x[-1], str) or x[-1] not in classes:
        return None
    k = x[-1]
    if x[0] == "poly":
        pts = x[1] if len(x) == 3 and isinstance(x[1], (list, tuple)) else None
        if not pts or not 3 <= len(pts) <= MAX_POLY or not all(isinstance(p, (list, tuple)) and len(p) == 2 and _in01(p[0]) and _in01(p[1]) for p in pts):
            return None
        return ["poly", [[round(min(1.0, max(0.0, float(a))), 5), round(min(1.0, max(0.0, float(b))), 5)] for a, b in pts], k]
    if x[0] == "pt":
        if len(x) != 4 or not (_in01(x[1]) and _in01(x[2])):
            return None
        return ["pt", round(min(1.0, max(0.0, float(x[1]))), 5), round(min(1.0, max(0.0, float(x[2]))), 5), k]
    if x[0] == "circle":
        if len(x) != 5 or not (_in01(x[1]) and _in01(x[2]) and _finite(x[3]) and 0 < float(x[3]) <= 1):
            return None
        return ["circle", round(min(1.0, max(0.0, float(x[1]))), 5), round(min(1.0, max(0.0, float(x[2]))), 5), round(float(x[3]), 5), k]
    if len(x) != 5 or not all(_finite(t) for t in x[:4]):
        return None
    a, b, c, d = (float(t) for t in x[:4])
    if 0 <= a and 0 <= b and c > 0 and d > 0 and a + c <= 1 + 1e-6 and b + d <= 1 + 1e-6:
        return [round(a, 5), round(b, 5), round(c, 5), round(d, 5), k]
    return None


def _shape_kind(x) -> str:
    return x[0] if x[0] in ("poly", "pt", "circle") else "box"


def _circle_poly(cx: float, cy: float, r: float, W: float, H: float, n: int = 32) -> list:
    """원 (반지름은 그림 너비 비율) → 꼭짓점 n개 다각형 (0–1 비율, 그림 안으로 자름). 그림이 정사각형이 아니어도 진짜 원."""
    ry = r * W / max(H, 1e-9)
    return [[round(min(1.0, max(0.0, cx + r * np.cos(t))), 6), round(min(1.0, max(0.0, cy + ry * np.sin(t))), 6)]
            for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def _circle_box(cx: float, cy: float, r: float, W: float, H: float) -> list:
    """원을 감싸는 상자 [x, y, w, h] (0–1, 그림 안으로)."""
    ry = r * W / max(H, 1e-9)
    x0, y0, x1, y1 = max(0.0, cx - r), max(0.0, cy - ry), min(1.0, cx + r), min(1.0, cy + ry)
    return [x0, y0, x1 - x0, y1 - y0]


def _poly_area(pts) -> float:
    return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2


def _clean_value(schema: str, v, classes: set):
    """정답 값 하나를 정답 종류 규칙대로 정리 (맞지 않으면 None). 작업 파일·합치기·충돌에 같은 규칙."""
    if schema == "class":
        return v if isinstance(v, str) and v else None
    if schema == "number":
        return float(v) if _finite(v) else None
    if schema == "segments":
        if not isinstance(v, list):
            return None
        out = [[a, b, c] for a, b, c in (x for x in v if isinstance(x, list) and len(x) == 3)
               if isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int) and not isinstance(b, bool)
               and isinstance(c, str) and c in classes and 0 <= a < b]
        return out or None
    if schema == "boxes":                                   # 도형: 상자·다각형·점 (섞어 담음)
        if not isinstance(v, list):
            return None
        out = [s for s in (_clean_shape(x, classes) for x in v) if s is not None]
        return out or None
    return None


def _paint(segs: list, start: int, end: int, cls) -> list:
    """구간 목록에 [start, end)를 cls로 칠함 (None이면 지움). 겹치는 부분은 새 것이 이김, 같은 클래스 이웃은 합침."""
    out = []
    for a, b, c in segs:
        if b <= start or a >= end:
            out.append([a, b, c])
            continue
        if a < start:
            out.append([a, start, c])
        if b > end:
            out.append([end, b, c])
    if cls is not None:
        out.append([start, end, cls])
    out.sort()
    merged = []
    for a, b, c in out:
        if merged and merged[-1][2] == c and merged[-1][1] == a:
            merged[-1][1] = b
        else:
            merged.append([a, b, c])
    return merged


def _inside(path: str, folder: str) -> bool:
    """path가 folder 안(또는 folder 자체)인지."""
    try:
        p, f = Path(path).resolve(), Path(folder).resolve()
    except OSError:
        return False
    return p == f or f in p.parents


def _guard_outside(path: str, folder: str, what: str) -> None:
    """작업 파일·정답 표는 데이터 폴더 밖에만 (데이터 파일을 덮거나 다음에 샘플로 읽히지 않게)."""
    if _inside(path, folder):
        raise _err(f"{what}을(를) 데이터 폴더 안에 둘 수 없음: {path}", "데이터 폴더 밖의 경로를 고르세요 (예: 데이터 폴더 옆)")


def _stats(v: np.ndarray) -> dict:
    return {"length": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "std": float(v.std()), "first": float(v[0]), "last": float(v[-1]), "range": float(v.max() - v.min()),
            "argmax_frac": float(np.argmax(v) / max(1, len(v) - 1))}


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n, d)·(k, d) → (n, k) 제곱 거리. 내적으로 (n·k·d 크기의 중간 배열 없이)."""
    return np.maximum((X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None], 0.0)


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> tuple:
    """k-means++ 시작 + 반복. → (묶음 번호, 중심)."""
    rng = np.random.default_rng(seed)
    C = X[[rng.integers(len(X))]]
    for _ in range(1, k):
        d = _sqdist(X, C).min(1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))]])
    a = _sqdist(X, C).argmin(1)
    for _ in range(iters):
        C2 = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        a2 = _sqdist(X, C2).argmin(1)
        if (a2 == a).all():
            C = C2
            break
        a, C = a2, C2
    return a, C


def _softmax_fit(X: np.ndarray, y: np.ndarray, k: int, iters: int = 300, lr: float = 0.5, l2: float = 1e-2) -> tuple:
    """다항 로지스틱 회귀 (경사하강). X는 표준화된 특징."""
    W, b = np.zeros((X.shape[1], k)), np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(iters):
        P = _softmax(X @ W + b)
        G = (P - Y) / len(X)
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _downsample(v: np.ndarray, limit: int = DETAIL_MAX) -> list:   # (옛 이름, labeling_sources.downsample과 같음)
    """길면 limit/2 구간마다 [최소, 최대] (모양과 튀는 값을 지킴)."""
    if len(v) <= limit:
        return [float(x) for x in v]
    out = []
    for chunk in np.array_split(v, limit // 2):
        lo, hi = int(np.argmin(chunk)), int(np.argmax(chunk))
        out += [float(chunk[min(lo, hi)]), float(chunk[max(lo, hi)])]
    return out


class LabelProject:
    """작업 하나: 데이터 폴더(샘플마다 CSV) + 값 열 + 클래스 목록 + 정답."""

    def __init__(self, file: str, meta: dict):
        self.file = str(file)
        self.meta = meta
        self.samples: dict = {}            # 이름 → {"path", "stats", "thumb", "error"}
        self.missing: list = []            # 작업 파일엔 정답이 있지만 폴더에서 사라진 샘플
        self._undo: list = []
        self._redo: list = []
        self.clusters: dict = {}           # 이름 → 묶음 번호 (메모리에만, L1)
        self.who: str = ""                  # 지금 작업자 이름 (정답마다 남김, L6)
        self.source = make_source(meta["source"])
        self._initial: dict = {}           # 처음 정답 (하위 폴더 이름)
        self._scan()

    # ------------------------------------------------------------ 만들기·열기·저장
    @classmethod
    def new(cls, data_path: str, column: str, classes: list, file: str, kind: str = "series_folder",
            schema: str = "class", id_col: Optional[str] = None, purpose: str = "general",
            columns: Optional[list] = None) -> "LabelProject":
        """새 작업. kind: series_folder | image_folder | table, schema: class | number (회귀) | segments (구간).
        purpose "anomaly"(이상 탐지)면 클래스를 정상·이상으로 둠 (정상 후보 찾기·정상만 내보내기)."""
        if purpose not in PURPOSES:
            raise _err(f"작업 목적 '{purpose}'을(를) 모름", "general 또는 anomaly")
        if purpose == "anomaly":
            if schema == "number":
                raise _err("이상 탐지 작업은 클래스(정상·이상)나 구간 정답으로", "정답 종류를 클래스 또는 구간으로 고르세요")
            classes = list(dict.fromkeys(["정상", "이상"] + [str(c).strip() for c in classes if str(c).strip()]))
        if schema not in SCHEMAS:
            raise _err(f"정답 종류 '{schema}'을(를) 모름", "class(클래스), number(숫자), segments(구간)")
        if schema == "segments" and kind != "series_folder":
            raise _err("구간 정답은 시계열 폴더에서만", "데이터를 '시계열: 샘플마다 CSV 하나인 폴더'로 고르세요")
        if schema == "boxes" and kind != "image_folder":
            raise _err("상자 정답은 이미지 폴더에서만", "데이터를 '이미지 폴더'로 고르세요")
        names = [str(c).strip() for c in classes if str(c).strip()] if schema != "number" else []
        if len(set(names)) != len(names) or len(names) > MAX_CLASSES:
            raise _err(f"클래스는 서로 다른 이름으로 {MAX_CLASSES}개까지", "겹치는 이름을 빼거나 수를 줄이세요")
        src = {"kind": kind, "path": str(data_path), "column": column}
        if id_col:
            src["id_col"] = id_col
        extra = [str(c) for c in (columns or []) if str(c) and str(c) != column]
        if kind == "series_folder" and extra:                # 함께 볼 열 (기준 열은 column)
            src["columns"] = [column] + list(dict.fromkeys(extra))
        _guard_outside(file, make_source(src).guard_root(), "작업 파일")
        if Path(file).exists():
            raise _err(f"작업 파일이 이미 있음: {file}", "이어서 하려면 '작업 파일 열기'를 쓰고, 새로 하려면 다른 경로를 고르세요")
        meta = {"format": FORMAT, "version": VERSION, "source": src, "purpose": purpose,
                "schema": {"kind": schema, "classes": [{"name": n, "color": COLORS[i]} for i, n in enumerate(names)]},
                "items": {}, "updated": ""}
        p = cls(file, meta)
        if schema == "class":                              # 클래스별 하위 폴더 → 처음 정답으로 (확인 전)  (상자·구간·숫자는 아님)
            for n, c in p._initial.items():
                p.meta["items"][n] = {"label": c, "by": "import", "confirmed": False, "note": ""}
            p._add_missing_classes()
            if not p.classes:
                raise _err("클래스가 없음", "정답 종류를 하나 이상 넣으세요 (예: 정상, 이상)")
        if schema == "boxes" and not p.classes:
            raise _err("클래스가 없음", "상자에 붙일 물체 종류를 하나 이상 넣으세요 (예: 결함)")
        p._save()                                          # 구간 작업은 클래스 없이 시작해도 됨 (열에서 가져오기)
        return p

    @classmethod
    def open(cls, file: str) -> "LabelProject":
        try:
            meta = json.loads(Path(file).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise _err(f"작업 파일을 찾을 수 없음: {file}", "라벨링 작업 파일(.dllabel)의 전체 경로를 넣으세요")
        except (ValueError, UnicodeDecodeError):
            raise _err(f"작업 파일을 읽을 수 없음: {file}", "BlockLearning 라벨링 작업 파일(.dllabel)인지 확인하세요")
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            raise _err(f"BlockLearning 라벨링 작업 파일이 아님: {file}", "라벨링 작업실에서 만든 .dllabel 파일을 여세요")
        src, schema = meta.get("source", {}), meta.get("schema", {})
        if meta.get("version") != VERSION or src.get("kind") not in ("series_folder", "image_folder", "table") \
                or schema.get("kind") not in SCHEMAS:
            raise _err(f"이 작업실 버전에서는 아직 못 여는 작업: {src.get('kind')} / {schema.get('kind')}",
                       "시계열 폴더·이미지 폴더·표의 클래스·숫자 정답 작업만 열 수 있습니다")
        combo = {"segments": "series_folder", "boxes": "image_folder"}.get(schema.get("kind"))
        if combo and src.get("kind") != combo:               # 구간은 시계열에서만, 상자는 그림에서만
            raise _err(f"맞지 않는 작업: {src.get('kind')} 데이터에 {schema.get('kind')} 정답", "작업 파일이 손상되었는지 확인하세요")
        meta.setdefault("items", {})
        if meta.get("purpose") not in PURPOSES:
            meta["purpose"] = "general"
        cls._sanitize(meta)
        p = cls(file, meta)
        p._add_missing_classes()
        return p

    @staticmethod
    def _sanitize(meta: dict) -> None:
        """남이 만든 작업 파일도 안전하게: 색은 #rrggbb만, 출처는 아는 값만, 정답·메모는 글자만 (화면에 그대로 넣는 값)."""
        cl = [c for c in meta["schema"].get("classes", []) if isinstance(c, dict) and isinstance(c.get("name"), str)]
        for i, c in enumerate(cl):
            if not isinstance(c.get("color"), str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", c["color"]):
                c["color"] = COLORS[i % len(COLORS)]
        meta["schema"]["classes"] = [{"name": c["name"], "color": c["color"]} for c in cl[:MAX_CLASSES]]
        clean = {}
        for n, it in (meta.get("items") or {}).items():
            if not isinstance(n, str) or not isinstance(it, dict):
                continue
            lab = it.get("label")
            if meta["schema"]["kind"] == "number":
                lab = float(lab) if _finite(lab) else None
            elif not isinstance(lab, str):
                lab = None
            by = it.get("by") if it.get("by") in BY else None
            segs = []
            if meta["schema"]["kind"] == "segments" and isinstance(it.get("segments"), list):
                segs = [[int(a), int(b), c] for a, b, c in (x for x in it["segments"] if isinstance(x, list) and len(x) == 3)
                        if isinstance(a, int) and isinstance(b, int) and isinstance(c, str) and 0 <= a < b]
                lab = None
            has = lab is not None and lab != "" or bool(segs)
            clean[n] = {"label": lab if has and not segs else None, "by": by if has else None,
                        "confirmed": bool(it.get("confirmed")) and has, "note": it.get("note") if isinstance(it.get("note"), str) else ""}
            if segs:
                clean[n]["segments"] = segs
            if isinstance(it.get("who"), str) and it["who"].strip():
                clean[n]["who"] = it["who"].strip()[:40]
            kind = meta["schema"]["kind"]
            names_ok = {c["name"] for c in meta["schema"]["classes"]}
            cf = it.get("conflict")
            if isinstance(cf, dict):                       # 충돌 값도 정답 종류 규칙 그대로 (아니면 버림)
                key = {"segments": "segments", "boxes": "boxes"}.get(kind, "label")
                cv = _clean_value(kind, cf.get(key), names_ok if kind in REGION else set())
                if cv is not None:
                    clean[n]["conflict"] = {key: cv, "who": cf["who"][:40] if isinstance(cf.get("who"), str) else ""}
            if kind == "boxes" and isinstance(it.get("boxes"), list):
                bx = _clean_value("boxes", it["boxes"], names_ok)
                if bx:
                    clean[n].update(boxes=bx, label=None, by=by, confirmed=bool(it.get("confirmed")))
            p = it.get("p")
            if by == "model" and isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1:
                clean[n]["p"] = float(p)
        meta["items"] = clean

    def _add_missing_classes(self) -> None:
        """정답에 있는데 클래스 목록에 없는 이름은 목록에 더함."""
        if self.schema != "class":
            return
        known = {c["name"] for c in self.classes}
        for it in self.meta["items"].values():
            lab = it.get("label")
            if lab and lab not in known and len(self.classes) < MAX_CLASSES:
                self.classes.append({"name": lab, "color": COLORS[len(self.classes) % len(COLORS)]})
                known.add(lab)

    def _save(self) -> None:
        """임시 파일에 쓴 뒤 바꿔치기 (중간에 꺼져도 작업 파일이 깨지지 않게)."""
        self.meta["updated"] = datetime.now().isoformat(timespec="seconds")
        f = Path(self.file)
        tmp = f.with_name(f.name + ".tmp")
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, f)
        except OSError as e:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise _err(f"작업 파일을 저장할 수 없음: {self.file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로로 작업 파일을 만드세요")

    def _scan(self) -> None:
        """폴더에서 샘플 목록·요약값·썸네일을 읽음. 값 열을 못 읽는 샘플은 error만 두고 계속."""
        self.samples, self._initial = self.source.scan()
        if isinstance(self.meta.get("assigned"), list):   # 나눈 작업: 맡은 샘플만
            mine = set(self.meta["assigned"])
            self.samples = {n: s for n, s in self.samples.items() if n in mine}
        self.missing = sorted(n for n in self.meta["items"] if n not in self.samples)

    # ------------------------------------------------------------ 읽기
    @property
    def classes(self) -> list:
        return self.meta["schema"]["classes"]

    @property
    def schema(self) -> str:
        return self.meta["schema"]["kind"]

    @property
    def stat_names(self) -> tuple:
        return tuple(self.source.stat_names)

    def _it(self, name: str) -> dict:
        return self.meta["items"].get(name) or {}

    def state(self) -> dict:
        items = {n: self._it(n) for n in self.samples}
        lab = [it for it in items.values() if _has(it)]
        counts = {"total": len(items), "labeled": len(lab), "unlabeled": len(items) - len(lab),
                  "unconfirmed": sum(1 for it in lab if not it.get("confirmed")),
                  "error": sum(1 for s in self.samples.values() if s["error"])}
        counts.update({b: sum(1 for it in lab if it.get("by") == b) for b in BY})
        counts["conflict"] = sum(1 for it in items.values() if it.get("conflict"))
        per = {c["name"]: 0 for c in self.classes}
        for it in lab:
            if self.schema == "class":
                per[it["label"]] = per.get(it["label"], 0) + 1
            elif self.schema in REGION:                    # 구간·상자: 그 클래스가 있는 샘플 수
                for c in set(_regions(it)):
                    per[c] = per.get(c, 0) + 1
        return {"file": self.file, "source": dict(self.meta["source"]), "missing": len(self.missing),
                "schema": self.schema, "stat_names": list(self.stat_names), "purpose": self.meta.get("purpose", "general"),
                "labelers": sorted({it["who"] for it in items.values() if it.get("who")}), "who": self.who,
                "assigned": len(self.meta["assigned"]) if isinstance(self.meta.get("assigned"), list) else None,
                "classes": [dict(c, count=per.get(c["name"], 0)) for c in self.classes], "counts": counts,
                "can_undo": bool(self._undo), "can_redo": bool(self._redo)}

    def _row(self, n: str) -> dict:
        s, it = self.samples[n], self._it(n)
        return {"name": n, "label": it.get("label"), "by": it.get("by"), "confirmed": bool(it.get("confirmed")),
                "note": it.get("note", ""), "thumb": s["thumb"], "thumbs": s.get("thumbs"), "stats": s["stats"], "error": s["error"],
                "p": it.get("p"), "cluster": self.clusters.get(n), "segments": it.get("segments", []), "boxes": it.get("boxes", []),
                "who": it.get("who"), "conflict": it.get("conflict")}

    def _filtered(self, flt: Optional[dict]) -> list:
        f = flt or {}
        want, by = f.get("label", "all"), f.get("by")
        cls = want[4:] if isinstance(want, str) and want.startswith("cls:") else None   # 클래스는 "cls:이름" (예약어와 안 겹치게)
        out = []
        for n in self.samples:
            it = self._it(n)
            lab, has = it.get("label"), _has(it)
            if want == "none" and has or want == "any" and not has:
                continue
            if cls is not None and (lab != cls if self.schema not in REGION else cls not in set(_regions(it))):
                continue
            if by and it.get("by") != by:
                continue
            if f.get("unconfirmed") and not (has and not it.get("confirmed")):
                continue
            if f.get("error") and not self.samples[n]["error"]:
                continue
            if f.get("cluster") is not None and self.clusters.get(n) != f.get("cluster"):
                continue
            if f.get("who") and it.get("who") != f.get("who"):
                continue
            if f.get("conflict") and not it.get("conflict"):
                continue
            out.append(n)
        return out

    def items(self, filter: Optional[dict] = None, sort: Optional[dict] = None, page: int = 0, size: int = PAGE) -> dict:
        names = self._filtered(filter)
        key = (sort or {}).get("key", "name")
        desc = bool((sort or {}).get("desc"))
        if key == "uncertainty":                           # 모델 제안 중 헷갈리는 것(확률 낮은 것)부터, 나머지는 뒤
            withp = sorted((n for n in names if self._it(n).get("p") is not None), key=lambda n: (self._it(n)["p"], n))
            names = withp + sorted(n for n in names if self._it(n).get("p") is None)
        elif key == "label" and self.schema == "number":   # 숫자 정답 순 (정답 없는 것은 뒤)
            ok = sorted((n for n in names if _has(self._it(n))), key=lambda n: (self._it(n)["label"], n), reverse=desc)
            names = ok + sorted(n for n in names if not _has(self._it(n)))
        elif key in self.stat_names:                       # 오류 샘플(요약값 없음)은 맨 뒤
            ok = [n for n in names if self.samples[n]["stats"]]
            ok.sort(key=lambda n: (self.samples[n]["stats"][key], n), reverse=desc)
            names = ok + sorted(n for n in names if not self.samples[n]["stats"])
        else:
            names.sort(reverse=desc)
        size = max(1, int(size))
        pages = max(1, -(-len(names) // size))
        page = min(max(0, int(page)), pages - 1)
        return {"total": len(names), "page": page, "pages": pages,
                "items": [self._row(n) for n in names[page * size:(page + 1) * size]]}

    def _need(self, name: str) -> dict:
        if name not in self.samples:
            raise _err(f"샘플을 찾을 수 없음: {name}", "목록을 새로 고치세요")
        return self.samples[name]

    def item(self, name: str) -> dict:
        s = self._need(name)
        out = dict(self._row(name), values=[], n=0)
        out.pop("thumb")
        out.pop("thumbs")
        if not s["error"]:
            out.update(self.source.detail(name, s))
        return out

    def file_path(self, name: str) -> Path:
        """그림 원본 경로 (이미지 작업만, 크게 보기용)."""
        s = self._need(name)
        if self.source.kind != "image_folder":
            raise _err("그림 작업이 아님", "")
        return self.source.file(s)

    # ------------------------------------------------------------ 편집 (되돌리기 기록 + 저장)
    def _edit(self, names, fn, classes: bool = False) -> None:
        """바꾸기 전 상태를 기록하고 fn()을 하고 저장. 저장에 실패하면 되돌리고 오류."""
        before = {n: copy.deepcopy(self.meta["items"].get(n)) for n in names}
        cls_before = copy.deepcopy(self.classes) if classes else None
        fn()
        changed = any(self.meta["items"].get(n) != b for n, b in before.items()) or \
            (classes and self.classes != cls_before)
        if not changed:
            return
        rec = (before, cls_before)
        try:
            self._save()
        except DataError:
            self._restore(rec)
            raise
        self._undo.append(rec)
        del self._undo[:-UNDO_STEPS]
        self._redo.clear()

    def _restore(self, rec) -> tuple:
        """기록으로 되돌리고, 되돌리기 전 상태(반대 방향 기록)를 돌려줌."""
        before, cls_before = rec
        now = {n: copy.deepcopy(self.meta["items"].get(n)) for n in before}
        cls_now = copy.deepcopy(self.classes) if cls_before is not None else None
        for n, it in before.items():
            if it is None:
                self.meta["items"].pop(n, None)
            else:
                self.meta["items"][n] = it
        if cls_before is not None:
            self.meta["schema"]["classes"] = cls_before
        return now, cls_now

    def _norm_label(self, label):
        """정답 값 검사·정리: 클래스는 목록에 있는 이름, 숫자는 유한한 실수. None은 지우기."""
        if label is None:
            return None
        if self.schema == "number":
            v = label if isinstance(label, (int, float)) and not isinstance(label, bool) else _num(str(label))
            if v is None or not np.isfinite(v):
                raise _err(f"숫자 정답이 아님: {label!r}", "숫자를 넣으세요 (예: 3.5)")
            return float(v)
        if label not in {c["name"] for c in self.classes}:
            raise _err(f"클래스 '{label}'이(가) 없음", "클래스를 먼저 더하세요")
        return label

    def _check_label(self, label) -> None:
        self._norm_label(label)

    def _put(self, name: str, label: Optional[str], by: str, p: Optional[float] = None) -> None:
        it = dict(self.meta["items"].get(name) or {"note": ""})
        it.pop("p", None)
        self._stamp(it, by)
        if label is None:
            it.update(label=None, by=None, confirmed=False)
        else:
            it.update(label=label, by=by, confirmed=(by == "manual"))
            if p is not None:
                it["p"] = round(float(p), 4)
        if _has(it) or it.get("note"):
            self.meta["items"][name] = it
        else:
            self.meta["items"].pop(name, None)

    def _no_segments(self, what: str) -> None:
        if self.schema == "segments":
            raise _err(f"구간 작업에서는 {what}을(를) 쓸 수 없음", "크게 보기에서 구간을 칠하거나 '열에서 가져오기'를 쓰세요")
        if self.schema == "boxes":
            raise _err(f"상자 작업에서는 {what}을(를) 쓸 수 없음", "크게 보기에서 그림 위에 상자를 그리세요")

    def _stamp(self, it: dict, by: Optional[str]) -> None:
        """작업자 이름을 남기고, 사람이 직접 고치면 충돌은 풀린 것으로."""
        if self.who:
            it["who"] = self.who
        if by == "manual":
            it.pop("conflict", None)

    def set_labels(self, names: list, label) -> int:
        """사람이 붙이기(또는 지우기: label None). 바뀐 수."""
        self._no_segments("샘플 정답")
        label = self._norm_label(label)
        names = [n for n in dict.fromkeys(names) if self._need(n)]
        before = {n: (self._it(n).get("label"), self._it(n).get("by")) for n in names}
        todo = [n for n in names if before[n] != (label, "manual" if label is not None else None) or self._it(n).get("conflict")]
        self._edit(todo, lambda: [self._put(n, label, "manual") for n in todo])
        return len(todo)

    def confirm(self, names: list) -> int:
        """규칙·가져옴 정답을 사람이 확인 (정답 있는 것만). 확인한 수."""
        todo = [n for n in dict.fromkeys(names) if self._need(n) and _has(self._it(n)) and not self._it(n).get("confirmed")]

        def run():
            for n in todo:
                self.meta["items"][n] = dict(self.meta["items"][n], confirmed=True)
                if self.who:
                    self.meta["items"][n]["who"] = self.who
        self._edit(todo, run)
        return len(todo)

    def note(self, name: str, text: str) -> None:
        self._need(name)

        def run():
            it = dict(self.meta["items"].get(name) or {"label": None, "by": None, "confirmed": False})
            it["note"] = str(text)
            if _has(it) or it["note"]:
                self.meta["items"][name] = it
            else:
                self.meta["items"].pop(name, None)
        self._edit([name], run)

    def _swap(self, src: list, dst: list) -> bool:
        if not src:
            return False
        rec = src.pop()
        back = self._restore(rec)
        try:
            self._save()
        except DataError:
            self._restore(back)
            src.append(rec)
            raise
        dst.append(back)
        return True

    def undo(self) -> bool:
        return self._swap(self._undo, self._redo)

    def redo(self) -> bool:
        return self._swap(self._redo, self._undo)

    # ------------------------------------------------------------ 클래스
    def classes_op(self, op: str, name: str = "", new: str = "", color: str = "") -> None:
        names = [c["name"] for c in self.classes]
        name, new = str(name).strip(), str(new).strip()
        if op == "add":
            if not name or name in names:
                raise _err(f"클래스 이름 '{name}'을(를) 쓸 수 없음", "비어 있지 않고 겹치지 않는 이름으로")
            if len(names) >= MAX_CLASSES:
                raise _err(f"클래스는 {MAX_CLASSES}개까지", "단축키 1–9에 맞춰 9개까지만 만들 수 있습니다")
            self._edit([], lambda: self.classes.append({"name": name, "color": color or COLORS[len(names) % len(COLORS)]}),
                       classes=True)
            return
        if name not in names:
            raise _err(f"클래스 '{name}'이(가) 없음", "목록에 있는 클래스를 고르세요")
        i = names.index(name)
        if op == "color":
            self._edit([], lambda: self.classes[i].update(color=color), classes=True)
        elif op == "rename":
            if not new or new in names:
                raise _err(f"클래스 이름 '{new}'을(를) 쓸 수 없음", "비어 있지 않고 겹치지 않는 이름으로")
            hit = [n for n, it in self.meta["items"].items() if it.get("label") == name or name in _regions(it)]

            def run():
                self.classes[i]["name"] = new
                for n in hit:
                    it = dict(self.meta["items"][n])
                    if it.get("label") == name:
                        it["label"] = new
                    for key in ("segments", "boxes"):           # 구간·상자 안의 클래스 이름도
                        if it.get(key):
                            it[key] = [list(x[:-1]) + [new if x[-1] == name else x[-1]] for x in it[key]]
                    self.meta["items"][n] = it
            self._edit(hit, run, classes=True)
        elif op == "remove":
            if any(it.get("label") == name or name in _regions(it) for it in self.meta["items"].values()):
                raise _err(f"클래스 '{name}'에 붙은 정답이 있어 지울 수 없음", "그 정답들을 먼저 다른 클래스로 바꾸거나 지우세요")
            self._edit([], lambda: self.classes.pop(i), classes=True)
        else:
            raise _err(f"클래스 동작 '{op}'을(를) 모름", "add, rename, color, remove 중 하나")

    # ------------------------------------------------------------ 규칙·가져오기·내보내기
    def rule_matches(self, conds: list, overwrite: bool = False, label=None) -> list:
        """요약값 조건(모두 만족) → 샘플 이름. overwrite가 아니면 정답 있는 샘플은 뺌. 오류 샘플은 늘 뺌."""
        if not conds:
            raise _err("조건이 없음", "조건을 하나 이상 넣으세요 (예: max > 5000)")
        for c in conds:
            if c.get("stat") not in self.stat_names or c.get("op") not in OPS or not isinstance(c.get("value"), (int, float))                     or isinstance(c.get("value"), bool):
                raise _err(f"조건이 맞지 않음: {c}", f"요약값은 {', '.join(self.stat_names)}, 비교는 < <= > >= ==, 값은 숫자")
        out = []
        for n, s in self.samples.items():
            cur = self._it(n).get("label") if _has(self._it(n)) else None
            if not s["stats"] or (not overwrite and cur is not None) or (label is not None and cur == label):
                continue
            if all(bool(OPS[c["op"]](s["stats"][c["stat"]], float(c["value"]))) for c in conds):
                out.append(n)
        return sorted(out)

    def rule_apply(self, conds: list, label, overwrite: bool = False) -> int:
        self._no_segments("규칙")
        label = self._norm_label(label)
        if label is None:
            raise _err("붙일 정답이 없음", "클래스를 고르거나 숫자를 넣으세요")
        todo = self.rule_matches(conds, overwrite, label)
        self._edit(todo, lambda: [self._put(n, label, "rule") for n in todo])
        return len(todo)

    def import_table(self, label_file: str) -> dict:
        """정답 표(파일 이름, 정답) → 가져옴 정답. 표에만 있는 이름 수를 알려 줌. 처음 보는 클래스는 더함."""
        self._no_segments("정답 표 가져오기")
        key = self.source.key
        keys = {key(n): n for n in self.samples}           # "폴더/파일"이 먼저, 없으면 파일 이름만
        for n in self.samples:
            keys.setdefault(key(Path(n).name), n)
        table = _label_table(label_file, set(keys), key)
        matched = {keys[k]: v for k, v in table.items() if k in keys}
        if self.schema == "number":                        # 숫자 정답: 숫자가 아니면 이름과 함께 오류
            bad = [n for n, v in matched.items() if _num(str(v)) is None]
            if bad:
                raise _err(f"정답 표에 숫자가 아닌 정답: {bad[0]} = '{matched[bad[0]]}'", "숫자 정답 작업에는 숫자만 넣으세요")
            matched = {n: float(_num(str(v))) for n, v in matched.items()}
        kept = [n for n in matched if self._it(n).get("by") == "manual"]      # 사람이 붙인 정답은 덮지 않음
        hits = {n: v for n, v in matched.items() if n not in kept}
        new_cls = [v for v in dict.fromkeys(hits.values()) if v not in {c["name"] for c in self.classes}] \
            if self.schema == "class" else []
        if len(self.classes) + len(new_cls) > MAX_CLASSES:
            raise _err(f"정답 표의 클래스가 너무 많음 (지금 {len(self.classes)}개 + 새로 {len(new_cls)}개)", f"클래스는 {MAX_CLASSES}개까지")

        def run():
            for v in new_cls:
                self.classes.append({"name": v, "color": COLORS[len(self.classes) % len(COLORS)]})
            for n, v in hits.items():
                self._put(n, v, "import")
        self._edit(list(hits), run, classes=True)
        return {"applied": len(hits), "unknown": len(table) - len(matched), "kept_manual": len(kept)}

    # ------------------------------------------------------------ L1: 묶기·모델 제안
    def _features(self) -> tuple:
        """샘플 특징: 썸네일(샘플마다 평균 0·표준편차 1) + 요약값(전체 기준 표준화, 가중 2). 오류 샘플은 뺌."""
        names = [n for n, s in self.samples.items() if s["stats"] and s["thumb"]]
        if not names:
            raise _err("특징을 만들 샘플이 없음", "값 열을 읽을 수 있는 샘플이 있어야 합니다")
        def shape(n):                                       # 여러 열이면 열마다 모양을 이어 붙임
            ts = self.samples[n].get("thumbs") or [self.samples[n]["thumb"]]
            return np.concatenate([(np.asarray(t, float) - np.mean(t)) / (np.std(t) + 1e-9) for t in ts])
        T = np.array([shape(n) for n in names], dtype=float)
        S = np.array([[self.samples[n]["stats"][k] for k in self.stat_names] for n in names], dtype=float)
        S = (S - S.mean(0)) / (S.std(0) + 1e-9)
        return names, np.hstack([T, 2 * S])

    def cluster(self, k: int = 6) -> dict:
        """비슷한 모양끼리 k개 묶음 (k-means). 정답은 바꾸지 않음. → 묶음마다 크기·대표·정답 분포 (큰 묶음부터)."""
        names, X = self._features()
        if isinstance(k, bool) or not isinstance(k, int) or not CLUSTER_K[0] <= k <= min(CLUSTER_K[1], len(names)):
            raise _err(f"묶음 수 {k}는 쓸 수 없음", f"{CLUSTER_K[0]}–{min(CLUSTER_K[1], len(names))} 사이로 정하세요")
        a, C = _kmeans(X, k)
        order = sorted(range(k), key=lambda j: -int((a == j).sum()))        # 큰 묶음이 0번
        new_id = {j: i for i, j in enumerate(order)}
        D = _sqdist(X, C)
        self.clusters = {n: new_id[int(a[i])] for i, n in enumerate(names)}
        groups = []
        for j in order:
            idx = np.flatnonzero(a == j)
            if not len(idx):
                continue
            rep = names[int(idx[np.argmin(D[idx, j])])]
            dist: dict = {}
            for i in idx:
                lab = str(self._it(names[i])["label"]) if _has(self._it(names[i])) else "정답 없음"
                dist[lab] = dist.get(lab, 0) + 1
            groups.append({"id": new_id[j], "size": int(len(idx)), "rep": rep, "thumb": self.samples[rep]["thumb"], "labels": dist})
        return {"k": k, "groups": groups}

    def suggest(self) -> dict:
        """사람이 붙였거나 확인된 정답으로 작은 모델(로지스틱 회귀)을 학습해, 정답 없는 샘플(과 이전 모델 제안)에 제안.
        사람·규칙·가져온 정답은 덮지 않음. 예상 정확도는 학습 데이터의 20%를 떼어 잼 (10개 미만이면 None)."""
        if self.schema != "class":
            raise _err("모델 제안은 클래스 정답 작업에서만", "숫자·구간 정답은 규칙·가져오기로 붙이세요")
        names, X = self._features()
        cls = [c["name"] for c in self.classes]
        train = [i for i, n in enumerate(names) if self._it(n).get("label") in cls
                 and (self._it(n).get("by") == "manual" or self._it(n).get("confirmed"))]
        ys = [cls.index(self._it(names[i])["label"]) for i in train]
        counts = {c: ys.count(j) for j, c in enumerate(cls) if ys.count(j)}
        if len(counts) < 2 or min(counts.values()) < 2:
            raise _err("모델을 학습할 정답이 모자람", "클래스 2개 이상에, 클래스마다 2개 이상 사람이 붙이거나 확인한 정답이 있어야 합니다")
        Xt, yt = X[train], np.array(ys)
        acc = None
        if len(train) >= 10:                               # 적으면 떼어 낸 한두 개로 0%/100%가 나와 오해 → 모름
            perm = np.random.default_rng(0).permutation(len(train))
            cut = max(1, len(train) // 5)
            te, tr = perm[:cut], perm[cut:]
            if len(set(yt[tr])) >= 2:
                W, b = _softmax_fit(Xt[tr], yt[tr], len(cls))
                acc = float(np.mean((Xt[te] @ W + b).argmax(1) == yt[te]))
        W, b = _softmax_fit(Xt, yt, len(cls))
        targets = [i for i, n in enumerate(names) if not _has(self._it(n))
                   or (self._it(n).get("by") == "model" and not self._it(n).get("confirmed"))]
        P = _softmax(X[targets] @ W + b) if targets else np.zeros((0, len(cls)))
        todo = {names[i]: (cls[int(P[r].argmax())], float(P[r].max())) for r, i in enumerate(targets)}
        self._edit(list(todo), lambda: [self._put(n, lab, "model", p) for n, (lab, p) in todo.items()])
        return {"applied": len(todo), "trained_on": len(train), "acc": acc}

    # ------------------------------------------------------------ 딥러닝 제안 (studio 대표 모델로 학습 → 정답 없는 샘플에 제안)
    def deep_data(self) -> dict:
        """학습할 정답(사람이 붙였거나 확인한 것)과 제안할 샘플 → Dataset(학습/시험 나눔·정규화) + 제안할 값(같은 정규화).
        시계열은 기준 열을 중앙값 길이로 맞추고, 그림은 28×28 흑백, 표는 숫자 열."""
        from blocklearning.data import SERIES_MAX_LEN, _fit_length, _gray01, make_dataset
        if self.schema != "class":
            raise _err("딥러닝 제안은 클래스 정답 작업에서만", "숫자·구간·상자 정답은 규칙·가져오기로 붙이세요")
        cls = [c["name"] for c in self.classes]
        ok = [n for n, s in self.samples.items() if not s["error"]]
        train = [n for n in ok if self._it(n).get("label") in cls and (self._it(n).get("by") == "manual" or self._it(n).get("confirmed"))]
        targets = [n for n in ok if not _has(self._it(n)) or (self._it(n).get("by") == "model" and not self._it(n).get("confirmed"))]
        counts = {c: sum(1 for n in train if self._it(n)["label"] == c) for c in cls}
        if sum(1 for v in counts.values() if v >= 2) < 2 or len(train) < 10:
            raise _err(f"딥러닝 제안에는 사람이 붙이거나 확인한 정답이 10개 이상 필요함 (지금 {len(train)}개)",
                       "클래스 2개 이상에, 클래스마다 2개 이상 붙인 뒤 다시 누르세요. 적으면 [모델 제안]이 낫습니다")
        if not targets:
            raise _err("제안할 샘플이 없음", "정답 없는 샘플이 있어야 합니다")
        kind, image = self.source.kind, None
        if kind == "series_folder":
            lens = [self.samples[n]["stats"]["length"] for n in ok]
            L = int(min(SERIES_MAX_LEN, max(8, int(np.median(lens)) // 8 * 8)))
            X = {n: _fit_length(self.source.values(n, self.samples[n]), L, "interp") for n in train + targets}
        elif kind == "image_folder":
            Image = self.source._pil()
            X = {}
            for n in train + targets:
                with Image.open(self.source.file(self.samples[n])) as im:
                    X[n] = _gray01(im, 28, Image).reshape(-1)
            image = (28, 28)
        else:
            X = {n: np.array([self.samples[n]["stats"][k] for k in self.stat_names], float) for n in train + targets}
        d = make_dataset(np.stack([X[n] for n in train]), np.array([self._it(n)["label"] for n in train], dtype=object),
                         task="classification", test_ratio=0.2 if len(train) >= 20 else 0.15, seed=0, image=image,
                         features=list(self.stat_names) if kind == "table" else None)
        P = (np.stack([X[n] for n in targets]) - d.mean) / d.std
        return {"data": d, "X_pred": P.astype(np.float32), "targets": targets, "kind": kind}

    def apply_model(self, targets: list, classes: list, probs) -> dict:
        """딥러닝 모델의 확률 (샘플 수, 클래스 수) → 정답 없는 샘플에 '모델 제안'으로 (사람·규칙·가져온 정답은 덮지 않음)."""
        P = np.asarray(probs, dtype=float).reshape(len(targets), len(classes))
        todo = {}
        for n, p in zip(targets, P):
            it = self._it(n)
            if n in self.samples and (not _has(it) or (it.get("by") == "model" and not it.get("confirmed"))):
                todo[n] = (str(classes[int(p.argmax())]), float(p.max()))
        self._edit(list(todo), lambda: [self._put(n, lab, "model", p) for n, (lab, p) in todo.items()])
        return {"applied": len(todo)}

    def labels(self, confirmed_only: bool = False) -> dict:
        """폴더에 있는 샘플 중 정답이 있는 것 → {이름: 정답}."""
        return {n: it["label"] for n in self.samples for it in [self._it(n)]
                if _has(it) and (it.get("confirmed") or not confirmed_only) and not self.samples[n]["error"]}

    # ------------------------------------------------------------ L3: 시계열 구간
    def paint(self, name: str, start: int, end: int, cls) -> None:
        """구간 칠하기 [start, end) (행 번호, 끝은 제외). cls None이면 지우개. 사람 정답(확인됨)."""
        if self.schema != "segments":
            raise _err("구간 작업이 아님", "")
        s = self._need(name)
        n = (s["stats"] or {}).get("length", 0)
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) \
                or not 0 <= start < end <= n:
            raise _err(f"구간 {start}–{end}이(가) 맞지 않음 (이 샘플은 {n}행)", "0 이상, 시작 < 끝 ≤ 행 수")
        if cls is not None and cls not in {c["name"] for c in self.classes}:
            raise _err(f"클래스 '{cls}'이(가) 없음", "클래스를 먼저 더하세요")

        def run():
            it = dict(self.meta["items"].get(name) or {"note": ""})
            segs = _paint([list(x) for x in it.get("segments", [])], start, end, cls)
            self._stamp(it, "manual")
            it.update(segments=segs, label=None, by="manual" if segs else None, confirmed=bool(segs))
            if not segs:
                it.pop("segments", None)
            if segs or it.get("note"):
                self.meta["items"][name] = it
            else:
                self.meta["items"].pop(name, None)
        self._edit([name], run)

    def import_column(self, column: str) -> dict:
        """데이터 CSV의 열(예: 단계)에서 같은 값이 이어지는 구간을 만듦 (가져옴, 확인 전). 사람이 칠한 샘플은 그대로.
        처음 보는 값은 클래스에 더함 (9개까지)."""
        if self.schema != "segments":
            raise _err("구간 작업이 아님", "")
        todo, kept = {}, 0
        for n, smp in self.samples.items():
            if smp["error"]:
                continue
            if self._it(n).get("by") == "manual" and self._it(n).get("segments"):
                kept += 1
                continue
            names, rows = _read_table(smp["path"])
            if column not in names:
                raise _err(f"{n}에 열 '{column}'이(가) 없음", "모든 파일에 있는 열을 고르세요")
            j = names.index(column)
            vals = [(r[j].strip() if j < len(r) else "") for _, r in rows]
            segs, a = [], 0
            for i in range(1, len(vals) + 1):
                if i == len(vals) or vals[i] != vals[a]:
                    if vals[a]:
                        segs.append([a, i, vals[a]])
                    a = i
            todo[n] = segs
        new_cls = [c for c in dict.fromkeys(x[2] for segs in todo.values() for x in segs)
                   if c not in {k["name"] for k in self.classes}]
        if len(self.classes) + len(new_cls) > MAX_CLASSES:
            raise _err(f"열 '{column}'의 값 종류가 너무 많음 (새로 {len(new_cls)}개)", f"클래스는 {MAX_CLASSES}개까지: 단계·상태 같은 열을 고르세요")

        def run():
            for c in new_cls:
                self.classes.append({"name": c, "color": COLORS[len(self.classes) % len(COLORS)]})
            for n, segs in todo.items():
                it = dict(self.meta["items"].get(n) or {"note": ""})
                it.update(segments=segs, label=None, by="import" if segs else None, confirmed=False)
                if not segs:
                    it.pop("segments", None)
                if segs or it.get("note"):
                    self.meta["items"][n] = it
                else:
                    self.meta["items"].pop(n, None)
        self._edit(list(todo), run, classes=True)
        return {"samples": sum(1 for v in todo.values() if v), "kept_manual": kept, "new_classes": new_cls}

    def export_segments(self, file: str, confirmed_only: bool = False) -> int:
        """구간 표 CSV (file,start,end,label; 끝은 제외). 쓴 행 수."""
        if self.schema != "segments":
            raise _err("구간 작업이 아님", "")
        _guard_outside(file, self.source.guard_root(), "구간 표")
        rows = [(n, a, b, c) for n in self.samples for it in [self._it(n)]
                if it.get("segments") and (it.get("confirmed") or not confirmed_only) for a, b, c in it["segments"]]
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "start", "end", "label"])
                w.writerows(rows)
        except OSError as e:
            raise _err(f"구간 표를 저장할 수 없음: {file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 넣으세요")
        return len(rows)

    # ------------------------------------------------------------ L4: 이상 탐지
    def normal_candidates(self, k: int = 4) -> list:
        """정상 후보: 비슷한 것끼리 묶었을 때 가장 큰 묶음 (보통 정상이 대부분). 정답은 붙이지 않음."""
        names, _ = self._features()
        k = max(2, min(int(k), len(names), 20))
        r = self.cluster(k)
        big = r["groups"][0]["id"]
        return sorted(n for n, g in self.clusters.items() if g == big)

    def export_normals(self, file: str, confirmed_only: bool = False) -> int:
        """'정상' 샘플 목록 CSV (file) — 정상만으로 학습하는 이상 탐지(오토인코더)용. 쓴 행 수."""
        _guard_outside(file, self.source.guard_root(), "정상 목록")
        if self.schema != "class":
            raise _err("정상 목록은 클래스(정상·이상) 정답 작업에서만", "구간 작업은 '내보내기'로 구간 표를 저장하세요")
        names = [n for n, v in self.labels(confirmed_only).items() if v == NORMAL]
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file"])
                w.writerows([n] for n in names)
        except OSError as e:
            raise _err(f"정상 목록을 저장할 수 없음: {file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 넣으세요")
        return len(names)

    # ------------------------------------------------------------ L5: 이미지 상자 (객체 탐지)
    def set_boxes(self, name: str, boxes: list) -> None:
        """그림 하나의 도형 전체 (상자 [x, y, w, h, 클래스] · 다각형 ["poly", [[x, y], …], 클래스] · 점 ["pt", x, y, 클래스]
        · 원 ["circle", 가운데 x, 가운데 y, 반지름(너비 비율), 클래스],
        왼쪽 위 기준, 그림 크기에 대한 0–1 비율). 사람 정답(확인됨)."""
        if self.schema != "boxes":
            raise _err("도형(상자·다각형·점) 작업이 아님", "")
        self._need(name)
        known = {c["name"] for c in self.classes}
        clean = []
        for b in boxes or []:
            if isinstance(b, (list, tuple)) and b and isinstance(b[-1], str) and b[-1] not in known:
                raise _err(f"클래스 '{b[-1]}'이(가) 없음", "클래스를 먼저 더하세요")
            s = _clean_shape(b, known)
            if s is None:
                raise _err(f"도형 값이 맞지 않음: {str(b)[:80]}",
                           "상자 [x, y, 너비, 높이, 클래스] · 다각형 [\"poly\", [[x, y], …], 클래스] (꼭짓점 3개 이상) · 점 [\"pt\", x, y, 클래스] · 원 [\"circle\", x, y, 반지름, 클래스] (그림 안, 0–1 비율)")
            clean.append(s)

        def run():
            it = dict(self.meta["items"].get(name) or {"note": ""})
            self._stamp(it, "manual")
            it.update(boxes=clean, label=None, by="manual" if clean else None, confirmed=bool(clean))
            if not clean:
                it.pop("boxes", None)
            if clean or it.get("note"):
                self.meta["items"][name] = it
            else:
                self.meta["items"].pop(name, None)
        self._edit([name], run)

    def _boxed(self, confirmed_only: bool) -> list:
        return [(n, self._it(n)["boxes"]) for n in self.samples
                if self._it(n).get("boxes") and (self._it(n).get("confirmed") or not confirmed_only)]

    def export_yolo(self, folder: str, confirmed_only: bool = False) -> int:
        """YOLO: folder/classes.txt + folder/labels/<그림 이름>.txt. 상자만이면 탐지 형식 (클래스번호 cx cy w h),
        다각형이 하나라도 있으면 분할 형식 (클래스번호 x1 y1 x2 y2 …, 상자는 네 꼭짓점으로). 점은 YOLO에 넣지 않음 (COCO로). 쓴 도형 수."""
        if self.schema != "boxes":
            raise _err("도형(상자·다각형·점) 작업이 아님", "")
        _guard_outside(folder, self.source.guard_root(), "YOLO 폴더")
        idx = {c["name"]: i for i, c in enumerate(self.classes)}
        stems: dict = {}
        for n, _ in self._boxed(confirmed_only):
            stems.setdefault(str(Path(n).with_suffix("")).lower(), []).append(n)
        dup = [v for v in stems.values() if len(v) > 1]
        if dup:
            raise _err(f"확장자만 다른 같은 이름의 그림: {', '.join(dup[0])}", "YOLO 라벨 파일이 서로 덮어써지므로 그림 이름을 다르게 바꾸세요")
        n_box = 0
        try:
            root = Path(folder)
            (root / "labels").mkdir(parents=True, exist_ok=True)
            (root / "classes.txt").write_text(chr(10).join(c["name"] for c in self.classes) + chr(10), encoding="utf-8")
            seg = any(_shape_kind(s) in ("poly", "circle") for _, sh in self._boxed(confirmed_only) for s in sh)
            for n, shapes in self._boxed(confirmed_only):
                f = root / "labels" / Path(n).with_suffix(".txt")
                f.parent.mkdir(parents=True, exist_ok=True)
                W, H = self.samples[n].get("width") or 1, self.samples[n].get("height") or 1
                lines = []
                for s in shapes:
                    kind = _shape_kind(s)
                    if kind == "pt":
                        continue
                    if seg:
                        pts = s[1] if kind == "poly" else _circle_poly(s[1], s[2], s[3], W, H) if kind == "circle" else \
                            [[s[0], s[1]], [s[0] + s[2], s[1]], [s[0] + s[2], s[1] + s[3]], [s[0], s[1] + s[3]]]
                        lines.append(f"{idx[s[-1]]} " + " ".join(f"{round(a, 6):g} {round(b, 6):g}" for a, b in pts) + "\n")
                    else:
                        x, y, w, h, k = s
                        lines.append(f"{idx[k]} {round(x + w / 2, 6):g} {round(y + h / 2, 6):g} {w:g} {h:g}\n")
                f.write_text("".join(lines), encoding="utf-8")
                n_box += len(lines)
        except OSError as e:
            raise _err(f"YOLO 폴더에 쓸 수 없음: {folder} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 넣으세요")
        return n_box

    def export_coco(self, file: str, confirmed_only: bool = False) -> str:
        """COCO JSON (픽셀, 그림 크기는 원본): 상자 = bbox, 다각형 = segmentation(+ 둘레 상자·넓이), 점 = keypoints(클래스마다 점 하나).
        저장한 경로."""
        if self.schema != "boxes":
            raise _err("도형(상자·다각형·점) 작업이 아님", "")
        _guard_outside(file, self.source.guard_root(), "COCO 파일")
        used_pt = {s[-1] for _, sh in self._boxed(confirmed_only) for s in sh if _shape_kind(s) == "pt"}
        cats = [dict({"id": i + 1, "name": c["name"]}, **({"keypoints": [c["name"]], "skeleton": []} if c["name"] in used_pt else {}))
                for i, c in enumerate(self.classes)]
        cid = {c["name"]: c["id"] for c in cats}
        images, anns = [], []
        for i, (n, shapes) in enumerate(self._boxed(confirmed_only), start=1):
            W, H = self.samples[n].get("width") or 1, self.samples[n].get("height") or 1
            images.append({"id": i, "file_name": n, "width": W, "height": H})
            for s in shapes:
                kind, k = _shape_kind(s), s[-1]
                a = {"id": len(anns) + 1, "image_id": i, "category_id": cid[k], "iscrowd": 0}
                if kind == "poly":
                    px = [[round(x * W, 2), round(y * H, 2)] for x, y in s[1]]
                    xs, ys = [p[0] for p in px], [p[1] for p in px]
                    a.update(segmentation=[[v for p in px for v in p]], bbox=[min(xs), min(ys), round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)],
                             area=round(_poly_area(px), 2))
                elif kind == "pt":
                    x, y = round(s[1] * W, 2), round(s[2] * H, 2)
                    a.update(keypoints=[x, y, 2], num_keypoints=1, bbox=[x, y, 0, 0], area=0)
                elif kind == "circle":                         # 원: 다각형(32각형) segmentation + 감싸는 상자, 넓이는 원 넓이
                    px = [[round(x * W, 2), round(y * H, 2)] for x, y in _circle_poly(s[1], s[2], s[3], W, H)]
                    bx = _circle_box(s[1], s[2], s[3], W, H)
                    a.update(segmentation=[[v for p in px for v in p]], bbox=[round(bx[0] * W, 2), round(bx[1] * H, 2), round(bx[2] * W, 2), round(bx[3] * H, 2)],
                             area=round(float(np.pi) * (s[3] * W) ** 2, 2))
                else:
                    x, y, w, h, _ = s
                    bb = [round(x * W, 2), round(y * H, 2), round(w * W, 2), round(h * H, 2)]
                    a.update(bbox=bb, area=round(bb[2] * bb[3], 2))
                anns.append(a)
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            Path(file).write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        except OSError as e:
            raise _err(f"COCO 파일을 저장할 수 없음: {file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 넣으세요")
        return str(file)

    # ------------------------------------------------------------ L6: 여러 사람이 나눠 하기 (파일로)
    def _value(self, it: dict):
        """정답 종류마다 비교할 값."""
        if self.schema == "segments":
            return it.get("segments") or None
        if self.schema == "boxes":
            return it.get("boxes") or None
        return it.get("label") if _has(it) else None

    def split(self, n: int) -> list:
        """샘플을 n개로 나눈 작업 파일 n개 (같은 데이터, 맡은 샘플만, 이미 붙인 정답 포함). → 파일 경로들."""
        names = sorted(self.samples)
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= min(20, len(names)):
            raise _err(f"나눌 수 {n}는 쓸 수 없음", f"2–{min(20, len(names))} 사이로 정하세요")
        stem = Path(self.file).name.replace(".dllabel", "")
        files = [str(Path(self.file).with_name(f"{stem}_part{i + 1}.dllabel")) for i in range(n)]
        exist = [f for f in files if Path(f).exists()]
        if exist:
            raise _err(f"나눈 작업 파일이 이미 있음: {exist[0]}", "그 파일들을 옮기거나 지운 뒤 다시 나누세요")
        chunks = np.array_split(np.arange(len(names)), n)
        for f, idx in zip(files, chunks):
            part = [names[i] for i in idx]
            meta = copy.deepcopy(self.meta)
            meta["assigned"] = part
            meta["items"] = {k: v for k, v in self.meta["items"].items() if k in set(part)}
            meta["split_base"] = {k: {"v": self._value(self._it(k)), "confirmed": bool(self._it(k).get("confirmed"))} for k in part}
            LabelProject._write(f, meta)
        return files

    @staticmethod
    def _write(file: str, meta: dict) -> None:
        f = Path(file)
        tmp = f.with_name(f.name + ".tmp")
        try:
            tmp.write_text(json.dumps(dict(meta, updated=datetime.now().isoformat(timespec="seconds")), ensure_ascii=False, indent=1),
                           encoding="utf-8")
            os.replace(tmp, f)
        except OSError as e:
            raise _err(f"작업 파일을 저장할 수 없음: {file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 고르세요")

    def merge(self, file: str) -> dict:
        """다른 사람의 작업 파일을 합침. 내 쪽에 정답이 없으면 가져오고, 둘 다 있고 다르면 '충돌'로 표시.
        같은 데이터·같은 정답 종류의 작업만. → {added, agreed, conflicts, agreement}."""
        try:
            other = json.loads(Path(file).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            raise _err(f"작업 파일을 읽을 수 없음: {file}", "다른 사람이 만든 .dllabel 파일의 경로를 넣으세요")
        if not isinstance(other, dict) or other.get("format") != FORMAT:
            raise _err(f"BlockLearning 라벨링 작업 파일이 아님: {file}", "")
        a, b = self.meta["source"], other.get("source", {})
        same = a.get("kind") == b.get("kind") and a.get("column", "") == b.get("column", "") and _inside(b.get("path", ""), a["path"]) \
            and _inside(a["path"], b.get("path", ""))
        if not same or other.get("schema", {}).get("kind") != self.schema:
            raise _err("다른 데이터(또는 다른 정답 종류)의 작업이라 합칠 수 없음", "같은 데이터 폴더·값 열·정답 종류로 만든 작업만 합칠 수 있습니다")
        other.setdefault("items", {})
        LabelProject._sanitize(other)
        if isinstance(other.get("split_base"), dict) and isinstance(other.get("assigned"), list):
            return self._merge_part(other)
        theirs = {n: it for n, it in other["items"].items() if n in self.samples and _has(it)}
        new_cls = [c["name"] for c in other["schema"]["classes"] if c["name"] not in {k["name"] for k in self.classes}]
        if self.schema != "number" and len(self.classes) + len(new_cls) > MAX_CLASSES:
            raise _err("합치면 클래스가 9개를 넘음", "클래스 이름을 맞춘 뒤 다시 합치세요")
        added, agreed, conflicts = [], [], []
        for n, t in theirs.items():
            mine = self._it(n)
            if not _has(mine):
                added.append(n)
            elif self._value(mine) == self._value(t):
                agreed.append(n)
            else:
                conflicts.append(n)

        def run():
            for c in new_cls if self.schema != "number" else []:
                self.classes.append({"name": c, "color": COLORS[len(self.classes) % len(COLORS)]})
            for n in added:
                t, note = theirs[n], self._it(n).get("note", "")
                self.meta["items"][n] = dict({k: v for k, v in t.items() if k != "conflict"}, note=note or t.get("note", ""))
            for n in conflicts:
                t = theirs[n]
                key = {"segments": "segments", "boxes": "boxes"}.get(self.schema, "label")
                self.meta["items"][n] = dict(self.meta["items"][n], conflict={key: self._value(t), "who": t.get("who", "")})
            for n in agreed:                               # 같은 정답: 그쪽이 확인했으면 확인 표시를 가져옴
                if theirs[n].get("confirmed") and not self._it(n).get("confirmed"):
                    self.meta["items"][n] = dict(self.meta["items"][n], confirmed=True, who=theirs[n].get("who", ""))
        self._edit(added + conflicts + agreed, run, classes=True)
        judged = len(agreed) + len(conflicts)
        return {"added": len(added), "agreed": len(agreed), "conflicts": len(conflicts),
                "agreement": round(len(agreed) / judged, 4) if judged else None}

    def _merge_part(self, other: dict) -> dict:
        """나눈 작업 되합치기 (3방향): 나눌 때 정답(split_base)과 비교해 그쪽만 바뀐 것은 그쪽 것으로(지움·확인 포함),
        나만 바뀐 것은 내 것으로, 둘 다 바뀌었는데 다르면 충돌."""
        key = {"segments": "segments", "boxes": "boxes"}.get(self.schema, "label")
        base, theirs = other["split_base"], other["items"]
        taken, agreed, conflicts = [], [], []
        for n in other["assigned"]:
            if n not in self.samples or n not in base:
                continue
            t = theirs.get(n, {})
            tv = self._value(t) if _has(t) else None
            mv = self._value(self._it(n)) if _has(self._it(n)) else None
            bv, bc = base[n].get("v"), bool(base[n].get("confirmed"))
            if tv == bv and bool(t.get("confirmed")) == bc:
                continue                                   # 그쪽은 안 바꿈
            if mv == bv:
                taken.append(n)                            # 나는 안 바꿈 → 그쪽 것
            elif tv == mv:
                agreed.append(n)
            else:
                conflicts.append(n)
        new_cls = [c["name"] for c in other["schema"]["classes"] if c["name"] not in {k["name"] for k in self.classes}]
        if self.schema != "number" and len(self.classes) + len(new_cls) > MAX_CLASSES:
            raise _err("합치면 클래스가 9개를 넘음", "클래스 이름을 맞춘 뒤 다시 합치세요")

        def run():
            for c in new_cls if self.schema != "number" else []:
                self.classes.append({"name": c, "color": COLORS[len(self.classes) % len(COLORS)]})
            for n in taken:
                t, note = theirs.get(n, {}), self._it(n).get("note", "")
                if _has(t):
                    self.meta["items"][n] = dict({k: v for k, v in t.items() if k != "conflict"}, note=note or t.get("note", ""))
                elif note:
                    self.meta["items"][n] = {"label": None, "by": None, "confirmed": False, "note": note}
                else:
                    self.meta["items"].pop(n, None)
            for n in agreed:
                if theirs[n].get("confirmed"):
                    self.meta["items"][n] = dict(self.meta["items"][n], confirmed=True, who=theirs[n].get("who", ""))
            for n in conflicts:
                t = theirs.get(n, {})
                self.meta["items"][n] = dict(self.meta["items"].get(n) or {"note": ""},
                                             conflict={key: self._value(t) if _has(t) else None, "who": t.get("who", "")})
        self._edit(taken + agreed + conflicts, run, classes=True)
        judged = len(agreed) + len(conflicts)
        return {"added": len(taken), "agreed": len(agreed), "conflicts": len(conflicts),
                "agreement": round(len(agreed) / judged, 4) if judged else None}

    def resolve(self, name: str, take: str) -> None:
        """충돌 풀기: take "mine"(내 정답 유지) 또는 "theirs"(그쪽 정답으로, 사람 결정이므로 manual)."""
        self._need(name)
        cf = self._it(name).get("conflict")
        if not cf:
            raise _err(f"{name}에는 충돌이 없음", "")
        if take not in ("mine", "theirs"):
            raise _err(f"고르는 값 '{take}'을(를) 모름", "mine 또는 theirs")

        def run():
            it = dict(self.meta["items"][name])
            it.pop("conflict", None)
            if take == "theirs":                           # 그쪽 값도 정답 종류 규칙으로 다시 검사
                key = {"segments": "segments", "boxes": "boxes"}.get(self.schema, "label")
                names_ok = {c["name"] for c in self.classes}
                v = _clean_value(self.schema, cf.get(key), names_ok)
                if self.schema == "class" and v is not None and v not in names_ok:
                    v = None
                it[key] = v
                if v is None:
                    it.update(by=None, confirmed=False)
                    it.pop(key, None) if key != "label" else None
                else:
                    it.update(by="manual", confirmed=True)
                it.pop("p", None)
            if self.who:
                it["who"] = self.who
            self.meta["items"][name] = it
        self._edit([name], run)

    def export(self, file: str, confirmed_only: bool = False) -> int:
        """정답 표 CSV (file,label, UTF-8 BOM: Excel에서 한글이 깨지지 않게). 쓴 행 수."""
        if self.schema == "segments":
            return self.export_segments(file, confirmed_only)
        if self.schema == "boxes":
            raise _err("상자 작업은 정답 표 CSV로 내보내지 않음", "YOLO 또는 COCO로 내보내세요")
        _guard_outside(file, self.source.guard_root(), "정답 표")
        lab = self.labels(confirmed_only)
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "label"])
                for n, v in lab.items():
                    w.writerow([n, v])
        except OSError as e:
            raise _err(f"정답 표를 저장할 수 없음: {file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 넣으세요")
        return len(lab)

    def export_training(self, file: str, confirmed_only: bool = False) -> int:
        """표 작업: 정답 있는 행만 숫자 열 + 마지막 열 label로 (studio CSV 불러오기 형식). 쓴 행 수."""
        if self.source.kind != "table":
            raise _err("표 작업이 아님", "")
        _guard_outside(file, self.source.guard_root(), "학습용 CSV")
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            return self.source.write_training(file, self.labels(confirmed_only))
        except OSError as e:
            raise _err(f"학습용 CSV를 저장할 수 없음: {file} ({type(e).__name__})", "쓸 수 있는 폴더의 경로를 넣으세요")

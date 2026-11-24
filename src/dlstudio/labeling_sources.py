"""라벨링 작업실의 데이터 종류: 샘플 목록·요약값(규칙·정렬·묶기용 숫자)·썸네일·크게 보기.

- series_folder: 샘플마다 CSV 하나인 폴더 (값 열 하나)
- image_folder: 그림 폴더 (맨 위 + 하위 폴더 한 단계)
- table: CSV 한 파일, 한 행 = 샘플
화면·HTTP·정답은 모른다 (labeling.py가 씀). 데이터 파일은 읽기만 한다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dlstudio.data import (DataError, Issue, _fit_length, _gray01, _num, _numeric_columns, _read_table,
                              _series_files, _series_key, _series_key_img, _series_values, image_files)

THUMB = 120                # 시계열 썸네일 점 수
IMG_THUMB = 24             # 그림 썸네일 (24×24 흑백)
DETAIL_MAX = 2000          # 크게 보기: 이보다 길면 구간별 최소·최대로 줄임
SERIES_STATS = ("length", "min", "max", "mean", "std", "first", "last", "range", "argmax_frac")
IMAGE_STATS = ("mean", "std", "min", "max", "width", "height")


def _err(message: str, hint: str = "") -> DataError:
    return DataError([Issue(None, None, message, hint)])


def _series_stats(v: np.ndarray) -> dict:
    return {"length": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "std": float(v.std()), "first": float(v[0]), "last": float(v[-1]), "range": float(v.max() - v.min()),
            "argmax_frac": float(np.argmax(v) / max(1, len(v) - 1))}


def downsample(v: np.ndarray, limit: int = DETAIL_MAX) -> list:
    """길면 limit/2 구간마다 [최소, 최대] (모양과 튀는 값을 지킴)."""
    if len(v) <= limit:
        return [float(x) for x in v]
    out = []
    for chunk in np.array_split(v, limit // 2):
        lo, hi = int(np.argmin(chunk)), int(np.argmax(chunk))
        out += [float(chunk[min(lo, hi)]), float(chunk[max(lo, hi)])]
    return out


def _sample(path=None, stats=None, thumb=None, error=None, **extra) -> dict:
    return dict({"path": path, "stats": stats, "thumb": thumb, "error": error}, **extra)


class SeriesSource:
    kind = "series_folder"
    key = staticmethod(_series_key)

    def __init__(self, src: dict):
        self.path, self.column = str(src["path"]), str(src.get("column", ""))
        self.stat_names = SERIES_STATS

    def guard_root(self) -> str:                         # 작업 파일·내보내기를 두면 안 되는 곳
        return self.path

    def scan(self) -> tuple:
        """→ (이름 → 샘플, 이름 → 처음 정답(하위 폴더 이름)). 값 열을 못 읽는 샘플은 error만."""
        files, mode, _ = _series_files(self.path)
        samples = {}
        for n, f, _ in files:
            try:
                v = _series_values(n, f, self.column)
                samples[n] = _sample(f, _series_stats(v), [round(float(x), 5) for x in _fit_length(v, THUMB, "interp")])
            except DataError as e:
                samples[n] = _sample(f, error=e.issues[0].message)
        return samples, ({n: c for n, _, c in files} if mode == "folders" else {})

    def values(self, name: str, s: dict) -> np.ndarray:
        return _series_values(name, s["path"], self.column)

    def detail(self, name: str, s: dict) -> dict:
        v = self.values(name, s)
        return {"values": downsample(v), "n": int(len(v))}


class ImageSource:
    kind = "image_folder"
    key = staticmethod(_series_key_img)

    def __init__(self, src: dict):
        self.path = str(src["path"])
        self.stat_names = IMAGE_STATS

    def guard_root(self) -> str:
        return self.path

    @staticmethod
    def _pil():
        try:
            from PIL import Image
            return Image
        except ImportError:
            raise _err("그림을 읽으려면 Pillow가 필요함", "명령창에서 pip install pillow 를 실행한 뒤 다시 여세요")

    def scan(self) -> tuple:
        Image = self._pil()
        files = image_files(self.path)
        if not files:
            raise _err(f"그림이 없음: {self.path}", "png·jpg·jpeg·bmp가 든 폴더를 고르세요")
        samples = {}
        for n, f, _ in files:
            try:
                with Image.open(f) as im:
                    w, h = im.size
                    g = _gray01(im, 64, Image)
                    t = _gray01(im, IMG_THUMB, Image)
                samples[n] = _sample(f, {"mean": float(g.mean()), "std": float(g.std()), "min": float(g.min()),
                                         "max": float(g.max()), "width": int(w), "height": int(h)},
                                     [round(float(x), 3) for x in t.reshape(-1)], width=int(w), height=int(h))
            except Exception:
                samples[n] = _sample(f, error=f"그림을 읽을 수 없음: {n}")
        subs = {c for _, _, c in files if c}
        return samples, ({n: c for n, _, c in files if c} if len(subs) >= 2 else {})

    def detail(self, name: str, s: dict) -> dict:
        return {"image": True, "width": s.get("width"), "height": s.get("height"), "n": 0, "values": []}

    def file(self, s: dict) -> Path:
        return Path(s["path"])


class TableSource:
    kind = "table"

    def __init__(self, src: dict):
        self.path, self.id_col = str(src["path"]), (src.get("id_col") or None)
        self.stat_names: tuple = ()
        self.header: list = []
        self.rows: dict = {}

    @staticmethod
    def key(s: str) -> str:
        return s.strip().lower()

    def guard_root(self) -> str:                         # 표는 데이터 파일 하나: 그 파일만 막음
        return self.path

    def scan(self) -> tuple:
        if not Path(self.path).is_file():
            raise _err(f"표 파일을 찾을 수 없음: {self.path}", "CSV 파일의 전체 경로를 넣으세요")
        names, rows = _read_table(self.path)
        if not rows:
            raise _err(f"표에 행이 없음: {self.path}", "한 행 = 샘플 하나인 CSV를 고르세요")
        nums = [c for c in _numeric_columns(names, rows) if c != self.id_col]
        if not nums:
            raise _err("숫자 열이 없음", "측정값이 숫자로 든 열이 있어야 합니다")
        if self.id_col and self.id_col not in names:
            raise _err(f"ID 열 '{self.id_col}'이(가) 없음", "표의 머리글에 있는 열을 고르세요")
        self.stat_names, self.header = tuple(nums), list(names)
        idx = {c: names.index(c) for c in nums}
        vals = np.full((len(rows), len(nums)), np.nan)
        samples, seen = {}, {}
        for r, (rn, cells) in enumerate(rows):
            if self.id_col:
                n = cells[names.index(self.id_col)].strip() if names.index(self.id_col) < len(cells) else ""
                if not n:
                    raise _err(f"{rn}행의 ID 칸이 비어 있음", "ID 열의 모든 칸을 채우거나 ID 열 없이 만드세요")
                if n in seen:
                    raise _err(f"ID '{n}'이(가) {seen[n]}행과 {rn}행에 겹침", "ID 열 값은 행마다 달라야 합니다")
                seen[n] = rn
            else:
                n = f"행 {r + 1}"
            self.rows[n] = (rn, cells)
            bad = [c for c in nums if idx[c] >= len(cells) or _num(cells[idx[c]]) is None]
            if bad:
                samples[n] = _sample(error=f"{rn}행 '{bad[0]}' 칸이 숫자가 아님")
                continue
            vals[r] = [_num(cells[idx[c]]) for c in nums]
            samples[n] = _sample(stats={c: float(vals[r][j]) for j, c in enumerate(nums)}, row=rn)
        ok = ~np.isnan(vals).any(1)
        mu, sd = vals[ok].mean(0), vals[ok].std(0) + 1e-9   # 썸네일: 열마다 전체 기준 표준화한 막대
        for r, n in enumerate(samples):
            if samples[n]["stats"]:
                samples[n]["thumb"] = [round(float(x), 3) for x in (vals[r] - mu) / sd]
        return samples, {}

    def detail(self, name: str, s: dict) -> dict:
        rn, cells = self.rows[name]
        return {"row": {c: (cells[j] if j < len(cells) else "") for j, c in enumerate(self.header)}, "line": rn,
                "n": 0, "values": []}

    def write_training(self, file: str, labels: dict) -> int:
        """정답 있는 행만: 숫자 열 + 마지막 열 label (studio CSV 불러오기 형식)."""
        idx = [self.header.index(c) for c in self.stat_names]
        with open(file, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(self.stat_names) + ["label"])
            for n, lab in labels.items():
                _, cells = self.rows[n]
                w.writerow([cells[j] for j in idx] + [lab])
        return len(labels)


KINDS = {"series_folder": SeriesSource, "image_folder": ImageSource, "table": TableSource}


def make_source(src: dict):
    kind = src.get("kind")
    if kind not in KINDS:
        raise _err(f"이 작업실 버전에서는 못 여는 데이터 종류: {kind}", "시계열 폴더·이미지 폴더·표(CSV) 작업만 열 수 있습니다")
    return KINDS[kind](src)

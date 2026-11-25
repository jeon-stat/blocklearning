"""데이터: CSV(한 행 = 샘플 하나, 열 = 시점·특징, 정답 열 하나) 또는 합성 예시 데이터 → Dataset.

오류는 DataError(issues)로 모아서 알린다. 각 Issue는 파일 기준 행·열(1부터)과 쉬운 말 설명·고칠 방법을 가진다.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MAX_CLASSES = 20          # 정답이 정수이고 종류가 이 이하면 분류로 봄
MIN_SAMPLES = 10
MAX_ISSUES = 20           # 오류는 이만큼만 보여 줌
IMAGE_SIZES = (8, 16, 28, 32)   # 폴더 이미지를 맞출 크기 (정사각형)
MAX_IMAGES = 20000        # 폴더 이미지 한도 (메모리에 한 번에 올림)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
SERIES_MAX_FILES = 20000  # 시계열 폴더: 파일(샘플) 한도
SERIES_MAX_ROWS = 200000  # 시계열 폴더: 한 파일 행 한도
SERIES_MAX_LEN = 65536    # 맞출 길이 한도 (채널 하나 한도와 같음)


@dataclass
class Issue:
    row: Optional[int]
    col: Optional[int]
    message: str
    hint: str

    def to_dict(self) -> dict:
        return {"row": self.row, "col": self.col, "message": self.message, "hint": self.hint}


class DataError(Exception):
    def __init__(self, issues: list):
        self.issues = issues
        super().__init__("; ".join(i.message for i in issues[:3]))


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    task: str
    classes: list
    length: int
    mean: np.ndarray
    std: np.ndarray
    source: dict = field(default_factory=dict)
    features: list = field(default_factory=list)      # 열 이름 (머리글이 없으면 빈 목록)
    image: Optional[tuple] = None                      # 이미지면 (H, W): X는 H·W로 펼친 픽셀
    det: Optional[dict] = None                         # 탐지면 {"grid": 격자, "train"·"test": 그림마다 [[클래스 번호, x, y, w, h], …]}
    extra: Optional[dict] = None                       # 예측: {"horizon", "stride", "column", …} · 이상 탐지: {"normal", "others", …}

    @property
    def n_out(self) -> int:
        """모델 출력 크기: 분류면 클래스 수, 회귀면 1, 탐지면 격자² × (5 + 클래스 수) (DetOut: 격자를 함께 지님),
        예측이면 다음 H개, 이상 탐지면 입력 값 수 (입력을 되살림)."""
        if self.task == "detection":
            from dlstudio.detect import DetOut
            return DetOut(self.det["grid"], len(self.classes))
        if self.task == "forecast":
            return int(self.y_train.shape[1])
        if self.task == "anomaly":
            return int(self.X_train.shape[1])
        return len(self.classes) if self.task == "classification" else 1

    @property
    def target_train(self) -> np.ndarray:
        """학습의 맞힐 값: 이상 탐지는 입력 그대로 (정규화한 값), 나머지는 정답."""
        return self.X_train if self.task == "anomaly" else self.y_train

    def label_text(self, part: str, i: int):
        """샘플 i의 정답 (화면용): 분류·이상 탐지는 클래스 이름, 회귀는 값, 탐지는 상자 수, 예측은 다음 값들."""
        if self.task == "detection":
            return f"상자 {len(self.det[part][i])}개"
        y = self.y_train if part == "train" else self.y_test
        if self.task == "forecast":
            return f"다음 {y.shape[1]}개 (첫 값 {float(y[i][0]):.4g})"
        if self.task == "anomaly":
            return self.classes[int(y[i])]
        return self.classes[int(y[i])] if self.task == "classification" else float(y[i])

    def raw(self, X: np.ndarray) -> np.ndarray:
        """정규화를 되돌린 값 (그림용)."""
        return X * self.std + self.mean

    def _predict_kinds(self) -> list:
        from dlstudio.predict import new_data_kinds
        return new_data_kinds(self)

    def summary(self) -> dict:
        if self.task == "detection":
            return self._det_summary()
        if self.task == "forecast":
            return self._fc_summary()
        y_all = np.concatenate([self.y_train, self.y_test])
        if self.task in ("classification", "anomaly"):
            counts = {name: int((y_all == k).sum()) for k, name in enumerate(self.classes)}
            order, seen = [], set()
            for k in range(len(self.classes)):                 # 클래스마다 돌아가며 최대 6개
                idx = np.flatnonzero(self.y_train == k)[:3]
                order.append(list(idx))
            picks = [i for grp in zip(*[g + [None] * (3 - len(g)) for g in order]) for i in grp if i is not None]
            picks = [i for i in picks if not (i in seen or seen.add(i))][:6]
            label = lambda i: self.classes[int(self.y_train[i])]
            if self.task == "anomaly":                          # 학습에는 정상만: 미리보기는 시험의 이상도 함께
                return self._ae_summary(counts)
        else:
            counts = {}
            picks = list(range(min(6, len(self.y_train))))
            label = lambda i: f"{float(self.y_train[i]):.4g}"
        raw = self.raw(self.X_train)
        y_range = None if self.task == "classification" else [float(np.min(y_all)), float(np.max(y_all))]
        return {"n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)), "length": int(self.length),
                "task": self.task, "classes": list(self.classes), "class_counts": counts, "target_range": y_range,
                "features": [] if self.image else (list(self.features) or [f"열 {i}" for i in range(1, self.length + 1)]),
                "image": list(self.image) if self.image else None,
                "example": (self.source or {}).get("example"),            # 예시 데이터면 그 이름 (화면에서 ✓ 표시)
                "predict_kinds": self._predict_kinds(),
                "feature_means": None if self.image or self.length > 40 else
                [round(float(v), 4) for v in self.raw(self.X_train).mean(0)],    # ⑤ 직접 넣기 칸의 처음 값
                "preview": [{"label": label(i), "values": [round(float(v), 5) for v in raw[i]]} for i in picks]}

    def _base_summary(self) -> dict:
        return {"n_train": int(len(self.X_train)), "n_test": int(len(self.X_test)), "length": int(self.length),
                "task": self.task, "classes": list(self.classes), "image": list(self.image) if self.image else None,
                "features": [] if self.image else (list(self.features) or [f"열 {i}" for i in range(1, self.length + 1)]),
                "example": (self.source or {}).get("example"), "predict_kinds": self._predict_kinds()}

    def _fc_summary(self) -> dict:
        e, y = self.extra, np.concatenate([self.y_train, self.y_test])
        raw = self.raw(self.X_train)
        picks = sorted({int(round(i)) for i in np.linspace(0, len(raw) - 1, 6)})
        return dict(self._base_summary(), class_counts={}, feature_means=None, features=[],
                    target_range=[float(y.min()), float(y.max())],
                    forecast={k: e[k] for k in ("horizon", "stride", "column", "columns", "n_points")},
                    preview=[{"label": f"→ 다음 {e['horizon']}개", "values": [round(float(v), 5) for v in raw[i]],
                              "future": [round(float(v), 5) for v in self.y_train[i]]} for i in picks])

    def _ae_summary(self, counts: dict) -> dict:
        raw_tr, raw_te = self.raw(self.X_train), self.raw(self.X_test)
        ab = [int(i) for i in np.flatnonzero(self.y_test == 1)[:3]]
        items = [(raw_tr[i], self.classes[0]) for i in range(min(6 - len(ab), len(raw_tr)))] + [(raw_te[i], "이상") for i in ab]
        e = self.extra
        return dict(self._base_summary(), class_counts=counts, target_range=None,
                    feature_means=None if self.image or self.length > 40 else [round(float(v), 4) for v in raw_tr.mean(0)],
                    anomaly={"normal": e["normal"], "others": e["others"], "n_anomaly": e["n_anomaly"],
                             "train_normal": int(len(self.X_train))},
                    preview=[{"label": lab, "values": [round(float(v), 5) for v in x]} for x, lab in items])

    def _det_summary(self) -> dict:
        boxes = self.det["train"] + self.det["test"]
        counts = {name: sum(1 for bs in boxes for b in bs if b[0] == k) for k, name in enumerate(self.classes)}
        raw = self.raw(self.X_train[:6])
        return {"n_train": int(len(self.X_train)), "n_test": int(len(self.X_test)), "length": int(self.length),
                "task": "detection", "classes": list(self.classes), "class_counts": counts, "target_range": None,
                "features": [], "image": list(self.image), "example": (self.source or {}).get("example"),
                "predict_kinds": self._predict_kinds(), "feature_means": None,
                "detection": {"grid": int(self.det["grid"]), "n_boxes": sum(len(b) for b in boxes),
                              "no_box": sum(1 for b in boxes if not b)},
                "preview": [{"label": self.label_text("train", i), "values": [round(float(v), 5) for v in raw[i]],
                             "boxes": self.det["train"][i]} for i in range(len(raw))]}


def _num(s: str) -> Optional[float]:
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _is_header(first: list, feats: list, lc: int, rest: list) -> bool:
    """첫 줄이 머리글인지: 값 칸에 숫자가 아닌 것이 있거나, 정답 칸만 글자인데 다른 행의 정답은 숫자이거나,
    pandas 기본 머리글처럼 0, 1, 2, … (또는 1, 2, …)로 이어지는 정수."""
    if any(first[j] and _num(first[j]) is None for j in feats):
        return True
    label_others = [r[lc] for _, r in rest[:20] if len(r) > lc]
    if first[lc] and _num(first[lc]) is None and label_others and all(_num(v) is not None for v in label_others):
        return True
    vals = [_num(first[j]) for j in feats]
    if all(v is not None and float(v).is_integer() for v in vals) and len(vals) >= 3:
        ints = [int(v) for v in vals]
        if ints == list(range(ints[0], ints[0] + len(ints))) and ints[0] in (0, 1):
            return True
    return False


def _delimiter(first_line: str) -> str:
    for d in (",", ";", "\t"):
        if d in first_line:
            return d
    return ","


def _read_rows(path) -> list:
    """CSV → [(행 번호, [칸...])] (빈 줄 제외). 인코딩은 UTF-8 → cp949 순서, 구분자는 첫 줄에서."""
    raw = Path(path).read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp949"):                     # 한국어 Excel의 "CSV (쉼표로 분리)"는 cp949
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DataError([Issue(None, None, f"{Path(path).name}: 파일의 글자 인코딩을 읽을 수 없음",
                               "Excel이면 'CSV UTF-8 (쉼표로 분리)'로 다시 저장하세요")])
    lines = text.splitlines()
    if not lines:
        return []
    rows = csv.reader(lines, delimiter=_delimiter(lines[0]))
    return [(i + 1, [c.strip() for c in r]) for i, r in enumerate(rows) if any(c.strip() for c in r)]


def _read_table(path) -> tuple:
    """CSV → (열 이름, 값 행). 머리글 판단: 첫 줄은 글자인데 그 아래(앞 20행)는 숫자인 열이 있으면 머리글.
    (시각 같은 글자 열은 아래도 글자라 증거가 아님.) 머리글이 없으면 열 이름은 '열 1, 열 2, …'."""
    rows = _read_rows(path)
    if not rows:
        return [], []
    first, rest = rows[0][1], [r for _, r in rows[1:21]]
    if rest:
        def numeric_below(j):
            cells = [r[j] for r in rest if j < len(r) and r[j]]
            return bool(cells) and sum(_num(c) is not None for c in cells) >= 0.9 * len(cells)
        header = any(c and _num(c) is None and numeric_below(j) for j, c in enumerate(first))
    else:
        header = any(c and _num(c) is None for c in first)
    if header:
        return [c or f"열 {j + 1}" for j, c in enumerate(first)], rows[1:]
    return [f"열 {j + 1}" for j in range(len(first))], rows


def load_csv(path: str, label_col: Optional[int] = None, task: str = "auto", normalize: bool = True,
             test_ratio: float = 0.2, seed: int = 0, image_width: Optional[int] = None) -> Dataset:
    """CSV → Dataset. label_col은 0부터 (None이면 마지막 열). 첫 줄의 특징 칸에 숫자가 아닌 것이 있으면 머리글로 건너뜀.
    image_width: None이면 이미지가 아님. 0이면 픽셀 수가 제곱수일 때 정사각형, 아니면 그 너비 (높이 = 픽셀 수 ÷ 너비)."""
    numbered = _read_rows(path)
    if not numbered:
        raise DataError([Issue(None, None, "파일이 비어 있음", "샘플이 한 행에 하나씩 있는 CSV를 고르세요")])
    ncol = len(numbered[0][1])
    if ncol < 2:
        raise DataError([Issue(1, None, "열이 하나뿐임 (값 열과 정답 열이 모두 있어야 함)",
                               "쉼표로 나뉜 CSV인지, 정답 열이 있는지 확인하세요")])
    lc = ncol - 1 if label_col is None else label_col
    if not 0 <= lc < ncol:
        raise DataError([Issue(None, None, f"정답 열 번호 {lc + 1}이(가) 없음 (열 {ncol}개)", "정답 열을 다시 고르세요")])
    feats = [j for j in range(ncol) if j != lc]
    first = numbered[0][1]
    names: list = []
    if _is_header(first, feats, lc, numbered[1:]):
        names = [first[j] or f"열 {j + 1}" for j in feats]
        numbered = numbered[1:]

    issues, X, labels = [], [], []
    for rn, r in numbered:
        if len(r) != ncol:
            issues.append(Issue(rn, None, f"{rn}행: 칸 수가 {len(r)}개 (다른 행은 {ncol}개)",
                                "빠지거나 남는 쉼표가 있는지 확인하세요"))
            continue
        vals = []
        for j in feats:
            if r[j] == "":
                issues.append(Issue(rn, j + 1, f"{rn}행 {j + 1}열이 비어 있음", "빈 칸을 채우거나 그 행을 지우세요"))
                vals.append(np.nan)
                continue
            v = _num(r[j])
            if v is None:
                issues.append(Issue(rn, j + 1, f"{rn}행 {j + 1}열 '{r[j][:20]}'은(는) 숫자가 아님",
                                    "값 열에는 숫자만 있어야 합니다"))
                v = np.nan
            vals.append(v)
        if r[lc] == "":
            issues.append(Issue(rn, lc + 1, f"{rn}행의 정답 칸이 비어 있음", "정답을 채우거나 그 행을 지우세요"))
        X.append(vals)
        labels.append(r[lc])
        if len(issues) >= MAX_ISSUES:
            break
    if issues:
        raise DataError(issues[:MAX_ISSUES])
    src = {"path": str(path), "label_col": lc, "normalize": normalize, "test_ratio": test_ratio, "seed": seed, "task": task}
    X = np.asarray(X, dtype=np.float64)
    image = None
    if image_width is not None:
        image = _csv_image_shape(X.shape[1], int(image_width))
        if X.max() > 1:                                    # 0–255 같은 픽셀 → 0–1
            X = X / X.max()
        src["image_width"] = int(image_width)
        names = []
    return make_dataset(X, np.asarray(labels, dtype=object), task=task,
                        normalize=normalize, test_ratio=test_ratio, seed=seed, source=src, features=names, image=image)


def _csv_image_shape(n: int, width: int) -> tuple:
    if width == 0:
        side = int(round(n ** 0.5))
        if side * side != n:
            raise DataError([Issue(None, None, f"픽셀 {n}개는 정사각형이 아님", "이미지 너비(가로 픽셀 수)를 넣으세요")])
        return side, side
    if width < 1 or n % width:
        raise DataError([Issue(None, None, f"픽셀 {n}개를 너비 {width}로 나눌 수 없음",
                               "이미지 너비(가로 픽셀 수)를 확인하세요. 비우면 정사각형으로 봅니다")])
    return n // width, width


def _gray01(im, size: int, Image) -> np.ndarray:
    """그림 → 흑백 0–1 (size×size). 16비트·실수 그림은 비트 수에 맞춰 나누고, 투명한 곳은 흰 배경으로."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im, dtype=np.float64)
        top = 65535.0 if im.mode.startswith("I;16") or a.max() > 255 else 255.0
        if im.mode == "F" and a.max() <= 1:
            top = 1.0
        g = Image.fromarray((np.clip(a / top, 0, 1)).astype(np.float32), "F")
        return np.asarray(g.resize((size, size)), dtype=np.float64)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return np.asarray(im.convert("L").resize((size, size)), dtype=np.float64) / 255.0


# ---------------------------------------------------------------- 샘플마다 CSV 하나인 폴더 (시계열)

def _series_files(path: str, label_file: Optional[str] = None) -> tuple:
    """폴더 → ([(이름, 파일, 클래스 또는 None)], 방식 "folders"|"file"|"none", 건너뛴 것 설명)."""
    root = Path(path)
    if not root.is_dir():
        raise DataError([Issue(None, None, f"폴더를 찾을 수 없음: {path}", "샘플마다 CSV 하나가 든 폴더의 전체 경로를 넣으세요")])
    if label_file and not Path(label_file).is_file():
        raise DataError([Issue(None, None, f"정답 표를 찾을 수 없음: {label_file}", "정답 표 CSV(파일이름, 정답)의 전체 경로를 넣으세요")])
    lab = Path(label_file).resolve() if label_file else None
    junk = lambda n: n.startswith(".") or n == "__MACOSX"
    is_csv = lambda f: f.is_file() and not junk(f.name) and f.suffix.lower() == ".csv" and f.resolve() != lab
    entries = [e for e in root.iterdir() if not junk(e.name)]
    top = sorted((f for f in entries if is_csv(f)), key=lambda f: f.name)
    subs = {d.name: sorted((f for f in d.iterdir() if is_csv(f)), key=lambda f: f.name)
            for d in sorted((e for e in entries if e.is_dir()), key=lambda d: d.name)}
    subs = {k: v for k, v in subs.items() if v}
    others = sum(1 for f in entries if f.is_file() and not is_csv(f) and f.resolve() != lab)
    others += sum(1 for d in entries if d.is_dir() for f in d.iterdir() if f.is_file() and not is_csv(f))
    skipped = [f"CSV가 아닌 파일 {others}개는 건너뜀"] if others else []
    if label_file:                                         # 정답 표: 맨 위 + 하위 폴더 CSV (이름 = 폴더/파일)
        mode = "file"
        files = [(f.name, f, None) for f in top] + [(f"{c}/{f.name}", f, None) for c, fs in subs.items() for f in fs]
    elif len(subs) >= 2:
        mode = "folders"
        files = [(f"{c}/{f.name}", f, c) for c, fs in subs.items() for f in fs]
        if top:
            skipped.append(f"맨 위의 CSV {len(top)}개는 쓰지 않음 (클래스 하위 폴더 방식)")
    else:
        mode = "none"
        files = [(f.name, f, None) for f in top] + [(f"{c}/{f.name}", f, None) for c, fs in subs.items() for f in fs]
    if not files:
        more = f" ({skipped[0]})" if skipped else ""
        raise DataError([Issue(None, None, f"CSV 파일이 없음{more}", "샘플(사이클·환자 등)마다 CSV 하나가 든 폴더를 고르세요")])
    if len(files) > SERIES_MAX_FILES:
        raise DataError([Issue(None, None, f"CSV가 {len(files):,}개 (한도 {SERIES_MAX_FILES:,}개)", "파일 수를 줄이거나 나눠서 쓰세요")])
    return files, mode, skipped


def _numeric_columns(names: list, rows: list) -> list:
    """앞 20행 중 숫자로 읽히는 칸이 90% 이상인 열."""
    head = rows[:20]
    out = []
    for j, n in enumerate(names):
        cells = [r[j] for _, r in head if j < len(r)]
        if cells and sum(_num(c) is not None for c in cells) >= 0.9 * len(head):
            out.append(n)
    return out


def scan_series_folder(path: str, label_file: Optional[str] = None) -> dict:
    """폴더 살펴보기: 파일 수, 길이(행 수) [최소, 중앙값, 최대], 숫자 열 후보, 기본 열·길이, 정답 방식."""
    files, mode, skipped = _series_files(path, label_file)
    tables = [(n, _read_table(f)) for n, f, _ in files]    # 파일마다 한 번만 읽음
    with_nums = [(n, t) for n, t in tables if _numeric_columns(*t)]
    no_nums = [n for n, t in tables if not _numeric_columns(*t)]
    if not with_nums:
        raise DataError([Issue(None, None, f"숫자 열이 있는 CSV가 없음 ({files[0][0]} 등)", "값(측정값)이 숫자로 든 열이 있어야 합니다")])
    if no_nums:
        shown = ", ".join(no_nums[:3]) + (" …" if len(no_nums) > 3 else "")
        skipped.append(f"숫자 열이 없는 CSV {len(no_nums)}개: {shown} (정답 표라면 '정답 표 CSV' 칸에 넣으세요)")
    names, rows = with_nums[0][1]
    cols = _numeric_columns(names, rows)
    lengths = [len(t[1]) for _, t in with_nums]
    med = float(np.median(lengths))
    head = rows[:20]                                       # 기본 값 열: 값이 변하는 숫자 열 중 마지막 (번호·ID 열은 건너뜀)
    varying = [c for c in cols if len({r[names.index(c)] for _, r in head if names.index(c) < len(r)}) > 1]
    return {"n_files": len(files), "lengths": [int(min(lengths)), med, int(max(lengths))], "columns": cols,
            "default_column": (varying or cols)[-1], "default_length": min(SERIES_MAX_LEN, max(8, int(med) // 8 * 8)), "mode": mode,
            "classes": sorted({c for _, _, c in files}) if mode == "folders" else None, "skipped": skipped}


def _series_values(name: str, path, column: str) -> np.ndarray:
    return _series_matrix(name, path, [column])[0]


def _series_matrix(name: str, path, columns: list) -> np.ndarray:
    """CSV 한 번 읽어 여러 값 열 → (열 수, 길이). 빈 칸은 앞뒤 값으로 메움."""
    names, rows = _read_table(path)
    for column in columns:
        if column not in names:
            raise DataError([Issue(None, None, f"{name}에 열 '{column}'이(가) 없음", "모든 파일의 머리글(열 이름)을 같게 하세요")])
    if not rows:
        raise DataError([Issue(None, None, f"{name}: 값이 없음", "빈 파일을 지우세요")])
    if len(rows) > SERIES_MAX_ROWS:
        raise DataError([Issue(None, None, f"{name}: {len(rows):,}행 (한도 {SERIES_MAX_ROWS:,}행)", "파일을 나누거나 줄이세요")])
    return np.stack([_column_values(name, names, rows, c) for c in columns])


def _column_values(name: str, names: list, rows: list, column: str) -> np.ndarray:
    j = names.index(column)
    v = np.empty(len(rows))
    for k, (rn, r) in enumerate(rows):
        cell = r[j] if j < len(r) else ""
        if cell == "":
            v[k] = np.nan
            continue
        x = _num(cell)
        if x is None:
            raise DataError([Issue(rn, j + 1, f"{name} {rn}행 '{cell[:20]}'은(는) 숫자가 아님", "값 열을 다시 고르거나 그 행을 고치세요")])
        v[k] = x
    bad = np.isnan(v)
    if bad.mean() > 0.3:
        raise DataError([Issue(None, None, f"{name}: 값의 {bad.mean():.0%}가 비어 있음", "빈 칸이 30% 이하여야 합니다")])
    if bad.any():                                          # 빈 칸은 앞뒤 값으로 메움
        idx = np.arange(len(v))
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def _fit_length(v: np.ndarray, length: int, resample: str) -> np.ndarray:
    if resample == "crop":                                 # 앞에서 자르고 모자라면 마지막 값으로 채움
        return np.concatenate([v[:length], np.full(max(0, length - len(v)), v[-1])])
    if len(v) == 1:
        return np.full(length, v[0])
    return np.interp(np.linspace(0, len(v) - 1, length), np.arange(len(v)), v)


def _series_key(s: str) -> str:
    """정답 표와 파일 이름 맞추기: 앞뒤 공백·대소문자·끝의 .csv를 무시."""
    s = s.strip().lower()
    return s[:-4] if s.endswith(".csv") else s


def _label_table(label_file: str, keys: set, key=None) -> dict:
    """정답 표 → {파일 키: 정답}. 첫 행의 첫 칸이 폴더 안 파일이면 데이터, 아니면 머리글.
    정답 칸이 비었거나 같은 파일에 정답이 둘이면 행 번호와 함께 오류."""
    key = key or _series_key
    rows = _read_rows(label_file)
    name = Path(label_file).name
    if rows and key(rows[0][1][0]) not in keys:
        rows = rows[1:]
    if not rows or max(len(r) for _, r in rows) < 2:
        raise DataError([Issue(None, None, f"정답 표 {name}에 열이 2개 이상 있어야 함", "첫 열 = 파일 이름, 둘째 열 = 정답")])
    table, where = {}, {}
    for rn, r in rows:
        if not r[0].strip():
            continue
        k = key(r[0])
        v = r[1].strip() if len(r) > 1 else ""
        if not v:
            raise DataError([Issue(rn, 2, f"정답 표 {name} {rn}행: '{r[0].strip()}'의 정답 칸이 비어 있음", "정답을 채우거나 그 행을 지우세요")])
        if k in table and table[k] != v:
            raise DataError([Issue(rn, 2, f"정답 표 {name}에 '{r[0].strip()}'의 정답이 두 가지 ({table[k]}, {v}): {where[k]}행, {rn}행",
                                   "한 파일에는 정답 하나만 두세요")])
        table[k], where[k] = v, rn
    return table


def read_series_folder(path: str, column: str, length: int, resample: str = "interp",
                       label_file: Optional[str] = None, only_labeled: bool = False) -> tuple:
    """폴더 → (샘플 이름, X (n, length), 정답 문자열 배열 또는 None, 방식). 정답이 없어도 읽음 (라벨링 작업실용)."""
    if not isinstance(length, int) or not 8 <= length <= SERIES_MAX_LEN:
        raise DataError([Issue(None, None, f"길이 {length}는 쓸 수 없음", f"8–{SERIES_MAX_LEN:,} 사이로 정하세요")])
    if resample not in ("interp", "crop"):
        raise DataError([Issue(None, None, f"길이 맞추기 방식 {resample!r}을(를) 모름", "interp(다시 뽑기) 또는 crop(자르기·채우기)")])
    files, mode, _ = _series_files(path, label_file)
    labels = None
    if mode == "folders":
        labels = np.asarray([c for _, _, c in files], dtype=object)
    elif mode == "file":
        key = _series_key                                  # 이름 맞추기: "폴더/파일"이 먼저, 없으면 파일 이름만
        table = _label_table(label_file, {key(n) for n, _, _ in files} | {key(f.name) for _, f, _ in files})
        pick = lambda n, f: table.get(key(n), table.get(key(f.name)))
        # 하위 폴더 CSV는 표에 있을 때만 씀 (관계없는 backup/ 등). only_labeled면 맨 위도 표에 있는 것만
        files = [(n, f, c) for n, f, c in files if pick(n, f) is not None or ("/" not in n and not only_labeled)]
        if not files:
            raise DataError([Issue(None, None, "정답 표와 맞는 파일이 없음", "정답 표의 첫 열이 파일 이름인지 확인하세요")])
        missing = [n for n, f, _ in files if pick(n, f) is None]
        if missing:
            shown = ", ".join(Path(n).stem for n in missing[:5]) + (" …" if len(missing) > 5 else "")
            raise DataError([Issue(None, None, f"정답 표에 없는 파일 {len(missing)}개: {shown}", "정답 표에 그 파일들의 행을 더하세요")])
        labels = np.asarray([pick(n, f) for n, f, _ in files], dtype=object)
    X = np.stack([_fit_length(_series_values(n, f, column), length, resample) for n, f, _ in files])
    return [n for n, _, _ in files], X, labels, mode


def load_series_folder(path: str, column: str, length: int, resample: str = "interp", label_file: Optional[str] = None,
                       task: str = "auto", normalize: bool = True, test_ratio: float = 0.2, seed: int = 0,
                       only_labeled: bool = False) -> Dataset:
    """샘플마다 CSV 하나인 폴더 → 시계열 Dataset (값 열 하나, 길이 length로 맞춤)."""
    names, X, labels, mode = read_series_folder(path, column, length, resample, label_file, only_labeled)
    if mode == "none":
        raise DataError([Issue(None, None, f"정답이 없음 (CSV {len(names)}개)",
                               "클래스마다 하위 폴더(예: 정상/, 이상/)로 나누거나, 정답 표 CSV(파일이름, 정답)를 지정하세요. "
                               "라벨링 작업실은 준비 중입니다")])
    src = {"series_folder": str(path), "column": column, "length": length, "resample": resample,
           "label_file": label_file, "normalize": normalize, "test_ratio": test_ratio, "seed": seed, "task": task,
           "only_labeled": only_labeled}
    return make_dataset(X, labels, task=task, normalize=normalize, test_ratio=test_ratio, seed=seed, source=src)


def image_files(path: str) -> list:
    """폴더(맨 위 + 하위 폴더 한 단계)의 그림 → [(이름 '폴더/파일' 또는 '파일', 경로, 하위 폴더 이름 또는 None)]."""
    root = Path(path)
    if not root.is_dir():
        raise DataError([Issue(None, None, f"폴더를 찾을 수 없음: {path}", "그림이 든 폴더의 전체 경로를 넣으세요")])
    junk = lambda name: name.startswith(".") or name == "__MACOSX"
    is_img = lambda f: f.is_file() and not junk(f.name) and f.suffix.lower() in IMAGE_EXTS
    out = [(f.name, f, None) for f in sorted(root.iterdir(), key=lambda f: f.name) if is_img(f)]
    for d in sorted((e for e in root.iterdir() if e.is_dir() and not junk(e.name)), key=lambda d: d.name):
        out += [(f"{d.name}/{f.name}", f, d.name) for f in sorted(d.iterdir(), key=lambda f: f.name) if is_img(f)]
    return out


def load_image_folder(path: str, size: int = 28, normalize: bool = True, test_ratio: float = 0.2,
                      seed: int = 0, label_file: Optional[str] = None, only_labeled: bool = False,
                      task: str = "auto") -> Dataset:
    """이미지 폴더 → Dataset. 폴더/클래스이름/*.png·jpg·jpeg·bmp. 흑백으로 바꾸고 size×size로 맞춤, 픽셀 0–1.
    클래스는 하위 폴더 이름 순서. 이미지가 아닌 파일(Thumbs.db 등)은 건너뜀.
    label_file(정답 표: 파일 이름, 정답)을 주면 하위 폴더 대신 표의 정답을 씀 (라벨링 작업실). only_labeled면 표에 있는 그림만."""
    if label_file:
        return _load_image_folder_table(path, size, normalize, test_ratio, seed, label_file, only_labeled, task)
    if size not in IMAGE_SIZES:
        raise DataError([Issue(None, None, f"크기 {size}는 쓸 수 없음", f"{', '.join(map(str, IMAGE_SIZES))} 중에서 고르세요")])
    try:
        from PIL import Image
    except ImportError:
        raise DataError([Issue(None, None, "이미지 폴더를 읽으려면 Pillow가 필요함",
                               "명령창에서 pip install pillow 를 실행한 뒤 다시 불러오세요 (CSV·예시는 그대로 됨)")])
    root = Path(path)
    junk = lambda name: name.startswith(".") or name == "__MACOSX"      # 숨김 파일, macOS 압축의 ._파일·__MACOSX
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not junk(d.name))
    files = {c: sorted(f for f in (root / c).iterdir()
                       if f.is_file() and not junk(f.name) and f.suffix.lower() in IMAGE_EXTS) for c in classes}
    classes = [c for c in classes if files[c]]
    if len(classes) < 2:
        raise DataError([Issue(None, None, f"이미지가 든 클래스 폴더가 {len(classes)}개뿐임",
                               "폴더 안에 클래스마다 하위 폴더(예: 양품, 불량)를 두고 그림을 넣으세요")])
    few = [c for c in classes if len(files[c]) < 2]
    if few:
        raise DataError([Issue(None, None, f"클래스 '{few[0]}'에 그림이 1장뿐임", "클래스마다 2장 이상 넣으세요")])
    total = sum(len(v) for v in files.values())
    if total > MAX_IMAGES:
        raise DataError([Issue(None, None, f"그림이 {total:,}장 (한도 {MAX_IMAGES:,}장)", "그림 수를 줄이거나 나눠서 쓰세요")])
    X, y, issues = [], [], []
    for k, c in enumerate(classes):
        for f in files[c]:
            try:
                with Image.open(f) as im:
                    X.append(_gray01(im, size, Image).reshape(-1))
                    y.append(k)
            except Exception:
                issues.append(Issue(None, None, f"그림을 읽을 수 없음: {c}/{f.name}", "깨진 파일이면 지우거나 다시 저장하세요"))
                if len(issues) >= MAX_ISSUES:
                    raise DataError(issues)
    if issues:
        raise DataError(issues)
    src = {"folder": str(path), "size": size, "normalize": normalize, "test_ratio": test_ratio, "seed": seed}
    return make_dataset(np.asarray(X), np.asarray(y), task="classification", normalize=normalize, test_ratio=test_ratio,
                        seed=seed, classes=classes, source=src, image=(size, size))


def _load_image_folder_table(path, size, normalize, test_ratio, seed, label_file, only_labeled, task="auto") -> Dataset:
    if size not in IMAGE_SIZES:
        raise DataError([Issue(None, None, f"크기 {size}는 쓸 수 없음", f"{', '.join(map(str, IMAGE_SIZES))} 중에서 고르세요")])
    try:
        from PIL import Image
    except ImportError:
        raise DataError([Issue(None, None, "이미지 폴더를 읽으려면 Pillow가 필요함", "명령창에서 pip install pillow 를 실행하세요")])
    if not Path(label_file).is_file():
        raise DataError([Issue(None, None, f"정답 표를 찾을 수 없음: {label_file}", "정답 표 CSV(파일이름, 정답)의 전체 경로를 넣으세요")])
    files = image_files(path)
    key = _series_key_img
    seen: dict = {}                                        # 확장자만 다른 같은 이름은 정답이 섞이므로 거절
    for n, _, _ in files:
        seen.setdefault(key(n), []).append(n)
    dup = [v for v in seen.values() if len(v) > 1]
    if dup:
        raise DataError([Issue(None, None, f"확장자만 다른 같은 이름의 그림: {', '.join(dup[0])}", "그림 이름을 다르게 바꾸세요")])
    table = _label_table(label_file, {key(n) for n, _, _ in files} | {key(f.name) for _, f, _ in files}, key)
    pick = lambda n, f: table.get(key(n), table.get(key(f.name)))
    files = [(n, f) for n, f, _ in files if pick(n, f) is not None or not only_labeled]
    missing = [n for n, f in files if pick(n, f) is None]
    if missing:
        shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        raise DataError([Issue(None, None, f"정답 표에 없는 그림 {len(missing)}개: {shown}", "정답 표에 그 그림들의 행을 더하세요")])
    if len(files) > MAX_IMAGES:
        raise DataError([Issue(None, None, f"그림이 {len(files):,}장 (한도 {MAX_IMAGES:,}장)", "그림 수를 줄이거나 나눠서 쓰세요")])
    X, y = [], []
    for n, f in files:
        try:
            with Image.open(f) as im:
                X.append(_gray01(im, size, Image).reshape(-1))
                y.append(pick(n, f))
        except Exception:
            raise DataError([Issue(None, None, f"그림을 읽을 수 없음: {n}", "깨진 파일이면 지우거나 정답 표에서 빼세요")])
    src = {"folder": str(path), "size": size, "normalize": normalize, "test_ratio": test_ratio, "seed": seed,
           "label_file": label_file, "only_labeled": only_labeled, "task": task}
    return make_dataset(np.asarray(X), np.asarray(y, dtype=object), task=task, normalize=normalize,
                        test_ratio=test_ratio, seed=seed, source=src, image=(size, size))


def make_det_dataset(X: np.ndarray, boxes: list, classes: list, size: int, normalize: bool = True,
                     test_ratio: float = 0.2, seed: int = 0, source: Optional[dict] = None) -> Dataset:
    """탐지 데이터: 그림 (n, size²) 픽셀 0–1 + 그림마다 상자 [[클래스 번호, x, y, w, h], …] → Dataset (task "detection").
    정답(y)은 모델 출력과 같은 자리의 목표값 (detect.encode). 시험 데이터는 무작위로 test_ratio만큼."""
    from dlstudio.detect import encode, grid_of
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < MIN_SAMPLES:
        raise DataError([Issue(None, None, f"그림이 {n}장뿐임 (최소 {MIN_SAMPLES}장)", "그림을 더 모아 라벨링하세요")])
    if len(classes) < 1 or not any(boxes):
        raise DataError([Issue(None, None, "상자가 하나도 없음", "라벨링에서 사물에 상자(또는 다각형·원)를 그리세요")])
    G, K = grid_of(size), len(classes)
    Y = np.stack([encode(b, G, K) for b in boxes])
    test = np.sort(np.random.default_rng(seed).permutation(n)[:max(1, int(round(n * test_ratio)))])
    train = np.setdiff1d(np.arange(n), test)
    Xtr, Xte = X[train], X[test]
    m, s = (float(Xtr.mean()), float(Xtr.std()) or 1.0) if normalize else (0.0, 1.0)
    mean, std = np.full(X.shape[1], m), np.full(X.shape[1], s)
    src = dict(source or {}, normalize=normalize, test_ratio=test_ratio, seed=seed)
    return Dataset((Xtr - mean) / std, (Xte - mean) / std, Y[train], Y[test], "detection", list(classes), X.shape[1],
                   mean, std, src, [], (size, size),
                   {"grid": G, "train": [boxes[i] for i in train], "test": [boxes[i] for i in test]})


def read_det_file(det_file: str) -> tuple:
    """탐지 정답 파일 (라벨링에서 만듦, JSON) → (클래스 이름 목록, {그림 이름: [[클래스 번호, x, y, w, h], …]})."""
    import json
    try:
        d = json.loads(Path(det_file).read_text(encoding="utf-8"))
        classes = [str(c) for c in d["classes"]]
        items = {str(it["name"]): [[int(b[0])] + [float(v) for v in b[1:5]] for b in it["boxes"]] for it in d["items"]}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        raise DataError([Issue(None, None, f"탐지 정답 파일을 읽을 수 없음: {det_file}", "라벨링에서 [학습 데이터로 쓰기]를 다시 누르세요")])
    return classes, items


def load_detection_folder(path: str, det_file: str, size: int = 64, normalize: bool = True, test_ratio: float = 0.2,
                          seed: int = 0) -> Dataset:
    """그림 폴더 + 탐지 정답 파일 → 탐지 Dataset. 그림은 흑백 size×size로 (상자는 0–1 비율이라 그대로)."""
    from dlstudio.detect import DET_SIZES
    if size not in DET_SIZES:
        raise DataError([Issue(None, None, f"크기 {size}는 탐지에 쓸 수 없음", f"{', '.join(map(str, DET_SIZES))} 중에서 고르세요")])
    try:
        from PIL import Image
    except ImportError:
        raise DataError([Issue(None, None, "이미지 폴더를 읽으려면 Pillow가 필요함", "명령창에서 pip install pillow 를 실행하세요")])
    classes, items = read_det_file(det_file)
    files = {n: f for n, f, _ in image_files(path)}
    missing = [n for n in items if n not in files]
    if missing:
        raise DataError([Issue(None, None, f"폴더에 없는 그림 {len(missing)}개: {', '.join(missing[:5])}",
                               "라벨링할 때와 같은 폴더를 고르세요")])
    if len(items) > MAX_IMAGES:
        raise DataError([Issue(None, None, f"그림이 {len(items):,}장 (한도 {MAX_IMAGES:,}장)", "그림 수를 줄이세요")])
    X, boxes = [], []
    for n in sorted(items):
        try:
            with Image.open(files[n]) as im:
                X.append(_gray01(im, size, Image).reshape(-1))
        except Exception:
            raise DataError([Issue(None, None, f"그림을 읽을 수 없음: {n}", "깨진 파일이면 라벨링에서 빼세요")])
        boxes.append(items[n])
    src = {"folder": str(path), "size": size, "det_file": str(det_file)}
    return make_det_dataset(np.asarray(X), boxes, classes, size, normalize, test_ratio, seed, src)


def _series_key_img(s: str) -> str:
    """정답 표와 그림 이름 맞추기: 공백·대소문자·확장자 무시."""
    s = s.strip().lower()
    for ext in IMAGE_EXTS:
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def _labels(y: np.ndarray, task: str, classes: Optional[list]):
    """정답 → (task, classes, y 숫자)."""
    if classes is not None:                                     # 이미 0..K-1 정수 (예시 데이터)
        return "classification", list(classes), np.asarray(y, dtype=np.int64)
    as_num = [_num(str(v)) for v in y]
    numeric = all(v is not None for v in as_num)
    if task == "auto":
        if numeric:
            vals = np.asarray(as_num)
            task = "classification" if np.all(vals == np.round(vals)) and len(np.unique(vals)) <= MAX_CLASSES \
                else "regression"
        else:
            task = "classification"
    if task == "regression":
        if not numeric:
            raise DataError([Issue(None, None, "회귀인데 정답에 숫자가 아닌 값이 있음", "과제를 분류로 바꾸거나 정답 열을 확인하세요")])
        return task, [], np.asarray(as_num, dtype=np.float64)
    if task != "classification":
        raise ValueError(f"task는 auto, classification, regression 중 하나: {task}")
    if numeric:
        uniq = sorted(set(as_num))
        names = [str(int(v)) if float(v).is_integer() else str(v) for v in uniq]
        index = {v: k for k, v in enumerate(uniq)}
        return task, names, np.asarray([index[v] for v in as_num], dtype=np.int64)
    names = sorted({str(v) for v in y})
    index = {n: k for k, n in enumerate(names)}
    return task, names, np.asarray([index[str(v)] for v in y], dtype=np.int64)


def make_dataset(X: np.ndarray, y: np.ndarray, task: str = "auto", normalize: bool = True, test_ratio: float = 0.2,
                 seed: int = 0, classes: Optional[list] = None, source: Optional[dict] = None,
                 features: Optional[list] = None, image: Optional[tuple] = None) -> Dataset:
    """배열 → Dataset: 과제 판단, 문자열 정답 변환, (분류면 클래스 비율을 지켜) 나누기, 학습 데이터로 정규화.
    task "anomaly": 분류처럼 읽되 종류가 하나(정상만)여도 됨 — 이상 탐지로 바꾸는 일은 anomaly.to_anomaly."""
    one_ok = task == "anomaly"
    if one_ok:
        task = "classification"
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X는 (샘플, 길이) 2차원이어야 함: {X.shape}")
    n = len(X)
    if n < MIN_SAMPLES:
        raise DataError([Issue(None, None, f"샘플이 {n}개뿐임 (최소 {MIN_SAMPLES}개)", "샘플을 더 모아 주세요")])
    task, names, yv = _labels(np.asarray(y), task, classes)
    rng = np.random.default_rng(seed)
    if task == "classification":
        if len(names) < 2 and not one_ok:
            raise DataError([Issue(None, None, f"정답 종류가 하나뿐임 ({names[0] if names else '-'})",
                                   "분류하려면 정답이 두 종류 이상이어야 합니다")])
        test = []
        for k in range(len(names)):
            idx = rng.permutation(np.flatnonzero(yv == k))
            m = int(round(len(idx) * test_ratio)) if len(idx) >= 2 else 0
            test.extend(idx[:max(m, 1 if len(idx) >= 2 else 0)].tolist())
        test = np.sort(np.asarray(test, dtype=np.int64))
    else:
        test = np.sort(rng.permutation(n)[:max(1, int(round(n * test_ratio)))])
    train = np.setdiff1d(np.arange(n), test)
    Xtr, Xte = X[train], X[test]
    if normalize and image:                            # 이미지: 픽셀 전체에 평균·표준편차 하나 (늘 0인 테두리도 그대로)
        m, s = float(Xtr.mean()), float(Xtr.std()) or 1.0
        mean, std = np.full(X.shape[1], m), np.full(X.shape[1], s)
    elif normalize:
        mean, std = Xtr.mean(0), Xtr.std(0)
        std = np.where(std > 0, std, 1.0)
    else:
        mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])
    src = dict(source or {})
    src.setdefault("normalize", normalize)
    src.setdefault("test_ratio", test_ratio)
    src.setdefault("seed", seed)
    return Dataset((Xtr - mean) / std, (Xte - mean) / std, yv[train], yv[test], task, names, X.shape[1],
                   mean, std, src, list(features or []), tuple(image) if image else None)


# ---------------------------------------------------------------- 예시 데이터 (모두 공개된 실제 데이터)
# 원본을 작게 줄인 npz는 dlstudio/data/ (만드는 법: tools/build_examples.py), 출처·라이선스는 <파일>.SOURCE.txt.

DATA_DIR = Path(__file__).resolve().parent / "data"
CONCRETE_FEATURES = ["시멘트 (kg/m³)", "고로슬래그 (kg/m³)", "플라이애시 (kg/m³)", "물 (kg/m³)", "감수제 (kg/m³)",
                     "굵은 골재 (kg/m³)", "잔골재 (kg/m³)", "재령 (일)"]
HYDRAULIC_FEATURES = ["압력 PS1 (bar)", "압력 PS2 (bar)", "압력 PS3 (bar)", "압력 PS4 (bar)", "압력 PS5 (bar)",
                      "압력 PS6 (bar)", "모터 전력 (W)", "유량 FS1 (l/min)", "유량 FS2 (l/min)", "온도 TS1 (°C)",
                      "온도 TS2 (°C)", "온도 TS3 (°C)", "온도 TS4 (°C)", "진동 (mm/s)", "냉각 효율 (%)",
                      "냉각 동력 (kW)", "효율 계수 (%)"]
HYDRAULIC_STATES = ["정상", "냉각기 저하", "밸브 지연", "펌프 누설", "복합 고장"]


def _npz(name: str) -> dict:
    """예시 파일 하나 (웹: 화면이 누를 때 내려받아 둠 — example_files)."""
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise DataError([Issue(None, None, f"예시 데이터 파일이 없음: {name}.npz", "페이지를 새로 고친 뒤 다시 누르세요")])
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _float(X) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).reshape(len(X), -1)


def _hydraulic_states(z: dict) -> np.ndarray:
    """유압 사이클의 상태: 냉각기(100%)·밸브(100%)·펌프(누설 0)가 모두 정상이면 정상, 하나만 나쁘면 그것, 둘 이상이면 복합.
    어큐뮬레이터 압력 차이는 정상 범위의 흔들림으로 봄 (정상 293 사이클)."""
    bad = np.stack([z["cooler"] < 100, z["valve"] < 100, z["pump"] > 0], 1)
    return np.where(bad.sum(1) == 0, 0, np.where(bad.sum(1) > 1, 4, 1 + bad.argmax(1))).astype(np.int64)


PLATE_FEATURES = ["X 최소", "X 최대", "Y 최소", "Y 최대", "면적 (픽셀)", "X 둘레", "Y 둘레", "밝기 합", "최소 밝기", "최대 밝기",
                  "컨베이어 길이", "강종 A300", "강종 A400", "판 두께", "가장자리 지수", "빈 공간 지수", "정사각 지수",
                  "바깥 X 지수", "가장자리 X 지수", "가장자리 Y 지수", "바깥 전체 지수", "면적 로그", "X 로그 지수",
                  "Y 로그 지수", "방향 지수", "밝기 지수", "면적 시그모이드"]
PLATE_FAULTS = ["페이스트리", "Z 긁힘", "K 긁힘", "얼룩", "오염", "돌기", "기타 결함"]


# 종류 → (파일, (X, y) 만들기, 클래스 이름(회귀는 None), 열 이름 만들기, 그림 (H, W))
EXAMPLES = {
    "ecg": ("ecg_mitdb", lambda z: (_float(z["X"]), z["y"].astype(np.int64)), ["정상", "이상"], None, None),
    "bearing": ("bearing_uottawa", lambda z: (_float(z["X"]), z["y"].astype(np.int64)),
                ["정상", "내륜 결함", "외륜 결함", "볼 결함", "복합 결함"], None, None),
    "steel_plates": ("steel_plates", lambda z: (_float(z["X"]), z["y"].astype(np.int64)), PLATE_FAULTS,
                     lambda z: PLATE_FEATURES, None),
    "hydraulic": ("hydraulic", lambda z: (_float(z["X"]), _hydraulic_states(z)), HYDRAULIC_STATES,
                  lambda z: HYDRAULIC_FEATURES, None),
    "concrete": ("concrete", lambda z: (_float(z["X"]), z["y"].astype(np.float64)), None, lambda z: CONCRETE_FEATURES, None),
    "hydraulic_valve": ("hydraulic", lambda z: (_float(z["ps2"]), z["valve"].astype(np.float64)), None, None, None),
    "steelblast": ("steelblast", lambda z: (_float(z["X"]) / 255.0, z["y"].astype(np.int64)), ["양호", "불량"], None, (64, 64)),
    "pneumonia": ("pneumonia28", lambda z: (_float(z["X"]) / 255.0, z["y"].astype(np.int64)), ["정상", "폐렴"], None, (28, 28)),
    "digits": ("digits32", lambda z: (_float(z["X"]), z["y"].astype(np.int64)), [str(i) for i in range(10)], None, (32, 32)),
}
FC_EXAMPLES = {"steel_power": ("steel_power", "전력 사용량 (kWh)", 96, 24, 1)}   # 종류 → (파일, 열 이름, 입력·예측 길이, 간격)
AE_EXAMPLES = {"bearing_ae": "bearing", "hydraulic_ae": "hydraulic"}         # 이상 탐지: 분류 예시를 정상만 학습으로
DET_EXAMPLES = {"pcb_det": ("pcb_det", 64)}                                    # 탐지: 종류 → (파일, 그림 크기)
ALIASES = {"vibration": "bearing", "process_cls": "steel_plates", "process_reg": "concrete", "wear_rul": "hydraulic_valve",
           "surface": "steelblast", "parts_det": "pcb_det", "factory_power": "steel_power", "vibration_ae": "bearing_ae",
           "process_ae": "hydraulic_ae"}                                       # 예전(합성) 예시 이름 → 바꾼 실제 데이터
EXAMPLE_KINDS = sorted(set(EXAMPLES) | set(FC_EXAMPLES) | set(AE_EXAMPLES) | set(DET_EXAMPLES))


def example_files(kind: str) -> list:
    """예시 하나에 필요한 npz 파일 이름 (웹이 누를 때만 내려받음)."""
    kind = ALIASES.get(kind, kind)
    if kind in FC_EXAMPLES:
        return [FC_EXAMPLES[kind][0] + ".npz"]
    if kind in DET_EXAMPLES:
        return [DET_EXAMPLES[kind][0] + ".npz"]
    kind = AE_EXAMPLES.get(kind, kind)
    return [EXAMPLES[kind][0] + ".npz"] if kind in EXAMPLES else []


def _det_arrays(z: dict) -> tuple:
    """탐지 npz: X (그림, H, W) 0–255, boxes (상자, 6) = [그림 번호, 클래스, x, y, w, h] (0–1 비율)."""
    X = _float(z["X"]) / 255.0
    boxes = [[] for _ in range(len(X))]
    for i, k, x, y, w, h in z["boxes"]:
        boxes[int(i)].append([int(k), round(float(x), 5), round(float(y), 5), round(float(w), 5), round(float(h), 5)])
    return X, boxes, [str(c) for c in z["names"]]


def load_example(kind: str = "ecg", normalize: bool = True, test_ratio: float = 0.2, seed: int = 0,
                 L=None, H=None, stride=None) -> Dataset:
    """예시 데이터 (모두 공개된 실제 데이터, 출처는 dlstudio/data/*.SOURCE.txt). 예전 합성 예시 이름은 ALIASES로 바꿔 읽음."""
    kind = ALIASES.get(kind, kind)
    if kind not in EXAMPLE_KINDS:
        raise ValueError(f"예시 데이터 종류는 {', '.join(EXAMPLE_KINDS)} 중 하나: {kind}")
    if kind in FC_EXAMPLES:
        from dlstudio.forecast import make_forecast_dataset
        f, col, L0, H0, s0 = FC_EXAMPLES[kind]
        return make_forecast_dataset(_npz(f)["series"], int(L or L0), int(H or H0), int(stride or s0), test_ratio=test_ratio,
                                     normalize=normalize, source={"example": kind}, column=col)
    if kind in AE_EXAMPLES:
        from dlstudio.anomaly import to_anomaly
        d = load_example(AE_EXAMPLES[kind], normalize=normalize, test_ratio=test_ratio, seed=seed)
        d.source = dict(d.source, example=kind)
        return to_anomaly(d)
    if kind in DET_EXAMPLES:
        f, size = DET_EXAMPLES[kind]
        X, boxes, names = _det_arrays(_npz(f))
        return make_det_dataset(X, boxes, names, size, normalize, test_ratio, seed, {"example": kind})
    f, make, names, feats, image = EXAMPLES[kind]
    z = _npz(f)
    X, y = make(z)
    return make_dataset(X, y, task="classification" if names else "regression", normalize=normalize,
                        test_ratio=test_ratio, seed=seed, classes=names, source={"example": kind},
                        features=feats(z) if feats else None, image=image)

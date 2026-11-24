"""새 데이터로 예측(추론): 학습 때와 같은 모양으로 읽고 같은 정규화를 해서 모델에 넣을 값을 만들고, 결과를 표·CSV로.

받는 데이터 (학습 데이터의 종류에 맞춰)
- CSV: 한 행 = 샘플 하나. 값 열은 학습 때와 같은 순서·개수, 정답 열은 있어도 없어도 됨 (있으면 맞았는지 표시)
- 그림 폴더: 그림 파일들 (하위 폴더 이름이 클래스 이름이면 그것을 정답으로 봄)
- 시계열 폴더: 샘플마다 CSV 하나 (학습 때와 같은 값 열·길이 맞추기)
- 직접 넣은 값: [[값, …], …] (표 데이터)
"""
from __future__ import annotations

import io
import csv

import numpy as np

from dlstudio.data import (IMAGE_EXTS, MAX_ISSUES, DataError, Issue, _num, _read_rows, image_files,
                           read_series_folder)

MAX_ROWS = 20000           # 한 번에 예측하는 샘플 수 상한
SHOW_ROWS = 500            # 화면 표에 보이는 행 수 (CSV에는 전부)


def _err(msg: str, hint: str):
    return DataError([Issue(None, None, msg, hint)])


def read_new(d, b: dict) -> dict:
    """요청 → {"names": [...], "X": 정규화한 값 (n, 특징 수) float32, "labels": 정답 문자열 목록 또는 None}."""
    n_feat = int(d.X_train.shape[1])
    if b.get("rows") is not None:
        rows = b["rows"]
        try:
            X = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)
        except (TypeError, ValueError):
            raise _err("넣은 값 중 숫자가 아닌 것이 있음", "칸마다 숫자를 넣으세요") from None
        if X.shape[1] != n_feat:
            raise _err(f"값이 {X.shape[1]}개 (모델은 {n_feat}개)", "칸을 모두 채우세요")
        names, labels = [f"입력 {i + 1}" for i in range(len(X))], None
    elif b.get("folder"):
        names, X, labels = _read_images(d, str(b["folder"]))
    elif b.get("series_folder"):
        src = d.source
        if not src.get("series_folder"):
            raise _err("이 모델은 시계열 폴더로 학습하지 않았음", "학습 때와 같은 모양의 CSV로 예측하세요")
        names, X, labels, mode = read_series_folder(str(b["series_folder"]), src["column"], int(src["length"]),
                                                    src.get("resample", "interp"))
        if mode != "folders":
            labels = None
    elif b.get("path"):
        names, X, labels = _read_csv(d, str(b["path"]))
    else:
        raise _err("예측할 데이터가 없음", "CSV 파일이나 폴더를 고르세요")
    if not len(X):
        raise _err("예측할 샘플이 없음", "샘플이 한 행(또는 파일 하나)에 하나씩 있는지 확인하세요")
    if len(X) > MAX_ROWS:
        raise _err(f"샘플이 {len(X):,}개 (한 번에 {MAX_ROWS:,}개까지)", "나눠서 예측하세요")
    if not np.all(np.isfinite(X)):
        raise _err("빈 칸이나 숫자가 아닌 값이 있음", "값을 확인하세요")
    Xn = ((X - d.mean) / d.std).astype(np.float32)
    if labels is not None:
        labels = [str(v) for v in labels]
    return {"names": list(names), "X": Xn, "labels": labels}


def _read_csv(d, path: str) -> tuple:
    n_feat = int(d.X_train.shape[1])
    numbered = _read_rows(path)
    if not numbered:
        raise _err("파일이 비어 있음", "샘플이 한 행에 하나씩 있는 CSV를 고르세요")
    ncol = len(numbered[0][1])
    if ncol == n_feat:
        lc = None
    elif ncol == n_feat + 1:
        lc = d.source.get("label_col", ncol - 1) if d.source.get("path") else ncol - 1
        lc = lc if isinstance(lc, int) and 0 <= lc < ncol else ncol - 1
    else:
        raise _err(f"열이 {ncol}개 (학습은 값 열 {n_feat}개)",
                   f"학습 CSV와 같은 값 열 {n_feat}개 (정답 열은 있어도 없어도 됨)로 맞추세요")
    feats = [j for j in range(ncol) if j != lc]
    first = numbered[0][1]
    if any(first[j] and _num(first[j]) is None for j in feats) or (lc is not None and len(numbered) > 1 and
                                                                   _num(first[lc]) is None and
                                                                   _num(numbered[1][1][lc]) is not None):
        numbered = numbered[1:]                              # 머리글
    issues, X, labels, names = [], [], [], []
    for rn, r in numbered:
        if len(r) != ncol:
            issues.append(Issue(rn, None, f"{rn}행: 칸 수가 {len(r)}개 (다른 행은 {ncol}개)", "빠지거나 남는 쉼표가 있는지 확인하세요"))
        else:
            vals = [_num(r[j]) for j in feats]
            bad = [j for j, v in zip(feats, vals) if v is None]
            if bad:
                issues.append(Issue(rn, bad[0] + 1, f"{rn}행 {bad[0] + 1}열 '{r[bad[0]][:20]}'은(는) 숫자가 아님", "값 열에는 숫자만 있어야 합니다"))
            else:
                X.append(vals)
                labels.append(r[lc] if lc is not None else "")
                names.append(f"{rn}행")
        if len(issues) >= MAX_ISSUES:
            break
    if issues:
        raise DataError(issues[:MAX_ISSUES])
    X = np.asarray(X, dtype=np.float64)
    if d.image and len(X) and X.max() > 1:                   # 픽셀 0–255 → 0–1 (학습 때처럼)
        X = X / (255.0 if X.max() <= 255 else X.max())
    has_label = lc is not None and any(labels)
    return names, X, (labels if has_label else None)


def _read_images(d, folder: str) -> tuple:
    if not d.image:
        raise _err("이 모델은 그림으로 학습하지 않았음", "학습 때와 같은 모양의 CSV로 예측하세요")
    H, W = d.image
    try:
        from PIL import Image
    except ImportError:
        raise _err("그림을 읽으려면 Pillow가 필요함", "명령창에서 pip install pillow") from None
    files = image_files(folder)
    if not files:
        raise _err("폴더에 그림(png·jpg·bmp)이 없음", f"{', '.join(IMAGE_EXTS)} 파일이 든 폴더를 고르세요")
    classes = {str(c) for c in d.classes} if d.task == "classification" else set()
    X, names, labels, issues = [], [], [], []
    for name, f, sub in files:
        try:
            with Image.open(f) as im:
                if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
                    rgba = im.convert("RGBA")
                    im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
                X.append(np.asarray(im.convert("L").resize((W, H)), dtype=np.float64).reshape(-1) / 255.0)
                names.append(name)
                labels.append(sub)
        except Exception:
            issues.append(Issue(None, None, f"그림을 읽을 수 없음: {name}", "깨진 파일이면 지우거나 다시 저장하세요"))
            if len(issues) >= MAX_ISSUES:
                break
    if issues:
        raise DataError(issues)
    known = labels and all(s is not None and s in classes for s in labels)
    return names, np.asarray(X), (labels if known else None)


SHOW_PICS = 24             # 탐지: 화면에 그림으로 보이는 수


def _table_det(d, names: list, out: np.ndarray, X) -> dict:
    """탐지: 그림마다 찾은 상자 (점수 0.5 이상). CSV는 상자마다 한 줄, 위치는 그림 크기에 대한 0–1 비율 (왼쪽 위 기준)."""
    from dlstudio.detect import SHOW_SCORE, decode
    G, K = d.n_out.grid, d.n_out.n_cls
    cls = [str(c) for c in d.classes]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["그림", "클래스", "점수", "x", "y", "너비", "높이"])
    rows, counts = [], {c: 0 for c in cls}
    for i, nm in enumerate(names):
        bs = decode(out[i], G, K, SHOW_SCORE)
        for b in bs:
            counts[cls[b[0]]] += 1
            w.writerow([nm, cls[b[0]]] + [f"{v:.4f}" for v in b[1:]])
        if not bs:
            w.writerow([nm, "", "", "", "", "", ""])
        if i < SHOW_ROWS:
            rows.append({"name": nm, "boxes": [[int(b[0])] + [round(float(v), 4) for v in b[1:]] for b in bs]})
    pics = [] if X is None else [{"name": names[i], "values": np.round(d.raw(np.asarray(X[i], dtype=np.float64)), 3).tolist()}
                                 for i in range(min(SHOW_PICS, len(names)))]
    n_boxes = sum(counts.values())
    summary = {"n": len(names), "task": "detection", "counts": counts, "n_boxes": n_boxes, "score": SHOW_SCORE,
               "no_box": sum(1 for r in rows if not r["boxes"])}
    return {"summary": summary, "rows": rows, "shown": min(len(names), SHOW_ROWS), "csv": "\ufeff" + buf.getvalue(),
            "has_label": False, "pics": pics}


def table(d, names: list, out: np.ndarray, labels, X=None) -> dict:
    """모델 출력 (n, n_out) → 화면 표·요약·CSV 글자. X: 정규화한 입력 (탐지면 그림으로 보여 줌)."""
    out = np.asarray(out, dtype=np.float64).reshape(len(names), -1)
    if d.task == "detection":
        return _table_det(d, names, out, X)
    cls = d.task == "classification"
    rows, summary = [], {"n": len(names), "task": d.task}
    buf = io.StringIO()
    w = csv.writer(buf)
    if cls:
        names_c = [str(c) for c in d.classes]
        e = np.exp(out - out.max(1, keepdims=True))
        prob = e / e.sum(1, keepdims=True)
        pred = prob.argmax(1)
        w.writerow(["샘플", "예측", "확신도"] + [f"확률_{c}" for c in names_c] + (["정답", "맞음"] if labels else []))
        for i, nm in enumerate(names):
            p = names_c[pred[i]]
            ok = None if labels is None else (labels[i] == p)
            w.writerow([nm, p, f"{prob[i, pred[i]]:.4f}"] + [f"{v:.4f}" for v in prob[i]] +
                       ([labels[i], "O" if ok else "X"] if labels else []))
            if i < SHOW_ROWS:
                rows.append({"name": nm, "pred": p, "conf": float(prob[i, pred[i]]),
                             "label": None if labels is None else labels[i], "ok": ok})
        summary["counts"] = {c: int((pred == k).sum()) for k, c in enumerate(names_c)}
        if labels is not None:
            summary["accuracy"] = float(np.mean([labels[i] == names_c[pred[i]] for i in range(len(names))]))
        summary["low_conf"] = int((prob.max(1) < 0.6).sum())
    else:
        pred = out[:, 0]
        w.writerow(["샘플", "예측"] + (["정답", "오차"] if labels else []))
        ys = None
        if labels is not None:
            try:
                ys = np.asarray([float(v) for v in labels])
            except ValueError:
                labels = None
        for i, nm in enumerate(names):
            w.writerow([nm, f"{pred[i]:.6g}"] + ([labels[i], f"{pred[i] - ys[i]:.6g}"] if labels is not None else []))
            if i < SHOW_ROWS:
                rows.append({"name": nm, "pred": float(pred[i]), "label": None if ys is None else float(ys[i]),
                             "err": None if ys is None else float(pred[i] - ys[i])})
        summary["mean"] = float(pred.mean())
        summary["range"] = [float(pred.min()), float(pred.max())]
        if ys is not None:
            summary["mae"] = float(np.mean(np.abs(pred - ys)))
    return {"summary": summary, "rows": rows, "shown": min(len(names), SHOW_ROWS), "csv": "﻿" + buf.getvalue(),
            "has_label": labels is not None}


def new_data_kinds(d) -> list:
    """이 모델에 넣을 수 있는 새 데이터 종류 (화면이 단추를 고름): series(시계열 폴더) · folder(그림 폴더) · csv · rows(직접)."""
    if d.task == "detection":
        return ["folder"]
    if d.source.get("series_folder"):
        return ["series", "csv"]
    if d.image:
        return ["folder", "csv"]
    return ["csv", "rows"] if d.X_train.shape[1] <= 40 else ["csv"]

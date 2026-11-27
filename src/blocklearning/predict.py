"""새 데이터로 예측(추론): 학습 때와 같은 모양으로 읽고 같은 정규화를 해서 모델에 넣을 값을 만들고, 결과를 표·CSV로.

받는 데이터 (학습 데이터의 종류에 맞춰)
- CSV: 한 행 = 샘플 하나. 값 열은 학습 때와 같은 순서·개수, 정답 열은 있어도 없어도 됨 (있으면 맞았는지 표시)
- 그림 폴더: 그림 파일들 (하위 폴더 이름이 클래스 이름이면 그것을 정답으로 봄)
- 시계열 폴더: 샘플마다 CSV 하나 (학습 때와 같은 값 열·길이 맞추기)
- 직접 넣은 값: [[값, …], …] (표 데이터)
- 시계열 예측: 긴 시계열 CSV 하나 (한 행 = 한 시점) → 끝의 지난 L개로 그다음 H개
- 이상 탐지: 분류와 같은 모양 → 샘플마다 이상 점수와 판정 (학습 결과의 기준값으로)
"""
from __future__ import annotations

import io
import csv

import numpy as np

from blocklearning.data import (IMAGE_EXTS, MAX_ISSUES, DataError, Issue, _num, _pixels01, _read_rows, image_files,
                           read_series_folder)

MAX_ROWS = 20000           # 한 번에 예측하는 샘플 수 상한
SHOW_ROWS = 500            # 화면 표에 보이는 행 수 (CSV에는 전부)


def _err(msg: str, hint: str):
    return DataError([Issue(None, None, msg, hint)])


def read_new(d, b: dict) -> dict:
    """요청 → {"names": [...], "X": 정규화한 값 (n, 특징 수) float32, "labels": 정답 문자열 목록 또는 None}."""
    n_feat = int(d.X_train.shape[1])
    if b.get("example_new"):                                # 예시 데이터의 학습·시험에 쓰지 않은 10개
        from blocklearning.data import example_new
        e = example_new(d)
        Xn = ((np.asarray(e["X"], dtype=np.float64) - d.mean) / d.std).astype(np.float32)
        return {"names": e["names"], "X": Xn, "labels": e["labels"], "truth": e["truth"]}
    if d.task == "forecast":
        if not b.get("path"):
            raise _err("시계열 예측은 긴 시계열 CSV로 예측함", "한 행 = 한 시점인 CSV를 고르세요 (학습 때와 같은 값 열)")
        names, X, labels = _read_forecast(d, str(b["path"]))
    elif b.get("rows") is not None:
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


def _read_forecast(d, path: str) -> tuple:
    """긴 시계열 CSV → 끝의 지난 L개 (창 하나). 값 열은 학습 때 이름이 있으면 그것, 없으면 마지막 숫자 열."""
    from pathlib import Path
    from blocklearning.data import _column_values, _numeric_columns, _read_table
    names, rows = _read_table(path)
    nums = _numeric_columns(names, rows)
    if not nums:
        raise _err("숫자 열이 없음", "한 행 = 한 시점, 열 = 측정값인 CSV를 고르세요")
    col = d.extra.get("column") if d.extra.get("column") in nums else nums[-1]
    v = _column_values(Path(path).name, names, rows, col)
    L = int(d.length)
    if len(v) < L:
        raise _err(f"값이 {len(v)}개 (모델은 지난 {L}개를 봄)", f"'{col}' 열에 값이 {L}개 이상 있는 CSV를 고르세요")
    x = v[-L:]
    if d.extra.get("calendar"):                             # 시각·요일 채널로 학습한 모델: 이 CSV에도 시각 열이 있어야 함
        from blocklearning.forecast import calendar, find_times
        tcol, t = find_times(names, rows, nums)
        cal, cn = calendar(t[-L:]) if t is not None else (None, [])
        if cal is None or cn != list(d.extra["calendar"]):
            raise _err("시각 열이 없거나 간격이 학습 때와 다름", "학습 때처럼 날짜·시각 열이 있는 CSV를 고르세요 (예: 2018-01-01 00:00)")
        x = np.concatenate([x, cal.ravel()])
    return [f"'{col}' 끝 다음"], x[None], None


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
    color = getattr(d.length, "channels", 1) == 3
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
                if color:                                  # 컬러로 학습한 모델: 학습 때와 같은 RGB (채널이 먼저)
                    X.append(_pixels01(im, H, Image, True))
                    names.append(name)
                    labels.append(sub)
                    continue
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


def _table_det(d, names: list, out: np.ndarray, X, truth=None) -> dict:
    """탐지: 그림마다 찾은 상자 (점수 0.5 이상). CSV는 상자마다 한 줄, 위치는 그림 크기에 대한 0–1 비율 (왼쪽 위 기준)."""
    from blocklearning.detect import SHOW_SCORE, decode
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
    pics = [] if X is None else [{"name": names[i], "values": np.round(d.raw(np.asarray(X[i], dtype=np.float64)), 3).tolist(),
                                  "truth": truth[i] if truth is not None else None} for i in range(min(SHOW_PICS, len(names)))]
    n_boxes = sum(counts.values())
    summary = {"n": len(names), "task": "detection", "counts": counts, "n_boxes": n_boxes, "score": SHOW_SCORE,
               "no_box": sum(1 for r in rows if not r["boxes"])}
    return {"summary": summary, "rows": rows, "shown": min(len(names), SHOW_ROWS), "csv": "\ufeff" + buf.getvalue(),
            "has_label": False, "pics": pics}


def _table_fc_many(d, names, out: np.ndarray, X, truth) -> dict:
    """예측 (예시 새 데이터): 창마다 다음 H개와 실제 값, 평균 오차."""
    truth = np.asarray(truth, dtype=np.float64)
    from blocklearning.forecast import history
    hist = history(d, X)
    err = np.abs(out - truth)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["창", "몇 칸 뒤", "예측", "실제"])
    for i, nm in enumerate(names):
        for k in range(out.shape[1]):
            w.writerow([nm, k + 1, f"{out[i, k]:.6g}", f"{truth[i, k]:.6g}"])
    rows = [{"name": nm, "mae": float(err[i].mean()), "history": np.round(hist[i], 5).tolist(), "pred": np.round(out[i], 5).tolist(),
             "y": np.round(truth[i], 5).tolist()} for i, nm in enumerate(names)]
    summary = {"n": len(names), "task": "forecast", "many": True, "column": d.extra.get("column"), "mae": float(err.mean()),
               "horizon": int(out.shape[1])}
    return {"summary": summary, "rows": rows, "shown": len(rows), "csv": "﻿" + buf.getvalue(), "has_label": True}


def _table_fc(d, out: np.ndarray, X) -> dict:
    """예측: 다음 H개 (칸마다 한 줄) + 그림용 지난 값."""
    pred = out[0]
    from blocklearning.forecast import history
    hist = history(d, X[0])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["몇 칸 뒤", "예측"])
    rows = []
    for k, v in enumerate(pred, 1):
        w.writerow([k, f"{v:.6g}"])
        rows.append({"name": f"{k}칸 뒤", "pred": float(v)})
    summary = {"n": len(pred), "task": "forecast", "column": d.extra.get("column"), "history": np.round(hist, 5).tolist(),
               "pred": np.round(pred, 5).tolist(), "mean": float(pred.mean()), "range": [float(pred.min()), float(pred.max())]}
    return {"summary": summary, "rows": rows, "shown": len(rows), "csv": "\ufeff" + buf.getvalue(), "has_label": False}


def _table_ae(d, names: list, out: np.ndarray, labels, X, results) -> dict:
    """이상 탐지: 샘플마다 이상 점수, 학습 결과의 기준값보다 크면 이상. 정답이 있으면 (정상 이름이 아니면 이상) 맞았는지."""
    from blocklearning.anomaly import scores
    s = scores(out, np.asarray(X, dtype=np.float64))
    th = (results or {}).get("threshold")
    normal = d.extra["normal"]
    truth = None if labels is None else [str(v) != str(normal) for v in labels]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["샘플", "이상 점수", "판정"] + (["정답", "맞음"] if labels else []))
    rows = []
    for i, nm in enumerate(names):
        ab = None if th is None else bool(s[i] > th)
        ok = None if truth is None or ab is None else ab == truth[i]
        w.writerow([nm, f"{s[i]:.6g}", "" if ab is None else ("이상" if ab else "정상")] +
                   ([labels[i], "O" if ok else "X"] if labels else []))
        if i < SHOW_ROWS:
            rows.append({"name": nm, "score": float(s[i]), "anomaly": ab, "label": None if labels is None else labels[i], "ok": ok})
    summary = {"n": len(names), "task": "anomaly", "threshold": th, "normal": normal, "mean": float(s.mean())}
    if th is not None:
        summary["counts"] = {"정상": int((s <= th).sum()), "이상": int((s > th).sum())}
        if truth is not None:
            summary["accuracy"] = float(np.mean([(s[i] > th) == truth[i] for i in range(len(names))]))
    return {"summary": summary, "rows": rows, "shown": min(len(names), SHOW_ROWS), "csv": "\ufeff" + buf.getvalue(),
            "has_label": labels is not None}


def table(d, names: list, out: np.ndarray, labels, X=None, results=None, truth=None) -> dict:
    if d.task == "segmentation":
        from blocklearning import segment
        return segment.table(d, names, out, truth)
    """모델 출력 (n, n_out) → 화면 표·요약·CSV 글자. X: 정규화한 입력 (탐지면 그림으로 보여 줌, 이상 탐지는 점수에).
    results: 학습 결과 (이상 탐지의 기준값)."""
    out = np.asarray(out, dtype=np.float64).reshape(len(names), -1)
    if d.task == "multi":
        from blocklearning import multi
        return multi.table(d, names, out, truth)
    if d.task == "detection":
        return _table_det(d, names, out, X, truth)
    if d.task == "forecast":
        return _table_fc_many(d, names, out, X, truth) if truth is not None else _table_fc(d, out, X)
    if d.task == "anomaly":
        return _table_ae(d, names, out, labels, X, results)
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
        return ["example", "folder"] if (d.source or {}).get("example") else ["folder"]
    if d.task == "generation":                              # 생성은 새 데이터를 넣지 않음 (④ 결과에서 새 샘플을 만듦)
        return []
    ex = ["example"] if (d.source or {}).get("example") else []   # 예시 데이터: 학습·시험에 쓰지 않은 10개
    if d.task == "forecast":
        return ex + ["csv"]
    if d.task == "segmentation":
        return ex
    if d.task == "multi":                                   # 정답 열이 여럿인 CSV는 ② 역할 표가 생기면
        return ex + (["rows"] if d.X_train.shape[1] <= 40 else [])
    if d.source.get("series_folder"):
        return ex + ["series", "csv"]
    if d.image:
        return ex + ["folder", "csv"]
    return ex + (["csv", "rows"] if d.X_train.shape[1] <= 40 else ["csv"])

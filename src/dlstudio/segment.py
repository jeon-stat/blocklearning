"""구간 분할: 긴 신호의 시점마다 상태(공정 단계 등)를 맞힌다.

데이터: 신호를 길이 L의 창으로 자르고, 창 안 시점마다 상태 번호 (y: (창 수, L)).
모델: 길이를 줄이지 않는 층들 → 출력 블록이 시점마다 1×1 합성곱으로 상태 점수 K개 → (K, L)을 펼친 K·L개 (상태가 먼저).
손실: 시점마다 교차 엔트로피. 성적: 시점 정확도 (맞힌 시점 비율).
"""
from __future__ import annotations

import io
import csv

from pathlib import Path

import numpy as np

MIN_TRAIN = 10          # 학습 창이 이보다 적으면 안 됨
N_SHOW = 4              # 결과 화면에 그릴 시험 창 (상태가 가장 많이 바뀌는 것부터)


class SegOut(int):
    """구간 분할 과제의 모델 출력 크기 (상태 수 K × 길이 L). int처럼 쓰이고, K·L을 함께 지님 →
    모델 검사·만들기·코드가 출력 블록을 '시점마다 1×1 합성곱'으로 만든다."""

    def __new__(cls, n_cls: int, length: int):
        o = super().__new__(cls, int(n_cls) * int(length))
        o.n_cls, o.length, o.seg = int(n_cls), int(length), True
        return o

    def __getnewargs__(self):
        return (self.n_cls, self.length)

    def __reduce__(self):
        return (SegOut, (self.n_cls, self.length))


def make_segment_dataset(X, Y, classes: list, test_ratio: float = 0.2, seed: int = 0, normalize: bool = True,
                         source: dict | None = None, channels: int = 1):
    """창들 (n, C·L, 채널이 먼저) · 시점마다 상태 (n, L) → 구간 분할 Dataset (X는 채널마다 학습 몫 전체 평균·표준편차로 정규화)."""
    from dlstudio.data import DataError, Dataset, Issue, seq_length
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.int64)
    n, L = Y.shape
    if X.shape != (n, channels * L):
        raise DataError([Issue(None, None, "창마다 시점 수만큼 상태가 있어야 함", "신호와 상태의 길이를 맞추세요")])
    order = np.random.default_rng(seed).permutation(n)
    n_te = max(1, int(round(n * test_ratio)))
    te, tr = order[:n_te], order[n_te:]
    if len(tr) < MIN_TRAIN:
        raise DataError([Issue(None, None, f"학습 창이 {len(tr)}개뿐임", f"창이 {MIN_TRAIN}개 이상 되게 신호를 늘리거나 창 길이를 줄이세요")])
    mu, sd = np.zeros(channels * L), np.ones(channels * L)
    if normalize:
        for c in range(channels):
            part = X[tr][:, c * L:(c + 1) * L]
            mu[c * L:(c + 1) * L], sd[c * L:(c + 1) * L] = float(part.mean()), float(part.std()) or 1.0
    src = dict(source or {}, task="segmentation")
    src.setdefault("normalize", normalize)
    src.setdefault("test_ratio", test_ratio)
    return Dataset((X[tr] - mu) / sd, (X[te] - mu) / sd, Y[tr], Y[te], "segmentation", [str(c) for c in classes],
                   seq_length(L, channels), mu, sd, src, [], None, None, {"segment": True})


def predicted(out: np.ndarray, n: int, K: int) -> np.ndarray:
    """출력 (n, K·L) → 시점마다 상태 번호 (n, L)."""
    return np.asarray(out).reshape(n, K, -1).argmax(1)


def metric(out: np.ndarray, y: np.ndarray, K: int) -> float:
    """시점 정확도: 맞힌 시점의 비율."""
    y = np.asarray(y)
    return float((predicted(out, len(y), K) == y).mean())


def torch_loss(out, target, K: int):
    """시점마다 교차 엔트로피 (평균)."""
    import torch.nn.functional as F
    return F.cross_entropy(out.reshape(len(target), K, -1), target)


def runs(labels) -> list:
    """상태 줄 → [[상태, 시작, 길이], …] (같은 상태가 이어지는 구간)."""
    labels = np.asarray(labels).ravel()
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append([int(labels[s]), s, i - s])
            s = i
    return out


def states_text(labels, classes: list, limit: int = 5) -> str:
    """'대기 → 동작 1–10 → 긴 단계' (구간이 많으면 줄임)."""
    names = [classes[r[0]] for r in runs(labels)]
    return " → ".join(names[:limit]) + (f" … (구간 {len(names)}개)" if len(names) > limit else "")


def result(out: np.ndarray, out_tr: np.ndarray, d) -> dict:
    """결과 화면: 시점 혼동 행렬·상태별 재현율, 상태가 많이 바뀌는 시험 창 몇 개의 신호·정답·예측."""
    K = len(d.classes)
    p = predicted(out, len(d.y_test), K)
    y = np.asarray(d.y_test)
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y.ravel(), p.ravel()), 1)
    rec = [None if cm[k].sum() == 0 else float(cm[k, k] / cm[k].sum()) for k in range(K)]
    order = np.argsort([-len(runs(r)) for r in y], kind="stable")[:N_SHOW]
    raw = d.raw(d.X_test)
    show = [{"index": int(i), "x": np.round(raw[i][:d.length], 4).tolist(), "truth": y[i].tolist(), "pred": p[i].tolist(),
             "acc": float((p[i] == y[i]).mean())} for i in order]
    return {"confusion": cm.tolist(), "recall": rec, "seg_samples": show,
            "train_metric": metric(out_tr, d.y_train, K)}


def table(d, names: list, out: np.ndarray, truth=None) -> dict:
    """⑤ 예측: 창마다 예측한 상태 구간 (정답이 있으면 시점 정확도). CSV는 시점마다 한 줄."""
    K = len(d.classes)
    p = predicted(out, len(names), K)
    t = None if truth is None else np.asarray(truth)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["창", "시점", "예측 상태"] + (["정답 상태"] if t is not None else []))
    rows = []
    for i, nm in enumerate(names):
        for j, k in enumerate(p[i]):
            w.writerow([nm, j, d.classes[k]] + ([d.classes[int(t[i][j])]] if t is not None else []))
        acc = None if t is None else float((p[i] == t[i]).mean())
        rows.append({"name": nm, "pred": states_text(p[i], d.classes), "label": None if t is None else states_text(t[i], d.classes),
                     "ok": None if acc is None else acc >= 0.9, "acc": acc, "seg_pred": p[i].tolist(),
                     "seg_truth": None if t is None else t[i].tolist()})
    acc = None if t is None else float((p == t).mean())
    summary = {"n": len(names), "task": "segmentation", "accuracy": acc, "classes": list(d.classes)}
    return {"summary": summary, "rows": rows, "shown": len(rows), "csv": "﻿" + buf.getvalue(), "has_label": t is not None}


def load_segment_csv(path: str, state_col=None, skip=None, length=None, normalize: bool = True, test_ratio: float = 0.2,
                     seed: int = 0):
    """긴 시계열 CSV (한 행 = 한 시점, 시간 순서) → 구간 분할 Dataset. state_col: 상태 열 번호 (없으면 마지막 열),
    값 열 = 나머지 숫자 열 (글자뿐인 열·skip은 안 씀, 여러 개면 채널). length: 창 길이 (없으면 시점 수에 맞춰), 겹치지 않게 자름."""
    from dlstudio.data import DataError, Issue, _column_values, _numeric_columns, _num, _read_table
    name = Path(path).name
    names, rows = _read_table(path)
    if not rows or len(names) < 2:
        raise DataError([Issue(None, None, "열이 2개 이상 있어야 함 (값 열 + 상태 열)", "한 행 = 한 시점, 마지막 열 = 그 시점의 상태")])
    sc = len(names) - 1 if state_col is None else int(state_col)
    if not 0 <= sc < len(names):
        raise DataError([Issue(None, None, f"상태 열 번호 {sc + 1}이(가) 없음", "역할 표에서 상태 열을 다시 고르세요")])
    skips = {int(j) for j in skip or []}
    numeric = set(_numeric_columns(names, rows))
    cols = [j for j in range(len(names)) if j != sc and j not in skips and names[j] in numeric]
    if not cols:
        raise DataError([Issue(None, None, "값(숫자) 열이 없음", "측정값이 숫자로 든 열이 있어야 합니다")])
    V = np.stack([_column_values(name, names, rows, names[j]) for j in cols])
    states = [(r[sc] if sc < len(r) else "").strip() for _, r in rows]
    bad = next((i for i, v in enumerate(states) if not v), None)
    if bad is not None:
        raise DataError([Issue(rows[bad][0], sc + 1, f"{rows[bad][0]}행의 상태 칸이 비어 있음", "상태를 채우거나 그 행을 지우세요")])
    uniq = sorted(set(states), key=lambda v: (_num(v) is None, _num(v) if _num(v) is not None else 0, v))
    if len(uniq) < 2:
        raise DataError([Issue(None, None, "상태가 한 가지뿐임", "구간 분할에는 상태가 두 가지 이상 있어야 합니다")])
    if len(uniq) > 50:
        raise DataError([Issue(None, None, f"상태가 {len(uniq)}가지 (50가지 넘음)", "상태 열이 맞는지 확인하세요 (숫자 측정값 열이 아니어야 함)")])
    idx = {v: k for k, v in enumerate(uniq)}
    y = np.asarray([idx[v] for v in states], dtype=np.int64)
    T = len(y)
    L = int(length) if length else int(min(512, max(32, (T // 20) // 8 * 8)))
    n = T // L
    if L < 8 or n < MIN_TRAIN + 2:
        raise DataError([Issue(None, None, f"시점 {T:,}개를 길이 {L} 창으로 자르면 {n}개 (최소 {MIN_TRAIN + 2}개)",
                               "창 길이를 줄이거나 더 긴 기록을 넣으세요")])
    X = np.stack([np.concatenate([V[c, i * L:(i + 1) * L] for c in range(len(cols))]) for i in range(n)])
    Y = np.stack([y[i * L:(i + 1) * L] for i in range(n)])
    used = set(cols) | {sc}
    src = {"path": str(path), "task": "segmentation", "label_col": sc, "targets": [sc],
           "skip": [j for j in range(len(names)) if j not in used], "length": L, "columns": list(names),
           "channel_names": [names[j] for j in cols], "seed": seed}
    return make_segment_dataset(X, Y, uniq, test_ratio=test_ratio, seed=seed, normalize=normalize, source=src, channels=len(cols))

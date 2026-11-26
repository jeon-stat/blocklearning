"""정답 여러 개: 한 입력으로 여러 정답(분류·값 섞어서)을 함께 맞힌다 (docs/design-recipes.md 4).

데이터: y는 (샘플, 정답 수) 표 — 분류 정답은 클래스 번호, 값 정답은 숫자. Dataset.extra["targets"] =
[{"name", "kind": "class"|"value", "classes": [...]}].
모델: 출력 블록 하나가 정답마다의 머리를 정해진 순서로 이어 붙인 한 줄을 냄 (분류 정답은 클래스 수, 값 정답은 1).
손실: 정답마다의 항 (분류 교차 엔트로피 · 값 MSE, 정규화한 값) 을 더함 — 손실 조립(losses.py)에서 정답마다 바꿀 수 있음.
성적: 정답마다 (분류 정확도 · 값 R²), 전체는 그 평균.
"""
from __future__ import annotations

import csv
import io

import numpy as np

N_SCATTER = 200


class MultiOut(int):
    """정답 여러 개의 모델 출력 크기 (머리 크기의 합). int처럼 쓰이고 머리 목록을 함께 지님:
    heads = [{"name", "kind", "classes", "start", "size"}]."""

    def __new__(cls, targets: list):
        heads, start = [], 0
        for t in targets:
            size = len(t["classes"]) if t["kind"] == "class" else 1
            heads.append({"name": str(t["name"]), "kind": t["kind"], "classes": [str(c) for c in t.get("classes") or []],
                          "start": start, "size": size})
            start += size
        o = super().__new__(cls, start)
        o.heads, o.multi = heads, True
        return o

    def __getnewargs__(self):
        return (self.targets(),)

    def __reduce__(self):
        return (MultiOut, (self.targets(),))

    def targets(self) -> list:
        return [{"name": h["name"], "kind": h["kind"], "classes": h["classes"]} for h in self.heads]


def make_multi_dataset(X, Y, targets: list, test_ratio: float = 0.2, seed: int = 0, normalize: bool = True,
                       source: dict | None = None, features: list | None = None):
    """입력 (n, D) · 정답 표 (n, T) → 정답 여러 개 Dataset. 나누기는 무작위 (정답이 여럿이라 비율은 맞추지 않음)."""
    from blocklearning.data import MIN_SAMPLES, DataError, Dataset, Issue
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64).reshape(len(X), -1)
    n = len(X)
    if n < MIN_SAMPLES:
        raise DataError([Issue(None, None, f"샘플이 {n}개뿐임 (최소 {MIN_SAMPLES}개)", "샘플을 더 모아 주세요")])
    if Y.shape[1] != len(targets) or len(targets) < 2:
        raise DataError([Issue(None, None, "정답 열이 둘 이상이어야 함", "정답이 하나면 분류·회귀 과제를 쓰세요")])
    order = np.random.default_rng(seed).permutation(n)
    n_te = max(1, int(round(n * test_ratio)))
    te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])
    if normalize:
        mean, std = X[tr].mean(0), X[tr].std(0)
        std = np.where(std > 0, std, 1.0)
    else:
        mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])
    src = dict(source or {}, task="multi")
    src.setdefault("normalize", normalize)
    src.setdefault("test_ratio", test_ratio)
    src.setdefault("seed", seed)
    tg = [{"name": str(t["name"]), "kind": t["kind"], "classes": [str(c) for c in t.get("classes") or []]} for t in targets]
    return Dataset((X[tr] - mean) / std, (X[te] - mean) / std, Y[tr], Y[te], "multi", [], X.shape[1], mean, std, src,
                   list(features or []), None, None, {"targets": tg})


def heads_of(d) -> list:
    return d.n_out.heads


def target_scale(Y: np.ndarray, heads: list) -> tuple:
    """출력 한 줄의 칸마다 (평균, 표준편차): 값 정답 칸은 학습 몫의 평균·표준편차, 분류 칸은 (0, 1) (되돌리지 않음).
    정답 표 Y (n, T) → 학습에 쓸 정답 (값 정답만 정규화)도 함께."""
    n_out = sum(h["size"] for h in heads)
    mu, sd = np.zeros(n_out), np.ones(n_out)
    Yf = np.array(Y, dtype=np.float64).reshape(len(Y), -1)
    for j, h in enumerate(heads):
        if h["kind"] == "value":
            m, s = float(Yf[:, j].mean()), float(Yf[:, j].std()) or 1.0
            mu[h["start"]], sd[h["start"]] = m, s
            Yf[:, j] = (Yf[:, j] - m) / s
    return mu, sd, Yf


def head_metric(o: np.ndarray, y: np.ndarray, h: dict) -> float:
    """머리 하나의 성적: 분류 정확도, 값 R²."""
    o = np.asarray(o)[:, h["start"]:h["start"] + h["size"]]
    y = np.asarray(y, dtype=np.float64)
    if h["kind"] == "class":
        return float(np.mean(o.argmax(1) == y.astype(np.int64)))
    var = float(np.var(y))
    return 1.0 - float(np.mean((o[:, 0] - y) ** 2)) / var if var > 0 else float("nan")


def metric(out: np.ndarray, Y: np.ndarray, heads: list) -> float:
    """전체 성적: 머리마다 성적의 평균 (정확도와 R²는 모두 1이 가장 좋음)."""
    Y = np.asarray(Y).reshape(len(Y), -1)
    vals = [head_metric(out, Y[:, j], h) for j, h in enumerate(heads)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def torch_loss(out, target, heads: list):
    """기본 손실: 분류 머리는 교차 엔트로피, 값 머리는 MSE (정규화한 값), 모두 더함."""
    import torch.nn.functional as F
    total = 0
    for j, h in enumerate(heads):
        o = out[:, h["start"]:h["start"] + h["size"]]
        if h["kind"] == "class":
            total = total + F.cross_entropy(o, target[:, j].long())
        else:
            total = total + ((o[:, 0] - target[:, j]) ** 2).mean()
    return total


def result(out: np.ndarray, out_tr: np.ndarray, d) -> dict:
    """결과 화면: 머리마다 성적과 그 정답 종류의 화면 (분류: 혼동 행렬·재현율, 값: MAE·예측 vs 실제)."""
    from blocklearning.jobs import confusion, regression_view
    heads = heads_of(d)
    Y, Ytr = np.asarray(d.y_test), np.asarray(d.y_train)
    views = []
    for j, h in enumerate(heads):
        o = np.asarray(out)[:, h["start"]:h["start"] + h["size"]]
        v = {"name": h["name"], "kind": h["kind"], "metric": head_metric(out, Y[:, j], h),
             "train_metric": head_metric(out_tr, Ytr[:, j], h)}
        if h["kind"] == "class":
            v["classes"] = h["classes"]
            v["confusion"], v["recall"] = confusion(o.argmax(1), Y[:, j].astype(np.int64), len(h["classes"]))
        else:
            v["mae"], v["scatter"] = regression_view(o, Y[:, j])
        views.append(v)
    return {"heads": views, "train_metric": metric(out_tr, Ytr, heads)}


def label_text(y_row, heads: list) -> str:
    """정답 한 줄 → '냉각기 정상 · 밸브 100 · …' (화면용)."""
    parts = []
    for j, h in enumerate(heads):
        v = y_row[j]
        parts.append(f"{h['name']} {h['classes'][int(v)] if h['kind'] == 'class' else f'{float(v):.4g}'}")
    return " · ".join(parts)


def pred_values(out_row, heads: list) -> list:
    """출력 한 줄 → 머리마다 예측 (클래스 이름 또는 값)."""
    return [h["classes"][int(np.argmax(out_row[h["start"]:h["start"] + h["size"]]))] if h["kind"] == "class"
            else round(float(out_row[h["start"]]), 6) for h in heads]


def table(d, names: list, out: np.ndarray, truth=None) -> dict:
    """⑤ 예측: 샘플마다 머리마다의 예측 (정답이 있으면 함께). CSV는 샘플마다 한 줄."""
    heads = heads_of(d)
    t = None if truth is None else np.asarray(truth, dtype=np.float64).reshape(len(names), -1)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["샘플"] + [f"예측 {h['name']}" for h in heads] + ([f"정답 {h['name']}" for h in heads] if t is not None else []))
    rows = []
    for i, nm in enumerate(names):
        p = pred_values(out[i], heads)
        lab = None if t is None else [h["classes"][int(t[i][j])] if h["kind"] == "class" else round(float(t[i][j]), 6)
                                       for j, h in enumerate(heads)]
        w.writerow([nm] + p + (lab or []))
        rows.append({"name": nm, "pred": " · ".join(f"{h['name']} {v:g}" if isinstance(v, float) else f"{h['name']} {v}"
                                                   for h, v in zip(heads, p)),
                     "label": None if lab is None else " · ".join(f"{h['name']} {v:g}" if isinstance(v, float) else f"{h['name']} {v}"
                                                                  for h, v in zip(heads, lab)),
                     "multi_pred": p, "multi_label": lab,
                     "ok": None if lab is None else all(a == b for h, a, b in zip(heads, p, lab) if h["kind"] == "class")})
    summary = {"n": len(names), "task": "multi", "heads": [{"name": h["name"], "kind": h["kind"]} for h in heads],
               "score": None if t is None else metric(out, t, heads)}
    return {"summary": summary, "rows": rows, "shown": len(rows), "csv": "﻿" + buf.getvalue(), "has_label": t is not None}

"""이상 탐지 (오토인코더): 정상 샘플만으로 '입력을 그대로 되살리기(복원)'를 배우고, 잘 못 되살리는 샘플을 이상으로 본다.

데이터는 분류처럼 정답(정상·이상 종류)이 붙은 것에서 만든다. 정상 클래스 하나를 정하고 나머지는 모두 '이상'.
학습에는 학습 몫의 정상만 쓰고, 학습 몫에 있던 이상 샘플은 시험으로 옮긴다 (평가에 쓸 이상이 늘어남).
정상만 있는 데이터도 된다 (그때는 AUC 대신 정상의 복원 오차만 보여 줌).
이상 점수 = 정규화한 값에서 (입력 − 복원)²의 평균. 기준값(이 값보다 크면 이상)은 학습 정상 점수의 95% 지점.

출력 블록을 '정상과의 거리'로 하면 (nn_extra.NormalDistance) 되살리지 않고, 정상 샘플의 특징 분포(평균·주성분)에서
떨어진 정도를 점수로 바로 냄 (모델 출력이 점수 하나). 사전학습 블록 바로 뒤에 둠: 그림 이상 탐지에서 훨씬 잘 됨.
"""
from __future__ import annotations

import numpy as np

NORMAL_NAMES = ("정상", "normal", "ok", "good", "양품", "0", "false", "n")
QUANTILE = 0.95           # 기준값: 학습 정상의 95%가 이 아래
N_SHOW = 6                # 결과 화면에 그릴 샘플 (점수 높은 것부터)


def pick_normal(classes: list, y: np.ndarray, normal=None) -> int:
    """정상 클래스 번호: 고른 이름 → 이름이 정상·normal·양품 같은 것 → 가장 많은 것."""
    names = [str(c) for c in classes]
    if normal is not None and str(normal) in names:
        return names.index(str(normal))
    for k, c in enumerate(names):
        if c.strip().lower() in NORMAL_NAMES:
            return k
    return int(np.bincount(np.asarray(y, dtype=int), minlength=len(names)).argmax())


def to_anomaly(d, normal=None):
    """분류 Dataset → 이상 탐지 Dataset (task "anomaly", classes [정상 이름, "이상"], y 0 정상 · 1 이상)."""
    from blocklearning.data import DataError, Dataset, Issue
    y_all = np.concatenate([d.y_train, d.y_test]) if len(d.classes) else np.zeros(len(d.X_train) + len(d.X_test), int)
    names = [str(c) for c in d.classes] or ["정상"]
    k = pick_normal(names, y_all, normal)
    raw_tr, raw_te = d.raw(d.X_train), d.raw(d.X_test)
    ab_tr = np.asarray(d.y_train) != k if len(d.classes) else np.zeros(len(raw_tr), bool)
    ab_te = np.asarray(d.y_test) != k if len(d.classes) else np.zeros(len(raw_te), bool)
    Xtr = raw_tr[~ab_tr]
    Xte = np.concatenate([raw_te, raw_tr[ab_tr]])
    yte = np.concatenate([ab_te, np.ones(int(ab_tr.sum()), bool)]).astype(np.int64)
    if len(Xtr) < 10:
        raise DataError([Issue(None, None, f"학습에 쓸 정상 샘플이 {len(Xtr)}개뿐임 (정상 = '{names[k]}')",
                               "정상 샘플을 10개 이상 모으거나, 정상 클래스를 다시 고르세요")])
    norm = (d.source or {}).get("normalize", True)
    if norm and d.image:
        m, s = float(Xtr.mean()), float(Xtr.std()) or 1.0
        mean, std = np.full(Xtr.shape[1], m), np.full(Xtr.shape[1], s)
    elif norm:
        mean, std = Xtr.mean(0), Xtr.std(0)
        std = np.where(std > 0, std, 1.0)
    else:
        mean, std = np.zeros(Xtr.shape[1]), np.ones(Xtr.shape[1])
    src = dict(d.source or {}, task="anomaly", normal=names[k])
    others = [c for i, c in enumerate(names) if i != k]
    extra = {"normal": names[k], "others": others, "n_anomaly": int(yte.sum())}
    return Dataset((Xtr - mean) / std, (Xte - mean) / std, np.zeros(len(Xtr), np.int64), yte, "anomaly",
                   [names[k], "이상"], d.length, mean, std, src, list(d.features), d.image, None, extra)


def distance_mode(out: np.ndarray, X: np.ndarray) -> bool:
    """모델이 점수 하나를 바로 냄 (출력 '정상과의 거리')."""
    return np.asarray(out).reshape(len(X), -1).shape[1] == 1 and np.asarray(X).reshape(len(X), -1).shape[1] > 1


def scores(out: np.ndarray, X: np.ndarray) -> np.ndarray:
    """샘플마다 이상 점수 = 정규화한 값에서 (입력 − 복원)²의 평균. 출력이 '정상과의 거리'면 그 값."""
    o = np.asarray(out, dtype=float).reshape(len(X), -1)
    if distance_mode(o, X):
        return o[:, 0]
    return ((o - np.asarray(X, dtype=float).reshape(len(X), -1)) ** 2).mean(1)


def fit_distance(F: np.ndarray, k: int) -> dict:
    """정상 샘플의 특징 F (n, C) → '정상과의 거리' 값: 평균, 분산이 큰 방향(주성분) k개, 그 분산의 역수, 나머지 방향 평균 분산의 역수.
    점수 = (주성분 방향마다 거리²/분산의 합 + 나머지 거리²/나머지 분산) / C — Mahalanobis 거리를 주성분 k개로 줄인 것."""
    F = np.asarray(F, dtype=np.float64)
    n, C = F.shape
    mu = F.mean(0)
    D = F - mu
    lam, U = np.linalg.eigh(D.T @ D / max(n, 1))
    lam, U = np.clip(lam[::-1], 0, None), U[:, ::-1]
    eps = 1e-3 * max(float(lam[0]), 1e-12)
    m = min(k, C, max(n - 1, 1))
    basis, inv_var = np.zeros((k, C)), np.zeros(k)
    basis[:m], inv_var[:m] = U[:, :m].T, 1.0 / (lam[:m] + eps)
    rest = float(lam[m:].mean()) if C > m else 0.0
    return {"mean": mu.astype(np.float32), "basis": basis.astype(np.float32), "inv_var": inv_var.astype(np.float32),
            "inv_rest": np.array([1.0 / (rest + eps)], np.float32)}


def auc(s: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC: 이상 하나와 정상 하나를 뽑았을 때 이상의 점수가 더 클 확률 (같으면 반). 둘 중 하나가 없으면 NaN."""
    s, y = np.asarray(s, dtype=float), np.asarray(y, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    allv = np.concatenate([neg, pos])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):                                   # 같은 값은 평균 순위
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metric(out: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """학습 곡선의 성적: 이상이 있으면 AUC, 정상만 있으면 평균 이상 점수 (낮을수록 잘 되살림)."""
    s = scores(out, X)
    return auc(s, y) if (np.asarray(y) == 1).any() and (np.asarray(y) == 0).any() else float(s.mean())


def metric_name(d) -> str:
    return "auc" if (np.asarray(d.y_test) == 1).any() and (np.asarray(d.y_test) == 0).any() else "recon"


def at_threshold(s: np.ndarray, y: np.ndarray, th: float) -> dict:
    """기준값으로 나눈 결과: 이상을 이상으로(tp)·정상을 이상으로(fp)·이상을 놓침(fn)·정상을 정상으로(tn)."""
    p, y = np.asarray(s) > th, np.asarray(y) == 1
    tp, fp, fn, tn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def result(out: np.ndarray, out_train: np.ndarray, d) -> dict:
    """결과 화면: 점수(시험 전체·학습 정상), 기준값, AUC, 기준값에서 나눈 결과, 점수 높은 샘플의 입력·복원."""
    s = scores(out, d.X_test)
    s_tr = scores(out_train, d.X_train)
    th = float(np.quantile(s_tr, QUANTILE))
    o = np.asarray(out, dtype=float).reshape(len(d.X_test), -1)
    dist = distance_mode(o, d.X_test)
    show = [int(i) for i in np.argsort(-s)[:N_SHOW]]
    normal_low = [int(i) for i in np.argsort(s)[:2] if int(i) not in show]       # 비교용: 가장 잘 되살린 정상
    r5 = lambda a: [round(float(x), 5) for x in a]
    raw = d.raw(d.X_test)
    rec = None if dist else o * d.std + d.mean
    return {"scores": r5(s), "train_scores": r5(s_tr), "threshold": th, "labels": [int(v) for v in d.y_test],
            "auc": auc(s, d.y_test), "at": at_threshold(s, d.y_test, th), "mode": "distance" if dist else "recon",
            "ae_samples": [{"index": i, "score": round(float(s[i]), 5), "label": int(d.y_test[i]), "x": r5(raw[i]),
                            "recon": None if dist else r5(rec[i]), "err": None if dist else r5((o[i] - d.X_test[i]) ** 2)}
                           for i in show + normal_low]}

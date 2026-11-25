"""손실 조립: 손실 = Σ 가중치 × 항. 정답 종류마다 고를 수 있는 항만 (docs/design-recipes.md).

항: {"term": 이름, "w": 가중치, ...설정}. 설정이 없으면 과제의 기본 (지금까지와 같은 손실).
PyTorch(여기·jobs.fit)·웹 엔진(static/engine.js lossTerms)·내보내는 코드(codegen)가 같은 식을 쓴다.
교차 엔트로피·Focal은 직접 계산: 샘플(시점)마다 −Σ q·log p (q = 라벨 스무딩한 정답), 클래스 가중치를 쓰면
Σ w·손실 / Σ w (정답 클래스의 가중치로 가중 평균).
"""
from __future__ import annotations

import numpy as np

# 정답 종류 → 고를 수 있는 항 (첫 것이 기본). 모든 과제에 l2·l1 (가중치 크기)도 더할 수 있음
TERMS = {
    "class": ["ce", "focal"],
    "value": ["mse", "mae", "huber"],
    "seq_class": ["ce", "dice"],
    "boxes": ["det"],
}
COMMON = ["l2", "l1"]
NAMES = {"ce": "교차 엔트로피", "focal": "Focal", "mse": "MSE", "mae": "MAE", "huber": "Huber", "dice": "Dice",
         "det": "탐지 손실", "l2": "가중치 L2", "l1": "가중치 L1"}
# 설정: 이름 → (기본값, 최소, 최대)
OPTIONS = {"ce": {"smooth": (0.0, 0.0, 0.5), "balance": (False, None, None)},
           "focal": {"gamma": (2.0, 0.0, 5.0), "balance": (False, None, None)},
           "huber": {"delta": (1.0, 0.01, 10.0)},
           "dice": {}, "mse": {}, "mae": {}, "det": {}, "l2": {}, "l1": {}}
W_RANGE = (0.0, 100.0)


def target_kind(task: str) -> str:
    return {"classification": "class", "segmentation": "seq_class", "detection": "boxes"}.get(task, "value")


def allowed(task: str) -> list:
    return TERMS[target_kind(task)] + COMMON


def default(task: str) -> list:
    return [{"term": TERMS[target_kind(task)][0], "w": 1.0}]


def normalize(spec, task: str) -> list:
    """학습 설정의 손실 → 검사한 항 목록 (없거나 비면 기본). 맞지 않는 항·값이면 ValueError (화면이 알림)."""
    spec = spec or default(task)
    out, ok = [], allowed(task)
    for t in spec:
        name = str(t.get("term"))
        if name not in ok:
            raise ValueError(f"'{NAMES.get(name, name)}' 손실은 이 과제에 쓸 수 없음 (쓸 수 있는 것: {', '.join(NAMES[k] for k in ok)})")
        w = float(t.get("w", 1.0))
        if not W_RANGE[0] <= w <= W_RANGE[1]:
            raise ValueError(f"{NAMES[name]}: 가중치 {w}이(가) {W_RANGE[0]}–{W_RANGE[1]} 밖")
        item = {"term": name, "w": w}
        for k, (d, lo, hi) in OPTIONS[name].items():
            v = t.get(k, d)
            if isinstance(d, bool):
                v = bool(v)
            else:
                v = float(v)
                if not lo <= v <= hi:
                    raise ValueError(f"{NAMES[name]}: {k} {v}이(가) {lo}–{hi} 밖")
            item[k] = v
        out.append(item)
    if not any(t["term"] not in COMMON for t in out):
        raise ValueError("정답과 맞추는 손실이 하나는 있어야 함 (가중치 L2·L1만으로는 학습할 수 없음)")
    return out


def is_default(spec: list, task: str) -> bool:
    return spec == normalize(default(task), task)


def class_weights(y, K: int) -> np.ndarray:
    """드문 클래스를 무겁게: 가중치 = 전체 수 / (K × 그 클래스 수). 없는 클래스는 0."""
    c = np.bincount(np.asarray(y, dtype=np.int64).ravel(), minlength=K).astype(np.float64)
    w = np.where(c > 0, c.sum() / (K * np.maximum(c, 1)), 0.0)
    return w.astype(np.float32)


def label(t: dict) -> str:
    """'교차 엔트로피 ×1' 꼴 (화면·학습 곡선)."""
    return f"{NAMES[t['term']]} ×{t['w']:g}"


# ---------------------------------------------------------------- PyTorch

def _ce(logp, t, K, smooth, cw):
    """logp (M, K) · 정답 t (M,) → 가중 평균 교차 엔트로피 (라벨 스무딩)."""
    import torch
    q = torch.nn.functional.one_hot(t, K).to(logp.dtype) * (1 - smooth) + smooth / K
    per = -(q * logp).sum(1)
    if cw is None:
        return per.mean()
    w = cw[t]
    return (per * w).sum() / w.sum().clamp_min(1e-12)


def _focal(logp, t, K, gamma, cw):
    import torch
    lp = logp.gather(1, t[:, None])[:, 0]
    per = -((1 - lp.exp()).clamp_min(0) ** gamma) * lp
    if cw is None:
        return per.mean()
    w = cw[t]
    return (per * w).sum() / w.sum().clamp_min(1e-12)


def build(spec: list, task: str, n_cls: int = 0, cw=None, det=None, seg_K: int = 0):
    """항 목록 → loss_fn(out, target, net) → (전체 손실, {항 이름: 값}). cw: 클래스 가중치 (balance)."""
    import torch
    import torch.nn.functional as F

    def per_term(t, out, y, net):
        n = t["term"]
        if n in ("ce", "focal"):
            if task == "segmentation":
                logp = F.log_softmax(out.reshape(len(y), seg_K, -1), 1).transpose(1, 2).reshape(-1, seg_K)
                yy, K = y.reshape(-1), seg_K
            else:
                logp, yy, K = F.log_softmax(out, 1), y, n_cls
            w = cw if t.get("balance") else None
            return _ce(logp, yy, K, t.get("smooth", 0.0), w) if n == "ce" else _focal(logp, yy, K, t["gamma"], w)
        if n == "dice":                                     # 상태마다 1 − (2·겹침 + 1) / (예측 합 + 정답 합 + 1), 평균
            p = F.softmax(out.reshape(len(y), seg_K, -1), 1)
            oh = F.one_hot(y, seg_K).permute(0, 2, 1).to(p.dtype)
            inter, tot = (p * oh).sum((0, 2)), p.sum((0, 2)) + oh.sum((0, 2))
            return (1 - (2 * inter + 1) / (tot + 1)).mean()
        if n == "mse":
            return ((out - y) ** 2).mean()
        if n == "mae":
            return (out - y).abs().mean()
        if n == "huber":
            e, d = (out - y).abs(), t["delta"]
            return torch.where(e < d, 0.5 * e * e, d * (e - 0.5 * d)).mean()
        if n == "det":
            from dlstudio.detect import torch_loss
            return torch_loss(out, y, det.grid, det.per_cell)
        ps = [p for p in net.parameters() if p.requires_grad]
        return sum((p * p).sum() for p in ps) if n == "l2" else sum(p.abs().sum() for p in ps)

    def loss_fn(out, y, net):
        parts = {}
        total = 0
        for t in spec:
            v = per_term(t, out, y, net)
            parts[label(t)] = v
            total = total + t["w"] * v
        return total, parts
    return loss_fn

"""손실 조립: 손실 = Σ 가중치 × 항. 정답 종류마다 고를 수 있는 항만 (docs/design-recipes.md).

항: {"term": 이름, "w": 가중치, ...설정}. 설정이 없으면 과제의 기본 (지금까지와 같은 손실).
정답이 여러 개면 (과제 "multi", multi.py) 정답 항마다 "head": 정답 이름 — 그 정답의 머리(출력 칸)와 정답 열에 걸림.
가중치 L2·L1은 정답과 상관없음 (head 없음).
PyTorch(여기·jobs.fit)·웹 엔진(static/engine.js buildLoss)·내보내는 코드(codegen)가 같은 식을 쓴다.
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
COMMON = ["l2", "l1", "act_l1"]                 # act_l1: 고른 블록 출력의 크기 (희소한 특징) — "node": 블록 id
NAMES = {"ce": "교차 엔트로피", "focal": "Focal", "mse": "MSE", "mae": "MAE", "huber": "Huber", "dice": "Dice",
         "det": "탐지 손실", "l2": "가중치 L2", "l1": "가중치 L1", "act_l1": "블록 출력 L1"}
# 설정: 이름 → (기본값, 최소, 최대)
OPTIONS = {"ce": {"smooth": (0.0, 0.0, 0.5), "balance": (False, None, None)},
           "focal": {"gamma": (2.0, 0.0, 5.0), "balance": (False, None, None)},
           "huber": {"delta": (1.0, 0.01, 10.0)},
           "dice": {}, "mse": {}, "mae": {}, "det": {}, "l2": {}, "l1": {}, "act_l1": {"node": ("", None, None)}}
W_RANGE = (0.0, 100.0)


def target_kind(task: str) -> str:
    return {"classification": "class", "segmentation": "seq_class", "detection": "boxes"}.get(task, "value")


def allowed(task: str, kind: str = None) -> list:
    """고를 수 있는 항. 정답이 여러 개면 정답마다 (kind = 그 정답의 종류), 공통 항은 kind 없이 COMMON."""
    if task == "multi" and kind is None:
        return list(COMMON)
    return TERMS[kind or target_kind(task)] + COMMON


def default(task: str, heads: list = None) -> list:
    if heads:
        return [{"term": TERMS[h["kind"]][0], "w": 1.0, "head": h["name"]} for h in heads]
    return [{"term": TERMS[target_kind(task)][0], "w": 1.0}]


def normalize(spec, task: str, heads: list = None) -> list:
    """학습 설정의 손실 → 검사한 항 목록 (없거나 비면 기본). 맞지 않는 항·값이면 ValueError (화면이 알림).
    heads: 정답이 여러 개일 때 머리 목록 (multi.MultiOut.heads)."""
    spec = spec or default(task, heads)
    byname = {h["name"]: h for h in heads or []}
    out = []
    for t in spec:
        name = str(t.get("term"))
        head = None
        if heads and name not in COMMON:
            head = t.get("head")
            if head not in byname:
                raise ValueError(f"'{head}' 정답이 없음 (정답: {', '.join(byname)})")
            ok = allowed(task, byname[head]["kind"])
        else:
            ok = allowed(task)
        if name not in ok:
            where = f"정답 '{head}'" if head else "이 과제"
            raise ValueError(f"'{NAMES.get(name, name)}' 손실은 {where}에 쓸 수 없음 (쓸 수 있는 것: {', '.join(NAMES[k] for k in ok)})")
        w = float(t.get("w", 1.0))
        if not W_RANGE[0] <= w <= W_RANGE[1]:
            raise ValueError(f"{NAMES[name]}: 가중치 {w}이(가) {W_RANGE[0]}–{W_RANGE[1]} 밖")
        item = {"term": name, "w": w}
        if head is not None:
            item["head"] = head
        for k, (d, lo, hi) in OPTIONS[name].items():
            v = t.get(k, d)
            if isinstance(d, bool):
                v = bool(v)
            elif isinstance(d, str):
                v = str(v or "")
                if not v:
                    raise ValueError(f"{NAMES[name]}: 블록을 고르세요")
            else:
                v = float(v)
                if not lo <= v <= hi:
                    raise ValueError(f"{NAMES[name]}: {k} {v}이(가) {lo}–{hi} 밖")
            item[k] = v
        out.append(item)
    if not any(t["term"] not in COMMON for t in out):
        raise ValueError("정답과 맞추는 손실이 하나는 있어야 함 (가중치 L2·L1만으로는 학습할 수 없음)")
    missing = [h["name"] for h in heads or [] if not any(t.get("head") == h["name"] for t in out)]
    if missing:
        raise ValueError(f"정답 '{missing[0]}'에 손실 항이 없음 (그 정답의 출력을 배우지 못함)")
    return out


def is_default(spec: list, task: str, heads: list = None) -> bool:
    return spec == normalize(default(task, heads), task, heads)


def class_weights(y, K: int) -> np.ndarray:
    """드문 클래스를 무겁게: 가중치 = 전체 수 / (K × 그 클래스 수). 없는 클래스는 0."""
    c = np.bincount(np.asarray(y, dtype=np.int64).ravel(), minlength=K).astype(np.float64)
    w = np.where(c > 0, c.sum() / (K * np.maximum(c, 1)), 0.0)
    return w.astype(np.float32)


def multi_class_weights(Y, heads: list, spec: list) -> dict:
    """정답 여러 개: '드문 클래스 무겁게'를 켠 분류 정답마다 {정답 이름: 클래스 가중치}."""
    Y = np.asarray(Y).reshape(len(Y), -1)
    want = {t.get("head") for t in spec if t.get("balance")}
    return {h["name"]: class_weights(Y[:, j], len(h["classes"])) for j, h in enumerate(heads)
            if h["kind"] == "class" and h["name"] in want}


def label(t: dict) -> str:
    """'교차 엔트로피 ×1' 꼴 (화면·학습 곡선). 정답이 여러 개면 '냉각기 · 교차 엔트로피 ×1'."""
    node = f" ({t['node']})" if t.get("node") else ""
    return (f"{t['head']} · " if t.get("head") else "") + f"{NAMES[t['term']]}{node} ×{t['w']:g}"


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
    lp = logp.gather(1, t[:, None])[:, 0]
    per = -((1 - lp.exp()).clamp_min(0) ** gamma) * lp
    if cw is None:
        return per.mean()
    w = cw[t]
    return (per * w).sum() / w.sum().clamp_min(1e-12)


def build(spec: list, task: str, n_cls: int = 0, cw=None, det=None, seg_K: int = 0, heads: list = None):
    """항 목록 → loss_fn(out, target, net) → (전체 손실, {항 이름: 값}). cw: 클래스 가중치 (balance) —
    정답이 여러 개면 {정답 이름: 가중치}. heads: 정답이 여러 개일 때 머리 목록 (target은 (샘플, 정답 수))."""
    import torch
    import torch.nn.functional as F
    col = {h["name"]: (j, h) for j, h in enumerate(heads or [])}

    def per_term(t, out, y, net):
        n = t["term"]
        if n == "act_l1":                                   # 고른 블록의 출력 크기 평균 (학습 때 모델이 노드마다 값을 담아 둠)
            acts = getattr(net, "capture", None) or {}
            if t["node"] not in acts:
                raise ValueError(f"'{t['node']}' 블록의 값을 찾을 수 없음 (없는 블록이거나 고정한 사전학습 블록 안)")
            return acts[t["node"]].abs().mean()
        if n in ("l2", "l1"):
            ps = [p for p in net.parameters() if p.requires_grad]
            return sum((p * p).sum() for p in ps) if n == "l2" else sum(p.abs().sum() for p in ps)
        K, w = n_cls, cw
        if t.get("head") is not None:                       # 정답 여러 개: 그 정답의 머리 칸·정답 열만
            j, h = col[t["head"]]
            out = out[:, h["start"]:h["start"] + h["size"]]
            if h["kind"] == "class":
                y, K, w = y[:, j].long(), h["size"], (cw or {}).get(h["name"])
            else:
                out, y = out[:, 0], y[:, j]
        if n in ("ce", "focal"):
            if task == "segmentation":
                logp = F.log_softmax(out.reshape(len(y), seg_K, -1), 1).transpose(1, 2).reshape(-1, seg_K)
                yy, K = y.reshape(-1), seg_K
            else:
                logp, yy = F.log_softmax(out, 1), y
            w = w if t.get("balance") else None
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
        from dlstudio.detect import torch_loss                # det
        return torch_loss(out, y, det.grid, det.per_cell)

    def loss_fn(out, y, net):
        parts = {}
        total = 0
        for t in spec:
            v = per_term(t, out, y, net)
            parts[label(t)] = v
            total = total + t["w"] * v
        return total, parts
    return loss_fn

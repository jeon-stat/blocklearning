"""모델 정보(파라미터 수·크기)와 블록 사이 실제 값(activations)."""
from __future__ import annotations

import numpy as np

MAX_POINTS = 200          # 블록 사이 그림의 시점 수 상한


def model_panel(model) -> dict:
    """③ 모델 옆 '모델 정보': 파라미터 수와 크기(MB, float32 기준)."""
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    frozen = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))     # 사전학습 (고정)
    return {"ok": True, "params": params, "frozen": frozen, "mb": (params + frozen) * 4 / 1e6, "message": "", "warnings": []}


def _thin(a: np.ndarray) -> list:
    if a.ndim == 2 and a.shape[1] > MAX_POINTS:
        idx = np.round(np.linspace(0, a.shape[1] - 1, MAX_POINTS)).astype(int)
        a = a[:, idx]
    return np.round(a.astype(np.float64), 5).tolist()


def activations(model, x) -> dict:
    """샘플 하나(모델 입력 모양, 배치 없음)를 넣어 블록마다 출력 값. 키는 model.block_modules의 키."""
    import torch
    if hasattr(model, "trace"):                                 # 그래프 모델: 모든 노드 (Add·Concat 포함)
        model.eval()
        dtype = next(model.parameters()).dtype
        with torch.no_grad():
            vals = model.trace(torch.as_tensor(np.asarray(x)[None], dtype=dtype))
        return {k: _thin(v[0].detach().cpu().numpy()) for k, v in vals.items()}
    seen: dict = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, k=k: seen.__setitem__(k, o))
             for k, m in model.block_modules.items()]
    try:
        model.eval()
        dtype = next(model.parameters()).dtype
        with torch.no_grad():
            model(torch.as_tensor(np.asarray(x)[None], dtype=dtype))
    finally:
        for h in hooks:
            h.remove()
    return {k: _thin(v[0].detach().cpu().numpy()) for k, v in seen.items()}


ATTN_ROWS, ATTN_COLS = 64, 128      # 어텐션 지도 그림의 크기 상한 (질문 위치 × 본 위치)


def attn_view(a: np.ndarray) -> dict:
    """어텐션 가중치 (질문 T, 본 위치 S) → 화면용: 지도(줄여서)·위치마다 평균으로 받은 주의·첫 자리(CLS)가 본 곳."""
    a = np.asarray(a, dtype=np.float64)
    m = a
    if m.shape[0] > ATTN_ROWS:
        m = m[np.round(np.linspace(0, m.shape[0] - 1, ATTN_ROWS)).astype(int)]
    if m.shape[1] > ATTN_COLS:
        m = m[:, np.round(np.linspace(0, m.shape[1] - 1, ATTN_COLS)).astype(int)]
    r = lambda v: np.round(v, 5).tolist()
    return {"q": int(a.shape[0]), "k": int(a.shape[1]), "map": r(m), "mean": r(a.mean(0)), "first": r(a[0])}


def attention(model, x) -> dict:
    """샘플 하나를 넣어 어텐션 블록(MultiHeadAttention·CrossAttention·NonLocal·AttentionPool)마다 softmax 가중치
    (head 평균) → {노드 id: attn_view}. 모듈 코드는 그대로 두고, 그 블록이 도는 동안만 torch.softmax의 결과를 받아 둠."""
    import torch
    from blocklearning import nn_extra as NX
    kinds = (NX.MultiHeadAttention, NX.CrossAttention, NX.NonLocal, NX.AttentionPool)
    mods = getattr(model, "block_modules", {}) or {}
    cur, got = [None], {}
    hooks = []
    for k, m in mods.items():
        for sub in m.modules():
            if isinstance(sub, kinds):
                hooks.append(sub.register_forward_pre_hook(lambda mod, i, k=k: cur.__setitem__(0, k)))
                hooks.append(sub.register_forward_hook(lambda mod, i, o: cur.__setitem__(0, None)))
    if not hooks:
        return {}
    real = torch.softmax

    def soft(t, *a, **kw):
        y = real(t, *a, **kw)
        if cur[0] is not None:
            w = y[0].detach().cpu().numpy()
            got[cur[0]] = w.mean(0) if w.ndim == 3 else (w[None] if w.ndim == 1 else w)   # head 평균 · AttentionPool은 (1, S)
        return y
    try:
        torch.softmax = soft
        model.eval()
        dtype = next(model.parameters()).dtype
        with torch.no_grad():
            model(torch.as_tensor(np.asarray(x)[None], dtype=dtype))
    finally:
        torch.softmax = real
        for h in hooks:
            h.remove()
    return {k: attn_view(v) for k, v in got.items()}


def _score(out, x, d):
    """예측을 나타내는 수 하나 (그 기울기가 '어디를 봤나'): 분류는 고른 클래스 점수, 구간 분할은 시점마다 고른 상태 점수의 합,
    정답 여러 개는 머리마다 (고른 클래스 점수·값)의 합, 탐지는 칸마다 '있음' 점수의 합, 오토인코더는 복원 오차, 나머지는 출력의 합.
    웹 엔진(engine.js saliency)과 같은 식."""
    if d.task == "classification":
        return out.max()
    if d.task == "segmentation":
        return out.reshape(d.n_out.n_cls, -1).max(0).values.sum()
    if d.task == "multi":
        return sum(out[h["start"]:h["start"] + h["size"]].max() if h["kind"] == "class" else out[h["start"]] for h in d.n_out.heads)
    if d.task == "detection":
        return out.reshape(d.n_out.per_cell, -1)[0].sum()
    if d.task == "anomaly" and out.numel() == x.numel():
        return ((out - x.reshape(-1)) ** 2).sum()
    return out.sum()


def saliency(model, x, d) -> list:
    """샘플 하나(모델 입력 모양)에서 입력 값마다 |∂ 예측 / ∂ 입력|, 가장 큰 것이 1. 데이터 순서로 펼치고,
    여러 채널 시계열은 시점마다 채널을 더함 (길이 L)."""
    import torch
    model.eval()
    dtype = next(model.parameters()).dtype
    xt = torch.as_tensor(np.asarray(x)[None], dtype=dtype).requires_grad_(True)
    out = model(xt)[0].reshape(-1)
    xd = xt[0].transpose(0, 1) if getattr(model, "kind", "") == "rnn" else xt[0]      # 순환 모델 입력 (L, C) → 데이터 순서 (C, L)
    _score(out, xd, d).backward()
    g = xt.grad[0]
    if getattr(model, "kind", "") == "rnn":
        g = g.transpose(0, 1)
    g = g.abs().double().cpu().numpy()
    if g.ndim >= 2 and g.shape[0] > 1:                          # 여러 채널 (시계열·컬러 그림): 시점·픽셀마다 더함
        g = g.sum(0)
    g = g.reshape(-1)
    hi = float(g.max()) or 1.0
    return np.round(g / hi, 4).tolist()

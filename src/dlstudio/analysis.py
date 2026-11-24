"""모델 정보(파라미터 수·크기)와 블록 사이 실제 값(activations)."""
from __future__ import annotations

import numpy as np

MAX_POINTS = 200          # 블록 사이 그림의 시점 수 상한


def model_panel(model) -> dict:
    """③ 모델 옆 '모델 정보': 파라미터 수와 크기(MB, float32 기준)."""
    params = int(sum(p.numel() for p in model.parameters()))
    return {"ok": True, "params": params, "mb": params * 4 / 1e6, "message": "", "warnings": []}


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

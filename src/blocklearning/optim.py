"""학습 방법 설정: 옵티마이저·학습률 스케줄·weight decay·기울기 자르기·혼합 정밀도.

설치형(jobs.fit)과 웹(engine.js train)이 같은 규칙을 씀. 기본값이면 예전과 똑같이 Adam·고정 학습률.
"""
from __future__ import annotations

import math

OPTIMIZERS = ("adam", "adamw", "sgd")                 # sgd: momentum 0.9, Nesterov
SCHEDULES = ("constant", "cosine", "warmup_cosine")   # warmup_cosine: 처음 WARMUP 비율만큼 0에서 올린 뒤 cosine
WARMUP = 0.05
MOMENTUM = 0.9
AMP_MODES = ("off", "auto")                           # auto: NVIDIA GPU면 혼합 정밀도 (bf16 되면 bf16, 아니면 fp16)
DEFAULTS = {"optimizer": "adam", "weight_decay": 0.0, "schedule": "constant", "grad_clip": 0.0, "amp": "off"}


class OptimError(ValueError):
    pass


def normalize(cfg: dict = None) -> dict:
    """학습 설정에서 학습 방법만 골라 검사 (없으면 기본값)."""
    c = dict(DEFAULTS, **{k: v for k, v in (cfg or {}).items() if k in DEFAULTS and v is not None})
    if c["optimizer"] not in OPTIMIZERS:
        raise OptimError(f"옵티마이저 {c['optimizer']!r}을(를) 모름 ({'·'.join(OPTIMIZERS)} 중에서)")
    if c["schedule"] not in SCHEDULES:
        raise OptimError(f"학습률 스케줄 {c['schedule']!r}을(를) 모름 ({'·'.join(SCHEDULES)} 중에서)")
    if c["amp"] not in AMP_MODES:
        raise OptimError(f"혼합 정밀도 {c['amp']!r}을(를) 모름 (off·auto)")
    for k, hi in (("weight_decay", 1.0), ("grad_clip", 1000.0)):
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= hi:
            raise OptimError(f"{k} 값 {v!r}이(가) 맞지 않음 (0–{hi:g})")
        c[k] = float(v)
    return c


def is_default(c: dict) -> bool:
    return all(c[k] == v for k, v in DEFAULTS.items())


def lr_at(step: int, total: int, lr: float, schedule: str) -> float:
    """묶음(step) 번호 0, 1, …, total-1에서의 학습률."""
    if schedule == "constant" or total <= 1:
        return lr
    if schedule == "warmup_cosine":
        warm = max(1, int(math.floor(total * WARMUP + 0.5)))   # JS Math.round와 같게 (.5는 올림)
        if step < warm:
            return lr * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
    else:
        t = step / total
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def make_optimizer(params, c: dict, lr: float):
    import torch
    params = [p for p in params if p.requires_grad]
    if c["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=c["weight_decay"])
    if c["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, nesterov=True, weight_decay=c["weight_decay"])
    return torch.optim.Adam(params, lr=lr, weight_decay=c["weight_decay"])


def amp_dtype(c: dict, dev):
    """혼합 정밀도로 쓸 자료형 (안 쓰면 None): NVIDIA GPU에서만."""
    import torch
    if c["amp"] != "auto" or dev is None or getattr(dev, "type", str(dev).split(":")[0]) != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def code_lines(c: dict, params: str = "model.parameters()") -> list:
    """생성 코드(PyTorch)의 옵티마이저 줄."""
    wd = f", weight_decay={c['weight_decay']!r}" if c["weight_decay"] else ""
    if c["optimizer"] == "adamw":
        return [f"    opt = torch.optim.AdamW({params}, lr=lr{wd})"]
    if c["optimizer"] == "sgd":
        return [f"    opt = torch.optim.SGD({params}, lr=lr, momentum={MOMENTUM}, nesterov=True{wd})"]
    return [f"    opt = torch.optim.Adam({params}, lr=lr{wd})"]


def describe(c: dict) -> str:
    """학습 기록 비교 표에 쓰는 한 줄."""
    name = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD"}[c["optimizer"]]
    parts = [name + (f" (wd {c['weight_decay']:g})" if c["weight_decay"] else "")]
    if c["schedule"] != "constant":
        parts.append({"cosine": "cosine", "warmup_cosine": "warmup+cosine"}[c["schedule"]])
    if c["grad_clip"]:
        parts.append(f"clip {c['grad_clip']:g}")
    if c["amp"] == "auto":
        parts.append("AMP")
    return " · ".join(parts)

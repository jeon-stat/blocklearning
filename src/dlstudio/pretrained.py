"""사전학습 가중치 (pretrained/<종류>.bin + .json, tools/build_pretrained.py로 만듦). 학습하지 않는(고정) 가중치라
프로젝트·학습 기록에는 넣지 않고, 모델을 만들 때마다 이 파일에서 채운다. 웹 엔진(static/engine.js)도 같은 파일을 읽는다."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent / "pretrained"


@lru_cache(maxsize=None)
def arrays(kind: str) -> dict:
    """종류 → {이름: float32 배열} (이름은 블록 안 이름: stem.weight …)."""
    index = json.loads((DIR / f"{kind}.json").read_text(encoding="utf-8"))
    flat = np.frombuffer((DIR / f"{kind}.bin").read_bytes(), dtype="<f4")
    out, i = {}, 0
    for name, shape in index:
        n = int(np.prod(shape)) if shape else 1
        a = flat[i:i + n].reshape(shape)
        a.setflags(write=False)
        out[name] = a
        i += n
    if i != len(flat):
        raise ValueError(f"사전학습 가중치 파일이 목록과 다름: {kind}")
    return out


def fill(module, kind: str) -> None:
    """PyTorch 모듈에 사전학습 가중치를 넣음."""
    import torch
    module.load_state_dict({k: torch.tensor(np.array(v)) for k, v in arrays(kind).items()})


def frozen_names(graph: dict, length: int, n_out, image: tuple = None) -> set:
    """그래프의 고정 가중치 이름 (모델 state_dict 이름). 프로젝트·학습 기록에서 빼는 것."""
    from dlstudio.codegen import make_plan, param_specs
    plan = make_plan(graph, length, n_out, image)
    return {p[0] for s in plan.steps for p in param_specs(s) if p[2] == "frozen"}


def strip(state: dict, frozen: set) -> dict:
    return {k: v for k, v in state.items() if k not in frozen}


def load_state(model, state: dict) -> None:
    """저장한 가중치 → 모델. 고정 가중치는 저장하지 않으므로 없어도 됨 (모델을 만들 때 이미 채움)."""
    import torch
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    missing, unexpected = model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()}, strict=False)
    bad = [k for k in missing if k not in frozen]
    if bad or unexpected:
        raise ValueError(f"가중치가 모델과 맞지 않음: {(bad or unexpected)[:3]}")

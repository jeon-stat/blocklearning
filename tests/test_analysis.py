"""모델 정보(model_panel)와 블록 사이 실제 값(activations)."""
import json

import numpy as np
from dlstudio.analysis import activations, model_panel
from dlstudio.blocks import build_model, to_model_input

EXAMPLE = [
    {"type": "conv", "channels": 16, "kernel": 5},
    {"type": "act", "kind": "relu"},
    {"type": "repeat", "times": 2, "body": [
        {"type": "residual", "body": [{"type": "conv", "channels": 16, "kernel": 3}, {"type": "act", "kind": "relu"}]},
    ]},
    {"type": "global_avg"},
    {"type": "output"},
]


def _X(n=8, L=140):
    return np.random.default_rng(0).normal(size=(n, L))


def test_model_panel_counts_parameters():
    m = build_model(EXAMPLE, 140, 2).eval()
    p = model_panel(m)
    assert p["ok"] and p["params"] == sum(x.numel() for x in m.parameters())
    assert abs(p["mb"] - p["params"] * 4 / 1e6) < 1e-12
    json.dumps(p)


def test_activations_keys_and_length():
    m = build_model(EXAMPLE, 400, 2).eval()
    a = activations(m, to_model_input("cnn", _X(1, 400))[0])
    assert {"0", "1", "3", "4"} <= set(a)
    assert len(a["0"]) == 16 and len(a["0"][0]) == 200          # 시점 400 → 고르게 200점
    assert len(a["3"]) == 16 and len(a["4"]) == 2
    json.dumps(a)

"""학습 방법 (optim.py): 옵티마이저·학습률 스케줄·weight decay·기울기 자르기. 설치형(torch)과 웹 엔진(TF.js)이 같은 갱신을 하는지."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from blocklearning import codegen, optim
from blocklearning.blocks import to_model_input
from blocklearning.graph import build_graph_model
from blocklearning.jobs import TrainSettings, fit

JS = Path(__file__).parent / "js"
HAS_JS = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()

G = {"version": 2, "nodes": [{"id": "input", "type": "input"},
                             {"id": "c", "type": "conv", "channels": 4, "kernel": 3, "stride": 1, "inputs": ["input"]},
                             {"id": "a", "type": "act", "kind": "relu", "inputs": ["c"]},
                             {"id": "f", "type": "flatten", "inputs": ["a"]},
                             {"id": "o", "type": "output", "inputs": ["f"]}]}
L, K = 16, 3
SETTINGS = [
    {"key": "adam"},
    {"key": "adam_wd", "weight_decay": 0.01},
    {"key": "adamw_cos", "optimizer": "adamw", "weight_decay": 0.05, "schedule": "cosine"},
    {"key": "sgd_warm_clip", "optimizer": "sgd", "weight_decay": 0.001, "schedule": "warmup_cosine", "grad_clip": 0.5},
]


def test_lr_schedules():
    assert optim.lr_at(0, 100, 0.1, "constant") == 0.1
    assert optim.lr_at(0, 100, 0.1, "cosine") == pytest.approx(0.1)
    assert optim.lr_at(50, 100, 0.1, "cosine") == pytest.approx(0.05)
    assert optim.lr_at(99, 100, 0.1, "cosine") < 0.001
    warm = [optim.lr_at(i, 100, 0.1, "warmup_cosine") for i in range(7)]
    assert warm[0] == pytest.approx(0.02) and warm[4] == pytest.approx(0.1) and warm[6] < 0.1   # 5% = 5걸음 동안 오름
    assert math.isclose(optim.lr_at(0, 1, 0.1, "cosine"), 0.1)
    assert optim.lr_at(2, 50, 0.1, "warmup_cosine") == pytest.approx(0.1)          # 50 × 5% = 2.5 → 3걸음 (JS Math.round와 같게)


def test_generated_code_trains():
    """코드로 보기: 고른 학습 방법이 생성 코드에 들어가고 그대로 돈다."""
    tr = {"optimizer": "sgd", "weight_decay": 0.001, "schedule": "warmup_cosine", "grad_clip": 1.0, "epochs": 2}
    code = codegen.to_pytorch(G, L, K, data_info={"task": "classification"}, train=tr)
    assert "torch.optim.SGD(" in code and "def lr_at" in code and "clip_grad_norm_" in code and "import math" in code
    ns = {}
    exec(compile(code, "gen", "exec"), ns)
    X = np.random.default_rng(0).normal(size=(20, 1, L)).astype("float32")
    ns["train"](ns["StudioModel"](), X, np.arange(20) % K)
    plain = codegen.to_pytorch(G, L, K, data_info={"task": "classification"})
    assert "torch.optim.Adam(model.parameters(), lr=lr)" in plain and "lr_at" not in plain


def test_normalize_rejects_bad_values():
    assert optim.normalize(None) == optim.DEFAULTS and optim.is_default(optim.normalize({}))
    for bad in ({"optimizer": "lion"}, {"schedule": "step"}, {"weight_decay": -1}, {"grad_clip": "x"}, {"amp": "on"}):
        with pytest.raises(optim.OptimError):
            optim.normalize(bad)
    s = TrainSettings.from_dict({"optimizer": "adamw", "weight_decay": 0.01, "schedule": "cosine", "junk": 1})
    assert s.optim()["optimizer"] == "adamw" and s.optim()["schedule"] == "cosine"
    assert optim.describe(optim.normalize(s.optim())) == "AdamW (wd 0.01) · cosine"


def _setup():
    torch.manual_seed(0)
    net = build_graph_model(G, L, K, seed=0)
    X = np.random.default_rng(0).normal(size=(12, L))
    y = np.arange(12) % K
    return net, X, y


def _torch_weights(cfg, steps, lr):
    net, X, y = _setup()
    net = fit(net, to_model_input(net.kind, X), y, "classification", steps, lr, batch_size=len(X), optim=cfg)
    return {k: v.detach().numpy().ravel() for k, v in net.named_parameters()}


def test_default_matches_plain_adam():
    """기본값(Adam·고정 학습률)은 예전 학습과 똑같음."""
    net, X, y = _setup()
    ref = build_graph_model(G, L, K, seed=0)
    ref.load_state_dict(net.state_dict())
    got = _torch_weights(None, 5, 0.01)
    opt = torch.optim.Adam(ref.parameters(), lr=0.01)
    x, t = torch.as_tensor(to_model_input(ref.kind, X), dtype=torch.float32), torch.as_tensor(y)
    for _ in range(5):
        ref.train()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(ref(x), t).backward()
        opt.step()
    for k, v in ref.named_parameters():
        assert np.allclose(got[k], v.detach().numpy().ravel(), atol=1e-6), k


def test_options_change_training():
    base = _torch_weights(None, 6, 0.05)
    for s in SETTINGS[1:]:
        cfg = {k: v for k, v in s.items() if k != "key"}
        w = _torch_weights(cfg, 6, 0.05)
        assert any(not np.allclose(w[k], base[k]) for k in base), s["key"]


@pytest.mark.skipif(not HAS_JS, reason="node 또는 tests/js/node_modules 없음")
def test_engine_matches_torch_optimizers(tmp_path):
    steps, lr = 6, 0.05
    net, X, y = _setup()
    x = to_model_input(net.kind, X)
    state = {k: {"shape": list(v.shape), "data": v.detach().numpy().ravel().tolist()} for k, v in net.state_dict().items()}
    f = tmp_path / "optim.json"
    f.write_text(json.dumps({"plan": codegen.plan_dict(G, L, K), "state": state, "x": {"shape": list(x.shape), "data": x.ravel().tolist()},
                             "labels": y.tolist(), "steps": steps, "settings": [dict(s, lr=lr) for s in SETTINGS]}), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_optim.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout)
    for s in SETTINGS:
        cfg = {k: v for k, v in s.items() if k != "key"}
        exp = _torch_weights(cfg, steps, lr)
        for k, v in exp.items():
            assert np.allclose(got[s["key"]][k], v, atol=2e-5, rtol=1e-3), (s["key"], k)

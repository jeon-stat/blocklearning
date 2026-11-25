"""손실 조립: 항 검사 · PyTorch와 웹 엔진이 같은 값 · 조립한 손실로 학습 (PyTorch·엔진) · 기본이면 지금까지와 같은 손실."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from dlstudio import losses, tasks
from dlstudio.data import load_example
from dlstudio.graph import from_blocks
from dlstudio.jobs import Job, TrainSettings, train_job

JS = Path(__file__).parent / "js"
HAS_NODE = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()
T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}


def test_normalize_and_allowed():
    assert losses.normalize(None, "classification") == [{"term": "ce", "w": 1.0, "smooth": 0.0, "balance": False}]
    assert losses.is_default(losses.normalize([{"term": "ce"}], "classification"), "classification")
    with pytest.raises(ValueError):
        losses.normalize([{"term": "mse"}], "classification")             # 분류에 MSE는 못 고름
    with pytest.raises(ValueError):
        losses.normalize([{"term": "l2", "w": 0.001}], "regression")       # 정답과 맞추는 항이 하나는 있어야 함
    with pytest.raises(ValueError):
        losses.normalize([{"term": "ce", "smooth": 0.9}], "classification")
    assert "dice" in losses.allowed("segmentation") and "dice" not in losses.allowed("classification")
    w = losses.class_weights([0, 0, 0, 1], 3)
    assert w[2] == 0 and w[1] > w[0]


def _cases():
    rng = np.random.default_rng(0)
    out = []
    o, y = rng.normal(size=(6, 3)).astype(np.float32), rng.integers(0, 3, 6)
    cw = losses.class_weights(y, 3)
    for spec in ([{"term": "ce", "smooth": 0.1}], [{"term": "ce", "balance": True}], [{"term": "focal", "gamma": 2, "balance": True, "w": 0.5}]):
        out.append(("classification", o, y, spec, cw, None))
    r, t = rng.normal(size=(5, 2)).astype(np.float32), rng.normal(size=(5, 2)).astype(np.float32)
    out.append(("regression", r, t, [{"term": "mse"}, {"term": "mae", "w": 0.3}, {"term": "huber", "delta": 0.5}], None, None))
    K, L = 3, 7
    so, sy = rng.normal(size=(4, K * L)).astype(np.float32), rng.integers(0, K, (4, L))
    out.append(("segmentation", so, sy, [{"term": "ce", "balance": True}, {"term": "dice", "w": 0.5}], losses.class_weights(sy, K), (K, L)))
    return out


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
def test_engine_matches_torch(tmp_path):
    fx, want = [], []
    for task, o, y, spec, cw, seg in _cases():
        spec = [dict(t, label=losses.label(t)) for t in losses.normalize(spec, task)]
        K = 3
        fn = losses.build(spec, task, n_cls=K, seg_K=seg[0] if seg else 0, cw=None if cw is None else torch.as_tensor(cw))
        ot = torch.as_tensor(o).clone().requires_grad_(True)
        total, parts = fn(ot, torch.as_tensor(y), torch.nn.Linear(1, 1))
        total.backward()
        want.append((float(total.detach()), {k: float(v.detach()) for k, v in parts.items()}, ot.grad.numpy().ravel()))
        if task == "classification":
            yy = np.eye(K, dtype=np.float32)[y]
        elif task == "segmentation":
            yy = np.eye(seg[0], dtype=np.float32)[y]                           # (B, L, K)
        else:
            yy = y
        fx.append({"out": o.ravel().tolist(), "out_shape": list(o.shape), "y": yy.ravel().tolist(), "y_shape": list(yy.shape),
                   "spec": spec, "cw": None if cw is None else cw.tolist(), "seg": {"n_cls": seg[0], "length": seg[1]} if seg else None,
                   "reg": task == "regression"})
    f = tmp_path / "loss.json"
    f.write_text(json.dumps(fx), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_loss.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    for g, (tot, parts, grad) in zip(json.loads(r.stdout), want):
        assert g["total"] == pytest.approx(tot, rel=1e-4, abs=1e-5)
        assert np.allclose(g["grad"], grad, rtol=1e-3, atol=1e-5)
        for k, v in parts.items():
            assert g["parts"][k] == pytest.approx(v, rel=1e-4, abs=1e-5), k


def test_training_with_assembled_loss_torch_and_engine(tmp_path):
    d = load_example("ecg")
    d.X_train, d.y_train, d.X_test, d.y_test = d.X_train[:200], d.y_train[:200], d.X_test[:60], d.y_test[:60]
    spec = [{"term": "focal", "gamma": 2, "balance": True}, {"term": "l2", "w": 1e-4}]
    g = from_blocks(T["cnn"]["blocks"])
    j = Job("t")
    r = train_job(j, g, d, TrainSettings(epochs=2, loss=spec))
    assert set(j.progress[-1]["terms"]) == {"Focal ×1", "가중치 L2 ×0.0001"} and np.isfinite(r["test_metric"])
    if not HAS_NODE:
        return
    from tests.test_engine import _train_js
    norm = losses.normalize(spec, "classification")
    w = _train_js(tmp_path, g, d, {"epochs": 2, "lr": 1e-3, "batch_size": 32, "seed": 0,
                                   "loss": [dict(t, label=losses.label(t)) for t in norm],
                                   "class_weights": losses.class_weights(d.y_train, 2).tolist()})
    assert set(w["progress"][-1]["terms"]) == {"Focal ×1", "가중치 L2 ×0.0001"} and np.isfinite(w["result"]["test_metric"])


def test_block_output_l1_torch_engine_and_code(tmp_path):
    """블록 출력 L1 (act_l1): 고른 블록의 출력 크기를 손실에 더함 — PyTorch·엔진 학습, 검사, 내보내는 코드."""
    from dlstudio.core import Studio, UserError
    d = load_example("ecg")
    d.X_train, d.y_train, d.X_test, d.y_test = d.X_train[:160], d.y_train[:160], d.X_test[:60], d.y_test[:60]
    g = from_blocks([{"type": "conv", "channels": 8, "kernel": 5, "stride": 1}, {"type": "act", "kind": "relu"},
                     {"type": "global_avg"}, {"type": "output"}])
    node = [n["id"] for n in g["nodes"] if n["type"] == "act"][0]
    spec = [{"term": "ce"}, {"term": "act_l1", "w": 0.01, "node": node}]
    assert losses.normalize(spec, "classification")[1]["node"] == node
    with pytest.raises(ValueError):
        losses.normalize([{"term": "ce"}, {"term": "act_l1"}], "classification")        # 블록을 골라야 함
    j = Job("t")
    r = train_job(j, g, d, TrainSettings(epochs=2, keep_best=True, loss=spec))
    lab = losses.label(losses.normalize(spec, "classification")[1])
    assert lab in j.progress[-1]["terms"] and j.progress[-1]["terms"][lab] > 0 and np.isfinite(r["test_metric"])
    assert getattr(r["model"], "capture", None) is None
    s = Studio()
    s.api_data_load({"example": "ecg"})
    with pytest.raises(UserError):
        s.api_train({"graph": g, "settings": {"epochs": 1, "loss": [{"term": "ce"}, {"term": "act_l1", "node": "없음"}]}})
    code = s.api_code({"framework": "pytorch", "graph": g, "settings": {"epochs": 1, "loss": spec}})["code"]
    assert "self.acts = {" in code and "model.acts[" in code
    ns = {}
    exec(compile(code.replace('if __name__ == "__main__":', "if False:"), "studio.py", "exec"), ns)
    ns["train"](ns["StudioModel"](), d.X_train[:64, None, :], d.y_train[:64], epochs=1)
    if not HAS_NODE:
        return
    from tests.test_engine import _train_js
    w = _train_js(tmp_path, g, d, {"epochs": 2, "lr": 1e-3, "batch_size": 32, "seed": 0, "keep_best": True,
                                   "loss": [dict(t, label=losses.label(t)) for t in losses.normalize(spec, "classification")]})
    assert w["progress"][-1]["terms"][lab] > 0 and np.isfinite(w["progress"][-1]["val_loss"])

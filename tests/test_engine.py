"""웹 학습 엔진(static/engine.js, TensorFlow.js)이 PyTorch(GraphNet)와 같은 계산을 하는지 node로 확인.

같은 가중치를 넣고: 평가 출력·노드마다 값(샘플 따라가기)·학습 모드의 손실 기울기·BatchNorm 통계 갱신을 비교한다.
node와 tests/js/node_modules(@tensorflow/tfjs, `npm install --prefix tests/js`)가 없으면 건너뜀.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from dlstudio import codegen
from dlstudio.blocks import to_model_input
from dlstudio.graph import build_graph_model
from tests.test_codegen import CASES

JS = Path(__file__).parent / "js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (JS / "node_modules" / "@tensorflow" / "tfjs").exists(),
                                reason="node 또는 tests/js/node_modules 없음")


def _fixture(case, seed=1):
    name, g, L, n_out, image = case
    torch.manual_seed(seed)
    net = build_graph_model(g, L, n_out, seed=seed, image=image)
    with torch.no_grad():                                   # BatchNorm 통계도 무작위로 (평가 경로를 제대로 보게)
        for m in net.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.3, 0.3)
    X = np.random.default_rng(seed).normal(size=(4, image[0] * image[1] if image else L))
    x = torch.as_tensor(to_model_input(net.kind, X, image=image), dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        y = net(x).numpy()
        tr = {k: v[0].numpy().ravel().tolist() for k, v in net.trace(x[:1]).items() if k != "input"}
    state = {k: {"shape": list(v.shape), "data": v.detach().numpy().ravel().tolist()} for k, v in net.state_dict().items()}
    fx = {"name": name, "plan": codegen.plan_dict(g, L, n_out, image), "state": state,
          "x": {"shape": list(x.shape), "data": x.numpy().ravel().tolist()}}
    from dlstudio.analysis import attention
    exp = {"y": y, "trace": tr, "params": sum(p.numel() for p in net.parameters()), "attn": attention(net, x[0].numpy())}
    from dlstudio.nn_extra import RANDOM_MODULES
    has_dropout = any(isinstance(m, (torch.nn.modules.dropout._DropoutNd,) + RANDOM_MODULES) for m in net.modules())
    if not has_dropout:                                     # 학습 모드: 묶음 통계 BatchNorm, 교차 엔트로피 기울기
        labels = [i % n_out for i in range(len(x))]
        net.train()
        net.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), torch.tensor(labels))
        loss.backward()
        fx["grad"] = {"labels": labels}
        exp["loss"] = float(loss)
        exp["grads"] = {k: p.grad.numpy().ravel() for k, p in net.named_parameters() if p.grad is not None}
        exp["running"] = {k: v.numpy().ravel() for k, v in net.state_dict().items()
                          if k.endswith("running_mean") or k.endswith("running_var")}
    return fx, exp


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    fixtures, expected = [], {}
    for case in CASES:
        fx, exp = _fixture(case)
        fixtures.append(fx)
        expected[case[0]] = exp
    f = tmp_path_factory.mktemp("engine") / "cases.json"
    f.write_text(json.dumps(fixtures), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_engine.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout), expected


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_engine_matches_torch(results, name):
    got, expected = results
    g, e = got[name], expected[name]
    assert g["params"] == e["params"]
    assert np.allclose(np.array(g["y"]).reshape(e["y"].shape), e["y"], atol=2e-4, rtol=1e-3), name
    for k, v in e["trace"].items():                          # 샘플 따라가기 값 (노드 id마다)
        assert k in g["trace"], (name, k)
        assert np.allclose(g["trace"][k], v, atol=2e-4, rtol=1e-3), (name, k)
    assert sorted(g["attn"]) == sorted(e["attn"]), name                # 어텐션 지도 (샘플 따라가기)
    for k, a in e["attn"].items():
        b = g["attn"][k]
        assert (b["q"], b["k"]) == (a["q"], a["k"]), (name, k)
        for f in ("map", "mean", "first"):
            assert np.allclose(b[f], a[f], atol=2e-4), (name, k, f)
    if "grads" in e:
        assert abs(g["loss"] - e["loss"]) < 1e-4 * max(1, abs(e["loss"])), name
        for k, v in e["grads"].items():
            gk = np.asarray(g["grads"][k])                          # 큰 그림(64×64)은 더하는 순서 차이가 쌓임 → 전체 상대 오차로도 봄
            assert np.allclose(gk, v, atol=2e-5, rtol=2e-3) or                 np.linalg.norm(gk - v) <= 5e-3 * max(np.linalg.norm(v), 1e-8), (name, k)
        for k, v in e["running"].items():
            assert np.allclose(g["running"][k], v, atol=1e-5, rtol=1e-4), (name, k)


def _train_js(tmp_path, graph, d, settings):
    plan = codegen.plan_dict(graph, d.length, d.n_out, d.image)
    data = {"X_train": d.X_train.ravel().tolist(), "X_test": d.X_test.ravel().tolist(), "y_train": np.ravel(d.y_train).tolist(),
            "y_test": np.ravel(d.y_test).tolist(), "n_train": len(d.y_train), "n_test": len(d.y_test), "task": d.task, "n_out": d.n_out,
            "length": d.length, "image": list(d.image) if d.image else None}
    f = tmp_path / "train.json"
    f.write_text(json.dumps({"plan": plan, "data": data, "settings": settings}), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_train.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


def test_engine_trains_classification_and_weights_run_in_torch(tmp_path):
    """엔진 학습: 정확도가 오르고, 학습한 가중치를 PyTorch 모델에 넣으면 같은 정확도 (웹 → PyTorch 내보내기 경로)."""
    from dlstudio.data import make_dataset
    from tests.synth import example_arrays
    from dlstudio.tasks import DEFAULT_TEMPLATES, _view
    d = make_dataset(*example_arrays(n=300, length=140))
    g = _view(DEFAULT_TEMPLATES[0])["graph"]
    out = _train_js(tmp_path, g, d, {"epochs": 6, "lr": 0.01, "batch_size": 32, "seed": 0})
    r = out["result"]
    assert [p["epoch"] for p in out["progress"]] == list(range(1, 7)) and r["epochs_done"] == 6
    assert r["metric_name"] == "accuracy" and r["test_metric"] > 0.7 and len(r["confusion"]) == 2
    assert 0.5 < r["train_metric"] <= 1
    net = build_graph_model(g, d.length, d.n_out).eval()
    net.load_state_dict({k: torch.tensor(v["data"]).reshape(v["shape"]).to(net.state_dict()[k].dtype) for k, v in out["state"].items()})
    with torch.no_grad():
        pred = net(torch.as_tensor(to_model_input(net.kind, d.X_test), dtype=torch.float32)).argmax(1).numpy()
    assert abs(float(np.mean(pred == d.y_test)) - r["test_metric"]) < 1e-6
    assert r["pred"] == pred.tolist()                                   # 시험 샘플마다 예측 (틀린 샘플 모아 보기)
    with torch.no_grad():
        pred_tr = net(torch.as_tensor(to_model_input(net.kind, d.X_train), dtype=torch.float32)).argmax(1).numpy()
    assert abs(float(np.mean(pred_tr == d.y_train)) - r["train_metric"]) < 1e-6, (float(np.mean(pred_tr == d.y_train)), r["train_metric"])


def test_engine_trains_regression_in_original_units(tmp_path):
    from dlstudio.data import load_example
    d = load_example("process_reg")
    g = next(t for f in __import__("dlstudio.tasks", fromlist=["x"]).catalog_view()["families"] for s in f["subtasks"]
             if s["id"].startswith("reg") and s["input"] == "tabular" for t in s["templates"])["graph"]
    out = _train_js(tmp_path, g, d, {"epochs": 15, "lr": 0.01, "batch_size": 32, "seed": 0})
    r = out["result"]
    assert r["metric_name"] == "r2" and r["test_metric"] > 0.3 and r["mae"] > 0 and len(r["scatter"]["y"]) > 0

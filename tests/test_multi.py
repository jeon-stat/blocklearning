"""정답 여러 개 (분류·값 섞어서): 예시 데이터 · 손실 조립 (정답마다) · PyTorch 학습·결과 · 웹 엔진 학습 · ⑤ 예측 · 코드."""
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from blocklearning import losses, multi, tasks
from blocklearning.data import load_example
from blocklearning.graph import build_graph_model, check_graph, from_blocks
from blocklearning.jobs import Job, TrainSettings, train_job

JS = Path(__file__).parent / "js"
HAS_NODE = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()
T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}


def test_example_data():
    d = load_example("hydraulic_multi")
    heads = d.n_out.heads
    assert d.task == "multi" and d.y_train.shape[1] == len(heads) == 4
    assert [h["kind"] for h in heads] == ["class", "value", "class", "value"]
    assert int(d.n_out) == sum(h["size"] for h in heads) == 3 + 1 + 3 + 1
    s = d.summary()
    json.dumps(s, ensure_ascii=False)
    assert [t["name"] for t in s["targets"]] == [h["name"] for h in heads] and " · " in s["preview"][0]["label"]
    assert [h["name"] for h in s["losses"]["heads"]] == [h["name"] for h in heads]


def test_loss_spec_per_head():
    d = load_example("hydraulic_multi")
    heads = d.n_out.heads
    spec = losses.normalize(None, "multi", heads)
    assert [(t["term"], t["head"]) for t in spec] == [("ce", "냉각기"), ("mse", "밸브 (%)"), ("ce", "펌프 누설"), ("mse", "축압기 (bar)")]
    assert losses.is_default(spec, "multi", heads)
    with pytest.raises(ValueError):                                     # 값 정답에 교차 엔트로피는 못 고름
        losses.normalize([dict(t, term="ce") if t["head"] == "밸브 (%)" else t for t in spec], "multi", heads)
    with pytest.raises(ValueError):                                     # 정답마다 항이 하나는 있어야 함
        losses.normalize(spec[1:], "multi", heads)
    s2 = spec + [{"term": "l2", "w": 1e-4}]
    assert losses.normalize(s2, "multi", heads)[-1] == {"term": "l2", "w": 1e-4}
    assert losses.label(spec[0]) == "냉각기 · 교차 엔트로피 ×1"


def test_assembled_loss_equals_default():
    d = load_example("hydraulic_multi")
    heads = d.n_out.heads
    rng = np.random.default_rng(0)
    out = torch.as_tensor(rng.normal(size=(9, int(d.n_out))), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(d.y_train[:9]), dtype=torch.float32)
    mu, sd, yn = multi.target_scale(d.y_train[:9], heads)
    yn = torch.as_tensor(yn, dtype=torch.float32)
    fn = losses.build(losses.normalize(None, "multi", heads), "multi", heads=heads)
    total, parts = fn(out, yn, torch.nn.Linear(1, 1))
    assert float(total) == pytest.approx(float(multi.torch_loss(out, yn, heads)), rel=1e-6)
    assert len(parts) == 4 and y.shape[1] == 4


def test_torch_training_result_predict_and_code():
    d = load_example("hydraulic_multi")
    g = from_blocks(T["multi_mlp"]["blocks"])
    assert check_graph(g, d.length, d.n_out).ok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = train_job(Job("t"), g, d, TrainSettings(epochs=15, lr=3e-3))
    assert r["metric_name"] == "mean" and len(r["heads"]) == 4
    cls = [h for h in r["heads"] if h["kind"] == "class"]
    assert all(np.asarray(h["confusion"]).sum() == len(d.y_test) for h in cls)
    assert r["test_metric"] == pytest.approx(np.mean([h["metric"] for h in r["heads"]]))
    assert r["heads"][0]["metric"] > 0.8                                     # 냉각기 상태는 쉬움 (냉각 효율 열)
    m = r["model"].eval()
    with torch.no_grad():
        out = m(torch.as_tensor(d.X_test[:, None, :], dtype=torch.float32)).double().numpy()
    assert multi.metric(out, d.y_test, d.n_out.heads) == pytest.approx(r["test_metric"])
    assert abs(out[:, d.n_out.heads[1]["start"]].mean() - d.y_test[:, 1].mean()) < 5    # 값 머리는 원래 단위 (밸브 %)
    from blocklearning.core import Studio
    s = Studio()
    s.api_data_load({"example": "hydraulic_multi"})
    s.model, s.model_graph, s.results = m, g, {k: v for k, v in r.items() if k != "model"}
    p = s.api_predict({"example_new": True})
    assert p["summary"]["task"] == "multi" and p["has_label"] and len(p["rows"]) == 10
    assert len(p["rows"][0]["multi_pred"]) == 4 and p["csv"].count("\n") == 11
    a = s.api_activations({"sample_index": 0})
    assert "냉각기" in a["prediction"]
    code = s.api_code({"framework": "pytorch", "graph": g, "settings": {"epochs": 2}})["code"]
    assert "def studio_loss" in code and "냉각기" in code
    ns = {}
    exec(compile(code.replace('if __name__ == "__main__":', "if False:"), "studio.py", "exec"), ns)
    net = ns["StudioModel"]()
    ns["train"](net, d.X_train[:64, None, :], d.y_train[:64], epochs=1)


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
def test_engine_loss_matches_torch(tmp_path):
    d = load_example("hydraulic_multi")
    heads = d.n_out.heads
    rng = np.random.default_rng(1)
    o = rng.normal(size=(7, int(d.n_out))).astype(np.float32)
    _, _, yn = multi.target_scale(d.y_train[:7], heads)
    yn = yn.astype(np.float32)
    spec = losses.normalize([{"term": "focal", "head": "냉각기", "gamma": 2, "balance": True}, {"term": "huber", "head": "밸브 (%)", "delta": 0.5},
                             {"term": "ce", "head": "펌프 누설", "smooth": 0.1}, {"term": "mae", "head": "축압기 (bar)", "w": 0.5}],
                            "multi", heads)
    spec = [dict(t, label=losses.label(t)) for t in spec]
    cw = losses.multi_class_weights(d.y_train, heads, spec)
    fn = losses.build(spec, "multi", heads=heads, cw={k: torch.as_tensor(v) for k, v in cw.items()})
    ot = torch.as_tensor(o).clone().requires_grad_(True)
    total, parts = fn(ot, torch.as_tensor(yn), torch.nn.Linear(1, 1))
    total.backward()
    f = tmp_path / "loss.json"
    f.write_text(json.dumps([{"out": o.ravel().tolist(), "out_shape": list(o.shape), "y": yn.ravel().tolist(), "y_shape": list(yn.shape),
                              "spec": spec, "cw": {k: v.tolist() for k, v in cw.items()}, "seg": None, "reg": False,
                              "heads": heads}]), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_loss.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    g = json.loads(r.stdout)[0]
    assert g["total"] == pytest.approx(float(total.detach()), rel=1e-4, abs=1e-5)
    assert np.allclose(g["grad"], ot.grad.numpy().ravel(), rtol=1e-3, atol=1e-5)
    for k, v in parts.items():
        assert g["parts"][k] == pytest.approx(float(v.detach()), rel=1e-4, abs=1e-5), k


def test_web_engine_training_matches_torch(tmp_path):
    from tests.test_engine import _train_js, pytestmark as engine_mark
    if engine_mark.mark.args[0]:
        pytest.skip(engine_mark.mark.kwargs.get("reason"))
    d = load_example("hydraulic_multi")
    g = from_blocks(T["multi_mlp"]["blocks"])
    heads = d.n_out.heads
    spec = [dict(t, label=losses.label(t)) for t in losses.normalize(None, "multi", heads)]
    w = _train_js(tmp_path, g, d, {"epochs": 3, "lr": 3e-3, "batch_size": 32, "seed": 0, "keep_best": True, "loss": spec})
    assert len(w["progress"]) == 3 and w["out"] is not None and set(w["progress"][-1]["terms"]) == {t["label"] for t in spec}
    net = build_graph_model(g, d.length, d.n_out).eval()
    net.load_state_dict({k: torch.as_tensor(np.asarray(v["data"], dtype=np.float32).reshape(v["shape"])) for k, v in w["state"].items()})
    with torch.no_grad():
        out = net(torch.as_tensor(d.X_test[:, None, :], dtype=torch.float32)).double().numpy()
    assert np.allclose(out.ravel(), np.asarray(w["out"]), atol=1e-2)                # 값 머리는 원래 단위로 접혀 있음
    assert multi.metric(out, d.y_test, heads) == pytest.approx(w["result"]["test_metric"], abs=1e-4)
    assert w["result"]["test_metric"] > 0.3

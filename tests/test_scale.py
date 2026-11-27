"""큰 데이터·긴 학습: 묶음으로 채점, 데이터를 CPU에 두고 묶음만 장치로, 지금 모델에서 이어 학습."""
import time

import numpy as np
import pytest

from blocklearning import jobs
from blocklearning.core import Studio
from blocklearning.tasks import DEFAULT_TEMPLATES, _view

torch = pytest.importorskip("torch")
G = _view(DEFAULT_TEMPLATES[0])["graph"]


def _wait(s, r):
    for _ in range(900):
        st = s.call("GET", f"/api/jobs/{r['id']}")[1]
        if st["state"] != "running":
            assert st["state"] == "done", st
            return st["result"]
        time.sleep(0.1)
    raise AssertionError("학습이 끝나지 않음")


def _train(s, graph, settings, resume=False):
    body = {"graph": graph, "settings": settings}
    if resume:
        body["resume"] = True
    c, r = s.call("POST", "/api/train", body)
    assert c == 200, r
    return _wait(s, r)


def test_predict_np_batches_like_one_pass():
    net = torch.nn.Sequential(torch.nn.Linear(5, 3))
    X = np.random.default_rng(0).normal(size=(1100, 5))
    with torch.no_grad():
        ref = net(torch.as_tensor(X, dtype=torch.float32)).double().numpy()
    assert np.allclose(jobs.predict_np(net, X, batch=128), ref)
    assert np.allclose(jobs.predict_np(net, torch.as_tensor(X, dtype=torch.float32)), ref)


def test_fit_keeps_big_data_off_device(monkeypatch):
    """데이터가 한도보다 크면 CPU에 두고 묶음만 옮겨도 같은 학습 (여기서는 CPU라 결과가 똑같아야 함)."""
    from blocklearning.blocks import to_model_input
    from blocklearning.graph import build_graph_model
    from blocklearning.data import make_dataset
    from tests.synth import example_arrays
    d = make_dataset(*example_arrays(n=120, length=60))

    def run():
        net = build_graph_model(G, d.length, d.n_out, seed=0)
        X = to_model_input(net.kind, d.X_train)
        return jobs.fit(net, X, d.y_train, "classification", 2, 0.01, 16, keep_best=True)
    a = run()
    monkeypatch.setattr(jobs, "GPU_RESIDENT_BYTES", 0)
    b = run()
    for (k, v), (_, w) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(v, w), k


@pytest.mark.parametrize("example", ["ecg", "concrete"])
def test_resume_continues_from_current_model(example):
    """이어 학습: 학습률을 아주 작게 1 epoch 이어 가면 성적이 그대로 (회귀는 출력층의 원래 단위 되돌림을 풀었다 다시 접음)."""
    s = Studio()
    assert s.call("POST", "/api/data/load", {"example": example})[0] == 200
    g = G if example == "ecg" else next(t for f in __import__("blocklearning.tasks", fromlist=["x"]).catalog_view()["families"]
                                         for st in f["subtasks"] if st["id"].startswith("reg") and st["input"] == "tabular"
                                         for t in st["templates"])["graph"]
    first = _train(s, g, {"epochs": 3, "lr": 0.01})
    again = _train(s, g, {"epochs": 1, "lr": 1e-9}, resume=True)
    assert abs(again["test_metric"] - first["test_metric"]) < 1e-3, (first["test_metric"], again["test_metric"])
    fresh = _train(s, g, {"epochs": 1, "lr": 1e-9})                  # 처음부터면 거의 못 배운 성적
    assert abs(fresh["test_metric"] - first["test_metric"]) > 1e-3
    runs = s.call("GET", "/api/runs")[1]["runs"]
    assert runs[1]["settings"].get("resume") is True and not runs[0]["settings"].get("resume")


def test_resume_needs_same_model():
    s = Studio()
    assert s.call("POST", "/api/data/load", {"example": "ecg"})[0] == 200
    c, e = s.call("POST", "/api/train", {"graph": G, "settings": {"epochs": 1}, "resume": True})
    assert c == 400 and "이어서 학습할 모델이 없음" in e["error"]["message"]
    _train(s, G, {"epochs": 1})
    other = _view(DEFAULT_TEMPLATES[1])["graph"]
    c, e = s.call("POST", "/api/train", {"graph": other, "settings": {"epochs": 1}, "resume": True})
    assert c == 400 and "블록 구성이 달라" in e["error"]["message"]


def test_engine_resume_keeps_model(tmp_path):
    """웹 엔진도 이어 학습: 받은 가중치(init_state)에서 시작하고, 회귀의 원래 단위 되돌림을 풀었다 다시 접음."""
    from tests.test_engine import HAS_NODE, _train_js
    if not HAS_NODE:
        pytest.skip("node 또는 tests/js/node_modules 없음")
    from blocklearning.data import load_example
    from blocklearning.tasks import catalog_view
    d = load_example("process_reg")
    g = next(t for f in catalog_view()["families"] for st in f["subtasks"]
             if st["id"].startswith("reg") and st["input"] == "tabular" for t in st["templates"])["graph"]
    first = _train_js(tmp_path, g, d, {"epochs": 8, "lr": 0.01, "batch_size": 32, "seed": 0})
    again = _train_js(tmp_path, g, d, {"epochs": 1, "lr": 1e-9, "batch_size": 32, "seed": 0, "init_state": first["state"]})
    assert abs(again["result"]["test_metric"] - first["result"]["test_metric"]) < 1e-3

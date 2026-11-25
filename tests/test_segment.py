"""구간 분할 (시점마다 상태): 예시 데이터 · 검사 규칙 · PyTorch 학습·결과 · 웹 엔진 학습 · ⑤ 예측 · 프로젝트 저장."""
import json
import warnings

import numpy as np
import pytest
import torch

from dlstudio import segment, tasks
from dlstudio.data import load_example
from dlstudio.graph import build_graph_model, check_graph, from_blocks
from dlstudio.jobs import Job, TrainSettings, train_job

T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}


def small(n_train=96, n_test=32):
    d = load_example("pneumatic_seg")
    d.X_train, d.y_train, d.X_test, d.y_test = d.X_train[:n_train], d.y_train[:n_train], d.X_test[:n_test], d.y_test[:n_test]
    return d


def test_example_data():
    d = load_example("pneumatic_seg")
    assert d.task == "segmentation" and d.y_train.shape == d.X_train.shape and d.length == 256
    assert int(d.n_out) == len(d.classes) * d.length and d.n_out.n_cls == 5
    s = d.summary()
    json.dumps(s, ensure_ascii=False)
    assert s["segment"]["n_classes"] == 5 and all(len(p["states"]) == 256 for p in s["preview"])
    assert sum(s["class_counts"].values()) == d.y_train.size + d.y_test.size


def test_runs_and_text():
    assert segment.runs([0, 0, 1, 1, 1, 0]) == [[0, 0, 2], [1, 2, 3], [0, 5, 1]]
    assert segment.states_text([0, 0, 2], ["a", "b", "c"]) == "a → c"


@pytest.mark.parametrize("tid", ["seg_tcn", "seg_unet", "seg_bigru"])
def test_templates_check_and_keep_length(tid):
    d = load_example("pneumatic_seg")
    c = check_graph(from_blocks(T[tid]["blocks"]), d.length, d.n_out)
    assert c.ok, c.issues


def test_length_rule():
    d = load_example("pneumatic_seg")
    for blocks in ([{"type": "conv", "channels": 8, "kernel": 3, "stride": 1}, {"type": "global_avg"}, {"type": "output"}],
                   [{"type": "conv", "channels": 8, "kernel": 3, "stride": 1}, {"type": "pool", "size": 2}, {"type": "output"}]):
        c = check_graph(from_blocks(blocks), d.length, d.n_out)
        assert not c.ok and "구간 분할" in c.issues[0].message


def test_torch_training_result_and_project(tmp_path):
    d = small()
    g = from_blocks(T["seg_tcn"]["blocks"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = train_job(Job("t"), g, d, TrainSettings(epochs=3))
    assert r["metric_name"] == "accuracy" and 0 <= r["test_metric"] <= 1 and r["epochs_done"] == 3
    assert np.asarray(r["confusion"]).sum() == d.y_test.size and len(r["seg_samples"]) == segment.N_SHOW
    m = r["model"].eval()
    with torch.no_grad():
        out = m(torch.as_tensor(d.X_test[:, None, :], dtype=torch.float32)).numpy()
    assert segment.metric(out, d.y_test, 5) == pytest.approx(r["test_metric"])
    from dlstudio.core import Studio
    s = Studio()
    s.api_data_load({"example": "pneumatic_seg"})
    s.model, s.model_graph, s.results = m, g, {k: v for k, v in r.items() if k != "model"}
    p = s.api_predict({"example_new": True})
    assert p["summary"]["task"] == "segmentation" and p["has_label"] and len(p["rows"]) == 10
    assert len(p["rows"][0]["seg_pred"]) == 256 and p["csv"].count("\n") == 10 * 256 + 1
    a = s.api_activations({"sample_index": 0})
    assert "→" in a["prediction"] or a["prediction"] in d.classes


def test_web_engine_training_matches_torch(tmp_path):
    from tests.test_engine import _train_js, pytestmark as engine_mark
    if engine_mark.mark.args[0]:
        pytest.skip(engine_mark.mark.kwargs.get("reason"))
    d = small(64, 24)
    g = from_blocks([{"type": "conv", "channels": 8, "kernel": 5, "stride": 1}, {"type": "act", "kind": "relu"},
                     {"type": "conv", "channels": 8, "kernel": 3, "stride": 1, "dilation": 2}, {"type": "act", "kind": "relu"},
                     {"type": "output"}])                       # node(CPU)에서 빨리 돌게 작은 모델
    w = _train_js(tmp_path, g, d, {"epochs": 2, "lr": 1e-3, "batch_size": 16, "seed": 0})
    assert len(w["progress"]) == 2 and w["out"] is not None
    net = build_graph_model(g, d.length, d.n_out).eval()
    net.load_state_dict({k: torch.as_tensor(np.asarray(v["data"], dtype=np.float32).reshape(v["shape"])) for k, v in w["state"].items()})
    with torch.no_grad():
        out = net(torch.as_tensor(d.X_test[:, None, :], dtype=torch.float32)).numpy()
    assert np.allclose(out.ravel(), np.asarray(w["out"]), atol=1e-3)
    assert segment.metric(out, d.y_test, 5) == pytest.approx(w["result"]["test_metric"], abs=1e-6)


def test_keras_code_uses_per_step_loss():
    """TensorFlow 코드: 출력 (상태 × 길이)를 시점마다 교차 엔트로피로 (펼친 출력에 바로 걸면 모양이 맞지 않음)."""
    from dlstudio.core import Studio
    s = Studio()
    s.api_data_load({"example": "pneumatic_seg"})
    code = s.api_code({"framework": "tensorflow", "graph": from_blocks(T["seg_bigru"]["blocks"]), "settings": {"epochs": 1}})["code"]
    assert "def seg_loss(target, out):" in code and "loss=seg_loss" in code and "N_STATES = 5" in code
    tcode = s.api_code({"framework": "pytorch", "graph": from_blocks(T["seg_tcn"]["blocks"]), "settings": {"epochs": 1}})["code"]
    assert "시점마다 상태 번호 (샘플, 길이)" in tcode

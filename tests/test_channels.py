"""여러 채널 입력 · CSV 역할 표: 입력 모양 (C, L) · 예시 (전류 + 공압 압력) · 시계열 폴더 값 열 여러 개 · PyTorch·엔진·코드 · 역할 표."""
import json
import warnings

import numpy as np
import pytest
import torch

from dlstudio import tasks
from dlstudio.blocks import to_model_input
from dlstudio.codegen import plan_dict
from dlstudio.data import channels_of, load_csv, load_example, load_series_folder
from dlstudio.graph import build_graph_model, check_graph, from_blocks
from dlstudio.jobs import Job, TrainSettings, train_job

T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}
SMALL = [{"type": "conv", "channels": 8, "kernel": 5, "stride": 1}, {"type": "act", "kind": "relu"},
         {"type": "conv", "channels": 8, "kernel": 3, "stride": 1, "dilation": 2}, {"type": "act", "kind": "relu"}, {"type": "output"}]


def small(kind="pneumatic_seg2", n_train=96, n_test=32):
    d = load_example(kind)
    d.X_train, d.y_train, d.X_test, d.y_test = d.X_train[:n_train], d.y_train[:n_train], d.X_test[:n_test], d.y_test[:n_test]
    return d


def test_example_two_channels():
    d1, d2 = load_example("pneumatic_seg"), load_example("pneumatic_seg2")
    assert channels_of(d1.length) == 1 and channels_of(d2.length) == 2 and int(d2.length) == 256
    assert d2.X_train.shape[1] == 512 and np.allclose(d2.X_train[:, :256].std(), 1, atol=0.1)
    assert np.allclose(d2.X_train[:, 256:].std(), 1, atol=0.1)                  # 채널마다 정규화
    s = d2.summary()
    json.dumps(s, ensure_ascii=False)
    assert s["channels"] == 2 and s["channel_names"] == ["전류 (A)", "공압 압력 (bar)"] and "channels" not in d1.summary()
    c = check_graph(from_blocks(T["seg_tcn"]["blocks"]), d2.length, d2.n_out)
    assert c.ok and c.shapes["input"] == (2, 256)
    assert plan_dict(from_blocks(SMALL), d2.length, d2.n_out)["in_shape"] == [2, 256]


def test_model_input_channel_first():
    X = np.arange(2 * 6).reshape(1, 12).astype(float)               # 채널 1: 0–5, 채널 2: 6–11
    assert to_model_input("cnn", X, channels=2).tolist() == [[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]]
    assert to_model_input("rnn", X, channels=2)[0, 0].tolist() == [0, 6]
    assert to_model_input("cnn", X).shape == (1, 1, 12)


def test_torch_training_and_predict_and_code():
    d = small()
    g = from_blocks(SMALL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = train_job(Job("t"), g, d, TrainSettings(epochs=2))
    assert 0 <= r["test_metric"] <= 1 and len(r["seg_samples"][0]["x"]) == 256
    from dlstudio.core import Studio
    s = Studio()
    s.api_data_load({"example": "pneumatic_seg2"})
    s.model, s.model_graph, s.results = r["model"], g, {k: v for k, v in r.items() if k != "model"}
    p = s.api_predict({"example_new": True})
    assert p["has_label"] and len(p["rows"]) == 10
    a = s.api_activations({"sample_index": 0})
    assert len(a["input"]) == 512
    code = s.api_code({"framework": "pytorch", "graph": g, "settings": {"epochs": 1}})["code"]
    assert "(샘플 수, 2, 256)" in code
    ns = {}
    exec(compile(code.replace('if __name__ == "__main__":', "if False:"), "studio.py", "exec"), ns)
    assert ns["StudioModel"]()(torch.zeros(3, 2, 256)).shape == (3, int(d.n_out))
    s.api_data_load({"example": "pneumatic_seg"})
    assert s.data_key() != json.dumps([])                       # 채널 하나는 예전 열쇠 그대로 (채널 수 없음)
    assert json.loads(s.data_key())[-1] is None


def test_web_engine_two_channels_matches_torch(tmp_path):
    from tests.test_engine import _train_js, pytestmark as engine_mark
    if engine_mark.mark.args[0]:
        pytest.skip(engine_mark.mark.kwargs.get("reason"))
    d = small(n_train=48, n_test=16)
    g = from_blocks(SMALL)
    w = _train_js(tmp_path, g, d, {"epochs": 1, "lr": 1e-3, "batch_size": 16, "seed": 0})
    net = build_graph_model(g, d.length, d.n_out).eval()
    net.load_state_dict({k: torch.as_tensor(np.asarray(v["data"], dtype=np.float32).reshape(v["shape"])) for k, v in w["state"].items()})
    with torch.no_grad():
        out = net(torch.as_tensor(to_model_input("cnn", d.X_test, channels=2), dtype=torch.float32)).numpy()
    assert np.allclose(out.ravel(), np.asarray(w["out"]), atol=1e-3)


def _series(tmp_path, n=12, L=40):
    rng = np.random.default_rng(0)
    for cls in ("정상", "이상"):
        (tmp_path / cls).mkdir()
        for i in range(n):
            t = np.arange(L)
            a = np.sin(t / 3) + (0.8 if cls == "이상" else 0) * (t > L // 2) + rng.normal(0, 0.1, L)
            b = np.cos(t / 5) + rng.normal(0, 0.1, L)
            rows = ["시간,전류,압력"] + [f"{k},{x:.4f},{y:.4f}" for k, x, y in zip(t, a, b)]
            (tmp_path / cls / f"{i}.csv").write_text("\n".join(rows), encoding="utf-8")


def test_series_folder_two_columns(tmp_path):
    _series(tmp_path)
    d = load_series_folder(str(tmp_path), ["전류", "압력"], 32)
    assert channels_of(d.length) == 2 and int(d.length) == 32 and d.X_train.shape[1] == 64
    assert d.summary()["channel_names"] == ["전류", "압력"]
    d1 = load_series_folder(str(tmp_path), "전류", 32)
    assert channels_of(d1.length) == 1 and d1.X_train.shape[1] == 32


def test_csv_roles(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    grade = np.where(a + b > 0, "좋음", "나쁨")
    rows = ["ID,a,b,c,등급,수율"] + [f"{i},{a[i]:.3f},{b[i]:.3f},{c[i]:.3f},{grade[i]},{2 * a[i] - c[i] + 50:.3f}" for i in range(n)]
    f = tmp_path / "t.csv"
    f.write_text("\n".join(rows), encoding="utf-8")
    d = load_csv(str(f))                                              # 기본: 마지막 열이 정답
    s = d.summary()
    assert d.task == "regression" and [c["role"] for c in s["columns"]] == ["input"] * 4 + ["skip", "target"]   # 글자 열은 안 씀
    d = load_csv(str(f), targets=[4, 5], skip=[0])                    # 정답 둘 (분류 + 값), ID는 안 씀
    assert d.task == "multi" and [h["kind"] for h in d.n_out.heads] == ["class", "value"] and d.X_train.shape[1] == 3
    assert [c["role"] for c in d.summary()["columns"]] == ["skip", "input", "input", "input", "target", "target"]
    from dlstudio.core import Studio
    st = Studio()
    st.api_data_load({"path": str(f)})
    r = st.api_data_load({"roles_reload": {"targets": [4, 5], "skip": [0], "channels": 1}})
    assert r["summary"]["task"] == "multi" and [t["name"] for t in r["summary"]["targets"]] == ["등급", "수율"]
    st.load_source(st.data.source)                                    # 프로젝트 다시 열기: 역할 그대로
    assert st.data.task == "multi" and st.data.X_train.shape[1] == 3

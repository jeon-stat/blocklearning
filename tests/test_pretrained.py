"""사전학습 블록 (Google MobileNetV3-Small, 고정): 가중치 파일·출처 · 학습해도 그대로 · 특징을 한 번만 뽑아도
같은 결과 · 프로젝트·학습 기록에는 고정 가중치를 넣지 않고 열 때 다시 채움 · 웹 엔진 학습."""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from dlstudio import codegen, layers, tasks
from dlstudio.data import load_example
from dlstudio.graph import build_graph_model, from_blocks
from dlstudio.jobs import Job, TrainSettings, train_job
from dlstudio.nn_extra import MobileNetV3Small
from dlstudio.pretrained import arrays, frozen_names

T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}
GRAPH = from_blocks(T["mnv3_transfer"]["blocks"])
ENGINE = Path(__file__).parents[1] / "src" / "dlstudio" / "static" / "engine.js"


def small(kind="steelblast", n_train=120, n_test=60):
    d = load_example(kind)
    d.X_train, d.y_train, d.X_test, d.y_test = d.X_train[:n_train], d.y_train[:n_train], d.X_test[:n_test], d.y_test[:n_test]
    return d


def test_weights_file_matches_block_definition():
    a = arrays("mnv3s")
    specs = layers.mnv3s_params("k")
    assert {n[2:]: s for n, s, kind, _ in specs} == {k: list(v.shape) for k, v in a.items()}
    assert all(kind == "frozen" for _, _, kind, _ in specs)
    assert MobileNetV3Small.CFG == layers.MNV3S_CFG
    js = re.search(r"const MNV3S = (\[\[.*?\]\]);", ENGINE.read_text(encoding="utf-8"), re.S).group(1)
    assert tuple(tuple(r) for r in json.loads(js)) == layers.MNV3S_CFG          # 웹 엔진도 같은 구조
    m = MobileNetV3Small()
    assert set(m.state_dict()) == set(a) and not any(p.requires_grad for p in m.parameters())


def test_source_and_license_files():
    d = Path(__file__).parents[1] / "src" / "dlstudio" / "pretrained"
    src = (d / "mnv3s.SOURCE.txt").read_text(encoding="utf-8")
    assert "Apache License 2.0" in src and "Google" in src
    assert "Apache License" in (d / "LICENSE-mnv3s.txt").read_text(encoding="utf-8")


def test_check_rules():
    from dlstudio.graph import check_graph
    d = load_example("steelblast")
    assert check_graph(GRAPH, d.length, d.n_out, image=d.image).ok
    e = load_example("ecg")                                       # 시계열에는 못 씀
    assert not check_graph(GRAPH, e.length, e.n_out, image=e.image).ok
    plan = codegen.plan_dict(GRAPH, d.length, d.n_out, d.image)
    assert plan["frozen"] == sum(v.size for v in arrays("mnv3s").values()) and plan["params"] == 1024 * 2 + 2
    assert codegen.keras_unsupported(GRAPH, d.length, d.n_out, d.image)


def test_training_keeps_pretrained_and_caching_gives_same_result(monkeypatch):
    d = small()
    s = TrainSettings(epochs=3, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = train_job(Job("a"), GRAPH, d, s)
        from dlstudio import graph as G
        cls = G._graph_net_class()
        monkeypatch.setattr(cls, "frozen_head", lambda self: None, raising=False)
        monkeypatch.setattr(G, "_graph_net_class", lambda: cls)
        slow = train_job(Job("b"), GRAPH, d, s)
    a = arrays("mnv3s")
    for k, v in fast["model"].state_dict().items():
        if ".m_" in k and k.split(".", 2)[2] in a:
            assert np.array_equal(v.numpy(), a[k.split(".", 2)[2]]), k          # 고정: 그대로
    assert fast["test_metric"] == pytest.approx(slow["test_metric"], abs=1e-9)
    for k, v in fast["model"].state_dict().items():
        assert np.allclose(v.numpy(), slow["model"].state_dict()[k].numpy(), atol=1e-4), k


def test_project_and_runs_leave_out_frozen_weights(tmp_path):
    from dlstudio.core import Studio
    s = Studio()
    s.api_data_load({"example": "steelblast"})
    s.data = small()
    s.graph = GRAPH
    r = train_job(Job("t"), GRAPH, s.data, TrainSettings(epochs=2))
    s.model, s.model_graph, s.results = r.pop("model"), GRAPH, r
    s.add_run(GRAPH, {}, r, {k: v.numpy() for k, v in s.model.state_dict().items()})
    frozen = frozen_names(GRAPH, s.data.length, s.data.n_out, s.data.image)
    assert len(frozen) == len(arrays("mnv3s"))
    assert not frozen & set(s.run_weights[s.current_run])
    path = tmp_path / "p.dlstudio"
    s.api_project_save({"path": str(path)})
    assert path.stat().st_size < 200_000                         # 사전학습 3.7MB는 넣지 않음
    from dlstudio.project import open_project
    pr = open_project(str(path))
    assert not frozen & set(pr["state_dict"]) and not frozen & set(pr["run_weights"][s.current_run])
    from dlstudio.graph import model_from_saved
    m = model_from_saved(pr, s.data.length, s.data.n_out, image=s.data.image).eval()
    x = torch.as_tensor(s.data.X_test[:8].reshape(-1, 1, *s.data.image), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(m(x), s.model.eval()(x), atol=1e-6)
    s.api_runs_restore({"n": s.current_run})                     # 학습 기록으로 돌아가기도 됨


def test_web_engine_trains_transfer_model(tmp_path):
    from tests.test_engine import _train_js, pytestmark as engine_mark
    if engine_mark.mark.args[0]:
        pytest.skip(engine_mark.mark.kwargs.get("reason"))
    d = small(n_train=96, n_test=48)
    r = _train_js(tmp_path, GRAPH, d, {"epochs": 3, "lr": 1e-3, "batch_size": 32, "seed": 0})
    assert r["result"]["epochs_done"] == 3 and len(r["progress"]) == 3
    net = build_graph_model(GRAPH, d.length, d.n_out, image=d.image).eval()
    net.load_state_dict({k: torch.as_tensor(np.asarray(v["data"], dtype=np.float32).reshape(v["shape"])) for k, v in r["state"].items()})
    a = arrays("mnv3s")
    for k, v in r["state"].items():
        if k.split(".", 2)[-1] in a and ".m_" in k:
            assert np.allclose(np.asarray(v["data"]), a[k.split(".", 2)[-1]].ravel()), k        # 엔진도 고정
    with torch.no_grad():
        out = net(torch.as_tensor(d.X_test.reshape(-1, 1, *d.image), dtype=torch.float32)).numpy()
    assert float(np.mean(out.argmax(1) == d.y_test)) == pytest.approx(r["result"]["test_metric"], abs=1e-9)


def test_image_anomaly_distance_head_torch_and_web_engine(tmp_path):
    """이미지 이상 탐지 '정상과의 거리': 경사 학습 없이 정상 특징 분포를 넣음 → AUC, 웹 엔진도 같은 점수."""
    from tests.test_engine import _train_js, pytestmark as engine_mark
    from dlstudio.graph import check_graph
    d = load_example("steelblast_ae")
    d.X_train, d.y_train = d.X_train[:96], d.y_train[:96]
    pick = np.r_[np.flatnonzero(d.y_test == 0)[:32], np.flatnonzero(d.y_test == 1)[:32]]   # 정상 32 · 이상 32
    d.X_test, d.y_test = d.X_test[pick], d.y_test[pick]
    g = from_blocks(T["ae_img_dist"]["blocks"])
    assert check_graph(g, d.length, d.n_out, image=d.image).ok
    bad = from_blocks([{"type": "mnv3s"}, {"type": "dense", "units": 8}, {"type": "output", "head": "distance"}])
    assert not check_graph(bad, d.length, d.n_out, image=d.image).ok          # 학습하는 층 뒤에는 못 둠
    job = Job("t")
    r = train_job(job, g, d, TrainSettings(epochs=30))
    assert r["epochs_done"] == 1 and len(job.progress) == 1 and r["mode"] == "distance" and r["auc"] > 0.8
    assert all(p["recon"] is None for p in r["ae_samples"])
    m = r["model"]
    head, _ = m.distance_head()
    x = torch.as_tensor(d.X_test.reshape(-1, 1, *d.image), dtype=torch.float32)
    with torch.no_grad():
        want = m(x).numpy()[:, 0]
    assert np.allclose(want, r["scores"], atol=1e-4)
    if engine_mark.mark.args[0]:
        return
    params = {k: getattr(head, k).numpy().ravel().tolist() for k in ("mean", "basis", "inv_var", "inv_rest")}
    w = _train_js(tmp_path, g, d, {}, distance=params)
    assert w["got"]["n"] == len(d.X_train) and w["got"]["c"] == 1024
    assert np.allclose(np.asarray(w["out"]), want, rtol=2e-3, atol=2e-3)
    assert w["result"]["test_metric"] == pytest.approx(r["auc"], abs=0.02)


def test_fine_tuning_trains_only_the_last_blocks_and_is_saved(tmp_path):
    """미세조정 (끝 블록 1개): 앞부분은 사전학습 값 그대로, 끝 블록·마지막 1×1 합성곱은 바뀜. 학습한 값은 프로젝트에 저장됨."""
    d = small(n_train=64, n_test=32)
    g = from_blocks([{"type": "mnv3s", "size": 64, "train_last": 1}, {"type": "output"}])
    plan = codegen.plan_dict(g, d.length, d.n_out, d.image)
    tuned = {p[0] for s in plan["steps"] for p in s["params"] if p[2] == "weight" and p[3:] == ["pretrained"]}
    assert any(".blocks.10." in k for k in tuned) and any(k.endswith(".head.weight") for k in tuned)
    assert not any(".blocks.9." in k for k in tuned)
    r = train_job(Job("t"), g, d, TrainSettings(epochs=2, lr=1e-3))
    a = arrays("mnv3s")
    sd = r["model"].state_dict()
    for k, v in sd.items():
        name = k.split(".", 2)[-1]
        if name in a:
            assert np.array_equal(v.numpy(), a[name]) == (k not in tuned), k
    frozen = frozen_names(g, d.length, d.n_out, d.image)
    assert frozen and not frozen & tuned
    from dlstudio.graph import model_from_saved
    m = model_from_saved({"graph": g, "state_dict": {k: v.numpy() for k, v in sd.items() if k not in frozen}},
                         d.length, d.n_out, image=d.image).eval()
    x = torch.as_tensor(d.X_test.reshape(-1, 1, *d.image), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(m(x), r["model"].eval()(x), atol=1e-6)


def test_web_engine_fine_tunes_last_block(tmp_path):
    from tests.test_engine import _train_js, pytestmark as engine_mark
    if engine_mark.mark.args[0]:
        pytest.skip(engine_mark.mark.kwargs.get("reason"))
    d = small(n_train=48, n_test=24)
    g = from_blocks([{"type": "mnv3s", "size": 64, "train_last": 1}, {"type": "output"}])
    r = _train_js(tmp_path, g, d, {"epochs": 2, "lr": 1e-3, "batch_size": 16, "seed": 0})
    a = arrays("mnv3s")
    for k, v in r["state"].items():
        name = k.split(".", 2)[-1]
        if name in a and ".m_" in k:
            same = np.allclose(np.asarray(v["data"]), a[name].ravel())
            assert same == (".blocks.10." not in k and ".last." not in k and ".head." not in k), k

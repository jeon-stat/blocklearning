"""객체 탐지: 상자 ↔ 격자 목표값, 상자 풀기·mAP (손으로 구한 값), 격자 출력 검사, torch 학습, 라벨링 도형 → 탐지 데이터,
예측 표, 생성한 PyTorch 코드, 웹 엔진(TF.js)이 Python과 같은 값을 내는지."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from blocklearning import codegen, detect
from blocklearning.data import make_det_dataset
from tests.synth import parts_det_arrays
from blocklearning.graph import build_graph_model, check_graph, from_blocks
from blocklearning.tasks import T_CNN2D, T_DET_RES, T_DET_SMALL

JS = Path(__file__).parent / "js"
HAS_NODE = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()


def _small(n=60, seed=0):
    X, B = parts_det_arrays(n=n, seed=seed)
    return make_det_dataset(X, B, ["원형 부품", "사각 부품"], 64, seed=seed)


def _perfect(d, part="test"):
    """목표값 → 그 목표를 정확히 내는 모델 출력 (logit)."""
    G, A = d.n_out.grid, d.n_out.per_cell
    y = d.y_test if part == "test" else d.y_train
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    out = []
    for t in y:
        t3, o = t.reshape(A, G, G), np.zeros((A, G, G))
        o[0] = np.where(t3[0] > 0, 9, -9)
        o[1:5] = lg(t3[1:5])
        o[5:] = t3[5:] * 12
        out.append(o.reshape(-1))
    return np.asarray(out)


def test_encode_decode_round_trip_and_perfect_map():
    d = _small()
    assert d.task == "detection" and d.n_out == 8 * 8 * 7 and d.n_out.grid == 8 and d.image == (64, 64)
    m, aps = detect.average_precision(detect.decode_all(_perfect(d), 8, 2), d.det["test"], 2)
    assert m == 1.0 and aps == [1.0, 1.0]
    got = detect.decode(_perfect(d)[0], 8, 2)
    want = sorted(d.det["test"][0], key=lambda b: b[1])
    assert len(got) == len(want)
    for g in got:                                              # 같은 클래스·거의 같은 상자
        assert any(g[0] == w[0] and detect.iou(g[2:], w[1:]) > 0.99 for w in want)


def test_average_precision_by_hand():
    """정답 2개, 예측 3개 (0.9 맞음 · 0.8 틀림 · 0.7 맞음) → 정밀도 1, ½, ⅔ · 재현율 ½, ½, 1 → AP = ½·1 + ½·⅔."""
    gts = [[[0, 0.1, 0.1, 0.2, 0.2], [0, 0.6, 0.6, 0.2, 0.2]]]
    preds = [[[0, 0.9, 0.1, 0.1, 0.2, 0.2], [0, 0.8, 0.35, 0.35, 0.1, 0.1], [0, 0.7, 0.61, 0.6, 0.2, 0.2]]]
    m, aps = detect.average_precision(preds, gts, 2)
    assert abs(m - (0.5 + 0.5 * 2 / 3)) < 1e-6 and aps[1] is None          # 정답이 없는 클래스는 평균에서 뺌
    pr = detect.precision_recall(preds, gts, 2, score=0.75)
    assert pr == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5}
    assert detect.average_precision([[]], gts, 1)[0] == 0.0
    dup = [[[0, 0.9, 0.1, 0.1, 0.2, 0.2], [0, 0.85, 0.1, 0.1, 0.2, 0.2]]]   # 같은 정답을 두 번 찾으면 두 번째는 틀림
    assert detect.precision_recall(dup, [gts[0][:1]], 1, score=0.5)["fp"] == 1


def test_nms_and_collision():
    c = [[0, 0.9, 0.1, 0.1, 0.3, 0.3], [0, 0.8, 0.12, 0.1, 0.3, 0.3], [1, 0.7, 0.12, 0.1, 0.3, 0.3]]
    assert [b[1] for b in detect.nms(c)] == [0.9, 0.7]            # 같은 클래스만 지움
    t = detect.encode([[0, 0.1, 0.1, 0.05, 0.05], [1, 0.08, 0.08, 0.1, 0.1]], 8, 2).reshape(7, 8, 8)
    assert t[0].sum() == 1 and t[6, 1, 1] == 1                      # 같은 칸이면 큰 상자만


def test_check_needs_grid_before_output():
    d = _small()
    for T in (T_DET_SMALL, T_DET_RES):
        assert check_graph(from_blocks(T["blocks"]), d.length, d.n_out, image=d.image).ok
    c = check_graph(from_blocks(T_CNN2D["blocks"]), d.length, d.n_out, image=d.image)
    assert not c.ok and "8×8 격자" in c.issues[0].message and "AvgPool2D" in c.issues[0].hint
    import copy
    import pickle
    assert copy.deepcopy(d.n_out).grid == 8 and pickle.loads(pickle.dumps(d.n_out)).per_cell == 7


def test_torch_training_learns_and_reports():
    torch = pytest.importorskip("torch")
    from blocklearning.jobs import Job, TrainSettings, train_job
    X, B = parts_det_arrays(n=160, seed=3)
    d = make_det_dataset(X, B, ["원형 부품", "사각 부품"], 64, seed=1)
    j = Job("train")
    r = train_job(j, from_blocks(T_DET_SMALL["blocks"]), d, TrainSettings(epochs=15, lr=0.003, seed=0))
    assert r["metric_name"] == "map50" and r["test_metric"] > 0.3 and len(r["ap"]) == 2, r["test_metric"]
    assert j.progress[-1]["test_metric"] > j.progress[0]["test_metric"]
    assert len(j.progress) == 15 and j.progress[-1]["test_metric"] == pytest.approx(r["test_metric"], abs=1e-9)
    assert len(r["det_pred"]) == len(d.det["test"]) and set(r["pr"]) == {"tp", "fp", "fn", "precision", "recall"}
    assert 0 <= r["train_metric"] <= 1 and isinstance(r["model"], torch.nn.Module)


def test_label_boxes_become_detection_data(tmp_path):
    """라벨링 도형 → [학습 데이터로 쓰기]: 다각형·원은 감싸는 상자, 점은 뺌, 상자가 없는 클래스는 뺌."""
    PIL = pytest.importorskip("PIL.Image")
    from blocklearning.core import Studio
    root = tmp_path / "그림"
    root.mkdir()
    for i in range(12):
        PIL.new("L", (40, 20), 255).save(root / f"g{i:02d}.png")
    s = Studio()
    c, st = s.call("POST", "/api/label/new", {"path": str(root), "classes": ["흠", "구멍", "안씀"], "file": str(tmp_path / "w.dllabel"),
                                              "kind": "image_folder", "schema": "boxes"})
    assert c == 200, st
    for i in range(12):
        shapes = [[0.1, 0.1, 0.2, 0.4, "흠"], ["circle", 0.6, 0.5, 0.1, "구멍"], ["pt", 0.9, 0.9, "안씀"]]
        if i == 0:
            shapes.append(["poly", [[0.5, 0.1], [0.9, 0.1], [0.7, 0.3]], "흠"])
        assert s.call("POST", "/api/label/set_boxes", {"name": f"g{i:02d}.png", "boxes": shapes})[0] == 200
    c, r = s.call("POST", "/api/label/use", {})
    assert c == 200, r
    sm = r["summary"]
    assert sm["task"] == "detection" and sm["classes"] == ["흠", "구멍"] and sm["class_counts"] == {"흠": 13, "구멍": 12}
    assert sm["image"] == [64, 64] and sm["n_train"] + sm["n_test"] == 12 and sm["detection"]["grid"] == 8
    det = json.loads(Path(r["label_file"]).read_text(encoding="utf-8"))
    g0 = next(it for it in det["items"] if it["name"] == "g00.png")["boxes"]
    assert [b[0] for b in g0] == [0, 1, 0]
    assert g0[1][1:] == pytest.approx([0.5, 0.3, 0.2, 0.4])              # 원: 너비 비율 0.1 → 세로는 0.2 (40×20 그림)
    assert g0[2][1:] == pytest.approx([0.5, 0.1, 0.4, 0.2])              # 다각형: 감싸는 상자
    assert s.load_source(s.data.source)["task"] == "detection"           # 프로젝트를 다시 열 때 같은 데이터
    few = Studio()
    few.call("POST", "/api/label/new", {"path": str(root), "classes": ["흠"], "file": str(tmp_path / "w2.dllabel"),
                                        "kind": "image_folder", "schema": "boxes"})
    few.call("POST", "/api/label/set_boxes", {"name": "g00.png", "boxes": [[0.1, 0.1, 0.2, 0.2, "흠"]]})
    c, e = few.call("POST", "/api/label/use", {})
    assert c == 400 and "1장" in e["error"]["message"]


def test_predict_table_lists_boxes():
    from blocklearning import predict
    d = _small()
    out = _perfect(d)
    t = predict.table(d, [f"n{i}.png" for i in range(len(out))], out, None, d.X_test)
    n = sum(len(b) for b in d.det["test"])
    assert t["summary"]["n_boxes"] == n and sum(t["summary"]["counts"].values()) == n
    assert len(t["pics"]) == len(out) and len(t["pics"][0]["values"]) == 64 * 64
    assert t["csv"].splitlines()[0].lstrip("﻿") == "그림,클래스,점수,x,y,너비,높이" and len(t["csv"].splitlines()) == n + 1
    assert predict.new_data_kinds(d) == ["folder"]


def test_pytorch_code_for_detection_matches_and_predicts(tmp_path):
    torch = pytest.importorskip("torch")
    d = _small()
    g = from_blocks(T_DET_SMALL["blocks"])
    net = build_graph_model(g, d.length, d.n_out, image=d.image).eval()
    info = {"task": "detection", "classes": d.classes}
    code = codegen.to_pytorch(g, d.length, d.n_out, d.image, info, weights_file="w.npz")
    assert "def det_loss" in code and "def decode" in code and "nn.Conv2d(32, 7, 1)" in code
    codegen.save_npz(codegen.code_state(net.state_dict(), g, d.length, d.n_out, d.image), tmp_path / "w.npz", data=d)
    ns = {}
    exec(compile(code, "<det>", "exec"), ns)
    model, z = ns["load_trained"](str(tmp_path / "w.npz"))
    x = torch.as_tensor(d.X_test[:4].reshape(-1, 1, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        a, b = model(x), net(x)
    assert torch.allclose(a, b, atol=1e-5)
    t = torch.as_tensor(d.y_test[:4], dtype=torch.float32)
    assert float(ns["det_loss"](a, t)) == pytest.approx(float(detect.torch_loss(b, t, 8, 7)), rel=1e-5)
    raw = d.raw(d.X_test[:2]).reshape(2, 64, 64)
    got = ns["predict"](model, z, raw)
    want = [detect.decode(r, 8, 2, detect.SHOW_SCORE) for r in b[:2].numpy()]
    assert [len(x) for x in got] == [len(x) for x in want]
    keras_code = codegen.to_keras(g, d.length, d.n_out, d.image, info)
    assert "Permute((3, 1, 2)" in keras_code and "def det_loss" in keras_code
    compile(keras_code, "<keras>", "exec")


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
def test_engine_detection_math_matches_python(tmp_path):
    """웹 엔진의 상자 풀기·mAP·정밀도·손실이 detect.py와 같음 (무작위 출력, 같은 float32 값)."""
    torch = pytest.importorskip("torch")
    d = _small(n=40)
    rng = np.random.default_rng(0)
    out = _perfect(d).reshape(len(d.X_test), 7, 8, 8)                 # 일부는 맞고 일부는 틀리게: 있음·클래스는 크게, 위치는 조금 흔듦
    out[:, 0] += rng.normal(0, 6, out[:, 0].shape)
    out[:, 1:5] += rng.normal(0, 0.35, out[:, 1:5].shape)
    out[:, 5:] += rng.normal(0, 8, out[:, 5:].shape)
    out = out.reshape(len(d.X_test), -1).astype(np.float32)
    det = {"grid": 8, "per_cell": 7, "n_cls": 2}
    f = tmp_path / "det.json"
    f.write_text(json.dumps({"out": out.ravel().tolist(), "gts": d.det["test"], "det": det,
                             "target": d.y_test.astype(np.float32).ravel().tolist()}), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_detect.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    js = json.loads(r.stdout)
    py = detect.result(out.astype(np.float64), d.det["test"], 8, 2)
    assert 0 < py["test_metric"] < 1
    assert js["res"]["test_metric"] == pytest.approx(py["test_metric"], abs=1e-6) and js["res"]["ap"] == pytest.approx(py["ap"], abs=1e-6)
    assert {k: js["res"]["pr"][k] for k in ("tp", "fp", "fn")} == {k: py["pr"][k] for k in ("tp", "fp", "fn")}
    for a, b in zip(js["decoded"], detect.decode_all(out.astype(np.float64), 8, 2)):
        assert len(a) == len(b) and all(x[0] == y[0] and np.allclose(x[1:], y[1:], atol=1e-5) for x, y in zip(a, b))
    want = float(detect.torch_loss(torch.as_tensor(out), torch.as_tensor(d.y_test, dtype=torch.float32), 8, 7))
    assert js["loss"] == pytest.approx(want, rel=1e-4)


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
def test_engine_trains_detection_and_weights_run_in_torch(tmp_path):
    """엔진(TF.js) 학습: mAP가 오르고, 학습한 가중치를 PyTorch 모델에 넣으면 같은 mAP (웹 → PyTorch 내보내기)."""
    torch = pytest.importorskip("torch")
    X, B = parts_det_arrays(n=70, size=32, seed=5)                    # node는 CPU라 작게 (32×32 → 격자 4×4)
    d = make_det_dataset(X, B, ["원형 부품", "사각 부품"], 32, seed=2)
    g = from_blocks(T_DET_SMALL["blocks"])
    plan = codegen.plan_dict(g, d.length, d.n_out, d.image)
    assert plan["steps"][-1]["cfg"]["head"] == "grid" and plan["steps"][-1]["params"][0][1] == [7, 32, 1, 1] and d.n_out.grid == 4
    data = {"X_train": d.X_train.ravel().tolist(), "X_test": d.X_test.ravel().tolist(), "y_train": d.y_train.ravel().tolist(),
            "y_test": d.y_test.ravel().tolist(), "n_train": len(d.X_train), "n_test": len(d.X_test), "task": "detection",
            "n_out": int(d.n_out), "length": d.length, "image": [32, 32],
            "det": {"grid": 4, "per_cell": 7, "n_cls": 2, "train": d.det["train"], "test": d.det["test"]}}
    f = tmp_path / "train.json"
    f.write_text(json.dumps({"plan": plan, "data": data, "settings": {"epochs": 10, "lr": 0.005, "batch_size": 16, "seed": 0}}),
                 encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_train.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout)
    res = out["result"]
    assert res["metric_name"] == "map50" and res["test_metric"] > 0.1 and len(res["det_pred"]) == len(d.X_test), res["test_metric"]
    assert len(out["progress"]) == 10 and out["progress"][-1]["test_metric"] > out["progress"][0]["test_metric"]
    net = build_graph_model(g, d.length, d.n_out, image=d.image).eval()
    net.load_state_dict({k: torch.tensor(v["data"]).reshape(v["shape"]).to(net.state_dict()[k].dtype) for k, v in out["state"].items()})
    with torch.no_grad():
        o = net(torch.as_tensor(d.X_test.reshape(-1, 1, 32, 32), dtype=torch.float32)).double().numpy()
    assert detect.metric(o, d.det["test"], 4, 2) == pytest.approx(res["test_metric"], abs=0.02)

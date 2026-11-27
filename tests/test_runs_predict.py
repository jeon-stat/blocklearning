"""학습 기록(되돌리기·프로젝트 저장), 새 데이터로 추론(샘플마다 추론 과정), 틀린 샘플 모아 보기, 블록 미리보기."""
import json
import time

import numpy as np
import pytest

from blocklearning import web
from blocklearning.core import Studio
from blocklearning.tasks import DEFAULT_TEMPLATES, _view

torch = pytest.importorskip("torch")
G = _view(DEFAULT_TEMPLATES[0])["graph"]


def _train(s, epochs, lr=0.01):
    c, r = s.call("POST", "/api/train", {"graph": G, "settings": {"epochs": epochs, "lr": lr}})
    assert c == 200, r
    for _ in range(600):
        st = s.call("GET", f"/api/jobs/{r['id']}")[1]
        if st["state"] != "running":
            assert st["state"] == "done", st
            return st["result"]
        time.sleep(0.1)
    raise AssertionError("학습이 끝나지 않음")


@pytest.fixture(scope="module")
def trained():
    s = Studio()
    assert s.call("POST", "/api/data/load", {"example": "ecg"})[0] == 200
    _train(s, 1)
    _train(s, 3)
    return s


def test_runs_record_restore_delete(trained, tmp_path):
    s = trained
    runs = s.call("GET", "/api/runs")[1]["runs"]
    assert [r["n"] for r in runs] == [1, 2] and runs[1]["current"] and all(r["data_ok"] and r["weights"] for r in runs)
    c, r = s.call("POST", "/api/runs/restore", {"n": 1})
    assert c == 200 and r["results"]["epochs_done"] == 1 and s.current_run == 1
    d = s.data
    x = torch.as_tensor(s.model_input(d.X_test), dtype=torch.float32)
    with torch.no_grad():
        acc = float((s.model(x).argmax(1).numpy() == d.y_test).mean())
    assert abs(acc - r["results"]["test_metric"]) < 1e-9                # 되돌린 모델 = 그때의 성적
    # 프로젝트에 기록과 가중치가 함께
    p = str(tmp_path / "p.blocklearning")
    assert s.call("POST", "/api/project/save", {"path": p})[0] == 200
    s2 = Studio()
    c, r = s2.call("POST", "/api/project/open", {"path": p})
    assert c == 200 and [x["n"] for x in r["runs"]] == [1, 2] and [x["current"] for x in r["runs"]] == [True, False]
    assert s2.call("POST", "/api/runs/restore", {"n": 2})[0] == 200
    # 다른 데이터의 기록은 되돌리지 못함
    s2.call("POST", "/api/data/load", {"example": "vibration"})
    assert not any(x["data_ok"] for x in s2.call("GET", "/api/runs")[1]["runs"])
    assert s2.call("POST", "/api/runs/restore", {"n": 2})[0] == 400
    assert [x["n"] for x in s2.call("POST", "/api/runs/delete", {"n": 1})[1]["runs"]] == [2]


def test_predict_csv_with_and_without_labels(trained, tmp_path):
    s = trained
    d = s.data
    raw = d.raw(d.X_test[:30])
    with_label = tmp_path / "a.csv"
    with_label.write_text("\n".join(",".join(f"{v:.6f}" for v in r) + "," + d.classes[int(y)] for r, y in zip(raw, d.y_test[:30])),
                          encoding="utf-8")
    c, r = s.call("POST", "/api/predict", {"path": str(with_label)})
    assert c == 200 and r["summary"]["n"] == 30 and r["has_label"]
    x = torch.as_tensor(s.model_input(d.X_test[:30]), dtype=torch.float32)
    with torch.no_grad():
        want = [d.classes[i] for i in s.model(x).argmax(1).numpy()]
    assert [row["pred"] for row in r["rows"]] == want                  # 원래 값 → 같은 정규화 → 같은 예측
    assert abs(r["summary"]["accuracy"] - np.mean([w == d.classes[int(y)] for w, y in zip(want, d.y_test[:30])])) < 1e-9
    assert r["csv"].startswith("﻿샘플,예측,확신도")
    no_label = tmp_path / "b.csv"
    no_label.write_text("\n".join(",".join(f"{v:.6f}" for v in r) for r in raw[:5]), encoding="utf-8")
    c, r = s.call("POST", "/api/predict", {"path": f'"{no_label}"'})
    assert c == 200 and not r["has_label"] and len(r["rows"]) == 5 and "accuracy" not in r["summary"]
    bad = tmp_path / "c.csv"
    bad.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    c, r = s.call("POST", "/api/predict", {"path": str(bad)})
    assert c == 400 and "열이 3개" in r["error"]["message"]


def test_predict_tabular_rows_and_regression():
    s = Studio()
    s.call("POST", "/api/data/load", {"example": "process_cls"})
    summ = s.data.summary()
    assert "rows" in summ["predict_kinds"] and len(summ["feature_means"]) == s.data.length
    from blocklearning.tasks import catalog_view
    g = next(t["graph"] for f in catalog_view()["families"] for sub in f["subtasks"] for t in sub.get("templates", []) if t["id"] == "mlp")
    c, r = s.call("POST", "/api/train", {"graph": g, "settings": {"epochs": 2}})
    while s.call("GET", f"/api/jobs/{r['id']}")[1]["state"] == "running":
        time.sleep(0.1)
    c, r = s.call("POST", "/api/predict", {"rows": [summ["feature_means"]]})
    assert c == 200 and r["rows"][0]["pred"] in [str(x) for x in s.data.classes]
    assert s.call("POST", "/api/predict", {"rows": [[1, 2]]})[0] == 400


def test_samples_cell_and_preview(trained):
    s = trained
    s.call("POST", "/api/runs/restore", {"n": 2})
    pred = np.asarray(s.results["pred"])
    c, r = s.call("POST", "/api/samples", {"cell": [1, 0]})
    want = np.flatnonzero((s.data.y_test == 1) & (pred == 0)).tolist()
    assert c == 200 and r["indices"] == want and len(r["items"]) == min(len(want), 24)
    node = G["nodes"][1]["id"]
    c, r = s.call("POST", "/api/preview", {"graph": G, "node": node})
    assert c == 200 and r["trained"] and len(r["value"]) == G["nodes"][1]["channels"]
    g2 = json.loads(json.dumps(G))
    g2["nodes"][1]["kernel"] = 7
    c, r = s.call("POST", "/api/preview", {"graph": g2, "node": node})
    assert c == 200 and not r["trained"]
    assert s.call("POST", "/api/preview", {"graph": G, "node": "없는것"})[0] == 400


def test_web_runs_and_predict_without_torch_engine():
    """웹 쪽: 엔진 가중치로 기록 더하기 → 되돌리기 응답, 예측 준비·마무리 (엔진 출력 대신 torch로 계산해 넣음)."""
    from blocklearning.graph import build_graph_model
    s = web.WebStudio()
    s.call("POST", "/api/data/load", {"example": "ecg"})
    d = s.data
    net = build_graph_model(G, d.length, d.n_out, seed=1).eval()
    state = {k: {"shape": list(v.shape), "data": v.detach().numpy().astype(np.int32 if "num_batches" in k else np.float32).tobytes(),
                 "int": "num_batches" in k} for k, v in net.state_dict().items()}
    web.set_trained(s, json.dumps(G), json.dumps({"epochs": 1}), json.dumps({"test_metric": 0.5, "metric_name": "accuracy"}), state)
    assert [r["n"] for r in s.runs_view()] == [1]
    r = json.loads(web.run_restore(s, 1))
    assert r["status"] == 200 and r["body"]["plan"]["steps"] and set(r["body"]["state"]) == set(net.state_dict())
    r = json.loads(web.predict_prepare(s, json.dumps({"rows": d.raw(d.X_test[:4]).tolist()})))
    assert r["status"] == 200 and r["body"]["n"] == 4
    X = np.frombuffer(web.predict_x(s), dtype=np.float32).reshape(4, -1)
    assert np.allclose(X, d.X_test[:4], atol=1e-4)
    with torch.no_grad():
        out = net(torch.as_tensor(X[:, None, :])).numpy().astype(np.float32)
    r = json.loads(web.predict_finish(s, out.tobytes()))
    assert r["status"] == 200 and [x["pred"] for x in r["body"]["rows"]] == [d.classes[i] for i in out.argmax(1)]


# ------------------------------------------------------------ ⑤ 추론 과정: 새 데이터 한 샘플의 샘플 따라가기
def _check_probs(probs, k=5):
    ps = [q["p"] for q in probs]
    assert 0 < len(ps) <= k and ps == sorted(ps, reverse=True) and sum(ps) <= 1 + 1e-5 and all(0 <= p <= 1 for p in ps)


def test_activations_on_new_data(trained):
    from blocklearning.predict import read_new
    s = trained
    d = s.data
    c, r = s.call("POST", "/api/predict", {"example_new": True})
    assert c == 200
    new = read_new(d, {"example_new": True})
    rows = r["rows"]
    assert [x["index"] for x in rows] == list(range(len(rows)))
    for x in rows:
        _check_probs(x["probs"], 3)
        assert x["probs"][0]["name"] == x["pred"] and abs(x["probs"][0]["p"] - x["conf"]) < 1e-5
        assert len(x["input"]) == min(d.length, 100)             # 시계열 썸네일: 원래 값 (140점 → 고르게 100점)
    c, a = s.call("POST", "/api/activations", {"new_index": 1})
    assert c == 200 and a["index"] == 1 and a["name"] == new["names"][1] and a["label"] == new["labels"][1]
    assert np.allclose(a["input"], d.raw(new["X"][1]), atol=1e-4)      # 시험 샘플이 아니라 새 데이터의 값
    assert a["prediction"] == rows[1]["pred"]
    _check_probs(a["probs"])
    assert a["probs"][:3] == rows[1]["probs"]                       # 표와 같은 softmax
    c, b = s.call("POST", "/api/activations", {"rows": a["input"]})    # 원래 단위 한 줄 (What-if)
    assert c == 200 and b["label"] is None and b["index"] is None and b["prediction"] == a["prediction"]
    assert np.allclose(b["input"], a["input"], atol=1e-4)
    assert all(abs(p["p"] - q["p"]) < 1e-3 for p, q in zip(a["probs"], b["probs"]))
    assert s.call("POST", "/api/activations", {"new_index": len(rows)})[0] == 400
    assert s.call("POST", "/api/activations", {"rows": [1, 2]})[0] == 400
    t = s.call("POST", "/api/activations", {"sample_index": 0})[1]       # ④ 시험 샘플은 그대로 (+ probs)
    assert t["index"] == 0 and np.allclose(t["input"], d.raw(d.X_test[0]), atol=1e-4) and t["label"] == d.label_text("test", 0)
    _check_probs(t["probs"])


def test_new_data_is_cleared_when_data_changes():
    s = Studio()
    s.call("POST", "/api/data/load", {"example": "ecg"})
    _train(s, 1)
    assert s.call("POST", "/api/predict", {"example_new": True})[0] == 200 and s.new_data is not None
    _train(s, 1)                                                    # 새 모델: 지난 추론 결과는 버림
    assert s.new_data is None and s.call("POST", "/api/activations", {"new_index": 0})[0] == 400
    s.call("POST", "/api/predict", {"example_new": True})
    s.call("POST", "/api/data/load", {"example": "ecg"})
    assert s.new_data is None


def test_regression_table_thumbs_and_rows():
    from blocklearning import predict
    from blocklearning.data import load_example
    d = load_example("concrete")
    new = predict.read_new(d, {"example_new": True})
    out = np.arange(len(new["names"]), dtype=np.float64)[:, None]
    r = predict.table(d, new["names"], out, new["labels"], new["X"])
    x = r["rows"][0]
    assert x["index"] == 0 and "probs" not in x and np.allclose(x["input"], d.raw(new["X"][0]), atol=1e-4)
    d = load_example("digits")                                       # 그림: 원래 값 그대로 (32×32는 64 이하)
    new = predict.read_new(d, {"example_new": True})
    r = predict.table(d, new["names"], np.zeros((len(new["names"]), len(d.classes))), new["labels"], new["X"])
    assert r["rows"][0]["input_hw"] == list(d.image) and len(r["rows"][0]["input"]) == d.image[0] * d.image[1]


def test_thumb_shrinks_big_pictures():
    from types import SimpleNamespace
    from blocklearning.predict import PIC_ROWS, THUMB_PX, thumb
    d = SimpleNamespace(image=(150, 100), length=150 * 100, raw=lambda x: x)
    t = thumb(d, np.arange(150 * 100, dtype=np.float64))
    assert max(t["input_hw"]) <= THUMB_PX and len(t["input"]) == t["input_hw"][0] * t["input_hw"][1]
    assert t["input"][1] == 3.0                                     # 세 칸마다 하나 (건너뛰어 뽑기)
    assert PIC_ROWS == 60


@pytest.mark.parametrize("kind", ["bearing_ae", "hydraulic_multi", "pneumatic_seg2", "steel_power", "pcb_det"])
def test_other_tasks_rows_have_index(kind):
    from blocklearning import predict
    from blocklearning.data import load_example
    d = load_example(kind)
    new = predict.read_new(d, {"example_new": True})
    out = np.zeros((len(new["names"]), int(d.n_out)), dtype=np.float32)
    r = predict.table(d, new["names"], out, new["labels"], new["X"], {"threshold": 1.0}, new.get("truth"))
    assert [x["index"] for x in r["rows"]] == list(range(len(r["rows"]))) and r["rows"]
    s = Studio()
    s.data, s.new_data = d, new
    p = s.picked_sample({"new_index": 0})
    assert p["name"] == new["names"][0] and np.allclose(p["x"], new["X"][0])
    assert p["label"] == predict.new_label(d, new, 0) and p["label"] is not None


def test_web_sample_new_matches_installed():
    """웹(sample_new)이 넘기는 새 데이터 값·정답·이름이 설치형(picked_sample)과 같음, 직접 넣은 값도."""
    from blocklearning.graph import build_graph_model
    s = web.WebStudio()
    s.call("POST", "/api/data/load", {"example": "ecg"})
    d = s.data
    assert json.loads(web.sample_new(s, 0, "null"))["status"] == 400            # 학습 전
    net = build_graph_model(G, d.length, d.n_out, seed=1).eval()
    web.set_trained(s, json.dumps(G), "{}", json.dumps({"metric_name": "accuracy"}))
    assert json.loads(web.sample_new(s, 0, "null"))["status"] == 400            # 추론 전
    assert json.loads(web.predict_prepare(s, json.dumps({"example_new": True})))["status"] == 200
    X = np.frombuffer(web.predict_x(s), dtype=np.float32).reshape(-1, d.length)
    with torch.no_grad():
        out = net(torch.as_tensor(X[:, None, :])).numpy().astype(np.float32)
    r = json.loads(web.predict_finish(s, out.tobytes()))["body"]
    assert r["rows"][2]["index"] == 2 and s.new_data is not None
    w = json.loads(web.sample_new(s, 2, "null"))
    assert w["status"] == 200
    w = w["body"]
    p = s.picked_sample({"new_index": 2})
    assert np.allclose(w["x"], X[2]) and np.allclose(w["x"], p["x"]) and w["label"] == p["label"] and w["name"] == p["name"]
    assert np.allclose(w["input"], d.raw(X[2]), atol=1e-4) and w["order"] and w["classes"] == [str(c) for c in d.classes]
    assert set(w) >= {"x", "input", "label", "order", "classes", "name"}
    v = json.loads(web.sample_new(s, -1, json.dumps(w["input"])))["body"]       # 직접 넣은 값 (원래 단위)
    assert np.allclose(v["x"], X[2], atol=1e-4) and v["label"] is None
    assert json.loads(web.sample_new(s, 99, "null"))["status"] == 400

"""시계열 예측·이상 탐지: 데이터 만들기(시간 순서 나누기·정상만 학습), 지표, 학습(PyTorch·웹 엔진이 같은 값), 결과 화면 값."""
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from dlstudio import anomaly, forecast, tasks
from dlstudio.data import DataError, load_example
from dlstudio.graph import check_graph, from_blocks

JS = Path(__file__).parent / "js"
ENGINE = Path(__file__).parents[1] / "src" / "dlstudio" / "static" / "engine.js"
HAS_NODE = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()
T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}


def _sub(sid):
    return next(s for f in tasks.CATALOG for s in f["subtasks"] if s["id"] == sid)


# ---------------------------------------------------------------- 시계열 예측

def test_windows_and_time_split_do_not_leak():
    v = np.arange(1000, dtype=float)
    d = forecast.make_forecast_dataset(v, 20, 5, stride=3, normalize=False)
    cut = d.extra["cut"]
    assert d.task == "forecast" and d.n_out == 5 and d.length == 20
    assert d.y_train.max() < cut <= d.y_test.min()                          # 맞힐 값: 학습은 나눈 지점 앞, 시험은 뒤
    assert np.allclose(d.y_train[:, 0], d.X_train[:, -1] + 1)               # 창: 지난 20개 → 바로 다음 5개
    assert np.allclose(np.diff(d.X_train[:, 0]), 3)                         # 간격 3
    with pytest.raises(DataError):
        forecast.make_forecast_dataset(np.arange(40.0), 30, 10)


def test_forecast_example_and_summary():
    d = load_example("factory_power")
    s = d.summary()
    json.dumps(s, ensure_ascii=False)
    assert s["task"] == "forecast" and s["forecast"]["horizon"] == 24 and len(s["preview"][0]["future"]) == 24
    assert d.label_text("test", 0).startswith("다음 24개")
    assert np.allclose(d.std, d.std[0]) and abs(d.X_train.mean()) < 0.2     # 정규화: 신호 전체에 하나
    assert np.allclose(forecast.naive(d, d.X_test), d.raw(d.X_test)[:, -24:])   # 단순 예측 = 지난 24개 되풀이
    r = forecast.result(d.y_test + 1.0, d)
    assert abs(r["mae"] - 1) < 1e-9 and len(r["mae_by_step"]) == 24 and len(r["fc_samples"]) == 6
    assert forecast.metric(d.y_test, d.y_test) == 1.0


def test_forecast_csv_loader_and_reload(tmp_path):
    from dlstudio.core import Studio
    rows = ["시각,온도,전력"] + [f"2026-01-01 {i},{20 + np.sin(i / 5):.4f},{100 + 10 * np.sin(i / 24 * 2 * np.pi):.4f}"
                               for i in range(600)]
    rows[50] = "2026-01-01 49,,"                                            # 빈 칸은 앞뒤 값으로 메움
    f = tmp_path / "log.csv"
    f.write_text("\n".join(rows), encoding="utf-8")
    s = Studio()
    r = s.api_data_load({"path": str(f), "task": "forecast"})
    fc = r["summary"]["forecast"]
    assert fc["column"] == "전력" and fc["columns"] == ["온도", "전력"] and fc["n_points"] == 600
    assert r["summary"]["length"] == forecast.defaults(600)[0]
    r = s.api_data_load({"forecast_reload": {"column": "온도", "L": 30, "H": 6}})
    assert r["summary"]["forecast"]["column"] == "온도" and r["summary"]["length"] == 30 and s.data.n_out == 6
    assert s.load_source(s.data.source)["forecast"]["horizon"] == 6        # 프로젝트를 다시 열 때도 같은 설정
    r = s.api_data_load({"example": "factory_power"})
    r = s.api_data_load({"forecast_reload": {"L": 48, "H": 12}})            # 예시도 길이만 바꿔 다시
    assert r["summary"]["length"] == 48 and r["summary"]["forecast"]["horizon"] == 12 and r["summary"]["example"] == "factory_power"


# ---------------------------------------------------------------- 이상 탐지

def test_to_anomaly_trains_on_normals_only():
    base = load_example("vibration")
    d = load_example("vibration_ae")
    assert d.task == "anomaly" and d.classes == ["정상", "이상"] and d.extra["others"] == ["결함"]
    assert (d.y_train == 0).all() and len(d.X_train) == int((base.y_train == 0).sum())
    assert int(d.y_test.sum()) == int((base.y_train == 1).sum() + (base.y_test == 1).sum())   # 이상은 모두 시험으로
    assert d.n_out == d.length == 200 and np.allclose(d.X_train.mean(0), 0, atol=1e-9)        # 정규화는 학습 정상으로
    assert np.allclose(d.target_train, d.X_train)
    s = d.summary()
    json.dumps(s, ensure_ascii=False)
    assert s["anomaly"]["n_anomaly"] == int(d.y_test.sum()) and s["class_counts"]["이상"] == int(d.y_test.sum())
    assert any(p["label"] == "이상" for p in s["preview"])
    assert load_example("process_ae").extra["others"] == ["전류 이상", "진동 이상", "압력 저하"]


def test_pick_normal_and_single_class_csv(tmp_path):
    assert anomaly.pick_normal(["결함", "정상"], np.array([0, 0, 0, 1])) == 1
    assert anomaly.pick_normal(["A", "B"], np.array([0, 1, 1])) == 1
    assert anomaly.pick_normal(["A", "B"], np.array([0, 1, 1]), normal="A") == 0
    from dlstudio.core import Studio
    rng = np.random.default_rng(0)
    f = tmp_path / "ok.csv"
    f.write_text("\n".join(["a,b,c,상태"] + [f"{x:.3f},{y:.3f},{x + y:.3f},양품" for x, y in rng.normal(size=(60, 2))]),
                 encoding="utf-8")
    s = Studio()
    r = s.api_data_load({"path": str(f), "task": "anomaly"})
    assert r["summary"]["task"] == "anomaly" and s.data.extra["normal"] == "양품" and s.data.y_test.sum() == 0
    assert anomaly.metric_name(s.data) == "recon"                          # 정상만: AUC 대신 복원 오차
    assert s.load_source(s.data.source)["anomaly"]["normal"] == "양품"


def test_auc_and_threshold():
    assert anomaly.auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert anomaly.auc([0.5, 0.5], [0, 1]) == 0.5                          # 같은 점수는 반
    assert anomaly.auc([0.3, 0.1, 0.2], [1, 0, 0]) == 1.0
    assert np.isnan(anomaly.auc([0.1, 0.2], [0, 0]))
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)                              # 같은 점수가 여럿
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert abs(anomaly.auc(s, y) - brute) < 1e-12
    at = anomaly.at_threshold(np.array([0.1, 0.9, 0.95, 0.2]), np.array([0, 1, 0, 1]), 0.5)
    assert (at["tp"], at["fp"], at["fn"], at["tn"]) == (1, 1, 1, 1)


# ---------------------------------------------------------------- 과제 목록·학습 (PyTorch)

@pytest.mark.parametrize("sid", ["fc_ts", "ae_ts", "ae_tab"])
def test_subtasks_available_and_templates_check(sid):
    sub = _sub(sid)
    assert sub["available"] and sub["task"] in ("forecast", "anomaly")
    d = load_example(sub["example_data"][0][0])
    assert d.task == sub["task"]
    for t in sub["templates"]:
        c = check_graph(from_blocks(t["blocks"]), d.length, d.n_out, image=d.image)
        assert c.ok, (t["id"], [i.message for i in c.issues])


@pytest.mark.parametrize("kind,tid,ep,need", [("factory_power", "fc_mlp", 12, 0.6), ("vibration_ae", "ae_dense", 25, 0.75),
                                              ("process_ae", "ae_tab_dense", 40, 0.75)])
def test_train_job_learns_and_result_is_json(kind, tid, ep, need):
    from dlstudio.jobs import Job, TrainSettings, train_job
    d = load_example(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = train_job(Job("train"), from_blocks(T[tid]["blocks"]), d, TrainSettings(epochs=ep, lr=1e-3, batch_size=32))
    json.dumps({k: v for k, v in r.items() if k != "model"})
    assert r["test_metric"] > need, r["test_metric"]
    if d.task == "forecast":
        assert r["metric_name"] == "r2" and r["mae"] < r["naive_mae"] and r["naive_name"].startswith("지난 24개")
    else:
        assert r["metric_name"] == "auc" and r["auc"] == r["test_metric"] and r["train_metric"] is None
        assert len(r["scores"]) == len(d.X_test) and r["threshold"] > 0 and len(r["ae_samples"]) >= 6
        assert np.mean(np.asarray(r["train_scores"]) <= r["threshold"]) >= 0.95   # 기준값: 학습 정상의 95%


# ---------------------------------------------------------------- 웹 엔진 (TF.js): 같은 지표·학습

def _node(script):
    code = (f"globalThis.tf = require('@tensorflow/tfjs');\nconst E = require({json.dumps(str(ENGINE))});\n{script}")
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=str(JS))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
def test_engine_metrics_match_python():
    rng = np.random.default_rng(3)
    X, out = rng.normal(size=(50, 7)), rng.normal(size=(50, 7))
    y = (rng.random(50) < 0.3).astype(int)
    out[:5] = X[:5]                                                         # 점수 0이 여럿 (같은 점수)
    Yf, Of = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
    arr = lambda a: f"Float32Array.from({json.dumps(np.ravel(a).tolist())})"
    got = _node(f"""const X = {arr(X)}, o = {arr(out)}, y = {arr(y)};
      process.stdout.write(JSON.stringify([Array.from(E.aeScores(o, X, 50)), E.aeMetric(o, X, y),
        E.aeMetric(o, X, new Float32Array(50)), E.fcMetric({arr(Of)}, {arr(Yf)})]));""")
    X32, o32 = X.astype(np.float32).astype(float), out.astype(np.float32).astype(float)
    assert np.allclose(got[0], anomaly.scores(o32, X32), atol=1e-9)
    assert abs(got[1] - anomaly.metric(o32, X32, y)) < 1e-9
    assert abs(got[2] - anomaly.scores(o32, X32).mean()) < 1e-9
    assert abs(got[3] - forecast.metric(Of.astype(np.float32), Yf.astype(np.float32))) < 1e-5


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
@pytest.mark.parametrize("kind,tid", [("factory_power", "fc_linear"), ("process_ae", "ae_tab_dense")])
def test_engine_trains_and_python_builds_same_result(tmp_path, kind, tid):
    """웹 엔진 학습 → 출력(원래 단위로 접힘)을 Python task_result에 넘김. 엔진 가중치를 PyTorch에 넣으면 같은 출력·성적."""
    import torch
    from dlstudio.blocks import to_model_input
    from dlstudio.graph import build_graph_model
    from dlstudio.jobs import task_result
    from tests.test_engine import _train_js
    d = load_example(kind)
    g = from_blocks(T[tid]["blocks"])
    o = _train_js(tmp_path, g, d, {"epochs": 4, "lr": 0.003, "batch_size": 32, "seed": 0})
    r = o["result"]
    assert r["metric_name"] == ("r2" if kind == "factory_power" else "auc") and len(o["out"]) == len(d.X_test) * d.n_out
    net = build_graph_model(g, d.length, d.n_out, image=d.image).eval()
    net.load_state_dict({k: torch.tensor(v["data"]).reshape(v["shape"]).to(net.state_dict()[k].dtype)
                         for k, v in o["state"].items()})
    with torch.no_grad():
        out = net(torch.as_tensor(to_model_input(net.kind, d.X_test), dtype=torch.float32)).numpy()
    assert np.allclose(out.ravel(), o["out"], rtol=1e-4, atol=1e-3 * max(1.0, float(np.abs(out).max())))
    full = task_result(d, np.asarray(o["out"]).reshape(len(d.X_test), -1), np.asarray(o["outTr"]).reshape(len(d.X_train), -1))
    json.dumps(full)
    if kind == "factory_power":
        assert abs(forecast.metric(out, d.y_test) - r["test_metric"]) < 1e-3 and full["mae"] > 0
    else:
        assert abs(anomaly.auc(anomaly.scores(out, d.X_test), d.y_test) - r["test_metric"]) < 1e-3 and full["auc"] > 0.5

"""웹 버전의 Python 쪽(web.py)을 CPython에서: torch 없이 돌고, 가중치·프로젝트 파일이 PyTorch와 맞물리는지."""
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from dlstudio import codegen, web
from dlstudio.blocks import to_model_input
from dlstudio.graph import build_graph_model


def _state_js(net) -> dict:
    """엔진(net.state())이 다리를 거쳐 넘기는 모양: {이름: {shape, data(바이트), int}}."""
    out = {}
    for k, v in net.state_dict().items():
        a = v.detach().numpy()
        is_int = np.issubdtype(a.dtype, np.integer)
        out[k] = {"shape": list(a.shape), "data": a.astype(np.int32 if is_int else np.float32).tobytes(), "int": is_int}
    return out


def _studio_with_example():
    s = web.WebStudio()
    code, body = s.call("POST", "/api/data/load", {"example": "ecg"})
    assert code == 200
    return s


def _graph():
    from dlstudio.tasks import DEFAULT_TEMPLATES, _view
    return _view(DEFAULT_TEMPLATES[0])["graph"]


def test_web_python_runs_without_torch():
    """Pyodide에는 torch가 없음: 웹 쪽 흐름(데이터·검사·계획·코드·샘플·가중치·프로젝트)이 torch 없이 돈다."""
    code = r'''
import sys, json
sys.modules["torch"] = None
from dlstudio import web
from dlstudio.tasks import DEFAULT_TEMPLATES, _view
s = web.WebStudio()
assert s.call("POST", "/api/data/load", {"example": "ecg"})[0] == 200
g = _view(DEFAULT_TEMPLATES[0])["graph"]
c, r = s.call("POST", "/api/model/check", {"graph": g})
assert c == 200 and r["ok"] and r["panel"]["params"] > 0, r
plan = json.loads(web.plan(s, json.dumps(g)))["body"]
p = web.data_payload(s)
assert len(p["X_train"]) == 4 * 800 * 140
web.set_trained(s, json.dumps(g), "{}", "{}")
assert json.loads(web.sample(s, 0))["status"] == 200
c, r = s.call("POST", "/api/code", {"graph": g, "framework": "pytorch", "weights_file": "w.npz"})
assert c == 200 and r["trained"] and 'np.load("w.npz")' in r["code"], r
state = {}
for st in plan["steps"]:
    for name, shape, kind in st["params"]:
        n = 1
        for v in shape: n *= v
        import numpy as np
        state[name] = {"shape": shape, "data": np.zeros(n, np.int32 if kind == "count" else np.float32).tobytes(), "int": kind == "count"}
assert len(web.weights_npz(s, json.dumps(g), state)) > 100
assert len(web.project_bytes(s, state)) > 100
print("ok")
'''
    import os
    from pathlib import Path
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=120, env=env)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]


def test_weights_npz_loads_into_generated_pytorch_code():
    s = _studio_with_example()
    d, g = s.data, _graph()
    net = build_graph_model(g, d.length, d.n_out, seed=3).eval()
    raw = web.weights_npz(s, json.dumps(g), _state_js(net))
    web.set_trained(s, json.dumps(g), "{}", "{}")
    code = s.call("POST", "/api/code", {"graph": g, "framework": "pytorch", "weights_file": "w.npz"})[1]["code"]
    ns: dict = {}
    exec(code, ns)
    m = ns["StudioModel"]().eval()
    with np.load(io.BytesIO(raw)) as z:
        m.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files})
    x = torch.as_tensor(to_model_input(net.kind, d.X_test[:8]), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(m(x), net(x), atol=1e-6)


def test_project_bytes_roundtrip_with_weights(tmp_path):
    s = _studio_with_example()
    d, g = s.data, _graph()
    net = build_graph_model(g, d.length, d.n_out, seed=4).eval()
    s.call("POST", "/api/model/check", {"graph": g})
    web.set_trained(s, json.dumps(g), json.dumps({"epochs": 3}), json.dumps({"test_metric": 0.5, "metric_name": "accuracy"}))
    raw = web.project_bytes(s, _state_js(net))
    assert set(zipfile.ZipFile(io.BytesIO(raw)).namelist()) == {"project.json", "model.npz"}
    p = tmp_path / "p.dlstudio"
    p.write_bytes(raw)
    s2 = web.WebStudio()                                     # 새로 연 페이지: 예시 데이터는 다시 불러옴
    r = json.loads(web.open_project(s2, str(p)))
    assert r["status"] == 200, r
    b = r["body"]
    assert b["response"]["trained"] and b["response"]["train_settings"] == {"epochs": 3} and b["response"]["summary"]["n_train"] + b["response"]["summary"]["n_test"] == 1000
    names = {p[0] for st in b["plan"]["steps"] for p in st["params"]}
    assert set(b["state"]) == names
    w = b["state"]["mods.m_b0.weight"]
    assert np.allclose(np.array(w["data"]).reshape(w["shape"]), net.state_dict()["mods.m_b0.weight"].numpy())


def test_open_project_needs_data_when_not_example(tmp_path):
    from dlstudio.project import save_project
    p = tmp_path / "q.dlstudio"
    save_project(str(p), _graph(), {"path": "/data/u1/x.csv"}, {}, None, {})
    r = json.loads(web.open_project(web.WebStudio(), str(p)))
    assert r["status"] == 400 and "데이터" in r["body"]["error"]["message"]
    bad = tmp_path / "bad.dlstudio"
    bad.write_bytes(b"not a zip")
    assert json.loads(web.open_project(web.WebStudio(), str(bad)))["status"] == 400


def test_plan_reports_graph_problem():
    s = _studio_with_example()
    r = json.loads(web.plan(s, json.dumps({"version": 2, "nodes": [{"id": "input", "type": "input"}]})))
    assert r["status"] == 400 and r["body"]["error"]["message"]

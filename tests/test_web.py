"""웹 버전의 Python 쪽(web.py)을 CPython에서: torch 없이 돌고, 가중치·프로젝트 파일이 PyTorch와 맞물리는지."""
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

from blocklearning import codegen, web
from blocklearning.blocks import to_model_input
from blocklearning.graph import build_graph_model


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
    from blocklearning.tasks import DEFAULT_TEMPLATES, _view
    return _view(DEFAULT_TEMPLATES[0])["graph"]


def test_web_python_runs_without_torch():
    """Pyodide에는 torch가 없음: 웹 쪽 흐름(데이터·검사·계획·코드·샘플·가중치·프로젝트)이 torch 없이 돈다."""
    code = r'''
import sys, json
sys.modules["torch"] = None
from blocklearning import web
from blocklearning.tasks import DEFAULT_TEMPLATES, _view
s = web.WebStudio()
assert s.call("POST", "/api/data/load", {"example": "ecg"})[0] == 200
g = _view(DEFAULT_TEMPLATES[0])["graph"]
c, r = s.call("POST", "/api/model/check", {"graph": g})
assert c == 200 and r["ok"] and r["panel"]["params"] > 0, r
plan = json.loads(web.plan(s, json.dumps(g)))["body"]
p = web.data_payload(s)
assert len(p["X_train"]) == 4 * 3992 * 140
web.set_trained(s, json.dumps(g), "{}", "{}")
assert json.loads(web.sample(s, 0))["status"] == 200
assert json.loads(web.predict_prepare(s, json.dumps({"example_new": True})))["status"] == 200     # ⑤ 추론 → 새 데이터 한 샘플
import numpy as np
n_new = len(np.frombuffer(web.predict_x(s), dtype=np.float32)) // 140
assert json.loads(web.predict_finish(s, np.zeros((n_new, len(s.data.classes)), np.float32).tobytes()))["status"] == 200
w = json.loads(web.sample_new(s, 0, "null"))
assert w["status"] == 200 and len(w["body"]["x"]) == 140 and w["body"]["name"], w
assert json.loads(web.sample_new(s, -1, json.dumps(w["body"]["input"])))["status"] == 200
c, r = s.call("POST", "/api/code", {"graph": g, "framework": "pytorch", "weights_file": "w.npz"})
assert c == 200 and r["trained"] and 'load_trained(path="w.npz")' in r["code"], r
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
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONIOENCODING="utf-8")
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
        assert np.allclose(z["_mean"], d.mean) and np.allclose(z["_std"], d.std) and list(z["_classes"]) == [str(c) for c in d.classes]
        m.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if not k.startswith("_")})
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
    p = tmp_path / "p.blocklearning"
    p.write_bytes(raw)
    s2 = web.WebStudio()                                     # 새로 연 페이지: 예시 데이터는 다시 불러옴
    r = json.loads(web.open_project(s2, str(p)))
    assert r["status"] == 200, r
    b = r["body"]
    assert b["response"]["trained"] and b["response"]["train_settings"] == {"epochs": 3} and b["response"]["summary"]["n_train"] + b["response"]["summary"]["n_test"] == 4990
    names = {p[0] for st in b["plan"]["steps"] for p in st["params"]}
    assert set(b["state"]) == names
    w = b["state"]["mods.m_b0.weight"]
    assert np.allclose(np.array(w["data"]).reshape(w["shape"]), net.state_dict()["mods.m_b0.weight"].numpy())


def test_open_project_needs_data_when_not_example(tmp_path):
    from blocklearning.project import save_project
    p = tmp_path / "q.blocklearning"
    save_project(str(p), _graph(), {"path": "/data/u1/x.csv"}, {}, None, {})
    r = json.loads(web.open_project(web.WebStudio(), str(p)))
    assert r["status"] == 400 and "데이터" in r["body"]["error"]["message"]
    bad = tmp_path / "bad.blocklearning"
    bad.write_bytes(b"not a zip")
    assert json.loads(web.open_project(web.WebStudio(), str(bad)))["status"] == 400


def test_plan_reports_graph_problem():
    s = _studio_with_example()
    r = json.loads(web.plan(s, json.dumps({"version": 2, "nodes": [{"id": "input", "type": "input"}]})))
    assert r["status"] == 400 and r["body"]["error"]["message"]


# ---------------------------------------------------------------- 프로젝트 기록 (브라우저 보관함·구글 드라이브)

@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = str(tmp_path / "data").replace("\\", "/")
    monkeypatch.setattr(web, "DATA_DIR", d)
    return d


def _csv_studio(data_dir):
    import os
    os.makedirs(f"{data_dir}/u7", exist_ok=True)
    rng = np.random.default_rng(0)
    rows = ["a,b,c,d,y"] + [",".join(f"{v:.3f}" for v in rng.normal(size=4)) + f",{'정상' if i % 2 else '이상'}" for i in range(40)]
    with open(f"{data_dir}/u7/내데이터.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    s = web.WebStudio()
    code, body = s.call("POST", "/api/data/load", {"path": f"{data_dir}/u7/내데이터.csv"})
    assert code == 200, body
    return s


def _dense_graph():
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}, {"id": "d", "type": "dense", "units": 5, "inputs": ["input"]},
                                    {"id": "a", "type": "act", "kind": "relu", "inputs": ["d"]}, {"id": "o", "type": "output", "inputs": ["a"]}]}


@pytest.mark.parametrize("together", [False, True])
def test_record_roundtrip_with_data_copy(tmp_path, data_dir, together):
    """CSV를 고른 프로젝트: 프로젝트 파일 + 데이터 사본으로 저장 → 새 페이지에서 열면 데이터째 돌아옴 (경로는 새 자리로)."""
    s = _csv_studio(data_dir)
    g = _dense_graph()
    s.call("POST", "/api/model/check", {"graph": g})
    net = build_graph_model(g, s.data.length, s.data.n_out, seed=2)
    web.set_trained(s, json.dumps(g), json.dumps({"epochs": 4}), json.dumps({"test_metric": 0.6}))
    proj = web.record_project(s, _state_js(net), "내 프로젝트", with_data=together)
    meta = json.loads(zipfile.ZipFile(io.BytesIO(proj)).read("project.json"))
    assert meta["name"] == "내 프로젝트" and meta["data_source"]["path"] == "@data/u7/내데이터.csv"
    pp = tmp_path / "p.blocklearning"
    pp.write_bytes(proj)
    dp = ""
    if not together:
        raw = web.data_zip(s)
        assert zipfile.ZipFile(io.BytesIO(raw)).namelist() == ["u7/내데이터.csv"]
        (tmp_path / "data.zip").write_bytes(raw)
        dp = str(tmp_path / "data.zip")
    s2 = web.WebStudio()
    r = json.loads(web.record_open(s2, str(pp), dp))
    assert r["status"] == 200, r
    b = r["body"]
    assert b["name"] == "내 프로젝트" and b["response"]["trained"] and b["response"]["summary"]["n_train"] + b["response"]["summary"]["n_test"] == 40
    assert s2.data.source["path"].startswith(data_dir + "/r") and s2.data.source["path"].endswith("u7/내데이터.csv")
    assert np.allclose(s2.data.X_train, s.data.X_train)
    w = b["state"]["mods.m_d.1.weight"]
    assert np.allclose(np.array(w["data"]).reshape(w["shape"]), net.state_dict()["mods.m_d.1.weight"].numpy())


def test_record_example_project_needs_no_data(tmp_path, data_dir):
    s = _studio_with_example()
    g = _graph()
    s.call("POST", "/api/model/check", {"graph": g})
    assert web.data_zip(s) == b""
    pp = tmp_path / "e.blocklearning"
    pp.write_bytes(web.record_project(s, None, "예시"))
    r = json.loads(web.record_open(web.WebStudio(), str(pp), ""))
    assert r["status"] == 200 and r["body"]["state"] is None and r["body"]["response"]["graph"] == g


def test_record_missing_data_copy_is_explained(tmp_path, data_dir):
    s = _csv_studio(data_dir)
    s.call("POST", "/api/model/check", {"graph": _dense_graph()})
    pp = tmp_path / "p.blocklearning"
    pp.write_bytes(web.record_project(s, None))
    r = json.loads(web.record_open(web.WebStudio(), str(pp), ""))
    assert r["status"] == 400 and "데이터 사본" in r["body"]["error"]["message"]


def test_record_open_ignores_paths_outside(tmp_path, data_dir):
    """데이터 사본 zip 안의 ../ 경로는 풀지 않음 (다른 곳에 파일을 쓰지 못하게)."""
    s = _csv_studio(data_dir)
    s.call("POST", "/api/model/check", {"graph": _dense_graph()})
    pp = tmp_path / "p.blocklearning"
    pp.write_bytes(web.record_project(s, None))
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("u7/내데이터.csv", open(f"{data_dir}/u7/내데이터.csv", "rb").read())
        z.writestr("../../escape.txt", b"x")
    (tmp_path / "evil.zip").write_bytes(evil.getvalue())
    r = json.loads(web.record_open(web.WebStudio(), str(pp), str(tmp_path / "evil.zip")))
    assert r["status"] == 200
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path.parent / "escape.txt").exists()

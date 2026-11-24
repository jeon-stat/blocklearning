"""DL studio HTTP API: 실제 서버를 스레드로 띄워 데이터 → 모델 검사 → 학습 → 그림 → 원격(가짜 실행기) 순서로 호출."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from dlstudio.graph import from_blocks

from dlstudio import cli
from dlstudio.server import make_server

CNN = [{"type": "conv", "channels": 8, "kernel": 5}, {"type": "act", "kind": "relu"}, {"type": "global_avg"},
       {"type": "output"}]


@pytest.fixture
def server(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>DL studio</title>", encoding="utf-8")
    (static / "app.js").write_text("console.log(1)", encoding="utf-8")
    s = make_server(0, static_dir=str(static))
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield s
    s.shutdown()
    s.server_close()


def _url(s, path):
    return f"http://127.0.0.1:{s.server_address[1]}{path}"


def _call(s, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(_url(s, path), data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait(s, jid, timeout=120):
    t = time.time()
    while True:
        code, st = _call(s, f"/api/jobs/{jid}")
        if st["state"] != "running":
            return st
        assert time.time() - t < timeout
        time.sleep(0.1)


def test_vibration_example_loads(server):
    code, d = _call(server, "/api/data/load", {"example": "vibration"})
    assert code == 200 and d["summary"]["classes"] == ["정상", "결함"]


def test_flow_example_to_trained(server, tmp_path):
    code, d = _call(server, "/api/data/load", {"example": True})
    assert code == 200 and d["summary"]["length"] == 140
    code, c = _call(server, "/api/model/check", {"blocks": CNN})
    assert code == 200 and c["ok"] and c["panel"]["ok"], c
    assert c["shapes"]["b0"] == [8, 140] and c["graph"] == from_blocks(CNN)
    code, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 2, "lr": 0.01}})
    st = _wait(server, j["id"])
    assert st["state"] == "done", st
    assert "model" not in st["result"] and st["result"]["test_metric"] > 0
    code, a = _call(server, "/api/activations", {"sample_index": 0})
    assert code == 200 and "b0" in a["values"] and a["label"] in ("정상", "이상") and a["order"][:2] == ["input", "b0"]
    code, _ = _call(server, "/api/project/save", {"path": str(tmp_path / "p.dlstudio")})
    assert code == 200
    code, o = _call(server, "/api/project/open", {"path": str(tmp_path / "p.dlstudio")})
    assert code == 200 and o["graph"] == from_blocks(CNN) and o["trained"] is True
    code, x = _call(server, "/api/project/export", {"path": str(tmp_path / "m.pt")})
    assert code == 200 and (tmp_path / "m.pt").exists() and x["info"]["task"] == "classification"


def test_bad_input_returns_400(server):
    code, e = _call(server, "/api/model/check", {"blocks": CNN})          # 데이터 먼저
    assert code == 400 and e["error"]["message"]
    _call(server, "/api/data/load", {"example": True})
    code, e = _call(server, "/api/train", {"blocks": [{"type": "nope"}]})
    assert code == 400 and e["error"]["message"]
    code, e = _call(server, "/api/data/load", {"path": "없는파일.csv"})
    assert code == 400


def test_data_errors_have_locations(server, tmp_path):
    p = tmp_path / "bad.csv"
    rows = ["1,2,3,%d" % (i % 2) for i in range(12)]
    rows[2] = "1,2,x,0"                                          # 첫 줄이면 머리글로 보므로 3행에
    p.write_text("\n".join(rows), encoding="utf-8")
    code, e = _call(server, "/api/data/load", {"path": str(p)})
    assert code == 400 and e["error"]["issues"][0]["row"] == 3 and e["error"]["issues"][0]["col"] == 3


def test_static_index_and_traversal(server):
    with urllib.request.urlopen(_url(server, "/")) as r:
        assert r.status == 200 and b"DL studio" in r.read()
    with urllib.request.urlopen(_url(server, "/static/app.js")) as r:
        assert r.status == 200
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    c.request("GET", "/static/../../server.py")
    assert c.getresponse().status == 404


def test_binds_localhost_only(server):
    assert server.server_address[0] == "127.0.0.1"


def test_cli_parses_port():
    assert cli.main(["--port", "0", "--no-browser", "--dry-run"]) == 0


def test_state_restores_after_reload(server):
    code, s = _call(server, "/api/state")
    assert code == 200 and s["summary"] is None and s["trained"] is False
    _call(server, "/api/data/load", {"example": "ecg"})
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 1}})
    _wait(server, j["id"])
    code, s = _call(server, "/api/state")
    assert s["summary"]["length"] == 140 and s["trained"] is True and s["graph"] == from_blocks(CNN)
    assert s["results"]["metric_name"] == "accuracy"


def _raw(server, path, body=b"{}", headers=None):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    c.request("POST", path, body=body, headers=headers or {})
    r = c.getresponse()
    return r.status, r.read()


def test_rejects_cross_site_requests(server):
    """다른 웹사이트가 로컬 API를 부르지 못하게: JSON만, Host는 127.0.0.1·localhost, Origin은 같은 출처."""
    port = server.server_address[1]
    body = json.dumps({"path": "x.dlstudio"}).encode()
    assert _raw(server, "/api/project/save", body, {"Content-Type": "text/plain"})[0] == 415
    assert _raw(server, "/api/project/save", body, {"Content-Type": "application/json", "Host": "evil.com"})[0] == 403
    assert _raw(server, "/api/project/save", body, {"Content-Type": "application/json",
                                                     "Origin": "http://evil.com"})[0] == 403
    ok = _raw(server, "/api/data/load", json.dumps({"example": "ecg"}).encode(),
              {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
    assert ok[0] == 200


def test_quoted_windows_path_accepted(server, tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("\n".join("1,2,3,%d" % (i % 2) for i in range(12)), encoding="utf-8")
    code, _ = _call(server, "/api/data/load", {"path": f'"{p}"'})
    assert code == 200


def test_loading_data_cancels_training_and_drops_its_model(server):
    _call(server, "/api/data/load", {"example": "ecg"})
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 200}})
    time.sleep(1.0)
    _call(server, "/api/data/load", {"example": "vibration"})
    st = _wait(server, j["id"])
    assert st["state"] == "cancelled"
    assert _call(server, "/api/state")[1]["trained"] is False


def test_json_never_contains_nan():
    from dlstudio.server import _clean
    out = json.dumps(_clean({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2}}), allow_nan=False)
    assert json.loads(out) == {"a": None, "b": [1.0, None], "c": {"d": 2}}


def test_state_returns_latest_edited_blocks_not_trained_ones(server):
    """새로 고침하면 마지막으로 편집한 블록이 떠야 함 (학습한 블록은 따로 알려서 '다시 학습 필요' 표시)."""
    _call(server, "/api/data/load", {"example": "ecg"})
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 1}})
    _wait(server, j["id"])
    edited = [{"type": "conv", "channels": 4, "kernel": 3}, {"type": "global_avg"}, {"type": "output"}]
    _call(server, "/api/model/check", {"blocks": edited})
    s = _call(server, "/api/state")[1]
    assert s["graph"] == from_blocks(edited) and s["trained_graph"] == from_blocks(CNN) and s["trained"] is True


def test_tasks_catalog_and_selection(server):
    code, cat = _call(server, "/api/tasks")
    assert code == 200 and cat["families"][0]["id"] == "classification"
    t = cat["families"][0]["subtasks"][0]["templates"][0]
    assert t["graph"]["version"] == 2 and "blocks" not in t
    assert [x["id"] for x in cat["defaults"]] == ["cnn", "lstm", "inception", "mlp"] and all("graph" in x for x in cat["defaults"])
    sel = {"family": "regression", "subtask": "reg_tab", "template": "mlp"}
    code, _ = _call(server, "/api/task/select", sel)
    assert code == 200 and _call(server, "/api/state")[1]["selection"] == sel
    code, d = _call(server, "/api/data/load", {"example": "process_reg"})
    assert code == 200 and d["summary"]["task"] == "regression" and d["summary"]["features"][0] == "온도"
    code, e = _call(server, "/api/task/select", {"family": "x", "subtask": "y", "template": "z"})
    assert code == 400


def test_image_flow(server, tmp_path):
    from dlstudio.tasks import find
    code, d = _call(server, "/api/data/load", {"example": "digits"})
    assert code == 200 and d["summary"]["image"] == [8, 8]
    blocks = find("classification", "cls_img", "cnn2d")[2]["blocks"]
    code, c = _call(server, "/api/model/check", {"blocks": blocks})
    assert code == 200 and c["ok"] and c["shapes"]["b0"] == [8, 8, 8], c
    _, j = _call(server, "/api/train", {"blocks": blocks, "settings": {"epochs": 1}})
    assert _wait(server, j["id"])["state"] == "done"
    code, a = _call(server, "/api/activations", {"sample_index": 0})
    assert code == 200 and len(a["values"]["b0"]) == 8 and len(a["values"]["b0"][0]) == 8     # (C, H, W)
    _call(server, "/api/project/save", {"path": str(tmp_path / "i.dlstudio")})
    code, o = _call(server, "/api/project/open", {"path": str(tmp_path / "i.dlstudio")})
    assert code == 200 and o["summary"]["image"] == [8, 8] and o["trained"]


def test_image_folder_and_csv_load(server, tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    for c in ("ok", "ng"):
        (tmp_path / "img" / c).mkdir(parents=True)
        for i in range(6):
            Image.new("L", (10, 10), color=i * 40).save(tmp_path / "img" / c / f"{i}.png")
    code, d = _call(server, "/api/data/load", {"folder": f'"{tmp_path / "img"}"', "size": 8})
    assert code == 200 and d["summary"]["image"] == [8, 8] and d["summary"]["classes"] == ["ng", "ok"], d
    code, e = _call(server, "/api/data/load", {"folder": str(tmp_path / "없음")})
    assert code == 400
    (tmp_path / "p.csv").write_text("\n".join(",".join(["0"] * 15 + ["255"]) + f",{i % 2}" for i in range(20)), encoding="utf-8")
    code, d = _call(server, "/api/data/load", {"path": str(tmp_path / "p.csv"), "image_width": 0})
    assert code == 200 and d["summary"]["image"] == [4, 4]


def _series_dir(tmp_path, with_labels=False):
    """클래스 폴더 방식(정상·이상 8개씩) 또는 정답 표 방식(맨 위 16개 + labels.csv)."""
    f = tmp_path / ("t" if with_labels else "s")
    for c, sign in (("정상", 1), ("이상", -1)):
        d = f if with_labels else f / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            (d / f"{c}_{i}.csv").write_text("v\n" + "\n".join(str(sign * np.sin(j / 3) + i * 0.01) for j in range(20 + i)),
                                            encoding="utf-8")
    if with_labels:
        (tmp_path / "labels.csv").write_text("file,y\n" + "\n".join(f"{c}_{i},{c}" for c in ("정상", "이상") for i in range(8)),
                                             encoding="utf-8")
    return f


def test_series_folder_scan_load_train_and_project(server, tmp_path):
    f = _series_dir(tmp_path)
    code, s = _call(server, "/api/data/scan_folder", {"path": f'"{f}"'})
    assert code == 200 and s["mode"] == "folders" and s["default_column"] == "v", s
    code, d = _call(server, "/api/data/load", {"series_folder": str(f), "column": "v", "length": 16})
    assert code == 200 and d["summary"]["length"] == 16 and d["summary"]["classes"] == ["이상", "정상"], d
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 1}})
    assert _wait(server, j["id"])["state"] == "done"
    _call(server, "/api/project/save", {"path": str(tmp_path / "s.dlstudio")})
    code, o = _call(server, "/api/project/open", {"path": str(tmp_path / "s.dlstudio")})
    assert code == 200 and o["summary"]["length"] == 16 and o["trained"]
    code, e = _call(server, "/api/data/scan_folder", {"path": str(tmp_path / "없음")})
    assert code == 400


def test_series_project_with_missing_label_file(server, tmp_path):
    f = _series_dir(tmp_path, with_labels=True)
    code, d = _call(server, "/api/data/load", {"series_folder": str(f), "column": "v", "length": 16,
                                               "label_file": str(tmp_path / "labels.csv")})
    assert code == 200 and d["summary"]["classes"] == ["이상", "정상"], d
    _call(server, "/api/model/check", {"blocks": CNN})                    # 저장하려면 블록이 있어야 함
    code, _ = _call(server, "/api/project/save", {"path": str(tmp_path / "t.dlstudio")})
    assert code == 200
    (tmp_path / "labels.csv").unlink()                                     # 정답 표가 사라짐
    code, e = _call(server, "/api/project/open", {"path": str(tmp_path / "t.dlstudio")})
    assert code == 400 and "labels.csv" in e["error"]["message"]
    assert _call(server, "/api/state")[0] == 200                           # 서버는 계속 동작


def test_series_regression_task_survives_project_reopen(server, tmp_path):
    """정수 정답(0–3)을 회귀로 불러와 저장 → 다시 열어도 회귀 (자동 판단으로 분류가 되면 모델이 맞지 않음)."""
    f = tmp_path / "r"
    f.mkdir()
    for i in range(16):
        (f / f"c{i}.csv").write_text("v\n" + "\n".join(str(np.sin(j / 3) * (i % 4)) for j in range(20)), encoding="utf-8")
    (tmp_path / "y.csv").write_text("file,y\n" + "\n".join(f"c{i},{i % 4}" for i in range(16)), encoding="utf-8")
    code, d = _call(server, "/api/data/load", {"series_folder": str(f), "column": "v", "length": 16,
                                               "label_file": str(tmp_path / "y.csv"), "task": "regression"})
    assert code == 200 and d["summary"]["task"] == "regression", d
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 1}})
    assert _wait(server, j["id"])["state"] == "done"
    _call(server, "/api/project/save", {"path": str(tmp_path / "r.dlstudio")})
    code, o = _call(server, "/api/project/open", {"path": str(tmp_path / "r.dlstudio")})
    assert code == 200 and o["summary"]["task"] == "regression" and o["trained"], o


def test_label_api_flow(server, tmp_path):
    code, e = _call(server, "/api/label/state", {})
    assert code == 400                                                     # 작업이 없음
    root = tmp_path / "cycles"
    root.mkdir()
    for i in range(12):
        (root / f"c{i:02d}.csv").write_text("v\n" + "\n".join(str(np.sin(j / 3) * (1 if i < 6 else -1) + i * 0.05)
                                                              for j in range(24)), encoding="utf-8")
    work = tmp_path / "w.dllabel"
    code, s = _call(server, "/api/label/new", {"path": f'"{root}"', "column": "v", "classes": ["정상", "이상"], "file": str(work)})
    assert code == 200 and s["counts"]["total"] == 12, s
    code, it = _call(server, "/api/label/items", {"sort": {"key": "max", "desc": True}, "page": 0, "size": 5})
    assert code == 200 and it["pages"] == 3 and len(it["items"][0]["thumb"]) == 120
    code, d = _call(server, "/api/label/item", {"name": "c00.csv"})
    assert code == 200 and d["n"] == 24
    _call(server, "/api/label/set", {"names": [f"c{i:02d}.csv" for i in range(6)], "label": "정상"})
    code, s = _call(server, "/api/label/set", {"names": [f"c{i:02d}.csv" for i in range(6, 12)], "label": "이상"})
    assert s["counts"]["labeled"] == 12
    code, s = _call(server, "/api/label/undo", {})
    assert s["counts"]["labeled"] == 6
    code, s = _call(server, "/api/label/redo", {})
    assert s["counts"]["labeled"] == 12
    code, r = _call(server, "/api/label/rule_preview", {"conds": [{"stat": "max", "op": ">", "value": 0.5}], "overwrite": True})
    assert code == 200 and r["count"] >= 1
    code, u = _call(server, "/api/label/use", {})
    assert code == 200 and u["summary"]["classes"] == ["이상", "정상"] and u["label_file"].endswith("_정답.csv"), u
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 1}})
    assert _wait(server, j["id"])["state"] == "done"
    code, o = _call(server, "/api/label/open", {"file": str(work)})
    assert code == 200 and o["counts"]["labeled"] == 12


def test_label_page_served(server):
    (server.static_dir / "label.html").write_text("<!doctype html><title>라벨링 작업실</title>", encoding="utf-8")
    with urllib.request.urlopen(urllib.request.Request(_url(server, "/label?folder=x")), timeout=10) as r:
        assert r.status == 200 and "라벨링 작업실" in r.read().decode("utf-8")


def test_label_api_concurrent_requests(server, tmp_path):
    """숫자키 연타처럼 요청이 겹쳐도 저장이 깨지지 않음 (작업 하나에 잠금)."""
    root = tmp_path / "cc"
    root.mkdir()
    for i in range(16):
        (root / f"c{i:02d}.csv").write_text("v\n" + "\n".join(str(j % 5) for j in range(10)), encoding="utf-8")
    _call(server, "/api/label/new", {"path": str(root), "column": "v", "classes": ["a", "b"], "file": str(tmp_path / "w.dllabel")})
    codes = []

    def hit(k):
        for r in range(6):
            codes.append(_call(server, "/api/label/set", {"names": [f"c{(k * 6 + r) % 16:02d}.csv"], "label": "ab"[r % 2]})[0])
    ts = [threading.Thread(target=hit, args=(k,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert codes == [200] * 36


def test_label_confirm_response_reflects_confirmation(server, tmp_path):
    root = tmp_path / "cf"
    for c in ("a", "b"):
        (root / c).mkdir(parents=True)
        for i in range(6):
            (root / c / f"{i}.csv").write_text("v\n" + "\n".join(str(j % 4) for j in range(10)), encoding="utf-8")
    code, s = _call(server, "/api/label/new", {"path": str(root), "column": "v", "classes": ["a", "b"], "file": str(tmp_path / "w.dllabel")})
    assert s["counts"]["unconfirmed"] == 12
    code, s = _call(server, "/api/label/confirm", {"names": ["a/0.csv", "a/1.csv"]})
    assert s["changed"] == 2 and s["counts"]["unconfirmed"] == 10            # 응답이 확인한 뒤의 상태


def test_label_cluster_and_suggest_api(server, tmp_path):
    root = tmp_path / "sh"
    root.mkdir()
    for i in range(8):
        for name, sign in (("s", 1), ("q", -1)):
            (root / f"{name}{i}.csv").write_text("v\n" + "\n".join(str(sign * np.sin(j / 4) + i * 0.01) for j in range(40)),
                                                 encoding="utf-8")
    _call(server, "/api/label/new", {"path": str(root), "column": "v", "classes": ["s", "q"], "file": str(tmp_path / "w.dllabel")})
    code, c = _call(server, "/api/label/cluster", {"k": 2})
    assert code == 200 and sorted(g["size"] for g in c["groups"]) == [8, 8]
    code, it = _call(server, "/api/label/items", {"filter": {"cluster": c["groups"][0]["id"]}, "size": 60})
    assert it["total"] == 8
    _call(server, "/api/label/set", {"names": ["s0.csv", "s1.csv"], "label": "s"})
    _call(server, "/api/label/set", {"names": ["q0.csv", "q1.csv"], "label": "q"})
    code, r = _call(server, "/api/label/suggest", {})
    assert code == 200 and r["applied"] == 12 and r["counts"]["model"] == 12


def test_label_image_and_table_and_number_use(server, tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    img = tmp_path / "imgs"
    for k, c in enumerate(("밝음", "어두움")):
        (img / c).mkdir(parents=True)
        for i in range(6):
            Image.new("L", (12, 12), color=220 - 180 * k + i).save(img / c / f"{i}.png")
    code, s = _call(server, "/api/label/new", {"path": str(img), "kind": "image_folder", "classes": [], "file": str(tmp_path / "i.dllabel")})
    assert code == 200 and s["counts"]["import"] == 12, s
    with urllib.request.urlopen(urllib.request.Request(_url(server, "/api/label/file?name=" + urllib.parse.quote("밝음/0.png"))), timeout=10) as r:
        assert r.status == 200 and r.read()[:4] == b"\x89PNG"
    with pytest.raises(urllib.error.HTTPError):                              # 작업에 없는 이름은 거절
        urllib.request.urlopen(urllib.request.Request(_url(server, "/api/label/file?name=" + urllib.parse.quote("../x.png"))), timeout=10)
    code, u = _call(server, "/api/label/use", {})
    assert code == 200 and u["summary"]["image"] == [28, 28] and u["summary"]["classes"] == ["밝음", "어두움"], u

    t = tmp_path / "공정.csv"
    t.write_text("온도,압력\n" + "\n".join(f"{20 + i},{1 + (i % 4) * 0.1}" for i in range(20)), encoding="utf-8")
    code, s = _call(server, "/api/label/new", {"path": str(t), "kind": "table", "schema": "number", "classes": [],
                                               "file": str(tmp_path / "t.dllabel")})
    assert code == 200 and s["schema"] == "number" and s["stat_names"] == ["온도", "압력"], s
    code, s = _call(server, "/api/label/rule_apply", {"conds": [{"stat": "온도", "op": ">=", "value": 0}], "label": 1.5})
    _call(server, "/api/label/set", {"names": ["행 1", "행 2", "행 3"], "label": 9.0})
    code, u = _call(server, "/api/label/use", {})
    assert code == 200 and u["summary"]["task"] == "regression" and u["summary"]["length"] == 2, u


def test_label_segments_api(server, tmp_path):
    root = tmp_path / "stg"
    root.mkdir()
    for i in range(12):
        st = ["A"] * 10 + ["B"] * 20
        (root / f"c{i:02d}.csv").write_text("v,stage\n" + "\n".join(f"{j % 7},{s}" for j, s in enumerate(st)), encoding="utf-8")
    code, s = _call(server, "/api/label/new", {"path": str(root), "column": "v", "classes": [], "schema": "segments",
                                               "file": str(tmp_path / "g.dllabel")})
    assert code == 200 and s["schema"] == "segments", s
    code, c = _call(server, "/api/label/columns", {})
    assert c["columns"] == ["v", "stage"]
    code, r = _call(server, "/api/label/import_column", {"column": "stage"})
    assert code == 200 and r["samples"] == 12 and [k["name"] for k in r["classes"]] == ["A", "B"]
    code, r = _call(server, "/api/label/paint", {"name": "c00.csv", "start": 5, "end": 12, "cls": "B"})
    assert code == 200
    code, d = _call(server, "/api/label/item", {"name": "c00.csv"})
    assert d["segments"] == [[0, 5, "A"], [5, 30, "B"]] and d["by"] == "manual"
    code, e = _call(server, "/api/label/use", {})
    assert code == 400 and "구간" in e["error"]["message"]


def test_label_anomaly_api(server, tmp_path):
    root = tmp_path / "an"
    root.mkdir()
    for i in range(12):
        top = 9.0 if i in (2, 5) else 1.0
        (root / f"c{i:02d}.csv").write_text("v\n" + "\n".join(str(top if j == 10 else np.sin(j / 3) * 0.1) for j in range(20)),
                                            encoding="utf-8")
    code, s = _call(server, "/api/label/new", {"path": str(root), "column": "v", "classes": [], "purpose": "anomaly",
                                               "file": str(tmp_path / "a.dllabel")})
    assert code == 200 and s["purpose"] == "anomaly" and [c["name"] for c in s["classes"]] == ["정상", "이상"]
    code, r = _call(server, "/api/label/normal_candidates", {"k": 2})
    assert code == 200 and r["count"] == 10
    _call(server, "/api/label/set", {"names": r["names"], "label": "정상"})
    code, e = _call(server, "/api/label/export_normals", {"file": str(tmp_path / "n.csv")})
    assert code == 200 and e["rows"] == 10


def test_label_boxes_api(server, tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    root = tmp_path / "det"
    root.mkdir()
    for i in range(3):
        Image.new("L", (40, 20), color=100).save(root / f"{i}.png")
    code, s = _call(server, "/api/label/new", {"path": str(root), "kind": "image_folder", "schema": "boxes", "classes": ["결함"],
                                               "file": str(tmp_path / "b.dllabel")})
    assert code == 200 and s["schema"] == "boxes", s
    code, s = _call(server, "/api/label/set_boxes", {"name": "0.png", "boxes": [[0.1, 0.1, 0.5, 0.5, "결함"]]})
    assert code == 200 and s["counts"]["labeled"] == 1
    code, r = _call(server, "/api/label/export_yolo", {"folder": str(tmp_path / "y")})
    assert code == 200 and r["boxes"] == 1 and (tmp_path / "y" / "labels" / "0.txt").exists()
    code, r = _call(server, "/api/label/export_coco", {"file": str(tmp_path / "c.json")})
    assert code == 200 and (tmp_path / "c.json").exists()
    code, e = _call(server, "/api/label/use", {})
    assert code == 400 and "객체 탐지" in e["error"]["message"]


def test_label_team_api(server, tmp_path):
    root = tmp_path / "tm"
    root.mkdir()
    for i in range(12):
        (root / f"c{i:02d}.csv").write_text("v\n" + "\n".join(str((j + i) % 5) for j in range(10)), encoding="utf-8")
    from dlstudio.labeling import LabelProject
    q = LabelProject.new(str(root), "v", ["a", "b"], str(tmp_path / "other.dllabel"))
    q.who = "동료"
    q.set_labels(["c00.csv"], "b")
    q.set_labels(["c01.csv"], "a")
    _call(server, "/api/label/new", {"path": str(root), "column": "v", "classes": ["a", "b"], "file": str(tmp_path / "me.dllabel")})
    code, s = _call(server, "/api/label/who", {"name": "나"})
    assert code == 200 and s["who"] == "나"
    _call(server, "/api/label/set", {"names": ["c00.csv"], "label": "a"})
    code, r = _call(server, "/api/label/merge", {"file": str(tmp_path / "other.dllabel")})
    assert code == 200 and r["added"] == 1 and r["conflicts"] == 1 and r["counts"]["conflict"] == 1
    code, s = _call(server, "/api/label/resolve", {"name": "c00.csv", "take": "theirs"})
    assert code == 200 and s["counts"]["conflict"] == 0
    code, r = _call(server, "/api/label/split", {"n": 2})
    assert code == 200 and len(r["files"]) == 2


def test_graph_check_and_train_far_concat(server):
    """선으로 이은 그래프 (멀리 건너뛰는 Concat)를 그대로 검사·학습."""
    from tests.test_graph_check import far_concat
    _call(server, "/api/data/load", {"example": "ecg"})
    g = far_concat()
    code, c = _call(server, "/api/model/check", {"graph": g})
    assert code == 200 and c["ok"] and c["shapes"]["cat"] == [16, 140] and c["graph"] == g, c
    _, j = _call(server, "/api/train", {"graph": g, "settings": {"epochs": 1}})
    assert _wait(server, j["id"])["state"] == "done"
    code, a = _call(server, "/api/activations", {"sample_index": 1})
    assert code == 200 and "cat" in a["values"]
    code, e = _call(server, "/api/train", {"graph": {"nodes": "x"}})
    assert code == 400


def test_save_after_reloading_data_keeps_edited_graph(server, tmp_path):
    """학습 뒤 데이터를 다시 불러오고 모델을 고쳐 저장하면, 예전에 학습한 그래프가 아니라 고친 그래프가 저장됨."""
    from dlstudio.project import open_project
    _call(server, "/api/data/load", {"example": "ecg"})
    _, j = _call(server, "/api/train", {"blocks": CNN, "settings": {"epochs": 1}})
    _wait(server, j["id"])
    _call(server, "/api/data/load", {"example": "ecg"})
    edited = from_blocks([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "global_avg"}, {"type": "output"}])
    _call(server, "/api/model/check", {"graph": edited})
    code, _ = _call(server, "/api/project/save", {"path": str(tmp_path / "e.dlstudio")})
    assert code == 200 and open_project(str(tmp_path / "e.dlstudio"))["graph"] == edited
    assert _call(server, "/api/state")[1]["trained_graph"] is None


def test_unfinished_graph_survives_reload(server):
    """아직 선을 다 잇지 않은(검사 문제 있는) 그래프도 서버가 기억해서, 새로 고쳐도 하던 작업이 남음."""
    from tests.test_graph_check import far_concat
    _call(server, "/api/data/load", {"example": "ecg"})
    g = far_concat()
    g["nodes"].append({"id": "loose", "type": "dropout", "rate": 0.2, "inputs": ["a"]})
    code, c = _call(server, "/api/model/check", {"graph": g})
    assert code == 200 and not c["ok"]
    assert _call(server, "/api/state")[1]["graph"] == g


# ---------------------------------------------------------------- 평문 모드

TANH_ANYWHERE = {"version": 2, "nodes": [
    {"id": "input", "type": "input"},
    {"id": "c", "type": "conv", "inputs": ["input"], "channels": 8, "kernel": 5, "stride": 1},
    {"id": "b", "type": "bn", "inputs": ["c"]},
    {"id": "t", "type": "act", "inputs": ["b"], "kind": "tanh"},
    {"id": "g", "type": "global_avg", "inputs": ["t"]},
    {"id": "o", "type": "output", "inputs": ["g"]}]}


def test_relaxed_graph_flow(server, tmp_path):
    """HE 때문에 막던 조립(Tanh 아무 곳)도 검사·학습·샘플 따라가기·내보내기·다시 열기가 됨."""
    _call(server, "/api/data/load", {"example": True})
    code, c = _call(server, "/api/model/check", {"graph": TANH_ANYWHERE})
    assert code == 200 and c["ok"] and c["panel"]["params"] > 0 and c["panel"]["mb"] > 0
    code, j = _call(server, "/api/train", {"graph": TANH_ANYWHERE, "settings": {"epochs": 2, "lr": 0.01}})
    st = _wait(server, j["id"])
    assert st["state"] == "done", st
    code, a = _call(server, "/api/activations", {"sample_index": 0})
    assert code == 200 and "t" in a["values"]
    code, x = _call(server, "/api/project/export", {"path": str(tmp_path / "m.pt")})
    assert code == 200 and (tmp_path / "m.pt").exists() and x["info"]["classes"] == ["정상", "이상"]
    _call(server, "/api/project/save", {"path": str(tmp_path / "p.dlstudio")})
    code, o = _call(server, "/api/project/open", {"path": str(tmp_path / "p.dlstudio")})
    assert code == 200 and o["trained"] is True


def test_removed_he_endpoints_are_gone(server):
    for path in ("/api/remote/check", "/api/remote/run", "/api/mode"):
        assert _call(server, path, {})[0] == 404


def test_code_api_and_save(server, tmp_path):
    """[코드로 보기]: 두 framework 코드, 학습한 모델이면 PyTorch 저장에 가중치도. 화면용 값(pos)은 모델 비교에서 뺌."""
    _call(server, "/api/data/load", {"example": True})
    g = from_blocks(CNN)
    code, r = _call(server, "/api/code", {"graph": g, "framework": "pytorch"})
    assert code == 200 and "class StudioModel" in r["code"] and r["trained"] is False and r["filename"].endswith(".py")
    code, r = _call(server, "/api/code", {"graph": g, "framework": "tensorflow"})
    assert code == 200 and "def build_model" in r["code"]
    assert _call(server, "/api/code", {"graph": g, "framework": "jax"})[0] == 400
    bad = {"version": 2, "nodes": [{"id": "input", "type": "input"}]}
    assert _call(server, "/api/code", {"graph": bad, "framework": "pytorch"})[0] == 400
    _, j = _call(server, "/api/train", {"graph": g, "settings": {"epochs": 1, "lr": 0.01}})
    assert _wait(server, j["id"])["state"] == "done"
    moved = json.loads(json.dumps(g))
    moved["nodes"][1]["pos"] = [10, 20]
    code, r = _call(server, "/api/code/save", {"graph": moved, "framework": "pytorch", "path": str(tmp_path / "m.py")})
    assert code == 200 and r["trained"] and len(r["files"]) == 2
    assert (tmp_path / "m_weights.pt").exists() and 'torch.load("m_weights.pt")' in (tmp_path / "m.py").read_text(encoding="utf-8")
    code, r = _call(server, "/api/code/save", {"graph": g, "framework": "tensorflow", "path": str(tmp_path / "k.py")})
    assert code == 200 and r["files"] == [str(tmp_path / "k.py")]
    assert _call(server, "/api/code/save", {"graph": g, "framework": "pytorch", "path": str(tmp_path / "m.txt")})[0] == 400

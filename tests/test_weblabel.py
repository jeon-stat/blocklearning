"""웹 라벨링: 브라우저 보관함용 작업(정답 JSON) + 데이터 사본(zip)으로 저장 → 다시 열기 → studio 학습 데이터로."""
import io
import json
import zipfile

import numpy as np
import pytest

from dlstudio import web, weblabel


@pytest.fixture()
def fs(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "DATA_DIR", str(tmp_path / "data").replace("\\", "/"))
    monkeypatch.setattr(weblabel, "OUT", str(tmp_path / "out").replace("\\", "/"))
    root = tmp_path / "data" / "u1" / "사이클"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(24):
        bad = i % 2 == 1
        v = np.sin(np.linspace(0, 6, 60)) + (rng.normal(0, 0.8, 60) if bad else 0)
        (root / f"c{i:02d}.csv").write_text("t,v\n" + "\n".join(f"{t},{x:.4f}" for t, x in enumerate(v)), encoding="utf-8")
    return tmp_path


def _wrapped(s):
    r = json.loads(s)
    assert r["status"] == 200, r
    return r["body"]


def _job(fs):
    s = web.WebStudio()
    path = f"{web.DATA_DIR}/u1/사이클"
    st = _wrapped(weblabel.new_job(s, json.dumps({"path": path, "column": "v", "classes": ["정상", "이상"], "kind": "series_folder"})))
    assert st["counts"]["total"] == 24 and st["file"].endswith("/u1/_work.dllabel")
    for i in range(24):
        assert s.call("POST", "/api/label/set", {"names": [f"c{i:02d}.csv"], "label": "이상" if i % 2 else "정상"})[0] == 200
    return s


def test_save_reopen_and_use_in_studio(fs):
    s = _job(fs)
    work, data = weblabel.work_text(s), weblabel.data_bytes(s)
    assert json.loads(work)["source"]["path"] == "@data/u1/사이클"
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert len(names) == 24 and all(n.startswith("사이클/") for n in names) and weblabel.WORK not in " ".join(names)
    summ = json.loads(weblabel.summary(s))
    assert summ["total"] == 24 and summ["labeled"] == 24 and summ["classes"] == ["정상", "이상"]
    dz = fs / "d.zip"
    dz.write_bytes(data)
    s2 = web.WebStudio()                                  # 다른 창(다른 Pyodide)에서 다시 엶
    st = _wrapped(weblabel.open_job(s2, str(dz), work))
    assert st["counts"]["labeled"] == 24 and "/L" in st["file"]
    s3 = web.WebStudio()                                  # studio ② 라벨링한 데이터
    r = _wrapped(weblabel.use_job(s3, str(dz), work, False))
    assert r["summary"]["n_train"] + r["summary"]["n_test"] == 24 and sorted(r["summary"]["classes"]) == ["이상", "정상"]
    src = s3.data.source
    assert web._data_roots(src) and src["label_file"].replace("\\", "/").startswith(web.DATA_DIR + "/L")   # studio 프로젝트의 데이터 사본에 들어감
    assert web._portable_source(src)["series_folder"].startswith("@data/L")


def test_bundle_split_merge(fs):
    s = _job(fs)
    r = _wrapped(weblabel.export_file(s, "bundle", json.dumps({"name": "사이클"})))
    (name, path), = r["files"]
    assert name == "사이클.dllabel.zip"
    s2 = web.WebStudio()
    st = _wrapped(weblabel.open_bundle(s2, path))
    assert st["counts"]["labeled"] == 24
    parts = _wrapped(weblabel.export_file(s, "split", json.dumps({"name": "사이클", "n": 2})))["files"]
    assert [p[0] for p in parts] == ["사이클_part1.dllabel.zip", "사이클_part2.dllabel.zip"]
    s3 = web.WebStudio()                                  # 받은 사람이 자기 부분을 열고 하나 고침
    st = _wrapped(weblabel.open_bundle(s3, parts[0][1]))
    assert st["counts"]["total"] == 12
    first = sorted(s3.label.samples)[0]
    s3.call("POST", "/api/label/set", {"names": [first], "label": "이상" if s.label.meta["items"][first]["label"] == "정상" else "정상"})
    back = fs / "back.dllabel.zip"
    back.write_bytes(weblabel.bundle_bytes(s3))
    r = _wrapped(weblabel.merge_file(s, str(back)))       # 처음 작업에 합치기: 그쪽만 고친 하나를 가져옴
    assert r["added"] == 1 and r["conflicts"] == 0 and s.label.meta["items"][first]["label"] == s3.label.meta["items"][first]["label"]
    r = _wrapped(weblabel.export_file(s, "export", json.dumps({"name": "사이클"})))
    assert r["rows"] == 24 and open(r["files"][0][1], encoding="utf-8-sig").readline().strip() == "file,label"


def test_bad_inputs(fs):
    s = web.WebStudio()
    r = json.loads(weblabel.new_job(s, json.dumps({"path": "/elsewhere/x", "kind": "series_folder", "classes": ["a", "b"]})))
    assert r["status"] == 400
    junk = fs / "junk.zip"
    with zipfile.ZipFile(junk, "w") as z:
        z.writestr("x.txt", "x")
    assert json.loads(weblabel.open_bundle(s, str(junk)))["status"] == 400

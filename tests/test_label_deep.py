"""라벨링: 여러 열(기준 열 + 함께 볼 열) 시계열, 딥러닝 제안 (PC판 torch · 웹판 준비·적용)."""
import json

import numpy as np
import pytest

from dlstudio import web, weblabel
from dlstudio.core import Studio

torch = pytest.importorskip("torch")


def _folder(root, n=40):
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        bad = i % 2
        t = np.linspace(0, 6, 80)
        a = np.sin(t) + (rng.normal(0, .8, 80) if bad else rng.normal(0, .05, 80))
        b = np.cos(t) * (2 if bad else 1)
        (root / f"c{i:02d}.csv").write_text("t,a,b\n" + "\n".join(f"{k},{x:.4f},{y:.4f}" for k, (x, y) in enumerate(zip(a, b))),
                                            encoding="utf-8")
    return root


def _label_some(s, k=16):
    for i in range(k):
        assert s.call("POST", "/api/label/set", {"names": [f"c{i:02d}.csv"], "label": "이상" if i % 2 else "정상"})[0] == 200


def test_multichannel_series_and_deep_suggest_pc(tmp_path):
    root = _folder(tmp_path / "cyc")
    s = Studio()
    c, st = s.call("POST", "/api/label/new", {"path": str(root), "column": "a", "columns": ["a", "b"], "classes": ["정상", "이상"],
                                              "file": str(tmp_path / "w.dllabel"), "kind": "series_folder"})
    assert c == 200 and st["source"]["columns"] == ["a", "b"] and "b·max" in st["stat_names"]
    it = s.call("POST", "/api/label/item", {"name": "c01.csv"})[1]
    assert [ch["name"] for ch in it["channels"]] == ["a", "b"] and len(it["channels"][1]["values"]) == 80
    assert len(s.call("POST", "/api/label/items", {})[1]["items"][0]["thumbs"]) == 2
    r = s.call("POST", "/api/label/rule_preview", {"conds": [{"stat": "b·max", "op": ">", "value": 1.5}], "label": "이상"})[1]
    assert r["count"] == 20                                        # 함께 볼 열의 요약값으로도 규칙
    assert s.call("POST", "/api/label/deep_suggest", {})[0] == 400  # 정답이 모자람
    _label_some(s)
    c, r = s.call("POST", "/api/label/deep_suggest", {})
    assert c == 200 and r["applied"] == 24 and r["model"] == "1D CNN" and r["trained_on"] == 16
    lab = s.label.labels()
    wrong = sum(lab.get(f"c{i:02d}.csv") != ("이상" if i % 2 else "정상") for i in range(16, 40))
    assert wrong <= 2
    assert all(s.label.meta["items"][f"c{i:02d}.csv"]["by"] == "model" for i in range(16, 40))
    assert s.label.meta["items"]["c00.csv"]["by"] == "manual"     # 사람이 붙인 것은 그대로
    reopened = Studio()                                           # 작업 파일에 열 목록이 남음
    assert reopened.call("POST", "/api/label/open", {"file": str(tmp_path / "w.dllabel")})[1]["source"]["columns"] == ["a", "b"]


def test_deep_suggest_web_prepare_and_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "DATA_DIR", str(tmp_path / "data").replace("\\", "/"))
    root = _folder(tmp_path / "data" / "u1" / "cyc")
    s = web.WebStudio()
    st = json.loads(weblabel.new_job(s, json.dumps({"path": f"{web.DATA_DIR}/u1/cyc", "column": "a", "classes": ["정상", "이상"],
                                                     "kind": "series_folder"})))
    assert st["status"] == 200, st
    _label_some(s)
    prep = json.loads(weblabel.deep_prepare(s))
    assert prep["status"] == 200 and prep["body"]["n_pred"] == 24 and prep["body"]["plan"]["steps"]
    p = weblabel.deep_payload()
    meta = json.loads(p["meta"])
    Xp = np.frombuffer(p["X_pred"], dtype=np.float32).reshape(meta["n_pred"], -1)
    probs = np.tile(np.array([[0.2, 0.8]], np.float32), (meta["n_pred"], 1))    # 엔진 대신: 모두 두 번째 클래스
    r = json.loads(weblabel.deep_apply(s, probs.tobytes(), 0.9))
    assert r["status"] == 200 and r["body"]["applied"] == 24 and r["body"]["acc"] == 0.9
    second = prep["body"]["classes"][1]
    assert all(s.label.meta["items"][f"c{i:02d}.csv"]["label"] == second for i in range(16, 40))
    assert Xp.shape[1] == meta["length"]

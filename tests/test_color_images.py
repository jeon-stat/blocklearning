"""컬러·큰 그림 폴더: RGB 3채널로 읽기, 모델 입력 (3, H, W), 학습·예측·어디를 봤나·코드·TorchScript, 메모리 한도."""
import time

import numpy as np
import pytest

from blocklearning import data as D
from blocklearning.core import Studio

PIL = pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

SIZE = 48


def _folder(root, n=24, size=40):
    """빨간 그림과 파란 그림 (밝기는 같게: 흑백으로는 구별이 안 됨)."""
    rng = np.random.default_rng(0)
    for name, rgb in (("빨강", (200, 40, 40)), ("파랑", (40, 40, 200))):
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            a = np.clip(np.array(rgb)[None, None, :] + rng.normal(0, 20, (size, size, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(a, "RGB").save(d / f"{i}.png")
    return root


def _wait(s, r):
    for _ in range(900):
        st = s.call("GET", f"/api/jobs/{r['id']}")[1]
        if st["state"] != "running":
            assert st["state"] == "done", st
            return st["result"]
        time.sleep(0.1)
    raise AssertionError("학습이 끝나지 않음")


G = {"version": 2, "nodes": [{"id": "input", "type": "input"},
                             {"id": "c", "type": "conv2d", "channels": 8, "kernel": 3, "stride": 2, "inputs": ["input"]},
                             {"id": "a", "type": "act", "kind": "relu", "inputs": ["c"]},
                             {"id": "g", "type": "global_avg2d", "inputs": ["a"]},
                             {"id": "o", "type": "output", "inputs": ["g"]}]}


def test_pixels_color_is_channel_first():
    im = Image.new("RGB", (4, 4), (255, 0, 0))
    v = D._pixels01(im, 4, Image, True)
    assert v.shape == (48,) and np.allclose(v[:16], 1) and np.allclose(v[16:], 0)
    assert D._pixels01(im, 4, Image).shape == (16,)                       # 흑백은 그대로


def test_color_folder_trains_and_predicts(tmp_path):
    root = _folder(tmp_path / "imgs")
    d = D.load_image_folder(str(root), size=SIZE, color=True)
    assert d.image == (SIZE, SIZE) and d.length.channels == 3 and d.X_train.shape[1] == 3 * SIZE * SIZE
    assert d.source["color"] is True
    s = Studio()
    c, r = s.call("POST", "/api/data/load", {"folder": str(root), "size": SIZE, "color": True})
    assert c == 200, r
    assert r["summary"]["image"] == [SIZE, SIZE] and r["summary"]["channels"] == 3
    c, chk = s.call("POST", "/api/model/check", {"graph": G})
    assert c == 200 and chk["ok"], chk
    c, r = s.call("POST", "/api/train", {"graph": G, "settings": {"epochs": 15, "lr": 0.01}})
    assert c == 200, r
    res = _wait(s, r)
    assert res["test_metric"] >= 0.9, res["test_metric"]                  # 색만 다른 그림을 구별
    c, a = s.call("POST", "/api/activations", {"sample_index": 0})
    assert c == 200 and len(a["input"]) == 3 * SIZE * SIZE and len(a["saliency"]) == SIZE * SIZE
    c, p = s.call("POST", "/api/predict", {"folder": str(root)})
    assert c == 200, p
    assert sum(p["summary"]["counts"].values()) == 48 and p["summary"]["accuracy"] >= 0.9, p["summary"]


def test_color_export_and_code(tmp_path):
    from blocklearning import codegen
    from blocklearning.project import _export_module as export_module
    root = _folder(tmp_path / "imgs", n=12)
    d = D.load_image_folder(str(root), size=32, color=True)
    from blocklearning.graph import build_graph_model
    net = build_graph_model(G, d.length, d.n_out, image=d.image)
    m = export_module(net, d)
    raw = d.X_test * d.std + d.mean
    with torch.no_grad():
        out = m(torch.as_tensor(raw, dtype=torch.float32))
    assert out.shape == (len(raw), 2)
    code = codegen.to_pytorch(G, d.length, d.n_out, d.image, data_info={"task": "classification"})
    assert "(샘플 수, 3, 32, 32) — 컬러 그림" in code
    ns = {}
    exec(compile(code, "gen", "exec"), ns)
    assert ns["StudioModel"]()(torch.zeros(2, 3, 32, 32)).shape == (2, 2)


def test_big_image_memory_limit(tmp_path, monkeypatch):
    root = _folder(tmp_path / "imgs", n=4, size=8)
    monkeypatch.setattr(D, "MAX_IMAGE_BYTES", 1000)
    with pytest.raises(D.DataError) as e:
        D.load_image_folder(str(root), size=224, color=True)
    assert "메모리" in e.value.issues[0].message

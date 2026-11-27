"""생성 (VAE): 잠재 공간 블록(학습 때 표본·KL, 평가 때 평균), 디코더만 실행해 새 샘플, 서버 흐름 (예시 → 학습 → 만들기),
그림만 든 폴더, 디코더가 앞쪽을 쓰면 안내."""
import time

import numpy as np
import pytest

from blocklearning.core import Studio
from blocklearning.graph import build_graph_model, check_graph
from blocklearning.tasks import catalog_view

torch = pytest.importorskip("torch")


def _templates(sub):
    fam = next(f for f in catalog_view()["families"] if f["id"] == "generation")
    return {t["id"]: t["graph"] for s in fam["subtasks"] if s["id"] == sub for t in s["templates"]}


def _wait(s, r):
    for _ in range(1200):
        st = s.call("GET", f"/api/jobs/{r['id']}")[1]
        if st["state"] != "running":
            assert st["state"] == "done", st
            return st["result"]
        time.sleep(0.1)
    raise AssertionError("학습이 끝나지 않음")


def test_latent_block_sampling_and_kl():
    from blocklearning.nn_extra import VAELatent
    m = VAELatent(8, 4, beta=0.5)
    x = torch.randn(32, 8)
    m.train()
    z1, z2 = m(x), m(x)
    assert z1.shape == (32, 4) and not torch.equal(z1, z2) and m.kl is not None and float(m.kl) >= 0
    m.eval()
    assert torch.equal(m(x), m.mu(x)) and m.kl is None


def test_generate_runs_only_decoder():
    g = _templates("gen_ts")["vae_dense"]
    net = build_graph_model(g, 140, 140)
    out = net.generate(torch.randn(5, 16))
    assert out.shape == (5, 140)
    bad = {"version": 2, "nodes": [{"id": "input", "type": "input"}, {"id": "f", "type": "flatten", "inputs": ["input"]},
                                   {"id": "z", "type": "vae_latent", "dim": 4, "inputs": ["f"]},
                                   {"id": "c", "type": "concat", "inputs": ["z", "f"]}, {"id": "o", "type": "output", "inputs": ["c"]}]}
    assert check_graph(bad, 20, 20).ok
    with pytest.raises(ValueError, match="앞의 블록"):
        build_graph_model(bad, 20, 20).generate(torch.randn(2, 4))


@pytest.mark.parametrize("example,tpl", [("digits_gen", "vae_conv"), ("ecg_gen", "vae_dense")])
def test_server_train_and_generate(example, tpl):
    s = Studio()
    c, r = s.call("POST", "/api/data/load", {"example": example})
    assert c == 200 and r["summary"]["task"] == "generation", r
    g = _templates("gen_img" if example == "digits_gen" else "gen_ts")[tpl]
    c, r = s.call("POST", "/api/train", {"graph": g, "settings": {"epochs": 3, "lr": 0.003, "batch_size": 64}})
    assert c == 200, r
    res = _wait(s, r)
    assert res["metric_name"] == "recon" and 0 < res["test_metric"] < 1.5 and len(res["recon"]) == 6
    c, a = s.call("POST", "/api/generate", {"n": 8, "seed": 3})
    assert c == 200, a
    width = 1024 if example == "digits_gen" else 140
    assert len(a["samples"]) == 8 and all(len(v) == width for v in a["samples"]) and a["csv"].count("\n") == 9
    _, b = s.call("POST", "/api/generate", {"n": 8, "seed": 3})
    assert b["samples"] == a["samples"]                                       # 같은 seed면 같은 샘플
    _, other = s.call("POST", "/api/generate", {"n": 8, "seed": 4})
    assert other["samples"] != a["samples"]


def test_generate_needs_generation_model():
    s = Studio()
    s.call("POST", "/api/data/load", {"example": "ecg"})
    c, e = s.call("POST", "/api/generate", {"n": 4})
    assert c == 400


def test_flat_image_folder_for_generation(tmp_path):
    from PIL import Image
    for i in range(12):
        Image.fromarray((np.random.default_rng(i).random((20, 20)) * 255).astype(np.uint8)).save(tmp_path / f"{i}.png")
    s = Studio()
    c, r = s.call("POST", "/api/data/load", {"folder": str(tmp_path), "size": 16, "task": "generation"})
    assert c == 200 and r["summary"]["task"] == "generation" and r["summary"]["n_train"] + r["summary"]["n_test"] == 12, r

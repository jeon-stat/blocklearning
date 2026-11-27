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


# ---------------------------------------------------------------- 조건부 생성·확산 (DDPM)
import json
import shutil
import subprocess
from pathlib import Path

JS = Path(__file__).parent / "js"
HAS_NODE = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()
need_node = pytest.mark.skipif(not HAS_NODE, reason="node와 tests/js/node_modules(@tensorflow/tfjs)가 있어야 함")


def test_cond_blocks_add_per_channel_and_pass_through_without_cond():
    from blocklearning.nn_extra import ClassCond, TimeEmbed
    x = torch.randn(2, 3, 5, 5)
    cc, te = ClassCond(3, 4), TimeEmbed(3)
    assert torch.equal(cc(x), x) and torch.equal(te(x), x)                   # 조건을 넣지 않으면 그대로
    cc.y = torch.tensor([0, 3])
    assert torch.equal(cc(x), x)                                             # 처음 값 0: 조건 없음과 같음
    with torch.no_grad():
        cc.emb.weight.normal_()
    d = cc(x) - x
    assert torch.allclose(d[1], cc.emb.weight[3].view(3, 1, 1).expand(3, 5, 5))
    te.t = torch.tensor([0.0, 150.0])
    e = te(x) - x
    assert e.shape == x.shape and not torch.allclose(e[0], e[1]) and torch.allclose(e[0, :, 0, 0], e[0, :, 4, 4])


def test_schedule_and_graph_issue():
    from blocklearning import generate
    from blocklearning.data import load_example
    beta, alpha, ab = generate.schedule()
    assert len(ab) == generate.T_STEPS and 0.99 < ab[0] < 1 and ab[-1] < 1e-3 and np.all(np.diff(ab) < 0)
    d = load_example("digits_gen")
    assert generate.labels_of(d) == [str(k) for k in range(10)] and set(np.unique(d.y_train)) == set(range(10))
    g = _templates("gen_img")["cvae_conv"]
    assert generate.graph_issue(d, g) is None
    small = json.loads(json.dumps(g).replace('"classes": 10', '"classes": 5'))
    assert "클래스 수" in generate.graph_issue(d, small)[0]
    both = {"nodes": [{"type": "time_embed"}, {"type": "vae_latent"}]}
    assert generate.graph_issue(d, both) is not None
    s = Studio()
    s.call("POST", "/api/data/load", {"example": "digits_gen"})
    c, e = s.call("POST", "/api/train", {"graph": small, "settings": {"epochs": 1}})
    assert c == 400 and "클래스" in e["error"]["message"]


@pytest.mark.parametrize("example,sub,tpl", [("ecg_gen", "gen_ts", "cvae_dense"), ("ecg_gen", "gen_ts", "ddpm_mlp"),
                                             ("digits_gen", "gen_img", "cvae_conv")])
def test_conditional_and_diffusion_train_and_generate(example, sub, tpl):
    s = Studio()
    c, r = s.call("POST", "/api/data/load", {"example": example})
    assert c == 200 and len(r["summary"]["gen_labels"]) > 1
    c, r = s.call("POST", "/api/train", {"graph": _templates(sub)[tpl], "settings": {"epochs": 2, "lr": 0.003, "batch_size": 64}})
    assert c == 200, r
    res = _wait(s, r)
    assert res["metric_name"] == "recon" and np.isfinite(res["test_metric"]) and len(res["recon"]) == 6
    c, a = s.call("POST", "/api/generate", {"n": 6, "seed": 2, "label": 1})
    assert c == 200, a
    names = r = s.data.extra["labels"]
    assert a["classes"] == [names[1]] * 6 and a["csv"].startswith("class,")
    _, mixed = s.call("POST", "/api/generate", {"n": 4, "seed": 2})
    assert mixed["classes"] == names[:4] if len(names) >= 4 else len(set(mixed["classes"])) == len(names)
    _, again = s.call("POST", "/api/generate", {"n": 6, "seed": 2, "label": 1})
    assert again["samples"] == a["samples"]
    c, e = s.call("POST", "/api/generate", {"n": 2, "label": 99})
    assert c == 400


@need_node
@pytest.mark.parametrize("image", [False, True])
def test_cond_blocks_match_engine(tmp_path, image):
    """클래스 조건·확산 시점 블록: 같은 가중치·클래스·시점이면 엔진(웹)과 PyTorch의 출력이 같음."""
    from blocklearning import codegen, generate
    from blocklearning.blocks import to_model_input
    from blocklearning.graph import from_blocks
    cond = [{"type": "time_embed"}, {"type": "class_cond", "classes": 3}, {"type": "act", "kind": "relu"}, {"type": "output"}]
    if image:
        g, L, img = from_blocks([{"type": "conv2d", "channels": 4, "kernel": 3}] + cond), 64, (8, 8)
    else:
        g, L, img = from_blocks([{"type": "flatten"}, {"type": "dense", "units": 8}] + cond), 12, None
    net = build_graph_model(g, L, L, image=img).eval()
    with torch.no_grad():
        for k, p in net.named_parameters():
            if "emb" in k:
                p.normal_()
    X = np.random.default_rng(0).standard_normal((4, L)).astype(np.float32)
    x = torch.as_tensor(to_model_input(net.kind, X, image=img))
    y, t = [0, 2, 1, 2], [0.0, 50.0, 199.0, 3.0]
    net.set_cond(y=torch.tensor(y), t=torch.tensor(t))
    with torch.no_grad():
        want = net(x).numpy().ravel()
    f = tmp_path / "cond.json"
    f.write_text(json.dumps({"plan": codegen.plan_dict(g, L, L, img), "x": x.numpy().ravel().tolist(), "shape": list(x.shape), "y": y, "t": t,
                             "state": {k: {"shape": list(v.shape), "data": v.numpy().ravel().tolist()} for k, v in net.state_dict().items()}}))
    r = subprocess.run(["node", str(JS / "run_cond.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout)
    assert np.allclose(got["out"], want, atol=1e-4), np.abs(np.asarray(got["out"]) - want).max()
    assert got["kind"] == "diffusion" and got["classes"] == 3
    assert np.allclose(got["ab"], generate.schedule()[2], rtol=1e-9)


@need_node
@pytest.mark.parametrize("tpl", ["cvae_dense", "ddpm_mlp"])
def test_engine_trains_conditional_and_diffusion(tmp_path, tpl):
    """웹 엔진도 조건부 VAE·확산 모델을 학습: 손실이 줄고, 가중치를 PyTorch에 넣으면 같은 평가 출력(VAE)·새 샘플을 만듦."""
    from blocklearning import generate
    from blocklearning.data import load_example
    from tests.test_engine import _train_js
    d = load_example("ecg_gen")
    g = _templates("gen_ts")[tpl]
    out = _train_js(tmp_path, g, d, {"epochs": 3, "lr": 0.003, "batch_size": 64, "seed": 0})
    r, prog = out["result"], out["progress"]
    assert r["metric_name"] == "recon" and np.isfinite(r["test_metric"]) and prog[-1]["loss"] < prog[0]["loss"]
    net = build_graph_model(g, d.length, d.n_out).eval()
    net.load_state_dict({k: torch.tensor(v["data"]).reshape(v["shape"]).to(net.state_dict()[k].dtype) for k, v in out["state"].items()})
    if tpl == "cvae_dense":                                                 # 평가 출력 (클래스를 넣어 되살린 것)이 같음
        mine = generate.predict(net, d.X_test, d.y_test)
        assert np.allclose(np.asarray(out["out"]).reshape(mine.shape), mine, atol=2e-3)
    X, y, _ = generate.sample(net, d, 4, seed=0, label=1)
    assert X.shape == (4, 140) and list(y) == [1] * 4 and np.all(np.isfinite(X))


def test_judge_scores_generated_samples_against_real_data():
    """분류기 채점: 진짜 시험 데이터는 높게, 고른 클래스와 다른 진짜 샘플은 틀렸다고, 클래스가 없으면 None."""
    from blocklearning import generate
    from blocklearning.data import load_example
    d = load_example("digits_gen")
    y = np.asarray(d.y_test[:40], np.int64)
    ok = generate.judge(d, d.X_test[:40], y)
    assert ok["real_acc"] > 0.85 and ok["acc"] > 0.85 and ok["n"] == 40 and len(ok["pred"]) == 40
    wrong = generate.judge(d, d.X_test[:40], (y + 1) % 10)
    assert wrong["acc"] < 0.2 and wrong["real_acc"] == ok["real_acc"]
    assert generate.judge(d, d.X_test[:4], None) is None
    s = Studio()
    s.call("POST", "/api/data/load", {"example": "ecg_gen"})
    c, r = s.call("POST", "/api/train", {"graph": _templates("gen_ts")["cvae_dense"], "settings": {"epochs": 2, "batch_size": 64}})
    _wait(s, r)
    _, a = s.call("POST", "/api/generate", {"n": 8, "seed": 0, "label": 0})
    j = a["judge"]
    assert j["n"] == 8 and 0 <= j["acc"] <= 1 and j["real_acc"] > 0.8 and len(j["pred"]) == 8
    s2 = Studio()
    s2.call("POST", "/api/data/load", {"example": "ecg_gen"})
    c, r = s2.call("POST", "/api/train", {"graph": _templates("gen_ts")["vae_dense"], "settings": {"epochs": 1}})
    _wait(s2, r)
    assert s2.call("POST", "/api/generate", {"n": 4})[1]["judge"] is None          # 클래스를 고르지 않는 모델

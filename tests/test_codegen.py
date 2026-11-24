"""그래프 → 코드: PyTorch 코드는 studio 모델과 계산이 같고(가중치를 옮겨 출력 비교), Keras 코드는 같은 구조로 만들어진다."""
import ast
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from dlstudio import codegen, tasks
from dlstudio.blocks import to_model_input
from dlstudio.data import load_example
from dlstudio.graph import build_graph_model, check_graph
from tests.test_graph_build import random_graph


def G(*nodes):
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}, *nodes]}


def N(id, type, *inputs, **kw):
    return dict(id=id, type=type, inputs=list(inputs), **kw)


def _cases():
    """(이름, 그래프, 길이, 클래스 수, 이미지): 대표 모델 전부 + 무작위 그래프 + 특수 층."""
    out, seen = [], set()
    for f in tasks.catalog_view()["families"]:
        for s in f["subtasks"]:
            if not s.get("available"):
                continue
            d = load_example(s["example_data"][0][0])
            for t in s["templates"]:
                if (t["id"], d.image) not in seen:
                    seen.add((t["id"], d.image))
                    out.append((f"tpl_{t['id']}_{s['id']}", t["graph"], d.length, d.n_out, d.image))
    for seed in range(30):
        image = seed % 3 == 2
        rng = random.Random(seed)
        L = 64 if image else 24
        for _ in range(20):
            g = random_graph(rng, image)
            if check_graph(g, L, 3, image=(8, 8) if image else None).ok:
                out.append((f"rand{seed}", g, L, 3, (8, 8) if image else None))
                break
    out.append(("rnn_stack", G(N("r", "lstm", "input", hidden=6, layers=2), N("d", "dense", "r", units=4),
                               N("a", "act", "d", kind="mish"), N("o", "output", "a")), 20, 2, None))
    out.append(("gru", G(N("r", "gru", "input", hidden=5, layers=1), N("o", "output", "r")), 20, 2, None))
    out.append(("simple_rnn", G(N("r", "simple_rnn", "input", hidden=5, layers=3), N("o", "output", "r")), 20, 2, None))
    out.append(("odd_ids", G(N("class", "conv", "input", channels=4, kernel=3, stride=1), N("1x", "act", "class", kind="gelu"),
                             N("x", "se", "1x", ratio=2, act="tanh"), N("g", "last_step", "x"), N("o", "output", "g")), 20, 2, None))
    out.append(("vec_concat", G(N("c", "conv", "input", channels=4, kernel=5, stride=2, dilation=1, padding="valid"),
                                N("g", "global_avg", "c"), N("a", "dense", "g", units=3), N("b", "dense", "g", units=5),
                                N("k", "concat", "a", "b"), N("s", "add", "k", "k"), N("o", "output", "s")), 30, 2, None))
    out.append(("stride_dilation", G(N("c", "conv", "input", channels=4, kernel=3, stride=2, dilation=2),
                                     N("g", "global_avg", "c"), N("o", "output", "g")), 30, 2, None))
    return out


CASES = _cases()
IDS = [c[0] for c in CASES]


def _model_input(net, L, image, n=3, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, image[0] * image[1] if image else L))
    return torch.as_tensor(to_model_input(net.kind, X, image=image), dtype=torch.float32)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pytorch_code_matches_studio_model(case):
    name, g, L, n_out, image = case
    net = build_graph_model(g, L, n_out, seed=1, image=image).eval()
    code = codegen.to_pytorch(g, L, n_out, image, {"task": "classification", "classes": list(range(n_out))})
    ns: dict = {"__name__": "studio_code"}
    exec(compile(code, f"<{name}>", "exec"), ns)
    m = ns["StudioModel"]()
    m.load_state_dict(codegen.torch_state_dict(net, g, L, n_out, image), strict=True)
    m.eval()
    x = _model_input(net, L, image)
    with torch.no_grad():
        assert torch.allclose(m(x), net(x), atol=1e-5), name
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_keras_code_parses(case):
    name, g, L, n_out, image = case
    code = codegen.to_keras(g, L, n_out, image, {"task": "regression"})
    ast.parse(code)
    assert "build_model" in code and "import tensorflow" in code


def test_pytorch_script_runs_with_saved_weights(tmp_path):
    d = load_example("ecg")
    g = next(t["graph"] for t in tasks.catalog_view()["defaults"] if t["id"] == "cnn")
    net = build_graph_model(g, d.length, d.n_out, seed=2).eval()
    codegen.save_npz(codegen.torch_state_dict(net, g, d.length, d.n_out), str(tmp_path / "model_weights.npz"), data=d)
    code = codegen.to_pytorch(g, d.length, d.n_out, None, {"task": "classification", "classes": d.classes},
                              {"epochs": 3}, weights_file="model_weights.npz")
    (tmp_path / "model.py").write_text(code, encoding="utf-8")
    r = subprocess.run([sys.executable, "model.py"], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=120,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith("['")                       # 클래스 이름 목록
    assert "정상, 이상" in code and "epochs=3" in code
    # predict(원래 값) == 학습 때 모델(정규화된 값)과 같은 답
    ns: dict = {"__name__": "gen"}
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(code, ns)
        model, info = ns["load_trained"]()
    finally:
        os.chdir(cwd)
    raw = d.raw(d.X_test[:16])
    want = [str(d.classes[i]) for i in net(torch.as_tensor(d.X_test[:16, None, :], dtype=torch.float32)).argmax(1)]
    assert ns["predict"](model, info, raw) == want


def test_pytorch_train_function_learns():
    g = G(N("d", "dense", "input", units=8), N("a", "act", "d", kind="relu"), N("o", "output", "a"))
    ns: dict = {}
    exec(codegen.to_pytorch(g, 4, 2, None, {"task": "classification"}, {"epochs": 40, "lr": 0.05}), ns)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1, 4)).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.int64)
    m = ns["train"](ns["StudioModel"](), X, y)
    with torch.no_grad():
        acc = float((m(torch.as_tensor(X)).argmax(1).numpy() == y).mean())
    assert acc > 0.9


def test_bad_graph_and_framework():
    with pytest.raises(ValueError):
        codegen.to_pytorch(G(N("o", "output", "nope")), 10, 2)
    with pytest.raises(ValueError):
        codegen.generate("jax", G(N("o", "output", "input")), 10, 2)


# ---------------------------------------------------------------- Keras (TensorFlow가 있으면)

def _keras_expected_params(net) -> int:
    """torch 파라미터 수 → Keras 학습 파라미터 수. LSTM·SimpleRNN은 torch가 bias를 두 벌(ih·hh) 가짐."""
    n = sum(p.numel() for p in net.parameters())
    for mod in net.modules():
        if isinstance(mod, (torch.nn.LSTM, torch.nn.RNN)) and not isinstance(mod, torch.nn.GRU):
            n -= sum(p.numel() for k, p in mod.named_parameters() if k.startswith("bias_hh"))
    return n


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_keras_code_builds_same_structure(case):
    tf = pytest.importorskip("tensorflow")
    name, g, L, n_out, image = case
    net = build_graph_model(g, L, n_out, seed=1, image=image).eval()
    ns: dict = {}
    exec(compile(codegen.to_keras(g, L, n_out, image, {"task": "classification"}), f"<{name}>", "exec"), ns)
    model = ns["build_model"]()
    assert model.output_shape == (None, n_out), name
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    assert trainable == _keras_expected_params(net), name
    x = _model_input(net, L, image).numpy()
    xk = np.transpose(x, (0, 2, 3, 1)) if image else (x if net.kind == "rnn" else np.transpose(x, (0, 2, 1)))
    assert tuple(model(xk, training=False).shape) == (3, n_out)
    tf.keras.backend.clear_session()


def test_keras_train_function_runs():
    pytest.importorskip("tensorflow")
    g = G(N("d", "dense", "input", units=8), N("a", "act", "d", kind="relu"), N("o", "output", "a"))
    ns: dict = {}
    exec(codegen.to_keras(g, 4, 2, None, {"task": "classification"}, {"epochs": 2}), ns)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4, 1)).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.int64)
    ns["train"](ns["build_model"](), X, y)


def test_keras_stride_with_dilation_is_split():
    """Keras는 strides와 dilation_rate를 함께 못 받음 → stride 1 + 건너뛰어 뽑기."""
    g = G(N("c", "conv", "input", channels=4, kernel=3, stride=2, dilation=2), N("g", "global_avg", "c"), N("o", "output", "g"))
    code = codegen.to_keras(g, 30, 2)
    assert "strides=2" not in code and "t[:, ::2, :]" in code and "dilation_rate=2" in code


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_plan_params_match_torch_state(case):
    """웹 엔진용 계획의 가중치 이름·모양·학습 파라미터 수가 PyTorch(GraphNet)와 같음."""
    name, g, L, n_out, image = case
    net = build_graph_model(g, L, n_out, image=image)
    plan = codegen.plan_dict(g, L, n_out, image)
    specs = {p[0]: tuple(p[1]) for st in plan["steps"] for p in st["params"]}
    assert specs == {k: tuple(v.shape) for k, v in net.state_dict().items()}, name
    assert plan["params"] == sum(p.numel() for p in net.parameters())
    import json
    json.dumps(plan)

"""그래프 → GraphNet: 모양이 검사와 같고, 검사를 통과한 그래프는 만들어지고 앞으로 계산한다 (무작위 불변식)."""
import random
import warnings

import numpy as np
import pytest
import torch

from dlstudio.analysis import activations
from dlstudio.blocks import to_model_input
from dlstudio.graph import build_graph_model, check_graph, output_linear
from tests.test_graph_check import G, N, conv, dense_res, downsample_res, far_concat


def _X(n=8, L=24, seed=0):
    return np.random.default_rng(seed).normal(scale=0.5, size=(n, L))


class _Run:
    """만든 모델을 그대로 앞으로 계산 (예전 heml 변환 시험 자리: 만들어지고 돌아가는지·모양 확인)."""

    def __init__(self, model):
        self.model = model.eval()

    def simulate(self, X):
        import torch
        with torch.no_grad():
            return self.model(torch.as_tensor(np.asarray(X), dtype=torch.float32)).double().numpy()


def _convert(model, Xi):
    return _Run(model)


def image_res():
    return G(N("c0", "conv2d", "input", channels=4, kernel=3), N("r0", "act", "c0", kind="relu"),
             N("a", "conv2d", "r0", channels=8, kernel=3, stride=2), N("ra", "act", "a", kind="relu"),
             N("sc", "conv2d", "r0", channels=8, kernel=1, stride=2), N("s", "add", "ra", "sc"),
             N("g", "global_avg2d", "s"), N("o", "output", "g"))


STRUCTS = {"downsample_res": (downsample_res, None), "dense_res": (dense_res, None), "far_concat": (far_concat, None),
           "image_res": (image_res, (8, 8))}


def _run(net, Xi):
    with torch.no_grad():
        return net(torch.as_tensor(Xi, dtype=torch.float32)).double().numpy()


@pytest.mark.parametrize("name", list(STRUCTS))
def test_build_matches_shapes(name):
    make, image = STRUCTS[name]
    L = 64 if image else 24
    c = check_graph(make(), L, 3, image=image)
    assert c.ok, c.issues
    net = build_graph_model(make(), L, 3, image=image).eval()
    Xi = to_model_input(c.kind, _X(4, L), image=image)
    with torch.no_grad():
        vals = net.trace(torch.as_tensor(Xi, dtype=torch.float32))
    for k, s in c.shapes.items():
        assert tuple(vals[k].shape[1:]) == s, k


@pytest.mark.parametrize("name", list(STRUCTS))
def test_three_structures_convert(name):
    make, image = STRUCTS[name]
    L = 64 if image else 24
    c = check_graph(make(), L, 3, image=image)
    net = build_graph_model(make(), L, 3, image=image).eval()
    Xi = to_model_input(c.kind, _X(8, L), image=image)
    want = _run(net, Xi)
    assert np.max(np.abs(_convert(net, Xi).simulate(Xi) - want)) <= 1e-2 * max(1.0, np.max(np.abs(want)))


def random_graph(rng: random.Random, image: bool = False) -> dict:
    """무작위 그래프: 줄 + 잔차(add) + 갈래(concat) + 멀리 건너뛰는 concat/add + Repeat. 모양은 따라가며 맞춤."""
    acts = ["relu", "sigmoid", "gelu", "silu"]
    nodes, ids = [], iter(f"n{i}" for i in range(1000))
    seen = [("input", (1, None))]                       # (id, (채널, 크기 표시))
    cur, C, size = "input", 1, 0                       # size: 줄어든 횟수 (같으면 같은 길이)
    cv = "conv2d" if image else "conv"

    def add(t, src, **kw):
        nid = next(ids)
        nodes.append(dict(id=nid, type=t, inputs=src if isinstance(src, list) else [src], **kw))
        return nid

    def conv_node(src, ch, stride=1, k=3):
        return add(cv, src, channels=ch, kernel=k, stride=stride)

    for _ in range(rng.randint(1, 5)):
        r = rng.random()
        if r < 0.2:                                     # 잔차: 같은 모양이면 그대로, 아니면 1×1 지름길
            ch, st = rng.choice([4, 8]), rng.choice([1, 1, 2])
            a = conv_node(cur, ch, st)
            b = add("act", a, kind=rng.choice(acts))
            sc = cur if (ch == C and st == 1) else conv_node(cur, ch, st, 1)
            cur, C, size = add("add", [b, sc]), ch, size + (st == 2)
        elif r < 0.35:                                  # 갈래 concat
            ch1, ch2 = rng.choice([2, 4]), rng.choice([2, 4])
            cur, C = add("concat", [conv_node(cur, ch1), conv_node(cur, ch2, k=rng.choice([1, 3]))]), ch1 + ch2
        elif r < 0.45:                                  # 멀리 건너뛰기: 같은 크기의 앞 노드와 concat
            far = [(k, c) for k, (c, s) in seen if s == size and k != cur and k != "input"]
            if far:
                k, c = rng.choice(far)
                cur, C = add("concat", [cur, k]), C + c
        elif r < 0.55:                                  # Repeat: 몸체 [conv(in, C) → act → add(act, in)]
            nid = next(ids)
            body = [dict(id=nid + "c", type=cv, inputs=["in"], channels=C, kernel=3, stride=1),
                    dict(id=nid + "a", type="act", inputs=[nid + "c"], kind=rng.choice(acts)),
                    dict(id=nid + "s", type="add", inputs=[nid + "a", "in"])]
            nodes.append(dict(id=nid, type="repeat", times=rng.randint(1, 3), inputs=[cur], body={"nodes": body}))
            cur = nid
        elif r < 0.62:
            cur, size = add("pool2d" if image else "pool", cur, size=2), size + 1
        elif r < 0.7:
            cur = add("bn", cur)
        elif r < 0.75 and not image:
            cur = add("se", cur, ratio=2, act=rng.choice(["sigmoid", "relu"]))
        else:
            ch = rng.choice([2, 4, 8])
            cur, C = add("act", conv_node(cur, ch), kind=rng.choice(acts)), ch
        seen.append((cur, (C, size)))
    cur = add("global_avg2d" if image else "global_avg", cur)
    if rng.random() < 0.4:
        cur = add("act", add("dense", cur, units=rng.choice([4, 8])), kind=rng.choice(acts))
        if rng.random() < 0.5:
            cur = add("add", [cur, add("act", add("dense", cur, units=nodes[-2]["units"]), kind="relu")])
    add("output", cur)
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}] + nodes}


def _random_ok(seed, image):
    rng = random.Random(seed)
    L = 64 if image else 24
    for _ in range(20):
        g = random_graph(rng, image)
        c = check_graph(g, L, 2, image=(8, 8) if image else None)
        if c.ok:
            return g, c
    return None, None


@pytest.mark.parametrize("seed", range(12))
def test_random_graphs_convert(seed):
    g, c = _random_ok(seed, False)
    assert g is not None
    net = build_graph_model(g, 24, 2, seed=seed).eval()
    Xi = to_model_input(c.kind, _X(L=24, seed=seed))
    sim = _convert(net, Xi).simulate(Xi)
    assert sim.shape == (8, 2) and np.all(np.isfinite(sim))


@pytest.mark.parametrize("seed", range(6))
def test_random_image_graphs_convert(seed):
    g, c = _random_ok(seed, True)
    assert g is not None
    net = build_graph_model(g, 64, 2, seed=seed, image=(8, 8)).eval()
    Xi = to_model_input("image", _X(L=64, seed=seed), image=(8, 8))
    sim = _convert(net, Xi).simulate(Xi)
    assert sim.shape == (8, 2) and np.all(np.isfinite(sim))


def test_rnn_graph_builds():
    g = G(N("r", "gru", "input", hidden=8, layers=1), N("d", "dense", "r", units=4), N("a", "act", "d", kind="relu"),
          N("o", "output", "a"))
    c = check_graph(g, 20, 2)
    assert c.ok and c.kind == "rnn"
    net = build_graph_model(g, 20, 2)
    Xi = to_model_input("rnn", _X(L=20))
    assert _convert(net, Xi).simulate(Xi).shape == (8, 2)


def test_activations_use_node_ids():
    net = build_graph_model(far_concat(), 24, 3)
    vals = activations(net, to_model_input("cnn", _X(1))[0])
    assert {"a", "b", "c", "cat", "g", "o"} <= set(vals)


def test_output_linear_and_invalid():
    net = build_graph_model(dense_res(), 24, 3)
    assert output_linear(net).out_features == 3
    with pytest.raises(ValueError):
        build_graph_model(G(conv("c", "input", 0), N("o", "output", "c")), 20, 2)

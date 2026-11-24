"""블록 → torch 모델: 모양이 check와 같고, 검증을 통과한 조립은 만들어지고 앞으로 계산된다."""
import random
import warnings

import numpy as np
import pytest
import torch

from dlstudio.blocks import build_model, check, to_model_input

EXAMPLE = [
    {"type": "conv", "channels": 16, "kernel": 5},
    {"type": "act", "kind": "relu"},
    {"type": "repeat", "times": 3, "body": [
        {"type": "residual", "body": [{"type": "conv", "channels": 16, "kernel": 3}, {"type": "act", "kind": "relu"}]},
    ]},
    {"type": "global_avg"},
    {"type": "output"},
]
POLY = [{"type": "conv", "channels": 8, "kernel": 3}, {"type": "act", "kind": "relu"}, {"type": "global_avg"},
        {"type": "output"}]
LSTM = [{"type": "lstm", "hidden": 8, "layers": 1}, {"type": "dense", "units": 6}, {"type": "act", "kind": "tanh"},
        {"type": "output"}]
GRU = [{"type": "gru", "hidden": 6, "layers": 2}, {"type": "output"}]
RES_CH = [{"type": "conv", "channels": 4, "kernel": 3}, {"type": "residual", "body": [
    {"type": "conv", "channels": 8, "kernel": 3}, {"type": "bn"}, {"type": "act", "kind": "relu"}]},
    {"type": "global_avg"}, {"type": "output"}]
STRIDE = [{"type": "conv", "channels": 6, "kernel": 3, "stride": 2}, {"type": "act", "kind": "relu"},
          {"type": "pool", "size": 2}, {"type": "dense", "units": 8}, {"type": "act", "kind": "sigmoid"},
          {"type": "output"}]

CASES = {"example": (EXAMPLE, 1e-2), "relu_small": (POLY, 1e-6), "lstm": (LSTM, 1e-3), "gru": (GRU, 1e-3),
         "res_channels": (RES_CH, 1e-2), "stride_pool": (STRIDE, 1e-2)}


def _X(n=8, L=24, seed=0):
    return np.random.default_rng(seed).normal(scale=0.5, size=(n, L))


class _Run:
    """만든 모델을 그대로 앞으로 계산 (만들어지고 돌아가는지·모양 확인)."""

    def __init__(self, model):
        self.model = model.eval()

    def simulate(self, X):
        import torch
        with torch.no_grad():
            return self.model(torch.as_tensor(np.asarray(X), dtype=torch.float32)).double().numpy()


def _convert(model, Xi):
    return _Run(model)


@pytest.mark.parametrize("name", list(CASES))
def test_build_matches_shapes(name):
    blocks, _ = CASES[name]
    c = check(blocks, 24, 3)
    assert c.ok, c.issues
    m = build_model(blocks, 24, 3).eval()
    with torch.no_grad():
        out = m(torch.as_tensor(to_model_input(c.kind, _X(4)), dtype=torch.float32))
    assert tuple(out.shape) == (4,) + c.shapes[(len(blocks) - 1,)]


@pytest.mark.parametrize("name", list(CASES))
def test_converts_and_simulates(name):
    blocks, tol = CASES[name]
    c = check(blocks, 24, 3)
    m = build_model(blocks, 24, 3).eval()
    Xi = to_model_input(c.kind, _X())
    he = _convert(m, Xi)
    with torch.no_grad():
        want = m(torch.as_tensor(Xi, dtype=torch.float32)).double().numpy()
    assert np.max(np.abs(he.simulate(Xi) - want)) <= tol * max(1.0, np.max(np.abs(want)))


def _random_blocks(rng: random.Random) -> list:
    """규칙을 지키는 무작위 조립 (새 조각 포함: dilation·valid padding·LastTimestep·Dropout·Flatten·SimpleRNN·여러 활성화)."""
    if rng.random() < 0.3:
        t = rng.choice(["lstm", "gru", "simple_rnn"])
        out = [{"type": t, "hidden": rng.choice([4, 8]), "layers": rng.choice([1, 2])}]
        if rng.random() < 0.3:
            out.append({"type": "dropout", "rate": 0.2})
        if rng.random() < 0.5:
            out += [{"type": "dense", "units": rng.choice([4, 8])}, {"type": "act", "kind": rng.choice(["relu", "tanh", "gelu"])}]
        return out + [{"type": "output"}]
    acts = ["relu", "sigmoid", "leaky_relu", "gelu", "silu", "elu", "mish", "softplus"]
    layer = lambda: rng.choice([                                  # tanh는 위치 규칙이 있어 머리(dense 뒤)에만
        {"type": "conv", "channels": rng.choice([2, 4, 8]), "kernel": rng.choice([1, 3, 5]), "dilation": rng.choice([1, 1, 2, 4])},
        {"type": "act", "kind": rng.choice(acts)},
        {"type": "bn"},
        {"type": "dropout", "rate": 0.1},
    ])
    out = []
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        if r < 0.15:
            out.append({"type": "residual", "body": [layer() for _ in range(rng.randint(1, 2))]})
        elif r < 0.25:
            out.append({"type": "repeat", "times": rng.randint(1, 3), "body": [layer() for _ in range(rng.randint(1, 2))]})
        elif r < 0.32:
            out.append({"type": "pool", "size": 2})
        elif r < 0.40:
            out.append({"type": "conv", "channels": rng.choice([2, 4]), "kernel": 3, "stride": 2})
        elif r < 0.46:
            out.append({"type": "conv", "channels": rng.choice([2, 4]), "kernel": 3, "padding": "valid"})
        elif r < 0.60:
            out.append({"type": "branch", "branches": [[layer() for _ in range(rng.randint(0, 2))]
                                                       for _ in range(rng.randint(2, 3))]})
        elif r < 0.53 + 0.07:
            out += [{"type": "conv", "channels": 4, "kernel": 3},
                    {"type": "se", "ratio": 2, "act": rng.choice(["sigmoid", "relu"])}]
        else:
            out.append(layer())
    head = rng.random()
    if head < 0.35:
        out.append({"type": "global_avg"})
    elif head < 0.55:
        out.append({"type": "last_step"})
    elif head < 0.75:
        out += [{"type": "flatten"}, {"type": "dense", "units": rng.choice([4, 8])}, {"type": "act", "kind": rng.choice(acts)}]
    else:
        out += [{"type": "dense", "units": rng.choice([4, 8])}, {"type": "act", "kind": rng.choice(acts + ["tanh"])}]
    return out + [{"type": "output"}]


@pytest.mark.parametrize("seed", range(20))
def test_random_valid_assemblies_convert(seed):
    blocks = _random_blocks(random.Random(seed))
    c = check(blocks, 32, 2)
    assert c.ok, (blocks, c.issues)
    m = build_model(blocks, 32, 2, seed=seed).eval()
    Xi = to_model_input(c.kind, _X(L=32, seed=seed))
    he = _convert(m, Xi)
    assert he.simulate(Xi).shape == (8, 2)


def test_invalid_raises():
    with pytest.raises(ValueError):
        build_model([{"type": "conv", "channels": 0, "kernel": 3}, {"type": "output"}], 20, 2)


def test_block_modules_cover_paths():
    m = build_model(EXAMPLE, 24, 2)
    assert {"0", "1", "2", "3", "4"} <= set(m.block_modules.keys())


def test_to_model_input_layouts():
    X = _X(3, 10)
    assert to_model_input("cnn", X).shape == (3, 1, 10)
    assert to_model_input("rnn", X).shape == (3, 10, 1)


TCN = [{"type": "conv", "channels": 6, "kernel": 3, "dilation": 1}, {"type": "act", "kind": "gelu"},
       {"type": "residual", "body": [{"type": "conv", "channels": 6, "kernel": 3, "dilation": 2},
                                     {"type": "act", "kind": "silu"}, {"type": "dropout", "rate": 0.1}]},
       {"type": "conv", "channels": 6, "kernel": 3, "dilation": 4, "padding": "valid"}, {"type": "last_step"},
       {"type": "output"}]
DEPTHWISE = [{"type": "conv", "channels": 8, "kernel": 3}, {"type": "act", "kind": "gelu"},
             {"type": "conv", "channels": 8, "kernel": 5, "groups": 8}, {"type": "act", "kind": "leaky_relu"},
             {"type": "global_avg"}, {"type": "output"}]
ACTS = [{"type": "conv", "channels": 4, "kernel": 3}, {"type": "act", "kind": "elu"}, {"type": "conv", "channels": 4, "kernel": 3},
        {"type": "act", "kind": "mish"}, {"type": "conv", "channels": 4, "kernel": 3}, {"type": "act", "kind": "softplus"},
        {"type": "flatten"}, {"type": "dense", "units": 8}, {"type": "act", "kind": "relu"}, {"type": "output"}]
SRNN = [{"type": "simple_rnn", "hidden": 6, "layers": 2}, {"type": "dropout", "rate": 0.2}, {"type": "output"}]


@pytest.mark.parametrize("name,blocks,tol", [("tcn", TCN, 5e-2), ("depthwise", DEPTHWISE, 5e-2), ("acts", ACTS, 5e-2),
                                             ("simple_rnn", SRNN, 1e-2)])
def test_more_blocks_convert(name, blocks, tol):
    c = check(blocks, 24, 3)
    assert c.ok, c.issues
    m = build_model(blocks, 24, 3).eval()
    Xi = to_model_input(c.kind, _X())
    he = _convert(m, Xi)
    with torch.no_grad():
        want = m(torch.as_tensor(Xi, dtype=torch.float32)).double().numpy()
    assert want.shape == (8, 3)
    assert np.max(np.abs(he.simulate(Xi) - want)) <= tol * max(1.0, np.max(np.abs(want)))


SE_NET = [{"type": "conv", "channels": 8, "kernel": 3}, {"type": "act", "kind": "relu"},
          {"type": "se", "ratio": 4, "act": "relu"},
          {"type": "residual", "body": [{"type": "conv", "channels": 8, "kernel": 3}, {"type": "se", "ratio": 2, "act": "relu"}]},
          {"type": "global_avg"}, {"type": "output"}]


def test_se_block_converts():
    c = check(SE_NET, 24, 3)
    assert c.ok, c.issues
    m = build_model(SE_NET, 24, 3).eval()
    Xi = to_model_input("cnn", _X())
    he = _convert(m, Xi)
    with torch.no_grad():
        want = m(torch.as_tensor(Xi, dtype=torch.float32)).double().numpy()
    assert np.max(np.abs(he.simulate(Xi) - want)) <= 5e-2 * max(1.0, np.max(np.abs(want)))


INCEPTION = [{"type": "conv", "channels": 8, "kernel": 1},
             {"type": "branch", "branches": [
                 [{"type": "conv", "channels": 4, "kernel": 3}, {"type": "act", "kind": "relu"}],
                 [{"type": "conv", "channels": 4, "kernel": 1}, {"type": "conv", "channels": 6, "kernel": 5}],
                 []]},
             {"type": "bn"}, {"type": "act", "kind": "relu"}, {"type": "global_avg"}, {"type": "output"}]


def test_branch_concat_converts():
    c = check(INCEPTION, 24, 3)
    assert c.ok, c.issues
    m = build_model(INCEPTION, 24, 3).eval()
    assert "1.1.1" in m.block_modules
    Xi = to_model_input("cnn", _X())
    he = _convert(m, Xi)
    with torch.no_grad():
        want = m(torch.as_tensor(Xi, dtype=torch.float32)).double().numpy()
    assert np.max(np.abs(he.simulate(Xi) - want)) <= 5e-2 * max(1.0, np.max(np.abs(want)))


IMG_MIX = [{"type": "conv2d", "channels": 8, "kernel": 3}, {"type": "bn"}, {"type": "act", "kind": "relu"},
           {"type": "residual", "body": [{"type": "conv2d", "channels": 4, "kernel": 3}]},
           {"type": "branch", "branches": [[{"type": "conv2d", "channels": 2, "kernel": 1}], []]},
           {"type": "pool2d", "size": 2}, {"type": "flatten"}, {"type": "dense", "units": 8}, {"type": "act", "kind": "relu"},
           {"type": "output"}]


def test_image_model_runs_and_converts():
    c = check(IMG_MIX, 64, 3, image=(8, 8))
    assert c.ok, c.issues
    m = build_model(IMG_MIX, 64, 3, image=(8, 8)).eval()
    Xi = to_model_input("image", _X(L=64), image=(8, 8))
    assert Xi.shape == (8, 1, 8, 8)
    with torch.no_grad():
        want = m(torch.as_tensor(Xi, dtype=torch.float32)).double().numpy()
    assert want.shape == (8, 3)
    he = _convert(m, Xi)
    assert np.max(np.abs(he.simulate(Xi) - want)) <= 1e-5 * max(1.0, np.max(np.abs(want)))


def _random_image_blocks(rng: random.Random) -> list:
    """규칙을 지키는 무작위 2D 조립: 16×16에서 크기를 줄이는 층(pool2d·stride 2·valid)은 합쳐 3번까지."""
    acts = ["relu", "sigmoid", "gelu", "silu"]
    layer = lambda: rng.choice([
        {"type": "conv2d", "channels": rng.choice([2, 4, 8]), "kernel": rng.choice([1, 3, 5])},
        {"type": "act", "kind": rng.choice(acts)}, {"type": "bn"}, {"type": "dropout", "rate": 0.1}])
    out, shrink = [], 0
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        if r < 0.15:
            out.append({"type": "residual", "body": [layer() for _ in range(rng.randint(1, 2))]})
        elif r < 0.25:
            out.append({"type": "repeat", "times": rng.randint(1, 3), "body": [layer() for _ in range(rng.randint(1, 2))]})
        elif r < 0.35 and shrink < 3:
            out.append({"type": "pool2d", "size": 2}); shrink += 1
        elif r < 0.43 and shrink < 3:
            out.append({"type": "conv2d", "channels": rng.choice([2, 4]), "kernel": 3, "stride": 2}); shrink += 1
        elif r < 0.48 and shrink < 3:
            out.append({"type": "conv2d", "channels": rng.choice([2, 4]), "kernel": 3, "padding": "valid"}); shrink += 1
        elif r < 0.60:
            out.append({"type": "branch", "branches": [[layer() for _ in range(rng.randint(0, 2))] for _ in range(rng.randint(2, 3))]})
        else:
            out.append(layer())
    if shrink < 2 or rng.random() < 0.5:                         # 16×16에서 덜 줄였으면 펼치면 한도(2,048)를 넘을 수 있음
        out.append({"type": "global_avg2d"})
    else:
        out += [{"type": "flatten"}, {"type": "dense", "units": rng.choice([4, 8])}, {"type": "act", "kind": rng.choice(acts)}]
    return out + [{"type": "output"}]


@pytest.mark.parametrize("seed", range(20))
def test_random_image_assemblies_convert(seed):
    blocks = _random_image_blocks(random.Random(seed))
    c = check(blocks, 256, 2, image=(16, 16))
    assert c.ok, (blocks, c.issues)
    m = build_model(blocks, 256, 2, seed=seed, image=(16, 16)).eval()
    Xi = to_model_input("image", _X(L=256, seed=seed), image=(16, 16))
    assert _convert(m, Xi).simulate(Xi).shape == (8, 2)


@pytest.mark.parametrize("blocks,image,L", [
    ([{"type": "conv", "channels": 8, "kernel": 3},
      {"type": "repeat", "times": 2, "body": [{"type": "act", "kind": "relu"}, {"type": "conv", "channels": 3, "kernel": 3}]},
      {"type": "global_avg"}, {"type": "output"}], None, 24),
    ([{"type": "conv2d", "channels": 8, "kernel": 3},
      {"type": "repeat", "times": 2, "body": [{"type": "pool2d", "size": 2}, {"type": "conv2d", "channels": 16, "kernel": 3},
                                              {"type": "act", "kind": "relu"}]},
      {"type": "global_avg2d"}, {"type": "output"}], (16, 16), 256),
])
def test_repeat_body_that_changes_shape_builds_and_converts(blocks, image, L):
    """Repeat 몸체가 채널·크기를 바꾸면 반복마다 모양이 다름: 검사를 통과하면 만들고 변환까지 돼야 함."""
    c = check(blocks, L, 2, image=image)
    assert c.ok, c.issues
    m = build_model(blocks, L, 2, image=image).eval()
    Xi = to_model_input(c.kind, _X(L=L), image=image)
    assert _convert(m, Xi).simulate(Xi).shape == (8, 2)

"""블록 JSON 검증과 블록마다 출력 모양 (설계 문서 4·5절)."""
from blocklearning.blocks import check

EXAMPLE = [
    {"type": "conv", "channels": 16, "kernel": 5},
    {"type": "act", "kind": "relu"},
    {"type": "repeat", "times": 3, "body": [
        {"type": "residual", "body": [
            {"type": "conv", "channels": 16, "kernel": 3},
            {"type": "act", "kind": "relu"},
        ]},
    ]},
    {"type": "global_avg"},
    {"type": "output"},
]


def _paths(c):
    return [i.path for i in c.issues]


def test_example_cnn_shapes():
    c = check(EXAMPLE, 140, 2)
    assert c.ok, c.issues
    assert c.kind == "cnn"
    assert c.shapes[(0,)] == (16, 140)
    assert c.shapes[(2, 0, 0)] == (16, 140)
    assert c.shapes[(3,)] == (16,)
    assert c.shapes[(4,)] == (2,)


def test_stride_and_pool_shapes():
    c = check([{"type": "conv", "channels": 8, "kernel": 3, "stride": 2}, {"type": "pool", "size": 4},
               {"type": "output"}], 140, 2)
    assert c.ok, c.issues
    assert c.shapes[(0,)] == (8, 70)
    assert c.shapes[(1,)] == (8, 17)


def test_rnn_shapes():
    c = check([{"type": "lstm", "hidden": 32, "layers": 1}, {"type": "dense", "units": 16},
               {"type": "act", "kind": "tanh"}, {"type": "output"}], 50, 3)
    assert c.ok, c.issues
    assert c.kind == "rnn"
    assert c.shapes[(0,)] == (32,) and c.shapes[(1,)] == (16,) and c.shapes[(3,)] == (3,)


def test_rnn_after_conv_allowed():
    """순환 층은 (채널, 길이) 뒤 어디에나: 시점마다 채널 수만큼 값을 읽음. 숫자 묶음 뒤에는 안 됨."""
    c = check([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "lstm", "hidden": 8, "layers": 1},
               {"type": "output"}], 20, 2)
    assert c.ok and c.kind == "cnn" and c.shapes[(1,)] == (8,)
    c = check([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "global_avg"}, {"type": "lstm", "hidden": 8, "layers": 1},
               {"type": "output"}], 20, 2)
    assert (2,) in _paths(c)
    c = check([{"type": "lstm", "hidden": 8, "layers": 1, "return_sequences": True, "bidirectional": True},
               {"type": "conv", "channels": 4, "kernel": 3}, {"type": "global_avg"}, {"type": "output"}], 20, 2)
    assert c.ok and c.kind == "rnn" and c.shapes[(0,)] == (16, 20)


def test_cnn_after_rnn_rejected():
    c = check([{"type": "gru", "hidden": 8, "layers": 1}, {"type": "conv", "channels": 4, "kernel": 3},
               {"type": "output"}], 20, 2)
    assert (1,) in _paths(c)


def test_pool_in_residual_rejected():
    c = check([{"type": "residual", "body": [{"type": "pool", "size": 2}]}, {"type": "global_avg"},
               {"type": "output"}], 20, 2)
    assert (0, 0) in _paths(c)


def test_output_missing_or_not_last():
    assert not check([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "global_avg"}], 20, 2).ok
    assert not check([{"type": "output"}, {"type": "act", "kind": "relu"}], 20, 2).ok
    assert not check([{"type": "global_avg"}, {"type": "output"}, {"type": "output"}], 20, 2).ok


def test_bad_values_rejected():
    c = check([{"type": "conv", "channels": 0, "kernel": 4}, {"type": "global_avg"}, {"type": "output"}], 20, 2)
    assert (0,) in _paths(c)
    c = check([{"type": "act", "kind": "no_such_fn"}, {"type": "global_avg"}, {"type": "output"}], 20, 2)
    assert (0,) in _paths(c)


def test_unknown_type():
    c = check([{"type": "maxpool"}, {"type": "output"}], 20, 2)
    assert (0,) in _paths(c)


def test_empty_body():
    c = check([{"type": "repeat", "times": 2, "body": []}, {"type": "global_avg"}, {"type": "output"}], 20, 2)
    assert (0,) in _paths(c)


def test_too_wide_rejected():
    c = check([{"type": "global_avg"}, {"type": "dense", "units": 4096}, {"type": "output"}], 20, 2)
    assert (1,) in _paths(c)                                     # units 한도 1,024
    c = check([{"type": "conv", "channels": 64, "kernel": 3}, {"type": "dense", "units": 8},
               {"type": "output"}], 2000, 2)                    # 펼치면 64 × 2,000 = 128,000 > 65,536
    assert (1,) in _paths(c)
    c = check([{"type": "conv", "channels": 32, "kernel": 3}, {"type": "dense", "units": 8},
               {"type": "output"}], 140, 2)                     # 4,480은 됨
    assert c.ok, c.issues


def test_issue_messages_are_plain_korean():
    c = check([{"type": "conv", "channels": 0, "kernel": 4}, {"type": "output"}], 20, 2)
    assert all(i.message and i.hint for i in c.issues)


def test_tanh_anywhere():
    for blocks in ([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "act", "kind": "tanh"},
                    {"type": "conv", "channels": 4, "kernel": 3}, {"type": "global_avg"}, {"type": "output"}],
                   [{"type": "lstm", "hidden": 4, "layers": 1}, {"type": "act", "kind": "tanh"}, {"type": "output"}]):
        assert check(blocks, 20, 2).ok


# ---------------------------------------------------------------- 더 넣은 조각 (Conv1D 옵션, LastTimestep, Dropout, Flatten, SimpleRNN)

def test_conv_dilation_padding_groups_shapes():
    c = check([{"type": "conv", "channels": 8, "kernel": 3, "dilation": 4, "padding": "valid"},
               {"type": "conv", "channels": 8, "kernel": 3, "dilation": 2},
               {"type": "conv", "channels": 8, "kernel": 3, "groups": 8},
               {"type": "global_avg"}, {"type": "output"}], 140, 2)
    assert c.ok, c.issues
    assert c.shapes[(0,)] == (8, 132) and c.shapes[(1,)] == (8, 132) and c.shapes[(2,)] == (8, 132)


def test_conv_groups_must_divide_channels():
    c = check([{"type": "conv", "channels": 6, "kernel": 3}, {"type": "conv", "channels": 6, "kernel": 3, "groups": 4},
               {"type": "global_avg"}, {"type": "output"}], 20, 2)
    assert (1,) in _paths(c)


def test_valid_conv_not_allowed_in_residual():
    c = check([{"type": "residual", "body": [{"type": "conv", "channels": 1, "kernel": 3, "padding": "valid"}]},
               {"type": "global_avg"}, {"type": "output"}], 20, 2)
    assert (0, 0) in _paths(c)


def test_last_step_dropout_flatten_shapes():
    c = check([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "dropout", "rate": 0.2},
               {"type": "last_step"}, {"type": "output"}], 20, 2)
    assert c.ok, c.issues
    assert c.shapes[(1,)] == (4, 20) and c.shapes[(2,)] == (4,)
    c = check([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "flatten"}, {"type": "dense", "units": 8},
               {"type": "output"}], 20, 2)
    assert c.ok and c.shapes[(1,)] == (80,)
    assert not check([{"type": "dropout", "rate": 1.5}, {"type": "global_avg"}, {"type": "output"}], 20, 2).ok


def test_new_activations_and_simple_rnn():
    for kind in ("leaky_relu", "gelu", "silu", "elu", "mish", "softplus"):
        c = check([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "act", "kind": kind}, {"type": "global_avg"},
                   {"type": "output"}], 20, 2)
        assert c.ok, (kind, c.issues)
    c = check([{"type": "simple_rnn", "hidden": 8, "layers": 1}, {"type": "output"}], 20, 2)
    assert c.ok and c.kind == "rnn" and c.shapes[(0,)] == (8,)


def test_se_block_keeps_shape_and_checks_ratio():
    c = check([{"type": "conv", "channels": 16, "kernel": 3}, {"type": "se", "ratio": 4},
               {"type": "residual", "body": [{"type": "conv", "channels": 16, "kernel": 3}, {"type": "se", "ratio": 2}]},
               {"type": "global_avg"}, {"type": "output"}], 30, 2)
    assert c.ok, c.issues
    assert c.shapes[(1,)] == (16, 30) and c.shapes[(2, 1)] == (16, 30)
    c = check([{"type": "conv", "channels": 2, "kernel": 3}, {"type": "se", "ratio": 4}, {"type": "global_avg"},
               {"type": "output"}], 30, 2)
    assert (1,) in _paths(c)                                     # 2채널 ÷ 4 = 0


INCEPTION = [{"type": "conv", "channels": 8, "kernel": 1},
             {"type": "branch", "branches": [
                 [{"type": "conv", "channels": 4, "kernel": 3}],
                 [{"type": "conv", "channels": 4, "kernel": 1}, {"type": "conv", "channels": 6, "kernel": 5}],
                 []]},
             {"type": "act", "kind": "relu"}, {"type": "global_avg"}, {"type": "output"}]


def test_branch_concat_shapes():
    c = check(INCEPTION, 30, 2)
    assert c.ok, c.issues
    assert c.shapes[(1, 0, 0)] == (4, 30) and c.shapes[(1, 1, 1)] == (6, 30)
    assert c.shapes[(1,)] == (4 + 6 + 8, 30)                    # 빈 갈래는 들어온 8채널 그대로


def test_branch_lanes_must_keep_length():
    bad = [{"type": "conv", "channels": 4, "kernel": 1},
           {"type": "branch", "branches": [[{"type": "pool", "size": 2}], [{"type": "conv", "channels": 4, "kernel": 3}]]},
           {"type": "global_avg"}, {"type": "output"}]
    assert (1, 0, 0) in _paths(check(bad, 30, 2))
    one = [{"type": "branch", "branches": [[{"type": "conv", "channels": 4, "kernel": 3}]]}, {"type": "global_avg"},
           {"type": "output"}]
    assert (0,) in _paths(check(one, 30, 2))                     # 갈래는 2개 이상


IMG = [{"type": "conv2d", "channels": 8, "kernel": 3}, {"type": "act", "kind": "relu"}, {"type": "pool2d", "size": 2},
       {"type": "conv2d", "channels": 16, "kernel": 3, "stride": 2}, {"type": "global_avg2d"}, {"type": "output"}]


def test_image_shapes():
    c = check(IMG, 64, 10, image=(8, 8))
    assert c.ok and c.kind == "image", c.issues
    assert c.shapes[(0,)] == (8, 8, 8) and c.shapes[(2,)] == (8, 4, 4) and c.shapes[(3,)] == (16, 2, 2) and c.shapes[(4,)] == (16,)
    assert c.shapes[(5,)] == (10,)


def test_image_rules():
    bad = lambda blocks, image=(8, 8): not check(blocks, 64, 10, image=image).ok
    assert bad([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "output"}])               # 1D 블록 금지
    assert bad([{"type": "lstm", "hidden": 4}, {"type": "output"}])                               # 순환 층 금지
    assert bad([{"type": "conv2d", "channels": 4, "kernel": 3}, {"type": "output"}], image=None)  # 시계열에 2D 금지
    assert bad([{"type": "residual", "body": [{"type": "pool2d", "size": 2}]}, {"type": "output"}])
    assert bad([{"type": "residual", "body": [{"type": "conv2d", "channels": 1, "kernel": 3, "stride": 2}]}, {"type": "output"}])
    assert not bad([{"type": "residual", "body": [{"type": "conv2d", "channels": 4, "kernel": 3}]}, {"type": "flatten"},
                    {"type": "dense", "units": 4}, {"type": "output"}])
    c = check([{"type": "pool2d", "size": 2}] * 4 + [{"type": "output"}], 64, 10, image=(8, 8))   # 8→4→2→1→0
    assert not c.ok and any("크기가 0" in i.message for i in c.issues)

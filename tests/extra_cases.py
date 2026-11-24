"""추가 블록마다 작은 모델 (이름, 그래프, 길이, 클래스 수, 이미지). test_codegen·test_engine·test_layers_extra가 같이 씀.

줄(시계열) 길이 24 · 그림 8×8. 앞에 채널을 4개로 늘리는 Conv를 두고, 블록 뒤에는 Flatten → Output."""


def _g(nodes):
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}] + nodes}


def chain(kind, blocks, pre=True, ch=4):
    """Input → (Conv 4채널) → 블록들 → Flatten → Output."""
    nodes, last = [], "input"
    if pre:
        t = "conv2d" if kind == "i" else "conv"
        nodes.append({"id": "pre", "type": t, "channels": ch, "kernel": 3, "stride": 1, "inputs": ["input"]})
        last = "pre"
    for i, (t, kw) in enumerate(blocks):
        nid = f"b{i}"
        nodes.append(dict({"id": nid, "type": t, "inputs": [last]}, **kw))
        last = nid
    nodes += [{"id": "fl", "type": "flatten", "inputs": [last]}, {"id": "o", "type": "output", "inputs": ["fl"]}]
    return _g(nodes)


S, I = (24, None), (64, (8, 8))


def case(name, kind, blocks, pre=True, ch=4, L=None):
    length, image = (S if kind == "s" else I)
    return (f"x_{name}", chain(kind, blocks, pre, ch), L or length, 3, image)


def merge_case(t):
    nodes = [{"id": "a", "type": "conv", "channels": 4, "kernel": 3, "stride": 1, "inputs": ["input"]},
             {"id": "b", "type": "conv", "channels": 4, "kernel": 5, "stride": 1, "inputs": ["input"]},
             {"id": "m", "type": t, "inputs": ["a", "b"]},
             {"id": "g", "type": "global_avg", "inputs": ["m"]}, {"id": "o", "type": "output", "inputs": ["g"]}]
    return (f"x_{t}", _g(nodes), 24, 3, None)


def cross_case():
    nodes = [{"id": "a", "type": "conv", "channels": 4, "kernel": 3, "stride": 1, "inputs": ["input"]},
             {"id": "b", "type": "conv", "channels": 6, "kernel": 3, "stride": 2, "inputs": ["input"]},
             {"id": "x", "type": "cross_attn", "heads": 2, "inputs": ["a", "b"]},
             {"id": "g", "type": "global_avg", "inputs": ["x"]}, {"id": "o", "type": "output", "inputs": ["g"]}]
    return ("x_cross_attn", _g(nodes), 24, 3, None)


def extra_cases():
    c = [
        case("glu", "s", [("glu", {"units": 6})]),
        case("swiglu_img", "i", [("glu", {"units": 5, "gate": "silu"})]),
        case("glu_vec", "s", [("global_avg", {}), ("glu", {"units": 3})]),
        case("dw_conv", "s", [("dw_conv", {"kernel": 5, "multiplier": 2})]),
        case("dw_conv2d", "i", [("dw_conv2d", {"kernel": 3})]),
        case("sep_conv", "s", [("sep_conv", {"channels": 5, "kernel": 5})]),
        case("sep_conv2d", "i", [("sep_conv2d", {"channels": 5, "kernel": 3})]),
        case("causal_conv", "s", [("conv", {"channels": 3, "kernel": 3, "stride": 1, "dilation": 2, "padding": "causal"})]),
        case("conv2d_dil_groups", "i", [("conv2d", {"channels": 6, "kernel": 3, "stride": 1, "dilation": 2, "groups": 2})]),
        case("max_pool", "s", [("max_pool", {"size": 2})]),
        case("max_pool2d", "i", [("max_pool2d", {})]),
        case("global_max", "s", [("global_max", {})]),
        case("global_max2d", "i", [("global_max2d", {})]),
        case("adaptive_avg", "s", [("adaptive_pool", {"size": 5, "mode": "avg"})]),
        case("adaptive_max2d", "i", [("adaptive_pool", {"size": 3, "mode": "max"})]),
        case("first_step", "s", [("first_step", {})]),
        case("attn_pool", "s", [("attn_pool", {})]),
        case("attn_pool2d", "i", [("attn_pool", {})]),
        case("upsample_nearest", "s", [("upsample", {"mode": "nearest"})]),
        case("upsample_linear", "s", [("upsample", {"mode": "linear"})]),
        case("upsample_bilinear", "i", [("upsample", {"mode": "linear"})]),
        case("upsample_nearest2d", "i", [("upsample", {})]),
        case("pixel_shuffle", "s", [("pixel_shuffle", {})]),
        case("pixel_shuffle2d", "i", [("pixel_shuffle", {})], ch=8),
        case("conv_t", "s", [("conv_t", {"channels": 3, "kernel": 3, "stride": 2})]),
        case("conv_t_k4", "s", [("conv_t", {"channels": 3, "kernel": 4, "stride": 2})]),
        case("conv_t_s1", "s", [("conv_t", {"channels": 3, "kernel": 3, "stride": 1})]),
        case("conv_t2d", "i", [("conv_t2d", {"channels": 3, "kernel": 2, "stride": 2})]),
        case("conv_t2d_k3", "i", [("conv_t2d", {"channels": 3, "kernel": 3, "stride": 2})]),
        case("se2d", "i", [("se", {"ratio": 2, "act": "relu"})]),
        case("eca", "s", [("eca", {"kernel": 3})]),
        case("eca2d", "i", [("eca", {"kernel": 3})]),
        case("cbam", "s", [("cbam", {"ratio": 2, "kernel": 3})]),
        case("cbam2d", "i", [("cbam", {"ratio": 2, "kernel": 7})]),
        case("spatial_attn2d", "i", [("spatial_attn", {"kernel": 5})]),
        case("mha", "s", [("mha", {"heads": 2})]),
        case("mha_causal_rope", "s", [("mha", {"heads": 2, "causal": True, "pos": "rope"})]),
        case("mha2d", "i", [("mha", {"heads": 4})]),
        case("linear_attn", "s", [("linear_attn", {"heads": 2})]),
        case("nonlocal2d", "i", [("nonlocal", {})]),
        cross_case(),
        case("pos_sin", "s", [("pos_enc", {"mode": "sin"})], ch=5),
        case("pos_learned", "s", [("pos_enc", {"mode": "learned"})]),
        case("vit", "i", [("patch_embed", {"patch": 2, "dim": 6}), ("cls_token", {}), ("pos_enc", {}),
                          ("layernorm", {}), ("mha", {"heads": 2}), ("first_step", {})], pre=False),
        case("layernorm", "s", [("layernorm", {})]),
        case("layernorm2d", "i", [("layernorm", {})]),
        case("layernorm_vec", "s", [("global_avg", {}), ("layernorm", {})]),
        case("rmsnorm", "s", [("rmsnorm", {})]),
        case("groupnorm", "s", [("groupnorm", {"groups": 2})]),
        case("instancenorm2d", "i", [("instancenorm", {})]),
        case("prelu", "s", [("prelu", {})]),
        case("acts", "s", [("act", {"kind": "selu"}), ("act", {"kind": "relu6"}), ("act", {"kind": "hardswish"}),
                           ("act", {"kind": "hardsigmoid"}), ("act", {"kind": "softmax"})]),
        case("spatial_dropout", "s", [("spatial_dropout", {"rate": 0.3})]),
        case("spatial_dropout2d", "i", [("spatial_dropout", {"rate": 0.3})]),
        case("drop_path", "s", [("drop_path", {"rate": 0.2})]),
        case("gaussian_noise", "s", [("gaussian_noise", {"std": 0.1})]),
        merge_case("multiply"), merge_case("average"), merge_case("maximum"),
        case("slice", "s", [("slice", {"start": 1, "count": 2})]),
        case("reshape", "s", [("reshape", {"channels": 8})]),
        case("permute", "s", [("permute", {})]),
        case("permute2d", "i", [("permute", {})]),
        case("to_seq", "i", [("to_seq", {}), ("conv", {"channels": 3, "kernel": 3, "stride": 1})]),
        case("crop", "s", [("crop", {"amount": 2})]),
        case("crop2d", "i", [("crop", {"amount": 1})]),
        case("pad", "s", [("pad", {"amount": 2})]),
        case("pad2d", "i", [("pad", {"amount": 1})]),
        case("embedding", "s", [("embedding", {"bins": 16, "dim": 5})], pre=False),
        case("spec_log", "s", [("spectrogram", {"n_fft": 16, "hop": 4, "mode": "log"}),
                               ("conv2d", {"channels": 3, "kernel": 3, "stride": 1})], pre=False, L=48),
        case("spec_mel", "s", [("spectrogram", {"n_fft": 32, "hop": 8, "mode": "mel", "n_mels": 8})], pre=False, L=64),
        case("spec_mfcc", "s", [("spectrogram", {"n_fft": 32, "hop": 8, "mode": "mfcc", "n_mels": 16, "n_mfcc": 8})], pre=False, L=64),
        case("aug_series", "s", [("rand_flip", {}), ("rand_shift", {"frac": 0.1}), ("rand_scale", {"frac": 0.2}),
                                 ("rand_stretch", {"frac": 0.1}), ("conv", {"channels": 3, "kernel": 3, "stride": 1})], pre=False),
        case("aug_image", "i", [("rand_flip", {"dir": "hv"}), ("rand_rotate", {"degrees": 10}), ("rand_shift", {"frac": 0.1}),
                                ("rand_zoom", {"frac": 0.1}), ("conv2d", {"channels": 3, "kernel": 3, "stride": 1})], pre=False),
        case("lstm_bi_seq", "s", [("lstm", {"hidden": 3, "layers": 2, "bidirectional": True, "return_sequences": True})]),
        case("gru_seq_lstm", "s", [("gru", {"hidden": 4, "layers": 1, "return_sequences": True}),
                                   ("lstm", {"hidden": 3, "layers": 1})]),
        case("rnn_bi_last", "s", [("simple_rnn", {"hidden": 3, "layers": 1, "bidirectional": True})]),
        case("lstm_after_conv", "s", [("lstm", {"hidden": 5, "layers": 1})]),
    ]
    return c

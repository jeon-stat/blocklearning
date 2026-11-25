"""그래프 → 파이썬 코드 (PyTorch · TensorFlow Keras).

화면에서 이은 모델을 사람이 읽고 고칠 수 있는 코드로 바꾼다. 층 하나가 코드 한 줄(변수 하나)이 되고, 줄마다 화면 이름과 출력 모양을 주석으로 단다.
Repeat 묶음은 반복마다 펼친다 (이름: <층>_r<반복 번호>, 예: conv_r0 · conv_r1).

- PyTorch: studio 모델과 계산이 같다. torch_state_dict()로 studio에서 학습한 가중치를 생성 코드의 이름으로 바꿔 그대로 불러올 수 있다.
- Keras: 같은 구조 (채널이 마지막인 모양). 가중치는 옮기지 않으므로 TF에서 새로 학습한다.
"""
from __future__ import annotations

import keyword
from dataclasses import dataclass, field
from typing import Optional

from dlstudio import layers as X
from dlstudio.blocks import RNN_TYPES
from dlstudio.graph import BODY_IN, INPUT, _Checker, _config, _key

FRAMEWORKS = ("pytorch", "tensorflow")
RESERVED = {"x", "self", "torch", "nn", "F", "tf", "keras", "layers", "model", "np", "h", "s"}


@dataclass
class Step:
    """펼친 층 하나: 코드 이름, 종류, 설정, 입력 이름들, 들어오는/나가는 모양(배치 제외, 채널 먼저), studio 가중치 키 앞부분."""
    name: str
    type: str
    cfg: dict
    inputs: list
    in_shape: tuple
    out_shape: tuple
    key: str                       # 예: "mods.m_b0" · "mods.m_R.1.m_c"
    node: str = ""                 # 그래프의 노드 id (Repeat 안이면 안쪽 id)
    in_shapes: list = field(default_factory=list)      # 입력마다 모양 (입력이 둘인 층)


@dataclass
class Plan:
    steps: list = field(default_factory=list)
    kind: str = "cnn"                # cnn | rnn | image
    in_shape: tuple = ()
    out: str = ""
    n_out: int = 1
    aliases: dict = field(default_factory=dict)      # Repeat 노드 id → 마지막 반복의 출구 이름


def _ident(nid: str, used: set) -> str:
    n = nid if nid.isidentifier() and not keyword.iskeyword(nid) and nid not in RESERVED else "n_" + nid
    base, k = n, 2
    while n in used:
        n, k = f"{base}_{k}", k + 1
    used.add(n)
    return n


# 생성 코드의 변수 이름: 층 종류 + 번호 (conv1, relu1, pool1 …). 가중치 이름(key)과는 따로라 바꿔도 저장한 프로젝트와 상관없음
PREFIX = {"conv": "conv", "conv2d": "conv", "pool": "pool", "pool2d": "pool", "global_avg": "gap", "global_avg2d": "gap",
          "last_step": "last", "bn": "bn", "dropout": "drop", "flatten": "flat", "dense": "fc", "lstm": "lstm", "gru": "gru",
          "simple_rnn": "rnn", "se": "se", "add": "add", "concat": "cat",
          "glu": "glu", "dw_conv": "dwconv", "dw_conv2d": "dwconv", "sep_conv": "sepconv", "sep_conv2d": "sepconv",
          "max_pool": "maxpool", "max_pool2d": "maxpool", "global_max": "gmp", "global_max2d": "gmp", "adaptive_pool": "apool",
          "first_step": "first", "attn_pool": "attnpool", "upsample": "up", "pixel_shuffle": "shuffle", "conv_t": "convt",
          "conv_t2d": "convt", "eca": "eca", "cbam": "cbam", "spatial_attn": "spattn", "mha": "attn", "linear_attn": "linattn",
          "nonlocal": "nonlocal", "cross_attn": "xattn", "pos_enc": "pos", "patch_embed": "patch", "cls_token": "cls",
          "layernorm": "ln", "rmsnorm": "rms", "groupnorm": "gn", "instancenorm": "inorm", "prelu": "prelu",
          "spatial_dropout": "sdrop", "drop_path": "droppath", "gaussian_noise": "noise", "slice": "split",
          "reshape": "reshape", "permute": "transpose", "to_seq": "seq", "crop": "crop", "pad": "pad", "embedding": "emb",
          "spectrogram": "spec", "rand_flip": "flip", "rand_rotate": "rotate", "rand_shift": "shift", "rand_zoom": "zoom",
          "rand_scale": "scale", "rand_stretch": "stretch", "multiply": "mul", "average": "avg", "maximum": "max",
          "mnv3s": "mnv3"}


def _pretty(t: str, cfg: dict, counts: dict) -> str:
    if t == "output":
        return "out"
    p = cfg.get("kind", "act") if t == "act" else PREFIX.get(t, t)
    counts[p] = counts.get(p, 0) + 1
    return f"{p}{counts[p]}"


def make_plan(graph: dict, length: int, n_out: int, image: tuple = None) -> Plan:
    """그래프 → 펼친 층 목록. 검사를 통과하지 못하면 ValueError(첫 문제)."""
    from dlstudio.graph import check_graph
    checked = check_graph(graph, length, n_out, image=image, rules=False)
    if not checked.ok:
        raise ValueError(checked.issues[0].message)
    c = _Checker(length, n_out, image)
    sc = c.scope(c.form(graph), INPUT)
    start = (1,) + tuple(image) if image else (1, length)
    plan = Plan(kind=checked.kind, in_shape=start, n_out=n_out)
    used: set = set()
    counts: dict = {}

    def walk(scope, in_shape, in_name: str, suffix: str, key_prefix: str) -> str:
        c.shape_pass(scope, in_shape)
        shapes = {scope.entry: tuple(in_shape)}
        names = {scope.entry: in_name}
        for k in scope.order:
            n = scope.nodes[k]
            t, ins = n["type"], list(n.get("inputs") or [])
            if t == "repeat":
                cur, s = names[ins[0]], shapes[ins[0]]
                body = scope.bodies[k]
                for i in range(n["times"]):
                    cur = walk(body, s, cur, f"{suffix}_r{i}", f"{key_prefix}{_key(k)}.{i}.")
                    s = tuple(c.shapes[body.exit])
                names[k], shapes[k] = cur, s
                plan.aliases[k] = cur
                continue
            nm = _ident(_pretty(t, n, counts), used)
            out = tuple(c.shapes[k])
            cfg = _config(n)
            if t == "output" and getattr(n_out, "grid", None):       # 탐지 출력: 격자 칸마다 1×1 필터
                cfg = dict(cfg, head="grid", per_cell=n_out.per_cell, grid=n_out.grid)
            if t == "output" and getattr(n_out, "seg", False):       # 구간 분할 출력: 시점마다 1×1 필터
                cfg = dict(cfg, head="seq", n_cls=n_out.n_cls)
            plan.steps.append(Step(nm, t, cfg, [names[x] for x in ins], shapes[ins[0]], out, key_prefix + _key(k), k,
                                   [tuple(shapes[x]) for x in ins]))
            names[k], shapes[k] = nm, out
        return names[scope.exit]

    plan.out = walk(sc, start, "x", "", "mods.")
    return plan


# ---------------------------------------------------------------- 공통 글자

ACT_LABEL = {"selu": "SELU", "relu6": "ReLU6", "hardswish": "HardSwish", "hardsigmoid": "HardSigmoid", "softmax": "Softmax",
             "relu": "ReLU", "leaky_relu": "LeakyReLU", "gelu": "GELU", "silu": "SiLU", "elu": "ELU", "mish": "Mish",
             "softplus": "Softplus", "sigmoid": "Sigmoid", "tanh": "Tanh"}


def _label(s: Step) -> str:
    b, t = s.cfg, s.type
    if t in ("conv", "conv2d"):
        extra = "".join(f" · {k} {b[k]}" for k in ("stride", "dilation", "groups") if b.get(k, 1) != 1)
        return f"{'Conv1D' if t == 'conv' else 'Conv2D'} {b['channels']} filters · k={b['kernel']}{extra}" + \
            (" · valid" if b.get("padding") == "valid" else "")
    return {"pool": f"AvgPool1D {b.get('size')}", "pool2d": "AvgPool2D 2", "global_avg": "GlobalAvgPool1D",
            "global_avg2d": "GlobalAvgPool2D", "last_step": "LastTimestep", "bn": "BatchNorm",
            "act": ACT_LABEL.get(b.get("kind"), "Activation"), "dropout": f"Dropout {b.get('rate', 0.5)}",
            "flatten": "Flatten", "dense": f"Dense {b.get('units')}", "output": "Output",
            "lstm": f"LSTM {b.get('hidden')}", "gru": f"GRU {b.get('hidden')}", "simple_rnn": f"SimpleRNN {b.get('hidden')}",
            "se": f"SE block (ratio {b.get('ratio', 4)})", "add": "Add", "concat": "Concat"}.get(t, X.name_of(t))


def _shape_text(shape: tuple) -> str:
    return "(" + ", ".join(str(v) for v in shape) + ")"


def _prod(shape) -> int:
    n = 1
    for v in shape:
        n *= v
    return n


def _io_doc(plan: Plan, data_info: Optional[dict], channels_last: bool) -> list:
    """파일 머리 설명: 입력·출력 모양과 데이터 준비."""
    L = plan.in_shape
    if plan.kind == "image":
        shape = f"(샘플 수, {L[1]}, {L[2]}, 1)" if channels_last else f"(샘플 수, 1, {L[1]}, {L[2]})"
        what = "흑백 그림 (높이 × 너비)"
    elif plan.kind == "rnn":
        shape, what = f"(샘플 수, {L[1]}, 1)", "시점마다 값 하나 (순환 층은 시점이 먼저)"
    else:
        shape = f"(샘플 수, {L[1]}, 1)" if channels_last else f"(샘플 수, 1, {L[1]})"
        what = "값 한 줄 (시계열이면 시점, 표면 특징)"
    lines = [f"입력: {shape} — {what}"]
    info = data_info or {}
    head = next((s.cfg for s in plan.steps if s.type == "output" and s.cfg.get("head") == "grid"), None)
    if head:
        G, A = head["grid"], head["per_cell"]
        lines.append(f"출력: (샘플 수, {plan.n_out}) = 칸마다 값 {A}개 × 격자 {G}×{G}칸을 펼친 것 (값 순서가 먼저, PyTorch 모양 (값, 세로, 가로)). "
                     f"칸마다: 사물 있음 · 칸 안 가운데 x·y · 너비·높이의 제곱근 · 클래스 점수 {A - 5}개"
                     + (f" ({', '.join(map(str, info.get('classes') or []))})" if info.get("classes") else ""))
        lines.append("정답: 그림마다 같은 모양의 목표값 (DL studio의 dlstudio.detect.encode). 찾은 상자는 아래 decode()로.")
        return lines
    seq = next((s.cfg for s in plan.steps if s.type == "output" and s.cfg.get("head") == "seq"), None)
    if seq:
        K = seq["n_cls"]
        lines.append(f"출력: (샘플 수, {plan.n_out}) = 시점마다 상태 점수 {K}개 × 길이 {plan.n_out // K}를 펼친 것 (상태가 먼저, 모양 (상태, 시점))"
                     + (f". 상태: {', '.join(map(str, info.get('classes') or []))}" if info.get("classes") else ""))
        lines.append("정답: 창마다 시점마다 상태 번호 (샘플 수, 길이). 예측 = 시점마다 점수가 가장 큰 상태.")
        return lines
    if info.get("task") == "classification" and info.get("classes"):
        lines.append(f"출력: (샘플 수, {plan.n_out}) 클래스 점수 — 가장 큰 칸이 예측. 순서: {', '.join(map(str, info['classes']))}")
    elif info.get("task") == "regression":
        lines.append("출력: (샘플 수, 1) 예측값")
    elif info.get("task") == "forecast":
        lines.append(f"출력: (샘플 수, {plan.n_out}) 다음 {plan.n_out}개 값 (원래 단위). 입력 = 바로 앞의 값들 (신호 전체의 평균·표준편차로 정규화)")
    elif info.get("task") == "anomaly":
        lines.append(f"출력: (샘플 수, {plan.n_out}) 되살린 입력 (정규화한 값). 이상 점수 = (입력 − 출력)²의 평균, 클수록 이상. "
                     "정상만으로 학습하세요")
    else:
        lines.append(f"출력: (샘플 수, {plan.n_out})")
    lines.append("데이터 준비: DL studio처럼 학습 데이터의 열마다 평균·표준편차로 정규화한 뒤 위 모양으로 바꿔 넣으세요.")
    return lines


# ---------------------------------------------------------------- PyTorch

_TORCH_ACT = {"relu": "torch.relu({})", "leaky_relu": "F.leaky_relu({})", "gelu": "F.gelu({})", "silu": "F.silu({})",
              "elu": "F.elu({})", "mish": "F.mish({})", "softplus": "F.softplus({})", "sigmoid": "torch.sigmoid({})",
              "tanh": "torch.tanh({})", "selu": "F.selu({})", "relu6": "F.relu6({})", "hardswish": "F.hardswish({})",
              "hardsigmoid": "F.hardsigmoid({})", "softmax": "torch.softmax({}, dim=1)"}

_TORCH_SE = '''

class SE(nn.Module):
    """Squeeze-and-Excitation: 채널마다 평균 → Linear → activation → Linear → sigmoid → 채널마다 곱함."""

    def __init__(self, channels, ratio, act):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(channels, hidden), act, nn.Linear(hidden, channels)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(self.act(self.fc1(x.flatten(2).mean(-1)))))
        return x * s.reshape(*s.shape, *([1] * (x.dim() - 2)))
'''

_TORCH_ACT_MODULE = {"relu": "nn.ReLU()", "leaky_relu": "nn.LeakyReLU()", "gelu": "nn.GELU()", "silu": "nn.SiLU()",
                     "elu": "nn.ELU()", "mish": "nn.Mish()", "softplus": "nn.Softplus()", "sigmoid": "nn.Sigmoid()",
                     "tanh": "nn.Tanh()"}


def _torch_layer(s: Step, n_out: int):
    """(모듈 만드는 식 또는 None, forward 식, studio 키 → 코드 키 짝 [(studio 앞부분, 코드 앞부분)])."""
    b, t, a = s.cfg, s.type, s.inputs[0] if s.inputs else ""
    me = f"self.{s.name}"
    if t == "conv" and b.get("padding") == "causal":
        opts = "".join(f", {n}={v}" for n, v, dflt in (("stride", b.get("stride", 1), 1), ("dilation", b.get("dilation", 1), 1),
                                                         ("groups", b.get("groups", 1), 1)) if v != dflt)
        return f"CausalConv1d({s.in_shape[0]}, {b['channels']}, {b['kernel']}{opts})", f"{me}({a})", [(s.key, s.name)]
    if t == "conv":
        k, d = b["kernel"], b.get("dilation", 1)
        pad = 0 if b.get("padding", "same") == "valid" else d * (k - 1) // 2
        opts = "".join(f", {n}={v}" for n, v, dflt in (("stride", b.get("stride", 1), 1), ("padding", pad, 0),
                                                         ("dilation", d, 1), ("groups", b.get("groups", 1), 1)) if v != dflt)
        return f"nn.Conv1d({s.in_shape[0]}, {b['channels']}, {k}{opts})", f"{me}({a})", [(s.key, s.name)]
    if t == "conv2d":
        k, d = b["kernel"], b.get("dilation", 1)
        pad = 0 if b.get("padding", "same") == "valid" else d * (k - 1) // 2
        opts = "".join(f", {n}={v}" for n, v, dflt in (("stride", b.get("stride", 1), 1), ("padding", pad, 0), ("dilation", d, 1),
                                                         ("groups", b.get("groups", 1), 1)) if v != dflt)
        return f"nn.Conv2d({s.in_shape[0]}, {b['channels']}, {k}{opts})", f"{me}({a})", [(s.key, s.name)]
    if t == "pool":
        return None, f"F.avg_pool1d({a}, {b['size']})", []
    if t == "pool2d":
        return None, f"F.avg_pool2d({a}, 2)", []
    if t == "global_avg":
        return None, f"{a}.mean(-1)", []
    if t == "global_avg2d":
        return None, f"{a}.mean(dim=(-1, -2))", []
    if t == "last_step":
        return None, f"{a}[:, :, -1]", []
    if t == "bn":
        cls = "nn.BatchNorm2d" if len(s.in_shape) == 3 else "nn.BatchNorm1d"
        return f"{cls}({s.in_shape[0]})", f"{me}({a})", [(s.key, s.name)]
    if t == "act":
        return None, _TORCH_ACT[b["kind"]].format(a), []
    if t == "dropout":
        return f"nn.Dropout({float(b.get('rate', 0.5))})", f"{me}({a})", []
    if t == "flatten":
        return None, f"torch.flatten({a}, 1)", []
    if t == "output" and b.get("head") == "grid":
        return f"nn.Conv2d({s.in_shape[0]}, {b['per_cell']}, 1)", f"torch.flatten({me}({a}), 1)", [(s.key + ".0", s.name)]
    if t == "output" and b.get("head") == "seq":
        return f"nn.Conv1d({s.in_shape[0]}, {b['n_cls']}, 1)", f"torch.flatten({me}({a}), 1)", [(s.key + ".0", s.name)]
    if t == "output" and b.get("head") == "distance":
        C = _prod(s.in_shape)
        return f"NormalDistance({C}, {X.dist_k(C)})", f"{me}({a})", [(s.key, s.name)]
    if t in ("dense", "output"):
        units = n_out if t == "output" else b["units"]
        flat = len(s.in_shape) >= 2
        src = f"torch.flatten({a}, 1)" if flat else a
        return f"nn.Linear({_prod(s.in_shape)}, {units})", f"{me}({src})", [(s.key + (".1" if flat else ""), s.name)]
    if t in RNN_TYPES:
        opts = "".join(f", {n}={v}" for n, v, dflt in (("layers", b.get("layers", 1), 1), ("bidirectional", bool(b.get("bidirectional")), False),
                                                         ("return_sequences", bool(b.get("return_sequences")), False)) if v != dflt)
        return f'RNNBlock("{t}", {s.in_shape[0]}, {b["hidden"]}{opts})', f"{me}({a})", [(s.key, s.name)]
    if t in X.EXTRA_TYPES:
        name, kw = X.torch_args(t, b, s.in_shape, s.in_shapes[1] if len(s.in_shapes) > 1 else None)
        expr = f"{me}({a}, {s.inputs[1]})" if t in X.TWO_INPUT else f"{me}({a})"
        return f"{name}({X.args_text(kw)})", expr, [(s.key, s.name)]
    if t == "multiply":
        return None, " * ".join(s.inputs), []
    if t == "average":
        return None, f"({' + '.join(s.inputs)}) / {len(s.inputs)}", []
    if t == "maximum":
        return None, f"torch.stack([{', '.join(s.inputs)}]).amax(0)", []
    if t == "se":
        act = _TORCH_ACT_MODULE[b.get("act", "relu")]
        return f"SE({s.in_shape[0]}, {b.get('ratio', 4)}, {act})", f"{me}({a})", [(s.key, s.name)]
    if t == "add":
        return None, " + ".join(s.inputs), []
    if t == "concat":
        return None, f"torch.cat([{', '.join(s.inputs)}], dim=1)", []
    raise ValueError(f"코드로 바꿀 수 없는 층: {t}")


def _extra_classes(plan: Plan) -> list:
    """생성 코드에 넣을 nn_extra 클래스 이름 (쓴 순서)."""
    out = []
    for s in plan.steps:
        if s.type in RNN_TYPES:
            name = "RNNBlock"
        elif s.type == "conv" and s.cfg.get("padding") == "causal":
            name = "CausalConv1d"
        elif s.type in X.EXTRA_TYPES:
            name = X.torch_args(s.type, s.cfg, s.in_shape, s.in_shapes[1] if len(s.in_shapes) > 1 else None)[0]
        elif s.type == "output" and s.cfg.get("head") == "distance":
            name = "NormalDistance"
        else:
            continue
        if not name.startswith("nn.") and name not in out:
            out.append(name)
    if any(n in out for n in ("RandomRotation", "RandomShift", "RandomZoom")):
        out.insert(0, "_affine")
    return out


def _class_sources(names: list) -> list:
    """nn_extra.py에서 그 클래스·함수의 소스 (torch 없이 글자로 읽음: 웹(Pyodide)에서도 됨)."""
    import ast
    from pathlib import Path
    text = (Path(__file__).resolve().parent / "nn_extra.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    found = {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    return [found[n] for n in names]


def _key_map(plan: Plan) -> list:
    out = []
    for s in plan.steps:
        out += _torch_layer(s, plan.n_out)[2]
    return out


def to_pytorch(graph: dict, length: int, n_out: int, image: tuple = None, data_info: Optional[dict] = None,
               train: Optional[dict] = None, weights_file: Optional[str] = None) -> str:
    """PyTorch 코드 (모델 클래스 StudioModel + 학습 함수 + 실행 예). weights_file: 함께 저장한 가중치 파일 이름."""
    plan = make_plan(graph, length, n_out, image)
    task = (data_info or {}).get("task", "classification")
    tr = dict({"epochs": 30, "lr": 1e-3, "batch_size": 32}, **(train or {}))
    init, fwd = [], []
    for s in plan.steps:
        make, expr, _ = _torch_layer(s, plan.n_out)
        if make:
            init.append(f"        self.{s.name} = {make}")
        fwd.append((f"        {s.name} = {expr}", f"# {_label(s)} → {_shape_text(s.out_shape)}"))
    width = min(80, max(len(a) for a, _ in fwd) + 2)
    fwd_lines = [a.ljust(width) + c for a, c in fwd]
    doc = _io_doc(plan, data_info, channels_last=False)
    ex_shape = ", ".join(["4"] + [str(v) for v in plan.in_shape]) if plan.kind != "rnn" else f"4, {plan.in_shape[1]}, 1"
    if weights_file:
        doc[-1] = "데이터 준비: 아래 predict()에 학습 CSV와 같은 열 순서의 원래 값(정답 열 빼고)을 넣으면 DL studio와 같은 정규화를 해 줍니다."
        doc.append(f"학습한 가중치·정규화 값: {weights_file} (DL studio에서 학습한 값, load_trained()가 불러옴)")
    lines = ['"""DL studio에서 만든 모델 (PyTorch).', ""] + doc + ['"""'] + (["import numpy as np"] if weights_file else []) + [
        "import torch", "import torch.nn.functional as F", "from torch import nn"]
    if any(s.type == "se" for s in plan.steps):
        lines += _TORCH_SE.rstrip("\n").split("\n")
    used = _extra_classes(plan)
    if used:
        lines.insert(lines.index("import torch"), "import math")
        for src in _class_sources(used):
            lines += ["", ""] + src.split("\n")
    lines += ["", "", "class StudioModel(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    lines += init or ["        pass"]
    first = ["        x = x.transpose(1, 2)            # 순환 모델: (샘플 수, 길이, 1) → 층들이 쓰는 (샘플 수, 1, 길이)"] if plan.kind == "rnn" else []
    lines += ["", "    def forward(self, x):"] + first + fwd_lines + [f"        return {plan.out}", "", ""]
    head = next((s.cfg for s in plan.steps if s.type == "output" and s.cfg.get("head") == "grid"), None)
    if head:
        return "\n".join(lines + _torch_det(head, tr, weights_file, plan))
    seq = next((s.cfg for s in plan.steps if s.type == "output" and s.cfg.get("head") == "seq"), None)
    if seq:                                                    # 구간 분할: 시점마다 교차 엔트로피
        loss, ytype = f"lambda out, t: F.cross_entropy(out.reshape(len(t), {seq['n_cls']}, -1), t)", "torch.long"
    elif task in ("regression", "forecast", "anomaly"):       # 이상 탐지는 y = X (입력을 되살림)
        loss, ytype = "nn.MSELoss()", "torch.float32"
    else:
        loss, ytype = "nn.CrossEntropyLoss()", "torch.long"
    lines += [
        f"def train(model, X, y, epochs={tr['epochs']}, lr={tr['lr']}, batch_size={tr['batch_size']}):",
        f'    """X: 위 입력 모양의 배열, y: {"정답 클래스 번호" if task not in ("regression", "forecast", "anomaly") else "다음 값들 (샘플, 예측 길이)" if task == "forecast" else "X를 펼친 것 (샘플, 값 수) — 정상만 넣으세요" if task == "anomaly" else "정답 값"}. Adam으로 학습."""',
        "    X = torch.as_tensor(X, dtype=torch.float32)",
        f"    y = torch.as_tensor(y, dtype={ytype})" + (".reshape(len(X), -1)" if task in ("regression", "forecast", "anomaly") else ""),
        f"    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), {loss}",
        "    for epoch in range(epochs):",
        "        model.train()",
        "        for idx in torch.randperm(len(X)).split(batch_size):",
        "            opt.zero_grad()",
        "            loss = loss_fn(model(X[idx]), y[idx])",
        "            loss.backward()",
        "            opt.step()",
        f"        print(f\"epoch {{epoch + 1}}: loss {{loss.item():.4f}}\")",
        "    return model.eval()",
        "", "",
    ]
    if weights_file:
        if plan.kind == "image":
            shape, n_feat = f"-1, 1, {plan.in_shape[1]}, {plan.in_shape[2]}", plan.in_shape[1] * plan.in_shape[2]
        elif plan.kind == "rnn":
            shape, n_feat = f"-1, {plan.in_shape[1]}, 1", plan.in_shape[1]
        else:
            shape, n_feat = f"-1, 1, {plan.in_shape[1]}", plan.in_shape[1]
        answer = ("out[:, 0].tolist()                                   # 예측값 (원래 단위)" if task == "regression"
                  else "out.tolist()                                         # 다음 값들 (원래 단위)" if task == "forecast"
                  else "((out - x.reshape(len(x), -1)) ** 2).mean(1).tolist()  # 이상 점수 (클수록 이상)" if task == "anomaly"
                  else f'[[str(info["classes"][k]) for k in row] for row in out.reshape(len(x), {seq["n_cls"]}, -1).argmax(1).tolist()]  # 시점마다 상태' if seq
                  else '[str(info["classes"][i]) for i in out.argmax(1)]      # 클래스 이름')
        lines += [
            f'def load_trained(path="{weights_file}"):',
            '    """DL studio에서 학습한 모델과 정규화 값을 불러옴 → (model, info)."""',
            "    z = np.load(path)",
            "    model = StudioModel()",
            '    model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if not k.startswith("_")})',
            '    info = {k[1:]: z[k] for k in z.files if k.startswith("_")}       # mean, std, classes',
            "    return model.eval(), info",
            "", "",
            "def predict(model, info, rows):",
            f'    """원래 값 행들 (샘플 수, {n_feat}: 학습 CSV와 같은 열 순서, 정답 열 빼고) → 예측. DL studio와 같은 정규화."""',
            '    x = (np.asarray(rows, dtype=np.float32) - info["mean"]) / info["std"]',
            f"    x = torch.as_tensor(x.reshape({shape}), dtype=torch.float32)",
            "    with torch.no_grad():",
            "        out = model(x)",
            f"    return {answer}",
            "", "",
            'if __name__ == "__main__":',
            "    model, info = load_trained()",
            f"    rows = np.zeros((2, {n_feat}), dtype=np.float32)         # ← 여기에 내 데이터 (원래 값, 한 행 = 샘플 하나)",
            "    print(predict(model, info, rows))",
            "",
        ]
        return "\n".join(lines)
    lines += ['if __name__ == "__main__":', "    model = StudioModel().eval()",
              f"    x = torch.randn({ex_shape})", "    with torch.no_grad():",
              f"        print(model(x).shape)       # torch.Size([4, {plan.n_out}])", ""]
    return "\n".join(lines)


def _torch_det(head: dict, tr: dict, weights_file: Optional[str], plan: Plan) -> list:
    """탐지: 손실(YOLO 첫 버전식, DL studio와 같음)·상자 풀기·학습·예측 코드."""
    from dlstudio import detect
    G, A = head["grid"], head["per_cell"]
    H, W = plan.in_shape[1], plan.in_shape[2]
    out = [
        f"GRID, PER_CELL = {G}, {A}          # 격자 {G}×{G}, 칸마다 값 {A}개",
        "", "",
        "def det_loss(out, target):",
        '    """있음(BCE, 없는 칸은 무게 %s) + 사물 칸의 위치·크기 제곱 오차 × %s + 클래스 교차 엔트로피, 그림 수로 나눔."""'
        % (detect.LAMBDA_NOOBJ, detect.LAMBDA_COORD),
        "    N = out.shape[0]",
        "    o, t = out.reshape(N, PER_CELL, GRID, GRID), target.reshape(N, PER_CELL, GRID, GRID)",
        "    pos = t[:, 0]",
        "    bce = F.binary_cross_entropy_with_logits(o[:, 0], pos, reduction=\"none\")",
        f"    loss = (bce * (pos + {detect.LAMBDA_NOOBJ} * (1 - pos))).sum()",
        f"    loss = loss + {detect.LAMBDA_COORD} * (((torch.sigmoid(o[:, 1:5]) - t[:, 1:5]) ** 2).sum(1) * pos).sum()",
        "    if PER_CELL > 6:",
        "        loss = loss + (-(t[:, 5:] * F.log_softmax(o[:, 5:], 1)).sum(1) * pos).sum()",
        "    return loss / N",
        "", "",
        f"def decode(out_row, score_min={detect.SHOW_SCORE}, nms_iou={detect.NMS_IOU}):",
        '    """출력 한 줄 → 찾은 상자 [(클래스 번호, 점수, x, y, w, h)] (0–1 비율, 왼쪽 위 기준). 점수 = 있음 × 클래스 확률."""',
        "    o = torch.as_tensor(out_row, dtype=torch.float32).reshape(PER_CELL, GRID, GRID)",
        "    obj = torch.sigmoid(o[0])",
        "    prob = torch.softmax(o[5:], 0) if PER_CELL > 6 else torch.ones(1, GRID, GRID)",
        "    p, k = prob.max(0)",
        "    score = obj * p",
        "    cand = []",
        "    for gy, gx in (score >= score_min).nonzero().tolist():",
        "        cx, cy = (gx + torch.sigmoid(o[1, gy, gx]).item()) / GRID, (gy + torch.sigmoid(o[2, gy, gx]).item()) / GRID",
        "        w, h = torch.sigmoid(o[3, gy, gx]).item() ** 2, torch.sigmoid(o[4, gy, gx]).item() ** 2",
        "        x0, y0, x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)",
        "        cand.append((int(k[gy, gx]), float(score[gy, gx]), x0, y0, x1 - x0, y1 - y0))",
        "    keep = []",
        "    for c in sorted(cand, key=lambda b: -b[1]):                     # 같은 클래스끼리 많이 겹치면 점수 높은 것만",
        "        if all(c[0] != q[0] or _iou(c[2:], q[2:]) <= nms_iou for q in keep):",
        "            keep.append(c)",
        "    return keep",
        "", "",
        "def _iou(a, b):",
        "    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])",
        "    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)",
        "    union = a[2] * a[3] + b[2] * b[3] - inter",
        "    return inter / union if union > 0 else 0.0",
        "", "",
        f"def train(model, X, Y, epochs={tr['epochs']}, lr={tr['lr']}, batch_size={tr['batch_size']}):",
        '    """X: 위 입력 모양의 그림, Y: 그림마다 목표값 (샘플 수, %d). Adam으로 학습."""' % (G * G * A),
        "    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)",
        "    opt = torch.optim.Adam(model.parameters(), lr=lr)",
        "    for epoch in range(epochs):",
        "        model.train()",
        "        for idx in torch.randperm(len(X)).split(batch_size):",
        "            opt.zero_grad()",
        "            loss = det_loss(model(X[idx]), Y[idx])",
        "            loss.backward()",
        "            opt.step()",
        '        print(f"epoch {epoch + 1}: loss {loss.item():.4f}")',
        "    return model.eval()",
        "", "",
    ]
    if weights_file:
        out += [
            f'def load_trained(path="{weights_file}"):',
            '    """DL studio에서 학습한 모델과 정규화 값을 불러옴 → (model, info)."""',
            "    z = np.load(path)",
            "    model = StudioModel()",
            '    model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if not k.startswith("_")})',
            '    info = {k[1:]: z[k] for k in z.files if k.startswith("_")}       # mean, std, classes',
            "    return model.eval(), info",
            "", "",
            "def predict(model, info, images):",
            f'    """흑백 그림들 (샘플 수, {H}, {W}) 픽셀 0–1 → 그림마다 찾은 상자 [(클래스 이름, 점수, x, y, w, h)]."""',
            '    x = (np.asarray(images, dtype=np.float32).reshape(len(images), -1) - info["mean"]) / info["std"]',
            f"    x = torch.as_tensor(x.reshape(-1, 1, {H}, {W}), dtype=torch.float32)",
            "    with torch.no_grad():",
            "        out = model(x)",
            '    return [[(str(info["classes"][b[0]]),) + b[1:] for b in decode(row)] for row in out]',
            "", "",
            'if __name__ == "__main__":',
            "    model, info = load_trained()",
            f"    images = np.zeros((2, {H}, {W}), dtype=np.float32)         # ← 여기에 내 그림 (흑백 {H}×{W}, 0–1)",
            "    print(predict(model, info, images))",
            "",
        ]
        return out
    return out + ['if __name__ == "__main__":', "    model = StudioModel().eval()",
                  f"    x = torch.randn(4, 1, {H}, {W})", "    with torch.no_grad():",
                  "        print([len(decode(row)) for row in model(x)])       # 그림마다 찾은 상자 수 (학습 전이라 무작위)", ""]


def code_state(state: dict, graph: dict, length: int, n_out: int, image: tuple = None) -> dict:
    """studio 가중치(GraphNet state_dict 이름: mods.m_<id>...) → 생성한 PyTorch 코드(StudioModel)의 이름. 값은 그대로."""
    plan = make_plan(graph, length, n_out, image)
    pairs = sorted(_key_map(plan), key=lambda p: -len(p[0]))
    out = {}
    for k, v in state.items():
        for src, dst in pairs:
            if k.startswith(src + "."):
                out[dst + k[len(src):]] = v
                break
        else:
            raise KeyError(f"가중치 {k}를 코드 이름으로 옮길 수 없음")
    return out


def torch_state_dict(net, graph: dict, length: int, n_out: int, image: tuple = None) -> dict:
    """studio 모델(GraphNet)의 가중치 → 생성한 PyTorch 코드(StudioModel)의 이름으로."""
    return {k: v.detach().clone() for k, v in code_state(net.state_dict(), graph, length, n_out, image).items()}


def save_npz(state: dict, file, data=None) -> None:
    """가중치 dict(이름 → 배열·텐서) → .npz (numpy만 있으면 읽음, torch 형식에 기대지 않음).
    data(Dataset)를 주면 정규화 값(_mean·_std)과 클래스 이름(_classes)도 함께 — 생성 코드의 predict가 씀."""
    import numpy as np
    arrays = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)) for k, v in state.items()}
    if data is not None:
        arrays["_mean"] = np.asarray(data.mean, dtype=np.float32)
        arrays["_std"] = np.asarray(data.std, dtype=np.float32)
        if data.task in ("classification", "detection"):
            arrays["_classes"] = np.asarray([str(c) for c in data.classes])
    np.savez(file, **arrays)


def param_specs(step: Step) -> list:
    """층 하나의 가중치 [(studio 이름, 모양, 종류)]. 종류: weight(학습) · buffer(BatchNorm 통계) · count · frozen(사전학습, 고정).
    이름·모양은 PyTorch(GraphNet)와 같다 — 웹 엔진(TF.js)이 이대로 만들어 PyTorch로 그대로 내보낸다."""
    b, t, k, ins = step.cfg, step.type, step.key, step.in_shape
    w = lambda name, *shape: (f"{k}.{name}", list(shape), "weight")
    if t == "conv":
        return [w("weight", b["channels"], ins[0] // b.get("groups", 1), b["kernel"]), w("bias", b["channels"])]
    if t == "conv2d":
        return [w("weight", b["channels"], ins[0] // b.get("groups", 1), b["kernel"], b["kernel"]), w("bias", b["channels"])]
    if t == "bn":
        C = ins[0]
        return [w("weight", C), w("bias", C), (f"{k}.running_mean", [C], "buffer"), (f"{k}.running_var", [C], "buffer"),
                (f"{k}.num_batches_tracked", [], "count")]
    if t == "output" and b.get("head") == "grid":
        return [(f"{k}.0.weight", [b["per_cell"], ins[0], 1, 1], "weight"), (f"{k}.0.bias", [b["per_cell"]], "weight")]
    if t == "output" and b.get("head") == "seq":
        return [(f"{k}.0.weight", [b["n_cls"], ins[0], 1], "weight"), (f"{k}.0.bias", [b["n_cls"]], "weight")]
    if t == "output" and b.get("head") == "distance":         # 정상과의 거리: 정상 샘플에서 계산해 넣는 값
        C = _prod(ins)
        K = X.dist_k(C)
        return [(f"{k}.mean", [C], "buffer"), (f"{k}.basis", [K, C], "buffer"), (f"{k}.inv_var", [K], "buffer"),
                (f"{k}.inv_rest", [1], "buffer")]
    if t in ("dense", "output"):
        units = step.out_shape[0]
        kk = k + (".1" if len(ins) >= 2 else "")
        return [(f"{kk}.weight", [units, _prod(ins)], "weight"), (f"{kk}.bias", [units], "weight")]
    if t in RNN_TYPES:
        G, H = {"lstm": 4, "gru": 3, "simple_rnn": 1}[t], b["hidden"]
        out = []
        dirs = ("", "_reverse") if b.get("bidirectional") else ("",)
        for layer in range(b.get("layers", 1)):
            inp = ins[0] if layer == 0 else H * len(dirs)
            for sfx in dirs:
                out += [(f"{k}.rnn.weight_ih_l{layer}{sfx}", [G * H, inp], "weight"), (f"{k}.rnn.weight_hh_l{layer}{sfx}", [G * H, H], "weight"),
                        (f"{k}.rnn.bias_ih_l{layer}{sfx}", [G * H], "weight"), (f"{k}.rnn.bias_hh_l{layer}{sfx}", [G * H], "weight")]
        return out
    if t in X.EXTRA_TYPES:
        return X.params(t, b, ins, step.out_shape, k, step.in_shapes[1] if len(step.in_shapes) > 1 else None)
    if t == "se":
        C = ins[0]
        h = max(1, C // b.get("ratio", 4))
        return [w("fc1.weight", h, C), w("fc1.bias", h), w("fc2.weight", C, h), w("fc2.bias", C)]
    return []


def plan_dict(graph: dict, length: int, n_out: int, image: tuple = None) -> dict:
    """웹 엔진용 계획 (JSON): 층 순서·설정·모양·가중치 이름과 모양, 학습 파라미터 수."""
    plan = make_plan(graph, length, n_out, image)
    steps = [{"name": s.name, "type": s.type, "cfg": s.cfg, "inputs": s.inputs, "in_shape": list(s.in_shape),
              "in_shapes": [list(x) for x in s.in_shapes], "out_shape": list(s.out_shape), "node": s.node, "key": s.key,
              "params": [list(p) for p in param_specs(s)]} for s in plan.steps]
    n_params = sum(_prod(p[1]) for st in steps for p in st["params"] if p[2] == "weight")
    n_frozen = sum(_prod(p[1]) for st in steps for p in st["params"] if p[2] == "frozen")
    return {"kind": plan.kind, "in_shape": list(plan.in_shape), "n_out": plan.n_out, "out": plan.out, "steps": steps,
            "aliases": plan.aliases, "params": n_params, "frozen": n_frozen}


# ---------------------------------------------------------------- TensorFlow Keras

_KERAS_ACT = {"selu": 'layers.Activation("selu"', "relu6": "layers.ReLU(6.0", "softmax": "layers.Softmax(axis=-1",
              "hardswish": "layers.Lambda(lambda t: t * tf.nn.relu6(t + 3.0) / 6.0",
              "hardsigmoid": "layers.Lambda(lambda t: tf.nn.relu6(t + 3.0) / 6.0",
              "relu": 'layers.Activation("relu"', "gelu": 'layers.Activation(gelu', "silu": 'layers.Activation("swish"',
              "elu": 'layers.Activation("elu"', "softplus": 'layers.Activation("softplus"',
              "sigmoid": 'layers.Activation("sigmoid"', "tanh": 'layers.Activation("tanh"',
              "mish": 'layers.Activation(mish', "leaky_relu": "layers.LeakyReLU(0.01"}


def _keras_shape(shape: tuple) -> tuple:
    """studio 모양(채널 먼저) → Keras 모양(채널 마지막)."""
    return tuple(shape[1:]) + (shape[0],) if len(shape) >= 2 else tuple(shape)


def _keras_lines(s: Step, n_out: int) -> list:
    b, t, n = s.cfg, s.type, s.name
    a = s.inputs[0] if s.inputs else ""
    nm = f'name="{n}"'
    if t in ("conv", "conv2d"):
        cls = "Conv1D" if t == "conv" else "Conv2D"
        stride, dil = b.get("stride", 1), b.get("dilation", 1)
        both = stride != 1 and dil != 1                  # Keras는 strides와 dilation_rate를 함께 못 받음 → stride 1로 계산 뒤 건너뛰어 뽑음
        opts = "".join(f", {k}={v}" for k, v, dflt in (("strides", 1 if both else stride, 1), ("dilation_rate", dil, 1),
                                                        ("groups", b.get("groups", 1), 1)) if v != dflt)
        pad = "valid" if b.get("padding", "same") == "valid" else "same"
        if b.get("padding") == "causal":
            pad = "causal"
        if both:
            return [f'{n}_full = layers.{cls}({b["channels"]}, {b["kernel"]}, padding="{pad}"{opts}, name="{n}_full")({a})',
                    f"{n} = layers.Lambda(lambda t: t[:, ::{stride}, :], {nm})({n}_full)"]
        return [f'{n} = layers.{cls}({b["channels"]}, {b["kernel"]}, padding="{pad}"{opts}, {nm})({a})']
    if t == "pool":
        return [f"{n} = layers.AveragePooling1D({b['size']}, {nm})({a})"]
    if t == "pool2d":
        return [f"{n} = layers.AveragePooling2D(2, {nm})({a})"]
    if t == "global_avg":
        return [f"{n} = layers.GlobalAveragePooling1D({nm})({a})"]
    if t == "global_avg2d":
        return [f"{n} = layers.GlobalAveragePooling2D({nm})({a})"]
    if t == "last_step":
        return [f"{n} = layers.Lambda(lambda t: t[:, -1, :], {nm})({a})"]
    if t == "bn":
        return [f"{n} = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, {nm})({a})"]
    if t == "act":
        return [f"{n} = {_KERAS_ACT[b['kind']]}, {nm})({a})"]
    if t == "dropout":
        return [f"{n} = layers.Dropout({float(b.get('rate', 0.5))}, {nm})({a})"]
    if t == "flatten":
        return [f"{n} = layers.Flatten({nm})({a})"]
    if t == "output" and b.get("head") == "seq":               # 시점마다 1×1 필터, 상태를 앞으로 옮겨 펼침 (PyTorch와 같은 순서)
        return [f'{n}_steps = layers.Conv1D({b["n_cls"]}, 1, name="{n}_steps")({a})',
                f'{n}_first = layers.Permute((2, 1), name="{n}_first")({n}_steps)',
                f"{n} = layers.Flatten({nm})({n}_first)"]
    if t == "output" and b.get("head") == "grid":              # 칸마다 1×1 필터, 채널을 앞으로 옮겨 펼침 (PyTorch와 같은 순서)
        return [f'{n}_cells = layers.Conv2D({b["per_cell"]}, 1, name="{n}_cells")({a})',
                f'{n}_first = layers.Permute((3, 1, 2), name="{n}_first")({n}_cells)',
                f"{n} = layers.Flatten({nm})({n}_first)"]
    if t in ("dense", "output"):
        units = n_out if t == "output" else b["units"]
        out = []
        if len(s.in_shape) >= 2:
            out.append(f'{n}_flat = layers.Flatten(name="{n}_flat")({a})')
            a = f"{n}_flat"
        return out + [f"{n} = layers.Dense({units}, {nm})({a})"]
    if t in RNN_TYPES:
        cls = {"lstm": "LSTM", "gru": "GRU", "simple_rnn": "SimpleRNN"}[t]
        k = b.get("layers", 1)
        out, cur = [], a
        for i in range(k):
            last = i == k - 1
            var = n if last else f"{n}_l{i}"
            seq = "" if last and not b.get("return_sequences") else ", return_sequences=True"
            layer = f'layers.{cls}({b["hidden"]}{seq})'
            if b.get("bidirectional"):
                out.append(f'{var} = layers.Bidirectional({layer}, name="{var}")({cur})')
            else:
                out.append(f'{var} = layers.{cls}({b["hidden"]}{seq}, name="{var}")({cur})')
            cur = var
        return out
    if t in X.EXTRA_TYPES or t in X.NEW_MERGES:
        return _keras_extra(s)
    if t == "se":
        C, h = s.in_shape[0], max(1, s.in_shape[0] // b.get("ratio", 4))
        act = b.get("act", "relu")
        act_kw = {"relu": '"relu"', "sigmoid": '"sigmoid"', "tanh": '"tanh"'}.get(act, '"relu"')
        two = len(s.in_shape) == 3
        return [f'{n}_s = layers.GlobalAveragePooling{"2" if two else "1"}D(name="{n}_squeeze")({a})',
                f'{n}_s = layers.Dense({h}, activation={act_kw}, name="{n}_fc1")({n}_s)',
                f'{n}_s = layers.Dense({C}, activation="sigmoid", name="{n}_fc2")({n}_s)',
                f'{n} = layers.Multiply({nm})([{a}, layers.Reshape(({"1, 1" if two else "1"}, {C}), name="{n}_reshape")({n}_s)])']
    if t == "add":
        return [f"{n} = layers.Add({nm})([{', '.join(s.inputs)}])"]
    if t == "concat":
        return [f"{n} = layers.Concatenate(axis=-1, {nm})([{', '.join(s.inputs)}])"]
    raise ValueError(f"코드로 바꿀 수 없는 층: {t}")


KERAS_UNSUPPORTED = ("mnv3s", "adaptive_pool", "attn_pool", "rmsnorm", "eca", "cbam", "spatial_attn", "linear_attn", "nonlocal",
                     "cross_attn", "pos_enc", "cls_token", "embedding", "spectrogram", "rand_scale", "rand_stretch", "drop_path")


def keras_unsupported(graph: dict, length: int, n_out: int, image: tuple = None) -> list:
    """Keras 코드로 바꿀 수 없는 블록 이름들 (PyTorch 코드는 모두 됨)."""
    plan = make_plan(graph, length, n_out, image)
    bad = []
    for s in plan.steps:
        rank = len(s.in_shape)
        if s.type in KERAS_UNSUPPORTED or (s.type == "mha" and s.cfg.get("pos") == "rope") \
                or (s.type == "output" and s.cfg.get("head") == "distance") \
                or (s.type == "upsample" and rank == 2 and s.cfg.get("mode") == "linear") \
                or (s.type in ("pixel_shuffle", "rand_flip", "rand_shift") and rank == 2):
            bad.append("출력 '정상과의 거리'" if s.type == "output" else X.name_of(s.type))
    return sorted(set(bad))


def _keras_extra(s: Step) -> list:
    """추가 블록 → Keras 줄 (채널이 마지막: 줄 (N, L, C), 그림 (N, H, W, C))."""
    b, t, n = s.cfg, s.type, s.name
    a = s.inputs[0] if s.inputs else ""
    nm = f'name="{n}"'
    o = lambda k: X.opt(b, k)
    rank = len(s.in_shape)
    d = "2D" if rank == 3 else "1D"
    C = s.in_shape[0]
    if t in ("multiply", "average", "maximum"):
        return [f"{n} = layers.{t.capitalize()}({nm})([{', '.join(s.inputs)}])"]
    if t == "glu":
        g = "tf.sigmoid" if o("gate") == "sigmoid" else "tf.nn.silu"
        return [f'{n}_h = layers.Dense({2 * o("units")}, name="{n}_proj")({a})',
                f"{n} = layers.Lambda(lambda t: t[..., :{o('units')}] * {g}(t[..., {o('units')}:]), {nm})({n}_h)"]
    if t in ("dw_conv", "dw_conv2d"):
        return [f'{n} = layers.DepthwiseConv{"2D" if t == "dw_conv2d" else "1D"}({o("kernel")}, depth_multiplier={o("multiplier")}, padding="same", {nm})({a})']
    if t in ("sep_conv", "sep_conv2d"):
        return [f'{n} = layers.SeparableConv{"2D" if t == "sep_conv2d" else "1D"}({o("channels")}, {o("kernel")}, padding="same", {nm})({a})']
    if t in ("max_pool", "max_pool2d"):
        return [f"{n} = layers.MaxPooling{d}({o('size')}, {nm})({a})"]
    if t in ("global_max", "global_max2d"):
        return [f"{n} = layers.GlobalMaxPooling{d}({nm})({a})"]
    if t == "first_step":
        return [f"{n} = layers.Lambda(lambda t: t[:, 0, :], {nm})({a})"]
    if t == "upsample":
        if rank == 3:
            mode = "bilinear" if o("mode") == "linear" else "nearest"
            return [f'{n} = layers.UpSampling2D(2, interpolation="{mode}", {nm})({a})']
        return [f"{n} = layers.UpSampling1D(2, {nm})({a})"]
    if t == "pixel_shuffle":
        return [f"{n} = layers.Lambda(lambda t: tf.nn.depth_to_space(t, 2), {nm})({a})"]
    if t in ("conv_t", "conv_t2d"):
        return [f'{n} = layers.Conv{d}Transpose({o("channels")}, {o("kernel")}, strides={o("stride")}, padding="same", {nm})({a})']
    if t == "mha":
        h = o("heads")
        causal = ", use_causal_mask=True" if o("causal") else ""
        if rank == 3:
            H, W = s.in_shape[1], s.in_shape[2]
            return [f'{n}_s = layers.Reshape(({H * W}, {C}), name="{n}_seq")({a})',
                    f'{n}_s = layers.MultiHeadAttention({h}, {C // h}, name="{n}_mha")({n}_s, {n}_s{causal})',
                    f"{n} = layers.Reshape(({H}, {W}, {C}), {nm})({n}_s)"]
        return [f"{n} = layers.MultiHeadAttention({h}, {C // h}, {nm})({a}, {a}{causal})"]
    if t == "patch_embed":
        p = o("patch")
        return [f'{n}_c = layers.Conv2D({o("dim")}, {p}, strides={p}, name="{n}_proj")({a})',
                f"{n} = layers.Reshape((-1, {o('dim')}), {nm})({n}_c)"]
    if t == "layernorm":
        return [f"{n} = layers.LayerNormalization(axis=-1, epsilon=1e-5, {nm})({a})"]
    if t == "groupnorm":
        return [f"{n} = layers.GroupNormalization({o('groups')}, axis=-1, epsilon=1e-5, {nm})({a})"]
    if t == "instancenorm":
        return [f"{n} = layers.GroupNormalization({C}, axis=-1, epsilon=1e-5, {nm})({a})"]
    if t == "prelu":
        axes = list(range(1, rank + 1))
        return [f"{n} = layers.PReLU(shared_axes={axes}, {nm})({a})"]
    if t == "spatial_dropout":
        return [f"{n} = layers.SpatialDropout{d}({float(o('rate'))}, {nm})({a})"]
    if t == "gaussian_noise":
        return [f"{n} = layers.GaussianNoise({float(o('std'))}, {nm})({a})"]
    if t == "slice":
        return [f"{n} = layers.Lambda(lambda t: t[..., {o('start')}:{o('start') + o('count')}], {nm})({a})"]
    if t == "reshape":
        c = o("channels")
        first = "(0, 2, 1)" if rank == 2 else "(0, 3, 1, 2)" if rank == 3 else None
        pre = f"tf.transpose(t, {first})" if first else "t"
        return [f"{n} = layers.Lambda(lambda t: tf.transpose(tf.reshape({pre}, (-1, {c}, {s.out_shape[1]})), (0, 2, 1)), {nm})({a})"]
    if t == "permute":
        return [f"{n} = layers.Permute({'(2, 1)' if rank == 2 else '(2, 1, 3)'}, {nm})({a})"]
    if t == "to_seq":
        return [f"{n} = layers.Reshape(({s.in_shape[1] * s.in_shape[2]}, {C}), {nm})({a})"]
    if t == "crop":
        return [f"{n} = layers.Cropping{d}({o('amount')}, {nm})({a})"]
    if t == "pad":
        return [f"{n} = layers.ZeroPadding{d}({o('amount')}, {nm})({a})"]
    if t == "rand_flip":
        mode = {"h": "horizontal", "v": "vertical", "hv": "horizontal_and_vertical"}[o("dir")]
        return [f'{n} = layers.RandomFlip("{mode}", {nm})({a})']
    if t == "rand_rotate":
        return [f"{n} = layers.RandomRotation({o('degrees') / 360:.6g}, fill_mode='constant', {nm})({a})"]
    if t == "rand_zoom":
        return [f"{n} = layers.RandomZoom({float(o('frac'))}, fill_mode='constant', {nm})({a})"]
    if t == "rand_shift":
        return [f"{n} = layers.RandomTranslation({float(o('frac'))}, {float(o('frac'))}, fill_mode='constant', {nm})({a})"]
    raise ValueError(f"TensorFlow 코드로 바꿀 수 없는 블록: {X.name_of(t)} (PyTorch 코드를 쓰세요)")


def to_keras(graph: dict, length: int, n_out: int, image: tuple = None, data_info: Optional[dict] = None,
             train: Optional[dict] = None) -> str:
    """TensorFlow Keras 코드 (build_model 함수형 API + 학습 함수 + 실행 예). 채널이 마지막인 모양."""
    plan = make_plan(graph, length, n_out, image)
    bad = keras_unsupported(graph, length, n_out, image)
    if bad:
        raise ValueError(f"TensorFlow 코드로 바꿀 수 없는 블록: {', '.join(bad)} — PyTorch 코드로 내보내세요 (모든 블록이 됨)")
    task = (data_info or {}).get("task", "classification")
    tr = dict({"epochs": 30, "lr": 1e-3, "batch_size": 32}, **(train or {}))
    if plan.kind == "rnn":
        in_shape = (plan.in_shape[1], 1)
    else:
        in_shape = _keras_shape(plan.in_shape)
    body = []
    for s in plan.steps:
        ls = _keras_lines(s, plan.n_out)
        comment = f"# {_label(s)} → {_shape_text(_keras_shape(s.out_shape))}"
        body += [f"    {l}" for l in ls[:-1]] + [(f"    {ls[-1]}", comment)]
    width = min(84, max(len(x[0]) for x in body if isinstance(x, tuple)) + 2)
    body = [x[0].ljust(width) + x[1] if isinstance(x, tuple) else x for x in body]
    kinds = {s.cfg.get("kind") for s in plan.steps if s.type == "act"}
    helpers = []
    if "gelu" in kinds:
        helpers += ["", "", "def gelu(t):", '    """GELU (PyTorch와 같은 정확한 식)."""', "    return keras.activations.gelu(t, approximate=False)"]
    if "mish" in kinds:
        helpers += ["", "", "def mish(t):", "    return t * tf.math.tanh(tf.math.softplus(t))"]
    doc = _io_doc(plan, data_info, channels_last=True)
    doc.append("가중치: DL studio에서 학습한 값은 들어 있지 않음 (TensorFlow에서 아래 train으로 새로 학습)")
    head = next((s.cfg for s in plan.steps if s.type == "output" and s.cfg.get("head") == "grid"), None)
    if head:
        from dlstudio import detect
        G, A = head["grid"], head["per_cell"]
        helpers += ["", "", f"GRID, PER_CELL = {G}, {A}", "", "",
                    "def det_loss(target, out):",
                    '    """DL studio와 같은 탐지 손실 (YOLO 첫 버전식). 출력·목표값은 (값, 세로, 가로) 순서로 펼친 것."""',
                    "    o, t = tf.reshape(out, (-1, PER_CELL, GRID, GRID)), tf.reshape(target, (-1, PER_CELL, GRID, GRID))",
                    "    pos = t[:, 0]",
                    "    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos, logits=o[:, 0])",
                    f"    loss = tf.reduce_sum(bce * (pos + {detect.LAMBDA_NOOBJ} * (1 - pos)))",
                    f"    loss += {detect.LAMBDA_COORD} * tf.reduce_sum(tf.reduce_sum((tf.sigmoid(o[:, 1:5]) - t[:, 1:5]) ** 2, 1) * pos)",
                    "    if PER_CELL > 6:",
                    "        loss += tf.reduce_sum(-tf.reduce_sum(t[:, 5:] * tf.nn.log_softmax(o[:, 5:], axis=1), 1) * pos)",
                    "    return loss / tf.cast(tf.shape(out)[0], tf.float32)"]
        loss, metric = "det_loss", "[]"
    elif task in ("regression", "forecast", "anomaly"):
        loss, metric = '"mse"', '["mae"]'
    else:
        loss, metric = "keras.losses.SparseCategoricalCrossentropy(from_logits=True)", '["accuracy"]'
    lines = ['"""DL studio에서 만든 모델 (TensorFlow Keras).', ""] + doc + ['"""', "import tensorflow as tf",
                                                                       "from tensorflow import keras",
                                                                       "from tensorflow.keras import layers"]
    lines += helpers
    lines += ["", "", "def build_model():", f"    x = keras.Input(shape={_shape_text(in_shape) if len(in_shape) > 1 else f'({in_shape[0]},)'})"]
    lines += body
    lines += [f'    return keras.Model(x, {plan.out}, name="studio_model")', "", "",
              f"def train(model, X, y, epochs={tr['epochs']}, lr={tr['lr']}, batch_size={tr['batch_size']}):",
              f'    """X: 위 입력 모양의 배열, y: {"정답 클래스 번호" if task not in ("regression", "forecast", "anomaly") else "다음 값들 (샘플, 예측 길이)" if task == "forecast" else "X를 펼친 것 (샘플, 값 수) — 정상만 넣으세요" if task == "anomaly" else "정답 값"}. Adam으로 학습."""',
              f"    model.compile(optimizer=keras.optimizers.Adam(lr), loss={loss}, metrics={metric})",
              "    model.fit(X, y, epochs=epochs, batch_size=batch_size)",
              "    return model", "", "",
              'if __name__ == "__main__":',
              "    model = build_model()",
              "    model.summary()", ""]
    return "\n".join(lines)


def generate(framework: str, graph: dict, length: int, n_out: int, image: tuple = None, data_info: Optional[dict] = None,
             train: Optional[dict] = None, weights_file: Optional[str] = None) -> str:
    if framework == "pytorch":
        return to_pytorch(graph, length, n_out, image, data_info, train, weights_file)
    if framework == "tensorflow":
        return to_keras(graph, length, n_out, image, data_info, train)
    raise ValueError(f"framework는 {' 또는 '.join(FRAMEWORKS)}: {framework!r}")

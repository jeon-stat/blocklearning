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


@dataclass
class Plan:
    steps: list = field(default_factory=list)
    kind: str = "cnn"                # cnn | rnn | image
    in_shape: tuple = ()
    out: str = ""
    n_out: int = 1


def _ident(nid: str, used: set) -> str:
    n = nid if nid.isidentifier() and not keyword.iskeyword(nid) and nid not in RESERVED else "n_" + nid
    base, k = n, 2
    while n in used:
        n, k = f"{base}_{k}", k + 1
    used.add(n)
    return n


def make_plan(graph: dict, length: int, n_out: int, image: tuple = None) -> Plan:
    """그래프 → 펼친 층 목록. 검사를 통과하지 못하면 ValueError(첫 문제)."""
    from dlstudio.graph import check_graph
    checked = check_graph(graph, length, n_out, image=image)
    if not checked.ok:
        raise ValueError(checked.issues[0].message)
    c = _Checker(length, n_out, image)
    sc = c.scope(c.form(graph), INPUT)
    start = (1,) + tuple(image) if image else (1, length)
    plan = Plan(kind=checked.kind, in_shape=start, n_out=n_out)
    used: set = set()

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
                continue
            nm = _ident(k + suffix, used)
            out = tuple(c.shapes[k])
            plan.steps.append(Step(nm, t, _config(n), [names[x] for x in ins], shapes[ins[0]], out, key_prefix + _key(k)))
            names[k], shapes[k] = nm, out
        return names[scope.exit]

    plan.out = walk(sc, start, "x", "", "mods.")
    return plan


# ---------------------------------------------------------------- 공통 글자

ACT_LABEL = {"relu": "ReLU", "leaky_relu": "LeakyReLU", "gelu": "GELU", "silu": "SiLU", "elu": "ELU", "mish": "Mish",
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
            "se": f"SE block (ratio {b.get('ratio', 4)})", "add": "Add", "concat": "Concat"}.get(t, t)


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
    if info.get("task") == "classification" and info.get("classes"):
        lines.append(f"출력: (샘플 수, {plan.n_out}) 클래스 점수 — 가장 큰 칸이 예측. 순서: {', '.join(map(str, info['classes']))}")
    elif info.get("task") == "regression":
        lines.append("출력: (샘플 수, 1) 예측값")
    else:
        lines.append(f"출력: (샘플 수, {plan.n_out})")
    lines.append("데이터 준비: DL studio처럼 학습 데이터의 열마다 평균·표준편차로 정규화한 뒤 위 모양으로 바꿔 넣으세요.")
    return lines


# ---------------------------------------------------------------- PyTorch

_TORCH_ACT = {"relu": "torch.relu({})", "leaky_relu": "F.leaky_relu({})", "gelu": "F.gelu({})", "silu": "F.silu({})",
              "elu": "F.elu({})", "mish": "F.mish({})", "softplus": "F.softplus({})", "sigmoid": "torch.sigmoid({})",
              "tanh": "torch.tanh({})"}

_TORCH_SE = '''

class SE(nn.Module):
    """Squeeze-and-Excitation: 채널마다 평균 → Linear → activation → Linear → sigmoid → 채널마다 곱함."""

    def __init__(self, channels, ratio, act):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(channels, hidden), act, nn.Linear(hidden, channels)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(self.act(self.fc1(x.mean(-1)))))
        return x * s.unsqueeze(-1)
'''

_TORCH_ACT_MODULE = {"relu": "nn.ReLU()", "leaky_relu": "nn.LeakyReLU()", "gelu": "nn.GELU()", "silu": "nn.SiLU()",
                     "elu": "nn.ELU()", "mish": "nn.Mish()", "softplus": "nn.Softplus()", "sigmoid": "nn.Sigmoid()",
                     "tanh": "nn.Tanh()"}


def _torch_layer(s: Step, n_out: int):
    """(모듈 만드는 식 또는 None, forward 식, studio 키 → 코드 키 짝 [(studio 앞부분, 코드 앞부분)])."""
    b, t, a = s.cfg, s.type, s.inputs[0] if s.inputs else ""
    me = f"self.{s.name}"
    if t == "conv":
        k, d = b["kernel"], b.get("dilation", 1)
        pad = 0 if b.get("padding", "same") == "valid" else d * (k - 1) // 2
        opts = "".join(f", {n}={v}" for n, v, dflt in (("stride", b.get("stride", 1), 1), ("padding", pad, 0),
                                                         ("dilation", d, 1), ("groups", b.get("groups", 1), 1)) if v != dflt)
        return f"nn.Conv1d({s.in_shape[0]}, {b['channels']}, {k}{opts})", f"{me}({a})", [(s.key, s.name)]
    if t == "conv2d":
        k = b["kernel"]
        pad = 0 if b.get("padding", "same") == "valid" else k // 2
        opts = "".join(f", {n}={v}" for n, v, dflt in (("stride", b.get("stride", 1), 1), ("padding", pad, 0)) if v != dflt)
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
    if t in ("dense", "output"):
        units = n_out if t == "output" else b["units"]
        flat = len(s.in_shape) >= 2
        src = f"torch.flatten({a}, 1)" if flat else a
        return f"nn.Linear({_prod(s.in_shape)}, {units})", f"{me}({src})", [(s.key + (".1" if flat else ""), s.name)]
    if t in RNN_TYPES:
        cls = {"lstm": "nn.LSTM", "gru": "nn.GRU", "simple_rnn": "nn.RNN"}[t]
        layers = b.get("layers", 1)
        opt = f", num_layers={layers}" if layers != 1 else ""
        return f"{cls}(1, {b['hidden']}{opt}, batch_first=True)", f"{me}({a})[0][:, -1]", [(s.key + ".rnn", s.name)]
    if t == "se":
        act = _TORCH_ACT_MODULE[b.get("act", "relu")]
        return f"SE({s.in_shape[0]}, {b.get('ratio', 4)}, {act})", f"{me}({a})", [(s.key, s.name)]
    if t == "add":
        return None, " + ".join(s.inputs), []
    if t == "concat":
        return None, f"torch.cat([{', '.join(s.inputs)}], dim=1)", []
    raise ValueError(f"코드로 바꿀 수 없는 층: {t}")


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
        doc.append(f"학습한 가중치: {weights_file} (DL studio에서 학습한 값, 아래 실행 예가 불러옴)")
    lines = ['"""DL studio에서 만든 모델 (PyTorch).', ""] + doc + ['"""', "import torch", "import torch.nn.functional as F",
                                                               "from torch import nn"]
    if any(s.type == "se" for s in plan.steps):
        lines += _TORCH_SE.rstrip("\n").split("\n")
    lines += ["", "", "class StudioModel(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    lines += init or ["        pass"]
    lines += ["", "    def forward(self, x):"] + fwd_lines + [f"        return {plan.out}", "", ""]
    if task == "regression":
        loss, ytype = "nn.MSELoss()", "torch.float32"
    else:
        loss, ytype = "nn.CrossEntropyLoss()", "torch.long"
    lines += [
        f"def train(model, X, y, epochs={tr['epochs']}, lr={tr['lr']}, batch_size={tr['batch_size']}):",
        f'    """X: 위 입력 모양의 배열, y: {"정답 클래스 번호" if task != "regression" else "정답 값"}. Adam으로 학습."""',
        "    X = torch.as_tensor(X, dtype=torch.float32)",
        f"    y = torch.as_tensor(y, dtype={ytype})" + (".reshape(-1, 1)" if task == "regression" else ""),
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
        'if __name__ == "__main__":',
        "    model = StudioModel().eval()",
    ]
    if weights_file:
        lines += [f'    model.load_state_dict(torch.load("{weights_file}"))       # DL studio에서 학습한 가중치']
    lines += [f"    x = torch.randn({ex_shape})", "    with torch.no_grad():",
              f"        print(model(x).shape)       # torch.Size([4, {plan.n_out}])", ""]
    return "\n".join(lines)


def torch_state_dict(net, graph: dict, length: int, n_out: int, image: tuple = None) -> dict:
    """studio 모델(GraphNet)의 가중치 → 생성한 PyTorch 코드(StudioModel)의 이름으로."""
    plan = make_plan(graph, length, n_out, image)
    pairs = sorted(_key_map(plan), key=lambda p: -len(p[0]))
    out = {}
    for k, v in net.state_dict().items():
        for src, dst in pairs:
            if k.startswith(src + "."):
                out[dst + k[len(src):]] = v.detach().clone()
                break
        else:
            raise KeyError(f"가중치 {k}를 코드 이름으로 옮길 수 없음")
    return out


# ---------------------------------------------------------------- TensorFlow Keras

_KERAS_ACT = {"relu": 'layers.Activation("relu"', "gelu": 'layers.Activation(gelu', "silu": 'layers.Activation("swish"',
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
            seq = "" if last else ", return_sequences=True"
            out.append(f'{var} = layers.{cls}({b["hidden"]}{seq}, name="{var}")({cur})')
            cur = var
        return out
    if t == "se":
        C, h = s.in_shape[0], max(1, s.in_shape[0] // b.get("ratio", 4))
        act = b.get("act", "relu")
        act_kw = {"relu": '"relu"', "sigmoid": '"sigmoid"', "tanh": '"tanh"'}.get(act, '"relu"')
        return [f'{n}_s = layers.GlobalAveragePooling1D(name="{n}_squeeze")({a})',
                f'{n}_s = layers.Dense({h}, activation={act_kw}, name="{n}_fc1")({n}_s)',
                f'{n}_s = layers.Dense({C}, activation="sigmoid", name="{n}_fc2")({n}_s)',
                f'{n} = layers.Multiply({nm})([{a}, layers.Reshape((1, {C}), name="{n}_reshape")({n}_s)])']
    if t == "add":
        return [f"{n} = layers.Add({nm})([{', '.join(s.inputs)}])"]
    if t == "concat":
        return [f"{n} = layers.Concatenate(axis=-1, {nm})([{', '.join(s.inputs)}])"]
    raise ValueError(f"코드로 바꿀 수 없는 층: {t}")


def to_keras(graph: dict, length: int, n_out: int, image: tuple = None, data_info: Optional[dict] = None,
             train: Optional[dict] = None) -> str:
    """TensorFlow Keras 코드 (build_model 함수형 API + 학습 함수 + 실행 예). 채널이 마지막인 모양."""
    plan = make_plan(graph, length, n_out, image)
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
    if task == "regression":
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
              f'    """X: 위 입력 모양의 배열, y: {"정답 클래스 번호" if task != "regression" else "정답 값"}. Adam으로 학습."""',
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

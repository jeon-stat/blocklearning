"""블록: 화면에서 쌓은 블록 JSON을 검증하고, 블록마다 출력 모양을 계산하고, torch 모델로 만든다.

블록 JSON은 위에서 아래 순서의 목록. 감싸는 블록(residual·repeat)은 "body" 목록을 가진다.
블록 위치(Path)는 최상위 i → (i,), body 안 j → (i, j), 더 깊으면 계속.
검증을 통과한 조립은 모델 만들기(build_model)가 실패하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dlstudio import layers as X

BLOCK_TYPES = ("conv", "pool", "global_avg", "last_step", "bn", "act", "dropout", "flatten", "dense", "lstm", "gru",
               "simple_rnn", "se", "branch", "residual", "repeat", "output", "conv2d", "pool2d", "global_avg2d") + X.EXTRA_TYPES
CNN_TYPES = ("conv", "pool", "global_avg", "last_step", "se", "branch")      # 시간 축이 있어야 하는 층
ONE_D_TYPES = ("conv", "pool", "global_avg", "last_step", "se")              # 시계열·표 전용 (이미지에서 못 씀)
IMAGE_TYPES = ("conv2d", "pool2d", "global_avg2d")                           # 이미지 전용
RNN_TYPES = ("lstm", "gru", "simple_rnn")
ACT_KINDS = ("relu", "leaky_relu", "gelu", "silu", "elu", "mish", "softplus", "sigmoid", "tanh",
             "selu", "relu6", "hardswish", "hardsigmoid", "softmax")
# 한도: 실수로 PC가 멈출 만큼 큰 모델만 막는 안전선
MAX_WIDTH = 65536           # Dense 입력·출력 개수
MAX_CHANNEL_LEN = 1048576   # 채널 하나의 길이(이미지면 높이×너비)

NAMES = {"conv": "Conv1D", "pool": "AvgPool1D", "global_avg": "GlobalAvgPool1D", "last_step": "LastTimestep",
         "bn": "BatchNorm", "act": "Activation", "dropout": "Dropout", "flatten": "Flatten", "dense": "Dense",
         "lstm": "LSTM", "gru": "GRU", "simple_rnn": "SimpleRNN", "se": "SE block", "branch": "Concat branches", "residual": "Residual block",
         "repeat": "Repeat", "conv2d": "Conv2D", "pool2d": "AvgPool2D", "global_avg2d": "GlobalAvgPool2D",
         "output": "Output", **X.EXTRA_NAMES}

Path = tuple


@dataclass
class BlockIssue:
    path: tuple
    message: str
    hint: str

    def to_dict(self) -> dict:
        return {"path": list(self.path), "message": self.message, "hint": self.hint}


@dataclass
class Checked:
    issues: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)        # Path → 블록 뒤 모양 (배치 제외): (C, L) 또는 (n,)
    kind: str = "cnn"

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "issues": [i.to_dict() for i in self.issues],
                "shapes": [{"path": list(p), "shape": list(s)} for p, s in self.shapes.items()]}


def _int(b: dict, key: str, lo: int, hi: int, default=None, choices=None):
    v = b.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    if choices is not None:
        return v if v in choices else None
    return v if lo <= v <= hi else None


# 값 규칙: (키, 최소, 최대, 기본값, 허용 값, 설명)
PARAMS = {
    "conv": [("channels", 1, 256, None, None, "filters는 1–256"), ("kernel", 1, 31, None, tuple(range(1, 32, 2)),
             "kernel size는 1–31의 홀수"), ("stride", 1, 2, 1, (1, 2), "stride는 1 또는 2"),
             ("dilation", 1, 64, 1, None, "dilation은 1–64"), ("groups", 1, 256, 1, None, "groups는 1–256")],
    "pool": [("size", 2, 4, None, (2, 4), "pool size는 2 또는 4")],
    "conv2d": [("channels", 1, 256, None, None, "filters는 1–256"), ("kernel", 1, 7, None, (1, 3, 5, 7), "kernel size는 1·3·5·7"),
               ("stride", 1, 2, 1, (1, 2), "stride는 1 또는 2"), ("dilation", 1, 8, 1, (1, 2, 4, 8), "dilation은 1·2·4·8"),
               ("groups", 1, 256, 1, None, "groups는 1–256")],
    "pool2d": [("size", 2, 2, 2, (2,), "pool size는 2")],
    "dense": [("units", 1, 1024, None, None, "units는 1–1024")],
    "lstm": [("hidden", 1, 256, None, None, "units는 1–256"), ("layers", 1, 3, 1, None, "layers는 1–3")],
    "gru": [("hidden", 1, 256, None, None, "units는 1–256"), ("layers", 1, 3, 1, None, "layers는 1–3")],
    "simple_rnn": [("hidden", 1, 256, None, None, "units는 1–256"), ("layers", 1, 3, 1, None, "layers는 1–3")],
    "repeat": [("times", 1, 10, None, None, "repeat count는 1–10")],
    "se": [("ratio", 1, 64, 4, None, "ratio는 1–64")],
    **X.EXTRA_PARAMS,
}


class _Walker:
    def __init__(self, n_out: int, image: bool = False):
        self.n_out = n_out
        self.image = image
        self.max_width = MAX_WIDTH
        self.max_len = MAX_CHANNEL_LEN
        self.issues: list = []
        self.shapes: dict = {}

    def issue(self, path, message, hint):
        self.issues.append(BlockIssue(tuple(path), message, hint))

    def values_ok(self, b: dict, path) -> bool:
        ok = True
        for key, lo, hi, default, choices, desc in PARAMS.get(b["type"], []):
            if _int(b, key, lo, hi, default, choices) is None:
                self.issue(path, f"{NAMES[b['type']]}: {key} 값 {b.get(key)!r}이(가) 맞지 않음", desc)
                ok = False
        if b["type"] == "act" and b.get("kind") not in ACT_KINDS:
            self.issue(path, f"Activation 종류 {b.get('kind')!r}을(를) 모름", "목록에 있는 함수 중에서 고르세요")
            ok = False
        if b["type"] == "se" and b.get("act", "relu") not in ACT_KINDS:
            self.issue(path, f"SE block: 안쪽 activation {b.get('act')!r}을(를) 모름", "ReLU 등 목록에 있는 함수로")
            ok = False
        pads = ("same", "valid", "causal") if b["type"] == "conv" else ("same", "valid")
        if b["type"] in ("conv", "conv2d") and b.get("padding", "same") not in pads:
            self.issue(path, f"{NAMES[b['type']]}: padding {b.get('padding')!r}이(가) 맞지 않음", "padding은 " + "·".join(pads))
            ok = False
        if b["type"] in RNN_TYPES:
            for key in ("bidirectional", "return_sequences"):
                if not isinstance(b.get(key, False), bool):
                    self.issue(path, f"{NAMES[b['type']]}: {key} 값 {b.get(key)!r}이(가) 맞지 않음", "참·거짓으로")
                    ok = False
        if b["type"] in X.EXTRA_TYPES and not X.values_ok(self, b, path):
            ok = False
        if b["type"] == "dropout":
            r = b.get("rate", 0.5)
            if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0 <= r < 1:
                self.issue(path, f"Dropout: rate {r!r}이(가) 맞지 않음", "rate는 0 이상 1 미만 (예: 0.2)")
                ok = False
        return ok

    def walk(self, blocks: list, shape, prefix: tuple, ctx: str):
        """블록 목록을 따라 모양을 계산. shape: (C, L) 또는 (n,), 모르면 None. ctx: top | residual | repeat."""
        for i, b in enumerate(blocks):
            path = prefix + (i,)
            if not isinstance(b, dict) or b.get("type") not in BLOCK_TYPES:
                t = b.get("type") if isinstance(b, dict) else b
                self.issue(path, f"알 수 없는 블록 {t!r}", "팔레트에 있는 블록만 쓸 수 있습니다")
                shape = None
                continue
            shape = self.block(b, shape, path, ctx, first=(prefix == () and i == 0), last=(i == len(blocks) - 1))
            if shape is not None:
                self.shapes[path] = tuple(shape)          # repeat 안은 마지막 반복 값이 남음
        return shape

    def block(self, b: dict, shape, path, ctx, first: bool, last: bool):
        t = b["type"]
        if not self.values_ok(b, path):
            return None
        if t == "output":
            if ctx != "top" or not last:
                self.issue(path, "출력 블록은 맨 아래에 하나만", "출력 블록을 맨 아래로 옮기세요")
            if getattr(self.n_out, "grid", None):
                return self.det_head(shape, path)
            return self.dense_like(shape, self.n_out, path)
        if shape is None:
            return None
        if t in RNN_TYPES:                                    # (채널, 길이)를 시점마다 채널 수만큼의 값으로 읽음
            if len(shape) != 2:
                self.issue(path, f"{NAMES[t]}은(는) (채널, 길이) 모양에만 쓸 수 있음 (지금 {tuple(shape)})",
                           "이미지는 ToSequence 뒤에, Dense·GlobalPool 뒤에는 쓸 수 없음 (앞 순환 층은 return_sequences를 켜세요)")
                return None
            h = b["hidden"] * (2 if b.get("bidirectional") else 1)
            return (h, shape[1]) if b.get("return_sequences") else (h,)
        if t in X.EXTRA_TYPES:
            return X.shape(self, b, shape, path, ctx)
        if t in IMAGE_TYPES and len(shape) != 3:
            self.issue(path, f"{NAMES[t]}은(는) (채널, 높이, 너비) 모양에만 쓸 수 있음 (지금 {tuple(shape)})",
                       "시계열에는 Conv1D·AvgPool1D를 쓰세요 (Spectrogram 뒤에는 됨). GlobalAvgPool2D·Flatten·Dense 뒤에는 쓸 수 없습니다")
            return None
        if t == "se" and len(shape) not in (2, 3):
            self.issue(path, f"SE block은 (채널, 길이)나 (채널, 높이, 너비) 모양에만 (지금 {tuple(shape)})", X.SPATIAL_HINT)
            return None
        if t in CNN_TYPES and t != "se" and len(shape) != 2 and not (t == "branch" and len(shape) == 3):
            self.issue(path, f"{NAMES[t]}은(는) (채널, 길이) 모양에만 쓸 수 있음 (지금 {tuple(shape)})",
                       "이미지에는 Conv2D·AvgPool2D 등 2D 층을 쓰세요 (ToSequence 뒤에는 됨). LSTM·GlobalAvgPool·Dense 뒤에는 쓸 수 없습니다")
            return None
        C, L = shape if len(shape) == 2 else (shape[0], None) if len(shape) == 3 else (None, None)
        if t == "conv2d":
            stride, valid, k = b.get("stride", 1), b.get("padding", "same") == "valid", b["kernel"]
            d, g = b.get("dilation", 1), b.get("groups", 1)
            if ctx == "residual" and (stride != 1 or valid):
                self.issue(path, "Residual block 안에서는 stride 2나 padding valid를 쓸 수 없음 (크기가 바뀜)",
                           "stride 1, padding same으로 하거나 Residual block 밖으로 옮기세요")
                return None
            if C % g or b["channels"] % g:
                self.issue(path, f"groups {g}가 들어오는 채널 {C}과 filters {b['channels']}을(를) 나누지 못함",
                           "groups는 두 채널 수의 공약수로 (depthwise는 DepthwiseConv2D 블록)")
                return None
            size = (lambda n: (n - d * (k - 1) - 1) // stride + 1) if valid else (lambda n: -(-n // stride))
            return self.grid(b["channels"], size(shape[1]), size(shape[2]), path)
        if t == "pool2d":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 AvgPool2D를 쓸 수 없음 (크기가 바뀜)", "AvgPool2D를 Residual block 밖으로 옮기세요")
                return None
            return self.grid(C, shape[1] // 2, shape[2] // 2, path)
        if t == "global_avg2d":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 GlobalAvgPool2D를 쓸 수 없음", "GlobalAvgPool2D를 Residual block 밖으로 옮기세요")
                return None
            return (C,)
        if t == "conv":
            stride, d, g = b.get("stride", 1), b.get("dilation", 1), b.get("groups", 1)
            valid = b.get("padding", "same") == "valid"
            if ctx == "residual" and (stride != 1 or valid):
                self.issue(path, "Residual block 안에서는 stride 2나 padding valid를 쓸 수 없음 (길이가 바뀜)",
                           "stride 1, padding same으로 하거나 Residual block 밖으로 옮기세요")
                return None
            if C % g or b["channels"] % g:
                self.issue(path, f"groups {g}가 들어오는 채널 {C}과 filters {b['channels']}을(를) 나누지 못함",
                           "groups는 두 채널 수의 공약수로 (depthwise는 groups = filters = 들어오는 채널)")
                return None
            out_len = (L - d * (b["kernel"] - 1) - 1) // stride + 1 if valid else -(-L // stride)
            return self.seq(b["channels"], out_len, path)
        if t == "pool":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 AvgPool1D를 쓸 수 없음 (길이가 바뀜)", "AvgPool1D를 Residual block 밖으로 옮기세요")
                return None
            return self.seq(C, L // b["size"], path)
        if t == "global_avg":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 GlobalAvgPool1D를 쓸 수 없음", "GlobalAvgPool1D를 Residual block 밖으로 옮기세요")
                return None
            return (C,)
        if t == "branch":
            lanes = b.get("branches")
            if not isinstance(lanes, list) or len(lanes) < 2 or not all(isinstance(x, list) for x in lanes):
                self.issue(path, "Concat branches는 갈래가 2개 이상이어야 함", "갈래를 추가하세요")
                return None
            total = 0
            for j, lane in enumerate(lanes):                   # 갈래는 크기를 바꾸면 안 됨 (Residual block과 같은 규칙)
                out = self.walk(lane, shape, path + (j,), "residual") if lane else shape
                if out is None or len(out) != len(shape) or tuple(out[1:]) != tuple(shape[1:]):
                    return None
                total += out[0]
            return (total,) + tuple(shape[1:])
        if t == "se":
            if C // b.get("ratio", 4) < 1:
                self.issue(path, f"SE block: 채널 {C}개를 ratio {b.get('ratio', 4)}로 나누면 0", "ratio를 채널 수보다 작게 하세요")
                return None
            return shape
        if t == "last_step":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 LastTimestep을 쓸 수 없음", "LastTimestep을 Residual block 밖으로 옮기세요")
                return None
            return (C,)
        if t in ("bn", "act", "dropout"):
            return shape
        if t == "flatten":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 Flatten을 쓸 수 없음", "Flatten을 Residual block 밖으로 옮기세요")
                return None
            return (_prod(shape),)
        if t == "dense":
            if ctx == "residual":
                self.issue(path, "Residual block 안에서는 Dense를 쓸 수 없음", "Dense를 Residual block 밖으로 옮기세요")
                return None
            return self.dense_like(shape, b["units"], path)
        if t == "residual":
            if not b.get("body"):
                self.issue(path, "Residual block 안이 비어 있음", "안에 층을 넣으세요")
                return None
            if len(shape) not in (2, 3):
                self.issue(path, "Residual block은 시간 축이나 (높이, 너비)가 있는 데이터에만",
                           "Residual block을 GlobalAvgPool·Flatten·Dense 앞으로 옮기세요")
                return None
            out = self.walk(b["body"], shape, path, "residual")
            return out if out is None or len(out) == len(shape) and tuple(out[1:]) == tuple(shape[1:]) else None
        if t == "repeat":
            body = b.get("body")
            if not body:
                self.issue(path, "Repeat 안이 비어 있음", "안에 층을 넣으세요")
                return None
            if any(isinstance(x, dict) and x.get("type") in RNN_TYPES + ("output",) for x in body):
                self.issue(path, "Repeat 안에는 LSTM·GRU·SimpleRNN·Output을 넣을 수 없음", "그 층을 Repeat 밖으로 옮기세요")
                return None
            n_issues = len(self.issues)
            for _ in range(b["times"]):                        # 펼친 것과 같은 규칙. 모양은 마지막 반복 기준
                del self.issues[n_issues:]
                shape = self.walk(body, shape, path, "residual" if ctx == "residual" else "repeat")
                if shape is None:
                    break
            return shape
        return shape

    def seq(self, C, L, path):
        if L < 1:
            self.issue(path, "길이가 0이 됨 (너무 많이 줄였음)", "AvgPool1D·stride 2를 줄이세요")
            return None
        if L > self.max_len:
            self.issue(path, f"길이 {L}이(가) 한도 {self.max_len:,}을(를) 넘음", "AvgPool1D로 길이를 줄이세요")
            return None
        return (C, L)

    def grid(self, C, H, W, path):
        if H < 1 or W < 1:
            self.issue(path, "크기가 0이 됨 (너무 많이 줄였음)", "AvgPool2D·stride 2·padding valid를 줄이세요")
            return None
        if H * W > self.max_len:
            self.issue(path, f"크기 {H}×{W}가 한도 {self.max_len:,}칸을 넘음", "AvgPool2D로 크기를 줄이세요")
            return None
        return (C, H, W)

    def det_head(self, shape, path):
        """탐지 출력: 바로 앞이 격자 모양 (채널, G, G)이어야 함 → 칸마다 1×1 필터로 (값 수, G, G)를 펼쳐 냄."""
        if shape is None:
            return None
        G = self.n_out.grid
        if len(shape) != 3 or shape[1] != G or shape[2] != G:
            now = "×".join(map(str, shape[1:])) if len(shape) == 3 else f"숫자 {_prod(shape)}개 (위치가 없음)"
            self.issue(path, f"탐지 모델의 출력 블록 바로 앞은 {G}×{G} 격자여야 함 (지금 {now})",
                       f"그림 {G * 8}×{G * 8}을 AvgPool2D로 세 번 줄이면 {G}×{G}가 됩니다. "
                       "GlobalAvgPool2D·Flatten·Dense는 위치를 잃어 탐지에 쓸 수 없습니다")
            return None
        return (int(self.n_out),)

    def dense_like(self, shape, units, path):
        if shape is None:
            return None
        width = _prod(shape)
        if width > self.max_width or units > self.max_width:
            pool = "GlobalAvgPool2D나 AvgPool2D" if self.image else "GlobalAvgPool1D나 AvgPool1D"
            self.issue(path, f"입력 {width:,}개가 한 층 한도 {self.max_width:,}을(를) 넘음", f"앞에 {pool}를 넣어 크기를 줄이세요")
            return None
        return (units,)


def _prod(shape) -> int:
    n = 1
    for v in shape:
        n *= v
    return n


def check(blocks: list, length: int, n_out: int, image: tuple = None) -> Checked:
    """블록 검증과 블록마다 출력 모양. length: 데이터 길이(이미지면 H·W), n_out: 분류 클래스 수(회귀면 1).
    image: 이미지면 (H, W) → kind "image", 시작 모양 (1, H, W)."""
    w = _Walker(n_out, image=bool(image))
    if not isinstance(blocks, list) or not blocks:
        w.issue((), "블록이 없음", "예시 모델을 불러오거나 블록을 쌓으세요")
        return Checked(w.issues, {}, "image" if image else "cnn")
    if image:
        kind = "image"
    else:
        kind = "rnn" if isinstance(blocks[0], dict) and blocks[0].get("type") in RNN_TYPES else "cnn"
    w.walk(blocks, (1,) + tuple(image) if image else (1, length), (), "top")
    for i, b in enumerate(blocks):                          # 증강은 맨 앞(증강끼리)에만
        if isinstance(b, dict) and b.get("type") in X.AUG_TYPES and any(
                not (isinstance(p, dict) and p.get("type") in X.AUG_TYPES) for p in blocks[:i]):
            w.issue((i,), f"{NAMES[b['type']]}은(는) Input 바로 뒤(다른 증강 블록 뒤)에만", "증강 블록을 맨 위로 옮기세요")
    outs = [i for i, b in enumerate(blocks) if isinstance(b, dict) and b.get("type") == "output"]
    if not outs:
        w.issue((), "출력 블록이 없음", "맨 아래에 출력 블록을 놓으세요")
    elif len(outs) > 1:
        for i in outs[:-1]:
            if not any(x.path == (i,) for x in w.issues):
                w.issue((i,), "출력 블록은 하나만", "출력 블록을 하나만 남기세요")
    return Checked(w.issues, w.shapes, kind)


# ---------------------------------------------------------------- torch 모델 만들기

def _torch():
    import torch
    from torch import nn
    return torch, nn


def to_model_input(kind: str, X, image: tuple = None):
    """(n, L) → CNN은 (n, 1, L), RNN은 (n, L, 1), 이미지는 (n, 1, H, W)."""
    import numpy as np
    X = np.asarray(X)
    if kind == "image":
        return X.reshape(len(X), 1, image[0], image[1])
    return X[:, None, :] if kind == "cnn" else X[:, :, None]


def _modules():
    torch, nn = _torch()

    class GlobalAvg(nn.Module):
        def forward(self, x):
            return x.mean(-1)

    class GlobalAvg2d(nn.Module):
        def forward(self, x):
            return x.mean(dim=(-1, -2))

    class Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)

    class Residual(nn.Module):
        def __init__(self, body, shortcut=None):
            super().__init__()
            self.body = body
            self.shortcut = shortcut

        def forward(self, x):
            s = x if self.shortcut is None else self.shortcut(x)
            return s + self.body(x)

    class SE(nn.Module):
        """Squeeze-and-Excitation: 채널마다 평균 → Dense → act → Dense → sigmoid → 채널 가중치를 곱함."""

        def __init__(self, c, ratio, act):
            super().__init__()
            h = max(1, c // ratio)
            self.fc1, self.act, self.fc2 = nn.Linear(c, h), act, nn.Linear(h, c)

        def forward(self, x):
            s = torch.sigmoid(self.fc2(self.act(self.fc1(x.flatten(2).mean(-1)))))
            return x * s.reshape(*s.shape, *([1] * (x.dim() - 2)))

    class Branch(nn.Module):
        """갈래마다 계산해 채널 방향으로 이어 붙임 (Inception). 빈 갈래는 들어온 값 그대로."""

        def __init__(self, lanes):
            super().__init__()
            self.lanes = nn.ModuleList(lanes)

        def forward(self, x):
            return torch.cat([lane(x) for lane in self.lanes], dim=1)

    class LastTimestep(nn.Module):
        def forward(self, x):
            return x[:, :, -1]

    class LastStep(nn.Module):
        def __init__(self, rnn):
            super().__init__()
            self.rnn = rnn

        def forward(self, x):
            out, _ = self.rnn(x)
            return out[:, -1]

    class StudioNet(nn.Module):
        """블록으로 만든 모델. block_modules: 블록 위치 키("0", "2.0.1") → 그 블록 출력을 내는 모듈 (등록 안 한 dict)."""

        def __init__(self, layers, kind: str, block_modules: dict):
            super().__init__()
            self.layers = nn.Sequential(*layers)
            self.kind = kind
            self.block_modules = block_modules

        def forward(self, x):
            return self.layers(x.transpose(1, 2) if self.kind == "rnn" else x)

    return GlobalAvg, Flatten, Residual, LastStep, StudioNet, LastTimestep, SE, Branch, GlobalAvg2d


def act_module(kind: str):
    """활성화 이름 → 모듈."""
    torch, nn = _torch()
    if kind == "softmax":
        return nn.Softmax(dim=1)
    return {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
            "silu": nn.SiLU, "elu": nn.ELU, "mish": nn.Mish, "softplus": nn.Softplus, "selu": nn.SELU, "relu6": nn.ReLU6,
            "hardswish": nn.Hardswish, "hardsigmoid": nn.Hardsigmoid}[kind]()


def make_layer(b: dict, shape, n_out: int):
    """감싸지 않는 층 하나(b: 블록 설정, shape: 들어오는 모양) → torch 모듈. 블록 목록과 그래프가 함께 쓴다."""
    torch, nn = _torch()
    GlobalAvg, Flatten, Residual, LastStep, StudioNet, LastTimestep, SE, Branch, GlobalAvg2d = _modules()
    t = b["type"]
    if t == "conv":
        k, d = b["kernel"], b.get("dilation", 1)
        if b.get("padding") == "causal":
            from dlstudio.nn_extra import CausalConv1d
            return CausalConv1d(shape[0], b["channels"], k, stride=b.get("stride", 1), dilation=d, groups=b.get("groups", 1))
        pad = 0 if b.get("padding", "same") == "valid" else d * (k - 1) // 2
        return nn.Conv1d(shape[0], b["channels"], k, stride=b.get("stride", 1), padding=pad, dilation=d,
                         groups=b.get("groups", 1))
    if t == "pool":
        return nn.AvgPool1d(b["size"])
    if t == "conv2d":
        k, d = b["kernel"], b.get("dilation", 1)
        return nn.Conv2d(shape[0], b["channels"], k, stride=b.get("stride", 1),
                         padding=0 if b.get("padding", "same") == "valid" else d * (k - 1) // 2, dilation=d,
                         groups=b.get("groups", 1))
    if t == "pool2d":
        return nn.AvgPool2d(2)
    if t == "global_avg2d":
        return GlobalAvg2d()
    if t == "global_avg":
        return GlobalAvg()
    if t == "last_step":
        return LastTimestep()
    if t == "se":
        return SE(shape[0], b.get("ratio", 4), act_module(b.get("act", "relu")))
    if t == "dropout":
        return nn.Dropout(float(b.get("rate", 0.5)))
    if t == "flatten":
        return Flatten()
    if t == "bn":
        return nn.BatchNorm2d(shape[0]) if len(shape) == 3 else nn.BatchNorm1d(shape[0])
    if t == "act":
        return act_module(b["kind"])
    if t == "output" and getattr(n_out, "grid", None):           # 탐지: 격자 칸마다 1×1 필터 → (값 수, G, G)를 펼침
        return nn.Sequential(nn.Conv2d(shape[0], n_out.per_cell, 1), Flatten())
    if t in ("dense", "output"):
        lin = nn.Linear(_prod(shape), n_out if t == "output" else b["units"])
        return nn.Sequential(Flatten(), lin) if len(shape) >= 2 else lin
    if t in RNN_TYPES:
        from dlstudio.nn_extra import RNNBlock
        return RNNBlock(t, shape[0], b["hidden"], b.get("layers", 1), bool(b.get("bidirectional")), bool(b.get("return_sequences")))
    if t in X.EXTRA_TYPES:
        return extra_layer(b, shape)
    raise ValueError(f"알 수 없는 층 {t}")


def extra_layer(b: dict, shape, shape2=None):
    """추가 블록 → PyTorch 모듈 (layers.torch_args의 이름·인자로)."""
    torch, nn = _torch()
    from dlstudio import nn_extra
    name, kw = X.torch_args(b["type"], b, shape, shape2)
    cls = getattr(nn, name[3:]) if name.startswith("nn.") else getattr(nn_extra, name)
    return cls(*kw.get("_", []), **{k: v for k, v in kw.items() if k != "_"})


def build_model(blocks: list, length: int, n_out: int, seed: int = 0, image: tuple = None):
    """블록 JSON → StudioNet (float32). 검증을 통과하지 못하면 ValueError(첫 문제). image: 이미지면 (H, W)."""
    c = check(blocks, length, n_out, image=image)
    if not c.ok:
        raise ValueError(c.issues[0].message)
    torch, nn = _torch()
    GlobalAvg, Flatten, Residual, LastStep, StudioNet, LastTimestep, SE, Branch, GlobalAvg2d = _modules()
    torch.manual_seed(seed)
    paths: dict = {}

    def build(seq: list, shape, prefix: tuple, shapes: dict):
        """shapes: 블록 위치 → 출력 모양. Repeat 몸체는 반복마다 모양이 달라 반복마다 따로 계산한 것을 씀."""
        mods = []
        for i, b in enumerate(seq):
            path = prefix + (i,)
            t = b["type"]
            if t == "branch":
                m = Branch([build(lane, shape, path + (j,), shapes)[0] if lane else nn.Identity()
                            for j, lane in enumerate(b["branches"])])
            elif t == "residual":
                body, out = build(b["body"], shape, path, shapes)
                conv = nn.Conv2d if len(shape) == 3 else nn.Conv1d
                m = Residual(body, None if out[0] == shape[0] else conv(shape[0], out[0], 1))
            elif t == "repeat":
                reps, s = [], shape
                for _ in range(b["times"]):                     # 가중치는 반복마다 따로, 대응표는 마지막 반복
                    w = _Walker(n_out, image=bool(image))       # 이번 반복의 입력 모양에서 몸체 모양을 다시 계산
                    w.walk(b["body"], s, path, "repeat")
                    body, s = build(b["body"], s, path, w.shapes)
                    reps.append(body)
                m = nn.Sequential(*reps)
            else:
                m = make_layer(b, shape, n_out)
            paths[".".join(map(str, path))] = m
            mods.append(m)
            shape = shapes[path]
        return nn.Sequential(*mods), shape

    seq, _ = build(blocks, (1,) + tuple(image) if image else (1, length), (), c.shapes)
    return StudioNet(list(seq), c.kind, paths).float()

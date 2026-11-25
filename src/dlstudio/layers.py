"""추가 블록 정의 (torch 없이): 설정 규칙 · 모양 계산 · 가중치 목록 · PyTorch 모듈 이름과 인자.

blocks._Walker(모양 검사), codegen.param_specs(가중치 이름·모양, 웹 엔진이 씀), blocks.make_layer(PyTorch 모듈),
codegen(코드 내보내기)이 모두 여기서 읽는다. PyTorch 모듈 자체는 nn_extra.py (생성 코드에도 그 소스를 그대로 넣음).

모양 (배치 제외, 채널 먼저): 숫자 묶음 (n,) · 줄 (C, L) · 그림 (C, H, W).
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- 종류

EXTRA_TYPES = (
    # 완전연결
    "glu",
    # 합성곱
    "dw_conv", "dw_conv2d", "sep_conv", "sep_conv2d",
    # 줄이기·모으기
    "max_pool", "max_pool2d", "global_max", "global_max2d", "adaptive_pool", "first_step", "attn_pool",
    # 키우기
    "upsample", "pixel_shuffle", "conv_t", "conv_t2d",
    # 어텐션
    "eca", "cbam", "spatial_attn", "mha", "linear_attn", "nonlocal", "cross_attn",
    # Transformer 부품
    "pos_enc", "patch_embed", "cls_token",
    # 정규화
    "layernorm", "rmsnorm", "groupnorm", "instancenorm",
    # 활성화 (배우는 값이 있는 것)
    "prelu",
    # 과적합 막기
    "spatial_dropout", "drop_path", "gaussian_noise",
    # 나누기
    "slice",
    # 모양 바꾸기
    "reshape", "permute", "to_seq", "crop", "pad",
    # 입력 변환·증강
    "embedding", "spectrogram", "rand_flip", "rand_rotate", "rand_shift", "rand_zoom", "rand_scale", "rand_stretch",
    # 사전학습 (가중치 고정)
    "mnv3s",
)
PRETRAINED_TYPES = ("mnv3s",)                               # 가중치를 pretrained/<종류>.bin에서 읽고 학습하지 않음
# Google MobileNetV3-Small 특징 추출기 (nn_extra.MobileNetV3Small.CFG와 같음): (들어옴, kernel, 넓힘, 나감, SE 채널, HardSwish인지, stride)
MNV3S_CFG = ((16, 3, 16, 16, 8, 0, 2), (16, 3, 72, 24, 0, 0, 2), (24, 3, 88, 24, 0, 0, 1), (24, 5, 96, 40, 24, 1, 2),
             (40, 5, 240, 40, 64, 1, 1), (40, 5, 240, 40, 64, 1, 1), (40, 5, 120, 48, 32, 1, 1), (48, 5, 144, 48, 40, 1, 1),
             (48, 5, 288, 96, 72, 1, 2), (96, 5, 576, 96, 144, 1, 1), (96, 5, 576, 96, 144, 1, 1))
MNV3S_OUT = 1024
DIST_K = 128                                                # 이상 탐지 출력 '정상과의 거리': 주성분 수 (특징이 적으면 그 수)


def dist_k(c: int) -> int:
    return min(DIST_K, int(c))
NEW_MERGES = ("multiply", "average", "maximum")             # 모양이 같은 값들을 원소마다 곱·평균·최댓값
TWO_INPUT = ("cross_attn",)                                 # 입력이 정확히 둘 (질문, 참고)
RANDOM_TYPES = ("spatial_dropout", "drop_path", "gaussian_noise", "rand_flip", "rand_rotate", "rand_shift", "rand_zoom",
                "rand_scale", "rand_stretch")                # 학습 때만 무작위 (예측 때는 그대로 통과)
AUG_TYPES = ("rand_flip", "rand_rotate", "rand_shift", "rand_zoom", "rand_scale", "rand_stretch")   # Input 바로 뒤(증강끼리)만

EXTRA_NAMES = {
    "glu": "GLU", "dw_conv": "DepthwiseConv1D", "dw_conv2d": "DepthwiseConv2D", "sep_conv": "SeparableConv1D",
    "sep_conv2d": "SeparableConv2D", "max_pool": "MaxPool1D", "max_pool2d": "MaxPool2D", "global_max": "GlobalMaxPool1D",
    "global_max2d": "GlobalMaxPool2D", "adaptive_pool": "AdaptivePool", "first_step": "FirstTimestep",
    "attn_pool": "AttentionPool", "upsample": "Upsample", "pixel_shuffle": "PixelShuffle", "conv_t": "ConvTranspose1D",
    "conv_t2d": "ConvTranspose2D", "eca": "ECA", "cbam": "CBAM", "spatial_attn": "SpatialAttention",
    "mha": "MultiHeadAttention", "linear_attn": "LinearAttention", "nonlocal": "NonLocal", "cross_attn": "CrossAttention",
    "pos_enc": "PositionalEncoding", "patch_embed": "PatchEmbedding", "cls_token": "CLS token", "layernorm": "LayerNorm",
    "rmsnorm": "RMSNorm", "groupnorm": "GroupNorm", "instancenorm": "InstanceNorm", "prelu": "PReLU",
    "spatial_dropout": "SpatialDropout", "drop_path": "DropPath", "gaussian_noise": "GaussianNoise", "slice": "Split",
    "reshape": "Reshape", "permute": "Transpose", "to_seq": "ToSequence", "crop": "Crop", "pad": "Pad",
    "embedding": "Embedding", "spectrogram": "Spectrogram", "rand_flip": "RandomFlip", "rand_rotate": "RandomRotation",
    "rand_shift": "RandomShift", "rand_zoom": "RandomZoom", "rand_scale": "RandomScale", "rand_stretch": "RandomStretch",
    "multiply": "Multiply", "average": "Average", "maximum": "Maximum", "mnv3s": "MobileNetV3-Small (사전학습)",
}

# 정수 설정: (키, 최소, 최대, 기본값, 허용 값, 설명) — blocks.PARAMS와 같은 꼴
EXTRA_PARAMS = {
    "glu": [("units", 1, 1024, None, None, "units는 1–1024")],
    "dw_conv": [("kernel", 1, 31, 3, tuple(range(1, 32, 2)), "kernel size는 1–31의 홀수"),
                ("multiplier", 1, 8, 1, None, "multiplier는 1–8")],
    "dw_conv2d": [("kernel", 1, 7, 3, (1, 3, 5, 7), "kernel size는 1·3·5·7"), ("multiplier", 1, 8, 1, None, "multiplier는 1–8")],
    "sep_conv": [("channels", 1, 256, None, None, "filters는 1–256"), ("kernel", 1, 31, 3, tuple(range(1, 32, 2)), "kernel size는 1–31의 홀수")],
    "sep_conv2d": [("channels", 1, 256, None, None, "filters는 1–256"), ("kernel", 1, 7, 3, (1, 3, 5, 7), "kernel size는 1·3·5·7")],
    "max_pool": [("size", 2, 4, 2, (2, 4), "pool size는 2 또는 4")],
    "max_pool2d": [("size", 2, 2, 2, (2,), "pool size는 2")],
    "adaptive_pool": [("size", 1, 64, 4, None, "출력 크기는 1–64")],
    "upsample": [("scale", 2, 2, 2, (2,), "배율은 2")],
    "pixel_shuffle": [("scale", 2, 2, 2, (2,), "배율은 2")],
    "conv_t": [("channels", 1, 256, None, None, "filters는 1–256"), ("kernel", 1, 5, 2, (1, 2, 3, 4, 5), "kernel size는 1–5"),
               ("stride", 1, 2, 2, (1, 2), "stride는 1 또는 2")],
    "conv_t2d": [("channels", 1, 256, None, None, "filters는 1–256"), ("kernel", 1, 5, 2, (1, 2, 3, 4, 5), "kernel size는 1–5"),
                 ("stride", 1, 2, 2, (1, 2), "stride는 1 또는 2")],
    "eca": [("kernel", 3, 7, 3, (3, 5, 7), "kernel size는 3·5·7")],
    "cbam": [("ratio", 1, 64, 4, None, "ratio는 1–64"), ("kernel", 3, 7, 7, (3, 5, 7), "kernel size는 3·5·7")],
    "spatial_attn": [("kernel", 3, 7, 7, (3, 5, 7), "kernel size는 3·5·7")],
    "mha": [("heads", 1, 8, 2, (1, 2, 4, 8), "heads는 1·2·4·8")],
    "linear_attn": [("heads", 1, 8, 2, (1, 2, 4, 8), "heads는 1·2·4·8")],
    "cross_attn": [("heads", 1, 8, 2, (1, 2, 4, 8), "heads는 1·2·4·8")],
    "patch_embed": [("patch", 2, 8, 4, (2, 4, 8), "조각 크기는 2·4·8"), ("dim", 1, 256, 32, None, "dim은 1–256")],
    "groupnorm": [("groups", 1, 32, 4, (1, 2, 4, 8, 16, 32), "groups는 1·2·4·8·16·32")],
    "slice": [("start", 0, 1024, 0, None, "시작 채널은 0 이상"), ("count", 1, 1024, None, None, "채널 수는 1 이상")],
    "reshape": [("channels", 1, 65536, None, None, "채널 수는 1 이상")],
    "crop": [("amount", 1, 64, 1, None, "자르는 칸은 1–64")],
    "pad": [("amount", 1, 64, 1, None, "채우는 칸은 1–64")],
    "rand_rotate": [("degrees", 1, 180, 15, None, "각도는 1–180")],
    "embedding": [("bins", 2, 256, 32, (8, 16, 32, 64, 128, 256), "구간 수는 8·16·32·64·128·256"),
                  ("dim", 1, 256, 16, None, "dim은 1–256")],
    "spectrogram": [("n_fft", 8, 256, 32, (8, 16, 32, 64, 128, 256), "창 길이(n_fft)는 8·16·32·64·128·256"),
                    ("hop", 1, 128, 8, (1, 2, 4, 8, 16, 32, 64, 128), "hop은 1·2·4·…·128"),
                    ("n_mels", 4, 64, 16, (8, 16, 32, 64), "mel 칸 수는 8·16·32·64"),
                    ("n_mfcc", 4, 40, 13, (8, 13, 20, 40), "MFCC 개수는 8·13·20·40")],
    "mnv3s": [("size", 64, 224, 128, (64, 96, 128, 160, 224), "크기는 64·96·128·160·224")],
}
# 실수 설정: (키, 최소(포함), 최대(미포함), 기본값, 설명)
FLOAT_PARAMS = {
    "spatial_dropout": [("rate", 0.0, 1.0, 0.1, "rate는 0 이상 1 미만")],
    "drop_path": [("rate", 0.0, 1.0, 0.1, "rate는 0 이상 1 미만")],
    "gaussian_noise": [("std", 0.0, 5.0, 0.1, "std는 0 이상 5 미만")],
    "rand_shift": [("frac", 0.0, 0.5, 0.1, "비율은 0 이상 0.5 미만")],
    "rand_zoom": [("frac", 0.0, 0.5, 0.1, "비율은 0 이상 0.5 미만")],
    "rand_scale": [("frac", 0.0, 0.9, 0.2, "비율은 0 이상 0.9 미만")],
    "rand_stretch": [("frac", 0.0, 0.5, 0.1, "비율은 0 이상 0.5 미만")],
}
# 글자·참거짓 설정: 키 → (허용 값, 기본값, 설명)
STR_PARAMS = {
    "glu": {"gate": (("sigmoid", "silu"), "sigmoid", "gate는 sigmoid(GLU) 또는 silu(SwiGLU)")},
    "adaptive_pool": {"mode": (("avg", "max"), "avg", "mode는 avg 또는 max")},
    "upsample": {"mode": (("nearest", "linear"), "nearest", "mode는 nearest 또는 linear")},
    "mha": {"causal": ((False, True), False, "causal은 참·거짓"), "pos": (("none", "rope"), "none", "pos는 none 또는 rope")},
    "pos_enc": {"mode": (("sin", "learned"), "sin", "mode는 sin 또는 learned")},
    "rand_flip": {"dir": (("h", "v", "hv"), "h", "방향은 h·v·hv")},
    "spectrogram": {"mode": (("log", "mel", "mfcc"), "log", "mode는 log·mel·mfcc")},
}

TRANSPOSE_SPECS = {"conv_t": 1, "conv_t2d": 2}
# ConvTranspose (stride 2): kernel → (padding, output_padding) — 출력이 정확히 2배가 되게
T_PAD = {1: (0, 1), 2: (0, 0), 3: (1, 1), 4: (1, 0), 5: (2, 1)}


def name_of(t: str) -> str:
    return EXTRA_NAMES.get(t, t)


def values_ok(w, b: dict, path) -> bool:
    """정수 밖의 설정(실수·글자·참거짓) 검사. 정수는 blocks.PARAMS(EXTRA_PARAMS가 합쳐짐)가 함."""
    ok = True
    t = b["type"]
    for key, lo, hi, default, desc in FLOAT_PARAMS.get(t, []):
        v = b.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v < hi:
            w.issue(path, f"{name_of(t)}: {key} 값 {v!r}이(가) 맞지 않음", desc)
            ok = False
    for key, (choices, default, desc) in STR_PARAMS.get(t, {}).items():
        v = b.get(key, default)
        if v not in choices or (isinstance(v, bool) != isinstance(default, bool)):
            w.issue(path, f"{name_of(t)}: {key} 값 {v!r}이(가) 맞지 않음", desc)
            ok = False
    return ok


def opt(b: dict, key: str):
    """설정 값 (없으면 기본값)."""
    t = b["type"]
    for k, lo, hi, d, ch, desc in EXTRA_PARAMS.get(t, []):
        if k == key:
            return b.get(key, d)
    for k, lo, hi, d, desc in FLOAT_PARAMS.get(t, []):
        if k == key:
            return b.get(key, d)
    if key in STR_PARAMS.get(t, {}):
        return b.get(key, STR_PARAMS[t][key][1])
    return b.get(key)


def _prod(s) -> int:
    n = 1
    for v in s:
        n *= v
    return n


# ---------------------------------------------------------------- 모양

SEQ_HINT = "이미지는 ToSequence·PatchEmbedding 뒤에, 숫자 묶음(Dense·GlobalPool 뒤)에는 쓸 수 없음"
IMG_HINT = "이미지(채널, 높이, 너비)에만. 시계열은 Spectrogram 뒤에는 됨"
SPATIAL_HINT = "시간 축이나 (높이, 너비)가 있는 값에만. GlobalPool·Flatten·Dense 앞으로 옮기세요"


def _need(w, t, shape, ranks, path) -> bool:
    want = {2: "(채널, 길이)", 3: "(채널, 높이, 너비)", 1: "숫자 묶음"}
    if len(shape) in ranks:
        return True
    hint = SEQ_HINT if ranks == (2,) else IMG_HINT if ranks == (3,) else SPATIAL_HINT
    w.issue(path, f"{name_of(t)}은(는) {' 또는 '.join(want[r] for r in ranks)} 모양에만 쓸 수 있음 (지금 {tuple(shape)})", hint)
    return False


def _spatial(shape):
    return _prod(shape[1:])


def shape(w, b: dict, shape_in, path, ctx):
    """추가 블록 하나의 출력 모양 (문제가 있으면 w.issue 후 None). w: blocks._Walker."""
    t = b["type"]
    s = tuple(shape_in)
    C = s[0]
    rank = len(s)
    o = lambda k: opt(b, k)
    if t in ("layernorm", "rmsnorm", "prelu", "drop_path", "gaussian_noise", "rand_scale"):
        return s
    if t == "glu":
        return (o("units"),) + s[1:]
    if t == "slice":
        if o("start") + o("count") > C:
            w.issue(path, f"Split: 채널 {o('start')}부터 {o('count')}개는 들어온 채널 {C}개를 넘음", f"시작 + 개수를 {C} 이하로")
            return None
        return (o("count"),) + s[1:]
    if t == "reshape":
        n = _prod(s)
        if n % o("channels"):
            w.issue(path, f"Reshape: 값 {n}개를 채널 {o('channels')}개로 나눌 수 없음", f"{n}의 약수로 정하세요")
            return None
        return (o("channels"), n // o("channels"))
    if t in ("rand_flip", "rand_shift"):
        return s if _need(w, t, s, (2, 3), path) else None
    if t in ("rand_rotate", "rand_zoom"):
        return s if _need(w, t, s, (3,), path) else None
    if t == "rand_stretch":
        return s if _need(w, t, s, (2,), path) else None
    if t == "embedding":
        if not _need(w, t, s, (2, 3), path):
            return None
        if C != 1:
            w.issue(path, f"Embedding은 채널이 1개인 값에만 (지금 {C}개)", "Input 바로 뒤에 두세요")
            return None
        return (o("dim"),) + s[1:]
    if t == "mnv3s":
        if not _need(w, t, s, (3,), path):
            return None
        if C not in (1, 3):
            w.issue(path, f"MobileNetV3-Small은 흑백(채널 1)이나 컬러(채널 3) 그림에만 (지금 채널 {C}개)", "Input 바로 뒤에 두세요")
            return None
        return (MNV3S_OUT,)
    if t == "spectrogram":
        if not _need(w, t, s, (2,), path):
            return None
        if C != 1:
            w.issue(path, f"Spectrogram은 채널이 1개인 신호에만 (지금 {C}개)", "Input 바로 뒤에 두세요")
            return None
        n_fft, hop = o("n_fft"), o("hop")
        if s[1] < n_fft:
            w.issue(path, f"Spectrogram: 길이 {s[1]}이(가) 창 길이 {n_fft}보다 짧음", "n_fft를 줄이세요")
            return None
        T = 1 + (s[1] - n_fft) // hop
        F = n_fft // 2 + 1 if o("mode") == "log" else o("n_mels") if o("mode") == "mel" else o("n_mfcc")
        if o("mode") == "mfcc" and o("n_mfcc") > o("n_mels"):
            w.issue(path, "Spectrogram: MFCC 개수는 mel 칸 수 이하", "n_mfcc를 줄이거나 n_mels를 늘리세요")
            return None
        return w.grid(1, F, T, path)
    # ---- 아래는 (채널, 길이) 또는 (채널, 높이, 너비)
    if t in ("dw_conv", "sep_conv", "max_pool", "global_max", "first_step", "conv_t", "cls_token"):
        if not _need(w, t, s, (2,), path):
            return None
    elif t in ("dw_conv2d", "sep_conv2d", "max_pool2d", "global_max2d", "conv_t2d", "patch_embed", "to_seq"):
        if not _need(w, t, s, (3,), path):
            return None
    elif not _need(w, t, s, (2, 3), path):
        return None
    sp = s[1:]
    if t == "dw_conv":
        return (C * o("multiplier"), s[1])
    if t == "dw_conv2d":
        return (C * o("multiplier"),) + sp
    if t == "sep_conv":
        return (o("channels"), s[1])
    if t == "sep_conv2d":
        return (o("channels"),) + sp
    if t == "max_pool":
        return w.seq(C, s[1] // o("size"), path)
    if t == "max_pool2d":
        return w.grid(C, s[1] // 2, s[2] // 2, path)
    if t in ("global_max", "global_max2d", "first_step", "attn_pool"):
        return (C,)
    if t == "adaptive_pool":
        return (C,) + (o("size"),) * (rank - 1)
    if t == "upsample":
        return (C,) + tuple(v * 2 for v in sp)
    if t == "pixel_shuffle":
        r = 2 ** (rank - 1)
        if C % r:
            w.issue(path, f"PixelShuffle: 채널 {C}개가 {r}로 나누어떨어지지 않음", f"앞 층의 채널을 {r}의 배수로")
            return None
        return (C // r,) + tuple(v * 2 for v in sp)
    if t in ("conv_t", "conv_t2d"):
        k, st = o("kernel"), o("stride")
        if st == 1 and k % 2 == 0:
            w.issue(path, f"{name_of(t)}: stride 1이면 kernel은 홀수", "kernel을 1·3·5로 하거나 stride 2로")
            return None
        return (o("channels"),) + tuple(v * st for v in sp)
    if t == "eca":
        return s
    if t in ("cbam", "spatial_attn"):
        if t == "cbam" and C // o("ratio") < 1:
            w.issue(path, f"CBAM: 채널 {C}개를 ratio {o('ratio')}로 나누면 0", "ratio를 채널 수보다 작게")
            return None
        return s
    if t in ("mha", "linear_attn"):
        h = o("heads")
        if C % h:
            w.issue(path, f"{name_of(t)}: 채널 {C}개를 heads {h}로 나눌 수 없음", "heads를 채널 수의 약수로")
            return None
        if t == "mha" and o("pos") == "rope" and (C // h) % 2:
            w.issue(path, f"MultiHeadAttention: RoPE는 head마다 채널이 짝수여야 함 (지금 {C // h})", "채널이나 heads를 바꾸세요")
            return None
        if _spatial(s) > 4096:
            w.issue(path, f"{name_of(t)}: 위치 {_spatial(s):,}개는 너무 많음 (4,096개까지)", "Pool·PatchEmbedding으로 줄인 뒤에 쓰세요")
            return None
        return s
    if t == "nonlocal":
        if _spatial(s) > 4096:
            w.issue(path, f"NonLocal: 위치 {_spatial(s):,}개는 너무 많음 (4,096개까지)", "Pool로 줄인 뒤에 쓰세요")
            return None
        return s
    if t == "pos_enc":
        return s
    if t == "patch_embed":
        p = o("patch")
        if s[1] < p or s[2] < p:
            w.issue(path, f"PatchEmbedding: 그림 {s[1]}×{s[2]}이(가) 조각 {p}보다 작음", "조각 크기를 줄이세요")
            return None
        return (o("dim"), (s[1] // p) * (s[2] // p))
    if t == "cls_token":
        return (C, s[1] + 1)
    if t == "groupnorm":
        if C % o("groups"):
            w.issue(path, f"GroupNorm: 채널 {C}개를 groups {o('groups')}로 나눌 수 없음", "groups를 채널 수의 약수로")
            return None
        return s
    if t in ("instancenorm", "spatial_dropout"):
        return s
    if t == "permute":
        return (s[1], s[0]) if rank == 2 else (C, s[2], s[1])
    if t == "to_seq":
        return (C, s[1] * s[2])
    if t in ("crop", "pad"):
        a = o("amount") * (-2 if t == "crop" else 2)
        if any(v + a < 1 for v in sp):
            w.issue(path, f"Crop: 크기 {sp}에서 양쪽 {o('amount')}칸씩 자르면 남는 것이 없음", "amount를 줄이세요")
            return None
        return (C,) + tuple(v + a for v in sp)
    return s


def two_shape(issue, b: dict, q, ctx):
    """입력이 둘인 블록 (CrossAttention): 질문 q, 참고 ctx의 모양 → 출력 모양 또는 None (issue(message, hint) 후)."""
    if len(q) not in (2, 3) or len(ctx) not in (2, 3):
        issue("CrossAttention: 두 입력 모두 (채널, 길이)나 (채널, 높이, 너비)여야 함", SPATIAL_HINT)
        return None
    h = opt(b, "heads")
    if q[0] % h:
        issue(f"CrossAttention: 첫 입력(질문)의 채널 {q[0]}개를 heads {h}로 나눌 수 없음", "heads를 채널 수의 약수로")
        return None
    return tuple(q)


# ---------------------------------------------------------------- 가중치 (PyTorch state_dict 이름·모양 + 처음 값)
# 처음 값: uniform(기본: ±1/√fan_in, PyTorch Linear·Conv와 같음) · ones · zeros · normal1 · normal02 · const0.25

def params(t: str, b: dict, ins, out, k: str, ins2=None) -> list:
    o = lambda key: opt(b, key)
    W = lambda name, *shape: (f"{k}.{name}", list(shape), "weight", "uniform")
    C = ins[0]
    rank = len(ins)
    if t == "glu":
        return [W("proj.weight", 2 * o("units"), C), W("proj.bias", 2 * o("units"))]
    if t in ("dw_conv", "dw_conv2d"):
        kk = (o("kernel"),) * (2 if t == "dw_conv2d" else 1)
        return [W("weight", C * o("multiplier"), 1, *kk), W("bias", C * o("multiplier"))]
    if t in ("sep_conv", "sep_conv2d"):
        two = t == "sep_conv2d"
        kk, one = (o("kernel"),) * (2 if two else 1), (1,) * (2 if two else 1)
        return [W("depthwise.weight", C, 1, *kk), W("depthwise.bias", C),
                W("pointwise.weight", o("channels"), C, *one), W("pointwise.bias", o("channels"))]
    if t in ("conv_t", "conv_t2d"):
        kk = (o("kernel"),) * (2 if t == "conv_t2d" else 1)
        return [W("weight", C, o("channels"), *kk), W("bias", o("channels"))]
    if t == "attn_pool":
        return [W("score.weight", 1, C), W("score.bias", 1)]
    if t == "eca":
        return [W("conv.weight", 1, 1, o("kernel"))]
    if t == "cbam":
        h = max(1, C // o("ratio"))
        kk = (o("kernel"),) * (rank - 1)
        return [W("fc1.weight", h, C), W("fc1.bias", h), W("fc2.weight", C, h), W("fc2.bias", C),
                W("conv.weight", 1, 2, *kk), W("conv.bias", 1)]
    if t == "spatial_attn":
        kk = (o("kernel"),) * (rank - 1)
        return [W("conv.weight", 1, 2, *kk), W("conv.bias", 1)]
    if t in ("mha", "linear_attn"):
        return [W("qkv.weight", 3 * C, C), W("qkv.bias", 3 * C), W("proj.weight", C, C), W("proj.bias", C)]
    if t == "nonlocal":
        h = max(1, C // 2)
        return [W("theta.weight", h, C), W("theta.bias", h), W("phi.weight", h, C), W("phi.bias", h),
                W("g.weight", h, C), W("g.bias", h), W("out.weight", C, h), W("out.bias", C)]
    if t == "cross_attn":
        C2 = ins2[0]
        return [W("q.weight", C, C), W("q.bias", C), W("kv.weight", 2 * C, C2), W("kv.bias", 2 * C),
                W("proj.weight", C, C), W("proj.bias", C)]
    if t == "pos_enc":
        return [(f"{k}.pe", [C, _prod(ins[1:])], "weight", "normal02")] if o("mode") == "learned" else []
    if t == "patch_embed":
        p = o("patch")
        return [W("proj.weight", o("dim"), C, p, p), W("proj.bias", o("dim"))]
    if t == "cls_token":
        return [(f"{k}.cls", [C], "weight", "normal02")]
    if t in ("layernorm", "groupnorm", "instancenorm"):
        return [(f"{k}.weight", [C], "weight", "ones"), (f"{k}.bias", [C], "weight", "zeros")]
    if t == "rmsnorm":
        return [(f"{k}.weight", [C], "weight", "ones")]
    if t == "prelu":
        return [(f"{k}.weight", [1], "weight", "const0.25")]
    if t == "embedding":
        return [(f"{k}.weight", [o("bins"), o("dim")], "weight", "normal1")]
    if t == "mnv3s":
        return mnv3s_params(k)
    return []


def mnv3s_params(k: str) -> list:
    """MobileNetV3-Small 가중치 (종류 frozen: 학습하지 않음, 처음 값은 사전학습 파일)."""
    Fz = lambda name, *shape: (f"{k}.{name}", list(shape), "frozen", "pretrained")
    out = [Fz("stem.weight", 16, 3, 3, 3), Fz("stem.bias", 16)]
    for j, (i, kk, e, o, q, hs, s) in enumerate(MNV3S_CFG):
        p = f"blocks.{j}."
        if e != i:
            out += [Fz(p + "expand.weight", e, i, 1, 1), Fz(p + "expand.bias", e)]
        out += [Fz(p + "dw.weight", e, 1, kk, kk), Fz(p + "dw.bias", e)]
        if q:
            out += [Fz(p + "se1.weight", q, e, 1, 1), Fz(p + "se1.bias", q), Fz(p + "se2.weight", e, q, 1, 1), Fz(p + "se2.bias", e)]
        out += [Fz(p + "project.weight", o, e, 1, 1), Fz(p + "project.bias", o)]
    return out + [Fz("last.weight", 576, 96, 1, 1), Fz("last.bias", 576),
                  Fz("head.weight", MNV3S_OUT, 576, 1, 1), Fz("head.bias", MNV3S_OUT)]


# ---------------------------------------------------------------- PyTorch 모듈: (이름, 인자) — "nn.X"는 torch 내장, 나머지는 nn_extra

def torch_args(t: str, b: dict, ins, ins2=None):
    o = lambda key: opt(b, key)
    C, rank = ins[0], len(ins)
    if t == "glu":
        return "GLU", {"c": C, "units": o("units"), "gate": o("gate")}
    if t == "dw_conv":
        return "nn.Conv1d", {"_": [C, C * o("multiplier"), o("kernel")], "padding": o("kernel") // 2, "groups": C}
    if t == "dw_conv2d":
        return "nn.Conv2d", {"_": [C, C * o("multiplier"), o("kernel")], "padding": o("kernel") // 2, "groups": C}
    if t in ("sep_conv", "sep_conv2d"):
        return "SeparableConv", {"c": C, "out": o("channels"), "k": o("kernel"), "dims": 2 if t == "sep_conv2d" else 1}
    if t == "max_pool":
        return "nn.MaxPool1d", {"_": [o("size")]}
    if t == "max_pool2d":
        return "nn.MaxPool2d", {"_": [2]}
    if t in ("global_max", "global_max2d"):
        return "GlobalMax", {}
    if t == "adaptive_pool":
        cls = {("avg", 2): "AdaptiveAvgPool1d", ("avg", 3): "AdaptiveAvgPool2d",
               ("max", 2): "AdaptiveMaxPool1d", ("max", 3): "AdaptiveMaxPool2d"}[(o("mode"), rank)]
        return "nn." + cls, {"_": [o("size")]}
    if t == "first_step":
        return "FirstStep", {}
    if t == "attn_pool":
        return "AttentionPool", {"c": C}
    if t == "upsample":
        mode = "nearest" if o("mode") == "nearest" else ("linear" if rank == 2 else "bilinear")
        kw = {"scale_factor": 2, "mode": mode}
        if mode != "nearest":
            kw["align_corners"] = False
        return "nn.Upsample", kw
    if t == "pixel_shuffle":
        return "PixelShuffle", {}
    if t in ("conv_t", "conv_t2d"):
        k, st = o("kernel"), o("stride")
        p, op = T_PAD[k] if st == 2 else ((k - 1) // 2, 0)
        cls = "nn.ConvTranspose1d" if t == "conv_t" else "nn.ConvTranspose2d"
        kw = {"_": [C, o("channels"), k], "stride": st, "padding": p}
        if op:
            kw["output_padding"] = op
        return cls, kw
    if t == "eca":
        return "ECA", {"k": o("kernel")}
    if t == "cbam":
        return "CBAM", {"c": C, "ratio": o("ratio"), "k": o("kernel"), "dims": rank - 1}
    if t == "spatial_attn":
        return "SpatialAttention", {"k": o("kernel"), "dims": rank - 1}
    if t == "mha":
        return "MultiHeadAttention", {"c": C, "heads": o("heads"), "causal": bool(o("causal")), "rope": o("pos") == "rope"}
    if t == "linear_attn":
        return "LinearAttention", {"c": C, "heads": o("heads")}
    if t == "nonlocal":
        return "NonLocal", {"c": C}
    if t == "cross_attn":
        return "CrossAttention", {"c": C, "c2": ins2[0], "heads": o("heads")}
    if t == "pos_enc":
        return "PositionalEncoding", {"c": C, "n": _prod(ins[1:]), "learned": o("mode") == "learned"}
    if t == "patch_embed":
        return "PatchEmbedding", {"c": C, "dim": o("dim"), "patch": o("patch")}
    if t == "cls_token":
        return "ClsToken", {"c": C}
    if t == "layernorm":
        return "LayerNorm", {"c": C}
    if t == "rmsnorm":
        return "RMSNorm", {"c": C}
    if t == "groupnorm":
        return "nn.GroupNorm", {"_": [o("groups"), C]}
    if t == "instancenorm":
        return "nn.GroupNorm", {"_": [C, C]}
    if t == "prelu":
        return "nn.PReLU", {}
    if t == "spatial_dropout":
        return ("nn.Dropout1d" if rank == 2 else "nn.Dropout2d"), {"_": [float(o("rate"))]}
    if t == "drop_path":
        return "DropPath", {"rate": float(o("rate"))}
    if t == "gaussian_noise":
        return "GaussianNoise", {"std": float(o("std"))}
    if t == "slice":
        return "Split", {"start": o("start"), "count": o("count")}
    if t == "reshape":
        return "Reshape", {"c": o("channels")}
    if t == "permute":
        return "Transpose", {}
    if t == "to_seq":
        return "ToSequence", {}
    if t == "crop":
        return "Crop", {"a": o("amount")}
    if t == "pad":
        return "Pad", {"a": o("amount")}
    if t == "embedding":
        return "Embedding", {"bins": o("bins"), "dim": o("dim")}
    if t == "spectrogram":
        return "Spectrogram", {"n_fft": o("n_fft"), "hop": o("hop"), "mode": o("mode"), "n_mels": o("n_mels"), "n_mfcc": o("n_mfcc")}
    if t == "rand_flip":
        return "RandomFlip", {"dir": o("dir")}
    if t == "rand_rotate":
        return "RandomRotation", {"degrees": o("degrees")}
    if t == "rand_shift":
        return "RandomShift", {"frac": float(o("frac"))}
    if t == "rand_zoom":
        return "RandomZoom", {"frac": float(o("frac"))}
    if t == "rand_scale":
        return "RandomScale", {"frac": float(o("frac"))}
    if t == "rand_stretch":
        return "RandomStretch", {"frac": float(o("frac"))}
    if t == "mnv3s":
        return "MobileNetV3Small", {"size": o("size")}
    raise ValueError(f"알 수 없는 층 {t}")


def args_text(kw: dict) -> str:
    """인자 dict → 코드 글자 ("_"는 자리 인자)."""
    pos = [repr(v) for v in kw.get("_", [])]
    named = [f"{k}={v!r}" for k, v in kw.items() if k != "_"]
    return ", ".join(pos + named)


# ---------------------------------------------------------------- 고정 표 (Spectrogram·위치): 웹 엔진(engine.js)도 같은 식

SR = 16000          # mel 눈금에 쓰는 가정 표본 주파수 (신호의 실제 주파수와는 상관없이 낮은 칸을 촘촘히)


def mel_matrix(n_fft: int, n_mels: int) -> list:
    """(n_mels, n_fft//2+1) 삼각 mel 필터 (HTK 식)."""
    F = n_fft // 2 + 1
    mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    top = mel(SR / 2)
    pts = [hz(top * i / (n_mels + 1)) for i in range(n_mels + 2)]
    freqs = [SR / 2 * j / (F - 1) for j in range(F)]
    out = []
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        row = []
        for f in freqs:
            up = (f - lo) / (c - lo) if c > lo else 0.0
            down = (hi - f) / (hi - c) if hi > c else 0.0
            row.append(max(0.0, min(up, down)))
        out.append(row)
    return out


def dct_matrix(n_mfcc: int, n_mels: int) -> list:
    """(n_mfcc, n_mels) DCT-II (정규 직교)."""
    return [[math.sqrt((1.0 if k == 0 else 2.0) / n_mels) * math.cos(math.pi / n_mels * (m + 0.5) * k)
             for m in range(n_mels)] for k in range(n_mfcc)]


def sin_table(c: int, n: int) -> list:
    """(c, n) 위치 표: 채널 2i는 sin, 2i+1은 cos, 파장은 10000^(2i/c)."""
    return [[(math.sin if ch % 2 == 0 else math.cos)(p / 10000 ** ((ch - ch % 2) / c)) for p in range(n)] for ch in range(c)]

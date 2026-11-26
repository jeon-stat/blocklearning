// DL studio 모델 편집기: 층(상자)을 선으로 이은 그래프. 자리는 자동으로 위→아래 (갈래는 옆으로, 건너뛰는 선은 옆으로 돌아감).
// 그래프 JSON(v2): {version: 2, nodes: [{id, type, ...설정, inputs: [id...]}]}. 입력 노드 id "input", Repeat 묶음 안의 입력 "in".
// 앞부분은 순수 함수(자동 배치·편집 동작·글자)라 node로 시험하고, 뒷부분(mount)이 화면을 그린다.
(function () {          // 감싸서 이름이 전역으로 새지 않게 (app.js와 같은 전역 공간을 씀)
"use strict";

const BOX_W = 236, BOX_H = 50, MERGE_W = 104, MERGE_H = 30, GAP_Y = 34, GAP_X = 30, DUMMY_W = 14, HEAD = 30, PAD = 16,
  MARGIN = 24, IN_W = 14, IN_H = 10, ACT_D = 64, SNAP = 48, GAP_H = 60;

// ---- 블록 목록 (왼쪽): 범주 → 블록. 같은 종류에 설정만 다른 것(예: Causal Attention)도 따로 둠
const CATS = [["dense", "완전연결 (Dense)"], ["conv", "합성곱 (Conv)"], ["pool", "줄이기·모으기 (Pooling)"], ["up", "키우기 (Upsampling)"],
  ["rnn", "순환 (RNN)"], ["attn", "어텐션 (Attention)"], ["tf", "Transformer 부품"], ["norm", "정규화 (Normalization)"],
  ["act", "활성화 (Activation)"], ["reg", "과적합 막기 (Regularization)"], ["merge", "합치기·나누기"], ["shape", "모양 바꾸기"],
  ["prep", "입력 변환"], ["aug", "증강 (학습 때만)"], ["pre", "사전학습 (고정)"], ["pack", "묶음 (자주 쓰는 조합)"]];
const COLORS = { input: "gray", dense: "gray", conv: "teal", pool: "teal", up: "teal", rnn: "coral", attn: "blue", tf: "blue", norm: "amber",
  act: "purple", reg: "gray", merge: "add", shape: "gray", prep: "pink", aug: "pink", pre: "blue", out: "coral", pack: "gray" };
// 블록이 받는 모양: seq (채널, 길이) · img (채널, 높이, 너비) · sp (둘 다) · any
const RANK = { conv: "seq", pool: "seq", global_avg: "seq", last_step: "seq", conv2d: "img", pool2d: "img", global_avg2d: "img",
  lstm: "seq", gru: "seq", simple_rnn: "seq", se: "sp", dw_conv: "seq", sep_conv: "seq", max_pool: "seq", global_max: "seq", first_step: "seq",
  conv_t: "seq", cls_token: "seq", dw_conv2d: "img", sep_conv2d: "img", max_pool2d: "img", global_max2d: "img", conv_t2d: "img",
  patch_embed: "img", to_seq: "img", rand_rotate: "img", rand_zoom: "img", rand_stretch: "seq", spectrogram: "seq",
  adaptive_pool: "sp", attn_pool: "sp", upsample: "sp", pixel_shuffle: "sp", eca: "sp", cbam: "sp", spatial_attn: "sp", mha: "sp",
  linear_attn: "sp", nonlocal: "sp", cross_attn: "sp", pos_enc: "sp", groupnorm: "sp", instancenorm: "sp", spatial_dropout: "sp",
  permute: "sp", crop: "sp", pad: "sp", rand_flip: "sp", rand_shift: "sp", embedding: "sp", mnv3s: "img" };
const TWO_D = { conv: "conv2d", pool: "pool2d", global_avg: "global_avg2d", dw_conv: "dw_conv2d", sep_conv: "sep_conv2d", max_pool: "max_pool2d",
  global_max: "global_max2d", conv_t: "conv_t2d" };
const ONE_D = Object.fromEntries(Object.entries(TWO_D).map(([a, b]) => [b, a]));
const P = (cat, name, desc, make) => ({ cat, name, desc, make });
const PALETTE = {
  dense: P("dense", "Dense", "모든 값을 섞어 정한 개수(units)로 (Linear)", () => ({ type: "dense", units: 16 })),
  glu: P("dense", "GLU", "반은 값, 반은 문(σ): 값 × σ(문). 위치마다 채널에", () => ({ type: "glu", units: 16, gate: "sigmoid" })),
  "glu:swiglu": P("dense", "SwiGLU", "GLU의 문을 SiLU로 (LLaMA 등 최신 LLM의 FFN)", () => ({ type: "glu", units: 16, gate: "silu" })),
  conv: P("conv", "Conv1D", "짧은 구간의 모양(패턴)을 찾아 채널(filter)로 나눔", () => ({ type: "conv", channels: 16, kernel: 3, stride: 1 })),
  "conv:causal": P("conv", "Causal Conv1D", "과거만 보는 Conv1D (왼쪽만 채움). WaveNet·TCN", () => ({ type: "conv", channels: 16, kernel: 3, stride: 1, padding: "causal" })),
  conv2d: P("conv", "Conv2D", "작은 네모(3×3 등) 필터로 그림의 획·모서리 같은 모양을 찾음", () => ({ type: "conv2d", channels: 8, kernel: 3, stride: 1 })),
  dw_conv: P("conv", "DepthwiseConv1D", "채널마다 따로 필터 하나 (채널은 안 섞음, 가벼움)", () => ({ type: "dw_conv", kernel: 3, multiplier: 1 })),
  dw_conv2d: P("conv", "DepthwiseConv2D", "채널마다 따로 3×3 필터 (MobileNet)", () => ({ type: "dw_conv2d", kernel: 3, multiplier: 1 })),
  sep_conv: P("conv", "SeparableConv1D", "Depthwise + 1칸 Conv: 같은 일을 훨씬 적은 가중치로", () => ({ type: "sep_conv", channels: 16, kernel: 3 })),
  sep_conv2d: P("conv", "SeparableConv2D", "Depthwise 3×3 + 1×1 Conv (Xception·MobileNet)", () => ({ type: "sep_conv2d", channels: 16, kernel: 3 })),
  pool: P("pool", "AvgPool1D", "이웃 값을 평균 내어 길이를 줄임", () => ({ type: "pool", size: 2 })),
  max_pool: P("pool", "MaxPool1D", "이웃 중 가장 큰 값만 남겨 길이를 줄임 (가장 흔한 풀링)", () => ({ type: "max_pool", size: 2 })),
  pool2d: P("pool", "AvgPool2D", "2×2 칸을 평균 내어 그림 크기를 절반으로", () => ({ type: "pool2d", size: 2 })),
  max_pool2d: P("pool", "MaxPool2D", "2×2 칸 중 가장 큰 값만 남겨 그림을 절반으로", () => ({ type: "max_pool2d", size: 2 })),
  global_avg: P("pool", "GlobalAvgPool1D", "채널마다 모든 시점을 평균 내어 값 하나로", () => ({ type: "global_avg" })),
  global_max: P("pool", "GlobalMaxPool1D", "채널마다 모든 시점 중 가장 큰 값 (어딘가에 한 번이라도 나타났나)", () => ({ type: "global_max" })),
  global_avg2d: P("pool", "GlobalAvgPool2D", "채널마다 그림 전체를 평균 내어 값 하나로", () => ({ type: "global_avg2d" })),
  global_max2d: P("pool", "GlobalMaxPool2D", "채널마다 그림 전체에서 가장 큰 값", () => ({ type: "global_max2d" })),
  adaptive_pool: P("pool", "AdaptivePool", "입력 크기와 상관없이 정한 크기(예: 4)로 줄임", () => ({ type: "adaptive_pool", size: 4, mode: "avg" })),
  attn_pool: P("pool", "AttentionPool", "위치마다 점수를 배워 중요한 곳을 더 반영해 모음", () => ({ type: "attn_pool" })),
  last_step: P("pool", "LastTimestep", "채널마다 마지막 시점 값만 (TCN식 분류)", () => ({ type: "last_step" })),
  first_step: P("pool", "FirstTimestep", "채널마다 첫 시점 값만 (CLS token 꺼내기)", () => ({ type: "first_step" })),
  upsample: P("up", "Upsample", "크기를 2배로 (가까운 값 복사 또는 선형 보간)", () => ({ type: "upsample", mode: "nearest" })),
  pixel_shuffle: P("up", "PixelShuffle", "채널을 칸으로 옮겨 2배로 (채널은 2·4로 나뉨)", () => ({ type: "pixel_shuffle" })),
  conv_t: P("up", "ConvTranspose1D", "배우면서 길이를 2배로 키우는 Conv", () => ({ type: "conv_t", channels: 16, kernel: 2, stride: 2 })),
  conv_t2d: P("up", "ConvTranspose2D", "배우면서 그림을 2배로 키우는 Conv (U-Net·생성)", () => ({ type: "conv_t2d", channels: 8, kernel: 2, stride: 2 })),
  simple_rnn: P("rnn", "SimpleRNN", "가장 단순한 순환 층", () => ({ type: "simple_rnn", hidden: 16, layers: 1 })),
  lstm: P("rnn", "LSTM", "시점을 차례로 읽으며 기억", () => ({ type: "lstm", hidden: 16, layers: 1 })),
  gru: P("rnn", "GRU", "LSTM보다 가벼운 순환 층", () => ({ type: "gru", hidden: 16, layers: 1 })),
  "lstm:bi": P("rnn", "Bidirectional LSTM", "앞→뒤, 뒤→앞으로 둘 다 읽어 이어 붙임", () => ({ type: "lstm", hidden: 16, layers: 1, bidirectional: true })),
  "gru:bi": P("rnn", "Bidirectional GRU", "앞→뒤, 뒤→앞으로 둘 다 읽는 GRU", () => ({ type: "gru", hidden: 16, layers: 1, bidirectional: true })),
  se: P("attn", "SE block", "채널마다 중요도를 배워 곱함 (Squeeze-and-Excitation)", () => ({ type: "se", ratio: 4, act: "relu" })),
  eca: P("attn", "ECA", "SE의 가벼운 판: 이웃 채널끼리만 봄", () => ({ type: "eca", kernel: 3 })),
  cbam: P("attn", "CBAM", "무엇을(채널) + 어디를(공간) 볼지 차례로", () => ({ type: "cbam", ratio: 4, kernel: 7 })),
  spatial_attn: P("attn", "SpatialAttention", "어디를 볼지 지도를 만들어 곱함", () => ({ type: "spatial_attn", kernel: 7 })),
  mha: P("attn", "MultiHeadAttention", "위치끼리 서로 보고 섞음 (Self-attention, Transformer의 핵심)", () => ({ type: "mha", heads: 2 })),
  "mha:causal": P("attn", "Causal Attention", "앞 위치만 봄 (GPT식, 다음 값 맞히기)", () => ({ type: "mha", heads: 2, causal: true })),
  cross_attn: P("attn", "CrossAttention", "입력 둘: 첫째(질문)가 둘째(참고)의 어디를 볼지", () => ({ type: "cross_attn", heads: 2 })),
  linear_attn: P("attn", "LinearAttention", "길이에 비례하는 가벼운 어텐션 (긴 입력용)", () => ({ type: "linear_attn", heads: 2 })),
  nonlocal: P("attn", "NonLocal", "먼 칸끼리의 관계를 모아 더함 (이미지 어텐션)", () => ({ type: "nonlocal" })),
  pos_enc: P("tf", "PositionalEncoding", "위치마다 sin·cos 값을 더해 순서를 알려 줌", () => ({ type: "pos_enc", mode: "sin" })),
  "pos_enc:learned": P("tf", "PositionalEncoding (학습형)", "위치마다 더할 값을 배움 (BERT·ViT)", () => ({ type: "pos_enc", mode: "learned" })),
  "mha:rope": P("tf", "Attention + RoPE", "Q·K를 위치만큼 돌려 순서를 앎 (최신 LLM)", () => ({ type: "mha", heads: 2, pos: "rope" })),
  patch_embed: P("tf", "PatchEmbedding", "그림을 조각으로 잘라 조각마다 벡터로 (ViT 입구)", () => ({ type: "patch_embed", patch: 4, dim: 32 })),
  cls_token: P("tf", "CLS token", "줄 맨 앞에 전체를 대표할 자리 하나를 붙임", () => ({ type: "cls_token" })),
  bn: P("norm", "BatchNorm", "묶음(배치) 기준으로 값 크기를 고르게", () => ({ type: "bn" })),
  layernorm: P("norm", "LayerNorm", "위치마다 채널 값을 고르게 (Transformer)", () => ({ type: "layernorm" })),
  rmsnorm: P("norm", "RMSNorm", "LayerNorm의 가벼운 판 (최신 LLM)", () => ({ type: "rmsnorm" })),
  groupnorm: P("norm", "GroupNorm", "채널을 묶어서 (배치가 작을 때)", () => ({ type: "groupnorm", groups: 4 })),
  instancenorm: P("norm", "InstanceNorm", "샘플·채널마다 따로 (스타일 변환)", () => ({ type: "instancenorm" })),
  dropout: P("reg", "Dropout", "학습 때 일부 값을 무작위로 꺼서 과적합을 막음", () => ({ type: "dropout", rate: 0.2 })),
  spatial_dropout: P("reg", "SpatialDropout", "채널을 통째로 끔 (Conv 뒤에)", () => ({ type: "spatial_dropout", rate: 0.1 })),
  drop_path: P("reg", "DropPath", "Residual 길을 통째로 건너뜀 (Residual 안 마지막에)", () => ({ type: "drop_path", rate: 0.1 })),
  gaussian_noise: P("reg", "GaussianNoise", "학습 때 작은 잡음을 더함", () => ({ type: "gaussian_noise", std: 0.1 })),
  add: P("merge", "Add (+)", "들어온 값들을 더함 (모양이 같아야 함). 잔차", () => ({ type: "add" })),
  concat: P("merge", "Concat", "채널 방향으로 이어 붙임 (길이가 같아야 함)", () => ({ type: "concat" })),
  multiply: P("merge", "Multiply (×)", "원소마다 곱함 (문·가중치 적용)", () => ({ type: "multiply" })),
  average: P("merge", "Average", "원소마다 평균", () => ({ type: "average" })),
  maximum: P("merge", "Maximum", "원소마다 큰 값", () => ({ type: "maximum" })),
  slice: P("merge", "Split (채널 나누기)", "채널 일부만 꺼냄 (둘 이상 꺼내면 나누기)", () => ({ type: "slice", start: 0, count: 4 })),
  flatten: P("shape", "Flatten", "한 줄로 펼침 (Dense 앞)", () => ({ type: "flatten" })),
  reshape: P("shape", "Reshape", "값 전체를 (채널, 나머지) 줄로 다시 묶음", () => ({ type: "reshape", channels: 4 })),
  permute: P("shape", "Transpose", "줄은 채널↔길이, 그림은 높이↔너비", () => ({ type: "permute" })),
  to_seq: P("shape", "ToSequence", "그림 (C, H, W) → 칸들의 줄 (C, H·W). 어텐션 앞", () => ({ type: "to_seq" })),
  crop: P("shape", "Crop", "가장자리를 잘라냄", () => ({ type: "crop", amount: 1 })),
  pad: P("shape", "Pad", "가장자리에 0을 채움", () => ({ type: "pad", amount: 1 })),
  embedding: P("prep", "Embedding", "값을 구간 번호로 바꿔 번호마다 벡터로 (시계열 토큰)", () => ({ type: "embedding", bins: 32, dim: 16 })),
  spectrogram: P("prep", "Spectrogram", "신호 → 시간·주파수 그림 (그 뒤엔 Conv2D)", () => ({ type: "spectrogram", n_fft: 32, hop: 8, mode: "log" })),
  "spectrogram:mel": P("prep", "Mel spectrogram", "사람 귀처럼 낮은 소리를 촘촘히 (음성)", () => ({ type: "spectrogram", n_fft: 64, hop: 16, mode: "mel", n_mels: 16 })),
  "spectrogram:mfcc": P("prep", "MFCC", "mel을 DCT로 줄인 음성 특징", () => ({ type: "spectrogram", n_fft: 64, hop: 16, mode: "mfcc", n_mels: 32, n_mfcc: 13 })),
  rand_flip: P("aug", "RandomFlip", "학습 때 반씩 뒤집음 (그림 좌우·신호 시간)", () => ({ type: "rand_flip", dir: "h" })),
  rand_rotate: P("aug", "RandomRotation", "학습 때 그림을 조금씩 돌림", () => ({ type: "rand_rotate", degrees: 15 })),
  rand_shift: P("aug", "RandomShift", "학습 때 조금씩 옮김", () => ({ type: "rand_shift", frac: 0.1 })),
  rand_zoom: P("aug", "RandomZoom", "학습 때 그림을 조금 확대·축소", () => ({ type: "rand_zoom", frac: 0.1 })),
  rand_scale: P("aug", "RandomScale", "학습 때 값 전체 세기를 조금 바꿈", () => ({ type: "rand_scale", frac: 0.2 })),
  rand_stretch: P("aug", "RandomStretch", "학습 때 신호를 시간 방향으로 늘이고 줄임", () => ({ type: "rand_stretch", frac: 0.1 })),
  mnv3s: P("pre", "MobileNetV3-Small", "ImageNet으로 미리 학습한 특징 1024개 (가중치 고정)", () => ({ type: "mnv3s", size: 128 })),
};
const ACT_LIST = [["relu", "ReLU", "음수는 0 (가장 많이 씀)"], ["leaky_relu", "LeakyReLU", "음수도 조금 흘림"], ["gelu", "GELU", "부드러운 ReLU (Transformer)"],
  ["silu", "SiLU / Swish", "x·σ(x) (EfficientNet·LLM)"], ["elu", "ELU", "음수 쪽이 부드럽게 −1로"], ["mish", "Mish", "부드럽고 매끈한 ReLU 계열"],
  ["softplus", "Softplus", "부드러운 ReLU"], ["sigmoid", "Sigmoid", "0~1로"], ["tanh", "Tanh", "−1~1로"], ["selu", "SELU", "스스로 정규화되는 활성화"],
  ["relu6", "ReLU6", "0~6으로 자름 (모바일)"], ["hardswish", "HardSwish", "가벼운 Swish (MobileNetV3)"], ["hardsigmoid", "HardSigmoid", "가벼운 Sigmoid"],
  ["softmax", "Softmax", "채널 합이 1이 되게"]];
for (const [k, name, desc] of ACT_LIST) PALETTE[k === "relu" ? "act" : `act:${k}`] = P("act", name, desc, () => ({ type: "act", kind: k }));
PALETTE.prelu = P("act", "PReLU", "LeakyReLU의 기울기를 배움", () => ({ type: "prelu" }));
// 종류마다 이름·설명·색 (palette 대표 항목에서)
const TYPES = {};
for (const [id, p] of Object.entries(PALETTE)) {
  const t = p.make().type;
  if (!TYPES[t] || !id.includes(":")) TYPES[t] = { cat: p.cat, name: t === "act" ? "Activation" : p.name, desc: p.desc, make: p.make };
}
TYPES.act.desc = "꺾임(비선형)을 넣음. 잘 모르겠으면 ReLU";
TYPES.output = { cat: "out", name: "Output", desc: "classification이면 클래스 수, regression이면 값 하나", make: () => ({ type: "output" }) };
const ACT_NAMES = Object.fromEntries(ACT_LIST.map(([k, n]) => [k, n]));
const RNN_NAMES = { lstm: "LSTM", gru: "GRU", simple_rnn: "SimpleRNN" };
const ACTS = ACT_LIST.map(([k, n]) => [k, k === "relu" ? "ReLU (추천)" : n]);
const MERGES = ["add", "concat", "multiply", "average", "maximum"];
const TWO_INPUT = ["cross_attn"];
// 왼쪽 목록은 데이터 종류와 상관없이 모든 층 (직접 꾸밀 수 있게). 잘 안 맞는 층은 흐리게 한마디와 함께
const TABULAR_OK = ["dense", "glu", "act", "prelu", "bn", "layernorm", "rmsnorm", "dropout", "gaussian_noise", "drop_path", "add", "concat",
  "multiply", "average", "maximum", "slice", "flatten", "output"];
function fitNote(t, input) {
  const r = RANK[t];
  if (input === "image") return r === "seq" && !["conv", "pool", "global_avg"].includes(t) ? "줄(시계열) 모양에 씀 — 그림은 ToSequence 뒤에" : "";
  if (r === "img") return t === "conv2d" || t === "pool2d" || t === "global_avg2d" ? "이미지 데이터용 (Spectrogram 뒤에는 됨)" : "그림 모양에 씀 (Spectrogram 뒤에는 됨)";
  if (input === "tabular" && !TABULAR_OK.includes(t)) return "표 데이터엔 보통 안 씀 (열 순서를 시간처럼 다룸)";
  return "";
}
function paletteTypes(input) {
  const all = Object.keys(PALETTE);
  const typeOf = (id) => PALETTE[id].make().type;
  return all.filter((t) => !fitNote(typeOf(t), input)).concat(all.filter((t) => fitNote(typeOf(t), input)));
}
function chooseHeads(C) { return [4, 2, 1].find((h) => C % h === 0 && C / h >= 2) || 1; }

// ---- 묶음: 자주 쓰는 조합. make(C, rank, L) → {nodes (입력은 "SRC"), exit}. C: 넣는 자리로 들어오는 채널 수
const PACKS = [
  { id: "conv_bn_relu", name: "Conv–BN–ReLU", desc: "가장 흔한 합성곱 단위", ranks: [2, 3] },
  { id: "resnet", name: "ResNet 기본 블록", desc: "Conv–BN–ReLU–Conv–BN + 지름길 더하기 → ReLU", ranks: [2, 3] },
  { id: "bottleneck", name: "ResNet Bottleneck", desc: "1×1로 줄였다가 3×3 → 1×1로 되돌리고 지름길 (깊은 ResNet)", ranks: [2, 3] },
  { id: "se_resnet", name: "SE-ResNet 블록", desc: "ResNet 블록 + SE 채널 주의", ranks: [2, 3] },
  { id: "inception", name: "Inception 모듈", desc: "1·3·5칸 필터를 나란히 써서 이어 붙임", ranks: [2, 3] },
  { id: "mobilenet", name: "MobileNet Inverted Residual", desc: "1×1로 4배 넓힘 → Depthwise → 1×1로 좁힘 + 지름길", ranks: [2, 3] },
  { id: "dense", name: "DenseNet Dense 블록", desc: "층마다 앞의 모든 값을 이어 붙여 다음으로", ranks: [2, 3] },
  { id: "convnext", name: "ConvNeXt 블록", desc: "Depthwise 7칸 → LayerNorm → 4배 넓힘·GELU → 되돌림 + DropPath 지름길", ranks: [2, 3] },
  { id: "unet", name: "U-Net 한 단계", desc: "줄였다가(MaxPool) 키우고(Upsample) 같은 크기의 앞 값과 이어 붙임", ranks: [2, 3] },
  { id: "transformer", name: "Transformer Encoder 블록", desc: "LayerNorm → Self-attention + 지름길 → LayerNorm → FFN(4배·GELU) + 지름길", ranks: [2, 3] },
  { id: "decoder", name: "Transformer Decoder 블록 (GPT식)", desc: "Encoder 블록과 같고 어텐션이 앞만 봄 (Causal)", ranks: [2] },
  { id: "mixer", name: "MLP-Mixer 블록", desc: "위치끼리 섞는 MLP(Transpose) + 채널끼리 섞는 MLP, 각각 지름길", ranks: [2] },
  { id: "vit", name: "ViT 입구", desc: "PatchEmbedding → CLS token → PositionalEncoding(학습형)", ranks: [3] },
  { id: "mlp", name: "MLP 블록 (Residual)", desc: "LayerNorm → 넓힘·GELU → 되돌림 + 지름길", ranks: [1, 2, 3] },
];
function makePack(id, C, rank, L) {
  const nodes = [];
  let n = 0;
  const two = rank === 3;
  const add = (type, ins, kw = {}) => {
    const t = two && TWO_D[type] ? TWO_D[type] : type, node = Object.assign({ id: `p${++n}`, type: t, inputs: ins }, kw);
    if (t === "conv2d" && node.kernel > 7) node.kernel = 7;
    if (rank === 1 && (type === "conv")) { node.type = "dense"; node.units = node.channels; delete node.channels; delete node.kernel; delete node.stride; }
    nodes.push(node);
    return node.id;
  };
  const conv = (ins, ch, k = 3) => add("conv", [ins], { channels: Math.max(1, ch), kernel: k, stride: 1 });
  const c = Math.max(4, C || 16), q = Math.max(1, Math.round(c / 4));
  let x = "SRC", s;
  switch (id) {
    case "conv_bn_relu": x = add("act", [add("bn", [conv("SRC", c)])], { kind: "relu" }); break;
    case "resnet": case "se_resnet": {
      s = add("bn", [conv(add("act", [add("bn", [conv("SRC", c)])], { kind: "relu" }), c)]);
      if (id === "se_resnet") s = add("se", [s], { ratio: Math.min(4, c), act: "relu" });
      x = add("act", [add("add", ["SRC", s])], { kind: "relu" });
      break;
    }
    case "bottleneck": {
      s = add("act", [add("bn", [conv("SRC", q, 1)])], { kind: "relu" });
      s = add("act", [add("bn", [conv(s, q, 3)])], { kind: "relu" });
      s = add("bn", [conv(s, c, 1)]);
      x = add("act", [add("add", ["SRC", s])], { kind: "relu" });
      break;
    }
    case "inception": {
      const b = Math.max(4, Math.round(c / 2));
      const b1 = conv("SRC", b, 1), b2 = conv(conv("SRC", b, 1), b, 3), b3 = conv(conv("SRC", b, 1), b, 5);
      x = add("act", [add("concat", [b1, b2, b3])], { kind: "relu" });
      break;
    }
    case "mobilenet": {
      s = add("act", [add("bn", [conv("SRC", c * 4, 1)])], { kind: "relu6" });
      s = add("act", [add("bn", [add("dw_conv", [s], { kernel: 3, multiplier: 1 })])], { kind: "relu6" });
      s = add("bn", [conv(s, c, 1)]);
      x = add("add", ["SRC", s]);
      break;
    }
    case "dense": {
      const g = 8;
      const l1 = conv(add("act", [add("bn", ["SRC"])], { kind: "relu" }), g, 3), c1 = add("concat", ["SRC", l1]);
      const l2 = conv(add("act", [add("bn", [c1])], { kind: "relu" }), g, 3);
      x = add("concat", [c1, l2]);
      break;
    }
    case "convnext": {
      s = add("layernorm", [add("dw_conv", ["SRC"], { kernel: 7, multiplier: 1 })]);
      s = add("drop_path", [conv(add("act", [conv(s, c * 4, 1)], { kind: "gelu" }), c, 1)], { rate: 0.1 });
      x = add("add", ["SRC", s]);
      break;
    }
    case "unet": {
      const skip = add("act", [conv("SRC", c)], { kind: "relu" });
      const mid = add("act", [conv(add("max_pool", [skip], { size: 2 }), c * 2)], { kind: "relu" });
      x = add("act", [conv(add("concat", [add("upsample", [mid], { mode: "nearest" }), skip]), c)], { kind: "relu" });
      break;
    }
    case "transformer": case "decoder": {
      const kw = { heads: chooseHeads(c) };
      if (id === "decoder") kw.causal = true;
      const a = add("add", ["SRC", add("mha", [add("layernorm", ["SRC"])], kw)]);
      const f = conv(add("act", [conv(add("layernorm", [a]), c * 4, 1)], { kind: "gelu" }), c, 1);
      x = add("add", [a, f]);
      break;
    }
    case "mixer": {
      const tok = Math.max(2, L || 16);
      const t1 = add("permute", [add("layernorm", ["SRC"])]);
      const t2 = add("permute", [conv(add("act", [conv(t1, tok * 2, 1)], { kind: "gelu" }), tok, 1)]);
      const a = add("add", ["SRC", t2]);
      const f = conv(add("act", [conv(add("layernorm", [a]), c * 4, 1)], { kind: "gelu" }), c, 1);
      x = add("add", [a, f]);
      break;
    }
    case "vit": x = add("pos_enc", [add("cls_token", [add("patch_embed", ["SRC"], { patch: 4, dim: 32 })])], { mode: "learned" }); break;
    case "mlp": {
      const f = conv(add("act", [conv(add("layernorm", ["SRC"]), c * 2, 1)], { kind: "gelu" }), c, 1);
      x = add("add", ["SRC", f]);
      break;
    }
    default: throw new Error("모르는 묶음: " + id);
  }
  return { nodes, exit: x };
}
// 묶음을 선 from → to 사이에 넣음 → {graph, ids}
function insertPack(g0, from, to, pack) {
  const g = clone(g0), f = findNode(g, to);
  if (!f || !insOf(f.node).includes(from)) return { graph: g0, ids: [] };
  const mk = idMaker(g), map = {};
  for (const n of pack.nodes) map[n.id] = mk();
  const added = pack.nodes.map((n) => Object.assign({}, n, { id: map[n.id], inputs: n.inputs.map((x) => (x === "SRC" ? from : map[x])) }));
  replaceInput(f.node, from, map[pack.exit]);
  f.list.splice(f.list.indexOf(f.node), 0, ...added);
  return { graph: g, ids: added.map((n) => n.id) };
}

// ------------------------------------------------------------ 순수 함수: 그래프 다루기
const clone = (x) => JSON.parse(JSON.stringify(x));
const isMerge = (n) => n && MERGES.includes(n.type);
const maxIns = (n) => (isMerge(n) ? Infinity : n && TWO_INPUT.includes(n.type) ? 2 : 1);   // 받을 수 있는 입력 수
const insOf = (n) => (n && Array.isArray(n.inputs) ? n.inputs : []);
const nodesOf = (g) => (g && Array.isArray(g.nodes) ? g.nodes : []);

function eachNode(nodes, fn, parent = null) {             // 모든 노드 (묶음 안 포함): fn(node, list, parent)
  for (const n of nodes || []) {
    if (!n || typeof n !== "object") continue;
    fn(n, nodes, parent);
    if (n.body && Array.isArray(n.body.nodes)) eachNode(n.body.nodes, fn, n);
  }
}
function findNode(g, id) {                                // → {node, list, parent} 또는 null
  let f = null;
  eachNode(nodesOf(g), (n, list, parent) => { if (n.id === id && !f) f = { node: n, list, parent }; });
  return f;
}
const entryOf = (parent) => (parent ? "in" : "input");
function newId(g) {
  let m = 0;
  eachNode(nodesOf(g), (n) => { const r = /^n(\d+)$/.exec(n.id || ""); if (r) m = Math.max(m, Number(r[1])); });
  return "n" + (m + 1);
}
function idMaker(g) {                                     // 같은 편집 안에서 여러 개 만들 때
  let m = Number(newId(g).slice(1));
  return () => "n" + m++;
}
const users = (list, id) => list.filter((n) => insOf(n).includes(id));
function ancestors(list, id) {                            // id에 닿는(위쪽) 모든 노드 id
  const map = new Map(list.map((n) => [n.id, n])), out = new Set(), stack = [id];
  while (stack.length) {
    const k = stack.pop();
    if (out.has(k)) continue;
    out.add(k);
    const n = map.get(k);
    if (n) stack.push(...insOf(n));
  }
  return out;
}
function replaceInput(n, from, to) {                      // n.inputs에서 from을 to로 (첫 번째만)
  const i = insOf(n).indexOf(from);
  if (i >= 0) n.inputs[i] = to;
}

function insertOnEdge(g0, from, to, node) {
  const g = clone(g0), f = findNode(g, to);
  if (!f || !insOf(f.node).includes(from)) return g;
  const n = Object.assign({}, node, { id: node.id || newId(g), inputs: [from] });
  replaceInput(f.node, from, n.id);
  f.list.splice(f.list.indexOf(f.node), 0, n);
  return g;
}
function connect(g0, from, to) {                          // → {graph, error}
  const g = clone(g0), f = findNode(g, to);
  if (!f || f.node.type === "input") return { graph: g0, error: "Input에는 선을 이을 수 없습니다" };
  const entry = entryOf(f.parent);
  if (from !== entry && !f.list.some((n) => n.id === from)) return { graph: g0, error: "같은 묶음 안의 층끼리만 이을 수 있습니다" };
  const src = f.list.find((n) => n.id === from);
  if (src && src.type === "output") return { graph: g0, error: "출력 뒤에는 이을 수 없습니다" };
  if (from === to || ancestors(f.list, from).has(to)) return { graph: g0, error: "아래에서 위로는 이을 수 없습니다 (선이 돌게 됨)" };
  const t = f.node;
  if (insOf(t).includes(from)) return { graph: g0, error: "이미 이어져 있습니다" };
  if (isMerge(t)) t.inputs = insOf(t).concat([from]);        // 합치기 상자: 입력을 더함
  else if (maxIns(t) === 2) t.inputs = insOf(t).length < 2 ? insOf(t).concat([from]) : [insOf(t)[0], from];   // 입력 둘: 둘째를 바꿈
  else t.inputs = [from];                                     // 보통 층: 기존 선을 새 선으로 바꿈
  return { graph: g, error: null };
}
// 출력 동그라미를 빈 곳에 놓고 층을 고름: from 뒤에 이어진 새 층 → {graph, id}
function addConnected(g0, from, node, pos) {
  const g = clone(g0), f = findNode(g, from);
  if (!f) return { graph: g0, id: null };
  const n = Object.assign({}, node, { id: newId(g), inputs: [from] });
  if (validPos(pos)) n.pos = [Math.max(0, Math.round(pos[0])), Math.max(0, Math.round(pos[1]))];
  f.list.splice(f.list.indexOf(f.node) + 1, 0, n);
  return { graph: g, id: n.id };
}
// 입력 동그라미를 빈 곳에 놓고 층을 고름: to 앞을 채우는 새 층 (합치기 상자면 입력을 더함) → {graph, id}
function addFeeding(g0, to, node, pos) {
  const g = clone(g0), f = findNode(g, to);
  if (!f || f.node.type === "input") return { graph: g0, id: null };
  const n = Object.assign({}, node, { id: newId(g), inputs: [] });
  if (validPos(pos)) n.pos = [Math.max(0, Math.round(pos[0])), Math.max(0, Math.round(pos[1]))];
  f.list.splice(f.list.indexOf(f.node), 0, n);
  f.node.inputs = isMerge(f.node) || (maxIns(f.node) === 2 && insOf(f.node).length < 2) ? insOf(f.node).concat([n.id]) : [n.id];
  return { graph: g, id: n.id };
}
function removeNode(g0, id) {
  const f0 = findNode(g0, id);
  if (!f0 || f0.node.type === "input" || f0.node.type === "output") return g0;
  const g = clone(g0), f = findNode(g, id), rep = insOf(f.node)[0];
  for (const u of users(f.list, id)) {
    u.inputs = u.inputs.map((x) => (x === id ? rep : x)).filter((x) => x !== undefined);
    if (isMerge(u)) u.inputs = u.inputs.filter((x, i) => u.inputs.indexOf(x) === i);
  }
  f.list.splice(f.list.indexOf(f.node), 1);
  return g;
}
function removeEdge(g0, from, to) {
  const g = clone(g0), f = findNode(g, to);
  if (!f) return g0;
  const i = insOf(f.node).indexOf(from);
  if (i >= 0) f.node.inputs.splice(i, 1);
  return g;
}
// 고른 노드들이 "입구 하나·출구 하나"인 묶음인지: → {source, exit, list} 또는 null
function singleEntryExit(g, ids) {
  if (!ids || !ids.length) return null;
  const fs = ids.map((id) => findNode(g, id));
  if (fs.some((f) => !f || f.node.type === "input" || f.node.type === "output") || fs.some((f) => f.list !== fs[0].list)) return null;
  const list = fs[0].list, sel = new Set(ids);
  const ext = new Set();
  for (const f of fs) for (const x of insOf(f.node)) if (!sel.has(x)) ext.add(x);
  if (ext.size !== 1) return null;
  const outside = list.filter((n) => !sel.has(n.id));
  const exits = ids.filter((id) => outside.some((n) => insOf(n).includes(id)));
  const sinks = ids.filter((id) => !list.some((n) => sel.has(n.id) && insOf(n).includes(id)));
  if (exits.length > 1 || sinks.length !== 1 || (exits.length === 1 && exits[0] !== sinks[0])) return null;
  return { source: [...ext][0], exit: sinks[0], list, parent: fs[0].parent };
}
function wrapResidual(g0, ids, shapes) {
  const s0 = singleEntryExit(g0, ids);
  if (!s0) return g0;
  const g = clone(g0), s = singleEntryExit(g, ids), mk = idMaker(g);
  const a = (shapes || {})[s.source], b = (shapes || {})[s.exit];
  let sc = s.source;
  const scNode = [];
  if (a && b && JSON.stringify(a) !== JSON.stringify(b)) {
    sc = mk();
    if (b.length === 1) scNode.push({ id: sc, type: "dense", units: b[0], inputs: [s.source] });
    else {
      const ratio = a.length === b.length ? Math.round(a[1] / b[1]) : 1;
      scNode.push({ id: sc, type: b.length === 3 ? "conv2d" : "conv", channels: b[0], kernel: 1, stride: ratio === 2 ? 2 : 1,
        inputs: [s.source] });
    }
  }
  const add = { id: mk(), type: "add", inputs: [s.exit, sc] };
  for (const u of s.list) if (!ids.includes(u.id)) if (u.inputs) u.inputs = u.inputs.map((x) => (x === s.exit ? add.id : x));
  const at = s.list.findIndex((n) => n.id === s.exit) + 1;
  s.list.splice(at, 0, ...scNode, add);
  return g;
}
function wrapRepeat(g0, ids) {
  if (!singleEntryExit(g0, ids)) return g0;
  const g = clone(g0), s = singleEntryExit(g, ids), id = newId(g);
  const body = s.list.filter((n) => ids.includes(n.id)).map((n) => Object.assign({}, n, { inputs: insOf(n).map((x) => (x === s.source ? "in" : x)) }));
  const rep = { id, type: "repeat", times: 2, inputs: [s.source], body: { nodes: body } };
  const at = s.list.findIndex((n) => ids.includes(n.id));
  for (const u of s.list) if (!ids.includes(u.id)) if (u.inputs) u.inputs = u.inputs.map((x) => (x === s.exit ? id : x));
  for (let i = s.list.length - 1; i >= 0; i--) if (ids.includes(s.list[i].id)) s.list.splice(i, 1);
  s.list.splice(at, 0, rep);
  return g;
}
function unwrap(g0, id) {
  const f0 = findNode(g0, id);
  if (!f0 || f0.node.type !== "repeat" || !f0.node.body) return g0;
  const g = clone(g0), f = findNode(g, id), r = f.node, src = insOf(r)[0];
  const body = r.body.nodes || [];
  const sinks = body.filter((n) => !body.some((m) => insOf(m).includes(n.id)));
  const exit = sinks.length ? sinks[sinks.length - 1].id : src;
  const back = body.map((n) => Object.assign({}, n, { inputs: insOf(n).map((x) => (x === "in" ? src : x)) }));
  for (const u of f.list) {
    if (u === r || !u.inputs) continue;
    u.inputs = u.inputs.map((x) => (x === id ? exit : x));
    if (isMerge(u)) u.inputs = u.inputs.filter((x, i) => u.inputs.indexOf(x) === i);   // 몸체가 비었으면 같은 값 둘이 됨
  }
  f.list.splice(f.list.indexOf(r), 1, ...back);
  return g;
}
function splitBranch(g0, ids, newNode) {                  // 같은 입구에서 갈래 하나를 더 만들고 Concat
  if (!singleEntryExit(g0, ids)) return g0;
  const g = clone(g0), s = singleEntryExit(g, ids), mk = idMaker(g);
  const lane = Object.assign({}, newNode, { id: mk(), inputs: [s.source] });
  const cat = { id: mk(), type: "concat", inputs: [s.exit, lane.id] };
  for (const u of s.list) if (!ids.includes(u.id)) if (u.inputs) u.inputs = u.inputs.map((x) => (x === s.exit ? cat.id : x));
  s.list.splice(s.list.findIndex((n) => n.id === s.exit) + 1, 0, lane, cat);
  return g;
}
function copyNodes(g, ids) {
  const s = singleEntryExit(g, ids);
  if (!s) return null;
  return { nodes: clone(s.list.filter((n) => ids.includes(n.id))), source: s.source, exit: s.exit };
}
function pasteAfter(g0, id, clip) {                       // → {graph, ids}
  const f0 = findNode(g0, id);
  if (!clip || !f0 || f0.node.type === "output") return { graph: g0, ids: [] };
  const g = clone(g0), f = findNode(g, id), mk = idMaker(g), map = {};
  eachNode(clip.nodes, (n) => { map[n.id] = mk(); });
  const remap = (nodes, top) => nodes.map((n) => {
    const m = Object.assign({}, n, { id: map[n.id], inputs: insOf(n).map((x) => (top && x === clip.source ? id : map[x] || x)) });
    if (n.body) m.body = { nodes: remap(n.body.nodes || [], false) };
    return m;
  });
  const added = remap(clone(clip.nodes), true);
  const exit = map[clip.exit];
  for (const u of users(f.list, id)) replaceInput(u, id, exit);
  f.list.splice(f.list.indexOf(f.node) + 1, 0, ...added);
  return { graph: g, ids: added.map((n) => n.id) };
}
function setParam(g0, id, key, value) {
  const g = clone(g0), f = findNode(g, id);
  if (f) f.node[key] = value;
  return g;
}
// ---- 화면 위치 (pos: 판 위 왼쪽 위 [x, y], 맨 위 층위 상자만. 모델에는 영향 없음)
const validPos = (p) => Array.isArray(p) && p.length === 2 && p.every((v) => typeof v === "number" && isFinite(v));
function setPos(g0, id, pos) {
  const g = clone(g0), f = findNode(g, id);
  if (f && validPos(pos)) f.node.pos = [Math.round(pos[0]), Math.round(pos[1])];   // 음수도 됨 (희망 위치, 배치가 판 안으로 맞춤)
  return g;
}
function clearPos(g0) {
  const g = clone(g0);
  eachNode(nodesOf(g), (n) => { delete n.pos; delete n.shift; });
  return g;
}
const stripPos = clearPos;
// 줄바꿈한 줄 옮기기 (보기용): 그 줄의 상자 하나에 흐름 방향 거리(px, 0 이상)를 적음. 0이면 줄 첫머리 그대로
function setShift(g0, id, v) {
  const g = clone(g0), f = findNode(g, id);
  if (!f) return g;
  const s = Math.max(0, Math.round(Number(v) || 0));
  if (s > 0) f.node.shift = s; else delete f.node.shift;
  return g;
}
// 상자 크기 (보기용, 모델과 상관없음): 고른 상자들에 한 번에. null이면 기본 크기
const SIZE_MIN = [60, 24], SIZE_MAX = [600, 400];
function setSize(g0, ids, size) {
  const g = clone(g0);
  for (const id of ids) {
    const f = findNode(g, id);
    if (!f || f.node.type === "repeat") continue;
    if (!size) { delete f.node.size; continue; }
    f.node.size = [0, 1].map((k) => Math.max(SIZE_MIN[k], Math.min(SIZE_MAX[k], Math.round(Number(size[k]) || 0))));
  }
  return g;
}
function stripView(g0) {                                  // 모델 비교용: 화면용 값(위치·크기·줄바꿈)을 뺀 그래프
  const g = clone(g0);
  eachNode(nodesOf(g), (n) => { delete n.pos; delete n.size; delete n.brk; delete n.shift; });
  return g;
}
// 줄바꿈 (보기용): 이 상자부터 다음 줄(가로면 아래, 세로면 오른쪽)로
function setBreak(g0, id, on) {
  const g = clone(g0), f = findNode(g, id);
  if (f) { if (on) f.node.brk = true; else delete f.node.brk; }
  return g;
}
function addFree(g0, node, pos) {                         // 선과 먼 곳에 놓음: 아직 안 이어진 상자
  const g = clone(g0), n = Object.assign({}, node, { id: node.id || newId(g), inputs: [] });
  if (validPos(pos)) n.pos = [Math.max(0, Math.round(pos[0])), Math.max(0, Math.round(pos[1]))];
  g.nodes.push(n);
  return g;
}
function moveNode(g0, id, from, to) {                     // 놓인 상자를 다른 선(from → to) 사이로 옮김 → {graph, error}
  const f = findNode(g0, id), t = findNode(g0, to);
  if (!f || f.node.type === "input" || f.node.type === "output") return { graph: g0, error: "Input·Output은 옮길 수 없습니다" };
  if (from === id || to === id) return { graph: g0, error: "제자리입니다" };
  if (!t || !insOf(t.node).includes(from)) return { graph: g0, error: "그 선을 찾을 수 없습니다" };
  if (t.list !== f.list) return { graph: g0, error: "묶음 안팎으로는 옮길 수 없습니다 (같은 묶음 안의 선으로)" };
  const node = clone(f.node);
  delete node.pos;
  return { graph: insertOnEdge(removeNode(g0, id), from, to, node), error: null };
}
const SHORT_ACT = { selu: "SELU", relu6: "ReLU6", hardswish: "HSwish", hardsigmoid: "HSigm", softmax: "Softmax",
  relu: "ReLU", sigmoid: "Sigmoid", tanh: "Tanh", leaky_relu: "LReLU",
  gelu: "GELU", silu: "SiLU", elu: "ELU", mish: "Mish", softplus: "Softplus" };
function shortLabel(b) { return b.type === "act" ? SHORT_ACT[b.kind] || "Act" : label(b); }
// 점 p에서 선분 a-b까지 거리
function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
  const t = L2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2)) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}
// 끌어 놓을 선: SNAP(48px) 안에서 가장 가까운 선. skip(e) → true면 빼기
function nearestEdge(edges, p, maxDist = SNAP, skip = null) {
  let best = null;
  for (const e of edges) {
    if (e.kind === "exit" || (skip && skip(e))) continue;
    for (let i = 0; i + 1 < e.points.length; i++) {
      const d = segDist(p, e.points[i], e.points[i + 1]);
      if (d <= maxDist && (!best || d < best.d)) best = { e, d };
    }
  }
  return best ? best.e : null;
}

// 되돌리기 기록: push(바꾸기 전 상태), undo(지금) → 이전 상태 또는 null, redo(지금) → 다음 상태 또는 null
function history(limit = 100) {
  const back = [], fwd = [];
  return {
    push(state) { back.push(clone(state)); if (back.length > limit) back.shift(); fwd.length = 0; },
    undo(cur) { if (!back.length) return null; fwd.push(clone(cur)); return back.pop(); },
    redo(cur) { if (!fwd.length) return null; back.push(clone(cur)); return fwd.pop(); },
    canUndo: () => back.length > 0, canRedo: () => fwd.length > 0,
  };
}

function label(b, summary) {
  switch (b.type) {
    case "input": return "Input";
    case "conv": return `Conv1D · ${b.channels} filters · k=${b.kernel}` + (b.stride === 2 ? " · stride 2" : "") +
      ((b.dilation || 1) > 1 ? ` · dilation ${b.dilation}` : "") + (b.padding === "valid" ? " · valid" : b.padding === "causal" ? " · causal" : "") +
      ((b.groups || 1) > 1 ? ` · groups ${b.groups}` : "");
    case "last_step": return "LastTimestep";
    case "se": return `SE block · ratio ${b.ratio}`;
    case "dropout": return `Dropout · ${b.rate}`;
    case "flatten": return "Flatten";
    case "pool": return `AvgPool1D · pool ${b.size}`;
    case "conv2d": return `Conv2D · ${b.channels} filters · ${b.kernel}×${b.kernel}` + (b.stride === 2 ? " · stride 2" : "") +
      ((b.dilation || 1) > 1 ? ` · dilation ${b.dilation}` : "") + ((b.groups || 1) > 1 ? ` · groups ${b.groups}` : "") +
      (b.padding === "valid" ? " · valid" : "");
    case "pool2d": return `AvgPool2D · pool ${b.size || 2}`;
    case "global_avg2d": return "GlobalAvgPool2D";
    case "global_avg": return "GlobalAvgPool1D";
    case "bn": return "BatchNorm";
    case "act": return ACT_NAMES[b.kind] || "Activation";
    case "dense": return `Dense · ${b.units} units`;
    case "add": return "Add (+)";
    case "concat": return "Concat";
    case "lstm": case "gru": case "simple_rnn": {
      const n = b.layers || 1;
      return `${b.bidirectional ? "Bi" : ""}${RNN_NAMES[b.type]} · ${b.hidden} units` + (n > 1 ? ` × ${n} layers` : "") +
        (b.return_sequences ? " · 줄 그대로" : "");
    }
    case "repeat": return `Repeat × ${b.times}`;
    case "prelu": return "PReLU";
    case "output":
      if (b.head === "distance") return "Output · 정상과의 거리";
      if (!summary) return "Output";
      if (summary.task === "multi" && summary.targets) return `Output · 정답 ${summary.targets.length}개 (${summary.targets.map((t) => t.name).join(" · ")})`.slice(0, 60);
      if (summary.task !== "classification") return "Output · regression";
      {
        const names = summary.classes.join(" · ");           // 클래스가 많으면 상자를 넘지 않게 개수만
        return `Output · classification (${names.length > 24 ? `${summary.classes.length} classes` : names})`;
      }
    default: return LABELS[b.type] ? LABELS[b.type](b) : TYPES[b.type] ? TYPES[b.type].name : String(b.type);
  }
}
const LABELS = {
  glu: (b) => `${b.gate === "silu" ? "SwiGLU" : "GLU"} · ${b.units} units`,
  dw_conv: (b) => `DepthwiseConv1D · k=${b.kernel || 3}` + ((b.multiplier || 1) > 1 ? ` · ×${b.multiplier}` : ""),
  dw_conv2d: (b) => `DepthwiseConv2D · ${b.kernel || 3}×${b.kernel || 3}` + ((b.multiplier || 1) > 1 ? ` · ×${b.multiplier}` : ""),
  sep_conv: (b) => `SeparableConv1D · ${b.channels} filters · k=${b.kernel || 3}`,
  sep_conv2d: (b) => `SeparableConv2D · ${b.channels} filters · ${b.kernel || 3}×${b.kernel || 3}`,
  max_pool: (b) => `MaxPool1D · pool ${b.size || 2}`, max_pool2d: () => "MaxPool2D · pool 2",
  adaptive_pool: (b) => `Adaptive${(b.mode || "avg") === "max" ? "Max" : "Avg"}Pool · ${b.size || 4}`,
  upsample: (b) => `Upsample ×2 · ${(b.mode || "nearest") === "linear" ? "linear" : "nearest"}`,
  conv_t: (b) => `ConvTranspose1D · ${b.channels} filters · k=${b.kernel || 2}` + ((b.stride || 2) === 2 ? " · ×2" : ""),
  conv_t2d: (b) => `ConvTranspose2D · ${b.channels} filters · ${b.kernel || 2}×${b.kernel || 2}` + ((b.stride || 2) === 2 ? " · ×2" : ""),
  eca: (b) => `ECA · k=${b.kernel || 3}`, cbam: (b) => `CBAM · ratio ${b.ratio || 4}`, spatial_attn: (b) => `SpatialAttention · k=${b.kernel || 7}`,
  mha: (b) => `${b.causal ? "Causal " : ""}Attention · ${b.heads || 2} heads` + (b.pos === "rope" ? " · RoPE" : ""),
  linear_attn: (b) => `LinearAttention · ${b.heads || 2} heads`, cross_attn: (b) => `CrossAttention · ${b.heads || 2} heads`,
  pos_enc: (b) => `PositionalEncoding · ${(b.mode || "sin") === "learned" ? "학습형" : "sin·cos"}`,
  patch_embed: (b) => `PatchEmbedding · ${b.patch || 4}×${b.patch || 4} → ${b.dim || 32}`,
  groupnorm: (b) => `GroupNorm · ${b.groups || 4} groups`,
  spatial_dropout: (b) => `SpatialDropout · ${b.rate ?? 0.1}`, drop_path: (b) => `DropPath · ${b.rate ?? 0.1}`,
  gaussian_noise: (b) => `GaussianNoise · ${b.std ?? 0.1}`,
  multiply: () => "Multiply (×)", average: () => "Average", maximum: () => "Maximum",
  slice: (b) => `Split · 채널 ${b.start || 0}–${(b.start || 0) + b.count - 1}`,
  reshape: (b) => `Reshape · ${b.channels} 채널`, crop: (b) => `Crop · ${b.amount || 1}`, pad: (b) => `Pad · ${b.amount || 1}`,
  embedding: (b) => `Embedding · ${b.bins || 32} 구간 → ${b.dim || 16}`,
  spectrogram: (b) => `${{ log: "Spectrogram", mel: "Mel spectrogram", mfcc: "MFCC" }[b.mode || "log"]} · n_fft ${b.n_fft || 32}`,
  rand_flip: (b) => `RandomFlip · ${{ h: "좌우", v: "상하", hv: "좌우·상하" }[b.dir || "h"]}`,
  rand_rotate: (b) => `RandomRotation · ±${b.degrees || 15}°`, rand_shift: (b) => `RandomShift · ±${b.frac ?? 0.1}`,
  rand_zoom: (b) => `RandomZoom · ±${b.frac ?? 0.1}`, rand_scale: (b) => `RandomScale · ±${b.frac ?? 0.2}`,
  rand_stretch: (b) => `RandomStretch · ±${b.frac ?? 0.1}`,
  mnv3s: (b) => `MobileNetV3-Small · ${b.out === "map" ? "특징 지도" : `${b.size || 128}×${b.size || 128}`} · ${b.train_last ? `끝 ${b.train_last}블록 학습` : "고정"}`,
};
// 설정 칸: [종류, 키, 이름, …]
const num = (k, lab, min, max) => ["num", k, lab, min, max];
const sel = (k, lab, opts) => ["sel", k, lab, opts];
const yes = (k, lab) => ["sel", k, lab, [[false, "끔"], [true, "켬"]]];
const vals = (...xs) => xs.map((v) => [v, String(v)]);
const RATE = sel("rate", "Rate (끄는 비율)", vals(0.05, 0.1, 0.2, 0.3, 0.5));
const FRAC = sel("frac", "세기 (±비율)", vals(0.05, 0.1, 0.2, 0.3));
const HEADS = sel("heads", "Heads (나눠 보는 수, 채널의 약수)", vals(1, 2, 4, 8));
const FIELDS = {
  glu: [num("units", "Units (출력 채널 수)", 1, 1024), sel("gate", "문(gate)", [["sigmoid", "Sigmoid (GLU)"], ["silu", "SiLU (SwiGLU)"]])],
  dw_conv: [sel("kernel", "Kernel size", vals(1, 3, 5, 7, 9, 11, 15)), num("multiplier", "Multiplier (채널마다 필터 수)", 1, 8)],
  dw_conv2d: [sel("kernel", "Kernel size", [1, 3, 5, 7].map((v) => [v, `${v}×${v}`])), num("multiplier", "Multiplier (채널마다 필터 수)", 1, 8)],
  sep_conv: [num("channels", "Filters", 1, 256), sel("kernel", "Kernel size", vals(1, 3, 5, 7, 9, 11, 15))],
  sep_conv2d: [num("channels", "Filters", 1, 256), sel("kernel", "Kernel size", [1, 3, 5, 7].map((v) => [v, `${v}×${v}`]))],
  max_pool: [sel("size", "Pool size", vals(2, 4))],
  adaptive_pool: [num("size", "출력 크기", 1, 64), sel("mode", "방법", [["avg", "평균 (Avg)"], ["max", "최댓값 (Max)"]])],
  upsample: [sel("mode", "방법", [["nearest", "nearest (가까운 값 복사)"], ["linear", "linear (선형 보간)"]])],
  conv_t: [num("channels", "Filters", 1, 256), sel("kernel", "Kernel size", vals(1, 2, 3, 4, 5)), sel("stride", "Stride", [[2, "2 (길이 2배)"], [1, "1 (그대로)"]])],
  conv_t2d: [num("channels", "Filters", 1, 256), sel("kernel", "Kernel size", vals(1, 2, 3, 4, 5)), sel("stride", "Stride", [[2, "2 (크기 2배)"], [1, "1 (그대로)"]])],
  eca: [sel("kernel", "Kernel size (이웃 채널 수)", vals(3, 5, 7))],
  cbam: [sel("ratio", "Ratio (채널 ÷ ratio)", vals(2, 4, 8, 16)), sel("kernel", "공간 kernel size", vals(3, 5, 7))],
  spatial_attn: [sel("kernel", "Kernel size", vals(3, 5, 7))],
  mha: [HEADS, yes("causal", "Causal (앞만 봄, GPT식)"), sel("pos", "위치 정보", [["none", "없음 (PositionalEncoding을 따로)"], ["rope", "RoPE (Q·K를 위치만큼 돌림)"]])],
  linear_attn: [HEADS], cross_attn: [HEADS],
  pos_enc: [sel("mode", "방법", [["sin", "sin·cos (고정)"], ["learned", "학습형"]])],
  patch_embed: [sel("patch", "조각 크기", [2, 4, 8].map((v) => [v, `${v}×${v}`])), num("dim", "Dim (조각마다 값 수)", 1, 256)],
  groupnorm: [sel("groups", "Groups (채널의 약수)", vals(1, 2, 4, 8, 16, 32))],
  spatial_dropout: [RATE], drop_path: [RATE],
  gaussian_noise: [sel("std", "Std (잡음 크기)", vals(0.01, 0.05, 0.1, 0.2, 0.5))],
  slice: [num("start", "시작 채널 (0부터)", 0, 1024), num("count", "채널 수", 1, 1024)],
  reshape: [num("channels", "채널 수 (값 전체 개수의 약수)", 1, 65536)],
  crop: [num("amount", "양쪽에서 자를 칸", 1, 64)], pad: [num("amount", "양쪽에 채울 칸", 1, 64)],
  embedding: [sel("bins", "구간 수", vals(8, 16, 32, 64, 128, 256)), num("dim", "Dim (구간마다 값 수)", 1, 256)],
  spectrogram: [sel("mode", "종류", [["log", "Spectrogram (log 크기)"], ["mel", "Mel spectrogram"], ["mfcc", "MFCC"]]),
    sel("n_fft", "창 길이 (n_fft)", vals(8, 16, 32, 64, 128, 256)), sel("hop", "hop (창을 옮기는 칸)", vals(1, 2, 4, 8, 16, 32, 64, 128)),
    sel("n_mels", "mel 칸 수", vals(8, 16, 32, 64)), sel("n_mfcc", "MFCC 개수", vals(8, 13, 20, 40))],
  rand_flip: [sel("dir", "방향 (그림)", [["h", "좌우"], ["v", "상하"], ["hv", "좌우·상하"]])],
  rand_rotate: [num("degrees", "최대 각도 (±도)", 1, 180)],
  mnv3s: [sel("out", "출력", [["vector", "특징 1024개 (분류·이상 탐지)"], ["map", "특징 지도 (탐지: 8픽셀마다 한 칸)"]]),
    sel("size", "입력 크기 (특징 1024개일 때)", [64, 96, 128, 160, 224].map((v) => [v, `${v}×${v}`])),
    sel("train_last", "함께 학습할 끝 블록 (미세조정)", [[0, "없음 (모두 고정, 빠름)"], [1, "끝 1개"], [2, "끝 2개"], [3, "끝 3개"]])],
  rand_shift: [FRAC], rand_zoom: [FRAC], rand_scale: [FRAC], rand_stretch: [FRAC],
};

// ------------------------------------------------------------ 순수 함수: 자동 배치
const validSize = (s) => Array.isArray(s) && s.length === 2 && s.every((v) => typeof v === "number" && v > 0 && isFinite(v));
const NARROW = [132, 66], MERGE_NW = 80;                 // 가로 방향 기본: 좁고 조금 높게 (옆으로 덜 길어짐)
const sizeOfDefault = (n, dir) => {
  if (validSize(n.size)) return n.type === "act" ? [Math.min(...n.size), Math.min(...n.size)] : n.size.slice();
  if (n.type === "act") return [ACT_D, ACT_D];
  if (isMerge(n)) return [dir === "h" ? MERGE_NW : MERGE_W, MERGE_H];
  return dir === "h" ? NARROW.slice() : [BOX_W, BOX_H];
};
// 글자 폭 어림 (한글 등은 2칸)
const textUnits = (t) => [...String(t)].reduce((s, ch) => s + (ch.charCodeAt(0) > 0x2e80 ? 2 : 1), 0);
function fitText(t, w) {                                   // 폭 w(px)에 맞게 줄이고 '…'
  const max = Math.floor((w - 14) / 6.5);
  if (textUnits(t) <= max) return t;
  let out = "";
  for (const ch of String(t)) { if (textUnits(out + ch) > max - 1) break; out += ch; }
  return out + "…";
}
// 상자 안 이름: 넓으면 한 줄, 좁으면 '종류' / '설정' 두 줄 (넘치면 '…')
function labelLines(n, summary, w) {
  const full = label(n, summary);
  if (w >= 190) return [full];
  const i = full.indexOf(" · ");
  return i < 0 ? [fitText(full, w)] : [fitText(full.slice(0, i), w), fitText(full.slice(i + 3), w)];
}

// 한 층위(맨 위 또는 묶음 몸체)를 배치. 좌표는 이 층위의 (0, 0)에서.
// 층 배치(위→아래)를 늘 지킨다: 같은 층 상자는 겹치지 않게 옆으로 밀고, 선은 층 사이로만 지나가 짧고 아래로만 간다.
// 사용자가 옮긴 위치(pos)는 '희망': 가로는 그 자리(겹치면 옆으로 비킴), 세로는 그 높이에 맞는 층으로 내려감(위 상자보다 위로는 못 감).
function layoutScope(nodes, entryId, sizeOf, opts = {}) {
  const GY = opts.dir === "h" ? GAP_H : GAP_Y;            // 층 사이 간격 (가로는 동그라미·화살표가 보이게 넓게)
  const list = (nodes || []).filter((n) => n && typeof n === "object" && typeof n.id === "string");
  const items = new Map();                                // id → {id, node, w, h, ax, layer, preds: [], succ: [], dummy}
  const inner = new Map();                                // repeat id → 안쪽 배치
  const add = (id, node, w, h, ax) => items.set(id, { id, node, w, h, ax: ax === undefined ? w / 2 : ax, layer: 0, preds: [], succ: [], dummy: !node });
  if (entryId === "in") add("in", null, IN_W, IN_H), (items.get("in").entry = true);
  for (const n of list) {
    if (items.has(n.id)) continue;
    if (n.type === "repeat") {
      const L = layoutScope(n.body && n.body.nodes, "in", sizeOf, opts);
      inner.set(n.id, L);
      const en = L.entries[0];
      if (opts.dir === "h") {                                     // 가로 방향: (돌려 계산하는 좌표에서) 머리글은 옆축 앞쪽
        const cross = L.width + 2 * PAD + HEAD, w = Math.max(BOX_H, cross), off = HEAD + PAD + (w - cross) / 2;
        L.off = [off, PAD];
        add(n.id, n, w, L.height + 2 * PAD, en ? off + en.x + en.w / 2 : w / 2);
      } else {
        const w = Math.max(BOX_W, L.width + 2 * PAD);             // 가로 기준: 묶음 안 입구 (줄이 곧게 이어지게)
        L.off = [(w - L.width) / 2, HEAD];
        add(n.id, n, w, L.height + HEAD + PAD, en ? L.off[0] + en.x + en.w / 2 : w / 2);
      }
    } else {
      const [w, h] = sizeOf(n);
      add(n.id, n, w, h);
    }
  }
  const realIns = (it) => (it.node ? [...new Set(insOf(it.node))].filter((x) => items.has(x) && x !== it.id) : []);
  const hint = (it) => (it.node && validPos(it.node.pos) ? it.node.pos : null);
  const out = list.find((n) => n.type === "output");
  // 1. 층 번호: 가장 긴 경로 (순환은 끊음). lower(id): 이 층보다 위로는 안 감
  const assign = (lower) => {
    const memo = new Map(), visiting = new Set();
    const layerOf = (id) => {
      if (memo.has(id)) return memo.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      let v = lower(id);
      for (const p of realIns(items.get(id))) v = Math.max(v, layerOf(p) + 1);
      visiting.delete(id);
      memo.set(id, v);
      return v;
    };
    for (const id of items.keys()) items.get(id).layer = layerOf(id);
    if (out) {
      const mx = Math.max(0, ...[...items.values()].filter((it) => it.id !== out.id).map((it) => it.layer));
      if (items.get(out.id).layer <= mx) items.get(out.id).layer = mx + 1;
    }
  };
  assign(() => 0);
  // 2. 세로 희망: 자연 배치의 줄 높이로 '몇 번째 줄'인지 정해 그 층 아래로만
  if ([...items.values()].some((it) => hint(it) && hint(it)[1] > 0)) {
    const top = rowTops([...items.values()]);
    const pitch = BOX_H + GY;
    const layerAt = (y) => {
      let l = 0;
      while (l + 1 < top.length && top[l + 1] - GY / 2 <= y) l++;
      const last = top.length - 1;
      return l === last && y > top[last] + pitch ? last + Math.round((y - top[last]) / pitch) : l;
    };
    assign((id) => { const h = hint(items.get(id)); return h ? layerAt(h[1]) : 0; });
  }
  // 3. 선과 보이지 않는 점(dummy): 두 층 이상 건너는 선은 사이 층마다 자리를 차지해 상자와 겹치지 않음
  const edges = [];
  for (const it of [...items.values()]) {
    if (it.dummy) continue;
    for (const u of realIns(it)) {
      const a = items.get(u), chainIds = [u];
      if (it.layer > a.layer + 1) {
        for (let k = a.layer + 1; k < it.layer; k++) {
          const d = `${u}>${it.id}#${k}`;
          add(d, null, DUMMY_W, 0);
          items.get(d).layer = k;
          chainIds.push(d);
        }
      }
      chainIds.push(it.id);
      for (let k = 0; k + 1 < chainIds.length; k++) {
        const p = items.get(chainIds[k]), q = items.get(chainIds[k + 1]);
        if (q.layer > p.layer) { p.succ.push(q.id); q.preds.push(p.id); }
      }
      edges.push({ from: u, to: it.id, chain: chainIds });
    }
  }
  // 4. 층 안 순서: 깊이 우선(첫 입력을 따라감) → 무게중심 정렬
  const maxL = Math.max(0, ...[...items.values()].map((it) => it.layer));
  const rows = Array.from({ length: maxL + 1 }, () => []);
  const placed = new Set();
  const prio = (p, q) => {                                // p의 다음 중 q의 우선순위: 첫 입력 직선 < 다른 직선 < 건너뛰기
    const qi = items.get(q);
    if (qi.dummy) return 2;
    return insOf(qi.node)[0] === p ? 0 : 1;
  };
  const visit = (id) => {
    if (placed.has(id)) return;
    placed.add(id);
    const it = items.get(id);
    rows[it.layer].push(id);
    [...it.succ].sort((a, b) => prio(id, a) - prio(id, b)).forEach(visit);
  };
  [...items.values()].filter((it) => !it.preds.length).forEach((it) => visit(it.id));
  for (const it of items.values()) visit(it.id);
  const pos = new Map();
  const index = () => rows.forEach((r) => r.forEach((id, i) => pos.set(id, i)));
  index();
  const bary = (ids) => (ids.length ? ids.reduce((s, x) => s + pos.get(x), 0) / ids.length : null);
  const sortRow = (row, key) => {
    const k = row.map((id, i) => [id, key(id), i]);
    k.forEach((x) => { if (x[1] === null) x[1] = pos.get(x[0]); });
    k.sort((a, b) => a[1] - b[1] || a[2] - b[2]);
    row.splice(0, row.length, ...k.map((x) => x[0]));
  };
  for (let iter = 0; iter < 4; iter++) {
    for (let l = 1; l <= maxL; l++) sortRow(rows[l], (id) => bary(items.get(id).preds));
    index();
    for (let l = maxL - 1; l >= 0; l--) sortRow(rows[l], (id) => bary(items.get(id).succ));
    index();
  }
  // 5. 가로: 각 상자를 첫 입력(또는 희망 위치)의 가운데에 맞추고, 겹치면 오른쪽으로 밂
  const firstPred = (it) => {
    if (it.dummy) return it.preds[0];
    const p0 = insOf(it.node)[0];
    return it.preds.find((p) => p === p0 || p.startsWith(`${p0}>${it.id}#`)) || it.preds[0];
  };
  // 희망 위치가 있으면 줄마다 '가려는 가로 위치'(희망 위치 또는 첫 입력의 가운데) 순으로 세운 뒤 놓음:
  // 따라가는 상자와 건너뛰는 선이 줄마다 같은 쪽에 있어 선이 엇갈리지 않음
  const place = (useHints, auto) => {
    const cx = new Map();
    rows.forEach((r) => {
      const wantOf = (id) => {
        const it = items.get(id), h = useHints && hint(it), f = firstPred(it);
        return h ? h[0] + it.ax : f !== undefined && cx.has(f) ? cx.get(f) : null;
      };
      if (useHints) {
        const w = new Map(r.map((id) => [id, wantOf(id)]));
        const k = r.map((id, i) => [id, w.get(id) !== null ? w.get(id) : auto.get(id), i]);
        k.sort((a, b) => a[1] - b[1] || a[2] - b[2]);
        r.splice(0, r.length, ...k.map((x) => x[0]));
      }
      let cursor = -Infinity;
      for (const id of r) {
        const it = items.get(id), want = wantOf(id);
        const c = Math.max(want !== null ? want : it.ax, cursor + GAP_X + it.ax);
        cx.set(id, c);
        cursor = c + it.w - it.ax;
      }
    });
    return cx;
  };
  let cx = place(false);
  if ([...items.values()].some((it) => hint(it))) {
    cx = place(true, cx);
    index();
  }
  // 6. 세로: 줄마다 가장 높은 상자 기준
  const rowTop = rowTops([...items.values()]), rowH = rows.map((r) => Math.max(0, ...r.map((id) => items.get(id).h)));
  const minX = Math.min(0, ...[...items.values()].map((it) => cx.get(it.id) - it.ax));
  const box = (it) => ({ x: cx.get(it.id) - it.ax - minX, y: rowTop[it.layer] + (rowH[it.layer] - it.h) / 2, w: it.w, h: it.h, ax: it.ax });
  const res = { nodes: [], groups: [], edges: [], entries: [], width: 0, height: 0 };
  const entryAt = new Map();                              // 묶음 id → 묶음 안 입구 가운데
  for (const it of items.values()) {
    const b = box(it);
    if (it.entry) { res.entries.push({ x: b.x, y: b.y, w: b.w, h: b.h, anc: [] }); continue; }
    if (it.dummy) continue;
    res.nodes.push(Object.assign({ id: it.id, layer: it.layer, ox: -minX, oy: 0, anc: [] }, b));   // ox: 희망 위치(pos)의 기준점 (왼쪽 끝 보정 포함)
    if (inner.has(it.id)) {
      const L = inner.get(it.id), ox = b.x + L.off[0], oy = b.y + L.off[1];
      const up = (o) => ({ anc: [it.id].concat(o.anc || []) });           // 바깥 묶음부터 안쪽 묶음까지
      res.groups.push({ id: it.id, x: b.x, y: b.y, w: b.w, h: b.h, anc: [] });
      for (const n of L.nodes) res.nodes.push(Object.assign({}, n, { x: n.x + ox, y: n.y + oy, ox: n.ox + ox, oy: n.oy + oy, inGroup: n.inGroup || it.id }, up(n)));
      for (const gr of L.groups) res.groups.push(Object.assign({}, gr, { x: gr.x + ox, y: gr.y + oy }, up(gr)));
      for (const e of L.edges) res.edges.push(Object.assign({}, e, { points: e.points.map(([px, py]) => [px + ox, py + oy]) }, up(e)));
      L.entries.forEach((en, k) => {
        const m = Object.assign({}, en, { x: en.x + ox, y: en.y + oy }, up(en));
        res.entries.push(m);
        if (k === 0) entryAt.set(it.id, [m.x + m.w / 2, m.y + m.h / 2]);
      });
      const body = (it.node.body && it.node.body.nodes) || [];        // 출구(몸체에서 아무도 안 쓰는 노드) → 묶음 아래 가운데
      const sinks = body.filter((n) => n && !body.some((m) => insOf(m).includes(n.id)));
      const ex = sinks.length && L.nodes.find((n) => n.id === sinks[sinks.length - 1].id);
      if (ex) {
        const sx = ex.x + ox + ex.ax, sy = ex.y + oy + ex.h, gx = b.x + b.ax, gb = b.y + b.h;
        const pts = sx === gx ? [[sx, sy], [gx, gb]] : [[sx, sy], [sx, gb - PAD / 2], [gx, gb - PAD / 2], [gx, gb]];
        res.edges.push({ from: ex.id, to: it.id, kind: "exit", points: pts, anc: [it.id] });
      }
    }
  }
  // 7. 선: 아래로 → 줄 사이 높이에서 가로 → 아래로 (dummy를 따라 옆으로 돌아감)
  for (const e of edges) {
    const its = e.chain.map((id) => items.get(id));
    const pts = [];
    const first = box(its[0]);
    pts.push([first.x + first.ax, first.y + first.h]);
    if (its[its.length - 1].layer <= its[0].layer) {         // 순환 등: 바로 잇기
      const t = box(its[its.length - 1]);
      pts.push([t.x + t.ax, t.y]);
    } else {
      for (let k = 1; k < its.length; k++) {
        const prev = its[k - 1], cur = its[k], b = box(cur), midY = rowTop[prev.layer] + rowH[prev.layer] + GY / 2;
        const en = !cur.dummy && entryAt.get(cur.id);
        const x = en ? en[0] : b.x + b.ax;
        pts.push([pts[pts.length - 1][0], midY], [x, midY]);
        pts.push(cur.dummy ? [x, rowTop[cur.layer] + rowH[cur.layer]] : en ? [x, en[1]] : [x, b.y]);
      }
    }
    const uniq = pts.filter((p, i) => i === 0 || p[0] !== pts[i - 1][0] || p[1] !== pts[i - 1][1]);
    const clean = uniq.filter((p, i) => i === 0 || i === uniq.length - 1 ||                // 일직선 가운데 점 빼기
      !((uniq[i - 1][0] === p[0] && p[0] === uniq[i + 1][0]) || (uniq[i - 1][1] === p[1] && p[1] === uniq[i + 1][1])));
    res.edges.push({ from: e.from, to: e.to, points: clean });
  }
  const xs = [0], ys = [0];
  for (const o of res.nodes.concat(res.groups, res.entries)) { xs.push(o.x + o.w); ys.push(o.y + o.h); }
  for (const e of res.edges) for (const [x, y] of e.points) { xs.push(x); ys.push(y); }
  res.width = Math.max(...xs);
  res.height = Math.max(...ys);
  return res;

  function rowTops(its) {                                 // 층마다 위쪽 y (가장 높은 상자 기준 + 간격)
    const L = Math.max(0, ...its.map((it) => it.layer)), h = Array(L + 1).fill(0);
    for (const it of its) h[it.layer] = Math.max(h[it.layer], it.h);
    const t = [];
    let y = 0;
    h.forEach((v, l) => { t[l] = y; y += v + GY; });
    return t;
  }
}

// 선 그리기: 꺾인선(ortho) 또는 꺾이는 곳을 둥글게(curve)
function edgePath(pts, style) {
  if (style !== "curve" || pts.length < 3) return pts.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
  let d = `M${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [a, b, c] = [pts[i - 1], pts[i], pts[i + 1]];
    const r = Math.min(16, Math.hypot(b[0] - a[0], b[1] - a[1]) / 2, Math.hypot(c[0] - b[0], c[1] - b[1]) / 2);
    const u = (q) => { const L2 = Math.hypot(q[0] - b[0], q[1] - b[1]) || 1; return [b[0] + (q[0] - b[0]) * r / L2, b[1] + (q[1] - b[1]) * r / L2]; };
    const p1 = u(a), p2 = u(c);
    d += ` L${p1[0]} ${p1[1]} Q${b[0]} ${b[1]} ${p2[0]} ${p2[1]}`;
  }
  const z = pts[pts.length - 1];
  return d + ` L${z[0]} ${z[1]}`;
}

// 줄바꿈: 흐름축(이 좌표계의 y)을 층 경계에서 끊어 다음 줄을 옆축(x)으로 옮김. 직접(brk) 또는 자동(wrap: 한 줄 최대 길이).
// 줄 사이를 잇는 선은 '되돌아가는 선'(나가서 줄 사이 길로, 다음 줄 첫머리 앞에서 들어감)으로 다시 그림
const WRAP_GAP = 44, ROW_PAD = 30;                      // ROW_PAD: 줄 첫머리 앞 손잡이 자리
function wrapRows(L, g, wrapLen) {
  const top = L.nodes.filter((n) => !n.anc.length);
  const range = new Map();
  for (const n of top) {
    const r = range.get(n.layer) || [Infinity, -Infinity];
    range.set(n.layer, [Math.min(r[0], n.y), Math.max(r[1], n.y + n.h)]);
  }
  const layers = [...range.keys()].sort((a, b) => a - b);
  if (layers.length < 2) return L;
  const brk = new Set();
  for (const n of nodesOf(g)) {
    const t = n && n.brk && top.find((x) => x.id === n.id);
    if (t && t.layer !== layers[0]) brk.add(t.layer);
  }
  if (wrapLen > 0) {                                       // 자동: 한 줄이 길이를 넘으면 그 층부터 다음 줄
    let start = range.get(layers[0])[0];
    for (const l of layers.slice(1)) {
      if (brk.has(l)) start = range.get(l)[0];
      else if (range.get(l)[1] - start > wrapLen) { brk.add(l); start = range.get(l)[0]; }
    }
  }
  if (!brk.size) return L;
  const segOfLayer = (l) => [...brk].filter((b) => b <= l).length;
  const seg = new Map(top.map((n) => [n.id, segOfLayer(n.layer)]));
  const segOf = (o) => seg.get(o.anc && o.anc.length ? o.anc[0] : o.id);
  const nSeg = brk.size + 1;
  const crossing = (e) => !(e.anc && e.anc.length) && segOf({ id: e.from }) !== segOf({ id: e.to }) && seg.has(e.from) && seg.has(e.to);
  // 줄마다 흐름 시작과 옆축 범위
  const S = Array.from({ length: nSeg }, () => ({ y0: Infinity, x0: Infinity, x1: -Infinity }));
  const grow = (k, x0, x1, y0) => { const s = S[k]; s.x0 = Math.min(s.x0, x0); s.x1 = Math.max(s.x1, x1); if (y0 !== undefined) s.y0 = Math.min(s.y0, y0); };
  for (const n of L.nodes) grow(segOf(n), n.x, n.x + n.w, n.anc.length ? undefined : n.y);
  for (const o of L.groups.concat(L.entries)) grow(segOf(o), o.x, o.x + o.w);
  for (const e of L.edges) if (!crossing(e) && segOf({ id: e.from, anc: e.anc }) !== undefined) for (const [x] of e.points) grow(segOf({ id: e.from, anc: e.anc }), x, x);
  // 줄마다 옮긴 거리: 그 줄 상자 중 shift가 있는 것(층 순서로 처음), 없으면 줄 첫 상자가 기준
  const gnode = new Map();
  for (const n of nodesOf(g)) if (n && n.id) gnode.set(n.id, n);
  const segTop = Array.from({ length: nSeg }, () => []);
  for (const n of top) if (segOf(n) !== undefined) segTop[segOf(n)].push(n);
  segTop.forEach((a) => a.sort((p, q) => p.layer - q.layer || p.x - q.x));
  const shiftOfNode = (n) => { const m = n && gnode.get(n.id); return m && Number(m.shift) > 0 ? Number(m.shift) : 0; };
  const anchor = segTop.map((a) => a.find((n) => shiftOfNode(n) > 0) || a[0]);
  const moved = anchor.map(shiftOfNode);
  let cross = 0;
  const shift = S.map((s, k) => {
    const d = [cross - s.x0, S[0].y0 - s.y0 + ROW_PAD + moved[k]];
    cross += s.x1 - s.x0 + WRAP_GAP;
    return d;
  });
  L.rows = S.map((s, k) => ({ k, anchor: anchor[k] ? anchor[k].id : null, shift: moved[k],
    c0: s.x0 + shift[k][0], c1: s.x1 + shift[k][0], f0: S[0].y0 + ROW_PAD + moved[k] }));
  const mv = (o, k) => { const [dx, dy] = shift[k]; o.x += dx; o.y += dy; if (o.ox !== undefined) { o.ox += dx; o.oy += dy; } };
  for (const n of L.nodes) mv(n, segOf(n));
  for (const o of L.groups.concat(L.entries)) mv(o, segOf(o));
  const lanes = new Map();
  for (const e of L.edges) {
    if (crossing(e)) continue;
    const k = segOf({ id: e.from, anc: e.anc });
    if (k === undefined) continue;
    e.points = e.points.map(([x, y]) => [x + shift[k][0], y + shift[k][1]]);
  }
  for (const e of L.edges) {                                // 줄 사이 선: 나가서 → 줄 사이 길 → 다음 줄 첫머리 앞 → 들어감
    if (!crossing(e)) continue;
    const a = L.nodes.find((n) => n.id === e.from && !n.anc.length), b = L.nodes.find((n) => n.id === e.to && !n.anc.length);
    const en = L.entries.find((x) => x.anc && x.anc.length === 1 && x.anc[0] === e.to);
    const s = [a.x + a.ax, a.y + a.h], t = en ? [en.x + en.w / 2, en.y + en.h / 2] : [b.x + b.ax, b.y];
    const kb = segOf(b), used = lanes.get(kb) || 0;
    lanes.set(kb, used + 1);
    const lane = shift[kb][0] + S[kb].x0 - WRAP_GAP / 2 + ((used % 5) - 2) * 6;
    const out = s[1] + 12, inn = t[1] - 12 - (used % 3) * 4;
    e.points = [s, [s[0], out], [lane, out], [lane, inn], [t[0], inn], t];
    e.wrap = true;
  }
  const xs = [0], ys = [0];
  for (const o of L.nodes.concat(L.groups, L.entries)) { xs.push(o.x + o.w); ys.push(o.y + o.h); }
  for (const e of L.edges) for (const [x, y] of e.points) { xs.push(x); ys.push(y); }
  L.width = Math.max(...xs);
  L.height = Math.max(...ys);
  return L;
}

// opts.edge: "ortho"(꺾인선, 기본) | "curve"(곡선). opts.dir: "v"(위→아래, 기본) | "h"(왼쪽→오른쪽). opts.wrap: 자동 줄바꿈 길이(px)
// 가로는 x·y를 바꿔 같은 배치(겹침 없음·짧은 화살표)를 계산한 뒤 돌려놓음. ax는 늘 '옆축' 기준점(세로면 가로, 가로면 세로)
function layoutGraph(g, sizeOf = null, opts = {}) {
  opts = opts || {};
  const h = opts.dir === "h";
  sizeOf = sizeOf || ((n) => sizeOfDefault(n, h ? "h" : "v"));
  const swapPos = (nodes) => (nodes || []).map((n) => {
    if (!n || typeof n !== "object") return n;
    const m = Object.assign({}, n);
    if (validPos(n.pos)) m.pos = [n.pos[1], n.pos[0]];
    if (n.body && Array.isArray(n.body.nodes)) m.body = Object.assign({}, n.body, { nodes: swapPos(n.body.nodes) });
    return m;
  });
  const L = wrapRows(layoutScope(h ? swapPos(nodesOf(g)) : nodesOf(g), "input", h ? (n) => sizeOf(n).slice().reverse() : sizeOf, opts),
    g, opts.wrap ? opts.wrap - 2 * MARGIN - ROW_PAD : 0);
  const mv = (o) => Object.assign({}, o, { x: o.x + MARGIN, y: o.y + MARGIN }, o.ox !== undefined ? { ox: o.ox + MARGIN, oy: o.oy + MARGIN } : {});
  const t = (o) => (h ? Object.assign({}, o, { x: o.y, y: o.x, w: o.h, h: o.w }, o.ox !== undefined ? { ox: o.oy, oy: o.ox } : {}) : o);
  const res = {
    nodes: L.nodes.map(mv).map(t),
    groups: L.groups.map(mv).map(t),
    edges: L.edges.map((e) => Object.assign({}, e, { points: e.points.map(([x, y]) => (h ? [y + MARGIN, x + MARGIN] : [x + MARGIN, y + MARGIN])) })),
    entries: L.entries.map(mv).map(t),
    width: (h ? L.height : L.width) + 2 * MARGIN, height: (h ? L.width : L.height) + 2 * MARGIN, dir: h ? "h" : "v",
    // 줄바꿈한 줄들: 손잡이 자리 (x, y = 줄 첫머리 앞, 옆축 가운데)와 기준 상자·옮긴 거리
    rows: (L.rows || []).map((r) => {
      const f = r.f0 + MARGIN, c = (r.c0 + r.c1) / 2 + MARGIN;
      return { k: r.k, anchor: r.anchor, shift: r.shift, x: h ? f : c, y: h ? c : f };
    }),
  };
  res.nodes.sort((a, b) => a.anc.length - b.anc.length || a.layer - b.layer || (h ? a.x - b.x || a.y - b.y : a.y - b.y || a.x - b.x));
  return res;
}

// 고른 상자들을 바로 위 상자(첫 입력)와 가운데를 맞춤 → 화살표가 수직. 위에 있는 것부터 차례로
function alignToParent(g0, ids, opts) {
  let g = g0;
  const h = opts && opts.dir === "h", flow = (n) => (h ? n.x : n.y);
  const ys = layoutGraph(g, undefined, opts).nodes;
  const order = ids.filter((id) => ys.some((n) => n.id === id)).sort((a, b) => flow(ys.find((n) => n.id === a)) - flow(ys.find((n) => n.id === b)));
  for (const id of order) {
    const L = layoutGraph(g, undefined, opts), n = L.nodes.find((x) => x.id === id), f = findNode(g, id);
    const src = f && insOf(f.node)[0];
    if (!n || src === undefined) continue;
    let pc;                                                   // 위(가로면 왼쪽) 상자의 옆축 가운데
    if (src === "in") {
      const grp = n.anc[n.anc.length - 1], en = L.entries.find((x) => x.anc[x.anc.length - 1] === grp);
      if (!en) continue;
      pc = h ? en.y + en.h / 2 : en.x + en.w / 2;
    } else {
      const p = L.nodes.find((x) => x.id === src);
      if (!p) continue;
      pc = (h ? p.y : p.x) + p.ax;
    }
    // 흐름축은 0 = 바로 다음 층 (짧은 화살표)
    g = setPos(g, id, h ? [0, pc - n.ax - n.oy] : [pc - n.ax - n.ox, 0]);
  }
  return g;
}

// ------------------------------------------------------------ 화면
const LOOSE = ["입력이 연결되지 않음", "출력까지 이어지지 않음", "합칠 입력이 2개 이상"];   // 아직 선을 안 이은 것
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtShape = (s) => (!s ? "" : `(${s.join(", ")})`);
// 이미지면 기본 Conv를 Conv2D로
// 넣는 자리의 모양(rank: 2 줄 · 3 그림, 모르면 데이터 종류)에 맞춰 1D ↔ 2D 층을 바꿈
function forInput(n, input, rank) {
  const r = rank || (input === "image" ? 3 : 2);
  if (r === 3 && TWO_D[n.type]) {
    const t = TWO_D[n.type];
    if (t === "conv2d") return { type: "conv2d", channels: n.channels, kernel: 3, stride: 1 };
    if (t === "pool2d" || t === "max_pool2d") return { type: t, size: 2 };
    const m = Object.assign({}, n, { type: t });
    if (m.kernel > 7) m.kernel = 7;
    return m;
  }
  if (r === 2 && ONE_D[n.type] && rank) return Object.assign({}, n, { type: ONE_D[n.type] });
  return n;
}
const makeOf = (id) => PALETTE[id].make();

function mount(els, opts) {
  // els: {canvas, palette, settings}; opts: {onChange(graph), getSummary(), getInput(), onHint(msg), onHistory(canUndo, canRedo)}
  let graph = { version: 2, nodes: [{ id: "input", type: "input" }] }, sel = new Set(), selEdge = null, shapes = {}, issues = {}, L = null;
  let dropEdge = null, clip = null, edgeStyle = "ortho", flowDir = "v", autoWrap = false;
  let zoom = 1;                                      // 판만 확대·축소 (페이지 전체가 아니라): SVG 크기만 바꾸고 좌표(viewBox)는 그대로
  // 자동 줄바꿈 길이: 보이는 판의 흐름 방향 길이 (판 좌표로)
  const lopts = () => ({ edge: edgeStyle, dir: flowDir,
    wrap: autoWrap ? (flowDir === "h" ? els.canvas.clientWidth : els.canvas.clientHeight) / zoom - 30 : 0 });
  const hist = history();
  const input = () => (opts.getInput ? opts.getInput() : "timeseries");
  const hint = (msg) => opts.onHint && opts.onHint(msg);
  const emit = () => opts.onChange(clone(graph));
  function set(g) {                                  // 대표 모델 바꾸기 등: 되돌리기 가능
    if (nodesOf(graph).length > 1 && JSON.stringify(g) !== JSON.stringify(graph)) hist.push(graph);
    graph = clone(g); sel = new Set(); selEdge = null; draw();
  }
  function get() { return clone(graph); }
  function change(g, newSel) {                       // newSel: id 목록 또는 undefined(그대로)
    if (JSON.stringify(g) === JSON.stringify(graph)) return draw();
    hist.push(graph);
    graph = g;
    issues = {};                                     // 예전 그래프의 문제 표시는 바로 지움 (새 검사 결과가 오면 다시)
    if (newSel !== undefined) { sel = new Set(newSel); selEdge = null; }
    for (const id of [...sel]) if (!findNode(graph, id)) sel.delete(id);
    draw(); emit();
  }
  function undo() { const p = hist.undo(graph); if (p) { graph = p; sel = new Set(); selEdge = null; draw(); emit(); } return !!p; }
  function redo() { const p = hist.redo(graph); if (p) { graph = p; sel = new Set(); selEdge = null; draw(); emit(); } return !!p; }
  function decorate(checked) {
    shapes = (checked && checked.shapes) || {};
    issues = {};
    for (const i of (checked && checked.issues) || []) (issues[i.node] = issues[i.node] || []).push(i);
    draw();
  }
  function select(id) { sel = new Set(id ? [id] : []); selEdge = null; draw(); }
  const selIds = () => [...sel].filter((id) => findNode(graph, id));
  const outputNode = () => nodesOf(graph).find((n) => n.type === "output");

  function draw() {
    const summary = opts.getSummary();
    L = layoutGraph(graph, undefined, lopts());
    const top = (o) => (o.anc && o.anc.length ? ` data-anc="${esc(o.anc.join(" "))}"` : "");
    // 판: 보이는 칸을 다 채우고, 오른쪽·아래로 여유를 둬서 어디로든 끌어 둘 수 있게
    const W = Math.max(L.width + 240, (els.canvas.clientWidth - 24) / zoom), H = Math.max(L.height + 240, (els.canvas.clientHeight - 24) / zoom);   // 판 안쪽 여백(8px×2) 빼고
    let s = `<svg viewBox="0 0 ${W} ${H}" width="${Math.round(W * zoom)}" height="${Math.round(H * zoom)}" class="flow-svg">` +
      `<defs><marker id="fa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
      `<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.6" stroke-linecap="round"/></marker></defs>`;
    for (const gr of L.groups) {
      const n = findNode(graph, gr.id).node, on = sel.has(gr.id), bad = issues[gr.id];
      s += `<g class="fg${on ? " sel" : ""}${bad ? " bad" : ""}" data-node="${esc(gr.id)}"${top(gr)}>` +
        `<rect x="${gr.x}" y="${gr.y}" width="${gr.w}" height="${gr.h}" rx="12" class="grp"/>` +
        `<rect x="${gr.x}" y="${gr.y}" width="${gr.w}" height="${HEAD - 4}" rx="12" class="grp-head"><title>묶음 머리를 끌면 묶음째 옮김</title></rect>` +
        `<text x="${gr.x + 12}" y="${gr.y + 18}" class="glabel">${esc(label(n))}${shapes[gr.id] ? "  " + esc(fmtShape(shapes[gr.id])) : ""}${bad ? "  ⚠ " + esc(bad[0].message) : ""}</text>` +
        port(L.nodes.find((n) => n.id === gr.id) || gr) + `</g>`;
    }
    for (const e of L.edges) {
      const d = edgePath(e.points, edgeStyle);
      if (e.kind === "exit") { s += `<path d="${d}" class="edge exit"${top(e)}/>`; continue; }     // 묶음 출구 → 묶음 아래
      const to = findNode(graph, e.to), skip = to && insOf(to.node)[0] !== e.from;
      const on = selEdge && selEdge.from === e.from && selEdge.to === e.to;
      const key = esc(`${e.from}>${e.to}`);
      const into = L.groups.some((gr) => gr.id === e.to);         // 묶음으로 들어가는 선은 안쪽 입구 점까지 이어짐 (화살촉 없음)
      s += `<path d="${d}" class="edge${skip ? " skip" : ""}${on ? " sel" : ""}"${into ? "" : ' marker-end="url(#fa)"'} data-from="${esc(e.from)}" data-to="${esc(e.to)}"${top(e)}/>` +
        `<path d="${d}" class="edge-hit" data-edge="${key}"${top(e)}/>`;
    }
    for (const en of L.entries) s += `<circle cx="${en.x + en.w / 2}" cy="${en.y + en.h / 2}" r="4" class="entry"${top(en)}><title>묶음의 입력</title></circle>`;
    for (const r of L.rows || []) {                            // 줄바꿈한 줄마다 손잡이: 끌면 그 줄을 흐름 방향(가로면 좌우)으로 옮김
      if (!r.anchor) continue;
      const hz = L.dir === "h", gw = hz ? 14 : 44, gh = hz ? 44 : 14, gx = hz ? r.x - 24 : r.x - 22, gy = hz ? r.y - 22 : r.y - 24;
      s += `<g class="row-grip" data-row="${r.k}" data-anchor="${esc(r.anchor)}" data-shift="${r.shift}">` +
        `<title>끌어서 ${hz ? "이 줄을 좌우" : "이 열을 위아래"}로 옮기기 (두 번 누르면 맨 ${hz ? "왼쪽" : "위"}으로)</title>` +
        `<rect x="${gx}" y="${gy}" width="${gw}" height="${gh}" rx="4"/>` +
        `<text x="${gx + gw / 2}" y="${gy + gh / 2}">${hz ? "⇔" : "⇕"}</text></g>`;
    }
    for (const bx of L.nodes) {
      if (L.groups.some((gr) => gr.id === bx.id)) continue;
      const n = findNode(graph, bx.id).node, on = sel.has(bx.id);
      // 아직 선을 안 이은 것뿐이면 오류가 아니라 '선을 이어 주세요' (회색 점선)
      const loose = issues[bx.id] && issues[bx.id].every((i) => LOOSE.some((w) => i.message.includes(w)));
      const bad = issues[bx.id] && (loose ? [{ message: "선을 이어 주세요" }] : issues[bx.id]);
      const T = TYPES[n.type] || {}, cls = `fb${on ? " sel" : ""}${loose ? " loose" : bad ? " bad" : ""}`;
      const head = `<g class="${cls}" data-node="${esc(bx.id)}"${top(bx)}>`, cx = bx.x + bx.w / 2, cy = bx.y + bx.h / 2;
      if (isMerge(n)) {
        s += head + `<title>${esc(bad ? bad[0].message : fmtShape(shapes[bx.id]))}</title>` +
          `<rect x="${bx.x}" y="${bx.y}" width="${bx.w}" height="${bx.h}" rx="15" class="box c-add"/>` +
          `<text x="${cx}" y="${cy}" class="t1">${esc(label(n))}${bad ? " ⚠" : ""}</text>` + port(bx) + `</g>`;
        continue;
      }
      if (n.type === "act") {                                  // 활성화는 원
        s += head + `<title>${esc(label(n))}${bad ? " — " + esc(bad[0].message) : ""}</title>` +
          `<circle cx="${cx}" cy="${cy}" r="${bx.w / 2}" class="box c-purple"/>` +
          `<text x="${cx}" y="${cy - 7}" class="t1">${esc(shortLabel(n))}${bad ? " ⚠" : ""}</text>` +
          `<text x="${cx}" y="${cy + 10}" class="t3">${esc(fmtShape(shapes[bx.id]))}</text>` + port(bx) + `</g>`;
        continue;
      }
      const color = n.type === "input" ? "gray" : COLORS[T.cat] || "gray", narrow = bx.w < 190;
      const inShape = summary && fmtShape(summary.image ? [1].concat(summary.image) : [1, summary.length]);
      const subFull = n.type === "input"
        ? (!summary ? "데이터를 먼저 불러오세요" : narrow ? inShape : `${inShape}  channels × ${summary.image ? "height × width" : "length"}`)
        : bad ? bad[0].message : fmtShape(shapes[bx.id]);
      const lines = labelLines(n, summary, bx.w), sub = narrow ? fitText(subFull, bx.w) : subFull;
      const warn = bad && !loose ? " ⚠" : "";
      const text = lines.length === 1
        ? `<text x="${cx}" y="${cy - 8}" class="t1">${esc(lines[0])}${warn}</text><text x="${cx}" y="${cy + 9}" class="t2">${esc(sub)}</text>`
        : `<text x="${cx}" y="${cy - 16}" class="t1">${esc(lines[0])}${warn}</text><text x="${cx}" y="${cy}" class="t2b">${esc(lines[1])}</text>` +
          `<text x="${cx}" y="${cy + 16}" class="t2">${esc(sub)}</text>`;
      const brkMark = n.brk ? `<text x="${bx.x + 8}" y="${bx.y + 11}" class="brk">↵</text>` : "";   // 여기서 줄바꿈
      s += head + `<title>${esc(label(n, summary))} — ${esc(subFull)}</title>` + brkMark +
        `<rect x="${bx.x}" y="${bx.y}" width="${bx.w}" height="${bx.h}" rx="10" class="box c-${color}"/>` + text + port(bx) + `</g>`;
    }
    for (const bx of L.nodes) {                      // 고른 상자 크기 손잡이 (끌면 고른 상자 모두 같은 크기로)
      if (!sel.has(bx.id) || L.groups.some((gr) => gr.id === bx.id)) continue;
      s += `<rect class="rsz" data-rsz="${esc(bx.id)}" x="${bx.x + bx.w - 7}" y="${bx.y + bx.h - 7}" width="11" height="11" rx="2"><title>끌어서 크기 바꾸기 (고른 상자 모두)</title></rect>`;
    }
    s += `<rect id="band" class="band" style="display:none"/><path id="drop-edge" class="drop" style="display:none" fill="none"/>` +
      `<line id="wire" class="wire" style="display:none"/><g id="snap"></g>`;
    els.canvas.innerHTML = s + "</svg>";
    drawSettings();
    if (opts.onHistory) opts.onHistory(hist.canUndo(), hist.canRedo());
  }
  function port(bx) {                                // 소켓: 위 = 입력, 아래 = 출력. 끌어서 다른 상자에 놓으면 선, 빈 곳에 놓으면 층 고르기
    const f = findNode(graph, bx.id), t = f ? f.node.type : "", a = bx.ax === undefined ? (L.dir === "h" ? bx.h : bx.w) / 2 : bx.ax;
    const [ix, iy, ox, oy] = L.dir === "h" ? [bx.x, bx.y + a, bx.x + bx.w, bx.y + a] : [bx.x + a, bx.y, bx.x + a, bx.y + bx.h];
    let s = "";
    if (t !== "input") s += `<circle cx="${ix}" cy="${iy}" r="5" class="port in" data-inport="${esc(bx.id)}"><title>입력: 끌어서 위 상자에 놓으면 선을 이음 (빈 곳에 놓으면 앞에 넣을 층을 고름)</title></circle>`;
    if (t !== "output") s += `<circle cx="${ox}" cy="${oy}" r="5" class="port" data-port="${esc(bx.id)}"><title>출력: 끌어서 아래 상자에 놓으면 선을 이음 (빈 곳에 놓으면 이어서 만들 층을 고름)</title></circle>`;
    return s;
  }

  function field(lab, inp) { return `<label class="fld"><span>${lab}</span>${inp}</label>`; }
  function numberInput(k, v, min, max) { return `<input type="number" data-k="${k}" value="${esc(v)}" min="${min}" max="${max}" step="1">`; }
  function selectInput(k, v, options) {
    return `<select data-k="${k}">${options.map(([val, txt]) => `<option value="${val}"${String(v) === String(val) ? " selected" : ""}>${esc(txt)}</option>`).join("")}</select>`;
  }
  const KEYS = `<details class="keys"><summary>조작 방법</summary><p class="muted"><b>마우스만으로:</b> 왼쪽 목록에서 끌어 판에 놓기 (선 근처면 그 사이로) · 상자를 끌어 옮기기 (선 근처면 그 사이로, 여러 개 고른 채 끌면 함께) · ` +
    `상자 위·아래 동그라미를 끌어 다른 상자에 놓으면 선 (기존 선은 바뀜), 빈 곳에 놓으면 층을 골라 바로 이어서 만듦 · <b>오른쪽 클릭</b>: 복사·삭제·감싸기·정렬·줄바꿈(세로면 열바꿈)·선 지우기·층 넣기<br>` +
    `키보드도 됨: ` +
    `여러 개 고르기: 빈 곳에서 끌어서 네모, 또는 Shift+클릭 · Ctrl+C 복사 · Ctrl+X 잘라내기 · Ctrl+V 붙여넣기 · Ctrl+D 복제 · Delete 삭제 · Ctrl+Z 되돌리기 · Ctrl+Y 다시 하기 · Esc 선택 해제</p></details>`;
  // 처음 보는 사람에게 덜 중요한 칸은 접어 둠 (기본값이 아니면 펼침)
  const adv = (inner, open) => (inner ? `<details class="adv"${open ? " open" : ""}><summary>고급 설정</summary>${inner}</details>` : "");

  function drawSettings() {
    const summary = opts.getSummary();
    if (selEdge) {
      const a = findNode(graph, selEdge.from), b = findNode(graph, selEdge.to);
      els.settings.innerHTML = `<h4>선</h4><p class="muted">${esc(a ? label(a.node, summary) : "묶음 입력")} → ${esc(b ? label(b.node, summary) : "")}</p>` +
        `<div class="acts"><button data-a="del-edge" class="danger">선 지우기</button></div>` + KEYS;
      return;
    }
    const ids = selIds();
    if (!ids.length) {
      els.settings.innerHTML = KEYS;
      return;
    }
    if (ids.length > 1) {
      const ok = !!singleEntryExit(graph, ids);
      els.settings.innerHTML = `<h4>${ids.length}개 선택</h4>` +
        (ok ? `<p class="muted">입구 하나·출구 하나인 묶음입니다</p>` : `<p class="muted">감싸기·복사는 입구 하나·출구 하나로 이어진 층들만 됩니다</p>`) +
        `<div class="acts"><button data-a="wrap-res"${ok ? "" : " disabled"}>Residual로 감싸기</button><button data-a="wrap-rep"${ok ? "" : " disabled"}>Repeat로 감싸기</button>` +
        `<button data-a="split"${ok ? "" : " disabled"}>갈래로 나누기</button><button data-a="copy"${ok ? "" : " disabled"}>복사</button>` +
        `<button data-a="dup"${ok ? "" : " disabled"}>복제</button><button data-a="del" class="danger">삭제</button></div>` + sizeFields(ids) + KEYS;
      return;
    }
    const b = findNode(graph, ids[0]).node, T = TYPES[b.type] || {};
    if (b.type === "input") {
      els.settings.innerHTML = `<h4>Input</h4><p class="muted">② 데이터 단계에서 불러온 데이터입니다${summary ? ` (length ${summary.length})` : ""}.</p>` + KEYS;
      return;
    }
    let f = "";
    if (b.type === "conv") f = field("Filters (채널 수: 많을수록 정확, 느림)", numberInput("channels", b.channels, 1, 256)) +
      field("Kernel size (한 번에 보는 칸 수)", selectInput("kernel", b.kernel, [1, 3, 5, 7, 9, 11, 15].map((v) => [v, v]))) +
      adv(field("Stride", selectInput("stride", b.stride || 1, [[1, "1"], [2, "2 (길이 절반)"]])) +
        field("Dilation (칸 사이를 띄워 넓게 봄, TCN)", selectInput("dilation", b.dilation || 1, [1, 2, 4, 8, 16, 32].map((v) => [v, v]))) +
        field("Padding", selectInput("padding", b.padding || "same", [["same", "same (길이 그대로)"], ["valid", "valid (가장자리만큼 줄어듦)"], ["causal", "causal (과거만 봄)"]])) +
        field("Groups (filters·들어오는 채널을 나누는 수, 같으면 depthwise)", numberInput("groups", b.groups || 1, 1, 256)),
      (b.stride || 1) !== 1 || (b.dilation || 1) !== 1 || (b.padding || "same") !== "same" || (b.groups || 1) !== 1);
    if (b.type === "pool") f = field("Pool size", selectInput("size", b.size, [[2, "2"], [4, "4"]]));
    if (b.type === "conv2d") f = field("Filters (채널 수: 많을수록 정확, 느림)", numberInput("channels", b.channels, 1, 256)) +
      field("Kernel size (한 번에 보는 네모 크기)", selectInput("kernel", b.kernel, [1, 3, 5, 7].map((v) => [v, `${v}×${v}`]))) +
      adv(field("Stride", selectInput("stride", b.stride || 1, [[1, "1"], [2, "2 (크기 절반)"]])) +
        field("Dilation (칸 사이를 띄워 넓게 봄)", selectInput("dilation", b.dilation || 1, [1, 2, 4, 8].map((v) => [v, v]))) +
        field("Groups (같으면 depthwise)", numberInput("groups", b.groups || 1, 1, 256)) +
        field("Padding", selectInput("padding", b.padding || "same", [["same", "same (크기 그대로)"], ["valid", "valid (가장자리만큼 줄어듦)"]])),
      (b.stride || 1) !== 1 || (b.padding || "same") !== "same" || (b.dilation || 1) !== 1 || (b.groups || 1) !== 1);
    if (b.type === "act") f = field("Function", selectInput("kind", b.kind, ACTS));
    if (b.type === "se") f = field("Ratio (줄이는 비율: 채널 ÷ ratio)", selectInput("ratio", b.ratio, [2, 4, 8, 16].map((v) => [v, v]))) +
      adv(field("안쪽 activation", selectInput("act", b.act || "relu", [["relu", "ReLU"], ["sigmoid", "Sigmoid"], ["tanh", "Tanh"]])), (b.act || "relu") !== "relu");
    if (b.type === "dropout") f = field("Rate (끄는 비율)", selectInput("rate", b.rate, [0.1, 0.2, 0.3, 0.5].map((v) => [v, v])));
    if (b.type === "dense") f = field("Units (출력 개수)", numberInput("units", b.units, 1, 1024));
    if (b.type === "lstm" || b.type === "gru" || b.type === "simple_rnn") f = field("Units (hidden size)", numberInput("hidden", b.hidden, 1, 256)) +
      field("Layers", numberInput("layers", b.layers || 1, 1, 3)) +
      field("Bidirectional (앞→뒤·뒤→앞 둘 다)", selectInput("bidirectional", !!b.bidirectional, [[false, "끔"], [true, "켬"]])) +
      field("Return sequences (줄 그대로 내보내기: 뒤에 순환·어텐션·Conv를 쌓을 때)", selectInput("return_sequences", !!b.return_sequences, [[false, "끔 (마지막 값만)"], [true, "켬"]]));
    if (FIELDS[b.type]) {
      const d = TYPES[b.type] ? TYPES[b.type].make() : {};
      f = FIELDS[b.type].map(([kind, k, lab, a1, a2]) => field(lab, kind === "num" ? numberInput(k, b[k] ?? d[k], a1, a2) : selectInput(k, b[k] ?? d[k], a1))).join("");
    }
    if (TWO_INPUT.includes(b.type)) f += `<div class="fld"><span>들어오는 선 (첫째 = 질문, 둘째 = 참고)</span><ol class="ins">${insOf(b).map((x, i) => {
      const src = findNode(graph, x);
      return `<li>${esc(src ? label(src.node, summary) : x)} <span class="muted">${esc(fmtShape(shapes[x]))}</span> <button data-a="in-del" data-i="${i}" title="이 선 지우기">×</button></li>`;
    }).join("")}</ol></div>`;
    if (b.type === "repeat") f = field("Repeat count", numberInput("times", b.times, 1, 10));
    if (isMerge(b)) {
      f = field("합치는 방법", selectInput("type", b.type, [["add", "Add (+): 더하기, 모양이 같아야 함"], ["concat", "Concat: 채널로 이어 붙이기"],
        ["multiply", "Multiply (×): 원소마다 곱하기"], ["average", "Average: 원소마다 평균"], ["maximum", "Maximum: 원소마다 큰 값"]])) +
        `<div class="fld"><span>들어오는 선 (Concat은 이 순서로 붙임)</span><ol class="ins">${insOf(b).map((x, i) => {
          const src = findNode(graph, x);
          return `<li>${esc(src ? label(src.node, summary) : x === "in" ? "묶음 입력" : x)} <span class="muted">${esc(fmtShape(shapes[x]))}</span> <button data-a="in-del" data-i="${i}" title="이 선 지우기">×</button></li>`;
        }).join("")}</ol></div>`;
    }
    const isOutput = b.type === "output";
    const look = opts.onPreview ? `<div class="acts"><button data-a="preview" title="학습 샘플 하나가 이 층을 지나면 어떤 값·모양이 되는지 오른쪽 아래에 보여 줌">미리보기: 샘플이 이 층을 지나면</button></div>` : "";
    const btns = look + (isOutput ? "" : `<div class="acts"><button data-a="copy">복사</button><button data-a="dup">복제</button>`) + (isOutput ? "" :
      `<button data-a="del" class="danger">삭제</button>` + (b.type === "repeat" ? `<button data-a="unwrap">묶음 풀기</button>` : "") +
      `<button data-a="wrap-res">Residual로 감싸기</button><button data-a="wrap-rep">Repeat로 감싸기</button><button data-a="split">갈래로 나누기</button></div>`);
    const bad = issues[b.id] || [];
    els.settings.innerHTML = `<h4>${esc(label(b, summary))}</h4><p class="muted">${esc(T.desc || (b.type === "repeat" ? "안의 층을 여러 번 쌓음 (가중치는 따로)" : ""))}</p>${f}${btns}` +
      sizeFields(ids) +
      bad.map((i) => `<p class="err">${esc(i.message)}<br>→ ${esc(i.hint)}</p>`).join("") +
      (isOutput ? `<p class="muted">분류·회귀는 ① 과제에서 고릅니다. 출력 개수는 데이터에 맞춰 정해집니다.</p>` : "") + KEYS;
  }

  function sizeFields(ids) {                         // 상자 크기 (보기용): 고른 상자 모두에 한 번에
    const boxes = ids.map((id) => L.nodes.find((n) => n.id === id)).filter((n) => n && !L.groups.some((gr) => gr.id === n.id));
    if (!boxes.length) return "";
    return `<details class="adv size-fld"><summary>상자 크기 (보기용${boxes.length > 1 ? ` · ${boxes.length}개 모두` : ""})</summary>` +
      `<div class="row"><label>가로 <input type="number" data-size="w" value="${esc(Math.round(boxes[0].w))}" min="${SIZE_MIN[0]}" max="${SIZE_MAX[0]}" step="10"></label>` +
      `<label>세로 <input type="number" data-size="h" value="${esc(Math.round(boxes[0].h))}" min="${SIZE_MIN[1]}" max="${SIZE_MAX[1]}" step="5"></label>` +
      `<button data-a="size-reset">기본 크기로</button></div><span class="muted">고른 상자 오른쪽 아래 네모를 끌어도 됨</span></details>`;
  }
  let openCat = null, search = "";                     // 처음에는 모든 범주를 접어 둠
  function packNote(p, kind) {
    if (kind === "image" && !p.ranks.includes(3)) return "줄(시계열) 모양에 씀 — 그림은 ToSequence 뒤에";
    if (kind !== "image" && !p.ranks.includes(2) && !p.ranks.includes(1)) return "그림 모양에 씀";
    return "";
  }
  function itemHtml(id, cat, name, desc, note) {
    return `<div class="pitem p-${cat}${note ? " off" : ""}" data-t="${esc(id)}" title="끌어서 판 위에 놓기 (선 근처면 그 선 사이로)">` +
      `<b>${esc(name)}</b><span>${esc(desc)}</span>${note ? `<em class="fit">${esc(note)}</em>` : ""}</div>`;
  }
  function drawPalette() {
    if (!els.palette.querySelector(".psearch")) {
      els.palette.innerHTML = `<input class="psearch" type="search" placeholder="블록 찾기 (예: attention, LSTM, norm)" aria-label="블록 찾기"><div class="plist"></div>`;
      const box = els.palette.querySelector(".psearch");
      box.addEventListener("input", () => { search = box.value.trim().toLowerCase(); drawPalette(); });
      box.addEventListener("mousedown", (e) => e.stopPropagation());
    }
    const kind = input(), all = paletteTypes(kind), list = els.palette.querySelector(".plist");
    const items = (cat) => cat === "pack"
      ? PACKS.map((p) => ["pack:" + p.id, "pack", p.name, p.desc, packNote(p, kind)])
      : all.filter((id) => PALETTE[id].cat === cat).map((id) => [id, cat, PALETTE[id].name, PALETTE[id].desc, fitNote(makeOf(id).type, kind)]);
    let h = "";
    if (search) {
      const hits = CATS.flatMap(([cat, cname]) => items(cat).filter((it) => (it[2] + " " + it[3] + " " + cname).toLowerCase().includes(search)));
      h = hits.length ? hits.map((it) => itemHtml(...it)).join("") : `<p class="muted">'${esc(search)}'에 맞는 블록이 없습니다</p>`;
    } else {
      for (const [cat, name] of CATS) {
        const its = items(cat), on = openCat === cat;
        h += `<button type="button" class="pcat-btn p-${cat}${on ? " on" : ""}" data-cat="${cat}" aria-expanded="${on}"><span>${esc(name)}</span><em>${its.length}</em></button>`;
        if (on) h += `<div class="pitems">${its.map((it) => itemHtml(...it)).join("")}</div>`;
      }
    }
    list.innerHTML = h;
  }
  els.palette.addEventListener("click", (e) => {
    const b = e.target.closest("[data-cat]");
    if (!b) return;
    openCat = openCat === b.dataset.cat ? null : b.dataset.cat;
    drawPalette();
  });
  const paletteName = (id) => (id.startsWith("pack:") ? (PACKS.find((p) => "pack:" + p.id === id) || {}).name : PALETTE[id].name);
  const rankAt = (id) => (id === "input" || id === "in" ? (shapes[id] || []).length || (input() === "image" ? 3 : 2) : (shapes[id] || []).length || 0);
  function dropPack(id, edge) {                      // 묶음 넣기: 선 위면 그 사이, 아니면 Output 바로 앞
    let from, to;
    if (edge) { from = edge.from; to = edge.to; } else {
      const out = outputNode();
      if (!out || !insOf(out).length) return hint("묶음은 선 위에 놓으세요");
      from = insOf(out)[0]; to = out.id;
    }
    const sh = from === "input" ? (opts.getSummary() ? (opts.getSummary().image ? [1].concat(opts.getSummary().image) : [1, opts.getSummary().length]) : null) : shapes[from];
    const pk = PACKS.find((p) => "pack:" + p.id === id), rank = sh ? sh.length : (input() === "image" ? 3 : 2);
    let pack;
    try { pack = makePack(pk.id, sh ? sh[0] : 16, rank, sh && sh.length === 2 ? sh[1] : 0); } catch (err) { return hint(String(err.message || err)); }
    const r = insertPack(graph, from, to, pack);
    change(r.graph, r.ids);
    hint(`'${pk.name}'을(를) 넣었습니다${edge ? "" : " (Output 바로 앞)"}. 상자를 눌러 설정을 바꿀 수 있습니다`);
  }

  // ---------------- 편집 동작
  function removeSel() {
    if (selEdge) return change(removeEdge(graph, selEdge.from, selEdge.to), []);
    let g = graph;
    for (const id of selIds()) g = removeNode(g, id);
    if (g === graph) return hint("Input·Output은 지울 수 없습니다");
    change(g, []);
  }
  function copySel(cut) {
    const ids = selIds().filter((id) => { const n = findNode(graph, id).node; return n.type !== "input" && n.type !== "output"; });
    if (!ids.length) return hint("복사할 층을 먼저 고르세요");
    const c = copyNodes(graph, ids);
    if (!c) return hint("입구 하나·출구 하나로 이어진 층들만 복사할 수 있습니다");
    clip = c;
    if (cut) { let g = graph; for (const id of ids) g = removeNode(g, id); change(g, []); }
    hint(`${ids.length}개 ${cut ? "잘라냄" : "복사함"} · Ctrl+V로 붙여넣기 (고른 상자 바로 뒤)`);
  }
  function paste(c) {
    if (!c) return hint("붙여넣을 것이 없습니다 (먼저 Ctrl+C)");
    const ids = selIds();
    const s = ids.length > 1 ? singleEntryExit(graph, ids) : null;
    const after = s ? s.exit : ids.length === 1 ? ids[0] : (insOf(outputNode())[0] || "input");
    const r = pasteAfter(graph, after, c);
    if (!r.ids.length) return hint("출력 뒤에는 붙일 수 없습니다");
    change(r.graph, r.ids);
  }
  function act(a, extra) {
    const ids = selIds();
    if (a === "preview") return opts.onPreview && opts.onPreview(ids[0]);
    if (a === "del-edge" || a === "del") return removeSel();
    if (a === "copy") return copySel(false);
    if (a === "dup") { const c = copyNodes(graph, ids); return c ? paste(c) : hint("입구 하나·출구 하나로 이어진 층들만 복제할 수 있습니다"); }
    if (a === "unwrap") return change(unwrap(graph, ids[0]), []);
    if (a === "size-reset") return change(setSize(graph, ids, null), ids);
    if (a === "in-del") { const n = findNode(graph, ids[0]).node; return change(removeEdge(graph, insOf(n)[Number(extra.i)], n.id)); }
    if (!singleEntryExit(graph, ids)) return hint("입구 하나·출구 하나로 이어진 층들만 감쌀 수 있습니다");
    if (a === "wrap-res") return change(wrapResidual(graph, ids, Object.assign({ input: summaryShape() }, shapes)), []);
    if (a === "wrap-rep") return change(wrapRepeat(graph, ids), []);
    if (a === "split") return change(splitBranch(graph, ids, forInput({ type: "conv", channels: 8, kernel: 5, stride: 1 }, input())), []);
  }
  function summaryShape() {
    const sm = opts.getSummary();
    return sm ? (sm.image ? [1].concat(sm.image) : [1, sm.length]) : null;
  }

  // ---------------- 마우스로 끌기: 왼쪽 목록 → 판, 상자 옮기기, 선 잇기, 네모 고르기
  // 선 근처(48px)에 놓으면 그 선 사이로 들어가고, 아니면 놓은 자리에 둠 (판 위 자유 배치)
  function svgPoint(e) {
    const svg = els.canvas.querySelector("svg");
    if (!svg || !svg.getScreenCTM()) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM().inverse());
    return [q.x, q.y];
  }
  const overCanvas = (e) => {
    const r = els.canvas.getBoundingClientRect();
    return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
  };
  function showDrop(edge) {
    dropEdge = edge;
    const p = els.canvas.querySelector("#drop-edge");
    if (!p) return;
    if (!edge) { p.style.display = "none"; return; }
    p.style.display = "";
    p.setAttribute("d", edge.points.map((q, i) => `${i ? "L" : "M"}${q[0]} ${q[1]}`).join(" "));
  }
  let drag = null, band = null, wire = null, skipClick = false, ghost = null;
  function ignoreNextClick() {                       // 끌기 직후 브라우저가 보내는 click 하나만 버림 (안 오면 바로 풀림)
    skipClick = true;
    setTimeout(() => { skipClick = false; }, 0);
  }
  function movedGraph(d, p) {                         // 끈 상자들을 판 위 그 자리로 옮긴 그래프 (희망 위치)
    let g = graph;
    for (const id of d.ids) {
      const bx = L.nodes.find((n) => n.id === id);
      if (bx) g = setPos(g, id, [bx.x + p[0] - d.x0 - bx.ox, bx.y + p[1] - d.y0 - bx.oy]);
    }
    return g;
  }
  function showSnap(g, ids) {                         // 놓으면 실제로 들어갈 자리 (겹치지 않게 맞춘 뒤)
    const el = els.canvas.querySelector("#snap");
    if (!el) return;
    if (!g) { el.innerHTML = ""; return; }
    const L2 = layoutGraph(g, undefined, lopts());
    el.innerHTML = L2.nodes.filter((n) => ids.includes(n.id))
      .map((n) => `<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.w === n.h ? n.w / 2 : 10}" class="snap"/>`).join("");
  }
  function setGhost(text, e) {
    if (!ghost) { ghost = document.createElement("div"); ghost.className = "drag-ghost"; document.body.appendChild(ghost); }
    ghost.textContent = text;
    ghost.style.left = `${e.clientX + 14}px`;
    ghost.style.top = `${e.clientY + 10}px`;
  }
  function clearGhost() { if (ghost) ghost.remove(); ghost = null; els.canvas.classList.remove("dragging"); }

  els.canvas.addEventListener("dblclick", (e) => {            // 줄 손잡이 두 번: 그 줄을 맨 앞(왼쪽·위)으로
    const rg = e.target.closest("[data-row]");
    if (rg && Number(rg.dataset.shift) > 0) change(setShift(graph, rg.dataset.anchor, 0));
  });
  els.palette.addEventListener("mousedown", (e) => {
    const it = e.target.closest("[data-t]");
    if (!it || e.button !== 0) return;
    drag = { kind: "palette", type: it.dataset.t, el: it, x0: e.clientX, y0: e.clientY, moved: false };
    e.preventDefault();                                          // 글자 고르기 막기
  });
  els.canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const p = svgPoint(e);
    const rz = e.target.closest("[data-rsz]");
    if (rz && p) {
      const bx = L.nodes.find((n) => n.id === rz.dataset.rsz);
      drag = { kind: "size", id: bx.id, x0: p[0], y0: p[1], bx, ids: selIds().filter((id) => !L.groups.some((gr) => gr.id === id)) };
      e.preventDefault();
      return;
    }
    const rg = e.target.closest("[data-row]");
    if (rg && p) {
      drag = { kind: "row", anchor: rg.dataset.anchor, shift0: Number(rg.dataset.shift) || 0, x0: p[0], y0: p[1], base: graph, moved: false };
      e.preventDefault();
      return;
    }
    const ip = e.target.closest("[data-inport]");
    if (ip && p) { wire = { to: ip.dataset.inport, x0: p[0], y0: p[1] }; e.preventDefault(); return; }
    const pt = e.target.closest("[data-port]");
    if (pt && p) { wire = { from: pt.dataset.port, x0: p[0], y0: p[1] }; e.preventDefault(); return; }
    const g = e.target.closest("[data-node]");
    if (g && p) {
      const id = g.dataset.node, isGroup = g.classList.contains("fg");
      if (isGroup && !e.target.closest(".grp-head, .glabel")) return;       // 묶음은 머리로만 끌기 (안쪽 빈 곳은 네모 고르기 없음)
      const ids = sel.has(id) && sel.size > 1 ? selIds() : [id];
      drag = { kind: "node", id, ids, x0: p[0], y0: p[1], moved: false };
      e.preventDefault();
      return;
    }
    if (e.target.closest("[data-edge]")) return;
    if (p) band = { x0: p[0], y0: p[1], x1: p[0], y1: p[1], moved: false };
  });
  document.addEventListener("mousemove", (e) => {
    if (drag && drag.kind === "palette") {
      drag.moved = drag.moved || Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) > 4;
      if (!drag.moved) return;
      els.canvas.classList.add("dragging");
      const p = overCanvas(e) && L ? svgPoint(e) : null;
      const edge = p ? nearestEdge(L.edges, p) : null;
      if (edge !== dropEdge) showDrop(edge);
      const pk = drag.type.startsWith("pack:");
      setGhost(`${paletteName(drag.type)} — ${!p ? "판 위에 놓으세요" : edge ? "이 선 사이에 넣기" : pk ? "Output 바로 앞에 넣기" : "여기에 두기 (선은 나중에 이음)"}`, e);
      return;
    }
    if (drag && drag.kind === "row") {                         // 줄 옮기기: 놓기 전까지는 보기만 (되돌리기 기록은 놓을 때 한 번)
      const p = svgPoint(e);
      if (!p) return;
      const d = flowDir === "h" ? p[0] - drag.x0 : p[1] - drag.y0;
      drag.moved = drag.moved || Math.abs(d) > 3;
      if (!drag.moved) return;
      drag.g = setShift(drag.base, drag.anchor, drag.shift0 + d);
      graph = drag.g;
      draw();
      graph = drag.base;
      return;
    }
    if (drag && drag.kind === "size") {
      const p = svgPoint(e), el = els.canvas.querySelector("#snap");
      if (!p || !el) return;
      const w = Math.max(SIZE_MIN[0], drag.bx.w + p[0] - drag.x0), h = Math.max(SIZE_MIN[1], drag.bx.h + p[1] - drag.y0);
      drag.size = [w, h];
      el.innerHTML = `<rect x="${drag.bx.x}" y="${drag.bx.y}" width="${w}" height="${h}" rx="10" class="snap"/>`;
      return;
    }
    if (drag && drag.kind === "node") {
      const p = svgPoint(e);
      if (!p) return;
      const dx = p[0] - drag.x0, dy = p[1] - drag.y0;
      drag.moved = drag.moved || Math.abs(dx) + Math.abs(dy) > 5;
      if (!drag.moved) return;
      const id = drag.id, ids = drag.ids;
      const hit = (el) => ids.includes(el.dataset.node) || (el.dataset.anc || "").split(" ").some((a) => ids.includes(a));
      for (const el of els.canvas.querySelectorAll("[data-node], [data-anc]")) el.setAttribute("transform", hit(el) ? `translate(${dx} ${dy})` : "");
      const n = findNode(graph, id).node, fixed = ids.length > 1 || n.type === "input" || n.type === "output";   // 여럿이면 자리만 옮김
      const edge = fixed ? null : nearestEdge(L.edges, p, SNAP, (ed) => ed.from === id || ed.to === id || (ed.anc || []).includes(id));
      if (edge !== dropEdge) showDrop(edge);
      showSnap(edge ? null : movedGraph(drag, p), ids);
      return;
    }
    if (wire) {
      const p = svgPoint(e), w = els.canvas.querySelector("#wire");
      if (!p || !w) return;
      w.style.display = "";
      w.setAttribute("x1", wire.x0); w.setAttribute("y1", wire.y0); w.setAttribute("x2", p[0]); w.setAttribute("y2", p[1]);
      return;
    }
    if (!band) return;
    const p = svgPoint(e), r = els.canvas.querySelector("#band");
    if (!p || !r) return;
    band.x1 = p[0]; band.y1 = p[1];
    band.moved = band.moved || Math.abs(band.x1 - band.x0) + Math.abs(band.y1 - band.y0) > 6;
    if (!band.moved) return;
    r.style.display = "";
    r.setAttribute("x", Math.min(band.x0, band.x1)); r.setAttribute("y", Math.min(band.y0, band.y1));
    r.setAttribute("width", Math.abs(band.x1 - band.x0)); r.setAttribute("height", Math.abs(band.y1 - band.y0));
  });
  document.addEventListener("mouseup", (e) => {
    if (drag && drag.kind === "palette") {
      const d = drag, edge = dropEdge;
      drag = null;
      clearGhost();
      showDrop(null);
      if (!d.moved) {                                            // 누르기만 함: 안내
        d.el.classList.add("nudge");
        setTimeout(() => d.el.classList.remove("nudge"), 600);
        return hint("층은 끌어서 판 위에 놓으세요 (선 근처면 그 선 사이로 들어감)");
      }
      const p = overCanvas(e) ? svgPoint(e) : null;
      if (!p) return;
      if (d.type.startsWith("pack:")) return dropPack(d.type, edge);
      const node = forInput(makeOf(d.type), input(), edge ? rankAt(edge.from) : 0), nid = newId(graph);
      if (edge) {
        change(insertOnEdge(graph, edge.from, edge.to, Object.assign({ id: nid }, node)), [nid]);
        if (isMerge(node)) hint("합치기 상자는 입력이 2개 이상이어야 합니다: 다른 상자의 동그라미를 끌어 이 상자에 놓으세요");
        else if (maxIns(node) === 2) hint("입력이 둘인 상자입니다: 참고할 다른 상자의 동그라미를 끌어 이 상자에 놓으세요 (첫째 = 질문, 둘째 = 참고)");
        else if (/^rand_/.test(node.type)) hint("증강 블록은 Input 바로 아래에 두세요 (학습 때만 그림·신호를 바꿈)");
      } else {
        const [w, h] = sizeOfDefault(node, flowDir), o = L.nodes.find((n) => !n.anc.length) || { ox: 0, oy: 0 };
        change(addFree(graph, Object.assign({ id: nid }, node), [p[0] - w / 2 - o.ox, p[1] - h / 2 - o.oy]), [nid]);
        hint("아직 이어지지 않은 상자입니다: 위 상자 아래의 동그라미를 끌어 이 상자에 놓으세요");
      }
      return;
    }
    if (drag && drag.kind === "row") {
      const d = drag;
      drag = null;
      ignoreNextClick();
      if (!d.moved || !d.g) return;                         // 누르기만 함 (두 번 누르기가 같은 손잡이에 가도록 다시 그리지 않음)
      return change(d.g);
    }
    if (drag && drag.kind === "size") {
      const d = drag;
      drag = null;
      ignoreNextClick();
      if (!d.size) return draw();
      return change(setSize(graph, d.ids, d.size), d.ids);
    }
    if (drag && drag.kind === "node") {
      const d = drag, edge = dropEdge;
      drag = null;
      showDrop(null);
      if (!d.moved) return;                                      // 누르기만 함: click에서 고름
      ignoreNextClick();
      const p = svgPoint(e);
      if (edge) {
        const r = moveNode(graph, d.id, edge.from, edge.to);
        if (r.error) { hint(r.error); return draw(); }
        return change(r.graph, [d.id]);
      }
      if (!p) return draw();
      return change(movedGraph(d, p), d.ids);                    // 판 위 그 자리로 (겹치면 옆으로, 세로는 층에 맞춤)
    }
    if (wire) {
      const w = wire, out = w.from !== undefined;
      wire = null;
      ignoreNextClick();
      const target = e.target.closest && e.target.closest("[data-node]");
      const ww = els.canvas.querySelector("#wire");
      if (ww) ww.style.display = "none";
      const p = svgPoint(e);
      if (target) {                                  // 상자에 놓음: 선 잇기 (보통 층은 기존 선을 바꿈)
        const tid = target.dataset.node;
        if (tid === (out ? w.from : w.to)) return;
        const r = out ? connect(graph, w.from, tid) : connect(graph, tid, w.to);
        if (r.error) return hint(r.error);
        return change(r.graph, [out ? tid : w.to]);
      }
      if (!p || !overCanvas(e) || Math.hypot(p[0] - w.x0, p[1] - w.y0) < 12) return;
      layerMenu(e, (t) => {                          // 빈 곳에 놓음: 층을 골라 바로 이어서 만듦
        const node = forInput(makeOf(t), input(), out ? rankAt(w.from) : 0), [nw, nh] = sizeOfDefault(node, flowDir);
        const ref = L.nodes.find((n) => n.id === (out ? w.from : w.to)) || { ox: 0, oy: 0 };
        const pos = [p[0] - nw / 2 - ref.ox, p[1] - nh / 2 - ref.oy];
        const r = out ? addConnected(graph, w.from, node, pos) : addFeeding(graph, w.to, node, pos);
        if (r.id) change(r.graph, [r.id]);
      });
      return;
    }
    if (!band) return;
    const b = band;
    band = null;
    const r = els.canvas.querySelector("#band");
    if (r) r.style.display = "none";
    if (!b.moved) return;
    ignoreNextClick();                                          // 끈 뒤 따라오는 click이 선택을 지우지 않게
    const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1), y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
    sel = new Set(L.nodes.filter((n) => n.x >= x0 && n.y >= y0 && n.x + n.w <= x1 && n.y + n.h <= y1).map((n) => n.id)
      .filter((id) => { const n = findNode(graph, id); return n && n.node.type !== "input" && n.node.type !== "output"; }));
    // 묶음째 고른 경우 안쪽 노드는 빼기
    for (const gr of L.groups) if (sel.has(gr.id)) for (const n of L.nodes) if (n.inGroup === gr.id) sel.delete(n.id);
    selEdge = null;
    draw();
  });
  els.canvas.addEventListener("click", (e) => {
    if (skipClick) { skipClick = false; return; }
    const ed = e.target.closest("[data-edge]");
    if (ed) { const [from, to] = ed.dataset.edge.split(">"); sel = new Set(); selEdge = { from, to }; return draw(); }
    const g = e.target.closest("[data-node]");
    if (!g) return select(null);
    const id = g.dataset.node;
    if (e.shiftKey) { if (sel.has(id)) sel.delete(id); else sel.add(id); selEdge = null; return draw(); }
    select(id);
  });
  // ---------------- 오른쪽 클릭 메뉴, 층 고르기 메뉴 (키보드 없이)
  const menu = document.createElement("div");
  menu.id = "ctx-menu";
  menu.hidden = true;
  document.body.appendChild(menu);
  let menuItems = [];
  function openMenu(e, items, cls = "") {
    menuItems = items;
    menu.className = `ctx-menu ${cls}`;
    menu.innerHTML = items.map((it, i) => (it.sep ? "<hr>" : it.title ? `<div class="ctx-title">${esc(it.title)}</div>`
      : `<button data-i="${i}"${it.disabled ? " disabled" : ""}${it.danger ? ' class="danger"' : ""}>${esc(it.label)}</button>`)).join("");
    menu.hidden = false;
    menu.style.left = `${Math.max(4, Math.min(e.clientX, innerWidth - menu.offsetWidth - 8))}px`;
    menu.style.top = `${Math.max(4, Math.min(e.clientY, innerHeight - menu.offsetHeight - 8))}px`;
  }
  const closeMenu = () => { menu.hidden = true; };
  menu.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-i]");
    if (!b || b.disabled) return;
    const it = menuItems[Number(b.dataset.i)];
    closeMenu();
    it.fn();
  });
  document.addEventListener("mousedown", (ev) => { if (!menu.hidden && !menu.contains(ev.target)) closeMenu(); }, true);
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeMenu(); });
  els.canvas.addEventListener("scroll", closeMenu);
  function layerMenu(e, pick) {
    const kind = input(), all = paletteTypes(kind), items = [];
    for (const [cat, name] of CATS) {
      const ts = all.filter((t) => PALETTE[t].cat === cat);
      if (!ts.length) continue;
      items.push({ title: name });
      for (const t of ts) items.push({ label: PALETTE[t].name + (fitNote(makeOf(t).type, kind) ? " · 잘 안 맞음" : ""), fn: () => pick(t) });
    }
    openMenu(e, items, "layers");
  }
  els.canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const p = svgPoint(e);
    if (!p) return;
    const g = e.target.closest("[data-node]"), ed = e.target.closest("[data-edge]");
    if (g) {
      const id = g.dataset.node, n = findNode(graph, id).node, fixed = n.type === "input" || n.type === "output";
      if (!sel.has(id)) { sel = new Set([id]); selEdge = null; draw(); }
      const many = selIds().length > 1;
      return openMenu(e, [
        { title: many ? `${selIds().length}개 고름` : label(n, opts.getSummary()) },
        { label: many ? "위 상자와 가운데 맞추기 (고른 것 모두)" : "위 상자와 가운데 맞추기", fn: () => autoLayout(), disabled: n.type === "input" },
        ...(!n.anc && L.nodes.some((x) => x.id === id && !x.anc.length) && n.type !== "input"
          ? [{ label: flowDir === "h" ? (n.brk ? "줄바꿈 풀기" : "여기서 줄바꿈 (아래 줄로)") : (n.brk ? "열바꿈 풀기" : "여기서 열바꿈 (오른쪽 열로)"),
            fn: () => change(setBreak(graph, id, !n.brk), [id]) }] : []),
        { label: "복사", fn: () => copySel(false), disabled: fixed },
        { label: "복제 (바로 뒤에)", fn: () => act("dup"), disabled: fixed },
        { label: "이 뒤에 붙여넣기", fn: () => paste(clip), disabled: !clip || n.type === "output" },
        { sep: true },
        { label: "Residual로 감싸기", fn: () => act("wrap-res"), disabled: fixed },
        { label: "Repeat로 감싸기", fn: () => act("wrap-rep"), disabled: fixed },
        { label: "갈래로 나누기", fn: () => act("split"), disabled: fixed || input() === "tabular" },
        ...(n.type === "repeat" ? [{ label: "묶음 풀기", fn: () => act("unwrap") }] : []),
        { sep: true },
        { label: "삭제", fn: () => removeSel(), danger: true, disabled: fixed },
      ]);
    }
    if (ed) {
      const [from, to] = ed.dataset.edge.split(">");
      sel = new Set(); selEdge = { from, to }; draw();
      return openMenu(e, [
        { title: "선" },
        { label: "이 선에 층 넣기…", fn: () => layerMenu(e, (t) => {
          const nid = newId(graph);
          change(insertOnEdge(graph, from, to, Object.assign({ id: nid }, forInput(makeOf(t), input(), rankAt(from)))), [nid]);
        }) },
        { label: "선 지우기", fn: () => change(removeEdge(graph, from, to), []), danger: true },
      ]);
    }
    select(null);
    openMenu(e, [
      { title: "판" },
      { label: "여기에 층 추가…", fn: () => layerMenu(e, (t) => {
        const node = forInput(makeOf(t), input()), [nw, nh] = sizeOfDefault(node, flowDir), o = L.nodes.find((n) => !n.anc.length) || { ox: 0, oy: 0 };
        const nid = newId(graph);
        change(addFree(graph, Object.assign({ id: nid }, node), [p[0] - nw / 2 - o.ox, p[1] - nh / 2 - o.oy]), [nid]);
      }) },
      { label: "붙여넣기", fn: () => paste(clip), disabled: !clip },
      { label: "전체 자동 정렬", fn: () => autoLayout() },
    ]);
  });

  els.settings.addEventListener("change", (e) => {
    if (e.target.dataset.size) {                     // 크기 칸: 고른 상자 모두
      const w = Number(els.settings.querySelector('[data-size="w"]').value), h = Number(els.settings.querySelector('[data-size="h"]').value);
      return change(setSize(graph, selIds(), [w, h]), selIds());
    }
    const k = e.target.dataset.k, ids = selIds();
    if (!k || ids.length !== 1) return;
    const raw = e.target.value;
    const v = raw === "true" ? true : raw === "false" ? false : raw !== "" && !isNaN(Number(raw)) ? Number(raw) : raw;   // same·relu 같은 글자 값은 그대로
    change(setParam(graph, ids[0], k, v));
  });
  els.settings.addEventListener("click", (e) => {
    const a = e.target.dataset.a;
    if (a) act(a, e.target.dataset);
  });
  document.addEventListener("keydown", (e) => {
    if (!els.canvas.offsetParent || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;   // 모델 화면에서만
    const ctrl = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
    let done = true;
    if ((e.key === "Delete" || e.key === "Backspace") && (sel.size || selEdge)) removeSel();
    else if (ctrl && k === "z" && !e.shiftKey) { if (!undo()) hint("더 되돌릴 것이 없습니다"); }
    else if (ctrl && (k === "y" || (k === "z" && e.shiftKey))) { if (!redo()) hint("다시 할 것이 없습니다"); }
    else if (ctrl && k === "c") copySel(false);
    else if (ctrl && k === "x") copySel(true);
    else if (ctrl && k === "v") paste(clip);
    else if (ctrl && k === "d") act("dup");
    else if (ctrl && k === "a") { sel = new Set(nodesOf(graph).filter((n) => n.type !== "input" && n.type !== "output").map((n) => n.id)); selEdge = null; draw(); }
    else if (e.key === "Escape") select(null);
    else done = false;
    if (done) e.preventDefault();
  });
  // ---------------- 판 확대·축소: Ctrl(⌘)+휠·트랙패드 집기·두 손가락 집기는 판에서만 (페이지는 그대로), 단추·Ctrl+0/+/−
  const ZMIN = 0.2, ZMAX = 2;
  function setZoom(z, at) {                          // at: 화면 좌표 [x, y] — 그 점 아래의 판 위치가 그대로 있게
    z = Math.min(ZMAX, Math.max(ZMIN, Math.round(z * 100) / 100));
    if (z === zoom) return;
    const r = els.canvas.getBoundingClientRect(), cx = at ? at[0] : r.left + r.width / 2, cy = at ? at[1] : r.top + r.height / 2;
    const p = svgPoint({ clientX: cx, clientY: cy });
    zoom = z;
    draw();
    const svg = els.canvas.querySelector("svg");
    if (p && svg) {
      const b = svg.getBoundingClientRect();
      els.canvas.scrollLeft += b.left + p[0] * zoom - cx;
      els.canvas.scrollTop += b.top + p[1] * zoom - cy;
    }
    if (opts.onZoom) opts.onZoom(zoom);
  }
  function fitZoom() {                               // 모델 전체가 보이게 (100%보다 키우지는 않음)
    if (!L) return;
    const z = Math.min(1, (els.canvas.clientWidth - 40) / (L.width + 40), (els.canvas.clientHeight - 40) / (L.height + 40));
    zoom = Math.min(ZMAX, Math.max(ZMIN, Math.floor(z * 100) / 100));
    draw();
    els.canvas.scrollLeft = 0;
    els.canvas.scrollTop = 0;
    if (opts.onZoom) opts.onZoom(zoom);
  }
  els.canvas.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) return;           // 그냥 휠은 판 스크롤
    e.preventDefault();                              // 브라우저 확대 대신 판만
    setZoom(zoom * Math.exp(-Math.max(-60, Math.min(60, e.deltaY)) / 300), [e.clientX, e.clientY]);
  }, { passive: false });
  let pinch = null;                                  // 터치: 두 손가락 집기
  const tdist = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
  els.canvas.addEventListener("touchstart", (e) => { if (e.touches.length === 2) pinch = { d: tdist(e.touches), z: zoom }; }, { passive: true });
  els.canvas.addEventListener("touchmove", (e) => {
    if (!pinch || e.touches.length !== 2) return;
    e.preventDefault();
    const t = e.touches;
    setZoom(pinch.z * tdist(t) / pinch.d, [(t[0].clientX + t[1].clientX) / 2, (t[0].clientY + t[1].clientY) / 2]);
  }, { passive: false });
  els.canvas.addEventListener("touchend", (e) => { if (e.touches.length < 2) pinch = null; });
  document.addEventListener("keydown", (e) => {
    if (!els.canvas.offsetParent || !(e.ctrlKey || e.metaKey) || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if (e.key === "=" || e.key === "+") setZoom(zoom * 1.2);
    else if (e.key === "-" || e.key === "_") setZoom(zoom / 1.2);
    else if (e.key === "0") setZoom(1);
    else return;
    e.preventDefault();
  });

  drawPalette();
  draw();
  function autoLayout() {                            // 고른 상자가 있으면 바로 위 상자와 가운데 맞춤, 없으면 전체 자동 정렬
    const ids = selIds().filter((id) => findNode(graph, id).node.type !== "input");
    if (ids.length) { change(alignToParent(graph, ids, lopts()), ids); return hint(`고른 ${ids.length}개를 바로 ${flowDir === "h" ? "왼쪽" : "위"} 상자와 가운데 맞췄습니다`); }
    change(clearPos(graph));
    hint("자동 정렬했습니다 (판 위에 옮겨 둔 위치를 지움). 상자를 고르고 누르면 그 상자만 위 상자와 가운데 맞춤");
  }
  function setEdgeStyle(v) { edgeStyle = v === "curve" ? "curve" : "ortho"; draw(); }
  function setWrap(on) { autoWrap = !!on; draw(); }
  function setFlowDir(v, fromUser) {                 // 방향을 바꾸면 옮겨 둔 위치는 지움 (축이 바뀌어 뜻이 달라짐, 되돌리기 됨)
    flowDir = v === "h" ? "h" : "v";
    let had = false;
    eachNode(nodesOf(graph), (n) => { if (n.pos) had = true; });
    if (fromUser && had) { change(clearPos(graph)); hint("방향을 바꿔 옮겨 둔 위치를 지웠습니다 (되돌리기로 살릴 수 있음)"); }
    else draw();
  }
  window.addEventListener("resize", () => { if (els.canvas.offsetParent) draw(); });
  return { set, get, decorate, select, redraw: draw, redrawPalette: drawPalette, undo, redo, autoLayout, setEdgeStyle, setFlowDir, setWrap,
    zoomIn: () => setZoom(zoom * 1.2), zoomOut: () => setZoom(zoom / 1.2), zoomReset: () => setZoom(1), zoomFit: fitZoom, getZoom: () => zoom };
}

const API = { PALETTE, PACKS, makePack, insertPack, RANK, mount, fmtShape, forInput, labelLines, setBreak, setShift, alignToParent, edgePath, addConnected, addFeeding, setSize, stripView, setPos, clearPos, stripPos, addFree, moveNode, shortLabel, nearestEdge, TYPES, ACT_NAMES, CATS, paletteTypes, fitNote, label, layoutGraph, insertOnEdge, connect, removeNode, removeEdge,
  singleEntryExit, wrapResidual, wrapRepeat, unwrap, splitBranch, copyNodes, pasteAfter, setParam, newId, findNode, history, clone };
if (typeof module !== "undefined") module.exports = API;
if (typeof window !== "undefined") window.StudioGraph = API;
})();

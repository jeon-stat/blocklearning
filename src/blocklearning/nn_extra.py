"""추가 블록의 PyTorch 모듈. studio 모델(blocks.make_layer)이 쓰고, 코드 내보내기는 쓴 클래스의 소스를 그대로 붙여 넣는다
(그래서 클래스마다 혼자 읽혀야 함: math·torch·nn·F만 씀). 값의 모양은 채널이 먼저: (N, C, L) · (N, C, H, W).
웹 엔진(static/engine.js)이 같은 식을 TF.js로 옮겼다 (tests/test_engine.py가 값이 같은지 확인).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class GLU(nn.Module):
    """위치마다 채널을 2·units개로 바꿔 반은 값, 반은 문(gate): 값 × σ(문) (GLU) 또는 값 × SiLU(문) (SwiGLU)."""

    def __init__(self, c, units, gate="sigmoid"):
        super().__init__()
        self.proj, self.gate = nn.Linear(c, 2 * units), gate

    def forward(self, x):
        h = self.proj(x.movedim(1, -1))
        a, b = h.chunk(2, dim=-1)
        y = a * (torch.sigmoid(b) if self.gate == "sigmoid" else F.silu(b))
        return y.movedim(-1, 1)


class SeparableConv(nn.Module):
    """Depthwise(채널마다 따로 k칸) + Pointwise(1칸으로 채널 섞기): 같은 일을 훨씬 적은 가중치로 (MobileNet·Xception)."""

    def __init__(self, c, out, k, dims=1):
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv1d
        self.depthwise = conv(c, c, k, padding=k // 2, groups=c)
        self.pointwise = conv(c, out, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class GlobalMax(nn.Module):
    """채널마다 전체에서 가장 큰 값."""

    def forward(self, x):
        return x.flatten(2).amax(-1)


class FirstStep(nn.Module):
    """채널마다 첫 위치 값 (CLS token 자리)."""

    def forward(self, x):
        return x[:, :, 0]


class AttentionPool(nn.Module):
    """위치마다 점수를 매겨 softmax 가중치로 평균: 중요한 위치를 더 많이 반영."""

    def __init__(self, c):
        super().__init__()
        self.score = nn.Linear(c, 1)

    def forward(self, x):
        s = x.flatten(2)
        a = torch.softmax(self.score(s.transpose(1, 2)).squeeze(-1), dim=-1)
        return (s * a.unsqueeze(1)).sum(-1)


class PixelShuffle(nn.Module):
    """채널을 위치로 옮겨 2배로 키움: (C·2, L) → (C, 2L), (C·4, H, W) → (C, 2H, 2W)."""

    def forward(self, x):
        if x.dim() == 4:
            return F.pixel_shuffle(x, 2)
        n, c, length = x.shape
        return x.reshape(n, c // 2, 2, length).permute(0, 1, 3, 2).reshape(n, c // 2, length * 2)


class ECA(nn.Module):
    """Efficient Channel Attention: 채널 평균을 이웃 채널끼리 k칸 Conv1D → sigmoid → 채널마다 곱함 (SE의 가벼운 판)."""

    def __init__(self, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.sigmoid(self.conv(x.flatten(2).mean(-1).unsqueeze(1))).squeeze(1)
        return x * s.reshape(*s.shape, *([1] * (x.dim() - 2)))


class SpatialAttention(nn.Module):
    """위치마다 채널 평균·최댓값 → k칸 Conv → sigmoid: '어디를 볼지' 지도를 곱함 (CBAM의 공간 부분)."""

    def __init__(self, k=7, dims=2):
        super().__init__()
        self.conv = (nn.Conv2d if dims == 2 else nn.Conv1d)(2, 1, k, padding=k // 2)

    def forward(self, x):
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(m))


class CBAM(nn.Module):
    """채널 주의(평균·최댓값 → 같은 MLP → 더해 sigmoid) 다음 공간 주의 (Convolutional Block Attention Module)."""

    def __init__(self, c, ratio=4, k=7, dims=2):
        super().__init__()
        h = max(1, c // ratio)
        self.fc1, self.fc2 = nn.Linear(c, h), nn.Linear(h, c)
        self.conv = (nn.Conv2d if dims == 2 else nn.Conv1d)(2, 1, k, padding=k // 2)

    def forward(self, x):
        f = x.flatten(2)
        mlp = lambda v: self.fc2(torch.relu(self.fc1(v)))
        ca = torch.sigmoid(mlp(f.mean(-1)) + mlp(f.amax(-1)))
        x = x * ca.reshape(*ca.shape, *([1] * (x.dim() - 2)))
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(m))


class MultiHeadAttention(nn.Module):
    """Self-attention: 위치마다 질문(Q)·열쇠(K)·값(V)을 만들고, 질문과 맞는 열쇠의 값을 softmax 가중치로 모음 (head마다 따로).
    causal이면 앞 위치만 봄 (GPT식). rope면 Q·K를 위치만큼 돌려 순서를 앎 (RoPE). 그림은 칸들을 한 줄로 보고 돌려놓음."""

    def __init__(self, c, heads=2, causal=False, rope=False):
        super().__init__()
        self.qkv, self.proj = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.heads, self.causal, self.rope = heads, causal, rope

    @staticmethod
    def rotate(x):
        """RoPE: (N, heads, T, d)의 짝지은 채널 (2i, 2i+1)을 위치 × 10000^(-2i/d)만큼 돌림."""
        T, d = x.shape[-2], x.shape[-1]
        inv = 10000.0 ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
        ang = torch.arange(T, dtype=x.dtype, device=x.device)[:, None] * inv[None]
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x):
        shape = x.shape
        s = x.flatten(2).transpose(1, 2)                                   # (N, T, C)
        n, t, c = s.shape
        q, k, v = (z.reshape(n, t, self.heads, c // self.heads).transpose(1, 2) for z in self.qkv(s).chunk(3, dim=-1))
        if self.rope:
            q, k = self.rotate(q), self.rotate(k)
        att = q @ k.transpose(-1, -2) / math.sqrt(c // self.heads)
        if self.causal:
            att = att.masked_fill(torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1), -1e9)
        y = (torch.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(n, t, c)
        return self.proj(y).transpose(1, 2).reshape(shape)


class LinearAttention(nn.Module):
    """softmax 대신 φ(x) = ELU(x) + 1로 (Q·K)V를 Q·(KᵀV)로 바꿔 길이에 비례하는 계산 (Linear Transformer)."""

    def __init__(self, c, heads=2):
        super().__init__()
        self.qkv, self.proj, self.heads = nn.Linear(c, 3 * c), nn.Linear(c, c), heads

    def forward(self, x):
        shape = x.shape
        s = x.flatten(2).transpose(1, 2)
        n, t, c = s.shape
        q, k, v = (z.reshape(n, t, self.heads, c // self.heads).transpose(1, 2) for z in self.qkv(s).chunk(3, dim=-1))
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = k.transpose(-1, -2) @ v                                       # (N, heads, d, d)
        z = 1.0 / (q @ k.sum(-2, keepdim=True).transpose(-1, -2) + 1e-6)    # (N, heads, T, 1)
        y = ((q @ kv) * z).transpose(1, 2).reshape(n, t, c)
        return self.proj(y).transpose(1, 2).reshape(shape)


class NonLocal(nn.Module):
    """Non-local block: 모든 위치 쌍의 관계(θᵀφ, softmax)로 g를 모아 1×1로 되돌린 뒤 입력에 더함 (먼 칸끼리)."""

    def __init__(self, c):
        super().__init__()
        h = max(1, c // 2)
        self.theta, self.phi, self.g, self.out = nn.Linear(c, h), nn.Linear(c, h), nn.Linear(c, h), nn.Linear(h, c)

    def forward(self, x):
        s = x.flatten(2).transpose(1, 2)
        att = torch.softmax(self.theta(s) @ self.phi(s).transpose(1, 2), dim=-1)
        y = self.out(att @ self.g(s))
        return x + y.transpose(1, 2).reshape(x.shape)


class CrossAttention(nn.Module):
    """첫 입력(질문)의 위치마다 둘째 입력(참고)의 어디를 볼지 정해 값을 모음. 출력 모양은 첫 입력과 같음."""

    def __init__(self, c, c2, heads=2):
        super().__init__()
        self.q, self.kv, self.proj, self.heads = nn.Linear(c, c), nn.Linear(c2, 2 * c), nn.Linear(c, c), heads

    def forward(self, x, ctx):
        shape = x.shape
        s, r = x.flatten(2).transpose(1, 2), ctx.flatten(2).transpose(1, 2)
        n, t, c = s.shape
        split = lambda z: z.reshape(n, z.shape[1], self.heads, c // self.heads).transpose(1, 2)
        k, v = self.kv(r).chunk(2, dim=-1)
        q, k, v = split(self.q(s)), split(k), split(v)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(c // self.heads), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(n, t, c)
        return self.proj(y).transpose(1, 2).reshape(shape)


class PositionalEncoding(nn.Module):
    """위치마다 정해진 값을 더해 순서를 알려 줌: sin·cos 표 (고정) 또는 배우는 값 (learned)."""

    def __init__(self, c, n, learned=False):
        super().__init__()
        if learned:
            self.pe = nn.Parameter(torch.randn(c, n) * 0.02)
        else:
            pe = torch.tensor([[(math.sin if ch % 2 == 0 else math.cos)(p / 10000 ** ((ch - ch % 2) / c)) for p in range(n)]
                               for ch in range(c)])
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe.reshape(x.shape[1:])


class PatchEmbedding(nn.Module):
    """그림을 patch×patch 조각으로 잘라 조각마다 dim개 값으로 → (dim, 조각 수) 줄 (ViT의 입구)."""

    def __init__(self, c, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(c, dim, patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2)


class ClsToken(nn.Module):
    """줄 맨 앞에 배우는 자리 하나(CLS)를 붙임. 어텐션을 지난 뒤 FirstTimestep으로 꺼내 전체 요약으로 씀."""

    def __init__(self, c):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(c) * 0.02)

    def forward(self, x):
        return torch.cat([self.cls.reshape(1, -1, 1).expand(x.shape[0], -1, -1), x], dim=2)


class LayerNorm(nn.Module):
    """위치마다 채널 값들을 평균 0·분산 1로 맞춘 뒤 채널마다 배우는 배율·이동 (Transformer의 정규화)."""

    def __init__(self, c):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        y = F.layer_norm(x.movedim(1, -1), (x.shape[1],), self.weight, self.bias, 1e-5)
        return y.movedim(-1, 1)


class RMSNorm(nn.Module):
    """위치마다 채널 값들의 제곱평균으로만 나눔 (평균은 빼지 않음, LayerNorm보다 가벼움 · 최신 LLM)."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-5)
        return y * self.weight.reshape(-1, *([1] * (x.dim() - 2)))


class DropPath(nn.Module):
    """학습 때 샘플마다 rate 확률로 이 길 전체를 0으로 (남은 것은 1/(1−rate)배). Residual 안 마지막에 둠 (Stochastic Depth)."""

    def __init__(self, rate=0.1):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        keep = (torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) >= self.rate).to(x.dtype)
        return x * keep / (1 - self.rate)


class GaussianNoise(nn.Module):
    """학습 때만 평균 0·표준편차 std 잡음을 더함."""

    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std if self.training else x


class Split(nn.Module):
    """채널 일부만 꺼냄 (start부터 count개). 같은 값에서 둘 이상 꺼내면 채널 나누기."""

    def __init__(self, start, count):
        super().__init__()
        self.start, self.count = start, count

    def forward(self, x):
        return x[:, self.start:self.start + self.count]


class Reshape(nn.Module):
    """값 전체를 (c, 나머지) 줄로, h가 있으면 (c, h, 나머지) 그림으로 다시 묶음."""

    def __init__(self, c, h=0):
        super().__init__()
        self.c, self.h = c, h

    def forward(self, x):
        return x.reshape(x.shape[0], self.c, self.h, -1) if self.h else x.reshape(x.shape[0], self.c, -1)


class Transpose(nn.Module):
    """줄은 채널↔길이, 그림은 높이↔너비를 바꿈."""

    def forward(self, x):
        return x.transpose(1, 2) if x.dim() == 3 else x.transpose(2, 3)


class ToSequence(nn.Module):
    """그림 (C, H, W) → 칸들의 줄 (C, H·W)."""

    def forward(self, x):
        return x.flatten(2)


class Crop(nn.Module):
    """양쪽 가장자리를 a칸씩 잘라냄."""

    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, x):
        a = self.a
        return x[:, :, a:-a] if x.dim() == 3 else x[:, :, a:-a, a:-a]


class Pad(nn.Module):
    """양쪽 가장자리에 0을 a칸씩 채움."""

    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, x):
        return F.pad(x, (self.a, self.a) * (x.dim() - 2))


class Embedding(nn.Module):
    """값(정규화된 −3…3)을 bins개 구간 번호로 바꾸고, 번호마다 배우는 dim개 값으로 (Chronos식 시계열 토큰)."""

    def __init__(self, bins, dim):
        super().__init__()
        self.bins = bins
        self.weight = nn.Parameter(torch.randn(bins, dim))

    def forward(self, x):
        idx = torch.clamp(torch.floor((x[:, 0] + 3.0) / 6.0 * self.bins), 0, self.bins - 1).long()
        return self.weight[idx].movedim(-1, 1)


class Spectrogram(nn.Module):
    """신호 (1, L) → 시간·주파수 그림 (1, 주파수, 시간): Hann 창 n_fft칸을 hop칸씩 옮기며 FFT 크기²의 log(1+·).
    mel이면 mel 필터로 주파수를 묶고, mfcc면 그 위에 DCT. 배우는 값 없음."""

    def __init__(self, n_fft=32, hop=8, mode="log", n_mels=16, n_mfcc=13):
        super().__init__()
        self.hop, self.mode = hop, mode
        nf = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float64)
        win = 0.5 - 0.5 * torch.cos(2 * math.pi * n / n_fft)
        f = torch.arange(nf, dtype=torch.float64)[:, None]
        self.register_buffer("cos_k", (win * torch.cos(2 * math.pi * f * n / n_fft)).float().unsqueeze(1), persistent=False)
        self.register_buffer("sin_k", (win * torch.sin(2 * math.pi * f * n / n_fft)).float().unsqueeze(1), persistent=False)
        sr = 16000.0
        mel = lambda v: 2595.0 * math.log10(1.0 + v / 700.0)
        hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
        pts = [hz(mel(sr / 2) * i / (n_mels + 1)) for i in range(n_mels + 2)]
        freqs = [sr / 2 * j / (nf - 1) for j in range(nf)]
        fb = [[max(0.0, min((fr - pts[m]) / (pts[m + 1] - pts[m]) if pts[m + 1] > pts[m] else 0.0,
                            (pts[m + 2] - fr) / (pts[m + 2] - pts[m + 1]) if pts[m + 2] > pts[m + 1] else 0.0))
               for fr in freqs] for m in range(n_mels)]
        self.register_buffer("mel", torch.tensor(fb, dtype=torch.float32), persistent=False)
        dct = [[math.sqrt((1.0 if k == 0 else 2.0) / n_mels) * math.cos(math.pi / n_mels * (m + 0.5) * k) for m in range(n_mels)]
               for k in range(n_mfcc)]
        self.register_buffer("dct", torch.tensor(dct, dtype=torch.float32), persistent=False)

    def forward(self, x):
        re, im = F.conv1d(x, self.cos_k, stride=self.hop), F.conv1d(x, self.sin_k, stride=self.hop)
        p = re * re + im * im                                              # (N, 주파수, 시간)
        if self.mode == "log":
            return torch.log1p(p).unsqueeze(1)
        m = torch.log1p(torch.einsum("mf,nft->nmt", self.mel, p))
        return (m if self.mode == "mel" else torch.einsum("km,nmt->nkt", self.dct, m)).unsqueeze(1)


# ---------------------------------------------------------------- 증강 (학습 때만, 예측 때는 그대로 통과)

def _affine(x, theta):
    """그림 (N, C, H, W)에 샘플마다 2×3 변환 (출력 좌표 → 입력 좌표, −1…1)."""
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class RandomFlip(nn.Module):
    """학습 때 반씩 뒤집음: 그림은 좌우(h)·상하(v)·둘 다(hv), 줄은 시간을 거꾸로."""

    def __init__(self, dir="h"):
        super().__init__()
        self.dir = dir

    def forward(self, x):
        if not self.training:
            return x
        flip = lambda y, dim: torch.where((torch.rand(y.shape[0], *([1] * (y.dim() - 1)), device=y.device) < 0.5), y.flip(dim), y)
        if x.dim() == 3:
            return flip(x, 2)
        if "h" in self.dir:
            x = flip(x, 3)
        if "v" in self.dir:
            x = flip(x, 2)
        return x


class RandomRotation(nn.Module):
    """학습 때 그림을 ±degrees도 안에서 무작위로 돌림."""

    def __init__(self, degrees=15):
        super().__init__()
        self.degrees = degrees

    def forward(self, x):
        if not self.training:
            return x
        a = (torch.rand(x.shape[0], device=x.device) * 2 - 1) * math.radians(self.degrees)
        c, s, z = torch.cos(a), torch.sin(a), torch.zeros_like(a)
        return _affine(x, torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1)], 1))


class RandomShift(nn.Module):
    """학습 때 크기의 ±frac 안에서 무작위로 옮김 (빈 곳은 0)."""

    def __init__(self, frac=0.1):
        super().__init__()
        self.frac = frac

    def forward(self, x):
        if not self.training:
            return x
        n = x.shape[0]
        if x.dim() == 3:
            length = x.shape[2]
            m = int(round(length * self.frac))
            out = torch.zeros_like(x)
            for i, d in enumerate(torch.randint(-m, m + 1, (n,)).tolist()):
                if d >= 0:
                    out[i, :, d:] = x[i, :, :length - d]
                else:
                    out[i, :, :d] = x[i, :, -d:]
            return out
        t = (torch.rand(n, 2, device=x.device) * 2 - 1) * self.frac * 2
        one, z = torch.ones(n, device=x.device), torch.zeros(n, device=x.device)
        return _affine(x, torch.stack([torch.stack([one, z, t[:, 0]], -1), torch.stack([z, one, t[:, 1]], -1)], 1))


class RandomZoom(nn.Module):
    """학습 때 그림을 1±frac배로 무작위 확대·축소."""

    def __init__(self, frac=0.1):
        super().__init__()
        self.frac = frac

    def forward(self, x):
        if not self.training:
            return x
        n = x.shape[0]
        s = 1 + (torch.rand(n, device=x.device) * 2 - 1) * self.frac
        z = torch.zeros(n, device=x.device)
        return _affine(x, torch.stack([torch.stack([1 / s, z, z], -1), torch.stack([z, 1 / s, z], -1)], 1))


class RandomScale(nn.Module):
    """학습 때 샘플마다 값 전체에 1±frac배를 곱함 (신호 세기 바꾸기)."""

    def __init__(self, frac=0.2):
        super().__init__()
        self.frac = frac

    def forward(self, x):
        if not self.training:
            return x
        s = 1 + (torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) * 2 - 1) * self.frac
        return x * s


class RandomStretch(nn.Module):
    """학습 때 신호를 시간 방향으로 1±frac배 늘이거나 줄인 뒤 가운데를 원래 길이로 (빈 곳은 0)."""

    def __init__(self, frac=0.1):
        super().__init__()
        self.frac = frac

    def forward(self, x):
        if not self.training:
            return x
        n, c, length = x.shape
        out = torch.zeros_like(x)
        for i in range(n):
            new = max(2, int(round(length * (1 + (float(torch.rand(1)) * 2 - 1) * self.frac))))
            y = F.interpolate(x[i:i + 1], size=new, mode="linear", align_corners=False)[0]
            if new >= length:
                a = (new - length) // 2
                out[i] = y[:, a:a + length]
            else:
                a = (length - new) // 2
                out[i, :, a:a + new] = y
        return out


class RNNBlock(nn.Module):
    """순환 층 (LSTM·GRU·SimpleRNN): (N, C, L)을 시점마다 C개 값으로 읽음. return_sequences면 (H, L) 줄을 그대로,
    아니면 마지막 숨은 값 (양방향이면 앞→뒤의 끝과 뒤→앞의 끝을 이어 붙임)."""

    def __init__(self, kind, c, hidden, layers=1, bidirectional=False, return_sequences=False):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU, "simple_rnn": nn.RNN}[kind]
        self.rnn = cls(c, hidden, layers, batch_first=True, bidirectional=bidirectional)
        self.hidden, self.bi, self.seq = hidden, bidirectional, return_sequences

    def forward(self, x):
        out, _ = self.rnn(x.transpose(1, 2))
        if self.seq:
            return out.transpose(1, 2)
        if self.bi:
            return torch.cat([out[:, -1, :self.hidden], out[:, 0, self.hidden:]], dim=1)
        return out[:, -1]


class CausalConv1d(nn.Conv1d):
    """앞(과거)만 보는 Conv1D: 왼쪽에만 dilation×(kernel−1)칸 0을 채움 (WaveNet·TCN)."""

    def forward(self, x):
        pad = self.dilation[0] * (self.kernel_size[0] - 1)
        return F.conv1d(F.pad(x, (pad, 0)), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class MobileNetV3Small(nn.Module):
    """Google이 ImageNet으로 미리 학습한 MobileNetV3-Small 특징 추출기 (TF Hub mobilenet_v3_small_100_224/feature_vector,
    Apache 2.0, BatchNorm은 합성곱에 접혀 있음). 학습하지 않음(고정).
    그림 (N, 1 또는 3, H, W) → 그림마다 0~1로 → size×size (bilinear) → −1~1 → 특징 1024개.
    out="map": 그림을 2배(2H×2W)로 넣어 앞 블록 8개까지 → 특징 지도 (48, H/8, W/8) (탐지: 원래 그림 8픽셀마다 한 칸).
    train_last > 0: 끝 블록 train_last개(특징 1024개면 마지막 1×1 합성곱 둘도)를 함께 학습 (미세조정).
    가장자리는 TensorFlow 'SAME'처럼 채움 (stride 2면 아래·오른쪽을 더).
    CFG: (들어옴, kernel, 넓힘, 나감, SE 채널 (0이면 없음), HardSwish인지, stride)."""

    CFG = ((16, 3, 16, 16, 8, 0, 2), (16, 3, 72, 24, 0, 0, 2), (24, 3, 88, 24, 0, 0, 1), (24, 5, 96, 40, 24, 1, 2),
           (40, 5, 240, 40, 64, 1, 1), (40, 5, 240, 40, 64, 1, 1), (40, 5, 120, 48, 32, 1, 1), (48, 5, 144, 48, 40, 1, 1),
           (48, 5, 288, 96, 72, 1, 2), (96, 5, 576, 96, 144, 1, 1), (96, 5, 576, 96, 144, 1, 1))
    SE_SCALE = 0.16667                                   # 원래 모델의 hard sigmoid: relu6(x + 3) × 0.16667

    def __init__(self, size=128, train_last=0, out="vector"):
        super().__init__()
        self.size, self.map = size, out == "map"
        nb = 8 if self.map else len(self.CFG)
        self.stem = nn.Conv2d(3, 16, 3, 2)
        self.blocks = nn.ModuleList()
        for i, k, e, o, q, hs, s in self.CFG[:nb]:
            b = nn.Module()
            if e != i:
                b.expand = nn.Conv2d(i, e, 1)
            b.dw = nn.Conv2d(e, e, k, s, groups=e)
            if q:
                b.se1, b.se2 = nn.Conv2d(e, q, 1), nn.Conv2d(q, e, 1)
            b.project = nn.Conv2d(e, o, 1)
            self.blocks.append(b)
        if not self.map:
            self.last = nn.Conv2d(96, 576, 1)
            self.head = nn.Conv2d(576, 1024, 1)
        self.requires_grad_(False)
        if train_last:                                   # 미세조정: 끝 블록 train_last개 (특징 1024개면 마지막 1×1 합성곱 둘도)
            for m in list(self.blocks)[nb - train_last:] + ([] if self.map else [self.last, self.head]):
                m.requires_grad_(True)

    @staticmethod
    def same(x, k, s):
        """TensorFlow 'SAME' 채움: 모자란 칸을 위·왼쪽에 반, 아래·오른쪽에 나머지."""
        if k == 1:
            return x
        H, W = x.shape[2], x.shape[3]
        ph, pw = max((-(-H // s) - 1) * s + k - H, 0), max((-(-W // s) - 1) * s + k - W, 0)
        return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))

    def prep(self, x):
        n = x.shape[0]
        flat = x.reshape(n, -1)
        lo, hi = flat.min(1)[0].view(n, 1, 1, 1), flat.max(1)[0].view(n, 1, 1, 1)
        x = (x - lo) / (hi - lo).clamp_min(1e-6)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        hw = (2 * x.shape[2], 2 * x.shape[3]) if self.map else (self.size, self.size)
        if tuple(x.shape[2:]) != hw:
            x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
        return x * 2 - 1

    def features(self, x):
        x = F.hardswish(self.stem(self.same(x, 3, 2)))
        for (i, k, e, o, q, hs, s), b in zip(self.CFG, self.blocks):
            act = F.hardswish if hs else F.relu
            y = act(b.expand(x)) if e != i else x
            y = act(b.dw(self.same(y, k, s)))
            if q:
                y = y * (F.relu6(b.se2(F.relu(b.se1(y.mean((2, 3), keepdim=True)))) + 3) * self.SE_SCALE)
            y = b.project(y)
            x = x + y if s == 1 and i == o else y
        if self.map:
            return x
        x = F.hardswish(self.last(x)).mean((2, 3), keepdim=True)
        return F.hardswish(self.head(x)).flatten(1)

    def forward(self, x):
        return self.features(self.prep(x))


class NormalDistance(nn.Module):
    """이상 탐지 출력 '정상과의 거리': 정상 샘플의 특징 분포(평균·주성분 k개)에서 떨어진 정도를 점수 하나로 냄
    (Mahalanobis 거리를 주성분 k개로 줄인 것). 값은 학습 때 정상 샘플에서 한 번 계산해 넣음 (경사 학습 없음)."""

    def __init__(self, c, k=128):
        super().__init__()
        self.register_buffer("mean", torch.zeros(c))
        self.register_buffer("basis", torch.zeros(k, c))
        self.register_buffer("inv_var", torch.zeros(k))
        self.register_buffer("inv_rest", torch.zeros(1))

    def forward(self, f):
        d = f - self.mean
        p = d @ self.basis.T
        s = (p * p * self.inv_var).sum(1) + ((d * d).sum(1) - (p * p).sum(1)) * self.inv_rest
        return (s / d.shape[1]).unsqueeze(1)


RANDOM_MODULES = (DropPath, GaussianNoise, RandomFlip, RandomRotation, RandomShift, RandomZoom, RandomScale, RandomStretch)

"""생성 (새 샘플 만들기): 두 방식.

VAE ('VAE 잠재 공간' 블록): 데이터를 되살리도록 학습하면서 가운데 잠재 공간을 정규분포에 맞춤. 학습이 끝나면 정규분포에서 뽑은 점을
잠재 공간 뒤(디코더)에 넣어 새 샘플을 만든다. 손실 = 복원 MSE + β·KL ÷ 출력 수.

확산 (DDPM, '확산 시점' 블록): 데이터에 시점 t만큼 잡음을 섞고 (x_t = √ᾱ_t·x + √(1-ᾱ_t)·ε), 모델이 원래 값 x를 맞히도록 학습
(잡음 ε를 맞히는 방식보다 작은 모델·짧은 학습에서 샘플이 안정적). 새 샘플은 순수한 잡음에서 시작해 T번 잡음을 조금씩 걷어 냄.
손실 = 원래 값 MSE.

'클래스 조건' 블록이 있으면 학습 때는 샘플의 클래스를, 만들 때는 고른 클래스를 넣어 그 클래스를 닮은 샘플을 만든다 (조건부 생성).
성적은 두 방식 모두 시험 복원 오차 (확산은 시점 T/2만큼 잡음을 섞은 것에서 맞힌 원래 값과 실제의 차이).
설치형(torch)과 웹(engine.js)이 같은 규칙.
"""
from __future__ import annotations

import math

import numpy as np

MAX_SAMPLES = 64          # 한 번에 만드는 샘플 수 상한
SHOW_RECON = 6            # 결과 화면: 시험 샘플과 되살린 것
T_STEPS = 200             # 확산 시점 수
EVAL_T = T_STEPS // 2     # 성적을 재는 시점
EVAL_SEED = 12345         # 성적을 잴 때 섞는 잡음 (늘 같게)
JUDGE_K = 5               # 분류기 채점: 가장 가까운 학습 샘플 수 (kNN)
JUDGE_DIM = 1024          # 값이 이보다 많으면 무작위 투영으로 줄여서 거리를 잼
JUDGE_REF = 1000          # 기준(진짜 시험 데이터 정확도)에 쓰는 시험 샘플 수 상한


def schedule(T: int = T_STEPS):
    """cosine 잡음 일정 (Nichol & Dhariwal 2021): (β_t, α_t, ᾱ_t) 각 길이 T (float64)."""
    s = 0.008
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    beta = np.clip(1 - ab[1:] / ab[:-1], 1e-5, 0.999)
    alpha = 1 - beta
    return beta, alpha, np.cumprod(alpha)


def to_generation(d):
    """읽은 Dataset (분류·회귀 등) → 생성 Dataset (task "generation"). 분류였으면 클래스 번호를 남김 (클래스 조건 블록이 씀)."""
    from blocklearning.data import Dataset
    src = dict(d.source or {}, task="generation")
    cls = d.task == "classification" and len(d.classes) > 1
    ytr = np.asarray(d.y_train, np.int64) if cls else np.zeros(len(d.X_train), np.int64)
    yte = np.asarray(d.y_test, np.int64) if cls else np.zeros(len(d.X_test), np.int64)
    return Dataset(d.X_train, d.X_test, ytr, yte, "generation", [], d.length, d.mean, d.std, src, list(d.features), d.image, None,
                   {"labels": [str(c) for c in d.classes] if cls else []})


def labels_of(data) -> list:
    """생성 데이터의 클래스 이름 (클래스가 없으면 [])."""
    return list((data.extra or {}).get("labels") or [])


def _blocks(obj):
    """그래프 JSON 안의 모든 블록 (갈래·반복 안까지)."""
    if isinstance(obj, dict):
        if "type" in obj:
            yield obj
        for v in obj.values():
            yield from _blocks(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _blocks(v)


def graph_issue(data, graph: dict):
    """생성 블록을 이 데이터에 쓸 수 있는지: 문제가 있으면 (설명, 고칠 방법), 없으면 None."""
    bs = list(_blocks(graph.get("nodes", graph)))
    types = {b["type"] for b in bs}
    if data.task != "generation":                           # 다른 과제에서는 조건을 넣지 않아 그대로 지나감
        return None
    if {"time_embed", "vae_latent"} <= types:
        return ("한 모델에 '확산 시점'과 'VAE 잠재 공간'을 함께 둘 수 없음", "확산 모델이면 VAE 잠재 공간을, VAE면 확산 시점을 빼세요")
    n = len(labels_of(data))
    small = [b for b in bs if b["type"] == "class_cond" and int(b.get("classes", 10)) < n]
    if small:
        return (f"'클래스 조건' 블록의 클래스 수({small[0].get('classes', 10)})가 데이터의 클래스 수({n})보다 적음",
                f"클래스 조건 블록의 클래스 수를 {n} 이상으로 하세요")
    return None


def metric(out: np.ndarray, X: np.ndarray) -> float:
    """복원 오차: 정규화한 값의 평균 제곱 오차 (낮을수록 좋음)."""
    o = np.asarray(out, dtype=float).reshape(len(X), -1)
    return float(np.mean((o - np.asarray(X, dtype=float).reshape(len(X), -1)) ** 2))


def result(out: np.ndarray, out_tr: np.ndarray, data) -> dict:
    """결과 화면: 시험·학습 복원 오차, 시험 샘플 몇 개와 되살린 것 (원래 단위)."""
    k = min(SHOW_RECON, len(data.X_test))
    o = np.asarray(out, dtype=float).reshape(len(data.X_test), -1)
    return {"test_metric": metric(out, data.X_test), "train_metric": metric(out_tr, data.X_train), "metric_name": "recon",
            "recon": [{"input": [round(float(v), 5) for v in data.raw(data.X_test[i])],
                       "output": [round(float(v), 5) for v in data.raw(o[i])]} for i in range(k)]}


# ---------------------------------------------------------------- 확산: 학습 묶음·성적·만들기 (torch)

def noised(x, generator=None):
    """학습 묶음 x (모델 입력 모양) → (x_t, t, 맞힐 값): t는 0…T-1에서 고르게, ε ~ N(0, 1). 맞힐 값은 원래 값 x를
    (N, 값 수)로 편 것 (모델 출력과 같은 꼴)."""
    import torch
    _, _, ab = schedule()
    n = x.shape[0]
    t = torch.randint(0, T_STEPS, (n,), generator=generator, device="cpu").to(x.device)
    eps = torch.randn(x.shape, generator=generator, device="cpu").to(x.device, x.dtype)
    a = torch.as_tensor(ab, dtype=x.dtype, device=x.device)[t].view(n, *([1] * (x.dim() - 1)))
    return a.sqrt() * x + (1 - a).sqrt() * eps, t.to(x.dtype), x.reshape(n, -1)


def predict(net, X: np.ndarray, labels, dev=None, batch: int = 256) -> np.ndarray:
    """평가 출력 (N, 값 수): VAE는 되살린 것, 확산은 시점 T/2만큼 잡음을 섞은 것에서 맞힌 원래 값. labels: 클래스 조건에 넣을 클래스."""
    import torch
    kind = net.gen_kind()
    dev = dev or "cpu"
    net = net.to(dev).eval()
    _, _, ab = schedule()
    a = float(ab[EVAL_T])
    rng = np.random.default_rng(EVAL_SEED)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=dev)
            yb = None if labels is None else torch.as_tensor(np.asarray(labels[i:i + batch]), dtype=torch.long, device=dev)
            if kind == "diffusion":
                eps = torch.as_tensor(rng.standard_normal(xb.shape), dtype=torch.float32, device=dev)
                xt = math.sqrt(a) * xb + math.sqrt(1 - a) * eps
                net.set_cond(y=yb, t=torch.full((len(xb),), float(EVAL_T), device=dev))
                o = net(xt)
            else:
                net.set_cond(y=yb)
                o = net(xb)
            outs.append(o.reshape(len(xb), -1).double().cpu().numpy())
    net.set_cond()
    return np.concatenate(outs) if outs else np.zeros((0, 0))


def latent_dim(model) -> int:
    lat = model.latent()
    if lat is None:
        raise ValueError("모델에 VAE 잠재 공간 블록이 없음")
    return int(lat[1].mu.out_features)


def sample_labels(model, data, n: int, label=None):
    """만들 샘플마다 넣을 클래스 (클래스 조건 블록이 없으면 None)."""
    K = model.cond_classes()
    return labels_for(K, labels_of(data), n, label) if K else None


def labels_for(K: int, names: list, n: int, label=None) -> np.ndarray:
    """클래스 조건 블록(클래스 수 K)에 넣을 n개: label이 있으면 그 클래스, 없으면 클래스를 돌아가며 (웹도 이것을 씀)."""
    if label is not None and label != "":
        k = int(label)
        if not 0 <= k < max(1, len(names)):
            raise ValueError(f"클래스 번호 {k}는 없음")
        return np.full(n, k, np.int64)
    return np.arange(n, dtype=np.int64) % max(1, min(K, len(names) or 1))


def step(x, x0, t: int, lo, hi):
    """확산 한 걸음 (잡음 더하기 전): 모델이 맞힌 원래 값 x₀를 학습 데이터 범위(열마다 lo–hi)로 자르고,
    x₀와 x_t로 x_{t-1}의 평균을 냄 (DDPM 사후 평균)."""
    beta, alpha, ab = schedule()
    ab_prev = ab[t - 1] if t > 0 else 1.0
    x0 = x0.clamp(min=lo, max=hi) if hasattr(x0, "clamp") else np.clip(x0, lo, hi)
    return (math.sqrt(ab_prev) * beta[t] / (1 - ab[t])) * x0 + (math.sqrt(alpha[t]) * (1 - ab_prev) / (1 - ab[t])) * x


def sample(model, data, n: int, seed: int = 0, scale: float = 1.0, label=None) -> np.ndarray:
    """새 샘플 n개 (원래 단위, (n, 값 수)). scale: 뽑는 폭 (작을수록 평균에 가까운 무난한 샘플). label: 클래스 조건이 있으면 그 클래스."""
    import torch
    from blocklearning.blocks import to_model_input
    n = max(1, min(int(n), MAX_SAMPLES))
    y = sample_labels(model, data, n, label)
    yt = None if y is None else torch.as_tensor(y)
    model.eval()
    kind = model.gen_kind()
    if kind == "diffusion":
        shape = to_model_input(model.kind, np.zeros((1, data.X_train.shape[1]), np.float32), image=data.image,
                               channels=getattr(data.length, "channels", 1)).shape[1:]
        g = torch.Generator().manual_seed(int(seed))
        beta, alpha, ab = schedule()
        Xtr = np.asarray(data.X_train, np.float32)
        lo, hi = (torch.as_tensor(v).reshape((1,) + tuple(shape)) for v in (Xtr.min(0), Xtr.max(0)))
        x = torch.randn((n,) + tuple(shape), generator=g) * float(scale)
        with torch.no_grad():
            for t in range(T_STEPS - 1, -1, -1):
                model.set_cond(y=yt, t=torch.full((n,), float(t)))
                x = step(x, model(x).reshape(x.shape), t, lo, hi)
                if t > 0:
                    sigma = math.sqrt(beta[t] * (1 - ab[t - 1]) / (1 - ab[t]))
                    x = x + sigma * float(scale) * torch.randn(x.shape, generator=g)
        model.set_cond()
        out = x.reshape(n, -1).double().numpy()
    else:
        z = np.random.default_rng(seed).standard_normal((n, latent_dim(model))) * float(scale)
        with torch.no_grad():
            model.set_cond(y=yt)
            out = model.generate(torch.as_tensor(z, dtype=torch.float32)).double().numpy()
            model.set_cond()
    return data.raw(out.reshape(n, -1)), y, out.reshape(n, -1)


# ---------------------------------------------------------------- 분류기 채점 (조건부 생성이 고른 클래스를 따르는지)

def _judge_space(data):
    """채점 공간: 정규화한 학습 값 (값이 많으면 고정한 무작위 투영으로 JUDGE_DIM개로). (투영 행렬 또는 None, 학습 값)."""
    cache = getattr(data, "_judge", None)
    if cache is not None:
        return cache
    Xtr = np.asarray(data.X_train, np.float32).reshape(len(data.X_train), -1)
    P = None
    if Xtr.shape[1] > JUDGE_DIM:
        P = (np.random.default_rng(0).standard_normal((Xtr.shape[1], JUDGE_DIM)) / math.sqrt(JUDGE_DIM)).astype(np.float32)
        Xtr = Xtr @ P
    cache = {"P": P, "X": Xtr, "sq": (Xtr ** 2).sum(1), "ref": None}
    try:
        data._judge = cache
    except AttributeError:
        pass
    return cache


def _knn(sp, ytr: np.ndarray, Q: np.ndarray, K: int) -> np.ndarray:
    """정규화한 값 Q (N, 값 수) → 가장 가까운 학습 샘플 JUDGE_K개의 다수결 클래스."""
    Q = np.asarray(Q, np.float32).reshape(len(Q), -1)
    if sp["P"] is not None:
        Q = Q @ sp["P"]
    k = min(JUDGE_K, len(sp["X"]))
    out = []
    for i in range(0, len(Q), 256):
        q = Q[i:i + 256]
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ sp["X"].T + sp["sq"][None]
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        votes = ytr[idx]
        out.append(np.array([np.bincount(v.astype(np.intp), minlength=K).argmax() for v in votes], np.int64))   # 웹(32비트)은 intp가 int32
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def judge(data, Q: np.ndarray, y) -> dict | None:
    """만든 샘플(정규화한 값 Q)을 진짜 학습 데이터로 만든 분류기(kNN)로 채점: 고른 클래스(y)로 맞힌 비율과,
    같은 분류기가 진짜 시험 데이터를 맞힌 비율(기준). 클래스가 없거나 클래스를 고르지 않은 모델이면 None."""
    names = labels_of(data)
    if y is None or len(names) < 2:
        return None
    ytr = np.asarray(data.y_train, np.int64)
    sp = _judge_space(data)
    pred = _knn(sp, ytr, Q, len(names))
    if sp["ref"] is None:                                       # 기준: 진짜 시험 데이터 (한 번만 계산)
        n = min(JUDGE_REF, len(data.X_test))
        sp["ref"] = float(np.mean(_knn(sp, ytr, data.X_test[:n], len(names)) == np.asarray(data.y_test[:n], np.int64))) if n else None
    y = np.asarray(y, np.int64)
    return {"acc": float(np.mean(pred == y)), "hit": int(np.sum(pred == y)), "n": int(len(y)), "real_acc": sp["ref"],
            "pred": [names[int(p)] for p in pred]}


def view(data, X: np.ndarray, seed: int, y=None, score=None) -> dict:
    """화면에 보낼 새 샘플: 값들과 CSV (한 행 = 샘플 하나, 열 = 입력과 같은 순서, 클래스 조건이면 맨 앞에 클래스)."""
    import csv
    import io
    names = list(data.features) if data.features and len(data.features) == X.shape[1] else [f"x{j + 1}" for j in range(X.shape[1])]
    labels = labels_of(data)
    cls = None if y is None else [labels[int(k)] if int(k) < len(labels) else str(int(k)) for k in y]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow((["class"] if cls else []) + names)
    for i, row in enumerate(X):
        w.writerow(([cls[i]] if cls else []) + [f"{v:.6g}" for v in row])
    return {"samples": [[round(float(v), 5) for v in row] for row in X], "seed": int(seed), "csv": buf.getvalue(),
            "image": list(data.image) if data.image else None, "channels": getattr(data.length, "channels", 1),
            "features": names if data.features else [], "classes": cls, "judge": score}

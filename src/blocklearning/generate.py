"""생성 (새 샘플 만들기, VAE): 데이터를 되살리도록 학습하면서 가운데 'VAE 잠재 공간' 블록이 데이터를 정규분포 위의 점으로 모음.
학습이 끝나면 정규분포에서 뽑은 점을 잠재 공간 뒤(디코더)에 넣어 학습 데이터와 닮은 새 샘플을 만든다.

설치형(torch)과 웹(engine.js)이 같은 규칙: 맞힐 값 = 입력 (이상 탐지처럼), 손실 = 복원 MSE + β·KL ÷ 출력 수, 성적 = 시험 복원 오차.
"""
from __future__ import annotations

import numpy as np

MAX_SAMPLES = 64          # 한 번에 만드는 샘플 수 상한
SHOW_RECON = 6            # 결과 화면: 시험 샘플과 되살린 것


def to_generation(d):
    """읽은 Dataset (분류·회귀 등) → 생성 Dataset (task "generation"): 정답은 쓰지 않음."""
    from blocklearning.data import Dataset
    src = dict(d.source or {}, task="generation")
    return Dataset(d.X_train, d.X_test, np.zeros(len(d.X_train), np.int64), np.zeros(len(d.X_test), np.int64), "generation",
                   [], d.length, d.mean, d.std, src, list(d.features), d.image, None, {"labels": [str(c) for c in d.classes]})


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


def latent_dim(model) -> int:
    lat = model.latent()
    if lat is None:
        raise ValueError("모델에 VAE 잠재 공간 블록이 없음")
    return int(lat[1].mu.out_features)


def sample(model, data, n: int, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """정규분포에서 뽑은 잠재 값 n개 → 새 샘플 (원래 단위, (n, 값 수)). scale: 뽑는 폭 (작을수록 평균에 가까운 무난한 샘플)."""
    import torch
    n = max(1, min(int(n), MAX_SAMPLES))
    z = np.random.default_rng(seed).standard_normal((n, latent_dim(model))) * float(scale)
    model.eval()
    with torch.no_grad():
        out = model.generate(torch.as_tensor(z, dtype=torch.float32)).double().numpy()
    return data.raw(out.reshape(n, -1))


def view(data, X: np.ndarray, seed: int) -> dict:
    """화면에 보낼 새 샘플: 값들과 CSV (한 행 = 샘플 하나, 열 = 입력과 같은 순서)."""
    import csv
    import io
    names = list(data.features) if data.features and len(data.features) == X.shape[1] else [f"x{j + 1}" for j in range(X.shape[1])]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(names)
    for row in X:
        w.writerow([f"{v:.6g}" for v in row])
    return {"samples": [[round(float(v), 5) for v in row] for row in X], "seed": int(seed), "csv": buf.getvalue(),
            "image": list(data.image) if data.image else None, "channels": getattr(data.length, "channels", 1),
            "features": names if data.features else []}

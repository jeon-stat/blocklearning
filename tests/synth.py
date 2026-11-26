"""시험용 합성 데이터 (화면의 예시는 모두 실제 데이터 — blocklearning/data/). 빠르고 결과가 늘 같은 작은 데이터가 필요한 시험에서만."""
import numpy as np


def example_arrays(n: int = 1000, length: int = 140, seed: int = 0) -> tuple:
    """합성 심전도형 박동: 0 = 정상(뾰족한 R파와 T파), 1 = 이상(넓은 QRS 또는 ST 내려감 또는 뒤집힌 T파). 대략 반반."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)
    g = lambda c, w, a: a * np.exp(-((t - c) / w) ** 2)
    y = rng.permutation(np.arange(n) % 2)
    X = np.empty((n, length))
    for i in range(n):
        s = rng.normal(0, 0.015)                                  # 박동 위치 흔들림
        a = rng.normal(1.0, 0.08)
        x = g(0.2 + s, 0.035, 0.12 * a)                           # P파
        if y[i] == 0:
            x = x + g(0.4 + s, 0.012, a) - g(0.37 + s, 0.01, 0.15 * a) - g(0.43 + s, 0.012, 0.25 * a)
            x = x + g(0.65 + s, 0.05, 0.3 * a)                    # T파
        else:
            kind = rng.integers(3)
            if kind == 0:                                         # 넓은 QRS
                x = x + g(0.41 + s, 0.04, 0.7 * a) - g(0.48 + s, 0.03, 0.3 * a) + g(0.68 + s, 0.05, 0.25 * a)
            elif kind == 1:                                       # ST 내려감
                x = x + g(0.4 + s, 0.012, a) - g(0.43 + s, 0.012, 0.25 * a)
                x = x - 0.15 * a * ((t > 0.45 + s) & (t < 0.6 + s)) + g(0.68 + s, 0.05, 0.2 * a)
            else:                                                 # 뒤집힌 T파
                x = x + g(0.4 + s, 0.012, a) - g(0.43 + s, 0.012, 0.25 * a) - g(0.65 + s, 0.05, 0.3 * a)
        X[i] = x + rng.normal(0, 0.03, length)
    return X, y


def parts_det_arrays(n: int = 400, size: int = 64, seed: int = 0) -> tuple:
    """합성 부품 사진 (제조, 탐지): 결이 있는 판 위에 원형 부품(밝은 원·가운데 구멍)·사각 부품(어두운 사각형)이 1–3개.
    → (픽셀 0–1로 펼친 X (n, size²), 그림마다 상자 [[클래스 번호, x, y, w, h], …])."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    X, B = [], []
    for _ in range(n):
        img = 0.45 + rng.normal(0, 0.05, size)[None, :] * np.ones((size, 1)) + rng.normal(0, 0.04, (size, size))
        img += rng.uniform(-0.08, 0.08) * (xx / size - 0.5)
        boxes, taken = [], []
        for _ in range(int(rng.integers(1, 4))):
            for _try in range(20):
                k = int(rng.integers(0, 2))
                r = rng.uniform(5, 11) if k == 0 else rng.uniform(4.5, 9)        # 원은 반지름, 사각형은 반 변
                cx, cy = rng.uniform(r + 1, size - r - 1, 2)
                if all(np.hypot(cx - a, cy - b) > r + q + 2 for a, b, q in taken):
                    break
            else:
                continue
            taken.append((cx, cy, r))
            if k == 0:
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                img = np.where(d2 < r ** 2, 0.85 + rng.normal(0, 0.03, (size, size)), img)
                img = np.where(d2 < (0.35 * r) ** 2, 0.3, img)
            else:
                inside = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
                img = np.where(inside, 0.12 + rng.normal(0, 0.03, (size, size)), img)
            boxes.append([k, (cx - r) / size, (cy - r) / size, 2 * r / size, 2 * r / size])
        X.append(np.clip(img, 0, 1).reshape(-1))
        B.append([[b[0]] + [round(float(v), 5) for v in b[1:]] for b in boxes])
    return np.asarray(X), B

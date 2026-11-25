"""시계열 예측 (미래 값): 긴 신호 하나를 '지난 L개 → 다음 H개' 창으로 잘라 학습한다.

시험 창은 신호의 뒤쪽 구간에서만 만든다 (시간 순서 그대로 나눔 → 미래 값이 학습에 새지 않음). 시험 창의 입력은
나눈 지점 앞의 값을 써도 되지만, 맞힐 값(다음 H개)은 모두 나눈 지점 뒤에 있다.
정규화는 학습 구간의 평균·표준편차 하나 (시점마다가 아니라 신호 전체에 하나: 입력과 맞힐 값이 같은 단위).
정답(다음 H개)은 원래 단위 그대로 두고, 학습할 때 정규화했다가 출력층에 되돌림을 접어 넣는다 (회귀와 같음).
"""
from __future__ import annotations

import numpy as np

MIN_TRAIN = 10            # 학습 창 최소 개수
MAX_TRAIN = 4000          # 학습 창이 이보다 많으면 간격(stride)을 늘림 (브라우저 학습 시간)
N_SAMPLES = 6             # 결과 화면의 예시 창 수


def windows(v: np.ndarray, L: int, H: int, stride: int) -> tuple:
    """신호 → (X (창 수, L), Y (창 수, H)). 창 시작은 0, stride, 2·stride, …"""
    starts = np.arange(0, len(v) - L - H + 1, max(1, stride))
    if not len(starts):
        return np.zeros((0, L)), np.zeros((0, H))
    X = np.stack([v[s:s + L] for s in starts])
    Y = np.stack([v[s + L:s + L + H] for s in starts])
    return X, Y


def defaults(n: int) -> tuple:
    """점 n개인 신호의 기본 (입력 길이 L, 예측 길이 H, 간격): H는 L의 1/4, 학습 창은 MAX_TRAIN개 이하."""
    L = int(min(96, max(8, n // 10)))
    H = int(max(1, min(24, L // 4)))
    stride = int(max(1, np.ceil((n * 0.8 - L - H) / MAX_TRAIN)))
    return L, H, stride


def make_forecast_dataset(series, L: int, H: int, stride: int = 1, test_ratio: float = 0.2, normalize: bool = True,
                          source: dict | None = None, column: str = "값", columns: list | None = None):
    """신호 하나 → 예측 Dataset (task "forecast", X는 정규화한 입력 창, y는 원래 단위의 다음 H개)."""
    from dlstudio.data import DataError, Dataset, Issue
    v = np.asarray(series, dtype=np.float64).ravel()
    n = len(v)
    if L < 2 or H < 1:
        raise DataError([Issue(None, None, f"입력 길이 {L}·예측 길이 {H}이(가) 너무 짧음", "입력 길이 2 이상, 예측 길이 1 이상")])
    cut = int(round(n * (1 - test_ratio)))
    Xtr, Ytr = windows(v[:cut], L, H, stride)
    Xte, Yte = windows(v[max(0, cut - L):], L, H, stride)
    if len(Xtr) < MIN_TRAIN or len(Xte) < 1:
        need = (MIN_TRAIN - 1) * stride + L + H
        raise DataError([Issue(None, None, f"점 {n}개로는 '지난 {L}개 → 다음 {H}개' 창이 모자람 (학습 창 {len(Xtr)}개, 시험 창 {len(Xte)}개)",
                               f"입력·예측 길이나 간격을 줄이세요 (학습 구간에 점이 {need}개 이상 필요)")])
    mu, sd = (float(v[:cut].mean()), float(v[:cut].std()) or 1.0) if normalize else (0.0, 1.0)
    src = dict(source or {})
    src.update({"task": "forecast", "column": column, "L": int(L), "H": int(H), "stride": int(stride)})
    src.setdefault("normalize", normalize)
    src.setdefault("test_ratio", test_ratio)
    extra = {"horizon": int(H), "stride": int(stride), "column": column, "columns": list(columns or [column]),
             "n_points": int(n), "cut": int(cut)}
    return Dataset((Xtr - mu) / sd, (Xte - mu) / sd, Ytr, Yte, "forecast", [], int(L), np.full(L, mu), np.full(L, sd),
                   src, [], None, None, extra)


def power_series(days: int = 120, seed: int = 0) -> np.ndarray:
    """합성 공장 전력 사용량 (kW, 1시간마다): 근무 시간(8–18시)에 높고 점심에 조금 내려감, 주말은 낮음,
    천천히 느는 추세와 날씨 같은 느린 흔들림, 잡음."""
    rng = np.random.default_rng(seed)
    t = np.arange(24 * days)
    hour, dow = t % 24, (t // 24) % 7
    work = 1 / (1 + np.exp(-(hour - 7.5) * 2)) * 1 / (1 + np.exp((hour - 18.5) * 2))      # 근무 시간 (부드러운 계단)
    lunch = np.exp(-((hour - 12.5) ** 2) / 0.8)
    weekday = np.where(dow < 5, 1.0, 0.3)
    slow = np.cumsum(rng.normal(0, 4, len(t)))                 # 느린 흔들림 (날씨·생산량)
    k = min(24 * 14, len(t))
    slow = slow - np.convolve(slow, np.ones(k) / k, mode="same")
    ar = np.zeros(len(t))
    for i in range(1, len(t)):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 12)
    return 320 + 0.03 * t + weekday * (420 * work - 90 * lunch * work) + 0.5 * slow + ar


def metric(out: np.ndarray, y: np.ndarray) -> float:
    """다음 H개 전체를 펼쳐 R² (1이면 완벽, 0이면 늘 평균을 답하는 것과 같음)."""
    o = np.asarray(out, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((o - y) ** 2)) / ss if ss > 0 else float("nan")


def naive(d, X: np.ndarray) -> np.ndarray:
    """단순 예측 (이보다 나아야 모델이 쓸모 있음): 입력이 H개 이상이면 지난 H개를 그대로 되풀이 (하루 주기 데이터에서
    H = 24시간이면 '어제 같은 시각'), 아니면 마지막 값을 그대로 이어 씀."""
    H, raw = d.extra["horizon"], X * d.std + d.mean
    if raw.shape[1] >= H:
        return raw[:, -H:]
    return np.repeat(raw[:, -1:], H, axis=1)


def naive_name(d) -> str:
    H = d.extra["horizon"]
    return f"지난 {H}개를 그대로 되풀이" if d.length >= H else "마지막 값을 그대로"


def result(out: np.ndarray, d) -> dict:
    """결과 화면: MAE·RMSE(원래 단위)·단순 예측의 MAE·앞으로 몇 칸째인지에 따른 MAE·예시 창."""
    y = np.asarray(d.y_test, dtype=float)
    o = np.asarray(out, dtype=float).reshape(y.shape)
    err = np.abs(o - y)
    nv = np.abs(naive(d, d.X_test) - y)
    k = len(y)
    picks = sorted({int(round(i)) for i in np.linspace(0, k - 1, min(N_SAMPLES, k))})
    raw = d.raw(d.X_test)
    r5 = lambda a: [round(float(x), 5) for x in a]
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(((o - y) ** 2).mean())), "naive_mae": float(nv.mean()),
            "naive_name": naive_name(d),
            "mae_by_step": r5(err.mean(0)), "naive_by_step": r5(nv.mean(0)),
            "fc_samples": [{"index": i, "history": r5(raw[i]), "y": r5(y[i]), "pred": r5(o[i])} for i in picks]}


def load_forecast_csv(path: str, column=None, L=None, H=None, stride=None, normalize: bool = True,
                      test_ratio: float = 0.2, **_) -> "Dataset":
    """긴 시계열 CSV (한 행 = 한 시점, 열 = 측정 항목) → 예측 Dataset. 값 열을 안 고르면 마지막 숫자 열.
    입력·예측 길이·간격을 안 정하면 defaults(점 수). 빈 칸은 앞뒤 값으로 메움."""
    from pathlib import Path
    from dlstudio.data import DataError, Issue, _column_values, _numeric_columns, _read_table
    names, rows = _read_table(path)
    nums = _numeric_columns(names, rows)
    if not nums:
        raise DataError([Issue(None, None, "숫자 열이 없음", "한 행 = 한 시점, 열 = 측정값인 CSV를 고르세요 (시각 같은 글자 열은 있어도 됨)")])
    col = column if column in nums else nums[-1]
    v = _column_values(Path(path).name, names, rows, col)
    dL, dH, ds = defaults(len(v))
    L, H = int(L or dL), int(H or dH)
    stride = int(stride or max(1, int(np.ceil((len(v) * (1 - test_ratio) - L - H) / MAX_TRAIN))))
    return make_forecast_dataset(v, L, H, stride, test_ratio, normalize, source={"path": str(path)}, column=col,
                                 columns=nums)

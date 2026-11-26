"""시계열 예측 (미래 값): 긴 신호 하나를 '지난 L개 → 다음 H개' 창으로 잘라 학습한다.

시험 창은 신호의 뒤쪽 구간에서만 만든다 (시간 순서 그대로 나눔 → 미래 값이 학습에 새지 않음). 시험 창의 입력은
나눈 지점 앞의 값을 써도 되지만, 맞힐 값(다음 H개)은 모두 나눈 지점 뒤에 있다.
정규화는 학습 구간의 평균·표준편차 하나 (시점마다가 아니라 신호 전체에 하나: 입력과 맞힐 값이 같은 단위).
정답(다음 H개)은 원래 단위 그대로 두고, 학습할 때 정규화했다가 출력층에 되돌림을 접어 넣는다 (회귀와 같음).
시각을 알면 (예시 데이터·CSV의 시각 열) 하루 중 시각·요일을 sin·cos 채널로 함께 넣는다: 값 채널이 먼저,
X는 채널마다 L개를 이어 붙인 것 (다른 시계열 채널과 같은 모양). 공장 전력처럼 주말·밤에 크게 바뀌는 신호에서
지난 값만으로는 '다음이 무슨 요일인지'를 모르기 때문.
"""
from __future__ import annotations

from datetime import datetime

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


DAY = 86400


def calendar(times) -> tuple:
    """시각 (datetime64 배열) → (채널 (k, 점 수), 이름 k개). 하루 중 시각은 간격이 하루보다 짧을 때만, 요일은 일주일보다
    짧을 때만 (그보다 길면 늘 같은 값). 둘 다 아니면 (None, [])."""
    t = np.asarray(times, dtype="datetime64[s]")
    if len(t) < 2:
        return None, []
    step = float(np.median(np.diff(t).astype(np.int64)))
    days = t.astype("datetime64[D]")
    sec = (t - days).astype(np.int64).astype(float)
    dow = (days.astype(np.int64) + 3) % 7                    # 1970-01-01은 목요일 → 월요일 0
    rows, names = [], []
    if 0 < step < DAY:
        a = 2 * np.pi * sec / DAY
        rows += [np.sin(a), np.cos(a)]
        names += ["시각 sin", "시각 cos"]
    if 0 < step < 7 * DAY:
        a = 2 * np.pi * (dow + sec / DAY) / 7
        rows += [np.sin(a), np.cos(a)]
        names += ["요일 sin", "요일 cos"]
    return (np.stack(rows), names) if rows else (None, [])


TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d",
                "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
                "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y", "%Y%m%d%H%M", "%Y%m%d")


def parse_times(cells: list):
    """글자 칸들 → datetime64[s] 배열 (모두 같은 형식으로 읽히고 시간 순서대로 늘어날 때만), 아니면 None.
    일·월 순서가 모호하면 (01/02/2018) 시간 순서대로 늘어나는 쪽."""
    cells = [c.strip() for c in cells]
    if len(cells) < 2 or not all(cells):
        return None
    for fmt in TIME_FORMATS:
        try:
            datetime.strptime(cells[0], fmt)
            t = np.array([datetime.strptime(c, fmt) for c in cells], dtype="datetime64[s]")
        except ValueError:
            continue
        if np.all(np.diff(t).astype(np.int64) > 0):
            return t
    return None


def find_times(names: list, rows: list, nums: list):
    """CSV에서 시각 열 찾기: 숫자 열이 아니고 모든 행이 날짜·시각으로 읽히는 첫 열 → (열 이름, datetime64 배열) 또는 (None, None)."""
    for j, n in enumerate(names):
        if n in nums:
            continue
        t = parse_times([r[j] if j < len(r) else "" for _, r in rows])
        if t is not None:
            return n, t
    return None, None


def window_channels(v: np.ndarray, cal, L: int, H: int, stride: int) -> tuple:
    """값 신호와 (있으면) 시각 채널 → (X (창 수, (1 + k)·L), Y (창 수, H)). X는 값 L개 다음 채널마다 L개."""
    X, Y = windows(v, L, H, stride)
    if cal is not None:
        X = np.concatenate([X] + [windows(c, L, H, stride)[0] for c in cal], axis=1)
    return X, Y


def history(d, X: np.ndarray) -> np.ndarray:
    """창의 지난 값 (원래 단위): 시각 채널이 있어도 값 채널만."""
    return d.raw(np.asarray(X, dtype=np.float64))[..., :int(d.length)]


def defaults(n: int) -> tuple:
    """점 n개인 신호의 기본 (입력 길이 L, 예측 길이 H, 간격): H는 L의 1/4, 학습 창은 MAX_TRAIN개 이하."""
    L = int(min(96, max(8, n // 10)))
    H = int(max(1, min(24, L // 4)))
    stride = int(max(1, np.ceil((n * 0.8 - L - H) / MAX_TRAIN)))
    return L, H, stride


def make_forecast_dataset(series, L: int, H: int, stride: int = 1, test_ratio: float = 0.2, normalize: bool = True,
                          source: dict | None = None, column: str = "값", columns: list | None = None,
                          times=None, use_calendar: bool = True, time_column: str | None = None):
    """신호 하나 → 예측 Dataset (task "forecast", X는 정규화한 입력 창, y는 원래 단위의 다음 H개).
    times(시각, 신호와 같은 길이)가 있고 use_calendar면 시각·요일 채널을 함께 넣음."""
    from dlstudio.data import DataError, Dataset, Issue, seq_length
    v = np.asarray(series, dtype=np.float64).ravel()
    n = len(v)
    if L < 2 or H < 1:
        raise DataError([Issue(None, None, f"입력 길이 {L}·예측 길이 {H}이(가) 너무 짧음", "입력 길이 2 이상, 예측 길이 1 이상")])
    cal, cal_names = calendar(times) if times is not None and use_calendar else (None, [])
    cut = int(round(n * (1 - test_ratio)))
    lo = max(0, cut - L)
    Xtr, Ytr = window_channels(v[:cut], None if cal is None else cal[:, :cut], L, H, stride)
    Xte, Yte = window_channels(v[lo:], None if cal is None else cal[:, lo:], L, H, stride)
    if len(Xtr) < MIN_TRAIN or len(Xte) < 1:
        need = (MIN_TRAIN - 1) * stride + L + H
        raise DataError([Issue(None, None, f"점 {n}개로는 '지난 {L}개 → 다음 {H}개' 창이 모자람 (학습 창 {len(Xtr)}개, 시험 창 {len(Xte)}개)",
                               f"입력·예측 길이나 간격을 줄이세요 (학습 구간에 점이 {need}개 이상 필요)")])
    mu, sd = (float(v[:cut].mean()), float(v[:cut].std()) or 1.0) if normalize else (0.0, 1.0)
    src = dict(source or {})
    src.update({"task": "forecast", "column": column, "L": int(L), "H": int(H), "stride": int(stride),
                "calendar": bool(use_calendar)})
    src.setdefault("normalize", normalize)
    src.setdefault("test_ratio", test_ratio)
    if cal_names:
        src["channel_names"] = [column] + cal_names
    extra = {"horizon": int(H), "stride": int(stride), "column": column, "columns": list(columns or [column]),
             "n_points": int(n), "cut": int(cut), "has_times": times is not None, "calendar": cal_names,
             "time_column": time_column}
    k = len(cal_names)                                      # 시각 채널은 이미 −1–1: 정규화하지 않음
    mean = np.concatenate([np.full(L, mu), np.zeros(k * L)])
    std = np.concatenate([np.full(L, sd), np.ones(k * L)])
    return Dataset((Xtr - mean) / std, (Xte - mean) / std, Ytr, Yte, "forecast", [], seq_length(L, 1 + k), mean, std,
                   src, [], None, None, extra)


def metric(out: np.ndarray, y: np.ndarray) -> float:
    """다음 H개 전체를 펼쳐 R² (1이면 완벽, 0이면 늘 평균을 답하는 것과 같음)."""
    o = np.asarray(out, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((o - y) ** 2)) / ss if ss > 0 else float("nan")


def naive(d, X: np.ndarray) -> np.ndarray:
    """단순 예측 (이보다 나아야 모델이 쓸모 있음): 입력이 H개 이상이면 지난 H개를 그대로 되풀이 (하루 주기 데이터에서
    H = 24시간이면 '어제 같은 시각'), 아니면 마지막 값을 그대로 이어 씀."""
    H, raw = d.extra["horizon"], history(d, X)
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
    raw = history(d, d.X_test)
    r5 = lambda a: [round(float(x), 5) for x in a]
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(((o - y) ** 2).mean())), "naive_mae": float(nv.mean()),
            "naive_name": naive_name(d),
            "mae_by_step": r5(err.mean(0)), "naive_by_step": r5(nv.mean(0)),
            "fc_samples": [{"index": i, "history": r5(raw[i]), "y": r5(y[i]), "pred": r5(o[i])} for i in picks]}


def load_forecast_csv(path: str, column=None, L=None, H=None, stride=None, normalize: bool = True,
                      test_ratio: float = 0.2, calendar: bool = True, **_) -> "Dataset":
    """긴 시계열 CSV (한 행 = 한 시점, 열 = 측정 항목) → 예측 Dataset. 값 열을 안 고르면 마지막 숫자 열.
    입력·예측 길이·간격을 안 정하면 defaults(점 수). 빈 칸은 앞뒤 값으로 메움. 시각 열이 있으면 시각·요일 채널 (calendar)."""
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
    tcol, times = find_times(names, rows, nums)
    return make_forecast_dataset(v, L, H, stride, test_ratio, normalize, source={"path": str(path)}, column=col,
                                 columns=nums, times=times, use_calendar=calendar, time_column=tcol)

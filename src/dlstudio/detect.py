"""객체 탐지 (YOLO 첫 버전과 같은 격자 방식): 그림을 격자로 나누고, 칸마다 '여기에 사물의 가운데가 있나, 상자는 어디·얼마나,
무슨 클래스인가'를 한 번에 맞힌다.

- 모델 출력 한 줄 = (칸마다 값 수 A, 격자 G, 격자 G)를 펼친 것. A = 5 + 클래스 수:
  0 있음(logit) · 1·2 칸 안의 가운데 위치(sigmoid) · 3·4 너비·높이의 제곱근(sigmoid, 그림 크기 비율) · 5… 클래스(softmax)
- 정답 상자: [클래스 번호, x, y, w, h] (왼쪽 위 기준, 그림 크기에 대한 0–1 비율)
- 지표: mAP@0.5 (클래스마다 AP를 구해 평균, 예측과 정답 상자가 50% 이상 겹치면 맞음)

numpy만 쓴다 (브라우저 Pyodide에서도 돈다). torch 손실은 torch_loss, 같은 식을 웹 엔진(engine.js)도 쓴다.
"""
from __future__ import annotations

import numpy as np

DET_SIZES = (32, 48, 64, 96)    # 탐지에 쓰는 그림 크기 (정사각형)
DEFAULT_SIZE = 64
STRIDE = 8                      # 8픽셀 = 격자 한 칸 (2×2 Pool을 세 번)
LAMBDA_COORD = 5.0              # 손실: 상자 위치·크기 무게
LAMBDA_NOOBJ = 0.5              # 손실: 사물이 없는 칸의 '없음' 무게 (없는 칸이 훨씬 많음)
SCORE_MIN = 0.05                # mAP를 구할 때 쓰는 가장 낮은 점수
SHOW_SCORE = 0.5                # 화면에 보이는 상자의 기본 점수
NMS_IOU = 0.5                   # 같은 클래스 상자가 이만큼 겹치면 점수가 낮은 쪽을 버림
MATCH_IOU = 0.5                 # 정답과 이만큼 겹치면 맞음
MAX_DET = 50                    # 그림 하나에서 찾는 상자 수 상한


class DetOut(int):
    """탐지 과제의 모델 출력 크기 (격자² × 칸마다 값 수). int처럼 쓰이고, 격자·클래스 수를 함께 지님 →
    모델 검사·만들기·코드가 Output 블록을 '격자 칸마다 1×1 필터'로 만든다."""

    def __new__(cls, grid: int, n_cls: int):
        o = super().__new__(cls, grid * grid * (5 + n_cls))
        o.grid, o.n_cls, o.per_cell = int(grid), int(n_cls), 5 + int(n_cls)
        return o

    def __getnewargs__(self):
        return (self.grid, self.n_cls)

    def __reduce__(self):
        return (DetOut, (self.grid, self.n_cls))


def grid_of(size: int) -> int:
    return max(1, int(size) // STRIDE)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def encode(boxes: list, grid: int, n_cls: int) -> np.ndarray:
    """정답 상자들 → 목표값 한 줄 (모델 출력과 같은 자리). 가운데가 같은 칸에 둘이면 큰 상자만 (YOLO 첫 버전과 같음)."""
    A, G = 5 + n_cls, grid
    t = np.zeros((A, G, G))
    for k, x, y, w, h in sorted(boxes, key=lambda b: -b[3] * b[4]):
        cx, cy = x + w / 2, y + h / 2
        gx, gy = min(G - 1, max(0, int(cx * G))), min(G - 1, max(0, int(cy * G)))
        if t[0, gy, gx]:
            continue
        t[0, gy, gx] = 1
        t[1, gy, gx], t[2, gy, gx] = cx * G - gx, cy * G - gy
        t[3, gy, gx], t[4, gy, gx] = np.sqrt(max(w, 0.0)), np.sqrt(max(h, 0.0))
        t[5 + int(k), gy, gx] = 1
    return t.reshape(-1)


def iou(a, b) -> float:
    """[x, y, w, h] 두 상자의 겹친 넓이 / 합친 넓이."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def decode(row, grid: int, n_cls: int, score_min: float = SCORE_MIN) -> list:
    """모델 출력 한 줄 → 찾은 상자 [[클래스 번호, 점수, x, y, w, h], …] (점수 높은 순, 같은 클래스끼리 겹친 것은 뺌).
    점수 = '있음' 확률 × 그 클래스 확률."""
    A, G = 5 + n_cls, grid
    o = np.asarray(row, dtype=np.float64).reshape(A, G, G)
    obj = _sig(o[0])
    if n_cls > 1:
        e = np.exp(o[5:] - o[5:].max(0, keepdims=True))
        prob = e / e.sum(0, keepdims=True)
    else:
        prob = np.ones((1, G, G))
    k = prob.argmax(0)
    score = obj * prob.max(0)
    cand = []
    for gy, gx in zip(*np.nonzero(score >= score_min)):
        cx, cy = (gx + _sig(o[1, gy, gx])) / G, (gy + _sig(o[2, gy, gx])) / G
        w, h = _sig(o[3, gy, gx]) ** 2, _sig(o[4, gy, gx]) ** 2
        x0, y0, x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x1 > x0 and y1 > y0:
            cand.append([int(k[gy, gx]), float(score[gy, gx]), x0, y0, x1 - x0, y1 - y0])
    return nms(cand)


def nms(cand: list, thr: float = NMS_IOU, max_det: int = MAX_DET) -> list:
    """점수 높은 순으로 고르고, 이미 고른 같은 클래스 상자와 thr 넘게 겹치면 버림."""
    keep = []
    for c in sorted(cand, key=lambda b: (-b[1], b[0], b[2], b[3])):
        if all(c[0] != q[0] or iou(c[2:], q[2:]) <= thr for q in keep):
            keep.append(c)
            if len(keep) >= max_det:
                break
    return keep


def _match(preds: list, gts: list, k: int, thr: float):
    """그림 하나·클래스 k: 점수 높은 예측부터 아직 안 맞힌 정답 중 가장 많이 겹친 것과 짝 → [(점수, 맞음)]."""
    used = [False] * len(gts)
    out = []
    for p in sorted((p for p in preds if p[0] == k), key=lambda p: -p[1]):
        best, j = 0.0, -1
        for i, g in enumerate(gts):
            if g[0] == k and not used[i]:
                v = iou(p[2:], g[1:])
                if v > best:
                    best, j = v, i
        ok = bool(j >= 0 and best >= thr)
        if ok:
            used[j] = True
        out.append((p[1], ok))
    return out


def average_precision(preds: list, gts: list, n_cls: int, thr: float = MATCH_IOU) -> tuple:
    """그림마다 예측·정답 → (mAP, 클래스마다 AP (정답이 없는 클래스는 None)). AP는 정밀도-재현율 곡선 아래 넓이
    (재현율이 오를 때마다 그 뒤 가장 높은 정밀도를 씀, VOC 2010 이후·COCO와 같은 방식)."""
    aps = []
    for k in range(n_cls):
        n_pos = sum(1 for g in gts for b in g if b[0] == k)
        if not n_pos:
            aps.append(None)
            continue
        hits = sorted((h for p, g in zip(preds, gts) for h in _match(p, g, k, thr)), key=lambda h: -h[0])
        tp = np.cumsum([1 if ok else 0 for _, ok in hits]) if hits else np.zeros(0)
        fp = np.cumsum([0 if ok else 1 for _, ok in hits]) if hits else np.zeros(0)
        rec = np.concatenate([[0.0], tp / n_pos, [1.0]])
        prec = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1), [0.0]]) if hits else np.array([1.0, 0.0, 0.0])
        if not hits:
            rec = np.array([0.0, 0.0, 1.0])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
        aps.append(round(ap, 6))
    have = [a for a in aps if a is not None]
    return (float(np.mean(have)) if have else 0.0), aps


def precision_recall(preds: list, gts: list, n_cls: int, score: float = SHOW_SCORE, thr: float = MATCH_IOU) -> dict:
    """점수 score 이상만 볼 때: 맞게 찾음(tp)·잘못 찾음(fp)·놓침(fn), 정밀도·재현율."""
    tp = fp = 0
    for p, g in zip(preds, gts):
        shown = [q for q in p if q[1] >= score]
        for k in range(n_cls):
            for _, ok in _match(shown, g, k, thr):
                tp, fp = tp + ok, fp + (not ok)
    n_gt = sum(len(g) for g in gts)
    fn = n_gt - tp
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / n_gt if n_gt else None}


def decode_all(out, grid: int, n_cls: int, score_min: float = SCORE_MIN) -> list:
    out = np.asarray(out, dtype=np.float64).reshape(len(out), -1)
    return [decode(r, grid, n_cls, score_min) for r in out]


def metric(out, gts: list, grid: int, n_cls: int) -> float:
    """시험 지표 mAP@0.5 (모델 출력 전체, 그림마다 정답)."""
    return average_precision(decode_all(out, grid, n_cls), gts, n_cls)[0]


def rounded(preds: list, score_min: float = 0.25) -> list:
    """화면·저장용: 점수 score_min 이상만, 소수 넷째 자리."""
    return [[[int(p[0])] + [round(float(v), 4) for v in p[1:]] for p in ps if p[1] >= score_min] for ps in preds]


def result(out, gts: list, grid: int, n_cls: int) -> dict:
    """학습이 끝난 뒤 결과: mAP, 클래스마다 AP, 점수 0.5에서 정밀도·재현율, 시험 그림마다 찾은 상자."""
    preds = decode_all(out, grid, n_cls)
    m, aps = average_precision(preds, gts, n_cls)
    return {"test_metric": m, "ap": aps, "pr": precision_recall(preds, gts, n_cls), "det_pred": rounded(preds)}


def torch_loss(out, target, grid: int, per_cell: int):
    """YOLO 첫 버전식 손실 (묶음 평균): 있음(BCE) + 사물 칸의 위치·크기 제곱 오차 × 5 + 클래스 교차 엔트로피.
    없는 칸의 '없음'은 무게 0.5. 웹 엔진(engine.js detLoss)과 같은 식."""
    import torch
    import torch.nn.functional as F
    N = out.shape[0]
    o = out.reshape(N, per_cell, grid, grid)
    t = target.reshape(N, per_cell, grid, grid)
    pos = t[:, 0]
    bce = F.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    loss = (bce * (pos + LAMBDA_NOOBJ * (1 - pos))).sum()
    loss = loss + LAMBDA_COORD * (((torch.sigmoid(o[:, 1:5]) - t[:, 1:5]) ** 2).sum(1) * pos).sum()
    if per_cell > 6:
        loss = loss + (-(t[:, 5:] * F.log_softmax(o[:, 5:], 1)).sum(1) * pos).sum()
    return loss / N


# ---------------------------------------------------------------- 라벨링 도형 → 상자

def shape_box(s: list, W: float, H: float):
    """라벨링 도형 → [x, y, w, h] (0–1) 또는 None (점). 다각형·원은 감싸는 상자."""
    kind = s[0] if isinstance(s[0], str) else "box"
    if kind == "pt":
        return None
    if kind == "poly":
        xs, ys = [p[0] for p in s[1]], [p[1] for p in s[1]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    elif kind == "circle":
        cx, cy, r = s[1], s[2], s[3]
        ry = r * W / max(H, 1e-9)
        x0, y0, x1, y1 = cx - r, cy - ry, cx + r, cy + ry
    else:
        x0, y0, x1, y1 = s[0], s[1], s[0] + s[2], s[1] + s[3]
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
    return [x0, y0, x1 - x0, y1 - y0] if x1 - x0 > 1e-4 and y1 - y0 > 1e-4 else None

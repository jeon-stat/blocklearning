"""오래 걸리는 일(학습)을 스레드로 돌리고 진행 상황·멈추기를 관리한다. 학습 루프(fit)도 여기."""
from __future__ import annotations

import threading
import traceback
import uuid
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

MAX_PROGRESS = 100
JOIN_SECONDS = 60          # 같은 종류의 새 작업을 시작할 때 앞 작업이 멈추기를 기다리는 시간


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.state = "running"
        self.progress: list = []
        self.result = None
        self.error: Optional[str] = None
        self.cancelled = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def report(self, item: dict) -> None:
        with self._lock:
            self.progress.append(item)
            del self.progress[:-MAX_PROGRESS]

    def snapshot(self) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "state": self.state, "progress": list(self.progress),
                    "result": self.result, "error": self.error}


class JobManager:
    def __init__(self):
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._start_locks: dict = {}

    def _running(self, kind: str) -> list:
        with self._lock:
            return [j for j in self._jobs.values() if j.kind == kind and j.state == "running"]

    def cancel_kind(self, kind: str) -> None:
        for j in self._running(kind):
            j.cancelled = True

    def start(self, kind: str, fn: Callable[[Job], dict]) -> str:
        """작업 시작. 같은 종류가 돌고 있으면 그것을 취소하고 끝나기를 기다린 뒤 시작한다.
        같은 종류의 start는 한 번에 하나씩 (버튼을 빠르게 두 번 눌러도 두 작업이 함께 돌지 않음)."""
        with self._lock:
            start_lock = self._start_locks.setdefault(kind, threading.Lock())
        with start_lock:
            for j in self._running(kind):
                j.cancelled = True
                if j.thread is not None:
                    j.thread.join(JOIN_SECONDS)
            if any(j.thread is not None and j.thread.is_alive() for j in self._running(kind)):
                raise RuntimeError("앞 작업이 아직 멈추지 않음 (반복 하나가 길면 기다려야 함) → 잠시 뒤 다시 누르세요")
            return self._launch(kind, fn)

    def _launch(self, kind: str, fn: Callable[[Job], dict]) -> str:
        job = Job(kind)

        def run():
            try:
                result = fn(job)
                with job._lock:
                    job.state, job.result = ("cancelled", None) if job.cancelled else ("done", result)
            except Exception as e:                               # 화면에 보여 줄 이유
                with job._lock:
                    job.state = "cancelled" if job.cancelled else "failed"
                    job.error = f"{type(e).__name__}: {e}"
                    job.result = {"traceback": traceback.format_exc()[-2000:]}

        job.thread = threading.Thread(target=run, daemon=True)
        with self._lock:
            self._jobs[job.id] = job
        job.thread.start()
        return job.id

    def get(self, job_id: str) -> dict:
        return self._jobs[job_id].snapshot()

    def job(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> None:
        self._jobs[job_id].cancelled = True


@dataclass
class TrainSettings:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    device: str = "auto"
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TrainSettings":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def metric(out: np.ndarray, y: np.ndarray, task: str) -> float:
    if task == "classification":
        return float(np.mean(out.argmax(1) == y))
    o = out[:, 0]
    var = float(np.var(y))
    return 1.0 - float(np.mean((o - y) ** 2)) / var if var > 0 else float("nan")


SCATTER_ROWS = 200       # 회귀 '예측 vs 실제' 그림에 쓰는 점 수


def regression_view(out: np.ndarray, y: np.ndarray) -> tuple:
    """회귀 결과: 평균 절대 오차(MAE, 원래 단위)와 '예측 vs 실제' 점들."""
    o = np.asarray(out, dtype=float).reshape(len(y), -1)[:, 0]
    y = np.asarray(y, dtype=float)
    k = min(SCATTER_ROWS, len(y))
    return float(np.mean(np.abs(o - y))), {"y": [float(v) for v in y[:k]], "pred": [float(v) for v in o[:k]]}


def confusion(pred: np.ndarray, y: np.ndarray, k: int) -> tuple:
    """혼동 행렬 (행 = 정답, 열 = 예측)과 클래스별 재현율 (정답이 그 클래스인 것 중 맞힌 비율, 없으면 None)."""
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(np.asarray(y, dtype=int), np.asarray(pred, dtype=int)):
        cm[t, p] += 1
    rec = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(k)]
    return cm.tolist(), rec


def _device(name: str):
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else None
    return None if name in ("cpu", "", None) else name


def _fold_target_scale(net, mu: float, sd: float) -> None:
    """정규화한 정답으로 학습한 출력층에 되돌림(× sd + mu)을 접어 넣음: 모델이 원래 단위로 냄."""
    import torch
    from dlstudio.graph import output_linear
    lin = output_linear(net)
    with torch.no_grad():
        lin.weight.mul_(sd)
        lin.bias.mul_(sd).add_(mu)


def fit(net, X, y, task: str, epochs: int, lr: float, batch_size: int, seed: int = 0, device=None, callback=None,
        det=None):
    """평범한 torch 학습: Adam, 분류는 교차 엔트로피, 회귀는 MSE, 탐지는 YOLO식 손실 (det: DetOut). 반복마다
    callback({"epoch", "loss", "model"}), True를 돌려주면 멈춤. 돌려주는 모델은 CPU."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = torch.device(device) if device else torch.device("cpu")
    net = net.to(dev)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    if task == "detection":
        from dlstudio.detect import torch_loss
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev)
        loss_fn = lambda out, t: torch_loss(out, t, det.grid, det.per_cell)
    elif task == "classification":
        yt, loss_fn = torch.as_tensor(np.asarray(y), dtype=torch.long, device=dev), nn.CrossEntropyLoss()
    else:
        yt, loss_fn = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev).reshape(-1, 1), nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xt)
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for idx in torch.randperm(n, device=dev).split(max(1, batch_size)):
            if len(idx) == 1 and n > 1:                    # BatchNorm은 학습 때 샘플 하나짜리 묶음을 못 받음
                continue
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if not np.isfinite(np.mean(losses)):
            raise FloatingPointError("손실이 발산함 (NaN) → 학습률(Learning rate)을 낮추세요")
        if callback is not None and callback({"epoch": epoch, "loss": float(np.mean(losses)), "model": net}):
            break
    return net.cpu().eval()


def train_job(job: Job, graph: dict, data, settings: TrainSettings) -> dict:
    """그래프 → 모델 → 처음부터 학습 → 시험 지표 (분류: 정확도·혼동 행렬, 회귀: R²·MAE·산점도)."""
    import torch
    from dlstudio.blocks import to_model_input
    from dlstudio.graph import build_graph_model

    img = data.image
    model = build_graph_model(graph, data.length, data.n_out, seed=settings.seed, image=img)
    kind = model.kind
    Xtr, Xte = to_model_input(kind, data.X_train, image=img), to_model_input(kind, data.X_test, image=img)
    dev = _device(settings.device)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    done = []
    if data.task == "detection":
        return _train_det(job, model, Xtr, Xte, Xte_t, data, settings, dev)
    reg = data.task != "classification"
    mu = float(np.mean(data.y_train)) if reg else 0.0         # 회귀 정답은 정규화해 학습 (21–96 같은 값도 잘 배움)
    sd = (float(np.std(data.y_train)) or 1.0) if reg else 1.0
    y_fit = (data.y_train - mu) / sd if reg else data.y_train

    def cb(info):
        net = info["model"]
        net.eval()
        with torch.no_grad():
            out = net(Xte_t).detach().double().cpu().numpy() * sd + mu
        done.append(info["epoch"])
        job.report({"epoch": info["epoch"], "loss": float(info["loss"]), "test_metric": metric(out, data.y_test, data.task)})
        return job.cancelled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = fit(model, Xtr, y_fit, data.task, settings.epochs, settings.lr, settings.batch_size, seed=settings.seed,
                  device=dev, callback=cb)
    if reg:
        _fold_target_scale(net, mu, sd)
    with torch.no_grad():
        out = net(torch.as_tensor(Xte, dtype=torch.float32)).double().numpy()
    result = {"model": net, "epochs_done": len(done), "test_metric": metric(out, data.y_test, data.task),
              "metric_name": "accuracy" if data.task == "classification" else "r2", "warnings": [],
              "confusion": None, "recall": None}
    with torch.no_grad():                                   # 학습 데이터 성적 (시험과 차이가 크면 외운 것)
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
        out_tr = np.concatenate([net(Xtr_t[i:i + 1024]).double().numpy() for i in range(0, len(Xtr_t), 1024)])
    result["train_metric"] = metric(out_tr, data.y_train, data.task)
    if data.task == "classification":
        result["confusion"], result["recall"] = confusion(out.argmax(1), data.y_test, len(data.classes))
        result["pred"] = [int(v) for v in out.argmax(1)]          # 시험 샘플마다 예측 (틀린 샘플 모아 보기)
    else:
        result["mae"], result["scatter"] = regression_view(out, data.y_test)
    return result


def _train_det(job: Job, model, Xtr, Xte, Xte_t, data, settings: TrainSettings, dev) -> dict:
    """탐지 학습: 손실은 detect.torch_loss, 시험 지표는 mAP@0.5."""
    import torch
    from dlstudio import detect
    det, done = data.n_out, []
    G, K = det.grid, det.n_cls

    def cb(info):
        net = info["model"]
        net.eval()
        with torch.no_grad():
            out = net(Xte_t).detach().double().cpu().numpy()
        done.append(info["epoch"])
        job.report({"epoch": info["epoch"], "loss": float(info["loss"]),
                    "test_metric": detect.metric(out, data.det["test"], G, K)})
        return job.cancelled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = fit(model, Xtr, data.y_train, "detection", settings.epochs, settings.lr, settings.batch_size,
                  seed=settings.seed, device=dev, callback=cb, det=det)
    with torch.no_grad():
        out = net(torch.as_tensor(Xte, dtype=torch.float32)).double().numpy()
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
        out_tr = np.concatenate([net(Xtr_t[i:i + 512]).double().numpy() for i in range(0, len(Xtr_t), 512)])
    r = detect.result(out, data.det["test"], G, K)
    return dict(r, model=net, epochs_done=len(done), metric_name="map50", warnings=[], confusion=None, recall=None,
                train_metric=detect.metric(out_tr, data.det["train"], G, K))

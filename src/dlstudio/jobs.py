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
VAL_FRAC = 0.1             # 가장 좋은 epoch·조기 종료: 학습 몫에서 떼어 쓰는 검증 몫 (시험 몫은 고르는 데 쓰지 않음)
MIN_VAL_SPLIT = 20         # 학습 샘플이 이보다 적으면 검증 몫을 떼지 않음
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
        self.stopped = False            # 사용자가 [멈추기]: 거기까지 학습한 결과는 남김 (다른 이유의 취소는 버림)
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
                keep = not job.cancelled or (job.stopped and isinstance(result, dict))
                with job._lock:
                    job.state, job.result = ("done", result) if keep else ("cancelled", None)
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
        """사용자가 [멈추기]: 지금 반복이 끝나면 멈추고, 거기까지 학습한 결과를 남김."""
        j = self._jobs[job_id]
        j.stopped = j.cancelled = True


@dataclass
class TrainSettings:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    device: str = "auto"
    seed: int = 0
    keep_best: bool = False       # 검증 손실이 가장 낮았던 epoch의 모델을 남김
    early_stop: int = 0           # 검증 손실이 이 epoch 수 동안 나아지지 않으면 멈춤 (0이면 안 멈춤)
    loss: Optional[list] = None   # 손실 조립 [{"term", "w", ...}] (없으면 과제의 기본, losses.py)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TrainSettings":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def metric(out: np.ndarray, y: np.ndarray, task: str, X: Optional[np.ndarray] = None, heads=None) -> float:
    """시험 성적: 분류 정확도, 회귀 R², 예측 R²(다음 H개 전체), 이상 탐지 AUC (X = 모델 입력을 펼친 정규화 값),
    정답 여러 개는 정답마다 성적의 평균 (heads: 머리 목록)."""
    if task == "multi":
        from dlstudio.multi import metric as multi_metric
        return multi_metric(out, y, heads)
    if task == "classification":
        return float(np.mean(out.argmax(1) == y))
    if task == "forecast":
        from dlstudio.forecast import metric as fc_metric
        return fc_metric(out, y)
    if task == "anomaly":
        from dlstudio.anomaly import metric as ae_metric
        return ae_metric(out, X, y)
    if task == "segmentation":
        from dlstudio.segment import metric as seg_metric
        return seg_metric(out, y, np.asarray(out).reshape(len(y), -1).shape[1] // np.asarray(y).shape[1])
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


def _fold_target_scale(net, mu, sd) -> None:
    """정규화한 정답으로 학습한 출력층에 되돌림(× sd + mu, 출력마다)을 접어 넣음: 모델이 원래 단위로 냄."""
    import torch
    from dlstudio.graph import output_linear
    lin = output_linear(net)
    mu_t = torch.as_tensor(np.asarray(mu, dtype=np.float64).reshape(-1), dtype=lin.weight.dtype)
    sd_t = torch.as_tensor(np.asarray(sd, dtype=np.float64).reshape(-1), dtype=lin.weight.dtype)
    with torch.no_grad():
        lin.weight.mul_(sd_t[:, None])
        lin.bias.mul_(sd_t).add_(mu_t)


def target_scale(Y: np.ndarray) -> tuple:
    """맞힐 값 (샘플, 출력 수)의 출력마다 평균·표준편차 (표준편차 0이면 1)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    sd = Y.std(0)
    return Y.mean(0), np.where(sd > 0, sd, 1.0)


def fit(net, X, y, task: str, epochs: int, lr: float, batch_size: int, seed: int = 0, device=None, callback=None,
        det=None, keep_best: bool = False, early_stop: int = 0, val_score=None, loss_spec=None, n_cls: int = 0,
        class_w=None):
    """평범한 torch 학습: Adam, 분류는 교차 엔트로피, 회귀는 MSE, 탐지는 YOLO식 손실 (det: DetOut). 반복마다
    callback({"epoch", "loss", "val_loss", "best_epoch", "model"}), True를 돌려주면 멈춤. 돌려주는 모델은 CPU.
    keep_best·early_stop: 학습 몫의 VAL_FRAC를 검증 몫으로 떼어 epoch마다 검증 손실을 재고, 가장 낮았던 epoch의 모델을
    남기거나(keep_best) early_stop epoch 동안 나아지지 않으면 멈춤. 결과는 net.fit_info {best_epoch, early_stopped, n_val}.
    val_score(검증 출력, 검증 샘플 번호) → 클수록 좋은 성적: 주면 손실 대신 이것으로 고름 (탐지: 손실이 성적을 잘 대변하지 못함).
    loss_spec: 조립한 손실 (losses.normalize 결과, 기본이면 None) → 반복마다 항마다의 값도 info["terms"]."""
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
    elif task == "segmentation":                           # 시점마다 교차 엔트로피 (det = SegOut: 상태 수)
        from dlstudio.segment import torch_loss as seg_loss
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=dev)
        loss_fn = lambda out, t: seg_loss(out, t, det.n_cls)
    elif task == "multi":                                  # 정답 여러 개 (det = MultiOut: 머리 목록)
        from dlstudio.multi import torch_loss as multi_loss
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev).reshape(len(Xt), -1)
        loss_fn = lambda out, t: multi_loss(out, t, det.heads)
    elif task == "classification":
        yt, loss_fn = torch.as_tensor(np.asarray(y), dtype=torch.long, device=dev), nn.CrossEntropyLoss()
    else:
        yt, loss_fn = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev).reshape(len(Xt), -1), nn.MSELoss()
    parts_fn = None
    if loss_spec is not None:                                # 조립한 손실: 전체 = Σ 가중치 × 항
        from dlstudio import losses
        cw = (None if class_w is None else {k: torch.as_tensor(v, device=dev) for k, v in class_w.items()}
              if isinstance(class_w, dict) else torch.as_tensor(class_w, device=dev))
        parts_fn = losses.build(loss_spec, task, n_cls=n_cls, det=det, seg_K=getattr(det, "n_cls", 0), cw=cw,
                                heads=getattr(det, "heads", None))
        loss_fn = lambda out, t: parts_fn(out, t, net)[0]
        if any(t["term"] == "act_l1" for t in loss_spec):
            net.capture = {}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xv = yv = None
    if (keep_best or early_stop) and len(Xt) >= MIN_VAL_SPLIT:
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed + 1)).to(dev)
        nv = max(2, int(round(len(Xt) * VAL_FRAC)))
        Xv, yv, Xt, yt = Xt[perm[:nv]], yt[perm[:nv]], Xt[perm[nv:]], yt[perm[nv:]]
        val_idx = perm[:nv].cpu().numpy()
    n = len(Xt)
    best, best_ep, since, best_state, early = float("inf"), None, 0, None, False
    for epoch in range(1, epochs + 1):
        net.train()
        losses, terms = [], {}
        for idx in torch.randperm(n, device=dev).split(max(1, batch_size)):
            if len(idx) == 1 and n > 1:                    # BatchNorm은 학습 때 샘플 하나짜리 묶음을 못 받음
                continue
            opt.zero_grad()
            if parts_fn is not None:
                loss, parts = parts_fn(net(Xt[idx]), yt[idx], net)
                for k, v in parts.items():
                    terms[k] = terms.get(k, 0.0) + float(v.detach())
            else:
                loss = loss_fn(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if not np.isfinite(np.mean(losses)):
            raise FloatingPointError("손실이 발산함 (NaN) → 학습률(Learning rate)을 낮추세요")
        val_loss = None
        if Xv is not None:                                   # 검증 손실 (평가 모드)
            net.eval()
            with torch.no_grad():                            # 묶음마다 출력 → 바로 손실 (블록 출력 L1이 그 묶음의 값을 씀)
                outs, tot = [], 0.0
                for i in range(0, len(Xv), 256):
                    o = net(Xv[i:i + 256])
                    outs.append(o)
                    tot += float(loss_fn(o, yv[i:i + 256])) * len(o)
            val_loss = tot / len(Xv)
            key = val_loss if val_score is None else -val_score(torch.cat(outs).double().cpu().numpy(), val_idx)
            if key < best:
                best, best_ep, since = key, epoch, 0
                if keep_best:
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
        info = {"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": val_loss, "best_epoch": best_ep, "model": net}
        if terms:
            info["terms"] = {k: v / max(1, len(losses)) for k, v in terms.items()}
        if callback is not None and callback(info):
            break
        if early_stop and Xv is not None and since >= early_stop:
            early = True
            break
    if getattr(net, "capture", None) is not None:
        net.capture = None
    if best_state is not None:
        net.load_state_dict(best_state)
    net.fit_info = {"best_epoch": best_ep, "early_stopped": early, "n_val": 0 if Xv is None else len(Xv)}
    return net.cpu().eval()


def _progress(info: dict, test_metric) -> dict:
    """진행 한 줄: epoch, 학습 손실, (있으면) 검증 손실·가장 좋은 epoch, 시험 성적."""
    p = {"epoch": info["epoch"], "loss": float(info["loss"]), "test_metric": test_metric}
    if info.get("val_loss") is not None:
        p["val_loss"], p["best_epoch"] = float(info["val_loss"]), info.get("best_epoch")
    if info.get("terms"):
        p["terms"] = {k: float(v) for k, v in info["terms"].items()}
    return p


def loss_args(data, settings) -> dict:
    """학습 설정의 손실 → fit에 넘길 것 (기본이면 비워 둠: 지금까지와 똑같은 손실)."""
    from dlstudio import losses
    heads = data.n_out.heads if data.task == "multi" else None
    spec = losses.normalize(settings.loss, data.task, heads)
    if losses.is_default(spec, data.task, heads):
        return {}
    if heads:
        return {"loss_spec": spec, "n_cls": 0, "class_w": losses.multi_class_weights(data.y_train, heads, spec) or None}
    K = len(data.classes) if data.task in ("classification", "segmentation") else 0
    cw = losses.class_weights(data.y_train, K) if K and any(t.get("balance") for t in spec) else None
    return {"loss_spec": spec, "n_cls": K, "class_w": cw}


def _fit_result(net) -> dict:
    fi = getattr(net, "fit_info", None) or {}
    return {"best_epoch": fi.get("best_epoch"), "early_stopped": bool(fi.get("early_stopped")), "n_val": fi.get("n_val", 0)}


def train_job(job: Job, graph: dict, data, settings: TrainSettings) -> dict:
    """그래프 → 모델 → 처음부터 학습 → 시험 지표 (분류: 정확도·혼동 행렬, 회귀: R²·MAE·산점도)."""
    import torch
    from dlstudio.blocks import to_model_input
    from dlstudio.graph import build_graph_model

    img = data.image
    model = build_graph_model(graph, data.length, data.n_out, seed=settings.seed, image=img)
    kind = model.kind
    C = getattr(data.length, "channels", 1)
    Xtr, Xte = to_model_input(kind, data.X_train, image=img, channels=C), to_model_input(kind, data.X_test, image=img, channels=C)
    dev = _device(settings.device)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    done = []
    if data.task == "detection":
        return _train_det(job, model, Xtr, Xte, Xte_t, data, settings, dev)
    if model.distance_head() is not None:
        return _fit_distance(job, model, Xtr, Xte, data, dev)
    Xfit, head = Xtr, model.frozen_head()
    if head is not None:                                 # 고정한 사전학습 블록: 특징을 한 번만 뽑고 뒷부분만 학습
        Xfit, Fte = (frozen_features(model.block_modules[head], X, dev) for X in (Xtr, Xte))
        Xte_t = torch.as_tensor(Fte, dtype=torch.float32, device=dev)
        model.start = head
    reg = data.task not in ("classification", "segmentation", "multi")   # 회귀·예측·이상 탐지: 맞힐 값을 출력마다 정규화해 학습
    Y = np.asarray(data.target_train, dtype=np.float64).reshape(len(data.X_train), -1)
    mu, sd = target_scale(Y) if reg else (0.0, 1.0)
    y_fit = (Y - mu) / sd if reg else data.y_train
    heads = data.n_out.heads if data.task == "multi" else None
    if heads:                                              # 정답 여러 개: 값 정답만 정규화 (출력 칸마다 되돌림)
        from dlstudio import multi
        mu, sd, y_fit = multi.target_scale(data.y_train, heads)

    def cb(info):
        net = info["model"]
        net.eval()
        with torch.no_grad():
            out = net(Xte_t).detach().double().cpu().numpy() * sd + mu
        done.append(info["epoch"])
        job.report(_progress(info, metric(out, data.y_test, data.task, data.X_test, heads)))
        return job.cancelled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            net = fit(model, Xfit, y_fit, data.task, settings.epochs, settings.lr, settings.batch_size, seed=settings.seed,
                      device=dev, callback=cb, det=data.n_out if data.task in ("segmentation", "multi") else None,
                      keep_best=settings.keep_best, early_stop=settings.early_stop, **loss_args(data, settings))
        finally:
            model.start = None
    if reg or heads:
        _fold_target_scale(net, mu, sd)
    with torch.no_grad():
        out = net(torch.as_tensor(Xte, dtype=torch.float32)).double().numpy()
    result = {"model": net, "epochs_done": len(done), "test_metric": metric(out, data.y_test, data.task, data.X_test, heads),
              "metric_name": metric_name(data), "warnings": [], "confusion": None, "recall": None, **_fit_result(net)}
    with torch.no_grad():                                   # 학습 데이터 성적 (시험과 차이가 크면 외운 것)
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
        out_tr = np.concatenate([net(Xtr_t[i:i + 1024]).double().numpy() for i in range(0, len(Xtr_t), 1024)])
    if data.task in ("forecast", "anomaly", "segmentation", "multi"):
        return dict(result, **task_result(data, out, out_tr))
    result["train_metric"] = metric(out_tr, data.y_train, data.task)
    if data.task == "classification":
        result["confusion"], result["recall"] = confusion(out.argmax(1), data.y_test, len(data.classes))
        result["pred"] = [int(v) for v in out.argmax(1)]          # 시험 샘플마다 예측 (틀린 샘플 모아 보기)
    else:
        result["mae"], result["scatter"] = regression_view(out, data.y_test)
    return result


def _fit_distance(job: Job, model, Xtr, Xte, data, dev) -> dict:
    """이상 탐지 출력 '정상과의 거리': 경사 학습 없이, 학습 몫(정상)의 특징 분포를 한 번 계산해 넣음."""
    import torch
    from dlstudio.anomaly import fit_distance
    head, src = model.distance_head()
    model = model.to(dev or "cpu").eval()
    got = []
    hook = head.register_forward_pre_hook(lambda m, a: got.append(a[0].detach()))

    def feats(X):
        got.clear()
        with torch.no_grad():
            for i in range(0, len(X), 256):
                model(torch.as_tensor(X[i:i + 256], dtype=torch.float32, device=dev or "cpu"))
        return torch.cat(got)
    try:
        Ftr, Fte = feats(Xtr), feats(Xte)
    finally:
        hook.remove()
    p = fit_distance(Ftr.cpu().numpy(), head.basis.shape[0])
    with torch.no_grad():
        for k, v in p.items():
            getattr(head, k).copy_(torch.as_tensor(v))
        out, out_tr = (head(F).double().cpu().numpy() for F in (Fte, Ftr))
    model = model.cpu().eval()
    m = metric(out, data.y_test, data.task, data.X_test)
    job.report({"epoch": 1, "loss": 0.0, "test_metric": m})
    result = {"model": model, "epochs_done": 1, "test_metric": m, "metric_name": metric_name(data), "warnings": [],
              "confusion": None, "recall": None}
    return dict(result, **task_result(data, out, out_tr))


def frozen_features(module, X, dev, batch: int = 256) -> np.ndarray:
    """고정 블록의 출력 (학습 전에 한 번). X: 모델 입력 (numpy)."""
    import torch
    module = module.to(dev or "cpu").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(module(torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=dev or "cpu")).cpu().numpy())
    return np.concatenate(out)


def metric_name(data) -> str:
    if data.task == "anomaly":
        from dlstudio.anomaly import metric_name as ae_name
        return ae_name(data)
    if data.task == "multi":
        return "mean"
    return "accuracy" if data.task in ("classification", "segmentation") else "r2"


def task_result(data, out: np.ndarray, out_tr: np.ndarray) -> dict:
    """예측·이상 탐지·구간 분할의 결과 화면 값 (시험·학습 출력에서). 웹 학습도 출력만 넘겨 이것을 씀 (web.task_result)."""
    if data.task == "segmentation":
        from dlstudio import segment
        return segment.result(out, out_tr, data)
    if data.task == "multi":
        from dlstudio import multi
        return multi.result(out, out_tr, data)
    if data.task == "forecast":
        from dlstudio import forecast
        return dict(forecast.result(out, data), train_metric=forecast.metric(out_tr, data.y_train))
    from dlstudio import anomaly
    r = anomaly.result(out, out_tr, data)
    tr = anomaly.scores(out_tr, data.X_train)
    return dict(r, train_metric=None if metric_name(data) == "auc" else float(tr.mean()))


def _train_det(job: Job, model, Xtr, Xte, Xte_t, data, settings: TrainSettings, dev) -> dict:
    """탐지 학습: 손실은 detect.torch_loss, 시험 지표는 mAP@0.5."""
    import torch
    from dlstudio import detect
    det, done = data.n_out, []
    G, K = det.grid, det.n_cls
    Xfit, head = Xtr, model.frozen_head()
    if head is not None:                                 # 고정한 사전학습 블록: 특징 지도를 한 번만 뽑고 뒷부분만 학습
        Xfit, Fte = (frozen_features(model.block_modules[head], X, dev) for X in (Xtr, Xte))
        Xte_t = torch.as_tensor(Fte, dtype=torch.float32, device=dev)
        model.start = head

    def cb(info):
        net = info["model"]
        net.eval()
        with torch.no_grad():
            out = net(Xte_t).detach().double().cpu().numpy()
        done.append(info["epoch"])
        job.report(_progress(info, detect.metric(out, data.det["test"], G, K)))
        return job.cancelled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            net = fit(model, Xfit, data.y_train, "detection", settings.epochs, settings.lr, settings.batch_size,
                      seed=settings.seed, device=dev, callback=cb, det=det, keep_best=settings.keep_best,
                      early_stop=settings.early_stop, **loss_args(data, settings),
                      val_score=lambda o, idx: detect.metric(o, [data.det["train"][i] for i in idx], G, K))
        finally:
            model.start = None
    with torch.no_grad():
        out = net(torch.as_tensor(Xte, dtype=torch.float32)).double().numpy()
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
        out_tr = np.concatenate([net(Xtr_t[i:i + 512]).double().numpy() for i in range(0, len(Xtr_t), 512)])
    r = detect.result(out, data.det["test"], G, K)
    return dict(r, model=net, epochs_done=len(done), metric_name="map50", warnings=[], confusion=None, recall=None, **_fit_result(net),
                train_metric=detect.metric(out_tr, data.det["train"], G, K))

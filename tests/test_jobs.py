"""작업 관리자와 학습 작업: 진행 보고, 멈추기, 새 학습이 앞 학습을 취소, 실패 기록."""
import time

import numpy as np
import pytest

from dlstudio.graph import build_graph_model, from_blocks
from dlstudio.data import make_dataset
from tests.synth import example_arrays
from dlstudio.jobs import JobManager, TrainSettings, train_job

CNN = [{"type": "conv", "channels": 8, "kernel": 5}, {"type": "act", "kind": "relu"}, {"type": "global_avg"},
       {"type": "output"}]


def _data():
    return make_dataset(*example_arrays(n=300, length=140))          # 짧게 줄이면 좁은 R파가 빠져 배우기 어려움


def _wait(jm, jid, timeout=120):
    t = time.time()
    while jm.get(jid)["state"] == "running":
        if time.time() - t > timeout:
            raise AssertionError("작업이 끝나지 않음")
        time.sleep(0.05)
    return jm.get(jid)


def test_train_job_reports_and_finishes():
    jm, d = JobManager(), _data()
    jid = jm.start("train", lambda job: train_job(job, from_blocks(CNN), d, TrainSettings(epochs=5, lr=1e-2)))
    st = _wait(jm, jid)
    assert st["state"] == "done", st["error"]
    assert [p["epoch"] for p in st["progress"]] == [1, 2, 3, 4, 5]
    r = st["result"]
    assert r["epochs_done"] == 5 and r["metric_name"] == "accuracy" and r["test_metric"] > 0.6
    assert 0.5 < r["train_metric"] <= 1
    assert all(p.device.type == "cpu" for p in r["model"].parameters())
    assert set(build_graph_model(from_blocks(CNN), 140, 2).block_modules) <= set(r["model"].block_modules)   # 학습 뒤에도 블록 대응표


def test_regression_job():
    import numpy as np
    X, _ = example_arrays(n=200, length=40)
    d = make_dataset(X, X[:, 10] * 2.0 + 0.1, task="regression")
    jm = JobManager()
    jid = jm.start("train", lambda job: train_job(job, from_blocks(CNN), d, TrainSettings(epochs=2)))
    st = _wait(jm, jid)
    assert st["state"] == "done", st["error"]
    assert st["result"]["metric_name"] == "r2" and np.isfinite(st["result"]["test_metric"])


def test_stop_keeps_training_so_far():
    """[멈추기]: 지금 반복이 끝나면 멈추고, 거기까지 학습한 모델·결과를 남김."""
    jm, d = JobManager(), _data()
    jid = jm.start("train", lambda job: train_job(job, from_blocks(CNN), d, TrainSettings(epochs=50)))
    while not jm.get(jid)["progress"]:
        time.sleep(0.02)
    jm.cancel(jid)
    st = _wait(jm, jid)
    assert st["state"] == "done" and st["result"]["model"] is not None
    assert 1 <= st["result"]["epochs_done"] == len(st["progress"]) < 50 and 0 <= st["result"]["test_metric"] <= 1


def test_new_train_cancels_previous():
    jm, d = JobManager(), _data()
    a = jm.start("train", lambda job: train_job(job, from_blocks(CNN), d, TrainSettings(epochs=50)))
    while not jm.get(a)["progress"]:
        time.sleep(0.02)
    b = jm.start("train", lambda job: train_job(job, from_blocks(CNN), d, TrainSettings(epochs=1)))
    assert jm.get(a)["state"] == "cancelled"                    # 새 학습을 시작하기 전에 앞 학습이 끝남
    assert _wait(jm, b)["state"] == "done"


def test_failed_job_has_error():
    jm = JobManager()

    def boom(job):
        raise RuntimeError("일부러")

    st = _wait(jm, jm.start("x", boom))
    assert st["state"] == "failed" and "일부러" in st["error"]


def test_unknown_job():
    with pytest.raises(KeyError):
        JobManager().get("nope")


def test_rnn_job_weights_carry_over():
    rnn = [{"type": "gru", "hidden": 6, "layers": 1}, {"type": "dense", "units": 4}, {"type": "act", "kind": "tanh"},
           {"type": "output"}]
    d = make_dataset(*example_arrays(n=60, length=16))
    jm = JobManager()
    st = _wait(jm, jm.start("train", lambda job: train_job(job, from_blocks(rnn), d, TrainSettings(epochs=1))))
    assert st["state"] == "done", st["error"]
    assert len(st["result"]["confusion"]) == 2


def test_train_result_has_confusion():
    jm, d = JobManager(), _data()
    st = _wait(jm, jm.start("train", lambda job: train_job(job, from_blocks(CNN), d, TrainSettings(epochs=2, lr=1e-2))))
    r = st["result"]
    cm = r["confusion"]
    assert len(cm) == 2 and sum(map(sum, cm)) == len(d.y_test)
    assert len(r["recall"]) == 2
    assert isinstance(r["warnings"], list)


def test_concurrent_starts_never_overlap():
    """[학습]을 빠르게 두 번 눌러도 같은 종류 작업이 동시에 돌지 않음."""
    import threading
    jm, live, peak, lock = JobManager(), [0], [0], threading.Lock()

    def fn(job):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        for _ in range(30):
            if job.cancelled:
                break
            time.sleep(0.02)
        with lock:
            live[0] -= 1
        return {}

    ts = [threading.Thread(target=lambda: jm.start("train", fn)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    time.sleep(1.0)
    assert peak[0] == 1


def test_regression_result_has_mae_and_scatter():
    from dlstudio.data import load_example
    d = load_example("process_reg")
    dense = [{"type": "dense", "units": 16}, {"type": "act", "kind": "relu"}, {"type": "output"}]
    jm = JobManager()
    st = _wait(jm, jm.start("train", lambda job: train_job(job, from_blocks(dense), d, TrainSettings(epochs=5, lr=1e-2))))
    r = st["result"]
    assert r["metric_name"] == "r2" and r["mae"] >= 0
    assert len(r["scatter"]["y"]) == len(r["scatter"]["pred"]) == min(200, len(d.y_test))


def test_regression_with_offset_target_learns_and_outputs_original_units():
    """정답이 21–96처럼 0에서 먼 값이어도 학습이 됨 (정답을 정규화해 학습하고 출력층에서 되돌림)."""
    import torch
    from dlstudio.blocks import check, to_model_input
    from dlstudio.data import load_example
    d = load_example("process_reg")
    dense = [{"type": "dense", "units": 32}, {"type": "act", "kind": "relu"}, {"type": "dense", "units": 16},
             {"type": "act", "kind": "relu"}, {"type": "output"}]
    jm = JobManager()
    st = _wait(jm, jm.start("train", lambda job: train_job(job, from_blocks(dense), d, TrainSettings(epochs=30))))
    r = st["result"]
    assert r["test_metric"] > 0.5, r["test_metric"]
    with torch.no_grad():
        out = r["model"](torch.as_tensor(to_model_input(r["model"].kind, d.X_test), dtype=torch.float32)).numpy()[:, 0]
    assert abs(float(np.mean(np.abs(out - d.y_test))) - r["mae"]) < 1e-3       # 모델 자체가 원래 단위로 냄

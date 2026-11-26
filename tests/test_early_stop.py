"""가장 좋았던 epoch 남기기·조기 종료: 학습 몫의 10%를 검증 몫으로 떼어 검증 손실(탐지는 검증 mAP)로 고름. 시험 몫은 쓰지 않음."""
import warnings

import numpy as np
import pytest
import torch

from blocklearning import tasks
from blocklearning.data import load_example
from blocklearning.graph import build_graph_model, from_blocks
from blocklearning.jobs import Job, TrainSettings, fit, train_job

T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}


def test_fit_keeps_best_epoch_and_stops_early():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1, 8)).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.int64)
    g = from_blocks([{"type": "flatten"}, {"type": "dense", "units": 64}, {"type": "act", "kind": "relu"}, {"type": "output"}])
    seen = []
    net = build_graph_model(g, 8, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = fit(net, X, y, "classification", 60, 0.01, 16, keep_best=True, early_stop=4,
                  callback=lambda info: seen.append((info["epoch"], info["val_loss"], info["best_epoch"])) and False)
    fi = net.fit_info
    assert fi["n_val"] == 20 and fi["best_epoch"] == min(seen, key=lambda t: t[1])[0]
    assert fi["early_stopped"] == (len(seen) < 60)
    if fi["early_stopped"]:
        assert len(seen) - fi["best_epoch"] == 4                  # 4 epoch 동안 나아지지 않아 멈춤


def test_off_by_default_and_restores_best_weights():
    d = load_example("concrete")
    g = from_blocks(T["mlp"]["blocks"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = train_job(Job("a"), g, d, TrainSettings(epochs=3))
        assert r0["best_epoch"] is None and not r0["early_stopped"] and r0["n_val"] == 0
        j = Job("b")
        r = train_job(j, g, d, TrainSettings(epochs=12, keep_best=True))
    assert r["n_val"] == round(len(d.X_train) * 0.1) and r["best_epoch"] is not None
    assert all("val_loss" in p for p in j.progress)


def test_web_engine_same_options(tmp_path):
    from tests.test_engine import _train_js, pytestmark as engine_mark
    if engine_mark.mark.args[0]:
        pytest.skip(engine_mark.mark.kwargs.get("reason"))
    d = load_example("concrete")
    w = _train_js(tmp_path, from_blocks(T["mlp"]["blocks"]), d,
                  {"epochs": 40, "lr": 0.01, "batch_size": 32, "seed": 0, "keep_best": True, "early_stop": 3})
    r, prog = w["result"], w["progress"]
    assert r["n_val"] == round(len(d.X_train) * 0.1) and r["best_epoch"] == min(prog, key=lambda p: p["val_loss"])["epoch"]
    assert r["early_stopped"] == (len(prog) < 40)


def test_stop_checks_every_batch():
    """[멈추기]: epoch가 끝나기를 기다리지 않고 묶음마다 확인해 바로 멈춤 (멈춘 epoch는 기록하지 않음)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(320, 1, 8)).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.int64)
    g = from_blocks([{"type": "flatten"}, {"type": "dense", "units": 16}, {"type": "output"}])
    asked, seen = [], []

    def should_stop():
        asked.append(1)
        return len(asked) > 25                               # 한 epoch = 묶음 20개: 둘째 epoch 중간에 멈춤

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = fit(build_graph_model(g, 8, 2), X, y, "classification", 50, 0.01, 16,
                  callback=lambda info: seen.append(info["epoch"]) and False, should_stop=should_stop)
    assert seen == [1] and len(asked) == 26 and not net.training

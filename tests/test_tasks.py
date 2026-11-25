"""과제 고르기: 큰 틀 → 세부 과제 → 대표 모델(템플릿). 모든 템플릿은 자기 예시 데이터에서 검증·변환이 된다."""
import json
import warnings

import numpy as np

import pytest

from dlstudio import tasks
from dlstudio.blocks import build_model, check, to_model_input
from dlstudio.graph import build_graph_model, check_graph
from dlstudio.data import load_example
from dlstudio.tasks import CATALOG, find


def _available():
    for fam in CATALOG:
        for sub in fam["subtasks"]:
            if sub["available"]:
                for t in sub["templates"]:
                    yield fam, sub, t


def test_catalog_shape_and_json():
    json.dumps(CATALOG, ensure_ascii=False)
    ids = [f["id"] for f in CATALOG]
    assert ids == ["classification", "regression", "detection", "anomaly", "forecast", "segmentation"]
    for fam in CATALOG:
        assert fam["name"] and fam["desc"] and fam["subtasks"]
        for sub in fam["subtasks"]:
            assert sub["input"] in ("timeseries", "tabular", "image") and sub["examples_text"]
            if sub["available"]:
                assert sub["task"] in ("classification", "regression", "detection") and sub["example_data"] and sub["templates"]
                for t in sub["templates"]:
                    assert t["name"] and t["desc"] and t["when"] and "cost" not in t
    assert not any(s["available"] for f in CATALOG[3:] for s in f["subtasks"])   # 이상 탐지·예측·구간 분할은 준비 중


def test_find():
    fam, sub, t = find("classification", "cls_ts", "cnn")
    assert fam["id"] == "classification" and sub["id"] == "cls_ts" and t["id"] == "cnn"
    with pytest.raises(KeyError):
        find("classification", "nope", "cnn")


@pytest.mark.parametrize("fam,sub,t", list(_available()), ids=lambda x: x["id"] if isinstance(x, dict) else str(x))
def test_every_template_checks_and_converts_on_its_example(fam, sub, t):
    kind = sub["example_data"][0][0]
    d = load_example(kind)
    assert d.task == sub["task"]
    g = next(x for x in tasks.catalog_view()["families"] if x["id"] == fam["id"])
    g = next(x for x in g["subtasks"] if x["id"] == sub["id"])
    g = next(x for x in g["templates"] if x["id"] == t["id"])["graph"]
    c = check_graph(g, d.length, d.n_out, image=d.image)
    assert c.ok, (t["id"], c.issues)
    m = build_graph_model(g, d.length, d.n_out, image=d.image).eval()
    Xi = to_model_input(c.kind, d.X_train[:16], image=d.image)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        he = _Run(m)
    assert he.simulate(Xi[:4]).shape == (4, d.n_out)


def test_image_task_listed():
    fam, sub, t = find("classification", "cls_img", "cnn2d")
    assert sub["input"] == "image" and sub["example_data"][0][0] == "surface"
    assert [x["id"] for x in sub["templates"]] == ["cnn2d", "lenet", "resnet2d", "cnn_transformer", "vit"]


@pytest.mark.parametrize("kind", ["surface", "pneumonia", "digits"])
def test_image_examples_listed_and_small_cnn_converts(kind):
    sub = find("classification", "cls_img", "cnn2d")[1]
    assert kind in [k for k, _ in sub["example_data"]]
    d = load_example(kind)
    blocks = find("classification", "cls_img", "cnn2d")[2]["blocks"]
    m = build_model(blocks, d.length, d.n_out, image=d.image).eval()
    Xi = to_model_input("image", d.X_train[:8], image=d.image)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _Run(m).simulate(Xi[:2]).shape == (2, d.n_out)


def test_default_templates_are_graphs():
    view = tasks.catalog_view()
    assert [t["id"] for t in view["defaults"]] == ["cnn", "lstm", "inception", "mlp"]
    for t in view["defaults"]:
        L = 20 if t["id"] == "mlp" else 140
        assert check_graph(t["graph"], L, 2).ok, t["id"]



class _Run:
    """만든 모델을 그대로 앞으로 계산 (만들어지고 돌아가는지·모양 확인)."""

    def __init__(self, model):
        self.model = model.eval()

    def simulate(self, X):
        import torch
        with torch.no_grad():
            return self.model(torch.as_tensor(np.asarray(X), dtype=torch.float32)).double().numpy()

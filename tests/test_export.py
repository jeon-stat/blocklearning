"""대표 모델(ReLU, 암호문 이야기 없음)과 torch만으로 읽는 .pt 내보내기."""
import json
import warnings

import numpy as np
import pytest
import torch

from blocklearning import tasks
from blocklearning.blocks import to_model_input
from blocklearning.data import load_example
from blocklearning.graph import build_graph_model, check_graph
from blocklearning.project import export_torch


def G(*nodes):
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}, *nodes]}


def N(id, type, *inputs, **kw):
    return dict(id=id, type=type, inputs=list(inputs), **kw)


def _templates(view):
    for f in view["families"]:
        for s in f["subtasks"]:
            for t in s.get("templates", []):
                yield s, t
    for t in view["defaults"]:
        yield None, t


def _acts(graph):
    out = []

    def walk(nodes):
        for n in nodes:
            if n["type"] == "act":
                out.append(n["kind"])
            if n["type"] == "se":
                out.append(n.get("act", "relu"))
            if n.get("body"):
                walk(n["body"]["nodes"])
    walk(graph["nodes"])
    return out


def test_templates_use_relu_and_never_mention_he():
    view = tasks.catalog_view()
    text = json.dumps(view, ensure_ascii=False)
    assert "poly2" not in text and "암호문" not in text and '"cost"' not in text
    assert sum("relu" in _acts(t["graph"]) for _, t in _templates(view)) > 5


def test_concat_of_vectors_builds_and_runs():
    g = G(N("c", "conv", "input", channels=8, kernel=3, stride=1), N("g", "global_avg", "c"),
          N("a", "dense", "g", units=4), N("b", "dense", "g", units=6), N("k", "concat", "a", "b"), N("o", "output", "k"))
    c = check_graph(g, 40, 2)
    assert c.ok and c.shapes["k"] == (10,)
    m = build_graph_model(g, 40, 2).eval()
    with torch.no_grad():
        assert m(torch.randn(3, 1, 40)).shape == (3, 2)


@pytest.mark.parametrize("kind", ["ecg", "digits", "lstm", "wear_rul"])
def test_export_torch_runs_on_raw_rows(tmp_path, kind):
    if kind == "lstm":
        d = load_example("ecg")
        g = G(N("r", "lstm", "input", hidden=8, layers=1), N("o", "output", "r"))
    else:
        d = load_example(kind)
        tid = {"digits": "cnn2d"}.get(kind, "cnn")
        g = next(t["graph"] for _, t in _templates(tasks.catalog_view()) if t["id"] == tid)
    m = build_graph_model(g, d.length, d.n_out, seed=0, image=d.image).eval()
    p = tmp_path / "m.pt"
    export_torch(m, d, str(p))
    extra = {"studio.json": ""}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = torch.jit.load(str(p), _extra_files=extra)
    info = json.loads(extra["studio.json"])
    assert info["task"] == d.task and info["n_features"] == d.X_test.shape[1]
    assert info["classes"] == ([str(c) for c in d.classes] if d.task == "classification" else [])
    raw = d.raw(d.X_test[:5])                                     # CSV에 있던 원래 값 그대로
    with torch.no_grad():
        got = loaded(torch.as_tensor(raw, dtype=torch.float32)).numpy()
        want = m(torch.as_tensor(to_model_input(m.kind, d.X_test[:5], image=d.image), dtype=torch.float32)).numpy()
    assert np.allclose(got, want, atol=1e-4)

"""어텐션 열지도 (샘플 따라가기): 어텐션 블록마다 softmax 가중치를 꺼내 화면용으로 줄임. 엔진과 같은 값인지는 test_engine이 봄."""
import numpy as np
import pytest
import torch

from blocklearning import tasks
from blocklearning.analysis import attention, attn_view
from blocklearning.blocks import to_model_input
from blocklearning.graph import build_graph_model, from_blocks


def _tpl(tid):
    return next(t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", []) if t["id"] == tid)


def _run(tid, L, image=None):
    g = from_blocks(_tpl(tid)["blocks"])
    net = build_graph_model(g, L, 3, image=image)
    X = np.random.default_rng(0).normal(size=(1, image[0] * image[1] if image else L))
    return g, attention(net, to_model_input(net.kind, X, image=image)[0])


def test_vit_maps_cls_plus_patches():
    g, a = _run("vit", 64, (16, 16))
    mha = [n["id"] for n in g["nodes"] if n["type"] == "mha"]
    assert sorted(a) == sorted(mha) and len(mha) == 2
    for v in a.values():
        assert v["q"] == v["k"] == 1 + 16                      # CLS + 4×4 조각
        assert np.allclose(np.sum(v["map"], 1), 1, atol=1e-3)  # 줄마다 softmax
        assert len(v["first"]) == len(v["mean"]) == 17


def test_transformer_ts_and_hybrid_and_tabular():
    _, a = _run("transformer", 140)
    assert {v["q"] for v in a.values()} == {70}                # max_pool 2 뒤 70개 위치
    v = next(iter(a.values()))
    assert len(v["map"]) == 64 and len(v["map"][0]) == 70      # 질문 위치는 64개로 줄여 그림
    _, a = _run("cnn_transformer", 64, (32, 32))
    assert {v["k"] for v in a.values()} == {64}                # 32 → 8×8 칸
    _, a = _run("ft_transformer", 9)
    assert {v["k"] for v in a.values()} == {9}                 # 열 하나 = 토큰 하나


def test_no_attention_and_softmax_restored():
    real = torch.softmax
    _, a = _run("cnn", 64)
    assert a == {} and torch.softmax is real
    _run("vit", 64, (16, 16))
    assert torch.softmax is real


def test_attn_view_pool_shape():
    v = attn_view(np.array([[0.2, 0.8]]))
    assert v == {"q": 1, "k": 2, "map": [[0.2, 0.8]], "mean": [0.2, 0.8], "first": [0.2, 0.8]}


@pytest.mark.parametrize("tid,kind", [("transformer", "ecg"), ("ft_transformer", "steel_plates"), ("vit", "digits"),
                                      ("cnn_transformer", "steelblast")])
def test_api_activations_has_attention(tid, kind, tmp_path):
    from blocklearning.core import Studio
    from blocklearning.data import load_example
    s = Studio()
    d = load_example(kind)
    s.data = d
    g = from_blocks(_tpl(tid)["blocks"])
    s.model_graph = g
    s.model = build_graph_model(g, d.length, d.n_out, image=d.image)
    r = s.api_activations({"sample_index": 0})
    assert set(r["attention"]) == {n["id"] for n in g["nodes"] if n["type"] == "mha"}

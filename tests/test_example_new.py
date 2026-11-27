"""예측 예시용 새 데이터: 예시마다 학습·시험에 쓰지 않은 10개 (npz의 *_new) → ⑤ 예측에서 정답과 비교."""
import json
import warnings

import numpy as np
import pytest

from blocklearning import tasks
from blocklearning.data import EXAMPLE_KINDS, GEN_EXAMPLES, example_new, load_example

KINDS = [k for k in EXAMPLE_KINDS if k not in GEN_EXAMPLES]      # 생성 예시는 새 데이터로 예측하지 않음
from blocklearning.graph import build_graph_model, from_blocks

T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}
FIRST = {e[0]: s["templates"][0] for f in tasks.CATALOG for s in f["subtasks"] if s["available"] for e in s["example_data"]}


@pytest.mark.parametrize("kind", KINDS)
def test_example_new_is_held_out(kind):
    d = load_example(kind)
    e = example_new(d)
    assert len(e["names"]) == len(e["X"]) == 10
    X = np.asarray(e["X"], dtype=np.float64)
    seen = np.concatenate([d.raw(d.X_train), d.raw(d.X_test)])
    if d.task not in ("forecast",):                                # 같은 샘플이 학습·시험에 없음
        assert not any(np.isclose(seen, x, atol=1e-6).all(1).any() for x in X), kind
    if d.task == "forecast":
        assert np.asarray(e["truth"]).shape == (10, d.n_out)
    elif d.task == "segmentation":
        assert np.asarray(e["truth"]).shape == (10, d.length)
    elif d.task == "multi":
        assert np.asarray(e["truth"]).shape == (10, len(d.n_out.heads))
    elif d.task == "detection":
        assert len(e["truth"]) == 10 and sum(len(b) for b in e["truth"]) > 0
    else:
        assert len(e["labels"]) == 10


@pytest.mark.parametrize("kind", KINDS)
def test_predict_with_example_new(kind):
    from blocklearning.core import Studio
    s = Studio()
    s.api_data_load({"example": kind})
    d = s.data
    assert "example" in d.summary()["predict_kinds"]
    g = from_blocks(FIRST.get(kind, T["mlp"])["blocks"])            # hydraulic: 이상 탐지의 바탕 (과제 목록에는 없음)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.model, s.model_graph = build_graph_model(g, d.length, d.n_out, image=d.image).eval(), g
    s.results = {"threshold": 1.0}
    r = s.api_predict({"example_new": True})
    json.dumps(r, ensure_ascii=False)
    assert r["summary"]["n"] in (10, 10 * 1) and len(r["rows"]) == 10
    if d.task == "detection":
        assert all("truth" in p for p in r["pics"])
    elif d.task == "segmentation":
        assert r["has_label"] and r["summary"]["accuracy"] is not None
    else:
        assert r["has_label"]

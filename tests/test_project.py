"""프로젝트 저장·열기(.dlstudio = zip)."""
import zipfile

import numpy as np
import pytest
import torch

import io
import json

from dlstudio.blocks import build_model, to_model_input
from dlstudio.graph import build_graph_model, from_blocks, model_from_saved
from dlstudio.project import open_project, save_project

CNN = [{"type": "conv", "channels": 4, "kernel": 3}, {"type": "act", "kind": "relu"}, {"type": "global_avg"},
       {"type": "output"}]
GRAPH = from_blocks(CNN)
SRC = {"example": True, "normalize": True, "test_ratio": 0.2, "seed": 0}
SETTINGS = {"epochs": 3, "lr": 0.01}


def _X():
    return to_model_input("cnn", np.random.default_rng(0).normal(size=(5, 12)))


def test_save_open_roundtrip(tmp_path):
    m = build_graph_model(GRAPH, 12, 2, seed=3).eval()
    p = str(tmp_path / "a.dlstudio")
    save_project(p, GRAPH, SRC, SETTINGS, m, {"test_metric": 0.9})
    pr = open_project(p)
    assert pr["graph"] == GRAPH and pr["data_source"] == SRC and pr["train_settings"] == SETTINGS
    assert pr["results"]["test_metric"] == 0.9
    m2 = model_from_saved(pr, 12, 2).eval()
    with torch.no_grad():
        X = torch.as_tensor(_X(), dtype=torch.float32)
        assert torch.allclose(m(X), m2(X))


def test_save_without_model(tmp_path):
    p = str(tmp_path / "a.dlstudio")
    save_project(p, GRAPH, SRC, SETTINGS, None, {})
    assert open_project(p)["state_dict"] is None
    assert model_from_saved(open_project(p), 12, 2) is None


def test_open_rejects_other_zip(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(ValueError, match="프로젝트"):
        open_project(str(p))


def test_save_open_non_ascii_path(tmp_path):
    p = tmp_path / "크립토랩" / "모델.dlstudio"
    save_project(str(p), GRAPH, SRC, SETTINGS, build_graph_model(GRAPH, 12, 2), {})
    assert open_project(str(p))["graph"] == GRAPH


def test_selection_roundtrip(tmp_path):
    p = str(tmp_path / "s.dlstudio")
    sel = {"family": "classification", "subtask": "cls_ts", "template": "cnn"}
    save_project(p, GRAPH, SRC, SETTINGS, None, {}, selection=sel)
    assert open_project(p)["selection"] == sel
    save_project(p, GRAPH, SRC, SETTINGS, None, {})
    assert open_project(p)["selection"] is None

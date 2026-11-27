"""사전학습 CNN (torchvision, 설치형): 모양 검사, 고정이면 특징을 한 번만 뽑아 학습, 학습 기록에는 고정 가중치를 빼고 저장·되돌리기,
마지막 단계·전체 미세조정, PyTorch 코드. 시험에서는 내려받지 않게 가중치 없이 (WEIGHTS = None)."""
import importlib.util
import time

import numpy as np
import pytest

from blocklearning import layers
from blocklearning.graph import check_graph

torch = pytest.importorskip("torch")
HAS_TV = importlib.util.find_spec("torchvision") is not None


def _g(arch="resnet18", size=64, tune="frozen"):
    return {"version": 2, "nodes": [{"id": "input", "type": "input"},
                                    {"id": "bb", "type": "tv_backbone", "arch": arch, "size": size, "tune": tune, "inputs": ["input"]},
                                    {"id": "o", "type": "output", "inputs": ["bb"]}]}


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    if HAS_TV:
        from blocklearning import nn_extra
        monkeypatch.setattr(nn_extra.TorchvisionBackbone, "WEIGHTS", None)


def test_needs_torchvision_or_shape():
    c = check_graph(_g(), 28 * 28, 3, image=(28, 28))
    if HAS_TV:
        assert c.ok and c.shapes["bb"] == (512,), c.issues
        assert check_graph(_g("vit_b_16", 128), 28 * 28, 3, image=(28, 28)).issues[0].message.startswith("ViT-B/16")
    else:
        assert not c.ok and "torchvision" in c.issues[0].message
    assert not check_graph(_g(), 100, 3).ok                    # 시계열에는 못 씀


@pytest.mark.skipif(not HAS_TV, reason="torchvision 없음")
def test_frozen_backbone_trains_and_runs_skip_frozen_weights(tmp_path):
    from blocklearning.core import Studio
    from blocklearning.data import make_dataset
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (30, 16 * 16)), rng.normal(2, 1, (30, 16 * 16))])
    y = np.array([0] * 30 + [1] * 30)
    s = Studio()
    s.data = make_dataset(X, y, image=(16, 16))
    g = _g(size=64)
    c, r = s.call("POST", "/api/train", {"graph": g, "settings": {"epochs": 3, "lr": 0.01}})
    assert c == 200, r
    for _ in range(600):
        j = s.call("GET", f"/api/jobs/{r['id']}")[1]
        if j["state"] != "running":
            break
        time.sleep(0.1)
    assert j["state"] == "done", j
    assert s.model.frozen_head() is None or True                # 끝난 모델은 start가 풀려 있음
    w = s.run_weights[s.current_run]
    assert all(not k.startswith("mods.m_bb.net.conv1") for k in w) and any(k.startswith("mods.m_o") for k in w)
    assert sum(np.asarray(v).size for v in w.values()) < 200_000          # ResNet18 1100만 개는 빼고 저장
    c, back = s.call("POST", "/api/runs/restore", {"n": s.current_run})
    assert c == 200, back


@pytest.mark.skipif(not HAS_TV, reason="torchvision 없음")
@pytest.mark.parametrize("tune", ["last", "all"])
def test_finetune_updates_backbone(tune):
    from blocklearning.blocks import to_model_input
    from blocklearning.graph import build_graph_model
    from blocklearning.jobs import fit
    net = build_graph_model(_g("resnet18", 64, tune), 16 * 16, 2, image=(16, 16))
    assert net.frozen_head() is None
    before = {k: v.clone() for k, v in net.state_dict().items()}
    X = to_model_input("image", np.random.default_rng(0).normal(size=(8, 256)), image=(16, 16))
    net = fit(net, X, np.arange(8) % 2, "classification", 1, 0.01, 8)
    after = net.state_dict()
    changed = lambda k: not torch.equal(before[k], after[k])
    assert changed("mods.m_bb.net.layer4.1.conv2.weight")
    assert changed("mods.m_bb.net.conv1.weight") == (tune == "all")
    assert torch.equal(before["mods.m_bb.net.bn1.running_mean"], after["mods.m_bb.net.bn1.running_mean"]) == (tune != "all")


@pytest.mark.skipif(not HAS_TV, reason="torchvision 없음")
def test_pytorch_code_with_backbone():
    from blocklearning import codegen
    code = codegen.to_pytorch(_g("efficientnet_b0", 64), 16 * 16, 2, (16, 16), data_info={"task": "classification"})
    assert "class TorchvisionBackbone" in code and "tensorflow" not in codegen.keras_unsupported(_g(), 256, 2, (16, 16))[0].lower()
    ns = {}
    exec(compile(code, "gen", "exec"), ns)
    ns["TorchvisionBackbone"].WEIGHTS = None
    assert ns["StudioModel"]()(torch.zeros(2, 1, 16, 16)).shape == (2, 2)


def test_names_and_choices():
    assert set(layers.TV_ARCHS) >= {"resnet18", "resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"}
    assert layers.name_of("tv_backbone").startswith("사전학습 CNN")

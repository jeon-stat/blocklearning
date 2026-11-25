"""어디를 봤나 (입력 기울기): 과제마다 PyTorch(analysis.saliency)와 웹 엔진(engine.js saliency)이 같은 값."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from dlstudio import tasks
from dlstudio.analysis import saliency
from dlstudio.blocks import to_model_input
from dlstudio.codegen import plan_dict
from dlstudio.data import channels_of, load_example
from dlstudio.graph import build_graph_model, from_blocks

JS = Path(__file__).parent / "js"
HAS_NODE = shutil.which("node") is not None and (JS / "node_modules" / "@tensorflow" / "tfjs").exists()
T = {t["id"]: t for f in tasks.CATALOG for s in f["subtasks"] for t in s.get("templates", [])}
CASES = [("ecg", "cnn"), ("ecg", "lstm"), ("pneumatic_seg2", "seg_tcn"), ("hydraulic_multi", "multi_mlp"), ("digits", "cnn2d"),
         ("bearing_ae", "ae_dense"), ("concrete", "mlp")]


def _meta(d):
    return {"task": d.task, "length": int(d.length), "channels": channels_of(d.length), "image": list(d.image) if d.image else None,
            "seg": {"n_cls": d.n_out.n_cls, "length": d.n_out.length} if d.task == "segmentation" else None,
            "multi": {"heads": d.n_out.heads} if d.task == "multi" else None}


@pytest.mark.skipif(not HAS_NODE, reason="node 또는 tests/js/node_modules 없음")
def test_engine_saliency_matches_torch(tmp_path):
    fx, want = [], []
    for ex, tid in CASES:
        d = load_example(ex)
        g = from_blocks(T[tid]["blocks"])
        net = build_graph_model(g, d.length, d.n_out, seed=1, image=d.image).eval()
        x = to_model_input(net.kind, d.X_test[:1], image=d.image, channels=channels_of(d.length))[0]
        want.append(saliency(net, x, d))
        st = {k: {"shape": list(v.shape), "data": v.detach().numpy().ravel().tolist(), "int": not v.is_floating_point()}
              for k, v in net.state_dict().items()}
        fx.append({"plan": plan_dict(g, d.length, d.n_out, d.image), "state": st, "x": d.X_test[0].tolist(), "d": _meta(d)})
    f = tmp_path / "sal.json"
    f.write_text(json.dumps(fx), encoding="utf-8")
    r = subprocess.run(["node", str(JS / "run_sal.js"), str(f)], capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    for (ex, tid), got, w in zip(CASES, json.loads(r.stdout), want):
        assert len(got) == len(w), (ex, tid)
        assert np.allclose(got, w, atol=2e-3), (ex, tid, float(np.abs(np.asarray(got) - w).max()))

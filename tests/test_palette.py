"""편집기 왼쪽 목록(graph.js PALETTE)과 묶음(PACKS): 모든 블록이 알맞은 자리에서 검사를 통과하고, 묶음을 넣은 모델이
검사·만들기·계산까지 되는지 (node로 graph.js를 불러 그래프를 만들고 Python이 검사)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dlstudio.blocks import to_model_input
from dlstudio.graph import build_graph_model, check_graph

GRAPH_JS = Path(__file__).resolve().parents[1] / "src" / "dlstudio" / "static" / "graph.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")


def _node(script: str):
    code = f"const L = require({json.dumps(str(GRAPH_JS))});\n{script}"
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def _base(kind):
    """Input → (Conv 8채널) → [여기] → Flatten → Output. kind: s 줄 · i 그림 · v 숫자 묶음."""
    pre = {"s": [{"id": "c", "type": "conv", "channels": 8, "kernel": 3, "stride": 1, "inputs": ["input"]}],
           "i": [{"id": "c", "type": "conv2d", "channels": 8, "kernel": 3, "stride": 1, "inputs": ["input"]}],
           "v": [{"id": "c0", "type": "conv", "channels": 8, "kernel": 3, "stride": 1, "inputs": ["input"]},
                 {"id": "c", "type": "global_avg", "inputs": ["c0"]}]}[kind]
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}] + pre +
            [{"id": "f", "type": "flatten", "inputs": ["c"]}, {"id": "o", "type": "output", "inputs": ["f"]}]}


DIMS = {"s": (32, None), "i": (64, (8, 8)), "v": (32, None)}
SHAPE = {"s": [8, 32], "i": [8, 8, 8], "v": [8]}


def _ok(g, kind):
    L, image = DIMS[kind]
    c = check_graph(g, L, 3, image=image)
    return c, L, image


def test_every_pack_builds_and_runs():
    packs = _node("process.stdout.write(JSON.stringify(L.PACKS.map((p) => [p.id, p.ranks])));")
    assert len(packs) >= 14
    done = 0
    for pid, ranks in packs:
        for kind, rank in (("s", 2), ("i", 3), ("v", 1)):
            if rank not in ranks:
                continue
            g = _node(f"""const g = {json.dumps(_base(kind))};
              const pk = L.makePack({json.dumps(pid)}, {SHAPE[kind][0]}, {rank}, {SHAPE[kind][1] if rank == 2 else 0});
              process.stdout.write(JSON.stringify(L.insertPack(g, "c", "f", pk).graph));""")
            c, L_, image = _ok(g, kind)
            assert c.ok, (pid, kind, [(i.node, i.message) for i in c.issues])
            net = build_graph_model(g, L_, 3, image=image).eval()
            X = np.random.default_rng(0).normal(size=(2, image[0] * image[1] if image else L_))
            import torch
            with torch.no_grad():
                y = net(torch.as_tensor(to_model_input(net.kind, X, image=image), dtype=torch.float32))
            assert tuple(y.shape) == (2, 3), pid
            done += 1
    assert done >= 20


def test_every_palette_block_checks_somewhere():
    """목록의 블록마다: 줄·그림·숫자 묶음 중 맞는 자리(증강은 Input 바로 뒤)에서 기본 설정으로 검사 통과."""
    items = _node("process.stdout.write(JSON.stringify(Object.entries(L.PALETTE).map(([id, p]) => [id, p.cat, p.make()])));")
    assert len(items) >= 85
    bad = []
    for pid, cat, node in items:
        if node["type"] in ("add", "concat", "multiply", "average", "maximum", "cross_attn"):
            continue                                           # 입력 둘 이상 (다른 시험에서)
        ok = False
        for kind in ("s", "i", "v"):
            g = _base(kind)
            n = dict(node, id="x", inputs=["c"])
            if cat in ("aug",) or node["type"] in ("embedding", "spectrogram", "patch_embed"):
                n["inputs"] = ["input"]
                g["nodes"] = [g["nodes"][0], n, {"id": "f", "type": "flatten", "inputs": ["x"]}, g["nodes"][-1]]
            else:
                g["nodes"].insert(-2, n)
                g["nodes"][-2]["inputs"] = ["x"]
            L_, image = DIMS[kind]
            if node["type"] == "spectrogram":
                L_ = 256
            if check_graph(g, L_, 3, image=image).ok:
                ok = True
                break
        if not ok:
            bad.append(pid)
    assert not bad, bad


def test_two_input_connect_and_merge_list():
    r = _node("""const g = {version: 2, nodes: [{id: "input", type: "input"}, {id: "a", type: "conv", channels: 4, kernel: 3, stride: 1, inputs: ["input"]},
        {id: "b", type: "conv", channels: 4, kernel: 3, stride: 1, inputs: ["input"]}, {id: "x", type: "cross_attn", heads: 2, inputs: ["a"]},
        {id: "o", type: "output", inputs: ["x"]}]};
      const g2 = L.connect(g, "b", "x").graph, g3 = L.connect(g2, "input", "x").graph;
      process.stdout.write(JSON.stringify([g2.nodes[3].inputs, g3.nodes[3].inputs, L.label({type: "mha", heads: 4, causal: true, pos: "rope"})]));""")
    assert r[0] == ["a", "b"] and r[1] == ["a", "input"] and r[2] == "Causal Attention · 4 heads · RoPE"


def test_for_input_follows_shape():
    r = _node("""process.stdout.write(JSON.stringify([L.forInput({type: "max_pool", size: 2}, "timeseries", 3).type,
      L.forInput({type: "conv2d", channels: 4, kernel: 3}, "image", 2).type, L.forInput({type: "conv", channels: 4, kernel: 9}, "image").kernel,
      L.forInput({type: "mha", heads: 2}, "image", 3).type]));""")
    assert r == ["max_pool2d", "conv", 3, "mha"]

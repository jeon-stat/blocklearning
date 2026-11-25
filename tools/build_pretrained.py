"""사전학습 가중치 만들기: Google MobileNetV3-Small (ImageNet, Apache 2.0)의 TF.js 파일 → DL studio 가중치 (src/dlstudio/pretrained/).

    python tools/build_pretrained.py <TF.js 모델 폴더>

모델 폴더: Kaggle google/mobilenet-v3 · TfJs · small-100-224-feature-vector · 버전 1 (TF Hub
google/imagenet/mobilenet_v3_small_100_224/feature_vector/5를 TF.js로 바꾼 것)의 model.json과 group1-shard*.bin.
    https://www.kaggle.com/models/google/mobilenet-v3/TfJs/small-100-224-feature-vector/1/<파일>?tfjs-format=file
BatchNorm은 이미 접혀 있음. 합성곱 가중치를 PyTorch 모양(출력, 입력, 높이, 너비)으로 바꿔 nn_extra.MobileNetV3Small의 이름으로 저장.

<이름>.bin: float32 값(little-endian)을 차례로 이어 붙임 · <이름>.json: [[이름, 모양], …] (그 차례) · <이름>.SOURCE.txt: 출처.
만든 뒤 원래 모델(TF.js 그래프, node)과 같은 값인지 확인한다 (tests/js/node_modules의 @tensorflow/tfjs).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "src" / "dlstudio" / "pretrained"
P = "StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/"

SOURCE = """MobileNetV3-Small 특징 추출기 (ImageNet ILSVRC-2012로 미리 학습) — Google, Apache License 2.0 (LICENSE-mnv3s.txt).
Kaggle google/mobilenet-v3, TfJs small-100-224-feature-vector 버전 1 (TF Hub google/imagenet/mobilenet_v3_small_100_224/feature_vector/5의 TF.js판)
https://www.kaggle.com/models/google/mobilenet-v3/tfJs/small-100-224-feature-vector/1
Howard A, et al. Searching for MobileNetV3. ICCV 2019 (arXiv:1905.02244).
바꾼 내용: 합성곱 가중치의 축 순서를 PyTorch 모양(출력, 입력, 높이, 너비)으로 바꾸고 이름을 붙여 float32로 다시 저장 (값은 그대로).
원래 모델과 같은 계산: 입력 −1~1, TensorFlow 'SAME' 채움, 끝의 1×1 합성곱(1024)까지. DL studio는 앞에 그림마다 0~1로 맞추기와 크기 바꾸기를 더함.
"""

CHECK_JS = r"""
const fs = require("fs");
globalThis.tf = require(process.argv[2]);
const tf = globalThis.tf;
(async () => {
  const dir = process.argv[3], m = JSON.parse(fs.readFileSync(dir + "/model.json", "utf8"));
  const bufs = m.weightsManifest[0].paths.map((p) => fs.readFileSync(dir + "/" + p));
  const all = Buffer.concat(bufs);
  const model = await tf.loadGraphModel(tf.io.fromMemory({ modelTopology: m.modelTopology, weightSpecs: m.weightsManifest[0].weights,
    weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength), format: m.format, generatedBy: m.generatedBy,
    convertedBy: m.convertedBy, signature: m.signature }));
  const x = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
  const y = model.predict(tf.tensor(x.data, x.shape));
  process.stdout.write(JSON.stringify(Array.from(y.dataSync())));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def read_tfjs(folder: Path) -> dict:
    m = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    g = m["weightsManifest"][0]
    buf = b"".join((folder / p).read_bytes() for p in g["paths"])
    W, off = {}, 0
    for w in g["weights"]:
        assert w["dtype"] in ("float32", "int32") and "quantization" not in w, w["name"]
        n = int(np.prod(w["shape"])) if w["shape"] else 1
        W[w["name"]] = np.frombuffer(buf, "<f4" if w["dtype"] == "float32" else "<i4", n, off).reshape(w["shape"])
        off += 4 * n
    assert off == len(buf)
    se = {}                                               # SE·머리 가중치는 이름이 없어 (Func/…) 그래프의 입력으로 찾음
    for n in m["modelTopology"]["node"]:
        name = n["name"].replace(P, "")
        if name.endswith("squeeze_excite/Conv/Relu") or name.endswith("squeeze_excite/Conv_1/BiasAdd") or name == "Conv_2/BiasAdd":
            se[name] = n["input"][1:3]
    return W, se


def convert(folder: Path) -> None:
    import torch
    from dlstudio.nn_extra import MobileNetV3Small
    W, se = read_tfjs(folder)
    conv = lambda a: torch.as_tensor(np.ascontiguousarray(a.transpose(3, 2, 0, 1)))       # (h, w, 입력, 출력) → (출력, 입력, h, w)
    dwc = lambda a: torch.as_tensor(np.ascontiguousarray(a.transpose(2, 3, 0, 1)))        # (h, w, 채널, 1) → (채널, 1, h, w)
    vec = lambda a: torch.as_tensor(np.array(a))
    st = {"stem.weight": conv(W[P + "Conv/Conv2D_weights"]), "stem.bias": vec(W[P + "Conv/Conv2D_bn_offset"])}
    for j, (i, k, e, o, q, hs, s) in enumerate(MobileNetV3Small.CFG):
        t = "expanded_conv" + (f"_{j}" if j else "")
        if e != i:
            st[f"blocks.{j}.expand.weight"] = conv(W[P + f"{t}/expand/Conv2D_weights"])
            st[f"blocks.{j}.expand.bias"] = vec(W[P + f"{t}/expand/Conv2D_bn_offset"])
        st[f"blocks.{j}.dw.weight"] = dwc(W[P + f"{t}/depthwise/depthwise_weights"])
        st[f"blocks.{j}.dw.bias"] = vec(W[P + f"{t}/depthwise/depthwise_bn_offset"])
        if q:
            for ours, theirs in (("se1", "squeeze_excite/Conv/Relu"), ("se2", "squeeze_excite/Conv_1/BiasAdd")):
                w_, b_ = se[f"{t}/{theirs}"]
                st[f"blocks.{j}.{ours}.weight"], st[f"blocks.{j}.{ours}.bias"] = conv(W[w_]), vec(W[b_])
        st[f"blocks.{j}.project.weight"] = conv(W[P + f"{t}/project/Conv2D_weights"])
        st[f"blocks.{j}.project.bias"] = vec(W[P + f"{t}/project/Conv2D_bn_offset"])
    st["last.weight"], st["last.bias"] = conv(W[P + "Conv_1/Conv2D_weights"]), vec(W[P + "Conv_1/Conv2D_bn_offset"])
    w_, b_ = se["Conv_2/BiasAdd"]
    st["head.weight"], st["head.bias"] = conv(W[w_]), vec(W[b_])
    ours = MobileNetV3Small(size=224).eval()
    ours.load_state_dict(st)                                  # 이름·모양이 모두 맞아야 함

    rng = np.random.default_rng(0)
    x = rng.random((3, 224, 224, 3), dtype=np.float32)        # 원래 모델의 입력: 0~1, (N, H, W, 3)
    with torch.no_grad():
        got = ours.features(torch.as_tensor(x).permute(0, 3, 1, 2) * 2 - 1).numpy()
    tfjs = ROOT / "tests" / "js" / "node_modules" / "@tensorflow" / "tfjs"
    with tempfile.TemporaryDirectory() as tmp:
        js, xf = Path(tmp) / "check.js", Path(tmp) / "x.json"
        js.write_text(CHECK_JS, encoding="utf-8")
        xf.write_text(json.dumps({"shape": list(x.shape), "data": x.ravel().tolist()}), encoding="utf-8")
        r = subprocess.run(["node", str(js), str(tfjs), str(folder), str(xf)], capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-2000:]
    want = np.array(json.loads(r.stdout), dtype=np.float32).reshape(got.shape)
    err = float(np.abs(got - want).max() / np.abs(want).max())
    assert err < 1e-4, err

    order = list(ours.state_dict())
    arr = {k: st[k].numpy().astype("<f4") for k in order}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mnv3s.bin").write_bytes(b"".join(arr[k].tobytes() for k in order))
    (OUT / "mnv3s.json").write_text(json.dumps([[k, list(arr[k].shape)] for k in order]), encoding="utf-8")
    (OUT / "mnv3s.SOURCE.txt").write_text(SOURCE, encoding="utf-8", newline="\n")
    n = sum(v.size for v in arr.values())
    print(f"mnv3s: 값 {n:,}개, {(OUT / 'mnv3s.bin').stat().st_size / 1e6:.2f} MB, 원래 모델과 상대 차이 {err:.1e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    convert(Path(sys.argv[1]))

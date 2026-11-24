// tests/test_detect.py가 부름: 탐지 계산(상자 풀기·mAP·정밀도·손실)을 엔진으로 해서 출력 → Python(detect.py)과 비교.
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/dlstudio/static/engine.js"));
const f = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = Float32Array.from(f.out), n = f.gts.length, nOut = out.length / n;
const res = E.detResult(out, f.gts, f.det);
const loss = E.tf.tidy(() => E.detLoss(E.tf.tensor2d(out, [n, nOut]), E.tf.tensor2d(Float32Array.from(f.target), [n, nOut]),
  f.det.grid, f.det.per_cell).dataSync()[0]);
process.stdout.write(JSON.stringify({ res, loss, decoded: Array.from({ length: n }, (_, i) => E.detDecode(out, i, f.det)) }));

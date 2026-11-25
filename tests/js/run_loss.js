// tests/test_losses.py가 부름: 출력·정답·손실 항으로 엔진의 손실(buildLoss)을 계산해 출력.
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/dlstudio/static/engine.js"));
const tf = E.tf;
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = [];
for (const c of cases) {
  const o = tf.tensor(c.out, c.out_shape), y = tf.tensor(c.y, c.y_shape);
  const L = E.buildLoss(c.spec, { seg: c.seg, det: null, cw: c.cw, reg: c.reg });
  const { total, parts } = L(o, y, { trainable: [] });
  out.push({ total: total.dataSync()[0], parts: Object.fromEntries(parts.map(([k, v]) => [k, v.dataSync()[0]])) });
}
process.stdout.write(JSON.stringify(out));

// tests/test_saliency.py가 부름: 계획·가중치·샘플로 엔진의 '어디를 봤나'(saliency)를 계산해 출력.
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/dlstudio/static/engine.js"));
const f = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = [];
for (const c of f) {
  const net = new E.Net(c.plan, 0);
  net.load(Object.fromEntries(Object.entries(c.state).map(([k, v]) => [k, { shape: v.shape, data: v.int ? Int32Array.from(v.data) : Float32Array.from(v.data) }])));
  out.push(E.saliency(net, Float32Array.from(c.x), c.d));
  net.dispose();
}
process.stdout.write(JSON.stringify(out));

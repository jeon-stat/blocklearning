// tests/test_engine.py가 부름: 계획과 데이터(JSON)로 엔진 학습을 돌려 결과를 출력.
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/dlstudio/static/engine.js"));
const PRE = path.join(__dirname, "../../src/dlstudio/pretrained");      // 사전학습 가중치 (브라우저에서는 ensurePretrained가 받음)
for (const f of fs.readdirSync(PRE).filter((n) => n.endsWith(".bin"))) {
  const kind = f.slice(0, -4), b = fs.readFileSync(path.join(PRE, f));
  E.setPretrained(kind, JSON.parse(fs.readFileSync(path.join(PRE, kind + ".json"), "utf8")), b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
(async () => {
  const f = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const d = Object.assign({}, f.data, {
    X_train: Float32Array.from(f.data.X_train), X_test: Float32Array.from(f.data.X_test),
    y_train: Float32Array.from(f.data.y_train), y_test: Float32Array.from(f.data.y_test),
  });
  const progress = [];
  const { net, result, out, outTr } = await E.train(f.plan, d, f.settings, (p) => { progress.push(p); return false; });
  const state = net.state();
  process.stdout.write(JSON.stringify({ result, progress, out: out && Array.from(out), outTr: outTr && Array.from(outTr), state: Object.fromEntries(Object.entries(state).map(([k, v]) => [k, { shape: v.shape, data: Array.from(v.data) }])) }));
})().catch((e) => { console.error(e); process.exit(1); });

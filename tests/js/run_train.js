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
  let got = null;
  const onEpoch = (p) => { progress.push(p); return false; };
  const t0 = Date.now();
  let askedStop = null;                                              // stop_after_ms: 그 뒤 [멈추기] (처음 멈춤을 본 때부터 끝날 때까지를 잼)
  const isCancelled = f.stop_after_ms ? () => { const on = Date.now() - t0 > f.stop_after_ms; if (on && askedStop === null) askedStop = Date.now(); return on; } : undefined;
  const { net, result, out, outTr } = f.distance                    // 이상 탐지 '정상과의 거리': 넣을 값은 시험이 줌 (웹은 Python이 계산)
    ? await E.fitDistance(f.plan, d, async (F, n, c) => { got = { n, c, sum: F.reduce((a, b) => a + b, 0) }; return f.distance; }, onEpoch)
    : await E.train(f.plan, d, f.settings, onEpoch, isCancelled);
  const stopMs = askedStop === null ? null : Date.now() - askedStop;
  const state = net.state();
  process.stdout.write(JSON.stringify({ result, progress, got, stopMs, out: out && Array.from(out), outTr: outTr && Array.from(outTr), state: Object.fromEntries(Object.entries(state).map(([k, v]) => [k, { shape: v.shape, data: Array.from(v.data) }])) }));
})().catch((e) => { console.error(e); process.exit(1); });

// tests/test_engine.py가 부름: 픽스처(JSON)의 계획·가중치·입력으로 엔진을 돌려 결과를 JSON으로 출력.
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/dlstudio/static/engine.js"));
const tf = E.tf;
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = {};
for (const c of cases) {
  const net = new E.Net(c.plan, 0);
  net.load(c.state);
  const x = tf.tensor(c.x.data, c.x.shape);
  const r = { params: net.numParams() };
  r.y = Array.from(tf.tidy(() => net.forward(x, false)).dataSync());
  const tr = net.trace(x.slice(0, 1));
  r.trace = Object.fromEntries(Object.entries(tr).map(([k, v]) => [k, v.flat(Infinity)]));
  if (c.grad) {                                          // 학습 모드(BatchNorm은 묶음 통계) 교차 엔트로피의 기울기
    const y = tf.oneHot(tf.tensor1d(c.grad.labels, "int32"), c.plan.n_out).toFloat();
    const pending = [];
    const { value, grads } = tf.variableGrads(() => tf.losses.softmaxCrossEntropy(y, net.forward(x, true, pending)), net.trainable);
    r.loss = value.dataSync()[0];
    r.grads = Object.fromEntries(Object.entries(grads).map(([k, g]) => [net.nameOf[k], Array.from(g.dataSync())]));
    r.running = Object.fromEntries(pending.filter(([, v]) => v).map(([b, v]) => [net.nameOf[b.name], Array.from(v.dataSync())]));
    net.dispose();
  }
  out[c.name] = r;
}
process.stdout.write(JSON.stringify(out));

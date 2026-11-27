// tests/test_optim.py가 부름: 같은 가중치·같은 묶음으로 엔진의 학습 방법(옵티마이저·스케줄·weight decay·자르기)을 몇 걸음 돌려 가중치를 출력.
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/blocklearning/static/engine.js"));
const tf = E.tf;
const c = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = {};
for (const s of c.settings) {
  const net = new E.Net(c.plan, 0);
  net.load(c.state);
  const x = tf.tensor(c.x.data, c.x.shape), y = tf.oneHot(tf.tensor1d(c.labels, "int32"), c.plan.n_out).toFloat();
  const oc = E.optimConfig(s), lr0 = s.lr;
  const opt = oc.optimizer === "sgd" ? tf.train.momentum(lr0, E.OPT_MOMENTUM, true) : tf.train.adam(lr0, 0.9, 0.999, 1e-8);
  for (let i = 0; i < c.steps; i++) {
    tf.tidy(() => E.optimStep(opt, oc, net.trainable, E.lrAt(i, c.steps, lr0, oc.schedule), () => tf.losses.softmaxCrossEntropy(y, net.forward(x, true, []))));
  }
  out[s.key] = Object.fromEntries(net.trainable.map((v) => [net.nameOf[v.name], Array.from(v.dataSync())]));
  net.dispose();
}
process.stdout.write(JSON.stringify(out));

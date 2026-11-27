// tests/test_generate.py가 부름: 가중치를 넣은 엔진 모델에 클래스·시점을 넣고 forward한 출력 (PyTorch와 맞춰 봄)
const fs = require("fs");
const path = require("path");
globalThis.tf = require("@tensorflow/tfjs");
const E = require(path.join(__dirname, "../../src/blocklearning/static/engine.js"));
const f = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const tf = E.tf;
const net = new E.Net(f.plan, 0);
net.load(Object.fromEntries(Object.entries(f.state).map(([k, v]) => [k, { shape: v.shape, data: Float32Array.from(v.data) }])));
const x = tf.tensor(f.x, f.shape);
net.cond = { y: f.y ? tf.tensor1d(Int32Array.from(f.y), "int32") : null, t: f.t ? tf.tensor1d(f.t) : null };
const out = net.forward(x, false).dataSync();
const T = E.SCHED;
process.stdout.write(JSON.stringify({ out: Array.from(out), ab: Array.from(T.ab), kind: E.genKind(f.plan), classes: E.condClasses(f.plan) }));

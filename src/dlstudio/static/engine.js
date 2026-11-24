// DL studio 웹 학습 엔진 (TensorFlow.js): Python이 준 계획(codegen.plan_dict)대로 모델을 만들고 학습한다.
// PyTorch(GraphNet)와 계산이 같도록 층마다 PyTorch 식을 TF.js 연산으로 그대로 옮겼다 — 값의 모양은 PyTorch처럼 채널이 먼저
// (시계열 (N, C, L), 이미지 (N, C, H, W), 순환 층 입력 (N, L, 1)), 가중치 이름·모양도 PyTorch state_dict와 같다.
// 그래서 학습한 가중치를 그대로 PyTorch 코드로 내보낼 수 있다. 브라우저(전역 tf)와 node(require)에서 함께 쓴다.
(function () {
"use strict";
const tf = globalThis.tf || require("@tensorflow/tfjs");          // 브라우저: 전역 tf, node: 부르는 쪽이 globalThis.tf를 넣어 줌

// ------------------------------------------------------------ 난수 (seed로 같은 결과)
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
const prod = (a) => a.reduce((x, y) => x * y, 1);
// 잠깐 양보 (화면 그리기·단추가 돌게). tf.nextFrame은 창이 숨겨지면 멈춰서 학습도 멈춤 → MessageChannel·setImmediate
const yieldNow = () => new Promise((r) => {
  if (typeof MessageChannel !== "undefined") { const c = new MessageChannel(); c.port1.onmessage = () => { c.port1.close(); r(); }; c.port2.postMessage(0); }
  else setImmediate(r);
});

// ------------------------------------------------------------ 가중치 만들기 (PyTorch 기본 초기화와 같은 분포)
function initParam(step, name, shape, kind, rand) {
  const n = prod(shape);
  if (kind === "count") return tf.scalar(0, "int32");
  const leaf = name.split(".").pop();
  if (step.type === "bn") {
    if (leaf === "weight" || leaf === "running_var") return tf.ones(shape);
    return tf.zeros(shape);
  }
  let bound;
  if (["lstm", "gru", "simple_rnn"].includes(step.type)) bound = 1 / Math.sqrt(step.cfg.hidden);
  else {                                                     // Conv·Linear: kaiming_uniform(a=√5) → U(±1/√fan_in), bias도 같음
    const wshape = leaf === "bias" ? null : shape;
    let fanIn;
    if (wshape) fanIn = prod(wshape.slice(1));
    else {
      const wname = name.slice(0, -"bias".length) + "weight";
      const w = step.params.find((p) => p[0] === wname);
      fanIn = prod(w[1].slice(1));
    }
    bound = 1 / Math.sqrt(fanIn);
  }
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = (rand() * 2 - 1) * bound;
  return tf.tensor(v, shape);
}

// ------------------------------------------------------------ 층 계산 (PyTorch와 같은 식)
const ACT = {
  relu: (x) => tf.relu(x),
  leaky_relu: (x) => tf.leakyRelu(x, 0.01),
  gelu: (x) => x.mul(tf.erf(x.div(Math.SQRT2)).add(1)).mul(0.5),
  silu: (x) => x.mul(tf.sigmoid(x)),
  elu: (x) => tf.elu(x),
  mish: (x) => x.mul(tf.tanh(tf.softplus(x))),
  softplus: (x) => tf.softplus(x),
  sigmoid: (x) => tf.sigmoid(x),
  tanh: (x) => tf.tanh(x),
};

function conv1d(x, w, b, cfg) {                       // x (N, C, L), w (out, in/g, k)
  const k = cfg.kernel, d = cfg.dilation || 1, s = cfg.stride || 1, g = cfg.groups || 1;
  const pad = cfg.padding === "valid" ? 0 : (d * (k - 1)) / 2;
  let h = tf.transpose(x, [0, 2, 1]);                     // (N, L, C)
  if (pad) h = tf.pad(h, [[0, 0], [pad, pad], [0, 0]]);
  let kern = tf.transpose(w, [2, 1, 0]);                  // (k, in/g, out)
  if (d > 1) {                                            // TF.js는 dilation의 기울기를 못 함 → 커널 사이에 0을 끼운 같은 커널
    const z = tf.zeros([d - 1, kern.shape[1], kern.shape[2]]), parts = [];
    tf.unstack(kern, 0).forEach((kj, j) => { if (j) parts.push(z); parts.push(kj.expandDims(0)); });
    kern = tf.concat(parts, 0);                           // (d·(k−1)+1, in/g, out)
  }
  const one = (hh, kk) => tf.conv1d(hh, kk, s, "valid", "NWC");
  let y;
  if (g === 1) y = one(h, kern);
  else {                                                  // groups: 채널을 나눠 따로 계산하고 이어 붙임
    const hs = tf.split(h, g, 2), ks = tf.split(kern, g, 2);
    y = tf.concat(hs.map((hh, i) => one(hh, ks[i])), 2);
  }
  y = y.add(b);
  return tf.transpose(y, [0, 2, 1]);
}

function conv2d(x, w, b, cfg) {                       // x (N, C, H, W), w (out, in, k, k)
  const k = cfg.kernel, s = cfg.stride || 1, pad = cfg.padding === "valid" ? 0 : Math.floor(k / 2);
  let h = tf.transpose(x, [0, 2, 3, 1]);
  if (pad) h = tf.pad(h, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
  const y = tf.conv2d(h, tf.transpose(w, [2, 3, 1, 0]), s, "valid").add(b);
  return tf.transpose(y, [0, 3, 1, 2]);
}

function avgPool1d(x, k) {                            // (N, C, L) → (N, C, ⌊L/k⌋)
  const [N, C, L] = x.shape, Lk = Math.floor(L / k);
  return x.slice([0, 0, 0], [N, C, Lk * k]).reshape([N, C, Lk, k]).mean(3);
}

function avgPool2d(x) {                               // 2×2
  const [N, C, H, W] = x.shape, H2 = Math.floor(H / 2), W2 = Math.floor(W / 2);
  return x.slice([0, 0, 0, 0], [N, C, H2 * 2, W2 * 2]).reshape([N, C, H2, 2, W2, 2]).mean([3, 5]);
}

function batchNorm(x, P, key, training, pending) {
  const C = x.shape[1], axes = x.rank === 2 ? [0] : x.rank === 3 ? [0, 2] : [0, 2, 3];
  const bshape = x.rank === 2 ? [1, C] : x.rank === 3 ? [1, C, 1] : [1, C, 1, 1];
  let mean, varr;
  if (training) {
    const m = tf.moments(x, axes);
    mean = m.mean; varr = m.variance;                     // 정규화는 편향 분산 (PyTorch와 같음)
    const n = x.size / C;
    const unbiased = varr.mul(n / Math.max(1, n - 1));
    pending.push([P[key + ".running_mean"], tf.keep(P[key + ".running_mean"].mul(0.9).add(mean.mul(0.1)))]);
    pending.push([P[key + ".running_var"], tf.keep(P[key + ".running_var"].mul(0.9).add(unbiased.mul(0.1)))]);
    pending.push([P[key + ".num_batches_tracked"], null]);
  } else {
    mean = P[key + ".running_mean"]; varr = P[key + ".running_var"];
  }
  return x.sub(mean.reshape(bshape)).div(varr.reshape(bshape).add(1e-5).sqrt())
    .mul(P[key + ".weight"].reshape(bshape)).add(P[key + ".bias"].reshape(bshape));
}

function linear(x, W, b) { return tf.matMul(x, W, false, true).add(b); }

function rnn(x, P, key, cfg, type) {                  // x (N, L, 1) → 마지막 시점의 숨은 값 (N, H)
  const H = cfg.hidden, layers = cfg.layers || 1;
  let seq = x;
  let last;
  for (let l = 0; l < layers; l++) {
    const Wih = P[`${key}.rnn.weight_ih_l${l}`], Whh = P[`${key}.rnn.weight_hh_l${l}`];
    const bih = P[`${key}.rnn.bias_ih_l${l}`], bhh = P[`${key}.rnn.bias_hh_l${l}`];
    const [N, L, I] = seq.shape;
    const xw = tf.matMul(seq.reshape([N * L, I]), Wih, false, true).add(bih).reshape([N, L, -1]);   // 모든 시점의 입력 쪽을 한 번에
    const steps = tf.unstack(xw, 1);
    let h = tf.zeros([N, H]), c = tf.zeros([N, H]);
    const outs = [];
    for (let t = 0; t < L; t++) {
      const hw = tf.matMul(h, Whh, false, true).add(bhh);
      if (type === "lstm") {
        const [i, f, g, o] = tf.split(steps[t].add(hw), 4, 1);       // 순서 i, f, g, o
        c = tf.sigmoid(f).mul(c).add(tf.sigmoid(i).mul(tf.tanh(g)));
        h = tf.sigmoid(o).mul(tf.tanh(c));
      } else if (type === "gru") {
        const [xr, xz, xn] = tf.split(steps[t], 3, 1), [hr, hz, hn] = tf.split(hw, 3, 1);   // 순서 r, z, n
        const r = tf.sigmoid(xr.add(hr)), z = tf.sigmoid(xz.add(hz));
        const n = tf.tanh(xn.add(r.mul(hn)));
        h = tf.scalar(1).sub(z).mul(n).add(z.mul(h));
      } else {
        h = tf.tanh(steps[t].add(hw));
      }
      outs.push(h);
    }
    last = h;
    seq = tf.stack(outs, 1);
  }
  return last;
}

function se(x, P, key, cfg) {
  const s = x.mean(-1);
  const a = ACT[cfg.act || "relu"](linear(s, P[key + ".fc1.weight"], P[key + ".fc1.bias"]));
  const w = tf.sigmoid(linear(a, P[key + ".fc2.weight"], P[key + ".fc2.bias"]));
  return x.mul(w.expandDims(-1));
}

function layer(st, ins, P, training, pending) {
  const c = st.cfg, k = st.key, x = ins[0];
  switch (st.type) {
    case "conv": return conv1d(x, P[k + ".weight"], P[k + ".bias"], c);
    case "conv2d": return conv2d(x, P[k + ".weight"], P[k + ".bias"], c);
    case "pool": return avgPool1d(x, c.size);
    case "pool2d": return avgPool2d(x);
    case "global_avg": return x.mean(-1);
    case "global_avg2d": return x.mean([2, 3]);
    case "last_step": { const L = x.shape[2]; return x.slice([0, 0, L - 1], [-1, -1, 1]).squeeze([2]); }
    case "bn": return batchNorm(x, P, k, training, pending);
    case "act": return ACT[c.kind](x);
    case "dropout": return training && (c.rate || 0.5) > 0 ? tf.dropout(x, c.rate ?? 0.5) : x;
    case "flatten": return x.reshape([x.shape[0], -1]);
    case "dense": case "output": {
      const flat = st.in_shape.length >= 2, kk = k + (flat ? ".1" : "");
      return linear(flat ? x.reshape([x.shape[0], -1]) : x, P[kk + ".weight"], P[kk + ".bias"]);
    }
    case "lstm": case "gru": case "simple_rnn": return rnn(x, P, k, c, st.type);
    case "se": return se(x, P, k, c);
    case "add": return ins.slice(1).reduce((a, b) => a.add(b), ins[0]);
    case "concat": return tf.concat(ins, 1);
    default: throw new Error("모르는 층: " + st.type);
  }
}

// ------------------------------------------------------------ 모델
class Net {
  constructor(plan, seed = 0) {
    this.plan = plan;
    this.kind = plan.kind;
    this.P = {};                                          // 이름 → tf.Variable (PyTorch state_dict 이름)
    this.trainable = [];
    this.nameOf = {};                                     // TF.js 변수 이름 → PyTorch 이름
    const rand = rng(seed * 7919 + 17);
    for (const st of plan.steps) {
      for (const [name, shape, kind] of st.params) {
        // TF.js 변수 이름은 전체에서 겹치면 안 됨 → 자동 이름, 우리 이름은 nameOf로
        const v = tf.variable(initParam(st, name, shape, kind, rand), kind === "weight", undefined, kind === "count" ? "int32" : "float32");
        this.P[name] = v;
        this.nameOf[v.name] = name;
        if (kind === "weight") this.trainable.push(v);
      }
    }
  }

  forward(x, training = false, pending = [], record = null) {
    const vals = { x };
    for (const st of this.plan.steps) {
      const y = layer(st, st.inputs.map((n) => vals[n]), this.P, training, pending);
      vals[st.name] = y;
      if (record) record[st.node] = y;
    }
    if (record) {
      for (const [rid, nm] of Object.entries(this.plan.aliases || {})) record[rid] = vals[nm];
    }
    return vals[this.plan.out];
  }

  predict(X, batch = 256) {                           // eval, 묶음으로 나눠 계산 → Float32Array (N × n_out)
    const N = X.shape[0], out = [];
    for (let i = 0; i < N; i += batch) {
      const y = tf.tidy(() => this.forward(X.slice(i, Math.min(batch, N - i)), false));
      out.push(y.dataSync().slice());
      y.dispose();
    }
    const flat = new Float32Array(out.reduce((a, b) => a + b.length, 0));
    let o = 0;
    for (const a of out) { flat.set(a, o); o += a.length; }
    return flat;
  }

  trace(x1) {                                         // 샘플 하나 → 노드 id마다 값 (배치 차원 뺀 중첩 배열)
    return tf.tidy(() => {
      const rec = {};
      this.forward(x1, false, [], rec);
      const out = {};
      for (const [k, v] of Object.entries(rec)) out[k] = v.squeeze([0]).arraySync();
      return out;
    });
  }

  numParams() { return this.trainable.reduce((a, v) => a + v.size, 0); }

  state() {                                           // 이름 → {shape, data(Float32Array·Int32Array)}
    const out = {};
    for (const [k, v] of Object.entries(this.P)) out[k] = { shape: v.shape, data: v.dataSync().slice() };
    return out;
  }

  load(state) {                                       // {이름: {shape, data}} (모양이 맞아야 함)
    for (const [k, v] of Object.entries(this.P)) {
      const s = state[k];
      if (!s) throw new Error("가중치가 없음: " + k);
      if (prod(s.shape) !== v.size) throw new Error(`가중치 모양이 다름: ${k}`);
      v.assign(tf.tensor(s.data, v.shape, v.dtype));
    }
  }

  dispose() { Object.values(this.P).forEach((v) => v.dispose()); }
}

// ------------------------------------------------------------ 데이터 모양: (n, 특징) → 모델 입력
function toInput(X, n, kind, length, image) {         // X: Float32Array (n × 특징)
  if (kind === "image") return tf.tensor4d(X, [n, 1, image[0], image[1]]);
  if (kind === "rnn") return tf.tensor3d(X, [n, length, 1]);
  return tf.tensor3d(X, [n, 1, length]);
}

// ------------------------------------------------------------ 지표 (jobs.py와 같음)
function metric(out, y, task, nOut) {
  const n = y.length;
  if (task === "classification") {
    let ok = 0;
    for (let i = 0; i < n; i++) ok += argmax(out, i, nOut) === y[i] ? 1 : 0;
    return ok / n;
  }
  let mu = 0; for (let i = 0; i < n; i++) mu += y[i]; mu /= n;
  let ss = 0, se = 0;
  for (let i = 0; i < n; i++) { ss += (y[i] - mu) ** 2; se += (out[i * nOut] - y[i]) ** 2; }
  return ss > 0 ? 1 - se / ss : NaN;
}
function argmax(out, i, k) { let b = 0; for (let j = 1; j < k; j++) if (out[i * k + j] > out[i * k + b]) b = j; return b; }
function confusion(out, y, k) {
  const cm = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < y.length; i++) cm[y[i]][argmax(out, i, k)]++;
  const rec = cm.map((r, i) => { const s = r.reduce((a, b) => a + b, 0); return s ? r[i] / s : null; });
  return [cm, rec];
}
function regressionView(out, y, nOut) {
  let mae = 0;
  for (let i = 0; i < y.length; i++) mae += Math.abs(out[i * nOut] - y[i]);
  const k = Math.min(200, y.length), ys = [], ps = [];
  for (let i = 0; i < k; i++) { ys.push(y[i]); ps.push(out[i * nOut]); }
  return [mae / y.length, { y: ys, pred: ps }];
}

// ------------------------------------------------------------ 학습 (jobs.fit·train_job과 같은 흐름)
// d: {X_train, X_test (Float32Array), y_train, y_test (Float32Array·Int32Array), n_train, n_test, task, n_out, length, image}
// s: {epochs, lr, batch_size, seed}. onEpoch({epoch, loss, test_metric}) → true면 멈춤.
async function train(plan, d, s, onEpoch, isCancelled) {
  const net = new Net(plan, s.seed || 0);
  const reg = d.task !== "classification";
  let mu = 0, sd = 1;
  if (reg) {
    for (const v of d.y_train) mu += v; mu /= d.n_train;
    let q = 0; for (const v of d.y_train) q += (v - mu) ** 2;
    sd = Math.sqrt(q / d.n_train) || 1;
  }
  const Xtr = toInput(d.X_train, d.n_train, plan.kind, d.length, d.image);
  const Xte = toInput(d.X_test, d.n_test, plan.kind, d.length, d.image);
  const ytr = reg ? tf.tensor2d(Float32Array.from(d.y_train, (v) => (v - mu) / sd), [d.n_train, 1])
    : tf.oneHot(tf.tensor1d(Int32Array.from(d.y_train), "int32"), d.n_out).toFloat();
  const opt = tf.train.adam(s.lr || 1e-3, 0.9, 0.999, 1e-8);
  const rand = rng((s.seed || 0) * 104729 + 3);
  const bs = Math.max(1, s.batch_size || 32), n = d.n_train;
  let done = 0;
  try {
    for (let epoch = 1; epoch <= (s.epochs || 30); epoch++) {
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      let lossSum = 0, nb = 0;
      for (let i = 0; i < n; i += bs) {
        const idx = perm.slice(i, i + bs);
        if (idx.length === 1 && n > 1) continue;            // BatchNorm은 학습 때 샘플 하나짜리 묶음을 못 받음
        const pending = [];
        const loss = tf.tidy(() => {
          const ii = tf.tensor1d(idx, "int32");
          const xb = tf.gather(Xtr, ii), yb = tf.gather(ytr, ii);
          return opt.minimize(() => {
            const out = net.forward(xb, true, pending);
            return reg ? tf.losses.meanSquaredError(yb, out) : tf.losses.softmaxCrossEntropy(yb, out);
          }, true, net.trainable);
        });
        for (const [buf, val] of pending) {
          if (val) { buf.assign(val); val.dispose(); } else buf.assign(buf.add(1));
        }
        lossSum += (await loss.data())[0];
        loss.dispose();
        nb++;
        if (nb % 8 === 0) await yieldNow();                   // 화면이 멈추지 않게
      }
      const out = net.predict(Xte).map((v) => (reg ? v * sd + mu : v));
      done = epoch;
      const stop = onEpoch && onEpoch({ epoch, loss: lossSum / Math.max(1, nb), test_metric: metric(out, d.y_test, d.task, d.n_out) });
      if (stop || (isCancelled && isCancelled())) break;
      await yieldNow();
    }
    if (reg) {                                              // 정규화한 정답으로 배운 출력층에 되돌림을 접어 넣음 (원래 단위로 냄)
      const last = plan.steps.find((st) => st.name === plan.out);
      const kk = last.key + (last.in_shape.length >= 2 ? ".1" : "");
      const W = net.P[kk + ".weight"], b = net.P[kk + ".bias"];
      W.assign(W.mul(sd)); b.assign(b.mul(sd).add(mu));
    }
    const out = net.predict(Xte);
    const r = { epochs_done: done, test_metric: metric(out, d.y_test, d.task, d.n_out),
      metric_name: reg ? "r2" : "accuracy", warnings: [], confusion: null, recall: null };
    if (reg) [r.mae, r.scatter] = regressionView(out, d.y_test, d.n_out);
    else [r.confusion, r.recall] = confusion(out, d.y_test, d.n_out);
    return { net, result: r };
  } finally {
    [Xtr, Xte, ytr].forEach((t) => t.dispose());
    opt.dispose();
  }
}

// 샘플 따라가기 값: (C, L)이 200점보다 길면 고르게 200점, 소수 다섯째 자리 (analysis._thin과 같음)
function thin(v) {
  const r5 = (x) => Math.round(x * 1e5) / 1e5;
  if (!Array.isArray(v)) return r5(v);
  if (Array.isArray(v[0]) && !Array.isArray(v[0][0]) && v[0].length > 200) {
    const L = v[0].length, idx = Array.from({ length: 200 }, (_, i) => Math.round((i * (L - 1)) / 199));
    return v.map((row) => idx.map((j) => r5(row[j])));
  }
  return v.map(thin);
}

const API = { Net, train, toInput, metric, confusion, regressionView, thin, ACT, tf };
if (typeof module !== "undefined") module.exports = API;
if (typeof window !== "undefined") window.DLEngine = API;
})();

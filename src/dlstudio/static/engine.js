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
function initParam(step, name, shape, kind, rand, init) {
  const n = prod(shape);
  if (kind === "count") return tf.scalar(0, "int32");
  if (init === "ones") return tf.ones(shape);
  if (init === "zeros") return tf.zeros(shape);
  if (init === "const0.25") return tf.fill(shape, 0.25);
  if (init === "normal1" || init === "normal02") {               // Box–Muller (seed로 같은 값)
    const sd = init === "normal1" ? 1 : 0.02, v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand()) * sd;
    return tf.tensor(v, shape);
  }
  const leaf = name.split(".").pop();
  if (step.type === "bn") {
    if (leaf === "weight" || leaf === "running_var") return tf.ones(shape);
    return tf.zeros(shape);
  }
  let bound;
  if (["lstm", "gru", "simple_rnn"].includes(step.type)) bound = 1 / Math.sqrt(step.cfg.hidden);
  else if (step.type === "conv_t" || step.type === "conv_t2d") bound = 1 / Math.sqrt(prod(step.params[0][1].slice(1)));
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
  selu: (x) => tf.selu(x),
  relu6: (x) => tf.relu6(x),
  hardswish: (x) => x.mul(tf.relu6(x.add(3))).div(6),
  hardsigmoid: (x) => tf.relu6(x.add(3)).div(6),
  softmax: (x) => (x.rank === 2 ? tf.softmax(x) : chFirst(tf.softmax(chLast(x)))),          // 채널 방향 (PyTorch dim=1)
};

function conv1d(x, w, b, cfg) {                       // x (N, C, L), w (out, in/g, k)
  const k = cfg.kernel, d = cfg.dilation || 1, s = cfg.stride || 1, g = cfg.groups || 1;
  const pad = cfg.padding === "valid" ? 0 : cfg.padding === "causal" ? 0 : Math.floor((d * (k - 1)) / 2);
  let h = tf.transpose(x, [0, 2, 1]);                     // (N, L, C)
  if (pad) h = tf.pad(h, [[0, 0], [pad, pad], [0, 0]]);
  if (cfg.padding === "causal" && d * (k - 1)) h = tf.pad(h, [[0, 0], [d * (k - 1), 0], [0, 0]]);   // 과거만 봄
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

function conv2d(x, w, b, cfg) {                       // x (N, C, H, W), w (out, in/g, k, k)
  const k = cfg.kernel, s = cfg.stride || 1, d = cfg.dilation || 1, g = cfg.groups || 1;
  const pad = cfg.padding === "valid" ? 0 : Math.floor((d * (k - 1)) / 2);
  let h = tf.transpose(x, [0, 2, 3, 1]);
  if (pad) h = tf.pad(h, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
  let kern = tf.transpose(w, [2, 3, 1, 0]);               // (k, k, in/g, out)
  if (d > 1) {                                            // dilation: 커널 칸 사이에 0을 끼움 (기울기가 되는 연산만)
    const [kh, kw, ci, co] = kern.shape;
    kern = kern.reshape([kh, 1, kw, 1, ci, co]).pad([[0, 0], [0, d - 1], [0, 0], [0, d - 1], [0, 0], [0, 0]])
      .reshape([kh * d, kw * d, ci, co]).slice([0, 0, 0, 0], [d * (kh - 1) + 1, d * (kw - 1) + 1, ci, co]);
  }
  const one = (hh, kk) => tf.conv2d(hh, kk, s, "valid");
  let y;
  if (g === 1) y = one(h, kern);
  else { const hs = tf.split(h, g, 3), ks = tf.split(kern, g, 3); y = tf.concat(hs.map((hh, i) => one(hh, ks[i])), 3); }
  return tf.transpose(y.add(b), [0, 3, 1, 2]);
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

function rnn(x, P, key, cfg, type) {                  // x (N, C, L) → 마지막 숨은 값 (N, H·방향) 또는 줄 (N, H·방향, L)
  const H = cfg.hidden, layers = cfg.layers || 1, bi = !!cfg.bidirectional;
  let seq = tf.transpose(x, [0, 2, 1]);                    // (N, L, C)
  for (let l = 0; l < layers; l++) {
    const f = rnnDir(seq, P, `${key}.rnn`, `l${l}`, H, type);
    seq = bi ? tf.concat([f, tf.reverse(rnnDir(tf.reverse(seq, 1), P, `${key}.rnn`, `l${l}_reverse`, H, type), 1)], 2) : f;
  }
  const L = seq.shape[1];
  if (cfg.return_sequences) return tf.transpose(seq, [0, 2, 1]);
  if (bi) return tf.concat([seq.slice([0, L - 1, 0], [-1, 1, H]).squeeze([1]), seq.slice([0, 0, H], [-1, 1, H]).squeeze([1])], 1);
  return seq.slice([0, L - 1, 0], [-1, 1, -1]).squeeze([1]);
}
function rnnDir(seq, P, pre, sfx, H, type) {         // 한 방향 한 층: (N, L, I) → (N, L, H)
  {
    const Wih = P[`${pre}.weight_ih_${sfx}`], Whh = P[`${pre}.weight_hh_${sfx}`];
    const bih = P[`${pre}.bias_ih_${sfx}`], bhh = P[`${pre}.bias_hh_${sfx}`];
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
    return tf.stack(outs, 1);
  }
}

function se(x, P, key, cfg) {
  const s = spatialOf(x).mean(-1);
  const a = ACT[cfg.act || "relu"](linear(s, P[key + ".fc1.weight"], P[key + ".fc1.bias"]));
  const w = tf.sigmoid(linear(a, P[key + ".fc2.weight"], P[key + ".fc2.bias"]));
  return x.mul(chanW(w, x));
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
    case "output":
      if (c.head === "grid") {                                // 탐지: 격자 칸마다 1×1 필터 → (값, G, G)를 펼침 (PyTorch와 같은 순서)
        const y = conv2d(x, P[k + ".0.weight"], P[k + ".0.bias"], { kernel: 1 });
        return y.reshape([y.shape[0], -1]);
      }
      // falls through
    case "dense": {
      const flat = st.in_shape.length >= 2, kk = k + (flat ? ".1" : "");
      return linear(flat ? x.reshape([x.shape[0], -1]) : x, P[kk + ".weight"], P[kk + ".bias"]);
    }
    case "lstm": case "gru": case "simple_rnn": return rnn(x, P, k, c, st.type);
    case "se": return se(x, P, k, c);
    case "add": return ins.slice(1).reduce((a, b) => a.add(b), ins[0]);
    case "concat": return tf.concat(ins, 1);
    case "multiply": return ins.slice(1).reduce((a, b) => a.mul(b), ins[0]);
    case "average": return ins.slice(1).reduce((a, b) => a.add(b), ins[0]).div(ins.length);
    case "maximum": return ins.slice(1).reduce((a, b) => tf.maximum(a, b), ins[0]);
    default: {
      const f = EXTRA[st.type];
      if (!f) throw new Error("모르는 층: " + st.type);
      return f(x, st, P, k, training, ins);
    }
  }
}

// ------------------------------------------------------------ 추가 블록 (nn_extra.py와 같은 식). x는 채널이 먼저: (N, C, L) · (N, C, H, W)
const spatialOf = (x) => (x.rank === 2 ? x.expandDims(-1) : x.reshape([x.shape[0], x.shape[1], -1]));   // (N, C, 위치 수)
const chanW = (w, x) => w.reshape([w.shape[0], w.shape[1]].concat(new Array(x.rank - 2).fill(1)));      // (N, C) → x에 곱할 모양
function chLast(x) { return x.rank === 2 ? x : tf.transpose(x, x.rank === 3 ? [0, 2, 1] : [0, 2, 3, 1]); }
function chFirst(y) { return y.rank === 2 ? y : tf.transpose(y, y.rank === 3 ? [0, 2, 1] : [0, 3, 1, 2]); }
function linearCh(x, W, b) {                          // 위치마다 채널에 Linear (PyTorch: x.movedim(1, -1) → Linear)
  if (x.rank === 2) return linear(x, W, b);
  const h = chLast(x), sh = h.shape;
  const y = linear(h.reshape([-1, sh[sh.length - 1]]), W, b);
  return chFirst(y.reshape(sh.slice(0, -1).concat([W.shape[0]])));
}
const toSeq = (x) => tf.transpose(spatialOf(x), [0, 2, 1]);                      // (N, T, C)
const fromSeq = (y, shape) => tf.transpose(y, [0, 2, 1]).reshape([shape[0], y.shape[2]].concat(shape.slice(2)));
function splitHeads(z, h) { const [n, t, c] = z.shape; return z.reshape([n, t, h, c / h]).transpose([0, 2, 1, 3]); }
const joinHeads = (y) => { const [n, h, t, d] = y.shape; return y.transpose([0, 2, 1, 3]).reshape([n, t, h * d]); };
function rope(x) {                                     // (N, heads, T, d): 짝 (2i, 2i+1)을 위치 × 10000^(−2i/d)만큼 돌림
  const [n, h, T, d] = x.shape, ang = new Float32Array(T * (d / 2));
  for (let p = 0; p < T; p++) for (let i = 0; i < d / 2; i++) ang[p * (d / 2) + i] = p * Math.pow(10000, -(2 * i) / d);
  const a = tf.tensor(ang, [T, d / 2]), cos = tf.cos(a), sin = tf.sin(a);
  const [x1, x2] = tf.unstack(x.reshape([n, h, T, d / 2, 2]), 4);
  return tf.stack([x1.mul(cos).sub(x2.mul(sin)), x1.mul(sin).add(x2.mul(cos))], 4).reshape([n, h, T, d]);
}
// 샘플 따라가기의 어텐션 지도: trace(x, attn)가 도는 동안만 블록마다 softmax 가중치(head 평균)를 받아 둠
let ATT = null, ATT_NODE = null;
const keepAttn = (w) => { if (ATT && ATT_NODE) ATT[ATT_NODE] = w.slice([0], [1]).squeeze([0]); };   // 배치 첫 샘플
function attend(q, k, v, causal) {                    // (N, h, T, d) → softmax(QKᵀ/√d)V
  let att = tf.matMul(q, k, false, true).mul(1 / Math.sqrt(q.shape[3]));
  if (causal) {
    const T = att.shape[2], m = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);     // 아래 삼각(자기 포함) = 1
    att = att.mul(m).add(m.sub(1).mul(1e9));                                       // 미래 = −1e9 (PyTorch masked_fill과 같은 값)
  }
  const w = tf.softmax(att);
  if (ATT) keepAttn(w.mean(1));
  return tf.matMul(w, v);
}
function qkv(x, P, k, heads) {
  const s = toSeq(x), z = linear(s.reshape([-1, s.shape[2]]), P[k + ".qkv.weight"], P[k + ".qkv.bias"]).reshape([s.shape[0], s.shape[1], -1]);
  return tf.split(z, 3, 2).map((t) => splitHeads(t, heads));
}
const proj = (y, x, P, k) => {                         // (N, T, C) → Linear → 원래 모양
  const o = linear(y.reshape([-1, y.shape[2]]), P[k + ".proj.weight"], P[k + ".proj.bias"]).reshape(y.shape);
  return fromSeq(o, x.shape);
};
function poolMat(L, out) { return Array.from({ length: out }, (_, i) => [Math.floor((i * L) / out), Math.ceil(((i + 1) * L) / out)]); }
function adaptive(x, size, mode) {                    // PyTorch Adaptive*Pool과 같은 구간 [⌊iL/n⌋, ⌈(i+1)L/n⌉)
  const red = (t, ax) => (mode === "max" ? t.max(ax, true) : t.mean(ax, true));
  const along = (t, ax) => tf.concat(poolMat(t.shape[ax], size).map(([a, b]) => {
    const begin = new Array(t.rank).fill(0), sz = t.shape.slice();
    begin[ax] = a; sz[ax] = b - a;
    return red(t.slice(begin, sz), ax);
  }), ax);
  let y = along(x, 2);
  if (x.rank === 4) y = along(y, 3);
  return y;
}
function interpMat(L, out) {                          // 선형 보간 (align_corners=False): (L, out) 행렬
  const M = new Float32Array(L * out);
  for (let i = 0; i < out; i++) {
    const src = Math.max(0, ((i + 0.5) * L) / out - 0.5), i0 = Math.min(Math.floor(src), L - 1), i1 = Math.min(i0 + 1, L - 1), w1 = src - i0;
    M[i0 * out + i] += 1 - w1; M[i1 * out + i] += w1;
  }
  return tf.tensor(M, [L, out]);
}
function alongLast(x, M) { const sh = x.shape; return tf.matMul(x.reshape([-1, sh[sh.length - 1]]), M).reshape(sh.slice(0, -1).concat([M.shape[1]])); }
function convT(x, W, b, cfg, two) {                   // ConvTranspose: 입력 칸 사이에 0 (stride−1개) → 양쪽 채움 → 뒤집은 커널로 보통 Conv
  const k = cfg.kernel, s = cfg.stride || 2;
  const [p, op] = s === 2 ? ({ 1: [0, 1], 2: [0, 0], 3: [1, 1], 4: [1, 0], 5: [2, 1] })[k] : [(k - 1) / 2, 0];
  const dil = (t, ax) => {                             // ax 축에 0 끼우기
    if (s === 1) return t;
    const sh = t.shape, n = sh[ax];
    const e = t.expandDims(ax + 1), padv = new Array(e.rank).fill([0, 0]);
    padv[ax + 1] = [0, s - 1];
    const r = e.pad(padv), ns = sh.slice(); ns[ax] = n * s;
    const begin = new Array(sh.length).fill(0), size = ns.slice(); size[ax] = (n - 1) * s + 1;
    return r.reshape(ns).slice(begin, size);
  };
  let h = dil(x, 2);
  if (two) h = dil(h, 3);
  const lo = k - 1 - p, hi = k - 1 - p + op, padv = [[0, 0], [0, 0], [lo, hi]].concat(two ? [[lo, hi]] : []);
  h = h.pad(padv);
  const kern = tf.reverse(W, two ? [2, 3] : [2]).transpose(two ? [1, 0, 2, 3] : [1, 0, 2]);   // (out, in, k…)
  return two ? conv2d(h, kern, b, { kernel: k, padding: "valid" }) : conv1d(h, kern, b, { kernel: k, padding: "valid" });
}
function sinTable(c, n) {
  const v = new Float32Array(c * n);
  for (let ch = 0; ch < c; ch++) for (let p = 0; p < n; p++) {
    const a = p / Math.pow(10000, (ch - (ch % 2)) / c);
    v[ch * n + p] = ch % 2 === 0 ? Math.sin(a) : Math.cos(a);
  }
  return tf.tensor(v, [c, n]);
}
function specMats(nFft, nMels, nMfcc) {               // Spectrogram 고정 표 (nn_extra.Spectrogram과 같은 식)
  const nf = Math.floor(nFft / 2) + 1, cosk = new Float32Array(nf * nFft), sink = new Float32Array(nf * nFft);
  for (let f = 0; f < nf; f++) for (let n = 0; n < nFft; n++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / nFft), a = (2 * Math.PI * f * n) / nFft;
    cosk[f * nFft + n] = w * Math.cos(a); sink[f * nFft + n] = w * Math.sin(a);
  }
  const sr = 16000, mel = (v) => 2595 * Math.log10(1 + v / 700), hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
  const pts = Array.from({ length: nMels + 2 }, (_, i) => hz((mel(sr / 2) * i) / (nMels + 1)));
  const fb = new Float32Array(nMels * nf);
  for (let m = 0; m < nMels; m++) for (let j = 0; j < nf; j++) {
    const fr = (sr / 2) * j / (nf - 1);
    const up = pts[m + 1] > pts[m] ? (fr - pts[m]) / (pts[m + 1] - pts[m]) : 0, down = pts[m + 2] > pts[m + 1] ? (pts[m + 2] - fr) / (pts[m + 2] - pts[m + 1]) : 0;
    fb[m * nf + j] = Math.max(0, Math.min(up, down));
  }
  const dct = new Float32Array(nMfcc * nMels);
  for (let q = 0; q < nMfcc; q++) for (let m = 0; m < nMels; m++) dct[q * nMels + m] = Math.sqrt((q === 0 ? 1 : 2) / nMels) * Math.cos((Math.PI / nMels) * (m + 0.5) * q);
  return { cos: tf.tensor(cosk, [nf, 1, nFft]), sin: tf.tensor(sink, [nf, 1, nFft]), mel: tf.tensor(fb, [nMels, nf]), dct: tf.tensor(dct, [nMfcc, nMels]), nf };
}
const uni = (n) => tf.randomUniform([n]);
function perSample(x, fn) { return tf.stack(tf.unstack(x, 0).map((xi, i) => fn(xi, i))); }
function imgTransform(x, mats) {                      // 그림 (N, C, H, W)에 샘플마다 변환 (출력 칸 → 입력 칸, 픽셀 단위 8값)
  return tf.transpose(tf.image.transform(tf.transpose(x, [0, 2, 3, 1]), tf.tensor2d(mats), "bilinear", "constant"), [0, 3, 1, 2]);
}
const EXTRA = {
  glu: (x, st, P, k) => {
    const h = linearCh(x, P[k + ".proj.weight"], P[k + ".proj.bias"]);
    const [a, g] = tf.split(h, 2, 1);
    return a.mul(st.cfg.gate === "silu" ? g.mul(tf.sigmoid(g)) : tf.sigmoid(g));
  },
  dw_conv: (x, st, P, k) => conv1d(x, P[k + ".weight"], P[k + ".bias"], { kernel: st.cfg.kernel || 3, groups: x.shape[1] }),
  dw_conv2d: (x, st, P, k) => conv2d(x, P[k + ".weight"], P[k + ".bias"], { kernel: st.cfg.kernel || 3, groups: x.shape[1] }),
  sep_conv: (x, st, P, k) => conv1d(conv1d(x, P[k + ".depthwise.weight"], P[k + ".depthwise.bias"], { kernel: st.cfg.kernel || 3, groups: x.shape[1] }),
    P[k + ".pointwise.weight"], P[k + ".pointwise.bias"], { kernel: 1 }),
  sep_conv2d: (x, st, P, k) => conv2d(conv2d(x, P[k + ".depthwise.weight"], P[k + ".depthwise.bias"], { kernel: st.cfg.kernel || 3, groups: x.shape[1] }),
    P[k + ".pointwise.weight"], P[k + ".pointwise.bias"], { kernel: 1 }),
  max_pool: (x, st) => { const s = st.cfg.size || 2, [N, C, L] = x.shape, Lk = Math.floor(L / s); return x.slice([0, 0, 0], [N, C, Lk * s]).reshape([N, C, Lk, s]).max(3); },
  max_pool2d: (x) => { const [N, C, H, W] = x.shape, H2 = Math.floor(H / 2), W2 = Math.floor(W / 2); return x.slice([0, 0, 0, 0], [N, C, H2 * 2, W2 * 2]).reshape([N, C, H2, 2, W2, 2]).max([3, 5]); },
  global_max: (x) => spatialOf(x).max(-1),
  global_max2d: (x) => spatialOf(x).max(-1),
  adaptive_pool: (x, st) => adaptive(x, st.cfg.size || 4, st.cfg.mode || "avg"),
  first_step: (x) => x.slice([0, 0, 0], [-1, -1, 1]).squeeze([2]),
  attn_pool: (x, st, P, k) => {
    const s = spatialOf(x), sc = linearCh(s, P[k + ".score.weight"], P[k + ".score.bias"]).squeeze([1]);   // (N, T)
    const w = tf.softmax(sc);
    if (ATT) keepAttn(w.expandDims(1));
    return s.mul(w.expandDims(1)).sum(-1);
  },
  upsample: (x, st) => {
    const sp = x.shape.slice(2);
    if ((st.cfg.mode || "nearest") === "nearest") {
      const dbl = (t, ax) => { const sh = t.shape.slice(); sh[ax] *= 2; return tf.stack([t, t], ax + 1).reshape(sh); };   // 칸마다 두 번
      return x.rank === 3 ? dbl(x, 2) : dbl(dbl(x, 3), 2);
    }
    if (x.rank === 3) return alongLast(x, interpMat(sp[0], sp[0] * 2));
    const w = alongLast(x, interpMat(sp[1], sp[1] * 2));                       // 가로 → 세로 (bilinear는 축마다 나눠 됨)
    return tf.transpose(alongLast(tf.transpose(w, [0, 1, 3, 2]), interpMat(sp[0], sp[0] * 2)), [0, 1, 3, 2]);
  },
  pixel_shuffle: (x) => {
    if (x.rank === 3) { const [N, C, L] = x.shape; return x.reshape([N, C / 2, 2, L]).transpose([0, 1, 3, 2]).reshape([N, C / 2, L * 2]); }
    const [N, C, H, W] = x.shape;
    return x.reshape([N, C / 4, 2, 2, H, W]).transpose([0, 1, 4, 2, 5, 3]).reshape([N, C / 4, H * 2, W * 2]);
  },
  conv_t: (x, st, P, k) => convT(x, P[k + ".weight"], P[k + ".bias"], st.cfg, false),
  conv_t2d: (x, st, P, k) => convT(x, P[k + ".weight"], P[k + ".bias"], st.cfg, true),
  eca: (x, st, P, k) => {
    const s = spatialOf(x).mean(-1).expandDims(1);                                  // (N, 1, C)
    const kk = st.cfg.kernel || 3, y = conv1d(s, P[k + ".conv.weight"], tf.zeros([1]), { kernel: kk });
    return x.mul(chanW(tf.sigmoid(y.squeeze([1])), x));
  },
  spatial_attn: (x, st, P, k) => spatialPart(x, st, P, k),
  cbam: (x, st, P, k) => {
    const f = spatialOf(x), mlp = (v) => linear(tf.relu(linear(v, P[k + ".fc1.weight"], P[k + ".fc1.bias"])), P[k + ".fc2.weight"], P[k + ".fc2.bias"]);
    const ca = tf.sigmoid(mlp(f.mean(-1)).add(mlp(f.max(-1))));
    return spatialPart(x.mul(chanW(ca, x)), st, P, k);
  },
  mha: (x, st, P, k) => {
    let [q, kk, v] = qkv(x, P, k, st.cfg.heads || 2);
    if (st.cfg.pos === "rope") { q = rope(q); kk = rope(kk); }
    return proj(joinHeads(attend(q, kk, v, !!st.cfg.causal)), x, P, k);
  },
  linear_attn: (x, st, P, k) => {
    let [q, kk, v] = qkv(x, P, k, st.cfg.heads || 2);
    q = tf.elu(q).add(1); kk = tf.elu(kk).add(1);
    const kv = tf.matMul(kk, v, true, false), z = tf.scalar(1).div(tf.matMul(q, kk.sum(2, true), false, true).add(1e-6));
    return proj(joinHeads(tf.matMul(q, kv).mul(z)), x, P, k);
  },
  nonlocal: (x, st, P, k) => {
    const s = toSeq(x), L = (n) => linearCh(tf.transpose(s, [0, 2, 1]), P[k + `.${n}.weight`], P[k + `.${n}.bias`]);   // (N, h, T)
    const th = L("theta"), ph = L("phi"), g = L("g");
    const att = tf.softmax(tf.matMul(th, ph, true, false));                     // (N, T, T)
    if (ATT) keepAttn(att);
    const y = tf.matMul(att, g, false, true);                                    // (N, T, h)
    const o = linear(y.reshape([-1, y.shape[2]]), P[k + ".out.weight"], P[k + ".out.bias"]).reshape([y.shape[0], y.shape[1], -1]);
    return x.add(fromSeq(o, x.shape));
  },
  cross_attn: (x, st, P, k, training, ins) => {
    const h = st.cfg.heads || 2, s = toSeq(x), r = toSeq(ins[1]);
    const lin = (t, n) => linear(t.reshape([-1, t.shape[2]]), P[k + `.${n}.weight`], P[k + `.${n}.bias`]).reshape([t.shape[0], t.shape[1], -1]);
    const [kk, v] = tf.split(lin(r, "kv"), 2, 2);
    return proj(joinHeads(attend(splitHeads(lin(s, "q"), h), splitHeads(kk, h), splitHeads(v, h), false)), x, P, k);
  },
  pos_enc: (x, st, P, k) => {
    const c = x.shape[1], n = prod(x.shape.slice(2));
    const pe = st.cfg.mode === "learned" ? P[k + ".pe"] : sinTable(c, n);
    return x.add(pe.reshape(x.shape.slice(1)));
  },
  patch_embed: (x, st, P, k) => {
    const p = st.cfg.patch || 4, y = conv2d(x, P[k + ".proj.weight"], P[k + ".proj.bias"], { kernel: p, stride: p, padding: "valid" });
    return y.reshape([y.shape[0], y.shape[1], -1]);
  },
  cls_token: (x, st, P, k) => tf.concat([P[k + ".cls"].reshape([1, -1, 1]).tile([x.shape[0], 1, 1]), x], 2),
  layernorm: (x, st, P, k) => {
    const h = chLast(x), m = tf.moments(h, -1, true);
    return chFirst(h.sub(m.mean).div(m.variance.add(1e-5).sqrt()).mul(P[k + ".weight"]).add(P[k + ".bias"]));
  },
  rmsnorm: (x, st, P, k) => x.mul(tf.rsqrt(x.square().mean(1, true).add(1e-5))).mul(P[k + ".weight"].reshape([-1].concat(new Array(x.rank - 2).fill(1)))),
  groupnorm: (x, st, P, k) => groupNorm(x, st.cfg.groups || 4, P, k),
  instancenorm: (x, st, P, k) => groupNorm(x, x.shape[1], P, k),
  prelu: (x, st, P, k) => tf.relu(x).add(P[k + ".weight"].mul(tf.minimum(x, 0))),
  spatial_dropout: (x, st, P, k, training) => {
    const r = st.cfg.rate ?? 0.1;
    if (!training || !r) return x;
    const keep = tf.randomUniform([x.shape[0], x.shape[1]]).greaterEqual(r).toFloat().div(1 - r);
    return x.mul(chanW(keep, x));
  },
  drop_path: (x, st, P, k, training) => {
    const r = st.cfg.rate ?? 0.1;
    if (!training || !r) return x;
    return x.mul(uni(x.shape[0]).greaterEqual(r).toFloat().div(1 - r).reshape([x.shape[0]].concat(new Array(x.rank - 1).fill(1))));
  },
  gaussian_noise: (x, st, P, k, training) => (training ? x.add(tf.randomNormal(x.shape).mul(st.cfg.std ?? 0.1)) : x),
  slice: (x, st) => { const b = new Array(x.rank).fill(0), s = new Array(x.rank).fill(-1); b[1] = st.cfg.start || 0; s[1] = st.cfg.count; return x.slice(b, s); },
  reshape: (x, st) => x.reshape([x.shape[0], st.cfg.channels, -1]),
  permute: (x) => tf.transpose(x, x.rank === 3 ? [0, 2, 1] : [0, 1, 3, 2]),
  to_seq: (x) => x.reshape([x.shape[0], x.shape[1], -1]),
  crop: (x, st) => { const a = st.cfg.amount || 1, b = [0, 0].concat(x.shape.slice(2).map(() => a)), s = [-1, -1].concat(x.shape.slice(2).map((v) => v - 2 * a)); return x.slice(b, s); },
  pad: (x, st) => { const a = st.cfg.amount || 1; return x.pad([[0, 0], [0, 0]].concat(x.shape.slice(2).map(() => [a, a]))); },
  embedding: (x, st, P, k) => {
    const bins = st.cfg.bins || 32, idx = tf.clipByValue(tf.floor(x.squeeze([1]).add(3).div(6).mul(bins)), 0, bins - 1).toInt();
    const e = tf.gather(P[k + ".weight"], idx.reshape([-1])).reshape(idx.shape.concat([-1]));   // (N, …, dim)
    return chFirst(e);
  },
  spectrogram: (x, st) => {
    const c = st.cfg, nFft = c.n_fft || 32, hop = c.hop || 8, M = specMats(nFft, c.n_mels || 16, c.n_mfcc || 13), z = tf.zeros([M.nf]);
    const re = conv1d(x, M.cos, z, { kernel: nFft, stride: hop, padding: "valid" }), im = conv1d(x, M.sin, z, { kernel: nFft, stride: hop, padding: "valid" });
    const p = re.square().add(im.square());                                        // (N, 주파수, 시간)
    if ((c.mode || "log") === "log") return tf.log1p(p).expandDims(1);
    const mm = tf.log1p(tf.matMul(M.mel.expandDims(0).tile([p.shape[0], 1, 1]), p));
    return (c.mode === "mel" ? mm : tf.matMul(M.dct.expandDims(0).tile([p.shape[0], 1, 1]), mm)).expandDims(1);
  },
  rand_flip: (x, st, P, k, training) => {
    if (!training) return x;
    const dir = st.cfg.dir || "h", N = x.shape[0], pick = (t, ax) => {
      const m = uni(N).less(0.5).reshape([N].concat(new Array(t.rank - 1).fill(1))).toFloat();
      return tf.reverse(t, ax).mul(m).add(t.mul(tf.scalar(1).sub(m)));
    };
    if (x.rank === 3) return pick(x, 2);
    let y = x;
    if (dir.includes("h")) y = pick(y, 3);
    if (dir.includes("v")) y = pick(y, 2);
    return y;
  },
  rand_rotate: (x, st, P, k, training) => {
    if (!training) return x;
    const [N, , H, W] = x.shape, cx = (W - 1) / 2, cy = (H - 1) / 2, deg = st.cfg.degrees || 15;
    return imgTransform(x, Array.from({ length: N }, () => {
      const a = (Math.random() * 2 - 1) * deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
      return [c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0, 0];
    }));
  },
  rand_zoom: (x, st, P, k, training) => {
    if (!training) return x;
    const [N, , H, W] = x.shape, cx = (W - 1) / 2, cy = (H - 1) / 2, f = st.cfg.frac ?? 0.1;
    return imgTransform(x, Array.from({ length: N }, () => { const z = 1 / (1 + (Math.random() * 2 - 1) * f); return [z, 0, cx - z * cx, 0, z, cy - z * cy, 0, 0]; }));
  },
  rand_shift: (x, st, P, k, training) => {
    if (!training) return x;
    const f = st.cfg.frac ?? 0.1;
    if (x.rank === 4) {
      const [N, , H, W] = x.shape;
      return imgTransform(x, Array.from({ length: N }, () => [1, 0, (Math.random() * 2 - 1) * f * W, 0, 1, (Math.random() * 2 - 1) * f * H, 0, 0]));
    }
    const L = x.shape[2], m = Math.round(L * f);
    if (!m) return x;
    const padded = x.pad([[0, 0], [0, 0], [m, m]]);
    return perSample(padded, (xi) => xi.slice([0, Math.floor(Math.random() * (2 * m + 1))], [-1, L]));
  },
  rand_scale: (x, st, P, k, training) => (training ? x.mul(uni(x.shape[0]).mul(2).sub(1).mul(st.cfg.frac ?? 0.2).add(1).reshape([x.shape[0]].concat(new Array(x.rank - 1).fill(1)))) : x),
  rand_stretch: (x, st, P, k, training) => {
    if (!training) return x;
    const L = x.shape[2], f = st.cfg.frac ?? 0.1;
    return perSample(x, (xi) => {
      const n = Math.max(2, Math.round(L * (1 + (Math.random() * 2 - 1) * f))), y = alongLast(xi, interpMat(L, n));
      if (n >= L) { const a = Math.floor((n - L) / 2); return y.slice([0, a], [-1, L]); }
      const a = Math.floor((L - n) / 2);
      return y.pad([[0, 0], [a, L - n - a]]);
    });
  },
};
function spatialPart(x, st, P, k) {                   // 위치마다 [채널 평균, 채널 최댓값] → k칸 Conv → sigmoid → 곱함
  const m = tf.concat([x.mean(1, true), x.max(1, true)], 1), kk = st.cfg.kernel || 7;
  const y = x.rank === 4 ? conv2d(m, P[k + ".conv.weight"], P[k + ".conv.bias"], { kernel: kk }) : conv1d(m, P[k + ".conv.weight"], P[k + ".conv.bias"], { kernel: kk });
  return x.mul(tf.sigmoid(y));
}
function groupNorm(x, g, P, k) {                      // (N, C, …) → 묶음(g)마다 평균·분산 (편향 분산, eps 1e-5) → 채널마다 배율·이동
  const sh = x.shape, N = sh[0], C = sh[1], r = x.reshape([N, g, -1]), m = tf.moments(r, 2, true);
  const y = r.sub(m.mean).div(m.variance.add(1e-5).sqrt()).reshape(sh);
  const cs = [1, C].concat(new Array(x.rank - 2).fill(1));
  return y.mul(P[k + ".weight"].reshape(cs)).add(P[k + ".bias"].reshape(cs));
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
      for (const [name, shape, kind, init] of st.params) {
        // TF.js 변수 이름은 전체에서 겹치면 안 됨 → 자동 이름, 우리 이름은 nameOf로
        const v = tf.variable(initParam(st, name, shape, kind, rand, init), kind === "weight", undefined, kind === "count" ? "int32" : "float32");
        this.P[name] = v;
        this.nameOf[v.name] = name;
        if (kind === "weight") this.trainable.push(v);
      }
    }
  }

  forward(x, training = false, pending = [], record = null) {
    const vals = { x: this.kind === "rnn" ? tf.transpose(x, [0, 2, 1]) : x };   // 순환 모델 입력 (N, L, 1) → 층들이 쓰는 (N, 1, L)
    for (const st of this.plan.steps) {
      ATT_NODE = st.node;
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

  trace(x1, attn = null) {                            // 샘플 하나 → 노드 id마다 값 (배치 차원 뺀 중첩 배열). attn이 있으면 어텐션 지도도
    return tf.tidy(() => {
      const rec = {};
      const got = attn ? {} : null;
      ATT = got;
      try { this.forward(x1, false, [], rec); } finally { ATT = null; ATT_NODE = null; }
      const out = {};
      for (const [k, v] of Object.entries(rec)) out[k] = v.squeeze([0]).arraySync();
      if (attn) for (const [k, v] of Object.entries(got)) attn[k] = attnView(v.arraySync());
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
  const det = d.task === "detection" ? d.det : null;       // 탐지: {grid, per_cell, n_cls, train, test (그림마다 정답 상자)}
  const reg = !det && d.task !== "classification";
  const score = (out, which) => (det ? detMetric(out, det[which], det) : metric(out, which === "test" ? d.y_test : d.y_train, d.task, d.n_out));
  let mu = 0, sd = 1;
  if (reg) {
    for (const v of d.y_train) mu += v; mu /= d.n_train;
    let q = 0; for (const v of d.y_train) q += (v - mu) ** 2;
    sd = Math.sqrt(q / d.n_train) || 1;
  }
  const Xtr = toInput(d.X_train, d.n_train, plan.kind, d.length, d.image);
  const Xte = toInput(d.X_test, d.n_test, plan.kind, d.length, d.image);
  const ytr = det ? tf.tensor2d(d.y_train, [d.n_train, d.n_out])
    : reg ? tf.tensor2d(Float32Array.from(d.y_train, (v) => (v - mu) / sd), [d.n_train, 1])
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
            return det ? detLoss(out, yb, det.grid, det.per_cell)
              : reg ? tf.losses.meanSquaredError(yb, out) : tf.losses.softmaxCrossEntropy(yb, out);
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
      const stop = onEpoch && onEpoch({ epoch, loss: lossSum / Math.max(1, nb), test_metric: score(out, "test") });
      if (stop || (isCancelled && isCancelled())) break;
      await yieldNow();
    }
    if (reg) {                                              // 정규화한 정답으로 배운 출력층에 되돌림을 접어 넣음 (원래 단위로 냄)
      const last = plan.steps.find((st) => st.name === plan.out);
      const kk = last.key + (last.in_shape.length >= 2 ? ".1" : "");
      const W = net.P[kk + ".weight"], b = net.P[kk + ".bias"];
      W.assign(W.mul(sd)); b.assign(b.mul(sd).add(mu));
    }
    // 마지막 평가는 CPU로 (브라우저 GPU(WebGL)가 드물게 묶음 크기에 따라 틀린 값을 냄). GPU 값과 다르면 알림
    const outGpu = net.predict(Xte);
    const [out, outTr] = await onCpu(() => [net.predict(Xte), net.predict(Xtr)]);
    const r = { epochs_done: done, test_metric: score(out, "test"),
      metric_name: det ? "map50" : reg ? "r2" : "accuracy", warnings: [], confusion: null, recall: null,
      train_metric: score(outTr, "train") };   // 학습 데이터 성적 (시험과 차이가 크면 외운 것)
    if (!same(outGpu, out))
      r.warnings.push({ message: "학습 중 이 브라우저의 GPU 계산이 불안정했습니다. 아래 결과는 CPU로 다시 계산한 값이지만, 학습이 덜 됐을 수 있습니다",
        hint: "페이지를 새로 고친 뒤 다시 학습해 보세요. 계속되면 다른 브라우저(크롬·엣지)를 써 보세요" });
    if (det) Object.assign(r, detResult(out, det.test, det));
    else if (reg) [r.mae, r.scatter] = regressionView(out, d.y_test, d.n_out);
    else {
      [r.confusion, r.recall] = confusion(out, d.y_test, d.n_out);
      r.pred = Array.from({ length: d.n_test }, (_, i) => argmax(out, i, d.n_out));   // 시험 샘플마다 예측 (틀린 샘플 모아 보기)
    }
    return { net, result: r };
  } finally {
    [Xtr, Xte, ytr].forEach((t) => t.dispose());
    opt.dispose();
  }
}

// ------------------------------------------------------------ 객체 탐지 (detect.py와 같은 식·순서)
const DET = { coord: 5, noobj: 0.5, scoreMin: 0.05, show: 0.5, nms: 0.5, match: 0.5, maxDet: 50, keep: 0.25 };
function detLoss(out, t, G, A) {                      // 있음(BCE) + 사물 칸 위치·크기 제곱 오차 × 5 + 클래스 교차 엔트로피, 그림 수로 나눔
  const N = out.shape[0], o = out.reshape([N, A, G, G]), y = t.reshape([N, A, G, G]);
  const part = (a, n) => [o.slice([0, a, 0, 0], [N, n, G, G]), y.slice([0, a, 0, 0], [N, n, G, G])];
  const [o0, y0] = part(0, 1), pos = y0.squeeze([1]), x = o0.squeeze([1]);
  const bce = tf.relu(x).sub(x.mul(pos)).add(tf.log1p(tf.exp(x.abs().neg())));      // BCE with logits (안정한 식)
  let loss = bce.mul(pos.mul(1 - DET.noobj).add(DET.noobj)).sum();
  const [ob, yb] = part(1, 4);
  loss = loss.add(tf.sigmoid(ob).sub(yb).square().sum(1).mul(pos).sum().mul(DET.coord));
  if (A > 6) {
    const [oc, yc] = part(5, A - 5);
    loss = loss.add(yc.mul(oc.sub(tf.logSumExp(oc, 1, true))).sum(1).neg().mul(pos).sum());
  }
  return loss.div(N);
}
const sig = (v) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, v))));
function iou(a, b) {                                  // [x, y, w, h]
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0), union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}
function nms(cand) {
  const keep = [];
  cand.sort((a, b) => b[1] - a[1] || a[0] - b[0] || a[2] - b[2] || a[3] - b[3]);
  for (const c of cand) {
    if (keep.every((q) => c[0] !== q[0] || iou(c.slice(2), q.slice(2)) <= DET.nms)) keep.push(c);
    if (keep.length >= DET.maxDet) break;
  }
  return keep;
}
function detDecode(out, i, det, scoreMin = DET.scoreMin) {   // 출력 i번째 줄 → [[클래스 번호, 점수, x, y, w, h], …]
  const G = det.grid, A = det.per_cell, K = A - 5, GG = G * G, off = i * A * GG, cand = [];
  const at = (c, gy, gx) => out[off + c * GG + gy * G + gx];
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
    let k = 0, p = 1;
    if (K > 1) {
      let m = -Infinity;
      for (let j = 0; j < K; j++) m = Math.max(m, at(5 + j, gy, gx));
      let sum = 0, best = -1;
      for (let j = 0; j < K; j++) { const e = Math.exp(at(5 + j, gy, gx) - m); sum += e; if (e > best) { best = e; k = j; } }
      p = best / sum;
    }
    const s = sig(at(0, gy, gx)) * p;
    if (s < scoreMin) continue;
    const cx = (gx + sig(at(1, gy, gx))) / G, cy = (gy + sig(at(2, gy, gx))) / G;
    const w = sig(at(3, gy, gx)) ** 2, h = sig(at(4, gy, gx)) ** 2;
    const x0 = Math.max(0, cx - w / 2), y0 = Math.max(0, cy - h / 2), x1 = Math.min(1, cx + w / 2), y1 = Math.min(1, cy + h / 2);
    if (x1 > x0 && y1 > y0) cand.push([k, s, x0, y0, x1 - x0, y1 - y0]);
  }
  return nms(cand);
}
function detMatch(preds, gts, k) {                    // 그림 하나·클래스 k → [[점수, 맞음]]
  const used = gts.map(() => false), out = [];
  const mine = preds.filter((p) => p[0] === k).sort((a, b) => b[1] - a[1]);
  for (const p of mine) {
    let best = 0, j = -1;
    gts.forEach((g, gi) => { if (g[0] === k && !used[gi]) { const v = iou(p.slice(2), g.slice(1)); if (v > best) { best = v; j = gi; } } });
    const ok = j >= 0 && best >= DET.match;
    if (ok) used[j] = true;
    out.push([p[1], ok]);
  }
  return out;
}
function detAP(preds, gts, K) {                       // → [mAP, 클래스마다 AP (정답 없으면 null)]
  const aps = [];
  for (let k = 0; k < K; k++) {
    let nPos = 0;
    for (const g of gts) for (const b of g) if (b[0] === k) nPos++;
    if (!nPos) { aps.push(null); continue; }
    const hits = [];
    preds.forEach((p, i) => hits.push(...detMatch(p, gts[i], k)));
    hits.sort((a, b) => b[0] - a[0]);
    const rec = [0], prec = [1];
    let tp = 0, fp = 0;
    for (const [, ok] of hits) { if (ok) tp++; else fp++; rec.push(tp / nPos); prec.push(tp / Math.max(tp + fp, 1)); }
    if (!hits.length) rec.push(0);
    rec.push(1); prec.push(0);
    if (!hits.length) prec.splice(1, 0, 0);
    for (let i = prec.length - 2; i >= 0; i--) prec[i] = Math.max(prec[i], prec[i + 1]);
    let ap = 0;
    for (let i = 1; i < rec.length; i++) ap += (rec[i] - rec[i - 1]) * prec[i];
    aps.push(Math.round(ap * 1e6) / 1e6);
  }
  const have = aps.filter((a) => a !== null);
  return [have.length ? have.reduce((a, b) => a + b, 0) / have.length : 0, aps];
}
function detPR(preds, gts, K) {                       // 점수 0.5 이상만: 맞게 찾음·잘못 찾음·놓침, 정밀도·재현율
  let tp = 0, fp = 0, nGt = 0;
  preds.forEach((p, i) => {
    const shown = p.filter((q) => q[1] >= DET.show);
    for (let k = 0; k < K; k++) for (const [, ok] of detMatch(shown, gts[i], k)) { if (ok) tp++; else fp++; }
    nGt += gts[i].length;
  });
  return { tp, fp, fn: nGt - tp, precision: tp + fp ? tp / (tp + fp) : null, recall: nGt ? tp / nGt : null };
}
function detAll(out, n, det) { return Array.from({ length: n }, (_, i) => detDecode(out, i, det)); }
function detMetric(out, gts, det) { return detAP(detAll(out, gts.length, det), gts, det.per_cell - 5)[0]; }
function detResult(out, gts, det) {
  const preds = detAll(out, gts.length, det), K = det.per_cell - 5, [m, ap] = detAP(preds, gts, K);
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  return { test_metric: m, ap, pr: detPR(preds, gts, K),
    det_pred: preds.map((ps) => ps.filter((p) => p[1] >= DET.keep).map((p) => [p[0]].concat(p.slice(1).map(r4)))) };
}

// fn을 CPU 백엔드에서 (이미 CPU거나 CPU가 없으면 그대로). 가중치는 TF.js가 알아서 옮김
async function onCpu(fn) {
  const prev = tf.getBackend();
  if (prev === "cpu" || !tf.findBackend("cpu")) return fn();
  await tf.setBackend("cpu");
  try { return fn(); } finally { await tf.setBackend(prev); }
}
function same(a, b) {
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-3 * (1 + Math.abs(b[i]))) return false;
  return true;
}

// 샘플 따라가기 값: (C, L)이 200점보다 길면 고르게 200점, 소수 다섯째 자리 (analysis._thin과 같음)
// 어텐션 가중치 (질문 T, 본 위치 S) → 화면용 (analysis.attn_view와 같은 꼴): 지도(줄여서)·위치마다 평균·첫 자리(CLS)가 본 곳
function attnView(a) {
  const T = a.length, S = a[0].length, r5 = (v) => Math.round(v * 1e5) / 1e5;
  const pick = (n, m) => (n > m ? Array.from({ length: m }, (_, i) => Math.round((i * (n - 1)) / (m - 1))) : Array.from({ length: n }, (_, i) => i));
  const rows = pick(T, 64), cols = pick(S, 128);
  const mean = Array.from({ length: S }, (_, j) => r5(a.reduce((t, row) => t + row[j], 0) / T));
  return { q: T, k: S, map: rows.map((i) => cols.map((j) => r5(a[i][j]))), mean, first: a[0].map(r5) };
}
function thin(v) {
  const r5 = (x) => Math.round(x * 1e5) / 1e5;
  if (!Array.isArray(v)) return r5(v);
  if (Array.isArray(v[0]) && !Array.isArray(v[0][0]) && v[0].length > 200) {
    const L = v[0].length, idx = Array.from({ length: 200 }, (_, i) => Math.round((i * (L - 1)) / 199));
    return v.map((row) => idx.map((j) => r5(row[j])));
  }
  return v.map(thin);
}

const API = { Net, train, toInput, metric, confusion, regressionView, thin, attnView, onCpu, ACT, tf,
  DET, detLoss, detDecode, detAP, detPR, detMetric, detResult };
if (typeof module !== "undefined") module.exports = API;
if (typeof window !== "undefined") window.DLEngine = API;
})();

// 학습 전용 일꾼 (Web Worker): 화면과 다른 스레드에서 engine.js의 train을 돌려, 학습하는 동안 화면이 버벅이지 않게 한다.
// web.js의 trainInWorker가 부름. 끝나면 가중치(state)를 넘기고, 화면 쪽이 같은 모델을 다시 만들어 예측·미리보기·저장에 씀.
// 주고받는 것: init {tf: [주소, SRI], engine} → ready {backend} · train {id, plan, d, s, pretrained} → epoch {id, p} … done/error · cancel {id}
"use strict";
const cancelled = new Set();

async function init({ tf, engine }) {
  const r = await fetch(tf[0], { integrity: tf[1], mode: "cors" });   // CDN 파일은 내용 해시가 맞을 때만
  if (!r.ok) throw new Error("TensorFlow.js를 받지 못함 (" + r.status + ")");
  const url = URL.createObjectURL(new Blob([await r.text()], { type: "text/javascript" }));
  try { importScripts(url); } finally { URL.revokeObjectURL(url); }
  importScripts(engine);
  await self.tf.ready();
  return self.tf.getBackend();
}

async function train({ id, plan, d, s, pretrained }) {
  const E = self.DLEngine;
  await E.ensurePretrained(plan, pretrained);
  const { net, result, out, outTr } = await E.train(plan, d, s,
    (p) => { self.postMessage({ type: "epoch", id, p }); return cancelled.has(id); },
    () => cancelled.has(id));
  try {
    const state = net.state(), moved = [];
    for (const v of Object.values(state)) moved.push(v.data.buffer);
    if (out) moved.push(out.buffer);
    if (outTr) moved.push(outTr.buffer);
    self.postMessage({ type: "done", id, result, out: out || null, outTr: outTr || null, state }, moved);
  } finally { net.dispose(); }
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "cancel") { cancelled.add(m.id); return; }
  try {
    if (m.type === "init") self.postMessage({ type: "ready", backend: await init(m) });
    else if (m.type === "train") await train(m);
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, name: err.name || "Error", message: String(err.message || err) });
  } finally { if (m.id) cancelled.delete(m.id); }
};

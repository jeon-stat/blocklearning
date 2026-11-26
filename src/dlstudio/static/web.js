// DL studio 웹 버전(GitHub Pages)의 다리: 화면(app.js)의 api(path, body)를 서버 대신 브라우저 안에서 처리한다.
// - Python(dlstudio 패키지)은 Pyodide로 그대로 돌린다: 검사·데이터 읽기·과제·코드 생성·프로젝트 파일 (web.py)
// - 학습·샘플 따라가기는 TensorFlow.js 엔진(engine.js), 파일은 고르기(업로드 아님, 브라우저 안에만)·내려받기
// 데이터는 PC 밖으로 나가지 않는다. 로컬 실행(python -m dlstudio)에서는 이 파일을 쓰지 않는다.
(function () {
"use strict";
const PY_VERSION = "0.27.7";
const BASE = "src/dlstudio/";
const PY_FILES = ["__init__.py", "analysis.py", "anomaly.py", "blocks.py", "codegen.py", "core.py", "data.py", "detect.py", "forecast.py", "graph.py", "jobs.py", "layers.py", "losses.py", "nn_extra.py",
  "labeling.py", "labeling_sources.py", "predict.py", "pretrained.py", "project.py", "multi.py", "segment.py", "tasks.py", "web.py", "weblabel.py"];
const PRETRAINED = BASE + "pretrained/";                 // 사전학습 가중치 (그 블록을 쓸 때만 받음)
const gotData = new Set();                               // 받아 둔 예시 데이터 파일 (예시를 누를 때만 받음)
const E = () => window.DLEngine;

const W = { py: null, mod: null, studio: null, data: null, trained: null, jobs: {}, running: null, n: 0 };

// ------------------------------------------------------------ 준비 (Pyodide + 우리 Python 코드)
function bootMessage(text) {
  let el = document.getElementById("web-boot");
  if (!el) {
    el = document.createElement("div");
    el.id = "web-boot";
    el.className = "web-boot";
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.hidden = !text;
}

async function boot() {
  bootMessage("DL studio를 준비하는 중… (처음 열 때는 10초쯤 걸립니다. 계산은 모두 이 브라우저 안에서 합니다)");
  const py = await loadPyodide({ indexURL: `https://cdn.jsdelivr.net/pyodide/v${PY_VERSION}/full/` });
  await py.loadPackage(["numpy", "pillow"]);
  py.FS.mkdirTree("/app/dlstudio/data");
  await Promise.all(PY_FILES.map(async (f) => py.FS.writeFile(`/app/dlstudio/${f}`, await (await fetch(BASE + f, { cache: "no-cache" })).text())));
  py.FS.mkdirTree("/data");
  py.runPython("import sys; sys.path.insert(0, '/app')");
  W.mod = py.pyimport("dlstudio.web");
  W.lab = py.pyimport("dlstudio.weblabel");
  W.studio = W.mod.WebStudio();
  W.py = py;
  bootMessage("");
}
const ready = boot().catch((e) => { bootMessage("DL studio를 준비하지 못함: " + e.message + " (인터넷 연결을 확인하고 새로 고치세요)"); throw e; });

// ------------------------------------------------------------ 도움 함수
function pyError(e) {                                   // Python 예외 → 화면 오류 모양
  const msg = String((e && e.message) || e);
  const m = msg.match(/UserError: (.*)/) || msg.match(/DataError: (.*)/) || msg.match(/(\w+Error: .*)\s*$/);
  return { message: m ? m[1].trim() : "처리 중 오류", hint: "", detail: msg.slice(-3000) };
}
function wrapped(json) {                                // web.py의 _wrap 결과 {status, body}
  const r = JSON.parse(json);
  if (r.status !== 200) throw (r.body && r.body.error) || { message: "오류" };
  return r.body;
}
function download(name, data, type = "application/octet-stream") {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
}
const baseName = (p) => String(p || "").trim().replace(/^["']|["']$/g, "").split(/[\\/]/).pop();
function bytesOf(proxy) {                               // Python bytes → Uint8Array
  const u8 = proxy.toJs();
  if (proxy.destroy) proxy.destroy();
  return u8;
}
function getData() {
  if (W.data) return W.data;
  const p = W.mod.data_payload(W.studio);
  const f32 = (k) => { const u8 = bytesOf(p.get(k)); return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)); };
  const meta = JSON.parse(p.get("meta"));
  W.data = Object.assign(meta, { X_train: f32("X_train"), X_test: f32("X_test"), y_train: f32("y_train"), y_test: f32("y_test") });
  p.destroy();
  return W.data;
}
function stateForPython(net) {                          // 엔진 가중치 → Python에 넘길 {이름: {shape, data(바이트), int}}
  const out = {};
  for (const [k, v] of Object.entries(net.state())) {
    const isInt = v.data instanceof Int32Array;
    out[k] = { shape: v.shape, data: new Uint8Array(v.data.buffer.slice(0)), int: isInt };
  }
  return W.py.toPy(out);
}
function clearTrained() {
  if (W.running) W.running.cancelled = true;
  if (W.trained) W.trained.net.dispose();
  W.trained = null;
  W.data = null;
}
const pyCall = (method, path, body) => {
  const r = JSON.parse(W.mod.call_json(W.studio, method, path, body === undefined ? "" : JSON.stringify(body)));
  if (r.status !== 200) throw (r.body && r.body.error) || { message: `오류 ${r.status}` };
  return r.body;
};

// ------------------------------------------------------------ 학습 (작업처럼: id → 진행 읽기)
function startTrain(body) {
  const plan = wrapped(W.mod.plan(W.studio, JSON.stringify(body.graph)));
  const data = getData();
  if (W.running) W.running.cancelled = true;
  const ls = wrapped(W.mod.loss_setup(W.studio, JSON.stringify(body.settings || {})));    // 조립한 손실 검사 (기본이면 spec 없음)
  const s = Object.assign({ epochs: 30, lr: 1e-3, batch_size: 32, seed: 0 }, body.settings || {}, { loss: ls.spec, class_weights: ls.class_weights });
  const job = { id: "w" + (++W.n), kind: "train", state: "running", progress: [], result: null, error: null, cancelled: false };
  W.jobs[job.id] = job;
  W.running = job;
  const onEpoch = (p) => { job.progress.push(p); job.progress.splice(0, Math.max(0, job.progress.length - 100)); return job.cancelled; };
  const dist = plan.steps.find((st) => st.type === "output" && st.cfg.head === "distance");     // 이상 탐지 '정상과의 거리'
  E().ensurePretrained(plan, PRETRAINED).then(() => (dist ? E().fitDistance(plan, data, fitDistance(dist), onEpoch)
    : E().train(plan, data, s, onEpoch, () => job.cancelled))).then(({ net, result, out, outTr }) => {
    if ((job.cancelled && !job.stopped) || W.data !== data) { net.dispose(); job.state = "cancelled"; return; }   // 다시 시작·데이터 바뀜은 버림
    result.stopped = !!job.stopped;
    if (out) Object.assign(result, wrapped(W.mod.task_result(W.studio, new Uint8Array(out.buffer), new Uint8Array(outTr.buffer))));   // 예측·이상 탐지
    if (W.trained) W.trained.net.dispose();
    W.trained = { net, plan, graph: body.graph, settings: body.settings || {}, results: result };
    const st = stateForPython(net);
    try { W.mod.set_trained(W.studio, JSON.stringify(body.graph), JSON.stringify(body.settings || {}), JSON.stringify(result), st); }
    finally { st.destroy(); }
    job.result = result;
    job.state = "done";
    changed("train");
  }).catch((e) => {
    job.state = job.cancelled ? "cancelled" : "failed";
    job.error = `${e.name || "Error"}: ${e.message || e}`;
  }).finally(() => { if (W.running === job) W.running = null; });
  return { id: job.id };
}
function fitDistance(head) {                           // 정상 특징 → 정상과의 거리 값 (Python anomaly.fit_distance)
  const k = head.params.find((p) => p[0].endsWith(".basis"))[1][0];
  return async (F, n, c) => {
    const p = W.mod.fit_distance(W.studio, new Uint8Array(F.buffer, F.byteOffset, F.byteLength), n, c, k), out = {};
    try {
      for (const name of ["mean", "basis", "inv_var", "inv_rest"]) {
        const u8 = bytesOf(p.get(name));
        out[name] = new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
      }
    } finally { p.destroy(); }
    return out;
  };
}
function jobView(path) {
  const id = path.split("/")[3], job = W.jobs[id];
  if (!job) throw { message: "작업을 찾을 수 없음" };
  if (path.endsWith("/cancel")) { job.cancelled = job.stopped = true; return { ok: true }; }   // [멈추기]: 거기까지 학습한 것은 남김
  return { id: job.id, kind: job.kind, state: job.state, progress: job.progress.slice(), result: job.result, error: job.error };
}

// ------------------------------------------------------------ 샘플 따라가기
function activations(body) {
  if (!W.trained) throw { message: "학습된 모델이 없음", hint: "④ 학습 단계에서 먼저 학습하세요" };
  const i = Number(body.sample_index || 0);
  const s = wrapped(W.mod.sample(W.studio, i));
  const d = getData(), net = W.trained.net;
  const x = E().toInput(Float32Array.from(s.x), 1, net.kind, d.length, d.image, d.channels);
  const attention = {};
  const values = Object.fromEntries(Object.entries(net.trace(x, attention)).map(([k, v]) => [k, E().thin(v)]));   // 층마다 줄이기
  const out = net.predict(x);
  x.dispose();
  let pred;
  if (d.task === "detection") pred = `상자 ${E().detDecode(out, 0, d.det, E().DET.show).length}개 찾음`;
  else if (d.task === "forecast") pred = `다음 ${out.length}개 (첫 값 ${Number(out[0].toPrecision(4))})`;
  else if (d.task === "segmentation") {                       // 시점마다 가장 큰 상태 → 구간 이름
    const K = d.seg.n_cls, L = d.seg.length, names = [];
    for (let t = 0; t < L; t++) {
      let b = 0;
      for (let k = 1; k < K; k++) if (out[k * L + t] > out[b * L + t]) b = k;
      if (!names.length || names[names.length - 1] !== b) names.push(b);
    }
    pred = names.slice(0, 5).map((k) => s.classes[k]).join(" → ") + (names.length > 5 ? ` … (구간 ${names.length}개)` : "");
  }
  else if (d.task === "anomaly") {
    let t = 0;
    for (let j = 0; j < out.length; j++) t += (out[j] - s.x[j]) ** 2;
    const sc = out.length === 1 && s.x.length > 1 ? out[0] : t / out.length, th = W.trained.results && W.trained.results.threshold;   // '정상과의 거리'면 출력이 점수
    pred = `이상 점수 ${Number(sc.toPrecision(3))}` + (th === undefined || th === null ? "" : sc > th ? " → 이상" : " → 정상");
  }
  else if (d.task === "classification") { let b = 0; for (let j = 1; j < out.length; j++) if (out[j] > out[b]) b = j; pred = s.classes[b]; }
  else if (d.task === "multi") pred = d.multi.heads.map((h) => {      // 정답마다: 분류는 가장 큰 클래스, 값은 그 칸
    if (h.kind !== "class") return `${h.name} ${Number(out[h.start].toPrecision(4))}`;
    let b = 0;
    for (let k = 1; k < h.size; k++) if (out[h.start + k] > out[h.start + b]) b = k;
    return `${h.name} ${h.classes[b]}`;
  }).join(" · ");
  else pred = out[0];
  let sal = null;
  try { sal = E().saliency(net, Float32Array.from(s.x), d); } catch (e) { sal = null; }   // 기울기를 못 내는 블록이면 보이지 않음
  return { index: i, label: s.label, prediction: pred, input: s.input, values, attention, order: s.order, saliency: sal };
}

// ------------------------------------------------------------ 코드·가중치·프로젝트 (내려받기)
function codeSave(body) {
  let name = baseName(body.path);
  if (!/\.py$/i.test(name)) throw { message: "파이썬 파일(.py) 이름을 넣으세요", hint: "예: model_pytorch.py" };
  const stem = name.replace(/\.py$/i, ""), wname = stem + "_weights.npz";
  const r = pyCall("POST", "/api/code", Object.assign({}, body, { weights_file: wname }));
  download(name, r.code, "text/x-python");
  const files = [name];
  if (r.trained && r.framework === "pytorch" && W.trained) {
    const st = stateForPython(W.trained.net);
    try { download(wname, bytesOf(W.mod.weights_npz(W.studio, JSON.stringify(W.trained.graph), st))); } finally { st.destroy(); }
    files.push(wname);
  }
  return { files, trained: r.trained };
}
function exportModel() {
  if (!W.trained) throw { message: "학습된 모델이 없음", hint: "④ 학습 단계에서 먼저 학습하세요" };
  return codeSave({ graph: W.trained.graph, framework: "pytorch", settings: W.trained.settings, path: "model_pytorch.py" });
}
function projectSave(body) {
  let name = baseName(body.path) || "dlstudio-project.dlstudio";
  if (!/\.dlstudio$/i.test(name)) name += ".dlstudio";
  const st = W.trained ? stateForPython(W.trained.net) : null;
  try { download(name, bytesOf(W.mod.project_bytes(W.studio, st))); }
  catch (e) { throw pyError(e); }
  finally { if (st) st.destroy(); }
  return { ok: true, file: name };
}
async function loadTrained(r) {                        // {response, state, plan} → 엔진에 학습한 모델
  if (W.trained) { W.trained.net.dispose(); W.trained = null; }
  if (!r.state) return;
  await E().ensurePretrained(r.plan, PRETRAINED);         // 고정 가중치는 프로젝트에 없음 → 사전학습 파일에서
  const net = new (E().Net)(r.plan, 0);
  const st = {};
  for (const [k, v] of Object.entries(r.state)) st[k] = { shape: v.shape, data: v.int ? Int32Array.from(v.data) : Float32Array.from(v.data) };
  net.load(st);
  W.trained = { net, plan: r.plan, graph: r.response.graph, settings: r.response.train_settings || {}, results: r.response.results || {} };
}
async function projectOpen(body) {
  const r = wrapped(W.mod.record_open(W.studio, body.path, body.data_path || ""));
  W.openedName = r.name || "";
  W.data = null;
  await loadTrained(r);
  return r.response;
}
async function runRestore(body) {                      // 학습 기록의 모델로 돌아가기
  const r = wrapped(W.mod.run_restore(W.studio, Number(body.n)));
  await loadTrained(r);
  changed("train");
  return r.response;
}
async function predictNew(body) {                      // 새 데이터로 예측: Python이 읽고 정규화 → 엔진(CPU) → Python이 표로
  if (!W.trained) throw { message: "학습된 모델이 없음", hint: "④ 학습 단계에서 먼저 학습하세요" };
  const n = wrapped(W.mod.predict_prepare(W.studio, JSON.stringify(body))).n;
  const u8 = bytesOf(W.mod.predict_x(W.studio));
  const X = new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
  const d = getData(), net = W.trained.net;
  const x = E().toInput(X, n, net.kind, d.length, d.image, d.channels);
  let out;
  try { out = (await E().checkedPredict(net, x, n)).out; } finally { x.dispose(); }
  return wrapped(W.mod.predict_finish(W.studio, new Uint8Array(out.buffer)));
}
const VIEW = ["pos", "size", "brk", "shift"];
function bare(g) {                                     // 화면용 값을 뺀 그래프 (같은 모델인지 비교)
  return JSON.stringify(g, (k, v) => (VIEW.includes(k) ? undefined : v));
}
async function preview(body) {                         // ③ 블록 미리보기: 학습 샘플 하나가 그 블록을 지난 값
  const plan = wrapped(W.mod.plan(W.studio, JSON.stringify(body.graph)));
  await E().ensurePretrained(plan, PRETRAINED);
  const d = getData();
  const trained = !!W.trained && bare(W.trained.graph) === bare(body.graph);
  const net = trained ? W.trained.net : new (E().Net)(plan, 0);
  const x = E().toInput(d.X_train.slice(0, d.n_features), 1, net.kind, d.length, d.image, d.channels);
  try {
    const vals = net.trace(x);
    if (!(body.node in vals)) throw { message: "이 블록의 값을 찾을 수 없음", hint: "블록을 다시 고르세요" };
    const label = d.task === "detection" ? `상자 ${d.det.train[0].length}개` : d.task === "classification" ? d.classes[d.y_train[0]] : d.y_train[0];
    return { node: body.node, value: E().thin(vals[body.node]), trained, label };
  } finally { x.dispose(); if (!trained) net.dispose(); }
}

// ------------------------------------------------------------ 라벨링 (라벨링 페이지 · studio ② 라벨링한 데이터)
// 계산은 core.api_label 그대로 (numpy만 씀). 파일 경로가 필요한 동작만 여기서: 새로 만들기·합치기·내보내기(내려받기)·그림 보기
const LABEL_READ = new Set(["state", "items", "item", "rule_preview", "cluster", "normal_candidates", "columns"]);
const LABEL_OUT = new Set(["export", "export_yolo", "export_coco", "export_normals", "split", "bundle"]);
let labelUrls = {};                                      // 그림 이름 → blob URL (크게 보기)
function resetLabelUrls() { Object.values(labelUrls).forEach((u) => URL.revokeObjectURL(u)); labelUrls = {}; }
function labelCall(op, body) {
  let r;
  if (op === "new") { resetLabelUrls(); r = wrapped(W.lab.new_job(W.studio, JSON.stringify(body))); }
  else if (op === "merge") r = wrapped(W.lab.merge_file(W.studio, body.file || ""));
  else if (op === "use" || op === "open") throw { message: "웹 버전은 라벨링 페이지의 작업 목록으로 엽니다", hint: "" };
  else if (op === "deep_suggest") return deepSuggest(body);
  else if (LABEL_OUT.has(op)) {                          // 내보내기: Pyodide 안에 쓴 파일을 내려받기
    r = wrapped(W.lab.export_file(W.studio, op, JSON.stringify(body)));
    for (const [name, p] of r.files) download(name, W.py.FS.readFile(p));
    return r;
  } else r = pyCall("POST", "/api/label/" + op, body);
  if (!LABEL_READ.has(op)) changed("label");
  return r;
}
function loadScript(src, integrity) {
  return new Promise((ok, no) => {
    const s = document.createElement("script");
    s.src = src;
    if (integrity) { s.integrity = integrity; s.crossOrigin = "anonymous"; }
    s.onload = ok;
    s.onerror = () => no(new Error("불러오지 못함: " + src));
    document.body.appendChild(s);
  });
}
async function ensureEngine() {                         // 라벨링 페이지: 학습 엔진(TF.js + engine.js)은 처음 쓸 때 불러옴
  if (window.tf && window.DLEngine) return;
  const B = window.DL_BOOT || {};
  if (!window.tf) await loadScript(B.TF[0], B.TF[1]);
  if (!window.DLEngine) await loadScript((B.STATIC || "src/dlstudio/static/") + "engine.js?v=" + Date.now());
}
async function deepSuggest(body) {                      // 딥러닝 제안: 정답 붙은 샘플로 대표 모델 학습 → 정답 없는 샘플에 확률
  const prep = wrapped(W.lab.deep_prepare(W.studio));
  if (body.onProgress) body.onProgress(`${prep.model} 준비 중… (처음이면 학습 엔진을 받느라 몇 초 걸립니다)`);
  await ensureEngine();
  const p = W.lab.deep_payload();
  const f32 = (k) => { const u8 = bytesOf(p.get(k)); return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)); };
  const meta = JSON.parse(p.get("meta"));
  const d = Object.assign(meta, { X_train: f32("X_train"), X_test: f32("X_test"), y_train: f32("y_train"), y_test: f32("y_test") });
  const Xp = f32("X_pred");
  p.destroy();
  const epochs = 40;
  const { net, result } = await E().train(prep.plan, d, { epochs, lr: 0.003, batch_size: 32, seed: 0 },
    (pr) => { if (body.onProgress) body.onProgress(`${prep.model} 학습 중… ${pr.epoch}/${epochs}`); return false; });
  let probs;
  const x = E().toInput(Xp, meta.n_pred, net.kind, d.length, d.image, d.channels);
  try {
    const out = (await E().checkedPredict(net, x, meta.n_pred)).out;
    const k = d.n_out;
    probs = new Float32Array(out.length);
    for (let i = 0; i < meta.n_pred; i++) {             // softmax
      let m = -Infinity, s = 0;
      for (let j = 0; j < k; j++) m = Math.max(m, out[i * k + j]);
      for (let j = 0; j < k; j++) { probs[i * k + j] = Math.exp(out[i * k + j] - m); s += probs[i * k + j]; }
      for (let j = 0; j < k; j++) probs[i * k + j] /= s;
    }
  } finally { x.dispose(); net.dispose(); }
  const acc = meta.n_test ? result.test_metric : null;
  const r = wrapped(W.lab.deep_apply(W.studio, new Uint8Array(probs.buffer), acc));
  changed("label");
  return Object.assign(r, { model: prep.model });
}
async function labelFileUrl(name) {                     // 그림 원본 → blob URL
  if (labelUrls[name]) return labelUrls[name];
  const p = String(W.lab.file_path(W.studio, name));
  const ext = (p.split(".").pop() || "").toLowerCase();
  const type = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", bmp: "image/bmp" }[ext] || "application/octet-stream";
  labelUrls[name] = URL.createObjectURL(new Blob([W.py.FS.readFile(p)], { type }));
  return labelUrls[name];
}
function labelWork() {                                   // 지금 작업 → {work: 정답 JSON 글자, meta: 목록 요약}
  return { work: String(W.lab.work_text(W.studio)), meta: JSON.parse(String(W.lab.summary(W.studio))) };
}
const labelData = () => bytesOf(W.lab.data_bytes(W.studio));
async function labelOpen(work, data) {                   // 보관함의 작업 → 라벨링 페이지에서 엶
  await ready;
  resetLabelUrls();
  W.py.FS.mkdirTree("/work");
  const p = `/work/label_${++W.n}.zip`;
  if (data && data.length) W.py.FS.writeFile(p, data);
  return wrapped(W.lab.open_job(W.studio, data && data.length ? p : "", work));
}
async function labelOpenBundle(path) {                  // 작업 묶음(.dllabel.zip) 가져오기
  await ready;
  resetLabelUrls();
  return wrapped(W.lab.open_bundle(W.studio, path));
}
async function useLabelJob(body) {                      // studio ②: 라벨링한 작업 → 학습 데이터
  const rec = await window.DLStore.labels.load(body.label_job);
  if (!rec || !rec.work) throw { message: "라벨링 작업을 찾을 수 없음", hint: "라벨링 페이지에서 작업을 다시 열어 저장하세요" };
  clearTrained();
  W.py.FS.mkdirTree("/work");
  const p = `/work/label_${++W.n}.zip`;
  if (rec.data && rec.data.length) W.py.FS.writeFile(p, rec.data);
  const r = wrapped(W.lab.use_job(W.studio, rec.data && rec.data.length ? p : "", rec.work, !!body.confirmed_only));
  const item = (await window.DLStore.labels.list()).find((x) => x.id === body.label_job);
  r.label_name = item ? item.name : "";
  changed("data");
  return r;
}

// ------------------------------------------------------------ 바뀜 알림 (자동 저장이 듣는 곳)
const listeners = [];
function changed(kind) { for (const f of listeners) { try { f(kind); } catch (e) { console.error(e); } } }

// 지금 작업을 기록 바이트로 (store.js가 브라우저 보관함·구글 드라이브에 씀). 데이터가 없거나 모델이 없으면 null
// opts.withData: 데이터를 프로젝트 파일 안에 함께 (내보내기), opts.dataCopy: 데이터 사본(data.zip)도 따로 만듦 (데이터가 바뀐 때만)
function recordBytes(name, opts = {}) {
  if (!W.studio || !W.studio.data) return null;
  const st = W.trained ? stateForPython(W.trained.net) : null;
  try {
    const out = { project: bytesOf(W.mod.record_project(W.studio, st, name || "", !!opts.withData)) };
    if (opts.dataCopy) out.data = bytesOf(W.mod.data_zip(W.studio));
    return out;
  } catch (e) {
    const err = pyError(e);
    if (/저장할 모델이 없음/.test(err.message)) return null;
    throw err;
  } finally { if (st) st.destroy(); }
}
// 예시 데이터 파일 받기: 예시 하나에 필요한 npz만 (처음 열 때 모두 받지 않게)
async function ensureExample(kind) {
  if (!kind) return;
  for (const f of JSON.parse(W.mod.example_files_json(kind))) {
    if (gotData.has(f)) continue;
    const res = await fetch(BASE + "data/" + f, { cache: "no-cache" });
    if (!res.ok) throw { message: `예시 데이터를 받지 못함: ${f}`, hint: "인터넷 연결을 확인하고 다시 누르세요" };
    W.py.FS.writeFile(`/app/dlstudio/data/${f}`, new Uint8Array(await res.arrayBuffer()));
    gotData.add(f);
  }
}
// 기록 바이트 → 열기 (화면용 응답). data는 없어도 됨 (예시 데이터, 또는 프로젝트 zip 안에 데이터가 있음)
async function openRecord(project, data) {
  await ready;
  W.py.FS.mkdirTree("/work");
  W.py.FS.writeFile("/work/open.dlstudio", project);
  if (data && data.length) W.py.FS.writeFile("/work/open.zip", data);
  await ensureExample(W.mod.project_example("/work/open.dlstudio"));
  return api("/api/project/open", { path: "/work/open.dlstudio", data_path: data && data.length ? "/work/open.zip" : "" });
}

// ------------------------------------------------------------ api: 화면이 부르는 곳
async function api(path, body) {
  await ready;
  const method = body === undefined ? "GET" : "POST";
  try {
    if (path === "/api/train" && method === "POST") return startTrain(body);
    if (path.startsWith("/api/jobs/")) return jobView(path);
    if (path === "/api/activations") return activations(body);
    if (path === "/api/project/export") return exportModel();
    if (path === "/api/code/save") return codeSave(body);
    if (path === "/api/project/save") return projectSave(body);
    if (path === "/api/project/open") return await projectOpen(body);
    if (path === "/api/runs/restore") return await runRestore(body);
    if (path === "/api/predict") return await predictNew(body);
    if (path === "/api/preview") return await preview(body);
    if (path === "/api/anomaly/threshold") {                // 이상 탐지 기준값: Python 결과와 엔진 쪽 결과 둘 다
      const r = pyCall(method, path, body);
      if (W.trained && W.trained.results) W.trained.results.threshold = r.threshold;
      return r;
    }
    if (path.startsWith("/api/label/")) return labelCall(path.slice("/api/label/".length), body || {});
    if (path === "/api/data/load" && body && body.label_job) return await useLabelJob(body);
    if (path === "/api/data/load") { clearTrained(); if (body && body.example) await ensureExample(body.example); }
    const r = pyCall(method, path, body);
    if (method === "POST") {
      if (path === "/api/data/load") changed("data");
      else if (path === "/api/model/check" || path === "/api/task/select") changed("model");
    }
    if (path === "/api/state" && W.trained) Object.assign(r, { trained: true, trained_graph: W.trained.graph, results: W.trained.results });
    if (path === "/api/runs/delete") changed("train");
    return r;
  } catch (e) {
    if (e && e.message !== undefined && !(e instanceof Error)) throw e;                // 이미 화면 오류 모양
    throw pyError(e);
  }
}

// ------------------------------------------------------------ 파일 고르기 (Pyodide 파일 시스템 /data에 씀)
function pick(opts) {
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "file";
    if (opts.accept) inp.accept = opts.accept;
    if (opts.dir) { inp.webkitdirectory = true; inp.multiple = true; }
    inp.onchange = () => resolve(Array.from(inp.files || []));
    inp.click();
  });
}
async function writeFiles(files) {                      // → Pyodide 안의 경로 (폴더면 폴더 경로)
  await ready;
  const root = `/data/u${++W.n}`;
  let top = null;
  for (const f of files) {
    const rel = f.webkitRelativePath || f.name;
    const p = `${root}/${rel}`;
    W.py.FS.mkdirTree(p.slice(0, p.lastIndexOf("/")));
    W.py.FS.writeFile(p, new Uint8Array(await f.arrayBuffer()));
    if (f.webkitRelativePath) top = rel.split("/")[0];
  }
  return top ? `${root}/${top}` : `${root}/${files[0].name}`;
}
async function pickInto(inputId, opts) {
  const files = await pick(opts);
  if (!files.length) return false;
  const p = await writeFiles(files);
  const el = document.getElementById(inputId);
  el.dataset.path = p;                                  // app.js의 pathVal이 읽는 실제 위치
  el.value = opts.dir ? files[0].webkitRelativePath.split("/")[0] + ` (파일 ${files.length}개)` : files[0].name;
  return true;
}

// app.js를 불러온 뒤: 경로 칸을 '고르기'로, 열기는 파일 고르기로, 라벨링 단추는 안내로
function afterApp() {
  const $ = (id) => document.getElementById(id);
  for (const id of ["csv-path", "folder-path", "series-path", "series-label-file", "predict-path", "predict-folder", "predict-series"]) {
    const el = $(id);
    if (!el) continue;
    el.readOnly = true;
    el.placeholder = id === "series-label-file" ? "정답 표 (선택) — 단추로 고르기" : "단추로 고르세요 (브라우저 안에서만 읽음)";
  }
  const wrapBtn = (btnId, inputId, opts, label) => {
    const b = $(btnId);
    if (!b) return;
    const orig = b.onclick;
    if (label) b.textContent = label;
    b.onclick = async (ev) => { if (await pickInto(inputId, opts)) orig && orig.call(b, ev); };
  };
  const csvLoad = $("btn-csv") && $("btn-csv").onclick;
  wrapBtn("btn-csv", "csv-path", { accept: ".csv,.tsv,.txt" }, "CSV 파일 고르기");
  const panel = $("step-data"), dz = $("dropzone");      // CSV를 ② 데이터 화면에 끌어 놓기
  if (panel && dz && csvLoad) {
    dz.hidden = false;
    panel.addEventListener("dragover", (e) => { e.preventDefault(); panel.classList.add("drag"); });
    panel.addEventListener("dragleave", (e) => { if (!panel.contains(e.relatedTarget)) panel.classList.remove("drag"); });
    panel.addEventListener("drop", async (e) => {
      e.preventDefault();
      panel.classList.remove("drag");
      const f = Array.from((e.dataTransfer && e.dataTransfer.files) || []).find((x) => /\.(csv|tsv|txt)$/i.test(x.name));
      if (!f) return window.showMessage && window.showMessage("CSV 파일만 끌어 놓을 수 있습니다", "이미지 폴더는 [이미지 폴더 고르기]로 고르세요");
      const el = $("csv-path");
      el.dataset.path = await writeFiles([f]);
      el.value = f.name;
      csvLoad.call($("btn-csv"));
    });
  }
  wrapBtn("btn-folder", "folder-path", { dir: true }, "이미지 폴더 고르기");
  wrapBtn("btn-predict-csv", "predict-path", { accept: ".csv,.tsv,.txt" }, "CSV 파일 고르고 예측");      // ⑤ 예측
  wrapBtn("btn-predict-folder", "predict-folder", { dir: true }, "그림 폴더 고르고 예측");
  wrapBtn("btn-predict-series", "predict-series", { dir: true }, "폴더 고르고 예측");
  wrapBtn("btn-series-scan", "series-path", { dir: true }, "폴더 고르기");
  const lf = $("series-label-file");
  if (lf) {
    const b = document.createElement("button");
    b.textContent = "정답 표 고르기";
    b.onclick = async () => { if (await pickInto("series-label-file", { accept: ".csv,.tsv,.txt" })) lf.dispatchEvent(new Event("change")); };
    lf.after(b);
  }
  const dev = $("device");
  if (dev && dev.closest("label")) dev.closest("label").hidden = true;      // 웹은 브라우저가 알아서 (GPU가 있으면 WebGL)
  const ex = $("btn-export-pt");
  if (ex) ex.textContent = "학습한 모델 내려받기 (PyTorch 코드 + 가중치)";
  const be = $("btn-export");
  if (be) { be.textContent = "내려받기"; be.title = "학습한 모델을 PyTorch 코드와 가중치(.npz)로 내려받기"; }
  document.body.classList.add("web");
}

window.DLS_WEB = { api, afterApp, ready, writeFiles, recordBytes, openRecord, pick, pickInto,
  labelFileUrl, labelWork, labelData, labelOpen, labelOpenBundle,
  onChange: (f) => listeners.push(f), hasData: () => !!(W.studio && W.studio.data), openedName: () => W.openedName || "",
  download,
  pickProject: async () => { const f = await pick({ accept: ".dlstudio,.zip" }); return f.length ? writeFiles(f) : null; } };
})();

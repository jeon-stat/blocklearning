// DL studio 웹 버전(GitHub Pages)의 다리: 화면(app.js)의 api(path, body)를 서버 대신 브라우저 안에서 처리한다.
// - Python(dlstudio 패키지)은 Pyodide로 그대로 돌린다: 검사·데이터 읽기·과제·코드 생성·프로젝트 파일 (web.py)
// - 학습·샘플 따라가기는 TensorFlow.js 엔진(engine.js), 파일은 고르기(업로드 아님, 브라우저 안에만)·내려받기
// 데이터는 PC 밖으로 나가지 않는다. 로컬 실행(python -m dlstudio)에서는 이 파일을 쓰지 않는다.
(function () {
"use strict";
const PY_VERSION = "0.27.7";
const BASE = "src/dlstudio/";
const PY_FILES = ["__init__.py", "analysis.py", "blocks.py", "codegen.py", "core.py", "data.py", "graph.py", "jobs.py",
  "labeling.py", "labeling_sources.py", "project.py", "tasks.py", "web.py"];
const DATA_FILES = ["data/digits8.npz", "data/pneumonia28.npz"];
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
  await Promise.all(DATA_FILES.map(async (f) => py.FS.writeFile(`/app/dlstudio/${f}`, new Uint8Array(await (await fetch(BASE + f, { cache: "no-cache" })).arrayBuffer()))));
  py.FS.mkdirTree("/data");
  py.runPython("import sys; sys.path.insert(0, '/app')");
  W.mod = py.pyimport("dlstudio.web");
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
  const s = Object.assign({ epochs: 30, lr: 1e-3, batch_size: 32, seed: 0 }, body.settings || {});
  const job = { id: "w" + (++W.n), kind: "train", state: "running", progress: [], result: null, error: null, cancelled: false };
  W.jobs[job.id] = job;
  W.running = job;
  E().train(plan, data, s, (p) => { job.progress.push(p); job.progress.splice(0, Math.max(0, job.progress.length - 100)); return job.cancelled; },
    () => job.cancelled).then(({ net, result }) => {
    if (job.cancelled || W.data !== data) { net.dispose(); job.state = "cancelled"; return; }
    if (W.trained) W.trained.net.dispose();
    W.trained = { net, plan, graph: body.graph, settings: body.settings || {}, results: result };
    W.mod.set_trained(W.studio, JSON.stringify(body.graph), JSON.stringify(body.settings || {}), JSON.stringify(result));
    job.result = result;
    job.state = "done";
    changed("train");
  }).catch((e) => {
    job.state = job.cancelled ? "cancelled" : "failed";
    job.error = `${e.name || "Error"}: ${e.message || e}`;
  }).finally(() => { if (W.running === job) W.running = null; });
  return { id: job.id };
}
function jobView(path) {
  const id = path.split("/")[3], job = W.jobs[id];
  if (!job) throw { message: "작업을 찾을 수 없음" };
  if (path.endsWith("/cancel")) { job.cancelled = true; return { ok: true }; }
  return { id: job.id, kind: job.kind, state: job.state, progress: job.progress.slice(), result: job.result, error: job.error };
}

// ------------------------------------------------------------ 샘플 따라가기
function activations(body) {
  if (!W.trained) throw { message: "학습된 모델이 없음", hint: "④ 학습 단계에서 먼저 학습하세요" };
  const i = Number(body.sample_index || 0);
  const s = wrapped(W.mod.sample(W.studio, i));
  const d = getData(), net = W.trained.net;
  const x = E().toInput(Float32Array.from(s.x), 1, net.kind, d.length, d.image);
  const values = Object.fromEntries(Object.entries(net.trace(x)).map(([k, v]) => [k, E().thin(v)]));   // 층마다 줄이기
  const out = net.predict(x);
  x.dispose();
  let pred;
  if (d.task === "classification") { let b = 0; for (let j = 1; j < out.length; j++) if (out[j] > out[b]) b = j; pred = s.classes[b]; }
  else pred = out[0];
  return { index: i, label: s.label, prediction: pred, input: s.input, values, order: s.order };
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
function projectOpen(body) {
  const r = wrapped(W.mod.record_open(W.studio, body.path, body.data_path || ""));
  W.openedName = r.name || "";
  if (W.trained) { W.trained.net.dispose(); W.trained = null; }
  W.data = null;
  if (r.state) {
    const net = new (E().Net)(r.plan, 0);
    const st = {};
    for (const [k, v] of Object.entries(r.state)) st[k] = { shape: v.shape, data: v.int ? Int32Array.from(v.data) : Float32Array.from(v.data) };
    net.load(st);
    W.trained = { net, plan: r.plan, graph: r.response.graph, settings: r.response.train_settings || {}, results: r.response.results || {} };
  }
  return r.response;
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
// 기록 바이트 → 열기 (화면용 응답). data는 없어도 됨 (예시 데이터, 또는 프로젝트 zip 안에 데이터가 있음)
async function openRecord(project, data) {
  await ready;
  W.py.FS.mkdirTree("/work");
  W.py.FS.writeFile("/work/open.dlstudio", project);
  if (data && data.length) W.py.FS.writeFile("/work/open.zip", data);
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
    if (path === "/api/project/open") return projectOpen(body);
    if (path.startsWith("/api/label/")) throw { message: "라벨링 작업실은 웹 버전에 아직 없음", hint: "내 PC에서 python -m dlstudio로 실행하면 쓸 수 있습니다" };
    if (path === "/api/data/load") clearTrained();
    const r = pyCall(method, path, body);
    if (method === "POST") {
      if (path === "/api/data/load") changed("data");
      else if (path === "/api/model/check" || path === "/api/task/select") changed("model");
    }
    if (path === "/api/state" && W.trained) Object.assign(r, { trained: true, trained_graph: W.trained.graph, results: W.trained.results });
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
  for (const id of ["csv-path", "folder-path", "series-path", "series-label-file"]) {
    const el = $(id);
    if (!el) continue;
    el.readOnly = true;
    el.placeholder = id === "series-label-file" ? "정답 표 (선택) — 단추로 고르기" : "옆 단추로 고르세요 (브라우저 안에서만 읽음)";
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
  wrapBtn("btn-series-scan", "series-path", { dir: true }, "폴더 고르기");
  const lf = $("series-label-file");
  if (lf) {
    const b = document.createElement("button");
    b.textContent = "정답 표 고르기";
    b.onclick = async () => { if (await pickInto("series-label-file", { accept: ".csv,.tsv,.txt" })) lf.dispatchEvent(new Event("change")); };
    lf.after(b);
  }
  const lead = $("step-train") && $("step-train").querySelector(".lead");
  if (lead) lead.textContent = "이 브라우저 안에서 학습합니다 (데이터는 밖으로 나가지 않음). 데이터·모델이 크면 몇 분 걸릴 수 있습니다. 끝나면 PyTorch 코드와 가중치로 내려받을 수 있습니다.";
  const dev = $("device");
  if (dev && dev.closest("label")) dev.closest("label").hidden = true;      // 웹은 브라우저가 알아서 (GPU가 있으면 WebGL)
  const lb = $("btn-label");
  if (lb) lb.hidden = true;                                 // 라벨링 작업실은 내 PC 실행에만
  const ex = $("btn-export-pt");
  if (ex) ex.textContent = "학습한 모델 내려받기 (PyTorch 코드 + 가중치)";
  const be = $("btn-export");
  if (be) { be.textContent = "내려받기"; be.title = "학습한 모델을 PyTorch 코드와 가중치(.npz)로 내려받기"; }
  document.body.classList.add("web");
}

window.DLS_WEB = { api, afterApp, ready, writeFiles, recordBytes, openRecord,
  onChange: (f) => listeners.push(f), hasData: () => !!(W.studio && W.studio.data), openedName: () => W.openedName || "",
  download,
  pickProject: async () => { const f = await pick({ accept: ".dlstudio,.zip" }); return f.length ? writeFiles(f) : null; } };
})();

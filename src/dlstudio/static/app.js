// DL studio 화면: 단계 ①과제 → ②데이터 → ③모델 → ④학습(내보내기). 계산은 모두 로컬 Python(API)에서.
"use strict";

const $ = (id) => document.getElementById(id);
const F = StudioGraph;
const WEB = !!window.DLS_WEB;
const pathVal = (id) => (($(id).dataset.path || $(id).value) || "").trim();   // 웹: 칸에는 고른 이름, 실제 위치는 data-path                          // 웹 버전(GitHub Pages): 계산을 브라우저 안(web.js)에서
const st = { step: "data", summary: null, ed: null, graph: null, checked: null, graphJson: "", trainedJson: "", defaults: [],
  trained: false, trainJob: null, checkTimer: null, checkSeq: 0,
  catalog: [], fam: null, sub: null, tpl: null };
const inputKind = () => (st.summary && st.summary.image ? "image" : st.sub ? st.sub.input : "timeseries");
// 흑백 그림: 펼친 값(또는 [H][W]) → SVG 칸. 밝기는 그림 안의 최소~최대
function svgImage(vals, H, W, cell = 6) {
  const flat = Array.isArray(vals[0]) ? vals.flat() : vals;
  let lo = Math.min(...flat), hi = Math.max(...flat);
  if (hi - lo < 1e-12) hi = lo + 1;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W * cell}" height="${H * cell}" class="img" shape-rendering="crispEdges">`;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const g = Math.round(255 * (flat[y * W + x] - lo) / (hi - lo));           // 실제 밝기 그대로 (값이 클수록 밝게)
    s += `<rect x="${x}" y="${y}" width="1" height="1" fill="rgb(${g},${g},${g})"/>`;
  }
  return s + `</svg>`;
}

// ------------------------------------------------------------ 공통
async function api(path, body) {
  if (WEB) return window.DLS_WEB.api(path, body);
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: { message: `응답을 읽을 수 없음 (${r.status})` } }));
  if (!r.ok) throw j.error || { message: `오류 ${r.status}` };
  return j;
}

let msgTimer = null;
function hideMessage() { clearTimeout(msgTimer); $("message").hidden = true; }
function showMessage(msg, hint = "", detail = "", ok = false) {
  $("message").hidden = false;
  clearTimeout(msgTimer);                                // 떠 있는 알림: 성공 알림은 잠시 뒤 저절로 닫힘 (오류는 닫을 때까지)
  if (ok) msgTimer = setTimeout(() => ($("message").hidden = true), 4000);
  $("message").classList.toggle("ok", ok);
  $("msg-text").textContent = msg;
  $("msg-hint").textContent = hint ? "→ " + hint : "";
  $("msg-detail-box").hidden = !detail;
  $("msg-detail").textContent = detail || "";
}
const showError = (e) => showMessage(e.message || String(e), e.hint || "", e.detail || "");
$("msg-close").onclick = () => ($("message").hidden = true);

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (v) => (v === null || v === undefined ? "-" : (v * 100).toFixed(1) + "%");
const kv = (rows) => rows.map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${v}</div>`).join("");

// ------------------------------------------------------------ 그림 (SVG)
function pathOf(vals, x0, y0, w, h) {
  if (!vals.length) return "";
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  return vals.map((v, i) => `${(x0 + (i / Math.max(1, vals.length - 1)) * w).toFixed(1)},${(y0 + h - ((v - lo) / (hi - lo)) * h).toFixed(1)}`).join(" ");
}
function svgLine(vals, w = 240, h = 50, color = "#4f46e5") {
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><polyline fill="none" stroke="${color}" stroke-width="1.3" points="${pathOf(vals, 2, 2, w - 4, h - 4)}"/></svg>`;
}
function svgStrips(rows, w = 300, color = "#1d9e75") {
  const n = Math.min(rows.length, 16), sh = 9, h = n * sh + 4;
  let s = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`;
  for (let c = 0; c < n; c++) s += `<polyline fill="none" stroke="${color}" stroke-width="1" points="${pathOf(rows[c], 2, 2 + c * sh, w - 4, sh - 2)}"/>`;
  return s + `</svg>` + (rows.length > n ? `<div class="shape">… 채널 ${rows.length - n}개 더</div>` : "");
}
function svgBars(vals, w = 300, h = 46, labels = null, color = "#7f77dd") {
  const n = vals.length, bw = Math.max(2, Math.min(28, (w - 4) / n - 2));
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1;
  const base = 2 + (h - 16) * (hi / span);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`;
  vals.forEach((v, i) => {
    const x = 2 + i * (bw + 2), y = v >= 0 ? base - (v / span) * (h - 16) : base, hh = Math.abs(v / span) * (h - 16);
    s += `<rect x="${x}" y="${y.toFixed(1)}" width="${bw}" height="${Math.max(1, hh).toFixed(1)}" rx="2" fill="${color}"/>`;
    if (labels) s += `<text x="${x + bw / 2}" y="${h - 2}" font-size="10" text-anchor="middle" fill="#6b7280">${esc(labels[i])}</text>`;
  });
  return s + `</svg>`;
}

function svgScatter(y, pred, w = 300, h = 260) {       // 회귀: 가로 = 실제, 세로 = 예측, 대각선 = 완벽
  const L = 46, B = 34, T = 8, R = 8, all = y.concat(pred);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  const px = (v) => L + ((v - lo) / (hi - lo)) * (w - L - R), py = (v) => h - B - ((v - lo) / (hi - lo)) * (h - B - T);
  const f = (v) => Number(v.toPrecision(3));
  let s = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="scatter">` +
    `<rect x="${L}" y="${T}" width="${w - L - R}" height="${h - B - T}" fill="#fbfbfc" stroke="#e3e6eb"/>` +
    `<line x1="${px(lo)}" y1="${py(lo)}" x2="${px(hi)}" y2="${py(hi)}" stroke="#9aa3af" stroke-dasharray="4 3"/>`;
  y.forEach((v, i) => (s += `<circle cx="${px(v).toFixed(1)}" cy="${py(pred[i]).toFixed(1)}" r="2.6" fill="#4f46e5" fill-opacity=".55"/>`));
  s += `<text x="${L}" y="${h - 18}" font-size="11" fill="#6b7280">${f(lo)}</text><text x="${w - R}" y="${h - 18}" font-size="11" text-anchor="end" fill="#6b7280">${f(hi)}</text>` +
    `<text x="${(L + w) / 2}" y="${h - 4}" font-size="12" text-anchor="middle" fill="#1d2330">실제 값</text>` +
    `<text x="12" y="${(h - B) / 2}" font-size="12" text-anchor="middle" fill="#1d2330" transform="rotate(-90 12 ${(h - B) / 2})">예측 값</text>`;
  return s + `</svg>`;
}
function renderScatter(el, sc, mae, caption) {
  if (!sc) return ($(el).innerHTML = "");
  $(el).innerHTML = `<div class="scatter-box">${svgScatter(sc.y, sc.pred)}<div class="cm-cap">${esc(caption)} · 점이 점선(예측 = 실제)에 가까울수록 정확 · ` +
    `평균 오차(MAE) ${mae === null || mae === undefined ? "-" : Number(mae.toPrecision(4))}</div></div>`;
}

// ------------------------------------------------------------ 단계 이동
function go(step) {
  if (!$("step-" + step)) step = "task";
  st.step = step;
  document.querySelectorAll("main > .panel").forEach((el) => (el.hidden = el.id !== "step-" + step));
  document.querySelectorAll(".step").forEach((b) => b.classList.toggle("active", b.dataset.step === step));
  if (step === "model") ensureEditor();
  if (step === "predict") renderPredictStep();
  window.scrollTo(0, 0);
  const mainEl = document.querySelector("main");          // 사이드바 틀에서는 main이 스크롤됨
  if (mainEl) mainEl.scrollTop = 0;
}
function markSteps() {
  const mark = (name, done, stale) => {
    const b = document.querySelector(`.step[data-step="${name}"]`);
    b.classList.toggle("done", !!done && !stale);
    b.classList.toggle("stale", !!stale);
  };
  mark("task", st.tpl);
  mark("data", st.summary);
  mark("model", st.checked && st.checked.ok);
  mark("train", st.trained, st.trained && st.trainedJson !== st.graphJson);
  if (st.step === "predict") renderPredictStep();
  for (const id of ["btn-export", "btn-export-pt"]) {        // 학습 전에는 내려받을 모델이 없음
    $(id).disabled = !st.trained;
    if (!st.trained) $(id).title = "④ 학습을 마치면 쓸 수 있습니다";
  }
  renderGuard();
}
// ④ 학습: 데이터가 없거나 모델에 문제가 있으면 학습 단추를 막고, 무엇을 하면 되는지 바로 보여 줌
function trainBlock() {
  if (!st.summary) return { why: "데이터가 아직 없습니다.", fix: [["data", "② 데이터 넣으러 가기"]] };
  const c = st.checked;
  if (c && !c.ok && st.graphJson === sig(st.graph)) {
    const first = c.issues[0];
    return { why: `모델에 문제가 ${c.issues.length}개 있어 학습할 수 없습니다: ${first ? first.message : ""}`,
      fix: [["model", "③ 모델 고치러 가기"], ["tpl", "추천 모델로 되돌리기"]] };
  }
  return null;
}
function renderGuard() {
  const b = trainBlock();
  $("train-guard").hidden = !b;
  if (b) $("train-guard").innerHTML = `<span>⚠ ${esc(b.why)}</span>` + b.fix.map(([k, t]) => `<button data-fix="${k}">${esc(t)}</button>`).join("");
  $("btn-train").disabled = !!b || !!st.training;
}
$("train-guard").onclick = (e) => {
  const k = (e.target.closest("[data-fix]") || {}).dataset;
  if (!k || !k.fix) return;
  if (k.fix === "data") return go("data");
  if (k.fix === "model") {
    go("model");
    const first = st.checked && st.checked.issues.find((i) => i.node);
    if (first && st.ed) st.ed.select(first.node);
    return;
  }
  const t = st.tpl ? st.tpl.graph : st.defaults.length ? st.defaults[0].graph : null;
  if (!t) return go("model");
  setGraph(t);
  showMessage("추천 모델로 되돌렸습니다", "③ 모델에서 Ctrl+Z(되돌리기)로 고치던 모델을 되살릴 수 있습니다", "", true);
};
document.querySelectorAll(".step").forEach((b) => (b.onclick = () => go(b.dataset.step)));

// ------------------------------------------------------------ 과제 목록
async function loadCatalog() {                           // 과제 목록. 고른 것은 id로 유지
  const ids = [st.fam && st.fam.id, st.sub && st.sub.id, st.tpl && st.tpl.id];
  const cat = await api("/api/tasks");
  st.catalog = cat.families;
  st.defaults = cat.defaults || [];
  st.fam = st.catalog.find((f) => f.id === ids[0]) || null;
  st.sub = (st.fam && st.fam.subtasks.find((s) => s.id === ids[1])) || null;
  st.tpl = (st.sub && (st.sub.templates || []).find((t) => t.id === ids[2])) || null;
  renderTask();
  renderDataGuide();
  renderTemplateButtons();
  if (st.ed) st.ed.redrawPalette();
}
document.querySelectorAll(".go").forEach((b) => (b.onclick = () => go(b.dataset.go)));

// ------------------------------------------------------------ ① 과제
const readyFirst = (xs, ok) => xs.filter(ok).concat(xs.filter((x) => !ok(x)));   // '준비 중'은 뒤로
function renderTask() {
  $("btn-quick").disabled = !st.catalog.length;
  $("families").innerHTML = readyFirst(st.catalog, (f) => f.subtasks.some((s) => s.available)).map((f) => {
    const ready = f.subtasks.some((s) => s.available);
    return `<button class="tcard${st.fam === f ? " sel" : ""}${ready ? "" : " soon"}" data-f="${f.id}">` +
      `<div class="ti">${f.icon || ""}</div><div class="tn">${esc(f.name)}${ready ? "" : ` <span class="badge">준비 중</span>`}</div>` +
      `<div class="td">${esc(f.desc)}</div></button>`;
  }).join("");
  $("subtask-box").hidden = !st.fam;
  if (st.fam) $("subtasks").innerHTML = readyFirst(st.fam.subtasks, (s) => s.available).map((s) =>
    `<button class="tcard${st.sub === s ? " sel" : ""}${s.available ? "" : " soon"}" data-s="${s.id}">` +
    `<div class="tn">${esc(s.name)}${s.available ? "" : ` <span class="badge">준비 중</span>`}</div>` +
    `<div class="td">${esc(s.desc)}</div><div class="te">예: ${esc(s.examples_text)}</div></button>`).join("");
  $("template-box").hidden = !st.sub;
  if (st.sub) $("templates").innerHTML = st.sub.templates.map((t, i) =>
    `<button class="tcard${st.tpl === t ? " sel" : ""}" data-t="${t.id}">` +
    `<div class="tn">${esc(t.name)}${i === 0 ? ` <span class="badge rec">추천</span>` : ""}</div>` +
    `<div class="td">${esc(t.desc)}</div><div class="te">언제: ${esc(t.when)}</div></button>`).join("");
  markSteps();
}
$("families").onclick = (e) => {
  const b = e.target.closest("[data-f]");
  if (!b) return;
  const f = st.catalog.find((x) => x.id === b.dataset.f);
  if (!f.subtasks.some((s) => s.available)) return showMessage(`'${f.name}'은(는) 준비 중입니다`, "지금은 분류·회귀를 쓸 수 있습니다");
  if (st.fam !== f) { st.fam = f; st.sub = null; st.tpl = null; }
  st.autoTask = false;
  renderTask();
};
$("btn-skip-task").onclick = () => {                     // 과제 없이: 데이터를 보고 정함
  st.autoTask = true;
  if (st.sub) { st.fam = null; st.sub = null; st.tpl = null; renderTask(); renderDataGuide(); renderTemplateButtons(); }
  go("data");
};
$("subtasks").onclick = (e) => {
  const b = e.target.closest("[data-s]");
  if (!b) return;
  const s = st.fam.subtasks.find((x) => x.id === b.dataset.s);
  if (!s.available) return showMessage(`'${s.name}'은(는) 준비 중입니다`, "같은 큰 틀의 다른 과제를 고르세요");
  st.autoTask = false;
  if (st.sub !== s) chooseTemplate(st.fam, s, s.templates[0]);
};
$("templates").onclick = (e) => {
  const b = e.target.closest("[data-t]");
  if (!b) return;
  st.autoTask = false;
  chooseTemplate(st.fam, st.sub, st.sub.templates.find((x) => x.id === b.dataset.t));
};
$("btn-quick").onclick = async () => {                   // 예시 과제·데이터·추천 모델을 한 번에 → ④ 학습
  const fam = st.catalog.find((f) => f.subtasks.some((s) => s.available && (s.example_data || []).some((e) => e[0] === "ecg")));
  const sub = fam && fam.subtasks.find((s) => s.available && (s.example_data || []).some((e) => e[0] === "ecg"));
  if (!sub) return showMessage("예시를 찾지 못함", "아래에서 과제를 직접 고르세요");
  if (WEB && window.DLProjects && DLProjects.current() && st.summary) return DLProjects.startNew("quick");   // 연 프로젝트는 두고 새 프로젝트로
  $("btn-quick").disabled = true;
  try {
    await chooseTemplate(fam, sub, sub.templates[0]);
    if (!(await loadData(Object.assign(dataOptions(), { example: "ecg" })))) return;
    go("model");                                            // 모델 판을 한 번 그려 두고 (검사도 시작)
    go("train");
    showMessage("예시 데이터와 추천 모델을 준비했습니다", "[학습 시작]을 누르세요. 데이터는 ②, 모델은 ③에서 볼 수 있습니다", "", true);
  } finally { $("btn-quick").disabled = false; }
};
async function chooseTemplate(fam, sub, tpl) {
  try { await api("/api/task/select", { family: fam.id, subtask: sub.id, template: tpl.id }); } catch (e) { return showError(e); }
  const subChanged = st.sub !== sub;
  st.fam = fam; st.sub = sub; st.tpl = tpl;
  if (st.ed) setGraph(tpl.graph);
  else st.pendingGraph = tpl.graph;
  renderTask();
  renderDataGuide();
  renderTemplateButtons();
  if (st.ed) st.ed.redrawPalette();
  if (subChanged && !st.autoTask && st.summary && st.summary.task !== sub.task)
    showMessage(`지금 불러온 데이터는 ${st.summary.task === "classification" ? "분류" : "회귀"}용입니다`, "② 데이터에서 이 과제에 맞는 데이터를 다시 불러오세요");
}
function applySelection(sel) {                           // 저장된 선택(id)을 목록에서 찾아 화면에 반영 (블록은 그대로)
  if (!sel) return;
  const fam = st.catalog.find((f) => f.id === sel.family);
  const sub = fam && fam.subtasks.find((s) => s.id === sel.subtask);
  const tpl = sub && (sub.templates || []).find((t) => t.id === sel.template);
  if (!tpl) return;
  st.fam = fam; st.sub = sub; st.tpl = tpl;
  renderTask(); renderDataGuide(); renderTemplateButtons();
  if (st.ed) st.ed.redrawPalette();
}

// ------------------------------------------------------------ ② 데이터
const GUIDE = {
  timeseries: { classification: "한 행 = 신호 한 구간 (시점마다 값이 한 열씩), 마지막 열 = 정답 클래스 이름 (예: 정상, 이상).",
    regression: "한 행 = 신호 한 구간 (시점마다 값이 한 열씩), 마지막 열 = 맞힐 숫자 (예: 남은 수명)." },
  tabular: { classification: "한 행 = 샘플 하나, 열 = 측정 항목 (온도, 압력 …), 마지막 열 = 정답 클래스 이름 (예: 양품, 불량).",
    regression: "한 행 = 샘플 하나, 열 = 측정 항목 (온도, 압력 …), 마지막 열 = 맞힐 숫자 (예: 수율)." },
  image: { classification: "폴더 안에 클래스마다 하위 폴더를 두고 그림(png·jpg·bmp)을 넣으세요. 예: 사진/양품/*.png, 사진/불량/*.png. 흑백으로 바꾸고 고른 크기로 맞춥니다." },
};
function renderDataGuide() {
  const sub = st.sub;
  $("data-task-name").textContent = sub ? `${st.fam.name} · ${sub.name}` : "";
  const img = sub && sub.input === "image";
  $("data-guide").textContent = !sub || st.autoTask
    ? "과제를 고르지 않아도 됩니다. CSV·그림 폴더·예시 데이터를 넣으면 분류인지 회귀인지, 시계열·표·이미지 중 무엇인지 보고 과제와 추천 모델을 정해 드립니다. CSV는 한 행 = 샘플 하나, 마지막 열 = 정답."
    : img ? `이미지 폴더: ${GUIDE.image.classification} 픽셀 CSV(한 행 = 픽셀 값들 + 마지막 열 정답)도 됩니다. 처음이면 예시 데이터로 시작해 보세요.`
    : `CSV 파일: ${GUIDE[sub.input][sub.task]} 첫 줄이 열 이름이면 그대로 씁니다. 처음이면 예시 데이터로 시작해 보세요.`;
  const any = !sub || st.autoTask;                           // 과제 없이: 넣는 방법을 다 보임
  $("image-row").hidden = !img && !any;
  $("image-width-box").hidden = !img;
  const ts = (sub && sub.input === "timeseries") || any;     // 샘플마다 CSV 하나인 폴더: 시계열 과제에서만
  $("series-row").hidden = !ts;
  if (!ts) $("series-box").hidden = true;
  $("csv-path").placeholder = WEB ? "옆 단추로 CSV 파일을 고르세요 (브라우저 안에서만 읽음)" : img ? "C:\\data\\pixels.csv (픽셀 CSV)" : "C:\\data\\ecg.csv";
  st.examples = sub && !st.autoTask ? sub.example_data : allExamples();
  renderExamples();
  $("csv-format-body").innerHTML = csvFormat(any ? null : sub);
  if (sub && !st.autoTask) $("task").value = sub.task;
  $("normalize-text").textContent = img ? "정규화 (픽셀 전체 평균 0, 표준편차 1)"
    : inputKind() === "tabular" ? "정규화 (열마다 평균 0, 표준편차 1)" : "정규화 (시점별 평균 0, 표준편차 1)";
}
function allExamples() {                                 // 과제 없이: 쓸 수 있는 모든 예시 (과제 이름과 함께)
  const out = [], seen = new Set();
  for (const f of st.catalog) for (const s of f.subtasks) {
    if (!s.available) continue;
    for (const [k, name] of s.example_data || []) if (!seen.has(k)) { seen.add(k); out.push([k, `${name} · ${s.name}`]); }
  }
  return out.length ? out : [["ecg", "심전도 정상·이상 (의료)"]];
}
function subOfExample(k) {
  for (const f of st.catalog) for (const s of f.subtasks) if (s.available && (s.example_data || []).some((e) => e[0] === k)) return [f, s];
  return null;
}
function guessKind(s) {                                  // 이미지가 아니면: 열 이름이 항목 이름이고 열이 적으면 표, 아니면 시계열
  if (s.image) return "image";
  const generic = (s.features || []).every((n) => /^(열\s*\d+|[a-z_]*\d+)$/i.test(String(n).trim()));
  return s.length <= 12 || (s.length <= 40 && !generic) ? "tabular" : "timeseries";
}
function findSub(input, task) {
  for (const f of st.catalog) for (const s of f.subtasks) if (s.available && s.input === input && s.task === task) return [f, s];
  return null;
}
async function autoPick(s) {                             // 과제를 안 골랐으면 데이터를 보고 과제·추천 모델을 정함
  const ex = s.example && subOfExample(s.example);
  const hit = ex && ex[1].task === s.task ? ex : findSub(guessKind(s), s.task);
  if (!hit) { $("auto-task").hidden = true; return; }
  if (st.sub !== hit[1]) await chooseTemplate(hit[0], hit[1], hit[1].templates[0]);
}
function renderAutoTask() {
  const s = st.summary, sub = st.sub;
  $("auto-task").hidden = !st.autoTask || !sub || !s;
  if ($("auto-task").hidden) return;
  const other = !s.image && findSub(sub.input === "tabular" ? "timeseries" : "tabular", s.task);
  $("auto-task").innerHTML = `과제: <b>${esc(sub.name)}</b> — 데이터를 보고 정했습니다 (추천 모델: ${esc(st.tpl ? st.tpl.name : "")}).` +
    (other ? ` <button data-sub="${esc(other[1].id)}">${sub.input === "tabular" ? "시계열 (값이 시간 순서)" : "표 데이터 (열마다 다른 항목)"}로 바꾸기</button>` : "") +
    ` <button data-go="task">① 과제 직접 고르기</button>`;
}
$("auto-task").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.go) return go(b.dataset.go);
  const hit = findSub(st.sub.input === "tabular" ? "timeseries" : "tabular", st.summary.task);
  if (hit) chooseTemplate(hit[0], hit[1], hit[1].templates[0]).then(() => onData(st.summary));
};
function renderExamples() {                             // 불러온 예시는 ✓ 표시, 데이터가 없을 때만 눈에 띄게
  $("example-btns").innerHTML = (st.examples || []).map(([k, name]) => {
    const sel = st.summary && st.dataSrc === k;
    return `<button class="${sel ? "ex-sel" : st.summary ? "" : "primary"}" data-ex="${esc(k)}">${sel ? "✓ " : ""}예시: ${esc(name)}</button>`;
  }).join(" ");
}
function csvFormat(sub) {                                // CSV 모양 예시 (작은 표)
  const kind = sub ? sub.input : "timeseries", reg = sub && sub.task === "regression";
  const ex = {
    timeseries: [["t1", "t2", "t3", "…", "t140", reg ? "남은수명" : "정답"],
      ["0.12", "0.15", "0.11", "…", "0.09", reg ? "120" : "정상"], ["0.31", "0.29", "0.40", "…", "0.22", reg ? "35" : "이상"]],
    tabular: [["온도", "압력", "속도", reg ? "수율" : "정답"], ["182.1", "3.4", "1200", reg ? "92.5" : "양품"], ["190.6", "2.9", "1350", reg ? "81.0" : "불량"]],
    image: [["p1", "p2", "…", "p784", "정답"], ["0", "12", "…", "0", "숫자7"], ["0", "0", "…", "255", "숫자1"]],
  }[kind] || [];
  const table = `<table class="tab"><tr>${ex[0].map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` +
    ex.slice(1).map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("") + `</table>`;
  const what = { timeseries: "한 행 = 신호 한 구간, 열 = 시점마다의 값", tabular: "한 행 = 샘플 하나, 열 = 측정 항목",
    image: "한 행 = 그림 하나, 열 = 픽셀 값 (28×28이면 784열). 그림 파일 폴더도 됩니다" }[kind];
  return `<p>${esc(what)}, <b>마지막 열 = ${reg ? "맞힐 숫자" : "정답 이름"}</b>.</p>${table}` +
    `<p class="muted">첫 줄(열 이름)은 있어도 없어도 됩니다. 정답 열이 마지막이 아니면 [더 보기]에서 정답 열 번호를 바꾸세요. 엑셀에서는 [다른 이름으로 저장 → CSV UTF-8].</p>`;
}
$("example-btns").onclick = (e) => {
  const b = e.target.closest("[data-ex]");
  if (b) loadData(Object.assign(dataOptions(), { example: b.dataset.ex }));
};
function dataOptions() {
  const o = { normalize: $("normalize").checked, test_ratio: Number($("test-ratio").value) || 0.2 };
  if ($("label-col").value) o.label_col = Number($("label-col").value) - 1;
  if ($("task").value !== "auto") o.task = $("task").value;
  return o;
}
async function loadData(body) {                       // 성공하면 true
  $("data-issues").innerHTML = "";
  try {
    const r = await api("/api/data/load", body);
    if (!st.sub && st.catalog.length) st.autoTask = true;    // 과제 없이 넣은 데이터: 데이터를 보고 과제를 정함
    if (st.autoTask) { st.summary = r.summary; await autoPick(r.summary); }
    onData(r.summary);
    showMessage(`데이터를 불러왔음: 샘플 ${r.summary.n_train + r.summary.n_test}개`, "", "", true);
    return true;
  } catch (e) {
    showError(e);
    $("data-issues").innerHTML = (e.issues || []).map((i) => `<li>${esc(i.message)}<span class="hint">→ ${esc(i.hint)}</span></li>`).join("");
    return false;
  }
}
function onData(s) {
  st.summary = s;
  st.dataSrc = s.example || null;
  st.trained = false;
  $("data-summary").hidden = false;
  const task = s.task === "classification" ? `분류 (${s.classes.length}클래스)` : "회귀";
  const tab = inputKind() === "tabular" && !s.image;
  const cards = [["학습 샘플", s.n_train], ["시험 샘플", s.n_test],
    s.image ? ["크기 (픽셀)", `${s.image[0]}×${s.image[1]}`] : [tab ? "측정 항목 (열)" : "길이 (시점)", s.length], ["과제", task]];
  if (s.target_range) cards.push(["정답 범위", `${Number(s.target_range[0].toPrecision(3))} ~ ${Number(s.target_range[1].toPrecision(3))}`]);
  $("data-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("") +
    Object.entries(s.class_counts).map(([k, v]) => `<div class="card"><div class="k">클래스 '${esc(k)}'</div><div class="v">${v}</div></div>`).join("");
  const fmt = (v) => Number(Number(v).toPrecision(4));
  $("data-preview").classList.toggle("table", tab);
  $("data-preview").classList.toggle("thumbs", !!s.image);
  $("data-preview").innerHTML = s.image
    ? s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${svgImage(p.values, s.image[0], s.image[1], Math.max(3, Math.round(84 / s.image[1])))}</div>`).join("")
    : tab
    ? `<table class="tab"><tr>${s.features.map((f) => `<th>${esc(f)}</th>`).join("")}<th class="lab">정답</th></tr>` +
      s.preview.map((p) => `<tr>${p.values.map((v) => `<td>${fmt(v)}</td>`).join("")}<td class="lab">${esc(p.label)}</td></tr>`).join("") + `</table>`
    : s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${svgLine(p.values, 240, 60)}</div>`).join("");
  renderAutoTask();
  if (!st.autoTask && st.sub && st.sub.task !== s.task)
    showMessage(`불러온 데이터는 ${s.task === "classification" ? "분류" : "회귀"}용인데 고른 과제는 ${st.sub.name}입니다`, "① 과제를 데이터에 맞게 다시 고르거나, 맞는 데이터를 불러오세요");
  $("sample-index").max = s.n_test - 1;
  renderExamples();
  markSteps();
  if (st.ed) { st.ed.redrawPalette(); st.ed.redraw(); scheduleCheck(); }   // 입력 형태가 바뀌면 왼쪽 목록도
}
$("btn-csv").onclick = () => {
  if (!pathVal("csv-path")) return showMessage("CSV 파일 경로를 넣으세요", "예: C:\\data\\ecg.csv");
  const o = Object.assign(dataOptions(), { path: pathVal("csv-path") });
  if (st.sub && st.sub.input === "image") o.image_width = Number($("image-width").value) || 0;   // 비우면 정사각형
  loadData(o);
};
// 샘플마다 CSV 하나인 폴더 (시계열): 살펴보기 → 값 열·길이·정답 표 고르기 → 불러오기
const MODE_TEXT = { folders: "정답: 하위 폴더", file: "정답: 정답 표", none: "정답 없음" };
async function scanSeries() {
  const path = pathVal("series-path");
  if (!path) return showMessage("폴더 경로를 넣으세요", "예: C:\\data\\사이클 (사이클·환자마다 CSV 하나)");
  const label_file = pathVal("series-label-file");
  $("data-issues").innerHTML = "";
  try {
    const s = await api("/api/data/scan_folder", label_file ? { path, label_file } : { path });
    st.seriesScan = s;
    $("series-box").hidden = false;
    const [lo, med, hi] = s.lengths;
    $("series-info").innerHTML = `CSV <b>${s.n_files}</b>개 · 길이 <b>${lo}–${hi}</b> (중앙값 ${med}) · <b>${MODE_TEXT[s.mode]}</b>` +
      (s.classes ? ` (${s.classes.map(esc).join(", ")})` : "") +
      (s.skipped.length ? `<div class="muted">${s.skipped.map(esc).join(" · ")}</div>` : "");
    const prev = $("series-column").value, same = st.seriesScanPath === path;   // 같은 폴더를 다시 볼 때만 고른 값 유지
    st.seriesScanPath = path;
    $("series-column").innerHTML = s.columns.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("series-column").value = same && s.columns.includes(prev) ? prev : s.default_column;
    if (!same || !$("series-length").value) $("series-length").value = s.default_length;
    const none = s.mode === "none";
    $("btn-series-load").hidden = none;
    $("series-guide").hidden = !none;
    if (none) $("series-guide").innerHTML = `<b>정답(클래스)이 없어 학습할 수 없습니다.</b> 아래 중 하나로 정답을 알려 주세요.<ol>` +
      `<li>클래스마다 하위 폴더로 나누기: <code>폴더/정상/*.csv</code>, <code>폴더/이상/*.csv</code></li>` +
      `<li>정답 표 CSV를 만들어 위 '정답 표 CSV' 칸에 넣기: 첫 열 = 파일 이름, 둘째 열 = 정답 (예: <code>Cycle_10,정상</code>)</li>` +
      `<li><a href="/label?folder=${encodeURIComponent(path)}&column=${encodeURIComponent($("series-column").value)}" target="_blank">` +
      `라벨링 작업실에서 정답 붙이기</a> (새 탭): 격자로 훑고, 규칙으로 한꺼번에, 하나씩 크게 보며</li></ol>`;
  } catch (e) {
    showError(e);
    $("data-issues").innerHTML = (e.issues || []).map((i) => `<li>${esc(i.message)}<span class="hint">→ ${esc(i.hint)}</span></li>`).join("");
  }
}
$("btn-series-scan").onclick = scanSeries;
$("btn-label").onclick = () => (WEB ? showMessage("라벨링 작업실은 웹 버전에 아직 없음", "내 PC에서 python -m dlstudio로 실행하면 쓸 수 있습니다")
  : window.open("/label", "_blank"));
$("series-path").oninput = () => {                        // 폴더를 바꾸면 이전 살펴보기 결과는 버림
  st.seriesScan = null;
  $("series-box").hidden = true;
};
$("series-label-file").onchange = () => { if (pathVal("series-path")) scanSeries(); };
$("btn-series-load").onclick = async () => {
  const o = Object.assign(dataOptions(), { series_folder: pathVal("series-path"), column: $("series-column").value,
    length: Number($("series-length").value) || (st.seriesScan && st.seriesScan.default_length) || 64,
    resample: $("series-resample").value });
  const lf = pathVal("series-label-file");
  if (lf) o.label_file = lf;
  delete o.label_col;                                      // 폴더 방식에는 '정답 열 번호'가 없음
  if (!(await loadData(o))) return;                      // 실패하면 이전 데이터 카드에 덧붙이지 않음
  if (st.summary && st.seriesScan) {
    const [lo, , hi] = st.seriesScan.lengths;
    $("data-cards").insertAdjacentHTML("beforeend", `<div class="card"><div class="k">원래 길이</div><div class="v">${lo}–${hi} → ${st.summary.length}</div></div>`);
  }
};
$("btn-folder").onclick = () => {
  if (!pathVal("folder-path")) return showMessage("이미지 폴더 경로를 넣으세요", "예: C:\\data\\사진 (안에 클래스별 하위 폴더)");
  loadData(Object.assign(dataOptions(), { folder: pathVal("folder-path"), size: Number($("folder-size").value) }));
};

// ------------------------------------------------------------ ② 모델
function ensureEditor() {
  if (!st.ed) {
    st.ed = F.mount({ canvas: $("canvas"), palette: $("palette"), settings: $("settings") },
      { onChange: (g) => { st.graph = g; scheduleCheck(); }, getSummary: () => st.summary,
        onHint: (msg) => showMessage(msg, "", "", true), getInput: inputKind, onPreview: (id) => { st.previewNode = id; runPreview(); },
        onHistory: (u, r) => { $("btn-undo").disabled = !u; $("btn-redo").disabled = !r; } });
    let es = "ortho";
    try { es = localStorage.getItem("dlstudio-edge-style") || "ortho"; } catch (e) { /* 없음 */ }
    $("edge-style").value = es;
    let fd = "v";
    try { fd = localStorage.getItem("dlstudio-flow-dir") || "v"; } catch (e) { /* 없음 */ }
    $("flow-dir").value = fd;
    wrapText();
    st.ed.setFlowDir(fd, false);
    let aw = false;
    try { aw = localStorage.getItem("dlstudio-auto-wrap") === "1"; } catch (e) { /* 없음 */ }
    $("auto-wrap").checked = aw;
    st.ed.setWrap(aw);
    st.ed.setEdgeStyle(es);
    setGraph(st.pendingGraph || (st.tpl ? st.tpl.graph : st.defaults.length ? st.defaults[0].graph : EMPTY_GRAPH));
  }
}
const sig = (g) => JSON.stringify(F.stripView(g || { nodes: [] }));   // 모델 비교: 화면용 값(위치·크기)은 빼고
const EMPTY_GRAPH = { version: 2, nodes: [{ id: "input", type: "input" }, { id: "n1", type: "output", inputs: ["input"] }] };
function setGraph(g) {
  ensureEditor();
  st.graph = JSON.parse(JSON.stringify(g));
  st.ed.set(st.graph);
  scheduleCheck();
}
function scheduleCheck() {
  st.checkSeq++;                                         // 바뀌기 전 그래프의 검사 결과가 늦게 와도 그리지 않음
  clearTimeout(st.checkTimer);
  st.checkTimer = setTimeout(runCheck, 500);
}
async function runCheck() {
  if (!st.ed) return;
  const graph = st.graph;
  st.graphJson = sig(graph);
  if (!st.summary) {
    $("check-status").textContent = "② 데이터를 먼저 불러오세요";
    return;
  }
  const seq = ++st.checkSeq;
  $("check-status").textContent = "계산 중…";
  try {
    const c = await api("/api/model/check", { graph });
    if (seq !== st.checkSeq) return;
    st.checked = c;
    st.ed.decorate(c);
    $("check-status").textContent = c.ok ? "학습할 수 있음" : `문제 ${c.issues.length}개`;
    renderIssues(c);
    renderPanel(c.panel);
    if (st.previewNode && !$("preview-box").hidden) runPreview();          // 미리보기를 켜 두었으면 고친 모델로 다시
  } catch (e) {
    if (seq === st.checkSeq) { $("check-status").textContent = ""; showError(e); }
  }
  markSteps();
}
function renderIssues(c) {
  $("model-issues").innerHTML = c.issues.length
    ? c.issues.map((i) => `<li data-node="${esc(i.node)}">${esc(i.message)}<span class="hint">→ ${esc(i.hint)}</span></li>`).join("")
    : `<li class="muted">없음</li>`;
}
$("model-issues").onclick = (e) => {                     // 문제를 누르면 그 상자를 선택
  const li = e.target.closest("[data-node]");
  if (li && st.ed && li.dataset.node) st.ed.select(li.dataset.node);
};
function renderWarnings(el, warns) {
  $(el).innerHTML = (warns || []).map((w) => `<li>⚠ ${esc(w.message)}<span class="hint">→ ${esc(w.hint)}</span></li>`).join("");
}
function renderConfusion(el, cm, recall, caption, pickable = false) {
  $("cm-pick").hidden = true;
  if (!cm) return ($(el).innerHTML = "");
  const names = st.summary.classes;
  let h = `<table class="cm${pickable ? " pick" : ""}"><tr><th>정답↓ 예측→</th>${names.map((n) => `<th>${esc(n)}</th>`).join("")}<th>재현율</th></tr>`;
  cm.forEach((row, i) => {
    h += `<tr><th>${esc(names[i])}</th>${row.map((v, j) => `<td class="${i === j ? "diag" : ""}"${pickable && v ? ` data-cell="${i},${j}" title="이 칸의 샘플 보기"` : ""}>${v}</td>`).join("")}` +
      `<td>${recall[i] === null ? "-" : pct(recall[i])}</td></tr>`;
  });
  $(el).innerHTML = h + `</table><div class="cm-cap">${esc(caption)} · 재현율 = 실제 그 클래스 중 맞힌 비율 (결함·이상을 놓치지 않는지)` +
    (pickable ? ` · <b>칸을 누르면 그 샘플들을 봅니다</b> (대각선 밖 = 틀린 샘플)` : "") + `</div>`;
}
$("train-confusion").onclick = async (e) => {            // 혼동 행렬 칸 → 그 칸의 시험 샘플들
  const td = e.target.closest("[data-cell]");
  if (!td) return;
  const [i, j] = td.dataset.cell.split(",").map(Number), names = st.summary.classes;
  try {
    const r = await api("/api/samples", { cell: [i, j] });
    const im = st.summary.image, tab = inputKind() === "tabular" && !im;
    const pic = (v) => (im ? svgImage(v, im[0], im[1], Math.max(2, Math.round(40 / im[1]))) : tab ? "" : svgLine(v, 120, 34));
    $("cm-pick").hidden = false;
    $("cm-pick").innerHTML = `<div class="cm-pick-head"><b>정답 '${esc(names[i])}' → 예측 '${esc(names[j])}'</b> ${i === j ? "(맞힌 샘플)" : "(틀린 샘플)"} ${r.indices.length}개` +
      (r.indices.length > r.items.length ? ` · 처음 ${r.items.length}개` : "") + ` <button class="x" data-close="1" aria-label="닫기">×</button></div>` +
      `<div class="cm-items">${r.items.map((it) => `<button data-idx="${it.index}" title="시험 샘플 ${it.index} 따라가 보기">${pic(it.input)}<span>#${it.index}</span></button>`).join("")}</div>` +
      `<div class="muted">누르면 아래 '샘플 하나 따라가 보기'에서 그 샘플이 모델을 지나는 모습을 봅니다</div>`;
  } catch (err) { showError(err); }
};
$("cm-pick").onclick = (e) => {
  if (e.target.closest("[data-close]")) return ($("cm-pick").hidden = true);
  const b = e.target.closest("[data-idx]");
  if (!b) return;
  $("sample-index").value = b.dataset.idx;
  showSample();
  $("sample-box").scrollIntoView({ behavior: "smooth", block: "start" });
};
function renderPanel(p) {
  renderWarnings("panel-warnings", p && p.warnings);
  if (!p) return ($("panel").innerHTML = `<p class="muted">블록 문제를 고치면 계산됩니다</p>`);
  if (!p.ok) return ($("panel").innerHTML = `<p class="full">${esc(p.message)}</p>`);
  $("panel").innerHTML = kv([["파라미터 (배우는 값)", p.params.toLocaleString()],
    ["크기", `${p.mb < 0.1 ? p.mb.toFixed(3) : p.mb.toFixed(1)} MB`]]);
}
function orderItems(order, graph, depth = 0, out = []) {   // 위상 순서대로 [id, 노드, 묶음 깊이] (묶음 안은 그 안 순서)
  for (const x of order || []) {
    const id = Array.isArray(x) ? x[0] : x, f = F.findNode(graph, id);
    if (f) out.push([id, f.node, depth]);
    if (Array.isArray(x)) orderItems(x[1], graph, depth + 1, out);
  }
  return out;
}
const blockName = (b) => F.label(b, st.summary);
function renderTemplateButtons() {                       // 고른 과제의 대표 모델들 (과제를 안 골랐으면 기본 예시)
  const src = st.sub ? st.sub.templates : st.defaults;
  const list = src.map((t) => [t.id, t.name, t.graph]);
  st.tplList = list;
  $("tpl-btns").innerHTML = src.map((t) => `<button data-tpl="${esc(t.id)}" title="${esc(`${t.desc}\n언제: ${t.when}`)}">${esc(t.name)}</button>`).join(" ");
}
$("tpl-btns").onclick = (e) => {
  const b = e.target.closest("[data-tpl]");
  if (!b) return;
  const t = st.sub && st.sub.templates.find((x) => x.id === b.dataset.tpl);
  if (t) return chooseTemplate(st.fam, st.sub, t);
  setGraph(st.tplList.find((x) => x[0] === b.dataset.tpl)[2]);
};
$("btn-clear").onclick = () => { setGraph(EMPTY_GRAPH); showMessage("빈 모델로 새로 시작했습니다", "왼쪽 층을 끌어 Input과 Output 사이 선에 놓으세요. Ctrl+Z(되돌리기)로 이전 모델을 되살릴 수 있습니다", "", true); };
document.addEventListener("click", (e) => {                // 보기 설정: 바깥을 누르면 닫음
  const d = document.querySelector(".viewopt[open]");
  if (d && !d.contains(e.target)) d.open = false;
});
document.addEventListener("keydown", (e) => { const d = document.querySelector(".viewopt[open]"); if (e.key === "Escape" && d) d.open = false; });
$("btn-autolayout").onclick = () => st.ed && st.ed.autoLayout();
function wrapText() {                                    // 세로(위→아래)는 열바꿈, 가로는 줄바꿈
  $("auto-wrap-text").textContent = $("flow-dir").value === "h" ? "자동 줄바꿈 (판을 넘으면 아래 줄로)" : "자동 열바꿈 (판을 넘으면 오른쪽 열로)";
}
$("auto-wrap").onchange = () => {                        // 자동 줄바꿈: 판 폭(세로면 높이)을 넘으면 다음 줄로
  try { localStorage.setItem("dlstudio-auto-wrap", $("auto-wrap").checked ? "1" : "0"); } catch (e) { /* 저장 안 돼도 됨 */ }
  if (st.ed) st.ed.setWrap($("auto-wrap").checked);
};
$("flow-dir").onchange = () => {                         // 방향은 이 브라우저에 기억
  try { localStorage.setItem("dlstudio-flow-dir", $("flow-dir").value); } catch (e) { /* 저장 안 돼도 됨 */ }
  if (st.ed) st.ed.setFlowDir($("flow-dir").value, true);
  wrapText();
};
$("edge-style").onchange = () => {                       // 선 모양은 이 브라우저에 기억 (모델에는 영향 없음)
  try { localStorage.setItem("dlstudio-edge-style", $("edge-style").value); } catch (e) { /* 저장 안 돼도 됨 */ }
  if (st.ed) st.ed.setEdgeStyle($("edge-style").value);
};
$("btn-undo").onclick = () => st.ed && st.ed.undo();
$("btn-redo").onclick = () => st.ed && st.ed.redo();

// ------------------------------------------------------------ ③ 학습
function trainSettings() {
  return { epochs: Number($("epochs").value) || 30, lr: Number($("lr").value) || 0.001, batch_size: Number($("batch").value) || 32,
    device: $("device").value };
}
async function pollJob(id, onTick) {
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    onTick(j);
    if (j.state !== "running") return j;
    await new Promise((r) => setTimeout(r, 700));
  }
}
function drawCurve(progress, metricName) {           // 왼쪽 축 = 시험 성적 (0~100% 또는 R² 0~1), 오른쪽 축 = loss (가장 큰 값 기준)
  const W = 560, H = 220, L = 46, R = 46, T = 30, B = 26, pw = W - L - R, ph = H - T - B;
  const n = progress.length, acc = metricName !== "r2";
  const yv = (f) => T + (1 - f) * ph;
  let s = "";
  for (const f of [0, 0.5, 1]) {
    s += `<line x1="${L}" y1="${yv(f)}" x2="${L + pw}" y2="${yv(f)}" stroke="#e3e6eb"/>` +
      `<text x="${L - 6}" y="${yv(f) + 4}" font-size="11" text-anchor="end" fill="#4f46e5">${acc ? f * 100 + "%" : f}</text>`;
  }
  const loss = progress.map((p) => p.loss), lmax = Math.max(0, ...loss) || 1;
  s += `<text x="${L + pw + 6}" y="${yv(1) + 4}" font-size="11" fill="#f97316">${Number(lmax.toPrecision(2))}</text>` +
    `<text x="${L + pw + 6}" y="${yv(0) + 4}" font-size="11" fill="#f97316">0</text>`;
  if (!n) s += `<text x="${L + pw / 2}" y="${T + ph / 2}" font-size="13" text-anchor="middle" fill="#94a3b8">[학습 시작]을 누르면 epoch마다 곡선이 그려집니다</text>`;
  if (n) {
    const xs = (i) => L + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const lp = loss.map((v, i) => `${xs(i).toFixed(1)},${yv(v / lmax).toFixed(1)}`);
    const mp = progress.map((p, i) => `${xs(i).toFixed(1)},${yv(Math.max(0, Math.min(1, p.test_metric))).toFixed(1)}`);
    s += `<polyline fill="none" stroke="#f97316" stroke-width="2" points="${lp.join(" ")}"/>` +
      `<polyline fill="none" stroke="#4f46e5" stroke-width="2" points="${mp.join(" ")}"/>` +
      `<text x="${L}" y="${H - 8}" font-size="11" fill="#6b7280">epoch 1</text>` +
      `<text x="${L + pw}" y="${H - 8}" font-size="11" text-anchor="end" fill="#6b7280">${n}</text>`;
  }
  s += `<line x1="${L}" y1="12" x2="${L + 18}" y2="12" stroke="#4f46e5" stroke-width="2"/>` +
    `<text x="${L + 24}" y="16" font-size="12" fill="#4f46e5">시험 ${acc ? "정확도 (왼쪽 축, 높을수록 좋음)" : "R² (왼쪽 축, 1에 가까울수록 좋음)"}</text>` +
    `<line x1="${L + 300}" y1="12" x2="${L + 318}" y2="12" stroke="#f97316" stroke-width="2"/>` +
    `<text x="${L + 324}" y="16" font-size="12" fill="#f97316">loss (오른쪽 축, 낮을수록 좋음)</text>`;
  $("curve").innerHTML = s;
}
$("btn-train").onclick = async () => {
  if (!st.summary) return showMessage("데이터가 없음", "② 데이터 단계에서 먼저 불러오세요");
  ensureEditor();
  const graph = st.graph;
  const blocked = trainBlock();
  if (blocked) return showMessage(blocked.why, blocked.fix[0][1]);
  hideMessage();                                              // 지난 오류 알림은 지움 (이번 결과와 헷갈리지 않게)
  $("train-status").textContent = "학습 준비 중…";
  st.training = true;
  $("btn-train").disabled = true;                              // 두 번 눌러 두 학습이 돌지 않게
  try {
    const { id } = await api("/api/train", { graph, settings: trainSettings() });
    st.trainJob = id;
    const json = sig(graph);
    $("train-status").textContent = "학습 중…";
    const metric = st.summary.task === "classification" ? "accuracy" : "r2";
    const j = await pollJob(id, (j) => {
      drawCurve(j.progress, metric);
      const last = j.progress[j.progress.length - 1];
      if (last) $("train-status").textContent = `epoch ${last.epoch} · loss ${last.loss.toFixed(4)} · 시험 ${metric === "r2" ? last.test_metric.toFixed(3) : pct(last.test_metric)}`;
    });
    if (j.state === "done") {
      st.trained = true;
      st.trainedJson = json;
      st.trainedGraph = graph;
      renderTrainResult(j.result);
      $("sample-box").hidden = false;
      showSample();
      $("train-status").textContent = "끝남";
      loadRuns();
      const r = j.result;
      showMessage(`학습 끝남: 시험 ${r.metric_name === "r2" ? "R² " + r.test_metric.toFixed(3) : "정확도 " + pct(r.test_metric)}`,
        "아래 결과와 '샘플 하나 따라가 보기'를 보세요", "", true);
    } else if (j.state === "cancelled") $("train-status").textContent = "멈춤";
    else { $("train-status").textContent = "실패"; showMessage("학습 실패", "학습률을 낮추거나 블록을 바꿔 보세요", j.error); }
  } catch (e) { $("train-status").textContent = ""; showError(e); }
  st.training = false;
  markSteps();
};
$("btn-train-stop").onclick = () => st.trainJob && api(`/api/jobs/${st.trainJob}/cancel`, {}).catch(showError);
function renderTrainResult(r) {
  const isAcc = r.metric_name === "accuracy";
  const f = (v) => (v === null ? "-" : isAcc ? pct(v) : v.toFixed(3));
  const m = (v) => (v === null || v === undefined ? "-" : Number(v.toPrecision(4)));
  const has = (v) => v !== null && v !== undefined;
  const base = isAcc ? baselineOf(r) : null;
  const rows = [["Epochs", r.epochs_done], [`시험 ${isAcc ? "정확도" : "R²"}`, `<b>${f(r.test_metric)}</b>`]];
  if (has(r.train_metric)) rows.push([`학습 ${isAcc ? "정확도" : "R²"}`, f(r.train_metric)]);
  if (has(base)) rows.push(["찍기 기준", `${pct(base)} <span class="muted">(가장 많은 클래스만 답할 때)</span>`]);
  if (!isAcc) rows.push(["시험 평균 오차 MAE", m(r.mae)], ["찍기 기준", `R² 0 <span class="muted">(늘 평균값만 답할 때)</span>`]);
  const v = verdict(r, base);
  $("train-result").innerHTML = kv(rows) + (v ? `<div class="verdict ${v[0]}"><b>${esc(v[1])}</b><div>다음: ${esc(v[2])}</div></div>` : "");
  renderWarnings("train-warnings", r.warnings);
  renderConfusion("train-confusion", r.confusion, r.recall, "시험 데이터", Array.isArray(r.pred));
  renderScatter("train-scatter", r.scatter, r.mae, "시험 데이터");
}
function baselineOf(r) {                                // 분류: 시험 데이터에서 가장 많은 클래스만 답할 때 정확도
  if (!r.confusion) return null;
  const rows = r.confusion.map((row) => row.reduce((a, b) => a + b, 0)), n = rows.reduce((a, b) => a + b, 0);
  return n ? Math.max(...rows) / n : null;
}
function verdict(r, base) {                              // 결과를 쉬운 말로 + 다음에 해 볼 일
  const t = r.test_metric, tr = r.train_metric, acc = r.metric_name === "accuracy";
  if (t === null || t === undefined) return null;
  const show = (v) => (acc ? pct(v) : v.toFixed(2));
  const gap = tr === null || tr === undefined ? 0 : tr - t;
  if (acc ? base !== null && t <= base + 0.02 : t < 0.1)
    return ["bad", acc ? `찍기 기준(${pct(base)})과 비슷합니다. 모델이 아직 거의 배우지 못했습니다.`
      : `R²가 ${t.toFixed(2)}로, 늘 평균값만 답하는 것(R² 0)과 비슷하거나 못합니다.`,
      "Epochs를 늘리거나, [더 보기]에서 Learning rate를 바꿔 보거나(0.01 또는 0.0001), 다른 대표 모델을 써 보세요."];
  if (gap > (acc ? 0.1 : 0.2))
    return ["warn", `학습 데이터는 ${show(tr)}인데 시험 데이터는 ${show(t)}입니다. 학습 데이터를 외운 것일 수 있습니다 (과적합).`,
      "③ 모델에 Dropout 블록을 넣거나, 층·채널을 줄이거나, Epochs를 줄이거나, 데이터를 더 모아 보세요."];
  if (acc ? t >= 0.9 : t >= 0.8)
    return ["good", acc ? `학습에 쓰지 않은 시험 데이터의 ${pct(t)}를 맞혔습니다.` : `시험 데이터에서 R² ${t.toFixed(2)}로 잘 맞힙니다 (1이면 완벽).`,
      `${acc ? "혼동 행렬에서 어떤 클래스를 헷갈리는지" : "산점도에서 어디가 크게 틀리는지"} 보고, 위 [내보내기]로 학습한 모델을 받아 쓸 수 있습니다.`];
  return ["warn", acc ? `시험 데이터의 ${pct(t)}를 맞혔습니다 (찍기 기준 ${pct(base)}). 배우긴 했지만 더 좋아질 수 있습니다.`
    : `시험 데이터에서 R² ${t.toFixed(2)}입니다 (1이면 완벽, 0이면 평균값만 답하는 수준).`,
    "Epochs를 늘리거나, 다른 대표 모델이나 블록을 바꿔 시험 성적이 오르는지 보세요."];
}
function valuePic(v, b, w = 600) {                     // 층 하나의 값 그림: (C,H,W)는 채널마다 작은 그림, (C,L)은 줄, 나머지는 막대
  if (Array.isArray(v[0]) && Array.isArray(v[0][0])) {
    const n = Math.min(16, v.length), h = v[0].length, ww = v[0][0].length, cell = Math.max(2, Math.round(48 / ww));
    return `<div class="chans">${v.slice(0, n).map((ch) => svgImage(ch, h, ww, cell)).join("")}</div>` +
      (v.length > n ? `<div class="shape">… 채널 ${v.length - n}개 더</div>` : "");
  }
  if (Array.isArray(v[0])) return svgStrips(v, w);
  if (b.type === "output" && st.summary.task === "classification") {
    const m = Math.max(...v), ex = v.map((x) => Math.exp(x - m)), sum = ex.reduce((x, y) => x + y, 0);
    return svgBars(ex.map((x) => x / sum), Math.max(120, v.length * 60), 60, st.summary.classes, "#4f46e5");
  }
  return svgBars(v, Math.min(w, v.length * 14 + 10));
}
const shapeOf = (v) => (Array.isArray(v) ? [v.length].concat(shapeOf(v[0]) || []) : null);

// ------------------------------------------------------------ ③ 블록 미리보기 (누를 때만)
async function runPreview() {
  const id = st.previewNode, f = id && F.findNode(st.graph, id);
  if (!f) { $("preview-box").hidden = true; return; }
  $("preview-box").hidden = false;
  $("preview-name").textContent = "· " + blockName(f.node);
  if (!st.summary) { $("preview-body").innerHTML = `<p class="muted">② 데이터를 먼저 불러오세요</p>`; return; }
  if (st.checked && !st.checked.ok) { $("preview-body").innerHTML = `<p class="muted">모델의 문제를 고치면 보입니다</p>`; return; }
  $("preview-body").innerHTML = `<p class="muted">계산 중…</p>`;
  try {
    const r = await api("/api/preview", { graph: st.graph, node: id });
    if (st.previewNode !== id) return;
    const shape = shapeOf(r.value) || [];
    $("preview-body").innerHTML = valuePic(r.value, f.node, 300) +
      `<p class="muted">학습 샘플 하나(정답 ${esc(r.label)})가 이 층을 지난 값 · 모양 (${shape.join(", ")})</p>` +
      `<p class="muted">${r.trained ? "학습한 모델의 값입니다." : "아직 학습 전이라 값은 무작위입니다. 값이 아니라 모양이 어떻게 바뀌는지 보세요."}</p>`;
  } catch (e) { $("preview-body").innerHTML = `<p class="err">${esc(e.message || e)}</p>`; }
}
$("preview-close").onclick = () => { $("preview-box").hidden = true; st.previewNode = null; };

// ------------------------------------------------------------ ④ 학습 기록
async function loadRuns() {
  try { st.runs = (await api("/api/runs")).runs; } catch (e) { st.runs = []; }
  renderRuns();
}
function modelName(g) {                                  // 대표 모델과 같으면 그 이름, 아니면 '직접 만든 모델'
  const s = sig(g);
  for (const f of st.catalog) for (const sub of f.subtasks) for (const t of sub.templates || []) if (sig(t.graph) === s) return t.name;
  for (const t of st.defaults) if (sig(t.graph) === s) return t.name;
  const n = (g.nodes || []).filter((x) => x.type !== "input" && x.type !== "output").length;
  return `직접 만든 모델 (블록 ${n}개)`;
}
function renderRuns() {
  const runs = st.runs || [];
  $("runs-box").hidden = !runs.length;
  if (!runs.length) return;
  const ok = runs.filter((r) => r.data_ok && r.results.test_metric !== null && r.results.test_metric !== undefined);
  const best = ok.length ? ok.reduce((a, b) => (b.results.test_metric > a.results.test_metric ? b : a)) : null;
  const when = (t) => {
    const d = new Date(t), today = new Date().toDateString() === d.toDateString();
    return isNaN(d) ? "" : d.toLocaleString("ko-KR", today ? { hour: "2-digit", minute: "2-digit" } : { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  };
  const f = (r, v) => (v === null || v === undefined ? "-" : r.results.metric_name === "r2" ? "R² " + v.toFixed(3) : pct(v));
  $("runs").innerHTML = `<table class="tab runs"><tr><th>#</th><th>언제</th><th>모델</th><th>Epochs</th><th>Learning rate</th><th>시험</th><th>학습</th><th></th></tr>` +
    runs.slice().reverse().map((r) => `<tr class="${r.current ? "cur" : ""}${r.data_ok ? "" : " other"}">` +
      `<td>${r.n}</td><td>${esc(when(r.time))}</td><td>${esc(modelName(r.graph))}</td>` +
      `<td>${r.results.epochs_done ?? "-"}</td><td>${r.settings.lr ?? "-"}</td>` +
      `<td><b>${f(r, r.results.test_metric)}</b>${best && best.n === r.n && ok.length > 1 ? ' <span class="best" title="시험 성적이 가장 좋음">★ 최고</span>' : ""}</td>` +
      `<td>${f(r, r.results.train_metric)}</td><td class="acts-cell">` +
      (r.current ? `<span class="badge rec">지금 모델</span>` : !r.data_ok ? `<span class="muted">다른 데이터</span>`
        : r.weights ? `<button data-run="${r.n}">이 모델로</button>` : "") +
      ` <button class="x-small" data-run-del="${r.n}" title="이 기록 지우기">×</button></td></tr>`).join("") + `</table>`;
}
$("runs").onclick = async (e) => {
  const b = e.target.closest("[data-run], [data-run-del]");
  if (!b) return;
  try {
    if (b.dataset.runDel) {
      if (!confirm(`${b.dataset.runDel}번 학습 기록을 지울까요? 되살릴 수 없습니다.`)) return;
      st.runs = (await api("/api/runs/delete", { n: Number(b.dataset.runDel) })).runs;
      return renderRuns();
    }
    const r = await api("/api/runs/restore", { n: Number(b.dataset.run) });
    applyTrained(r.graph, r.results, r.train_settings);
    st.runs = r.runs;
    renderRuns();
    showMessage(`${b.dataset.run}번 학습의 모델로 돌아왔습니다`, "③ 모델·⑤ 예측·내려받기 모두 이 모델로 합니다", "", true);
  } catch (err) { showError(err); }
};
function applySettings(s) {                              // 학습 설정 칸에 (연 프로젝트·되돌린 기록)
  if (!s) return;
  if (s.epochs) $("epochs").value = s.epochs;
  if (s.lr) $("lr").value = s.lr;
  if (s.batch_size) $("batch").value = s.batch_size;
}
function applyTrained(graph, results, settings) {        // 학습한 모델을 지금 모델로 (기록 되돌리기)
  setGraph(graph);
  st.graphJson = sig(graph);
  st.trained = true;
  st.trainedJson = st.graphJson;
  st.trainedGraph = graph;
  applySettings(settings);
  drawCurve([], results && results.metric_name);
  if (results && results.metric_name) renderTrainResult(results);
  $("sample-box").hidden = false;
  showSample();
  markSteps();
}

// ------------------------------------------------------------ ⑤ 새 데이터로 예측
const PREDICT_FORMAT = {
  csv: (s) => `CSV: 한 행 = 샘플 하나, 값 열 ${s.image ? s.image[0] * s.image[1] + "개(픽셀)" : s.length + "개"} — 학습 CSV와 같은 순서. 정답 열이 더 있으면 맞았는지 표시합니다.`,
  folder: (s) => `그림 폴더: 그림 파일(png·jpg·bmp)을 넣은 폴더. 흑백으로 바꾸고 ${s.image[0]}×${s.image[1]}로 맞춥니다. 클래스 이름 하위 폴더에 나눠 넣으면 그것을 정답으로 봅니다.`,
  series: () => "시계열 폴더: 샘플마다 CSV 하나 (학습 때와 같은 값 열·길이 맞추기). 클래스 이름 하위 폴더에 나눠 넣으면 그것을 정답으로 봅니다.",
};
function renderPredictStep() {
  const s = st.summary, kinds = (s && s.predict_kinds) || [];
  const stale = st.trained && st.trainedJson !== st.graphJson;
  const guard = !s ? ["데이터와 모델이 아직 없습니다.", "data", "② 데이터 넣으러 가기"]
    : !st.trained ? ["학습한 모델이 없습니다. 먼저 학습하세요.", "train", "④ 학습하러 가기"] : null;
  $("predict-guard").hidden = !guard && !stale;
  $("predict-guard").innerHTML = guard ? `<span>⚠ ${esc(guard[0])}</span><button data-go="${guard[1]}">${esc(guard[2])}</button>`
    : stale ? `<span>예측은 마지막으로 학습한 모델로 합니다. ③에서 고친 내용은 다시 학습해야 반영됩니다.</span><button data-go="train">④ 다시 학습하러 가기</button>` : "";
  $("predict-inputs").hidden = !!guard;
  document.querySelectorAll("#predict-inputs [data-kind]").forEach((el) => (el.hidden = !kinds.includes(el.dataset.kind)));
  $("predict-format").textContent = s ? kinds.filter((k) => PREDICT_FORMAT[k]).map((k) => PREDICT_FORMAT[k](s)).join(" ") : "";
  if (s && kinds.includes("rows") && st.predictFormFor !== s) {
    st.predictFormFor = s;
    const means = s.feature_means || [];
    $("predict-form").innerHTML = s.features.map((f, i) => `<label>${esc(f)} <input type="number" step="any" data-f="${i}" value="${means[i] ?? 0}"></label>`).join("");
  }
}
$("predict-guard").onclick = (e) => { const b = e.target.closest("[data-go]"); if (b) go(b.dataset.go); };
async function runPredict(body) {
  $("predict-issues").innerHTML = "";
  $("predict-note").textContent = "예측하는 중…";
  try {
    const r = await api("/api/predict", body);
    st.predictCsv = r.csv;
    renderPredict(r);
  } catch (e) {
    $("predict-note").textContent = "";
    showError(e);
    $("predict-issues").innerHTML = (e.issues || []).map((i) => `<li>${esc(i.message)}<span class="hint">→ ${esc(i.hint)}</span></li>`).join("");
  }
}
function renderPredict(r) {
  const s = r.summary, cls = s.task === "classification", num = (v) => Number(Number(v).toPrecision(4));
  const cards = [["샘플", s.n]];
  if (cls) {
    for (const [c, n] of Object.entries(s.counts)) cards.push([`예측: ${c}`, n]);
    if (s.accuracy !== undefined) cards.push(["정확도 (정답과 비교)", pct(s.accuracy)]);
    if (s.low_conf) cards.push(["확신 60% 미만", s.low_conf]);
  } else {
    cards.push(["예측 평균", num(s.mean)], ["예측 범위", `${num(s.range[0])} ~ ${num(s.range[1])}`]);
    if (s.mae !== undefined) cards.push(["평균 오차 MAE", num(s.mae)]);
  }
  $("predict-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const head = cls ? ["샘플", "예측", "확신도"].concat(r.has_label ? ["정답", ""] : []) : ["샘플", "예측"].concat(r.has_label ? ["정답", "오차"] : []);
  $("predict-table").innerHTML = `<tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` + r.rows.map((x) => cls
    ? `<tr class="${x.ok === false ? "bad" : ""}"><td>${esc(x.name)}</td><td><b>${esc(x.pred)}</b></td><td>${pct(x.conf)}</td>` +
      (r.has_label ? `<td>${esc(x.label)}</td><td>${x.ok ? "✓" : "✗"}</td>` : "") + `</tr>`
    : `<tr><td>${esc(x.name)}</td><td><b>${num(x.pred)}</b></td>` + (r.has_label ? `<td>${num(x.label)}</td><td>${num(x.err)}</td>` : "") + `</tr>`).join("");
  $("predict-note").textContent = r.shown < s.n ? `표에는 처음 ${r.shown}개만 보입니다. 전부는 CSV로 내려받으세요.` : "";
  $("predict-result").hidden = false;
}
$("btn-predict-csv").onclick = () => (pathVal("predict-path") ? runPredict({ path: pathVal("predict-path") }) : showMessage("CSV 파일 경로를 넣으세요", "예: C:\\data\\new.csv"));
$("btn-predict-folder").onclick = () => (pathVal("predict-folder") ? runPredict({ folder: pathVal("predict-folder") }) : showMessage("그림 폴더 경로를 넣으세요"));
$("btn-predict-series").onclick = () => (pathVal("predict-series") ? runPredict({ series_folder: pathVal("predict-series") }) : showMessage("폴더 경로를 넣으세요"));
$("btn-predict-rows").onclick = () => runPredict({ rows: [[...document.querySelectorAll("#predict-form [data-f]")].map((el) => Number(el.value))] });
$("btn-predict-download").onclick = () => {
  if (!st.predictCsv) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([st.predictCsv], { type: "text/csv;charset=utf-8" }));
  a.download = "predictions.csv";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
};

async function showSample() {
  const i = Number($("sample-index").value) || 0;
  try {
    const a = await api("/api/activations", { sample_index: i });
    const num = (v) => (typeof v === "number" ? Number(v.toPrecision(4)) : v);
    $("sample-info").textContent = `정답 ${num(a.label)} · 예측 ${num(a.prediction)}`;
    const graph = st.trainedGraph || st.graph;              // 값은 학습한 모델 기준
    const im = st.summary.image;
    const inPic = im ? svgImage(a.input, im[0], im[1], Math.max(3, Math.round(96 / im[1])))
      : inputKind() === "tabular"                                // 표 데이터: 열 이름과 값 (크기가 제각각이라 그림보다 표)
      ? `<table class="tab"><tr>${st.summary.features.map((f) => `<th>${esc(f)}</th>`).join("")}</tr>` +
        `<tr>${a.input.map((v) => `<td>${num(v)}</td>`).join("")}</tr></table>` : svgLine(a.input, 600, 60);
    let html = `<div class="fs"><div class="name">입력</div>${inPic}</div>`;
    for (const [id, b, depth] of orderItems(a.order, graph)) {
      const v = a.values[id];
      if (!v || b.type === "input") continue;
      html += `<div class="fs" style="margin-left:${depth * 12}px"><div class="name">${esc(blockName(b))}</div>${valuePic(v, b)}</div>`;
    }
    $("sample-flow").innerHTML = html;
  } catch (e) { showError(e); }
}
$("sample-index").onchange = showSample;
$("btn-prev").onclick = () => { $("sample-index").value = Math.max(0, Number($("sample-index").value) - 1); showSample(); };
$("btn-next").onclick = () => { $("sample-index").value = Number($("sample-index").value) + 1; showSample(); };

// ------------------------------------------------------------ 저장·열기·내보내기
$("btn-save").onclick = async () => {
  const path = prompt("저장할 프로젝트 파일 경로 (.dlstudio)", "dlstudio-project.dlstudio");
  if (!path) return;
  try { await api("/api/project/save", { path }); showMessage(`저장했음: ${path}`, "", "", true); } catch (e) { showError(e); }
};
function applyOpened(r) {                                // 연 프로젝트를 화면에 (과제·데이터·모델·학습 결과)
  applySelection(r.selection);
  onData(r.summary);
  setGraph(r.graph);
  st.graphJson = sig(r.graph);
  st.trained = r.trained;
  st.trainedJson = r.trained ? st.graphJson : "";
  st.trainedGraph = r.trained ? r.graph : null;
  if (r.trained && r.results && r.results.metric_name) { renderTrainResult(r.results); $("sample-box").hidden = false; showSample(); }
  applySettings(r.train_settings);
  st.runs = r.runs || [];
  renderRuns();
  scheduleCheck();
  go(r.trained ? "train" : "model");
}
$("btn-open").onclick = async () => {
  const path = WEB ? await window.DLS_WEB.pickProject() : prompt("열 프로젝트 파일 경로 (.dlstudio)");
  if (!path) return;
  try {
    applyOpened(await api("/api/project/open", { path }));
    showMessage(`열었음: ${WEB ? path.split("/").pop() : path}`, "", "", true);
  } catch (e) { showError(e); }
};
$("btn-export").onclick = async () => {                // torch만 있으면 읽는 파일 (정규화·모양 맞추기 포함)
  if (WEB) {                                             // 웹: PyTorch 코드 + 학습한 가중치를 내려받기
    try {
      const r = await api("/api/project/export", {});
      showMessage(`내려받았음: ${r.files.join(", ")}`, "두 파일을 같은 폴더에 두고 python model_pytorch.py로 실행하면 학습한 가중치를 불러옵니다", "", true);
    } catch (e) { showError(e); }
    return;
  }
  const path = prompt("내보낼 모델 파일 경로 (.pt)", "model.pt");
  if (!path) return;
  try {
    await api("/api/project/export", { path });
    showMessage(`내보냈음: ${path}`, "파이썬에서 torch.jit.load로 읽고, 학습 CSV와 같은 열 순서의 원래 값(정답 열 빼고)을 넣으면 예측합니다",
      `import torch\nm = torch.jit.load("${path.replace(/\\/g, "/")}")\nscores = m(torch.tensor(rows, dtype=torch.float32))   # rows: (샘플 수, 특징 수)\n# 과제·클래스 이름: torch.jit.load(경로, _extra_files={"studio.json": ""})`, true);
  } catch (e) { showError(e); }
};
$("btn-export-pt").onclick = () => $("btn-export").onclick();

// ------------------------------------------------------------ 코드로 보기 (PyTorch · TensorFlow)
const CODE_NOTE = {
  pytorch: (trained) => trained
    ? "학습한 모델과 같은 코드입니다. [파일로 저장]하면 학습한 가중치(이름_weights.npz)도 함께 저장되고, 코드가 그 가중치를 불러옵니다."
    : "아직 학습하지 않았거나 학습한 뒤 고친 모델입니다. 코드 안의 train()으로 학습할 수 있습니다.",
  tensorflow: () => "같은 구조의 TensorFlow Keras 코드입니다. 학습한 가중치는 들어 있지 않아 train()으로 새로 학습합니다. 입력은 채널이 마지막인 모양입니다.",
};
let codeFw = "pytorch", codeFile = "";
async function showCode(fw) {
  if (!st.summary) return showMessage("데이터가 없음", "② 데이터 단계에서 먼저 불러오세요");
  ensureEditor();
  codeFw = fw || codeFw;
  document.querySelectorAll("#code-dialog .tab").forEach((b) => b.classList.toggle("active", b.dataset.fw === codeFw));
  $("code-dialog").hidden = false;
  $("code-note").textContent = "";
  $("code-text").textContent = "만드는 중…";
  try {
    const r = await api("/api/code", { graph: st.graph, framework: codeFw, settings: trainSettings() });
    codeFile = r.filename;
    $("code-text").textContent = r.code;
    $("code-note").textContent = CODE_NOTE[codeFw](r.trained);
  } catch (e) { $("code-dialog").hidden = true; showError(e); }
}
const closeCode = () => ($("code-dialog").hidden = true);
$("btn-code").onclick = () => showCode();
$("btn-code-train").onclick = () => showCode();
document.querySelectorAll("#code-dialog .tab").forEach((b) => (b.onclick = () => showCode(b.dataset.fw)));
$("code-close").onclick = closeCode;
$("code-dialog").onclick = (e) => { if (e.target === $("code-dialog")) closeCode(); };   // 바깥을 누르면 닫음
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("code-dialog").hidden) closeCode(); });
$("code-copy").onclick = async () => {
  try { await navigator.clipboard.writeText($("code-text").textContent); showMessage("코드를 복사했음", "", "", true); }
  catch (e) { showMessage("복사하지 못함", "코드를 직접 끌어 골라 Ctrl+C 하세요"); }
};
$("code-save").onclick = async () => {
  const path = prompt("저장할 파이썬 파일 경로 (.py)", codeFile || "model.py");
  if (!path) return;
  try {
    const r = await api("/api/code/save", { graph: st.graph, framework: codeFw, settings: trainSettings(), path });
    showMessage(`저장했음: ${r.files.join(", ")}`, r.files.length > 1 ? "가중치 파일(.pt)을 코드와 같은 폴더에 두세요" : "", "", true);
  } catch (e) { showError(e); }
};

// 화면을 새로 고쳐도 서버에 남은 상태(데이터, 블록, 학습된 모델)를 이어서 씀
(async () => {
  go("task");
  drawCurve([], "accuracy");
  if (!st.catalog.length) {                                  // 첫 로딩: 빈 화면 대신 무엇을 기다리는지
    $("btn-quick").disabled = true;
    $("families").innerHTML = `<p class="muted loading">과제 목록을 불러오는 중…${WEB ? " 처음 열 때는 계산 프로그램(Pyodide·TensorFlow.js)을 받느라 10~20초 걸립니다." : ""}</p>`;
  }
  try {
    const s = await api("/api/state");
    await loadCatalog();
    applySelection(s.selection);
    if (s.summary) { onData(s.summary); go("data"); }
    if (s.graph) { st.pendingGraph = s.graph; st.graphJson = sig(s.graph); }
    if (s.trained) {
      st.trained = true;
      st.trainedGraph = s.trained_graph || s.graph;
      st.trainedJson = sig(st.trainedGraph);               // 편집한 그래프와 다르면 '다시 학습' 표시
      if (s.results && s.results.metric_name) renderTrainResult(s.results);
      $("sample-box").hidden = false;
      showSample();
    }
    applySettings(s.train_settings);
    st.runs = s.runs || [];
    renderRuns();
    if (s.summary) showMessage("이전 작업을 이어서 불러왔음", "", "", true);
    markSteps();
  } catch (e) { /* 처음 실행 */ }
})();

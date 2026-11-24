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
function svgLine(vals, w = 240, h = 50, color = "#2f6fde") {
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
  y.forEach((v, i) => (s += `<circle cx="${px(v).toFixed(1)}" cy="${py(pred[i]).toFixed(1)}" r="2.6" fill="#2f6fde" fill-opacity=".55"/>`));
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
  st.step = step;
  for (const s of ["task", "data", "model", "train"]) $("step-" + s).hidden = s !== step;
  document.querySelectorAll(".step").forEach((b) => b.classList.toggle("active", b.dataset.step === step));
  if (step === "model") ensureEditor();
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
function renderTask() {
  $("families").innerHTML = st.catalog.map((f) => {
    const ready = f.subtasks.some((s) => s.available);
    return `<button class="tcard${st.fam === f ? " sel" : ""}${ready ? "" : " soon"}" data-f="${f.id}">` +
      `<div class="ti">${f.icon || ""}</div><div class="tn">${esc(f.name)}${ready ? "" : ` <span class="badge">준비 중</span>`}</div>` +
      `<div class="td">${esc(f.desc)}</div></button>`;
  }).join("");
  $("subtask-box").hidden = !st.fam;
  if (st.fam) $("subtasks").innerHTML = st.fam.subtasks.map((s) =>
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
  renderTask();
};
$("subtasks").onclick = (e) => {
  const b = e.target.closest("[data-s]");
  if (!b) return;
  const s = st.fam.subtasks.find((x) => x.id === b.dataset.s);
  if (!s.available) return showMessage(`'${s.name}'은(는) 준비 중입니다`, "같은 큰 틀의 다른 과제를 고르세요");
  if (st.sub !== s) chooseTemplate(st.fam, s, s.templates[0]);
};
$("templates").onclick = (e) => {
  const b = e.target.closest("[data-t]");
  if (b) chooseTemplate(st.fam, st.sub, st.sub.templates.find((x) => x.id === b.dataset.t));
};
$("btn-quick").onclick = async () => {                   // 예시 과제·데이터·추천 모델을 한 번에 → ④ 학습
  const fam = st.catalog.find((f) => f.subtasks.some((s) => s.available && (s.example_data || []).some((e) => e[0] === "ecg")));
  const sub = fam && fam.subtasks.find((s) => s.available && (s.example_data || []).some((e) => e[0] === "ecg"));
  if (!sub) return showMessage("예시를 찾지 못함", "아래에서 과제를 직접 고르세요");
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
  if (subChanged && st.summary && st.summary.task !== sub.task)
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
  $("data-guide").textContent = !sub ? "① 과제를 먼저 고르면 그 과제에 맞는 예시 데이터와 안내가 나옵니다. 또는 바로 CSV를 불러올 수도 있습니다."
    : img ? `이미지 폴더: ${GUIDE.image.classification} 픽셀 CSV(한 행 = 픽셀 값들 + 마지막 열 정답)도 됩니다. 처음이면 예시 데이터로 시작해 보세요.`
    : `CSV 파일: ${GUIDE[sub.input][sub.task]} 첫 줄이 열 이름이면 그대로 씁니다. 처음이면 예시 데이터로 시작해 보세요.`;
  $("image-row").hidden = !img;
  $("image-width-box").hidden = !img;
  const ts = sub && sub.input === "timeseries";              // 샘플마다 CSV 하나인 폴더: 시계열 과제에서만
  $("series-row").hidden = !ts;
  if (!ts) $("series-box").hidden = true;
  $("csv-path").placeholder = WEB ? "옆 단추로 CSV 파일을 고르세요 (브라우저 안에서만 읽음)" : img ? "C:\\data\\pixels.csv (픽셀 CSV)" : "C:\\data\\ecg.csv";
  const ex = sub ? sub.example_data : [["ecg", "심전도 정상·이상 (의료)"], ["vibration", "설비 진동 정상·결함 (제조)"]];
  $("example-btns").innerHTML = ex.map(([k, name]) => `<button class="primary" data-ex="${esc(k)}">예시: ${esc(name)}</button>`).join(" ");
  if (sub) $("task").value = sub.task;
  $("normalize-text").textContent = img ? "정규화 (픽셀 전체 평균 0, 표준편차 1)"
    : inputKind() === "tabular" ? "정규화 (열마다 평균 0, 표준편차 1)" : "정규화 (시점별 평균 0, 표준편차 1)";
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
  if (st.sub && st.sub.task !== s.task)
    showMessage(`불러온 데이터는 ${s.task === "classification" ? "분류" : "회귀"}용인데 고른 과제는 ${st.sub.name}입니다`, "① 과제를 데이터에 맞게 다시 고르거나, 맞는 데이터를 불러오세요");
  $("sample-index").max = s.n_test - 1;
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
        onHint: (msg) => showMessage(msg, "", "", true), getInput: inputKind,
        onHistory: (u, r) => { $("btn-undo").disabled = !u; $("btn-redo").disabled = !r; } });
    let es = "ortho";
    try { es = localStorage.getItem("dlstudio-edge-style") || "ortho"; } catch (e) { /* 없음 */ }
    $("edge-style").value = es;
    let fd = "v";
    try { fd = localStorage.getItem("dlstudio-flow-dir") || "v"; } catch (e) { /* 없음 */ }
    $("flow-dir").value = fd;
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
function renderConfusion(el, cm, recall, caption) {
  if (!cm) return ($(el).innerHTML = "");
  const names = st.summary.classes;
  let h = `<table class="cm"><tr><th>정답 \ 예측</th>${names.map((n) => `<th>${esc(n)}</th>`).join("")}<th>재현율</th></tr>`;
  cm.forEach((row, i) => {
    h += `<tr><th>${esc(names[i])}</th>${row.map((v, j) => `<td class="${i === j ? "diag" : ""}">${v}</td>`).join("")}` +
      `<td>${recall[i] === null ? "-" : pct(recall[i])}</td></tr>`;
  });
  $(el).innerHTML = h + `</table><div class="cm-cap">${esc(caption)} · 재현율 = 실제 그 클래스 중 맞힌 비율 (결함·이상을 놓치지 않는지)</div>`;
}
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
  const list = (st.sub ? st.sub.templates : st.defaults).map((t) => [t.id, t.name, t.graph]);
  st.tplList = list;
  $("tpl-btns").innerHTML = list.map(([id, name]) => `<button data-tpl="${esc(id)}">${esc(name)}</button>`).join(" ");
}
$("tpl-btns").onclick = (e) => {
  const b = e.target.closest("[data-tpl]");
  if (!b) return;
  const t = st.sub && st.sub.templates.find((x) => x.id === b.dataset.tpl);
  if (t) return chooseTemplate(st.fam, st.sub, t);
  setGraph(st.tplList.find((x) => x[0] === b.dataset.tpl)[2]);
};
$("btn-clear").onclick = () => { setGraph(EMPTY_GRAPH); showMessage("빈 모델로 새로 시작했습니다", "Ctrl+Z(되돌리기)로 이전 모델을 되살릴 수 있습니다", "", true); };
$("btn-autolayout").onclick = () => st.ed && st.ed.autoLayout();
$("auto-wrap").onchange = () => {                        // 자동 줄바꿈: 판 폭(세로면 높이)을 넘으면 다음 줄로
  try { localStorage.setItem("dlstudio-auto-wrap", $("auto-wrap").checked ? "1" : "0"); } catch (e) { /* 저장 안 돼도 됨 */ }
  if (st.ed) st.ed.setWrap($("auto-wrap").checked);
};
$("flow-dir").onchange = () => {                         // 방향은 이 브라우저에 기억
  try { localStorage.setItem("dlstudio-flow-dir", $("flow-dir").value); } catch (e) { /* 저장 안 돼도 됨 */ }
  if (st.ed) st.ed.setFlowDir($("flow-dir").value, true);
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
function drawCurve(progress, metricName) {
  const W = 560, H = 220, L = 40, B = 24;
  const n = progress.length;
  let s = `<line x1="${L}" y1="${H - B}" x2="${W - 8}" y2="${H - B}" stroke="#e3e6eb"/>`;
  if (n) {
    const xs = (i) => L + (n === 1 ? 0 : (i / (n - 1)) * (W - L - 10));
    const loss = progress.map((p) => p.loss), lmax = Math.max(...loss) || 1;
    const lp = loss.map((v, i) => `${xs(i).toFixed(1)},${(8 + (1 - v / lmax) * (H - B - 16)).toFixed(1)}`);   // 손실이 클수록 위
    const mp = progress.map((p, i) => `${xs(i).toFixed(1)},${(8 + (1 - Math.max(0, Math.min(1, p.test_metric))) * (H - B - 16)).toFixed(1)}`);
    s += `<polyline fill="none" stroke="#c0392b" stroke-width="2" points="${lp.join(" ")}"/>`;
    s += `<polyline fill="none" stroke="#2f6fde" stroke-width="2" points="${mp.join(" ")}"/>`;
    s += `<text x="${L}" y="${H - 6}" font-size="11" fill="#6b7280">epoch 1</text><text x="${W - 10}" y="${H - 6}" font-size="11" text-anchor="end" fill="#6b7280">${n}</text>`;
  }
  s += `<text x="${L}" y="14" font-size="12" fill="#c0392b">loss</text><text x="${L + 44}" y="14" font-size="12" fill="#2f6fde">test ${metricName === "r2" ? "R²" : "accuracy"}</text>`;
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
  renderConfusion("train-confusion", r.confusion, r.recall, "시험 데이터");
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
      const pad = depth * 12;
      let pic;
      if (Array.isArray(v[0]) && Array.isArray(v[0][0])) {       // (C, H, W): 채널마다 작은 그림
        const n = Math.min(16, v.length), h = v[0].length, w = v[0][0].length, cell = Math.max(2, Math.round(48 / w));
        pic = `<div class="chans">${v.slice(0, n).map((ch) => svgImage(ch, h, w, cell)).join("")}</div>` +
          (v.length > n ? `<div class="shape">… 채널 ${v.length - n}개 더</div>` : "");
      } else if (Array.isArray(v[0])) pic = svgStrips(v, 600);
      else if (b.type === "output" && st.summary.task === "classification") {
        const m = Math.max(...v), ex = v.map((x) => Math.exp(x - m)), sum = ex.reduce((x, y) => x + y, 0);
        pic = svgBars(ex.map((x) => x / sum), Math.max(120, v.length * 60), 60, st.summary.classes, "#2f6fde");
      } else pic = svgBars(v, Math.min(600, v.length * 14 + 10));
      html += `<div class="fs" style="margin-left:${pad}px"><div class="name">${esc(blockName(b))}</div>${pic}</div>`;
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
  scheduleCheck();
  go("model");
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
    if (s.summary) showMessage("이전 작업을 이어서 불러왔음", "", "", true);
    markSteps();
  } catch (e) { /* 처음 실행 */ }
})();

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

// 객체 탐지 그림: 흑백 그림 위에 정답(점선)·모델이 찾은 상자(실선, 점수) — 상자는 [클래스 번호, (점수,) x, y, w, h] 0–1 비율
const DET_COLORS = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"];
const TASK_NAME = { multi: "여러 정답", classification: "분류", regression: "회귀", detection: "객체 탐지", forecast: "시계열 예측", anomaly: "이상 탐지",
  segmentation: "구간 분할" };
const SEG_COLORS = ["#94a3b8", "#4f46e5", "#14b8a6", "#f97316", "#e11d48", "#a855f7", "#22c55e", "#eab308", "#0ea5e9", "#78716c"];
function segStrip(states, w = 240, h = 9) {              // 시점마다 상태 → 색 띠 (같은 상태가 이어지면 한 칸)
  const L = states.length;
  let s = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="seg-strip">`;
  for (let i = 0, j = 0; i < L; i = j) {
    while (j < L && states[j] === states[i]) j++;
    s += `<rect x="${((i * w) / L).toFixed(2)}" y="0" width="${(((j - i) * w) / L).toFixed(2)}" height="${h}" fill="${SEG_COLORS[states[i] % SEG_COLORS.length]}"/>`;
  }
  return s + "</svg>";
}
const segLegend = (names) => `<p class="muted fc-legend">${names.map((n, k) => `<span style="color:${SEG_COLORS[k % SEG_COLORS.length]}">■</span> ${esc(n)}`).join(" · ")}</p>`;
// 성적 글자: 정확도·mAP는 %, R²·AUC는 0–1, 복원 오차는 그대로 (낮을수록 좋음)
function metricText(name, v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  if (name === "mean") return "평균 " + v.toFixed(3);
  if (name === "r2") return "R² " + v.toFixed(3);
  if (name === "auc") return "AUC " + v.toFixed(3);
  if (name === "recon") return "복원 오차 " + Number(v.toPrecision(3));
  return (name === "map50" ? "mAP@0.5 " : "정확도 ") + pct(v);
}
// 여러 줄 그래프: series [{v: 값들, x0: 시작 위치, color, dash, width}] — 모두 같은 눈금, cut: 세로 점선 위치
function svgSeries(series, w = 300, h = 90, cut = null, bands = null, bandMax = null) {
  const all = series.flatMap((s) => s.v), n = Math.max(...series.map((s) => (s.x0 || 0) + s.v.length));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  const X = (i) => 3 + (i / Math.max(1, n - 1)) * (w - 6), Y = (v) => 3 + (1 - (v - lo) / (hi - lo)) * (h - 6);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="series">`;
  if (bands) { const m = bandMax || Math.max(...bands) || 1; bands.forEach((b, i) => (s += `<rect x="${X(i) - (w - 6) / n / 2}" y="0" width="${(w - 6) / n + 0.3}" height="${h}" fill="rgba(239,68,68,${(0.6 * b / m).toFixed(3)})"/>`)); }
  if (cut !== null) s += `<line x1="${X(cut - 0.5)}" y1="0" x2="${X(cut - 0.5)}" y2="${h}" stroke="#cbd5e1" stroke-dasharray="3 3"/>`;
  for (const sr of series) {
    const pts = sr.v.map((v, i) => `${X((sr.x0 || 0) + i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    s += `<polyline fill="none" stroke="${sr.color}" stroke-width="${sr.width || 1.4}"${sr.dash ? ` stroke-dasharray="${sr.dash}"` : ""} points="${pts}"/>`;
  }
  return s + `</svg>`;
}
const FC_COLORS = { hist: "#64748b", y: "#0f172a", pred: "#4f46e5", naive: "#f97316" };
function grayURL(vals, H, W) {                          // 펼친 값 → PNG 주소 (칸마다 rect보다 훨씬 가벼움)
  const flat = Array.isArray(vals[0]) ? vals.flat() : vals;
  let lo = Infinity, hi = -Infinity;
  for (const v of flat) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-12) hi = lo + 1;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d"), im = g.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const v = Math.round((255 * (flat[i] - lo)) / (hi - lo));
    im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v;
    im.data[i * 4 + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return c.toDataURL();
}
function detImage(vals, H, W, px, gts, preds, classes, minScore = 0.5) {
  const fs = Math.max(4.5, W / 12), col = (k) => DET_COLORS[k % DET_COLORS.length];
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${px}" height="${Math.round((px * H) / W)}" class="img det-img" role="img">` +
    `<image href="${grayURL(vals, H, W)}" width="${W}" height="${H}" preserveAspectRatio="none" style="image-rendering:pixelated"/>`;
  for (const [k, x, y, w, h] of gts || [])
    s += `<rect x="${x * W}" y="${y * H}" width="${w * W}" height="${h * H}" fill="none" stroke="${col(k)}" stroke-width="1.6" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"><title>정답: ${esc(classes[k])}</title></rect>`;
  for (const [k, sc, x, y, w, h] of (preds || []).filter((p) => p[1] >= minScore)) {
    const label = `${classes[k]} ${Math.round(sc * 100)}%`;
    const tw = [...label].reduce((a, ch) => a + (ch.charCodeAt(0) > 255 ? 1 : 0.6), 0) * fs;   // 글자 폭 어림 (한글 1, 나머지 0.6)
    const ty = y * H > fs + 1 ? y * H - 1.5 : Math.min(H - 1, (y + h) * H + fs);
    const right = x * W + tw > W;                                  // 오른쪽 끝을 넘으면 상자 오른쪽 끝에 맞춤
    s += `<rect x="${x * W}" y="${y * H}" width="${w * W}" height="${h * H}" fill="${col(k)}" fill-opacity=".12" stroke="${col(k)}" stroke-width="2.2" vector-effect="non-scaling-stroke"><title>찾음: ${esc(label)}</title></rect>` +
      `<text x="${right ? Math.min(W - 0.5, (x + w) * W) : x * W + 0.5}" y="${ty}" font-size="${fs}" fill="${col(k)}" class="det-text"${right ? ' text-anchor="end"' : ""}>${esc(label)}</text>`;
  }
  return s + `</svg>`;
}
function detLegend(classes, withPred = true) {
  return `<div class="det-legend">${classes.map((c, k) => `<span><i style="--c:${DET_COLORS[k % DET_COLORS.length]}"></i>${esc(c)}</span>`).join("")}` +
    `<span class="muted">점선 = 정답${withPred ? " · 실선 = 모델이 찾은 상자 (점수)" : ""}</span></div>`;
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
// 데이터가 없을 때 ③·④·⑤는 내용 대신 안내 화면. 사이드바에는 아직 못 쓰는 단계를 흐리게
function renderEmpty() {
  const none = !st.summary;
  for (const s of ["model", "train", "predict"]) {
    const sec = $("step-" + s);
    if (sec) sec.classList.toggle("is-empty", none);
  }
  const lock = { model: none, train: none, predict: none || !st.trained };
  document.querySelectorAll(".step[data-step]").forEach((b) => {
    const why = lock[b.dataset.step];
    b.classList.toggle("locked", !!why);
    if (b.dataset.step in lock) b.title = !why ? "" : none ? "먼저 ② 데이터를 넣으세요" : "먼저 ④에서 학습하세요";
  });
}
document.addEventListener("click", (e) => {               // 안내 화면의 단추
  const b = e.target.closest(".empty-step [data-go], .empty-step [data-quick]");
  if (!b) return;
  if (b.dataset.go) return go(b.dataset.go);
  go("task");
  $("btn-quick").click();
});
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
const CH_COLORS = ["#4f46e5", "#64748b", "#d97706", "#db2777", "#2563eb", "#65a30d"];
function svgChannels(vals, C, w = 240, h = 50) {           // 여러 채널 (채널이 먼저 이어 붙은 값): 채널마다 한 줄씩 위아래로, 저마다 눈금
  if (!C || C < 2) return svgLine(vals, w, h);
  const L = vals.length / C, hh = Math.max(18, Math.round(h / C * 1.2));
  return `<div class="ch-stack">${Array.from({ length: C }, (_, c) => svgLine(vals.slice(c * L, (c + 1) * L), w, hh, CH_COLORS[c % CH_COLORS.length])).join("")}</div>`;
}
const chLegend = (names) => (names && names.length > 1
  ? `<p class="muted fc-legend">${names.map((n, c) => `<span style="color:${CH_COLORS[c % CH_COLORS.length]}">━</span> ${esc(n)}`).join(" · ")}</p>` : "");
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
// 단계 이동은 브라우저 기록에도 남김: 뒤로·앞으로 가기(마우스 옆 단추·휴대폰 뒤로 밀기)가 사이트를 떠나지 않고 단계를 오감.
// 주소는 그대로 (새로 고치면 처음 화면부터). fromHistory: 뒤로·앞으로 가기로 온 것이라 기록을 더하지 않음
function go(step, fromHistory = false) {
  if (!$("step-" + step)) step = "task";
  st.step = step;
  if (step === "train") renderLoss();                       // 손실 칩의 블록 목록은 지금 모델에서
  setTimeout(() => {                                         // 좁은 화면: 단계 줄이 가로로 밀리면 지금 단계가 보이게
    const s = document.querySelector("#steps .step.active"), bar = $("steps");
    if (s && bar && bar.scrollWidth > bar.clientWidth) bar.scrollLeft = s.offsetLeft - bar.offsetLeft - 8;
  }, 0);
  if (!fromHistory) {
    try {
      const h = history.state && history.state.dls;
      if (!h) history.replaceState({ dls: step }, "");       // 처음 화면: 지금 칸을 씀 (뒤로 한 번 더면 사이트를 떠남)
      else if (h !== step) history.pushState({ dls: step }, "");
    } catch (e) { /* 기록을 못 쓰는 환경 */ }
  }
  document.querySelectorAll("main > .panel").forEach((el) => (el.hidden = el.id !== "step-" + step));
  document.querySelectorAll(".step").forEach((b) => b.classList.toggle("active", b.dataset.step === step));
  renderEmpty();
  if (step === "model" && st.summary) { ensureEditor(); if (st.fitPending) fitSoon(); }   // 데이터가 없으면 편집기 대신 안내 (모양을 정할 수 없음)
  if (step === "predict") renderPredictStep();
  window.scrollTo(0, 0);
  const mainEl = document.querySelector("main");          // 사이드바 틀에서는 main이 스크롤됨
  if (mainEl) mainEl.scrollTop = 0;
  if (window.DLProjects && window.DLProjects.showSteps) window.DLProjects.showSteps();   // 웹: 만들기 흐름 보이기·숨기기
}
window.addEventListener("popstate", (e) => {
  const step = e.state && e.state.dls;
  if (!step) return;
  if (step === "home" && window.DLProjects) return window.DLProjects.home(true);   // 프로젝트 목록을 새로 그림
  go(step, true);
});
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
  renderEmpty();
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
// 큰 틀 아이콘: 내 프로젝트 화면과 같은 선 그림 (24×24, 선 1.8)
const lineIcon = (d) => `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const FAMILY_ICON = {
  classification: lineIcon('<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9z"/><circle cx="7.5" cy="7.5" r="1.5"/>'),
  regression: lineIcon('<path d="M3 3v18h18"/><path d="M7 16 19 6"/><circle cx="8" cy="11" r="1"/><circle cx="12" cy="13" r="1"/><circle cx="15" cy="8" r="1"/>'),
  detection: lineIcon('<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><rect x="8.5" y="8.5" width="7" height="7" rx="1"/>'),
  anomaly: lineIcon('<path d="M3 12h4l2-5 4 10 2-5h6"/>'),
  forecast: lineIcon('<path d="M3 17l5-5 4 3 6-7"/><path d="M14 8h4v4"/><path d="M3 21h18" stroke-dasharray="2 3"/>'),
  segmentation: lineIcon('<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M9 6v12M15 6v12"/>'),

  multi: lineIcon('<circle cx="5" cy="12" r="2"/><path d="M7 12h4M11 12l5-6M11 12l5 6M11 12h5"/><circle cx="18" cy="5" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="18" cy="19" r="2"/>'),
};
const readyFirst = (xs, ok) => xs.filter(ok).concat(xs.filter((x) => !ok(x)));   // '준비 중'은 뒤로
function renderTask() {
  $("btn-quick").disabled = !st.catalog.length;
  $("families").innerHTML = readyFirst(st.catalog, (f) => f.subtasks.some((s) => s.available)).map((f) => {
    const ready = f.subtasks.some((s) => s.available);
    return `<button class="tcard${st.fam === f ? " sel" : ""}${ready ? "" : " soon"}" data-f="${f.id}">` +
      `<span class="fi ti">${FAMILY_ICON[f.id] || lineIcon('<circle cx="12" cy="12" r="8"/>')}</span><div class="tn">${esc(f.name)}${ready ? "" : ` <span class="badge">준비 중</span>`}</div>` +
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
  else { st.pendingGraph = tpl.graph; st.fitPending = true; }
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
GUIDE.timeseries.forecast = "한 행 = 한 시점 (시간 순서대로), 열 = 측정 항목 (예: 시각, 온도, 전력). 맞힐 값 열의 지난 값들로 다음 값들을 예측합니다. 불러온 뒤 맞힐 열·입력 길이·예측 길이를 바꿀 수 있습니다.";
GUIDE.timeseries.anomaly = "한 행 = 신호 한 구간 (시점마다 값이 한 열씩), 마지막 열 = 상태 (예: 정상, 결함). 정상만 학습하고 나머지는 이상으로 봅니다. 정상만 있어도 됩니다.";
GUIDE.tabular.multi = "한 행 = 샘플 하나, 열 = 측정 항목과 정답들. 불러온 뒤 역할 표에서 정답 열 이름을 눌러 '정답'으로 바꾸면 (둘 이상) 정답마다 분류·숫자를 스스로 정합니다.";
GUIDE.timeseries.segmentation = "한 행 = 한 시점 (시간 순서대로), 열 = 측정값들과 그 시점의 상태 (마지막 열). 같은 길이의 창으로 잘라 시점마다 상태를 맞힙니다. 불러온 뒤 역할 표에서 상태 열·창 길이를 바꿀 수 있습니다.";
GUIDE.tabular.anomaly = "한 행 = 샘플 하나, 열 = 측정 항목, 마지막 열 = 상태 (예: 정상, 불량). 정상만 학습하고 나머지는 이상으로 봅니다. 정상만 있어도 됩니다.";
function renderDataGuide() {
  const sub = st.sub;
  $("data-task-name").textContent = sub ? `${st.fam.name} · ${sub.name}` : "";
  const img = sub && sub.input === "image";
  $("data-guide").textContent = !sub || st.autoTask
    ? "과제를 고르지 않아도 됩니다. CSV·그림 폴더·예시 데이터를 넣으면 분류인지 회귀인지, 시계열·표·이미지 중 무엇인지 보고 과제와 추천 모델을 정해 드립니다. CSV는 한 행 = 샘플 하나, 마지막 열 = 정답."
    : img ? `이미지 폴더: ${GUIDE.image.classification} 픽셀 CSV(한 행 = 픽셀 값들 + 마지막 열 정답)도 됩니다. 처음이면 예시 데이터로 시작해 보세요.`
    : sub.task === "segmentation" ? `CSV 파일: ${GUIDE.timeseries.segmentation} 처음이면 예시 데이터로 시작해 보세요.`
    : `CSV 파일: ${GUIDE[sub.input][sub.task]} 첫 줄이 열 이름이면 그대로 씁니다. 처음이면 예시 데이터로 시작해 보세요.`;
  $("mine-title").hidden = $("mine-grid").hidden = false;
  const any = !sub || st.autoTask;                           // 과제 없이: 넣는 방법을 다 보임
  $("image-row").hidden = !img && !any;
  $("image-width-box").hidden = !img;
  const ts = ((sub && sub.input === "timeseries") || any) && !(sub && (sub.task === "forecast" || sub.task === "segmentation"));   // 샘플마다 CSV 하나인 폴더: 시계열 과제에서만 (예측은 긴 CSV 하나)
  $("series-row").hidden = !ts;
  if (!ts) $("series-box").hidden = true;
  $("csv-path").placeholder = WEB ? "단추로 CSV 파일 고르기" : img ? "C:\\data\\pixels.csv (픽셀 CSV)" : "C:\\data\\ecg.csv";
  st.examples = sub && !st.autoTask ? sub.example_data : allExamples();
  renderExamples();
  renderLabelJobs();
  $("csv-format-body").innerHTML = csvFormat(any ? null : sub);
  if (sub && !st.autoTask) $("task").value = sub.task;
  $("normalize-text").textContent = img ? "정규화 (픽셀 전체 평균 0, 표준편차 1)"
    : inputKind() === "tabular" ? "정규화 (열마다 평균 0, 표준편차 1)" : "정규화 (시점별 평균 0, 표준편차 1)";
}
async function renderLabelJobs() {                      // 웹: 라벨링 페이지에서 만든 작업 (이 브라우저 보관함)
  if (!WEB || !window.DLStore || !DLStore.labels) return;
  $("label-card").hidden = false;
  let items = [];
  try { items = await DLStore.labels.list(); } catch (e) { /* 보관함을 못 열면 빈 목록 */ }
  const kind = { series_folder: "시계열", image_folder: "이미지", table: "표" };
  const schema = { class: "클래스", number: "숫자", segments: "구간", boxes: "도형" };
  $("label-jobs").innerHTML = items.length ? items.map((p) => {
    const m = p.meta || {}, ok = m.schema === "class" || m.schema === "number" || (m.schema === "boxes" && m.kind === "image_folder");
    return `<div class="label-job"><div><b>${esc(p.name)}</b><span class="muted">${esc(`${kind[m.kind] || ""} · ${schema[m.schema] || ""} 정답 · ${m.labeled || 0}/${m.total || 0}개`)}</span></div>` +
      (ok ? `<button data-job="${esc(p.id)}">불러오기</button>` : `<span class="muted" title="구간 정답으로 학습하는 과제는 준비 중">학습 과제 준비 중</span>`) + `</div>`;
  }).join("") : `<p class="muted">아직 없습니다. 라벨링 페이지에서 정답을 붙이면 여기서 불러올 수 있습니다.</p>`;
}
$("label-jobs").onclick = (e) => {
  const b = e.target.closest("[data-job]");
  if (b) loadData(Object.assign(dataOptions(), { label_job: b.dataset.job, confirmed_only: $("label-confirmed").checked }));
};
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
function renderExamples() {                             // 예시 카드: 이름 + (분야 · 과제), 불러온 것은 ✓
  $("example-btns").innerHTML = (st.examples || []).map(([k, label]) => {
    const [main, sub] = label.split(" · ");
    const m = /^(.*?)\s*\(([^)]*)\)\s*$/.exec(main), name = m ? m[1] : main;
    const meta = [m ? m[2] : "", sub || ""].filter(Boolean).join(" · ");
    const sel = st.summary && st.dataSrc === k;
    return `<button class="ex-card${sel ? " sel" : ""}" data-ex="${esc(k)}"><span class="ex-name">${esc(name)}</span>` +
      `<span class="ex-meta">${esc(meta)}</span>${sel ? `<span class="ex-check">✓ 불러옴</span>` : ""}</button>`;
  }).join("");
}
function csvFormat(sub) {                                // CSV 모양 예시 (작은 표)
  if (sub && sub.task === "forecast") {
    const rows = [["시각", "온도", "전력"], ["2026-03-01 00:00", "21.4", "310.2"], ["2026-03-01 01:00", "21.1", "298.7"], ["…", "…", "…"]];
    return `<p>한 행 = 한 시점 (시간 순서대로), 열 = 측정 항목. <b>맞힐 값 열</b>은 불러온 뒤 고릅니다 (처음엔 마지막 숫자 열).</p>` +
      `<table class="tab"><tr>${rows[0].map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.slice(1).map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table>` +
      `<p class="muted">시각 같은 글자 열은 있어도 됩니다 (쓰지 않음). 빈 칸은 앞뒤 값으로 메웁니다. 시험은 뒤쪽 ${Math.round((Number($("test-ratio").value) || 0.2) * 100)}% 구간.</p>`;
  }
  if (sub && sub.task === "segmentation") {
    const rows = [["시각", "전류", "압력", "상태"], ["0.00", "0.46", "6.69", "대기"], ["0.01", "0.56", "6.68", "대기"], ["0.02", "1.83", "6.52", "동작"], ["…", "…", "…", "…"]];
    return `<p>한 행 = 한 시점 (시간 순서대로), <b>마지막 열 = 그 시점의 상태</b>. 값 열이 여럿이면 채널이 됩니다.</p>` +
      `<table class="tab"><tr>${rows[0].map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.slice(1).map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table>` +
      `<p class="muted">시각 같은 글자 열은 있어도 됩니다 (쓰지 않음). 빈 칸은 앞뒤 값으로 메웁니다.</p>`;
  }
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
  return `<p>${esc(what)}, 정답 열(${reg ? "맞힐 숫자" : "정답 이름"}) 하나 이상. 어느 열이 정답인지는 파일을 고른 뒤 정합니다 (처음엔 마지막 열).</p>${table}` +
    `<p class="muted">첫 줄(열 이름)은 있어도 없어도 됩니다. 엑셀에서는 [다른 이름으로 저장 → CSV UTF-8].</p>`;
}
$("example-btns").onclick = (e) => {
  const b = e.target.closest("[data-ex]");
  if (b) loadData(Object.assign(dataOptions(), { example: b.dataset.ex }));
};
function dataOptions() {
  const o = { normalize: $("normalize").checked, test_ratio: Number($("test-ratio").value) || 0.2 };
  if ($("label-col").value) o.label_col = Number($("label-col").value) - 1;
  if ($("task").value && $("task").value !== "auto") o.task = $("task").value;
  return o;
}
async function loadData(body) {                       // 성공하면 true
  $("data-issues").innerHTML = "";
  try {
    const r = await api("/api/data/load", body);
    st.dataName = r.label_name || null;                     // 라벨링한 작업이면 그 이름 (프로젝트 이름에)
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
// ④ 학습 화면 비우기: 새 데이터(다른 프로젝트 열기 포함)에는 지난 학습 결과·곡선·샘플을 남기지 않음
function clearTrainView() {
  $("train-result").innerHTML = `<p class="muted">학습이 끝나면 나옵니다</p>`;
  for (const id of ["train-warnings", "train-confusion", "train-scatter", "sample-flow", "sample-info", "train-status"]) $(id).innerHTML = "";
  $("cm-pick").hidden = true;
  $("sample-box").hidden = true;
  $("det-box").hidden = true;
  $("det-box").innerHTML = "";
  st.detPred = null;
  drawCurve([], "accuracy");
}
const lossKey = (s) => (s && s.losses ? JSON.stringify([s.losses.allowed, (s.losses.heads || []).map((h) => h.name)]) : "");
function onData(s) {
  st.summary = s;
  if (lossKey(s) !== st.lossKey) st.loss = null;              // 정답 종류가 바뀌면 손실은 기본으로 (자동 과제는 요약을 먼저 넣으므로 따로 기억)
  st.lossKey = lossKey(s);
  renderLoss();
  st.dataSrc = s.example || null;
  st.trained = false;
  clearTrainView();                                          // 지난 데이터·프로젝트의 학습 결과는 치움
  $("data-summary").hidden = false;
  $("btn-next-data").hidden = false;
  const det = s.task === "detection", fc = s.forecast, ae = s.anomaly;
  const seg = s.task === "segmentation", multi = s.task === "multi";
  const task = multi ? `여러 정답 (${s.targets.length}개)` : s.task === "classification" ? `분류 (${s.classes.length}클래스)` : det ? `객체 탐지 (${s.classes.length}클래스)`
    : fc ? "시계열 예측" : ae ? "이상 탐지" : seg ? `구간 분할 (${s.classes.length}상태)` : "회귀";
  const tab = inputKind() === "tabular" && !s.image;
  const cards = fc ? [["학습 창", s.n_train], ["시험 창 (뒤쪽 구간)", s.n_test], ["입력 길이", `지난 ${s.length}개`], ["예측 길이", `다음 ${fc.horizon}개`],
    ["맞힐 값 열", esc(fc.column)], ["점 수", fc.n_points]]
    : [[det ? "학습 그림" : ae ? "학습 샘플 (정상만)" : "학습 샘플", s.n_train], [det ? "시험 그림" : "시험 샘플", s.n_test],
      s.image ? ["크기 (픽셀)", `${s.image[0]}×${s.image[1]}`] : [tab ? "측정 항목 (열)" : "길이 (시점)", s.length], ["과제", task]];
  if (ae) cards.push(["정상으로 본 클래스", esc(ae.normal)], ["이상으로 본 클래스", ae.others.length ? ae.others.map(esc).join(", ") : "없음 (정상만)"]);
  renderFcBox(s);
  renderRoleBox(s);
  if (det) cards.push(["상자", `${s.detection.n_boxes}개`], ["격자", `${s.detection.grid}×${s.detection.grid}칸`]);
  if (s.channels > 1) cards.push(["채널", (s.channel_names || []).length ? s.channel_names.map(esc).join(" · ") : `${s.channels}개`]);
  if (multi) for (const t of s.targets) cards.push([`정답 '${esc(t.name)}'`, t.kind === "class" ? `${t.classes.length}클래스`
    : `${Number(t.range[0].toPrecision(3))} ~ ${Number(t.range[1].toPrecision(3))}`]);
  if (s.target_range) cards.push(["정답 범위", `${Number(s.target_range[0].toPrecision(3))} ~ ${Number(s.target_range[1].toPrecision(3))}`]);
  $("data-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("") +
    Object.entries(s.class_counts).map(([k, v]) => `<div class="card"><div class="k">${det ? `'${esc(k)}' 상자` : seg ? `'${esc(k)}' 시점` : `클래스 '${esc(k)}'`}</div><div class="v">${v}</div></div>`).join("");
  const fmt = (v) => Number(Number(v).toPrecision(4));
  $("data-preview").classList.toggle("table", tab);
  $("data-preview").classList.toggle("thumbs", !!s.image);
  $("data-preview").classList.toggle("det", det);
  $("data-preview").innerHTML = det
    ? s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${detImage(p.values, s.image[0], s.image[1], 150, p.boxes, null, s.classes)}</div>`).join("") +
      detLegend(s.classes, false)
    : s.image
    ? s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${svgImage(p.values, s.image[0], s.image[1], Math.max(3, Math.round(84 / s.image[1])))}</div>`).join("")
    : tab
    ? `<table class="tab"><tr>${s.features.map((f) => `<th>${esc(f)}</th>`).join("")}<th class="lab">정답</th></tr>` +
      s.preview.map((p) => `<tr>${p.values.map((v) => `<td>${fmt(v)}</td>`).join("")}<td class="lab">${esc(p.label)}</td></tr>`).join("") + `</table>`
    : fc
    ? s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${svgSeries([{ v: p.values, color: FC_COLORS.hist },
      { v: p.future, x0: p.values.length, color: FC_COLORS.y, width: 1.8 }], 240, 60, p.values.length)}</div>`).join("") +
      `<p class="muted fc-legend"><span style="color:${FC_COLORS.hist}">━</span> 모델이 보는 지난 값 · <span style="color:${FC_COLORS.y}">━</span> 맞힐 다음 값</p>`
    : seg
    ? s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${svgChannels(p.values, s.channels, 240, 60)}${segStrip(p.states, 240)}</div>`).join("") +
      segLegend(s.classes) + chLegend(s.channel_names)
    : s.preview.map((p) => `<div class="pv"><div class="lab">${esc(p.label)}</div>${svgChannels(p.values, s.channels, 240, 60)}</div>`).join("") +
      chLegend(s.channel_names);
  renderAutoTask();
  if (!st.autoTask && st.sub && st.sub.task !== s.task && !(st.sub.task === "multi" && s.columns))   // 여러 정답: 역할 표에서 정답을 더 고름
    showMessage(`불러온 데이터는 ${TASK_NAME[s.task] || s.task}용인데 고른 과제는 ${st.sub.name}입니다`, "① 과제를 데이터에 맞게 다시 고르거나, 맞는 데이터를 불러오세요");
  $("sample-index").max = s.n_test - 1;
  renderExamples();
  markSteps();
  if (st.ed) { st.ed.redrawPalette(); st.ed.redraw(); scheduleCheck(); }   // 입력 형태가 바뀌면 왼쪽 목록도
}
const ROLE_NEXT = { input: "target", target: "skip", skip: "input" }, ROLE_NAME = { input: "입력", target: "정답", skip: "안 씀" };
function renderRoleBox(s) {                               // ② 역할 표: CSV의 열마다 입력·정답·안 씀 (누르면 바뀜)
  const cols = s.columns;
  $("role-box").hidden = !cols;
  if (!cols) return;
  st.roles = cols.map((c) => c.role);
  $("role-channels").value = s.csv_channels || 1;
  const seg = s.task === "segmentation";                    // 구간 분할: 값 열이 곧 채널, 대신 창 길이
  $("role-ch-box").hidden = seg || inputKind() === "tabular";  // 표 데이터에는 채널이 없음
  $("role-len-box").hidden = !seg;
  if (seg) $("role-length").value = s.length;
  drawRoles();
}
function roleGroups(cols, roles) {                        // 같은 역할의 숫자 열이 8개 넘게 이어지면 칩 하나로 (시계열 CSV의 t1 … t140)
  const out = [], num = (c) => (c.kind || "number") === "number";
  for (let i = 0; i < cols.length;) {
    let j = i;
    while (j + 1 < cols.length && roles[j + 1] === roles[i] && num(cols[j + 1]) && num(cols[i])) j++;
    if (j - i + 1 >= 8) { out.push([i, j]); i = j + 1; } else { out.push([i, i]); i++; }
  }
  return out;
}
function drawRoleChips(el, noteEl, cols, roles) {         // 열 칩 (누르면 입력 → 정답 → 안 씀), 묶음 칩은 그 열들을 한꺼번에
  el.innerHTML = roleGroups(cols, roles).map(([a, b]) => {
    const name = b > a ? `${cols[a].name} … ${cols[b].name} · ${b - a + 1}열` : cols[a].name;
    const tip = b > a ? name : cols[a].sample && cols[a].sample.length ? `${cols[a].name}: ${cols[a].sample.join(", ")} …` : cols[a].name;
    return `<button type="button" class="role r-${roles[a]}" data-a="${a}" data-b="${b}" title="${esc(tip)}"><span>${esc(name)}</span><b>${ROLE_NAME[roles[a]]}</b></button>`;
  }).join("");
  const n = (r) => roles.filter((x) => x === r).length;
  noteEl.textContent = `입력 ${n("input")} · 정답 ${n("target")}` + (n("skip") ? ` · 안 씀 ${n("skip")}` : "");
}
function cycleRoles(e, roles, redraw) {
  const b = e.target.closest("[data-a]");
  if (!b) return;
  const a = Number(b.dataset.a), z = Number(b.dataset.b), next = ROLE_NEXT[roles[a]];
  for (let j = a; j <= z; j++) roles[j] = next;
  redraw();
}
const rolePick = (roles, r) => roles.map((x, j) => (x === r ? j : -1)).filter((j) => j >= 0);
function drawRoles() { drawRoleChips($("role-cols"), $("role-note"), st.summary.columns, st.roles); }
$("role-cols").onclick = (e) => cycleRoles(e, st.roles, drawRoles);
// CSV 열 고르기 (불러오기 전): 열마다 짐작한 역할을 채워 보여 주고, 사용자가 확인·바꾼 뒤 [불러오기]
function drawPick() {
  drawRoleChips($("pick-cols"), $("pick-note"), st.pick.cols, st.pick.roles);
  const seg = !$("pick-len-box").hidden, table = !!st.sub && !st.autoTask && st.sub.input === "tabular";
  $("pick-ch-box").hidden = seg || table || rolePick(st.pick.roles, "input").length < 16;   // 채널은 값 열이 많은 시계열에서만
}
$("pick-cols").onclick = (e) => cycleRoles(e, st.pick.roles, drawPick);
async function openPick(path) {
  $("data-issues").innerHTML = "";
  try {
    const r = await api("/api/data/columns", { path });
    st.pick = { path, cols: r.columns, roles: r.columns.map((c) => c.role) };
    const seg = !!st.sub && !st.autoTask && st.sub.task === "segmentation";
    $("pick-file").textContent = $("csv-path").value || path.split(/[\\/]/).pop();
    $("pick-info").textContent = `${r.n_rows}행 · 열 ${r.columns.length}개`;
    $("pick-ch-box").hidden = seg || (!!st.sub && !st.autoTask && st.sub.input === "tabular");
    $("pick-len-box").hidden = !seg;
    $("pick-channels").value = 1;
    $("pick-length").value = "";
    $("csv-pick").hidden = false;
    drawPick();
    $("csv-pick").scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (e) {
    showError(e);
    $("data-issues").innerHTML = (e.issues || []).map((i) => `<li>${esc(i.message)}<span class="hint">→ ${esc(i.hint)}</span></li>`).join("");
  }
}
$("btn-pick-load").onclick = async () => {
  const roles = st.pick.roles, seg = !$("pick-len-box").hidden;
  if (!rolePick(roles, "target").length) return showMessage("정답 열을 하나 이상 고르세요", "열 이름을 눌러 '정답'으로 바꾸세요");
  if (!rolePick(roles, "input").length) return showMessage("입력 열을 하나 이상 고르세요", "열 이름을 눌러 '입력'으로 바꾸세요");
  if (seg && rolePick(roles, "target").length !== 1) return showMessage("상태 열은 하나만 고르세요", "열 이름을 눌러 '정답'을 하나만 남기세요");
  const o = Object.assign(dataOptions(), { path: st.pick.path, targets: rolePick(roles, "target"), skip: rolePick(roles, "skip"),
    channels: $("pick-ch-box").hidden ? 1 : Number($("pick-channels").value) || 1 });
  delete o.label_col;
  if (seg) { o.task = "segmentation"; if (Number($("pick-length").value)) o.length = Number($("pick-length").value); }
  if (await loadData(o)) $("csv-pick").hidden = true;
};
$("btn-role-reload").onclick = () => {
  const pick = (r) => rolePick(st.roles, r);
  if (!pick("target").length) return showMessage("정답 열을 하나 이상 고르세요", "열 이름을 눌러 '정답'으로 바꾸세요");
  if (st.summary.task === "segmentation" && pick("target").length !== 1) return showMessage("상태 열은 하나만 고르세요", "열 이름을 눌러 '정답'을 하나만 남기세요");
  loadData({ roles_reload: { targets: pick("target"), skip: pick("skip"), channels: Number($("role-channels").value) || 1,
    length: st.summary.task === "segmentation" ? Number($("role-length").value) || 0 : 0 } });
};
function renderFcBox(s) {                               // 예측 데이터: 맞힐 열·입력·예측 길이·간격을 바꿔 다시 만들기
  const fc = s.forecast;
  $("fc-box").hidden = !fc;
  if (!fc) return;
  $("fc-column").innerHTML = fc.columns.map((c) => `<option${c === fc.column ? " selected" : ""}>${esc(c)}</option>`).join("");
  $("fc-column").disabled = fc.columns.length < 2;
  $("fc-L").value = s.length; $("fc-H").value = fc.horizon; $("fc-stride").value = fc.stride;
  $("fc-note").textContent = `시험은 뒤쪽 구간 (시간 순서 그대로 나눠 미래를 미리 보지 않음). 학습 창 ${s.n_train}개 · 시험 창 ${s.n_test}개`;
}
$("btn-fc-reload").onclick = () => loadData({ forecast_reload: { column: $("fc-column").value, L: Number($("fc-L").value) || 0,
  H: Number($("fc-H").value) || 0, stride: Number($("fc-stride").value) || 0 } });
$("btn-csv").onclick = () => {
  if (!pathVal("csv-path")) return showMessage("CSV 파일 경로를 넣으세요", "예: C:\\data\\ecg.csv");
  const o = Object.assign(dataOptions(), { path: pathVal("csv-path") });
  const own = st.sub && !st.autoTask;
  if (own && st.sub.input === "image") { o.image_width = Number($("image-width").value) || 0; return loadData(o); }   // 픽셀 CSV: 비우면 정사각형
  if (own && st.sub.task === "forecast") return loadData(o);                  // 시계열 예측: 맞힐 열은 불러온 뒤 고름
  openPick(pathVal("csv-path"));                                                // 그 밖: 열 고르기를 먼저
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
    const prev = seriesColumns(), same = st.seriesScanPath === path;   // 같은 폴더를 다시 볼 때만 고른 값 유지
    st.seriesScanPath = path;
    const keep = same && prev.length && prev.every((c) => s.columns.includes(c)) ? prev : [s.default_column];
    $("series-columns").innerHTML = s.columns.map((c) => `<label class="chk"><input type="checkbox" value="${esc(c)}"${keep.includes(c) ? " checked" : ""}> ${esc(c)}</label>`).join("");
    if (!same || !$("series-length").value) $("series-length").value = s.default_length;
    const none = s.mode === "none";
    $("btn-series-load").hidden = none;
    $("series-guide").hidden = !none;
    if (none) $("series-guide").innerHTML = `<b>정답(클래스)이 없어 학습할 수 없습니다.</b> 아래 중 하나로 정답을 알려 주세요.<ol>` +
      `<li>클래스마다 하위 폴더로 나누기: <code>폴더/정상/*.csv</code>, <code>폴더/이상/*.csv</code></li>` +
      `<li>정답 표 CSV를 만들어 위 '정답 표 CSV' 칸에 넣기: 첫 열 = 파일 이름, 둘째 열 = 정답 (예: <code>Cycle_10,정상</code>)</li>` +
      (WEB ? `<li><a href="label.html" target="_blank" rel="noopener">라벨링 페이지에서 정답 붙이기</a> (새 탭): 같은 폴더를 고르고 정답을 붙인 뒤, 여기 '라벨링한 데이터'에서 불러오기</li></ol>`
        : `<li><a href="/label?folder=${encodeURIComponent(path)}&column=${encodeURIComponent(seriesColumns()[0] || "")}" target="_blank">` +
          `라벨링 작업실에서 정답 붙이기</a> (새 탭): 격자로 훑고, 규칙으로 한꺼번에, 하나씩 크게 보며</li></ol>`);
  } catch (e) {
    showError(e);
    $("data-issues").innerHTML = (e.issues || []).map((i) => `<li>${esc(i.message)}<span class="hint">→ ${esc(i.hint)}</span></li>`).join("");
  }
}
$("btn-series-scan").onclick = scanSeries;
function seriesColumns() { return [...document.querySelectorAll("#series-columns input:checked")].map((el) => el.value); }   // 둘 이상이면 여러 채널
$("btn-label").onclick = () => window.open(WEB ? "label.html" : "/label", "_blank");   // 라벨링은 따로 페이지 (새 탭)
$("series-path").oninput = () => {                        // 폴더를 바꾸면 이전 살펴보기 결과는 버림
  st.seriesScan = null;
  $("series-box").hidden = true;
};
$("series-label-file").onchange = () => { if (pathVal("series-path")) scanSeries(); };
$("btn-series-load").onclick = async () => {
  const cols = seriesColumns();
  if (!cols.length) return showMessage("값 열을 하나 이상 고르세요", "둘 이상 고르면 여러 채널 시계열이 됩니다");
  const o = Object.assign(dataOptions(), { series_folder: pathVal("series-path"), column: cols.length > 1 ? cols : cols[0],
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
        onHistory: (u, r) => { $("btn-undo").disabled = !u; $("btn-redo").disabled = !r; },
        onZoom: (z) => ($("zoom-reset").textContent = Math.round(z * 100) + "%") });
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
  fitSoon();
}
// 모델을 통째로 바꾸면(대표 모델·프로젝트 열기) 처음에는 전체 보기. 판이 보일 때 맞춤 (안 보이면 모델 화면을 열 때)
function fitSoon() {
  st.fitPending = true;
  if (!st.ed || $("step-model").hidden || !$("canvas").clientWidth) return;
  requestAnimationFrame(() => { if (st.fitPending && st.ed) { st.ed.zoomFit(); st.fitPending = false; } });
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
function renderConfusion(el, cm, recall, caption, pickable = false, classNames = null) {
  $("cm-pick").hidden = true;
  if (!cm) return ($(el).innerHTML = "");
  const names = classNames || st.summary.classes;
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
    ...(p.frozen ? [["사전학습 (고정)", p.frozen.toLocaleString()]] : []),
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
$("zoom-in").onclick = () => st.ed && st.ed.zoomIn();
$("zoom-out").onclick = () => st.ed && st.ed.zoomOut();
$("zoom-reset").onclick = () => st.ed && st.ed.zoomReset();
$("zoom-fit").onclick = () => st.ed && st.ed.zoomFit();
$("btn-redo").onclick = () => st.ed && st.ed.redo();

// ------------------------------------------------------------ ③ 학습
function trainSettings() {
  lossTerms();                                              // 맞지 않는 손실 설정은 여기서 기본으로
  return { epochs: Number($("epochs").value) || 30, lr: Number($("lr").value) || 0.001, batch_size: Number($("batch").value) || 32,
    device: $("device").value, keep_best: $("keep-best").checked, early_stop: Number($("early-stop").value) || 0,
    ...(st.loss ? { loss: st.loss } : {}) };
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
  const n = progress.length, map = metricName === "map50", auc = metricName === "auc", recon = metricName === "recon";
  const acc = metricName !== "r2" && metricName !== "mean" && !auc && !recon;
  const mmax = recon ? Math.max(1e-12, ...progress.map((p) => p.test_metric)) : 1;      // 복원 오차: 가장 큰 값 기준
  const yv = (f) => T + (1 - f) * ph;
  let s = "";
  for (const f of [0, 0.5, 1]) {
    s += `<line x1="${L}" y1="${yv(f)}" x2="${L + pw}" y2="${yv(f)}" stroke="#e3e6eb"/>` +
      `<text x="${L - 6}" y="${yv(f) + 4}" font-size="11" text-anchor="end" fill="#4f46e5">${acc ? f * 100 + "%" : recon ? Number((f * mmax).toPrecision(2)) : f}</text>`;
  }
  const loss = progress.map((p) => p.loss), vloss = progress.map((p) => p.val_loss), hasVal = vloss.some((v) => v !== undefined && v !== null);
  const lmax = Math.max(0, ...loss, ...(hasVal ? vloss.filter((v) => v !== undefined && v !== null) : [])) || 1;
  s += `<text x="${L + pw + 6}" y="${yv(1) + 4}" font-size="11" fill="#f97316">${Number(lmax.toPrecision(2))}</text>` +
    `<text x="${L + pw + 6}" y="${yv(0) + 4}" font-size="11" fill="#f97316">0</text>`;
  if (!n) s += `<text x="${L + pw / 2}" y="${T + ph / 2}" font-size="13" text-anchor="middle" fill="#94a3b8">[학습 시작]을 누르면 epoch마다 곡선이 그려집니다</text>`;
  if (n) {
    const xs = (i) => L + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const lp = loss.map((v, i) => `${xs(i).toFixed(1)},${yv(v / lmax).toFixed(1)}`);
    const mp = progress.map((p, i) => `${xs(i).toFixed(1)},${yv(Math.max(0, Math.min(1, p.test_metric / mmax))).toFixed(1)}`);
    if (hasVal) {                                           // 검증 loss (점선)과 가장 좋았던 epoch (세로 점선)
      s += `<polyline fill="none" stroke="#f97316" stroke-width="1.5" stroke-dasharray="4 3" points="${vloss.map((v, i) => `${xs(i).toFixed(1)},${yv(v / lmax).toFixed(1)}`).join(" ")}"/>`;
      const be = progress[n - 1].best_epoch, bi = progress.findIndex((p) => p.epoch === be);
      if (bi >= 0) s += `<line x1="${xs(bi)}" y1="${T}" x2="${xs(bi)}" y2="${T + ph}" stroke="#16a34a" stroke-width="1.2" stroke-dasharray="3 3"/>`;
    }
    s += `<polyline fill="none" stroke="#f97316" stroke-width="2" points="${lp.join(" ")}"/>` +
      `<polyline fill="none" stroke="#4f46e5" stroke-width="2" points="${mp.join(" ")}"/>`;
    if (n < 3) s += lp.map((p) => `<circle cx="${p.split(",")[0]}" cy="${p.split(",")[1]}" r="3.5" fill="#f97316"/>`).join("") +   // 점이 적으면 선이 안 보임
      mp.map((p) => `<circle cx="${p.split(",")[0]}" cy="${p.split(",")[1]}" r="3.5" fill="#4f46e5"/>`).join("");
    s += n === 1 ? `<text x="${L + pw / 2}" y="${H - 8}" font-size="11" text-anchor="middle" fill="#6b7280">epoch 1</text>`
      : `<text x="${L}" y="${H - 8}" font-size="11" fill="#6b7280">epoch 1</text>` +
        `<text x="${L + pw}" y="${H - 8}" font-size="11" text-anchor="end" fill="#6b7280">${n}</text>`;
  }
  s += `<line x1="${L}" y1="12" x2="${L + 18}" y2="12" stroke="#4f46e5" stroke-width="2"/>` +
    `<text x="${L + 24}" y="16" font-size="12" fill="#4f46e5">시험 ${map ? "mAP@0.5 (왼쪽 축, 높을수록 좋음)" : auc ? "AUC (왼쪽 축, 1에 가까울수록 좋음)" : recon ? "복원 오차 (왼쪽 축, 낮을수록 좋음)" : acc ? "정확도 (왼쪽 축, 높을수록 좋음)" : metricName === "mean" ? "평균 (왼쪽 축, 1에 가까울수록 좋음)" : "R² (왼쪽 축, 1에 가까울수록 좋음)"}</text>` +
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
  const epochs = trainSettings().epochs;
  trainBusy(true);
  $("btn-train").disabled = true;                              // 두 번 눌러 두 학습이 돌지 않게
  try {
    const { id } = await api("/api/train", { graph, settings: trainSettings() });
    st.trainJob = id;
    const json = sig(graph);
    $("train-status").textContent = "학습 중…";
    const ae = st.summary.anomaly;                              // 이상 탐지: 이상 샘플이 있으면 AUC, 없으면 복원 오차
    const metric = { classification: "accuracy", detection: "map50", segmentation: "accuracy", multi: "mean" }[st.summary.task] || (ae ? (ae.n_anomaly ? "auc" : "recon") : "r2");
    const t0 = performance.now();
    const j = await pollJob(id, (j) => {
      drawCurve(j.progress, metric);
      renderLossParts(j.progress);
      const last = j.progress[j.progress.length - 1];
      const left = last && last.epoch < epochs ? ` · 약 ${fmtLeft(((performance.now() - t0) / 1000 / last.epoch) * (epochs - last.epoch))} 남음` : "";
      if (last) $("train-status").textContent = `epoch ${last.epoch} · loss ${last.loss.toFixed(4)} · 시험 ${metricText(metric, last.test_metric)}${left}`;
      trainBusy(true, last ? last.epoch / epochs : null);
    });
    if (j.state === "done") {
      st.trained = true;
      st.trainedJson = json;
      st.trainedGraph = graph;
      renderTrainResult(j.result);
      $("sample-box").hidden = false;
      showSample();
      const r = j.result;
      $("train-status").textContent = r.stopped ? `멈춤 (${r.epochs_done} epoch)` : "끝남";
      loadRuns();
      showMessage(`${r.stopped ? `학습 멈춤 (${r.epochs_done} epoch)` : "학습 끝남"}: 시험 ${metricText(r.metric_name, r.test_metric)}`,
        "아래 결과와 '샘플 하나 따라가 보기'를 보세요", "", true);
    } else if (j.state === "cancelled") $("train-status").textContent = "멈춤";
    else { $("train-status").textContent = "실패"; showMessage("학습 실패", "학습률을 낮추거나 블록을 바꿔 보세요", j.error); }
  } catch (e) { $("train-status").textContent = ""; showError(e); }
  st.training = false;
  trainBusy(false);
  markSteps();
};
function trainBusy(on, frac = null) {                         // 학습 중: 도는 표시 + epoch 막대 (첫 epoch 전에는 흐르는 막대)
  $("train-status").classList.toggle("busy", on);
  const bar = $("train-bar");
  bar.hidden = !on;
  bar.classList.toggle("indet", on && frac === null);
  bar.firstElementChild.style.width = on && frac !== null ? `${Math.min(100, frac * 100)}%` : "";
}
$("btn-train-stop").onclick = () => st.trainJob && api(`/api/jobs/${st.trainJob}/cancel`, {}).catch(showError);
function renderSegResult(r) {                             // 구간 분할: 시점 정확도·혼동 행렬, 시험 창의 정답·예측 띠
  const names = st.summary.classes, a = r.test_metric;
  const v = a >= 0.9 ? ["good", `시험 시점의 ${pct(a)}를 맞혔습니다.`, "아래 창에서 경계가 어긋나는 곳을 보세요. ⑤ 예측에서 새 데이터에 써 볼 수 있습니다."]
    : a >= 0.7 ? ["warn", `시험 시점의 ${pct(a)}를 맞혔습니다. 짧은 상태를 놓치는 곳이 있습니다.`, "Epochs를 늘리거나 다른 대표 모델(U-Net식)을 써 보세요."]
    : ["bad", `시험 시점의 ${pct(a)}만 맞혔습니다.`, "Epochs를 늘리거나, 필터를 더 넓게 보는 모델(TCN)을 써 보세요."];
  $("train-result").innerHTML = kv([["Epochs", epochsText(r)], ["시험 시점 정확도", `<b>${pct(a)}</b>`], ["학습 시점 정확도", pct(r.train_metric)]]) +
    `<div class="verdict ${v[0]}"><b>${esc(v[1])}</b><div>다음: ${esc(v[2])}</div></div>`;
  renderWarnings("train-warnings", r.warnings);
  renderConfusion("train-confusion", r.confusion, r.recall, "시점마다 센 혼동 행렬");
  $("train-scatter").innerHTML = "";
  $("det-box").hidden = false;
  $("det-box").innerHTML = `<h3>시험 창 (상태가 많이 바뀌는 것)</h3><div class="seg-gallery">` + r.seg_samples.map((p) =>
    `<div class="fc-card">${svgLine(p.x, 300, 60)}${segStrip(p.truth, 300)}${segStrip(p.pred, 300)}<div class="muted">맞힌 시점 ${pct(p.acc)}</div></div>`).join("") +
    `</div><p class="muted fc-legend">띠: 위 = 정답, 아래 = 예측</p>` + segLegend(names);
}
function renderMultiResult(r, pick = 0) {                  // 정답 여러 개: 정답마다 성적 탭 → 그 정답의 혼동 행렬·산점도
  st.multiResult = r;
  $("det-box").hidden = true;
  pick = Math.max(0, Math.min(pick, r.heads.length - 1));
  const one = (h, v) => (v === null || v === undefined ? "-" : h.kind === "class" ? pct(v) : `R² ${v.toFixed(3)}`);
  const h = r.heads[pick];
  const rows = [["Epochs", epochsText(r)], ["시험 평균", `<b>${r.test_metric.toFixed(3)}</b>`], ["학습 평균", r.train_metric === null || r.train_metric === undefined ? "-" : r.train_metric.toFixed(3)]];
  const val = (v) => (v === null || v === undefined ? "-" : h.kind === "class" ? pct(v) : v.toFixed(3));   // 이름 칸에 정확도·R²가 있으니 값만
  const detail = [[`시험 ${h.kind === "class" ? "정확도" : "R²"}`, `<b>${val(h.metric)}</b>`], [`학습 ${h.kind === "class" ? "정확도" : "R²"}`, val(h.train_metric)]];
  if (h.kind === "value") detail.push(["시험 평균 오차 MAE", Number(h.mae.toPrecision(4))]);
  $("train-result").innerHTML = kv(rows) +
    `<div class="head-tabs" role="tablist">${r.heads.map((x, i) => `<button type="button" role="tab" data-head="${i}" aria-selected="${i === pick}">${esc(x.name)} <b>${one(x, x.metric)}</b></button>`).join("")}</div>` +
    kv(detail);
  renderWarnings("train-warnings", r.warnings);
  if (h.kind === "class") {
    renderConfusion("train-confusion", h.confusion, h.recall, `시험 데이터 · ${h.name}`, false, h.classes);
    $("train-scatter").innerHTML = "";
  } else {
    renderConfusion("train-confusion", null);
    renderScatter("train-scatter", h.scatter, h.mae, `시험 데이터 · ${h.name}`);
  }
}
$("train-result").addEventListener("click", (e) => {
  const b = e.target.closest("[data-head]");
  if (b && st.multiResult) renderMultiResult(st.multiResult, Number(b.dataset.head));
});
function renderTrainResult(r) {
  if (r.heads) return renderMultiResult(r);
  if (r.seg_samples) return renderSegResult(r);
  if (r.metric_name === "map50") return renderDetResult(r);
  if (r.fc_samples) return renderFcResult(r);
  if (r.ae_samples) return renderAeResult(r);
  $("det-box").hidden = true;
  const isAcc = r.metric_name === "accuracy";
  const f = (v) => (v === null ? "-" : isAcc ? pct(v) : v.toFixed(3));
  const m = (v) => (v === null || v === undefined ? "-" : Number(v.toPrecision(4)));
  const has = (v) => v !== null && v !== undefined;
  const base = isAcc ? baselineOf(r) : null;
  const rows = [["Epochs", epochsText(r)], [`시험 ${isAcc ? "정확도" : "R²"}`, `<b>${f(r.test_metric)}</b>`]];
  if (has(r.train_metric)) rows.push([`학습 ${isAcc ? "정확도" : "R²"}`, f(r.train_metric)]);
  if (has(base)) rows.push(["찍기 기준", `${pct(base)} <span class="muted">(가장 많은 클래스만 답할 때)</span>`]);
  if (!isAcc) rows.push(["시험 평균 오차 MAE", m(r.mae)], ["찍기 기준", `R² 0 <span class="muted">(늘 평균값만 답할 때)</span>`]);
  const v = verdict(r, base);
  $("train-result").innerHTML = kv(rows) + (v ? `<div class="verdict ${v[0]}"><b>${esc(v[1])}</b><div>다음: ${esc(v[2])}</div></div>` : "");
  renderWarnings("train-warnings", r.warnings);
  renderConfusion("train-confusion", r.confusion, r.recall, "시험 데이터", Array.isArray(r.pred));
  renderScatter("train-scatter", r.scatter, r.mae, "시험 데이터");
}
// ---------------- 시계열 예측 결과: MAE (원래 단위)·단순 예측과 비교·몇 칸 뒤까지 맞히는지·시험 창 예시
function fcVerdict(r) {
  const better = r.naive_mae > 0 ? 1 - r.mae / r.naive_mae : 0, tr = r.train_metric, t = r.test_metric;
  if (better <= 0) return ["bad", `시험 구간의 평균 오차(MAE ${Number(r.mae.toPrecision(4))})가 단순 예측(${r.naive_name}, ${Number(r.naive_mae.toPrecision(4))})보다 크거나 같습니다.`,
    "Epochs를 늘리거나(50–100), 다른 대표 모델(MLP·Transformer)을 써 보거나, ② 데이터에서 입력 길이를 늘려 보세요."];
  if (tr !== null && tr !== undefined && tr - t > 0.2) return ["warn", `학습 구간은 R² ${tr.toFixed(2)}인데 시험 구간은 ${t.toFixed(2)}입니다. 학습 구간을 외운 것일 수 있습니다 (과적합).`,
    "층·채널을 줄이거나 Dropout을 넣거나, 데이터(기간)를 더 모아 보세요."];
  if (better >= 0.3) return ["good", `단순 예측(${r.naive_name})보다 오차가 ${Math.round(better * 100)}% 작습니다.`,
    "아래 그래프에서 몇 칸 뒤부터 오차가 커지는지 보고, ⑤ 예측에서 최근 데이터로 다음 값을 예측해 보세요."];
  return ["warn", `단순 예측(${r.naive_name})보다 오차가 ${Math.round(better * 100)}% 작습니다. 나아졌지만 더 좋아질 수 있습니다.`,
    "Epochs를 늘리거나 다른 대표 모델을 써 보고, 입력 길이(지난 몇 개를 볼지)를 바꿔 보세요."];
}
function renderFcResult(r) {
  const m = (v) => (v === null || v === undefined ? "-" : Number(v.toPrecision(4)));
  const better = r.naive_mae > 0 ? 1 - r.mae / r.naive_mae : 0;
  const rows = [["Epochs", epochsText(r)], ["시험 평균 오차 MAE", `<b>${m(r.mae)}</b> <span class="muted">(원래 단위)</span>`], ["시험 RMSE", m(r.rmse)],
    ["단순 예측 MAE", `${m(r.naive_mae)} <span class="muted">(${esc(r.naive_name)})</span>`],
    ["단순 예측보다", better >= 0 ? `오차 ${Math.round(better * 100)}% 작음` : `오차 ${Math.round(-better * 100)}% 큼`],
    ["시험 R²", r.test_metric.toFixed(3)]];
  if (r.train_metric !== null && r.train_metric !== undefined) rows.push(["학습 R²", r.train_metric.toFixed(3)]);
  const v = fcVerdict(r);
  $("train-result").innerHTML = kv(rows) + `<div class="verdict ${v[0]}"><b>${esc(v[1])}</b><div>다음: ${esc(v[2])}</div></div>`;
  renderWarnings("train-warnings", r.warnings);
  renderConfusion("train-confusion", null);
  const H = r.mae_by_step.length;
  $("train-scatter").innerHTML = `<h3>몇 칸 뒤까지 맞히나 (시험 구간)</h3>` +
    svgSeries([{ v: r.naive_by_step, color: FC_COLORS.naive, dash: "4 3" }, { v: r.mae_by_step, color: FC_COLORS.pred, width: 2 }], 300, 120) +
    `<p class="muted">가로 = 1칸 뒤 … ${H}칸 뒤, 세로 = 평균 오차 · <span style="color:${FC_COLORS.pred}">━</span> 모델 · <span style="color:${FC_COLORS.naive}">┅</span> 단순 예측</p>`;
  $("det-box").hidden = false;
  $("det-box").innerHTML = `<div class="row det-head"><h3>시험 구간에서 예측한 모습</h3></div>` +
    `<p class="muted fc-legend"><span style="color:${FC_COLORS.hist}">━</span> 모델이 본 지난 값 · <span style="color:${FC_COLORS.y}">━</span> 실제 · <span style="color:${FC_COLORS.pred}">━</span> 예측</p>` +
    `<div class="fc-gallery">${r.fc_samples.map((p) => {
      const hist = p.history.slice(-Math.max(H * 3, 24));
      return `<div class="fc-card">${svgSeries([{ v: hist, color: FC_COLORS.hist }, { v: p.y, x0: hist.length, color: FC_COLORS.y, width: 1.8 },
        { v: p.pred, x0: hist.length, color: FC_COLORS.pred, width: 1.8 }], 280, 100, hist.length)}<div class="muted">시험 창 ${p.index + 1}</div></div>`;
    }).join("")}</div>`;
}
// ---------------- 이상 탐지 결과: AUC·기준값·이상 점수 분포(기준값 바꿔 보기)·점수 높은 샘플의 입력과 되살린 값
function aeCounts(r, th) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  r.scores.forEach((s, i) => { const p = s > th, y = r.labels[i] === 1; if (p && y) tp++; else if (p) fp++; else if (y) fn++; else tn++; });
  return { tp, fp, fn, tn };
}
function aeVerdict(r) {
  const dist = r.mode === "distance";
  if (r.metric_name === "recon" && dist) return ["warn", "시험 데이터에 이상 샘플이 없어 얼마나 잘 찾는지는 잴 수 없습니다.",
    "이상 샘플(결함 등)이 조금이라도 있으면 함께 넣어 보세요. 정답 열에 정상이 아닌 이름이 있으면 이상으로 봅니다."];
  if (dist && r.auc < 0.9) return [r.auc < 0.6 ? "bad" : "warn", `AUC ${r.auc.toFixed(2)} — 이상과 정상의 점수가 ${r.auc < 0.6 ? "거의 " : ""}섞여 있습니다.`,
    "사전학습 블록의 입력 크기를 바꿔 보거나(예: 224), 정상 그림을 더 모아 보세요. 아래 분포에서 기준값도 조정해 보세요."];
  if (r.metric_name === "recon") return ["warn", "시험 데이터에 이상 샘플이 없어 얼마나 잘 찾는지는 잴 수 없습니다. 정상을 얼마나 잘 되살리는지만 봅니다.",
    "이상 샘플(결함 등)이 조금이라도 있으면 함께 넣어 보세요. 정답 열에 정상이 아닌 이름이 있으면 이상으로 봅니다."];
  const a = r.auc;
  if (a < 0.6) return ["bad", `AUC ${a.toFixed(2)} — 이상과 정상의 점수가 거의 섞여 있습니다.`,
    "Epochs를 늘리거나, 다른 대표 모델(Conv 오토인코더)을 써 보거나, 가운데 Dense를 더 작게 해 보세요."];
  if (a >= 0.9) return ["good", `AUC ${a.toFixed(2)} — 이상 샘플의 점수가 대부분 정상보다 높습니다.`,
    "아래 분포에서 기준값을 옮겨 놓치는 이상과 잘못 잡는 정상 사이를 맞추고, [이 기준값을 예측에 쓰기]를 누르세요."];
  return ["warn", `AUC ${a.toFixed(2)} — 이상을 어느 정도 가려내지만 섞이는 것이 있습니다.`,
    "Epochs를 늘리거나 다른 대표 모델을 써 보고, 아래 분포에서 기준값을 조정해 보세요."];
}
function aeHist(r, th) {                                   // 점수 분포: 정상(초록)·이상(빨강) 막대, 기준값 세로선
  const all = r.scores.concat(r.train_scores), sorted = all.slice().sort((a, b) => a - b);
  const hi = Math.max(sorted[Math.floor(sorted.length * 0.99)] || 1, th * 1.2), nb = 30, W = 560, Hh = 150, B = 22;
  const bins = (arr) => { const c = new Array(nb).fill(0); arr.forEach((s) => c[Math.min(nb - 1, Math.floor((s / hi) * nb))]++); return c; };
  const nor = bins(r.scores.filter((_, i) => r.labels[i] === 0)), ab = bins(r.scores.filter((_, i) => r.labels[i] === 1));
  const mx = Math.max(1, ...nor, ...ab), bw = (W - 8) / nb;
  let s = `<svg viewBox="0 0 ${W} ${Hh}" width="${W}" height="${Hh}" class="ae-hist">`;
  for (let k = 0; k < nb; k++) {
    const x = 4 + k * bw, hn = (nor[k] / mx) * (Hh - B - 6), ha = (ab[k] / mx) * (Hh - B - 6);
    s += `<rect x="${x}" y="${Hh - B - hn}" width="${bw / 2 - 1}" height="${hn}" fill="#16a34a" opacity=".75"/>` +
      `<rect x="${x + bw / 2}" y="${Hh - B - ha}" width="${bw / 2 - 1}" height="${ha}" fill="#e11d48" opacity=".75"/>`;
  }
  const tx = 4 + Math.min(1, th / hi) * (W - 8);
  s += `<line x1="${tx}" y1="0" x2="${tx}" y2="${Hh - B}" stroke="#0f172a" stroke-width="2" stroke-dasharray="4 3"/>` +
    `<text x="${tx + 4}" y="12" font-size="11" fill="#0f172a">기준값</text>` +
    `<text x="4" y="${Hh - 6}" font-size="11" fill="#6b7280">0</text><text x="${W - 4}" y="${Hh - 6}" font-size="11" text-anchor="end" fill="#6b7280">${Number(hi.toPrecision(2))}+</text>`;
  return s + `</svg>`;
}
function renderAeResult(r) {
  const m = (v) => (v === null || v === undefined ? "-" : Number(v.toPrecision(3)));
  const both = r.metric_name === "auc";
  const rows = [["Epochs", epochsText(r)]];
  if (both) rows.push(["시험 AUC", `<b>${r.auc.toFixed(3)}</b> <span class="muted">(이상 하나·정상 하나를 뽑을 때 이상 점수가 더 클 확률, 0.5 = 찍기)</span>`]);
  else rows.push(["시험 정상의 평균 이상 점수", `<b>${m(r.test_metric)}</b>`]);
  rows.push(["기준값 (처음)", `${m(r.threshold)} <span class="muted">(학습 정상의 95%가 이 아래)</span>`]);
  const v = aeVerdict(r);
  $("train-result").innerHTML = kv(rows) + `<div class="verdict ${v[0]}"><b>${esc(v[1])}</b><div>다음: ${esc(v[2])}</div></div>`;
  renderWarnings("train-warnings", r.warnings);
  renderConfusion("train-confusion", null);
  $("train-scatter").innerHTML = "";
  st.aeTh = r.threshold;
  const smax = Math.max(...r.scores, ...r.train_scores, r.threshold * 2);
  const tab = inputKind() === "tabular" && !st.summary.image, feats = st.summary.features || [];
  const emax = Math.max(1e-12, ...r.ae_samples.flatMap((p) => p.err || []));    // 차이 막대·배경은 모든 샘플에 같은 눈금
  const dist = r.mode === "distance", im = st.summary.image;                       // 정상과의 거리: 되살린 값이 없음
  $("det-box").hidden = false;
  $("det-box").innerHTML = `<div class="row det-head"><h3>이상 점수 분포 (시험)</h3>` +
    `<label class="muted det-th">기준값 <input id="ae-th" type="range" min="0" max="1000" step="1" value="${Math.round(Math.sqrt(r.threshold / smax) * 1000)}"> <b id="ae-th-v"></b></label>` +
    `<button id="btn-ae-th">이 기준값을 예측에 쓰기</button></div>` +
    `<div id="ae-hist"></div><div id="ae-at" class="kv ae-at"></div>` +
    `<p class="muted"><span style="color:#16a34a">■</span> 정상 · <span style="color:#e11d48">■</span> 이상 — 점수 = ${dist ? "정상 그림들의 특징 분포에서 떨어진 정도" : "되살린 값과 입력의 차이 (제곱 평균)"}. 기준값보다 크면 이상으로 판정</p>` +
    `<h3>점수가 높은 시험 샘플 (마지막 둘은 ${dist ? "가장 정상에 가까운 것" : "가장 잘 되살린 것"})</h3><div class="ae-gallery">${r.ae_samples.map((p) => {
      const tag = `${p.label === 1 ? "정답 이상" : "정답 정상"} · 점수 ${m(p.score)}`;
      const pic = im
        ? [p.x].concat(p.recon ? [p.recon] : []).map((v) => `<img src="${grayURL(v, im[0], im[1])}" width="96" height="${Math.round((96 * im[0]) / im[1])}" class="ae-img" alt="">`).join("")
        : tab
        ? `<table class="attn-cols">${p.err.map((e, j) => `<tr><td>${esc(feats[j] || "열 " + (j + 1))}</td><td>${m(p.x[j])} → ${m(p.recon[j])}</td>` +
          `<td><span style="width:${Math.max(1, Math.round(Math.min(1, e / emax) * 120))}px;background:#ef4444"></span></td></tr>`).join("")}</table>`
        : svgSeries([{ v: p.x, color: "#0f172a" }, { v: p.recon, color: "#4f46e5", dash: "4 2" }], 300, 90, null, p.err, emax);
      return `<div class="fc-card ${p.label === 1 ? "ab" : ""}">${pic}<div class="muted">${esc(tag)}</div></div>`;
    }).join("")}</div>` +
    (dist ? "" : `<p class="muted fc-legend">${im ? "왼쪽: 입력 · 오른쪽: 되살린 그림" : tab ? "열마다: 입력 → 되살린 값, 빨간 막대 = 그 열의 차이 (이상한 열)" : `<span style="color:#0f172a">━</span> 입력 · <span style="color:#4f46e5">┅</span> 되살린 값 · 빨간 배경 = 차이가 큰 곳`}</p>`);
  const draw = () => {
    const th = st.aeTh, c = aeCounts(r, th);
    $("ae-th-v").textContent = m(th);
    $("ae-hist").innerHTML = aeHist(r, th);
    $("ae-at").innerHTML = both ? kv([["이상을 이상으로 (찾음)", `${c.tp} / ${c.tp + c.fn} <span class="muted">(재현율 ${pct(c.tp + c.fn ? c.tp / (c.tp + c.fn) : null)})</span>`],
      ["정상을 이상으로 (잘못 잡음)", `${c.fp} / ${c.fp + c.tn}`], ["이상이라고 한 것 중 진짜 이상", pct(c.tp + c.fp ? c.tp / (c.tp + c.fp) : null)]])
      : kv([["이상으로 판정되는 시험 정상", `${c.fp} / ${c.fp + c.tn}`]]);
  };
  $("ae-th").oninput = (e) => { st.aeTh = (Number(e.target.value) / 1000) ** 2 * smax; draw(); };
  $("btn-ae-th").onclick = async () => {
    try {
      await api("/api/anomaly/threshold", { threshold: st.aeTh });
      r.threshold = st.aeTh;
      showMessage(`기준값을 ${m(st.aeTh)}(으)로 정했음`, "⑤ 예측과 샘플 따라가기의 이상 판정에 이 값을 씁니다", "", true);
    } catch (e) { showError(e); }
  };
  draw();
}
function detVerdict(r) {                               // 탐지 결과를 쉬운 말로 + 다음에 해 볼 일
  const t = r.test_metric, tr = r.train_metric;
  if (t === null || t === undefined) return null;
  if (t < 0.1) return ["bad", "아직 사물을 거의 찾지 못합니다.",
    "Epochs를 늘리거나(50–100), '깊은 탐지 모델'을 써 보거나, 상자를 그린 그림을 더 모아 보세요."];
  if (tr !== null && tr !== undefined && tr - t > 0.2)
    return ["warn", `학습 그림은 ${pct(tr)}인데 시험 그림은 ${pct(t)}입니다. 학습 그림을 외운 것일 수 있습니다 (과적합).`,
      "상자를 그린 그림을 더 모으거나, Epochs를 줄이거나, 채널을 줄여 보세요."];
  if (t >= 0.7) return ["good", `시험 그림에서 mAP@0.5 ${pct(t)} — 사물의 위치와 종류를 잘 찾습니다.`,
    "아래 시험 그림에서 놓친 것(점선만 있는 상자)과 잘못 찾은 것을 보고, ⑤ 예측에서 새 그림에 써 보세요."];
  return ["warn", `시험 그림에서 mAP@0.5 ${pct(t)}입니다. 찾긴 하지만 더 좋아질 수 있습니다.`,
    "Epochs를 늘리거나 '깊은 탐지 모델'로 바꿔 보고, 아래 그림에서 어떤 사물을 놓치는지 보세요."];
}
function renderDetResult(r) {
  const cls = st.summary.classes || [], pr = r.pr || {};
  const rows = [["Epochs", epochsText(r)], ["시험 mAP@0.5", `<b>${pct(r.test_metric)}</b>`]];
  if (r.train_metric !== null && r.train_metric !== undefined) rows.push(["학습 mAP@0.5", pct(r.train_metric)]);
  (r.ap || []).forEach((a, k) => { if (a !== null) rows.push([`AP '${cls[k]}'`, pct(a)]); });
  if (pr.precision !== undefined) rows.push(["점수 50% 이상만 볼 때",
    `찾은 것 중 맞음 ${pct(pr.precision)} · 정답 중 찾음 ${pct(pr.recall)} <span class="muted">(잘못 찾음 ${pr.fp} · 놓침 ${pr.fn})</span>`]);
  const v = detVerdict(r);
  $("train-result").innerHTML = kv(rows) + (v ? `<div class="verdict ${v[0]}"><b>${esc(v[1])}</b><div>다음: ${esc(v[2])}</div></div>` : "") +
    `<p class="muted det-help">mAP@0.5: 찾은 상자가 정답 상자와 절반 이상 겹치고 종류도 맞으면 맞힌 것으로 보고, 점수 높은 순으로 줄 세워 구한 정밀도·재현율 곡선의 넓이를 클래스마다 평균한 값 (1이면 완벽).</p>`;
  renderWarnings("train-warnings", r.warnings);
  renderConfusion("train-confusion", null);
  st.detPred = r.det_pred || null;
  $("train-scatter").innerHTML = "";
  $("det-box").hidden = !st.detPred;
  $("det-box").innerHTML = st.detPred ? `<div class="row det-head"><h3>시험 그림에서 찾은 사물</h3>` +
    `<label class="muted det-th">점수 <input id="det-th" type="range" min="0.25" max="0.95" step="0.05" value="${st.detTh || 0.5}"> <b id="det-th-v">${Math.round((st.detTh || 0.5) * 100)}%</b> 이상만 보기</label></div>` +
    detLegend(cls) + `<div id="det-gallery" class="det-gallery"><p class="muted">그림을 불러오는 중…</p></div>` : "";
  if (!st.detPred) return;
  $("det-th").oninput = (e) => { st.detTh = Number(e.target.value); $("det-th-v").textContent = Math.round(st.detTh * 100) + "%"; drawDetGallery(); };
  api("/api/samples", { indices: Array.from({ length: Math.min(24, st.summary.n_test) }, (_, i) => i) })
    .then((s) => { st.detItems = s.items; drawDetGallery(); })
    .catch((e) => { $("det-gallery").innerHTML = `<p class="err">${esc(e.message || e)}</p>`; });
}
function drawDetGallery() {
  const el = $("det-gallery"), s = st.summary;
  if (!el || !st.detItems || !st.detPred) return;
  const th = st.detTh || 0.5;
  el.innerHTML = st.detItems.map((it) => {
    const p = st.detPred[it.index] || [], found = p.filter((q) => q[1] >= th).length;
    return `<div class="det-card">${detImage(it.input, s.image[0], s.image[1], 168, it.label, p, s.classes, th)}` +
      `<div class="muted">시험 ${it.index + 1} · 정답 ${it.label.length} · 찾음 ${found}</div></div>`;
  }).join("");
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
  if (b.type === "output" && (st.summary.task === "forecast" || st.summary.task === "anomaly") && v.length > 1) return svgLine(v, w, 60);
  if (b.type === "output" && st.summary.task === "classification") {
    const m = Math.max(...v), ex = v.map((x) => Math.exp(x - m)), sum = ex.reduce((x, y) => x + y, 0);
    return svgBars(ex.map((x) => x / sum), Math.max(120, v.length * 60), 60, st.summary.classes, "#4f46e5");
  }
  return svgBars(v, Math.min(w, v.length * 14 + 10));
}
const shapeOf = (v) => (Array.isArray(v) ? [v.length].concat(shapeOf(v[0]) || []) : null);

// ------------------------------------------------------------ 어텐션 열지도 (샘플 따라가기)
// at = {q, k, map, mean, first}: 질문 위치 q개 × 본 위치 k개의 softmax 가중치(head 평균). 입력 위에 '어디를 봤나'를 겹쳐 그림
const heat = (t) => `rgba(239,68,68,${(0.08 + 0.72 * t).toFixed(3)})`;      // 0–1 → 빨강 진하기
function attnFocus(at, graph) {                         // 무엇을 보여 줄지: CLS 자리가 본 곳 · 점수 가중치 · 모든 위치의 평균
  const cls = (graph.nodes || []).some((n) => n.type === "cls_token");
  if (at.q === 1) return { v: at.first, what: "위치마다 점수(softmax) — 클수록 더 많이 반영" };
  if (cls && at.k > 1) return { v: at.first.slice(1), what: "맨 앞 요약 자리(CLS)가 본 곳" };
  return { v: at.mean, what: "위치마다 받은 주의 (모든 위치가 본 정도의 평균)" };
}
function gridOf(n, H, W) {                              // 토큰 n개 → 그림 비율에 맞는 격자 (gh × gw)
  for (const d of [0, -1, 1, -2, 2]) {
    const gh = Math.round(Math.sqrt((n * H) / W)) + d;
    if (gh >= 1 && n % gh === 0) return [gh, n / gh];
  }
  return null;
}
function attnOverlay(focus, input) {                    // 입력(그림·신호·표) 위에 받은 주의를 겹침
  const v = focus, n = v.length, hi = Math.max(...v) || 1, im = st.summary.image;
  if (im) {
    const [H, W] = im, g = gridOf(n, H, W);
    if (g) {
      const [gh, gw] = g, px = 180;
      let s = `<svg viewBox="0 0 ${W} ${H}" width="${px}" height="${Math.round((px * H) / W)}" class="img">` +
        `<image href="${grayURL(input, H, W)}" width="${W}" height="${H}" preserveAspectRatio="none" style="image-rendering:pixelated"/>`;
      for (let i = 0; i < gh; i++) for (let j = 0; j < gw; j++)
        s += `<rect x="${(j * W) / gw}" y="${(i * H) / gh}" width="${W / gw}" height="${H / gh}" fill="${heat(v[i * gw + j] / hi)}"><title>${(v[i * gw + j] * 100).toFixed(1)}%</title></rect>`;
      return s + `</svg>`;
    }
  }
  const feats = st.summary.features;
  if (!im && inputKind() === "tabular" && feats && feats.length === n)        // 표: 열마다 가로 막대 (열 이름이 길어도 겹치지 않게)
    return `<table class="attn-cols">${v.map((x, j) => `<tr><td>${esc(feats[j])}</td><td><span style="width:${Math.max(1, Math.round((x / hi) * 160))}px;background:${heat(x / hi)}"></span></td><td>${(x * 100).toFixed(1)}%</td></tr>`).join("")}</table>`;
  if (!im && !Array.isArray(input[0]) && input.length >= n) {          // 신호: 토큰 하나가 신호의 같은 길이 구간을 맡는다고 보고 배경에 칠함
    const w = 600, h = 70, bw = (w - 4) / n;
    let s = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`;
    v.forEach((x, j) => (s += `<rect x="${(2 + j * bw).toFixed(2)}" y="0" width="${(bw + 0.3).toFixed(2)}" height="${h}" fill="${heat(x / hi)}"><title>${j}번 위치 ${(x * 100).toFixed(1)}%</title></rect>`));
    return s + `<polyline fill="none" stroke="#1f2937" stroke-width="1.3" points="${pathOf(input, 2, 2, w - 4, h - 4)}"/></svg>`;
  }
  return svgBars(v, Math.min(600, n * 14 + 10), 60, null, "#ef4444");
}
function attnMatrix(m) {                                // 질문 위치(세로) × 본 위치(가로) 지도 → PNG
  const R = m.length, C = m[0].length, c = document.createElement("canvas");
  c.width = C; c.height = R;
  const g = c.getContext("2d"), im = g.createImageData(C, R);
  let hi = 0;
  for (const row of m) for (const x of row) if (x > hi) hi = x;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const t = m[i][j] / (hi || 1), o = (i * C + j) * 4;
    im.data[o] = 255 - Math.round(176 * t); im.data[o + 1] = 255 - Math.round(185 * t); im.data[o + 2] = 255 - Math.round(26 * t); im.data[o + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  const sc = Math.max(1, Math.floor(160 / Math.max(R, C)));
  return `<img src="${c.toDataURL()}" width="${C * sc}" height="${R * sc}" style="image-rendering:pixelated;border:1px solid #e5e7eb" alt="어텐션 지도">`;
}
function attnPic(at, graph, input) {
  const f = attnFocus(at, graph);
  let html = `<div class="attn"><div><div class="shape">어디를 봤나 · ${esc(f.what)}</div>${attnOverlay(f.v, input)}</div>`;
  if (at.q > 1) html += `<div><div class="shape">어텐션 지도 · 세로 질문 위치 ${at.q}개 × 가로 본 위치 ${at.k}개${at.q > at.map.length || at.k > at.map[0].length ? " (줄여 그림)" : ""}</div>${attnMatrix(at.map)}</div>`;
  return html + `</div>`;
}

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
  const f = (r, v) => metricText(r.results.metric_name, v);
  st.cmp = (st.cmp || []).filter((n) => runs.some((r) => r.n === n));
  $("runs").innerHTML = `<table class="tab runs"><tr><th title="둘을 고르면 비교">비교</th><th>#</th><th>언제</th><th>모델</th><th>Epochs</th><th>Learning rate</th><th>시험</th><th>학습</th><th></th></tr>` +
    runs.slice().reverse().map((r) => `<tr class="${r.current ? "cur" : ""}${r.data_ok ? "" : " other"}">` +
      `<td><input type="checkbox" data-cmp="${r.n}"${st.cmp.includes(r.n) ? " checked" : ""} aria-label="${r.n}번 비교"></td><td>${r.n}</td><td>${esc(when(r.time))}</td><td>${esc(modelName(r.graph))}</td>` +
      `<td>${r.results.epochs_done ?? "-"}</td><td>${r.settings.lr ?? "-"}</td>` +
      `<td><b>${f(r, r.results.test_metric)}</b>${best && best.n === r.n && ok.length > 1 ? ' <span class="best" title="시험 성적이 가장 좋음">★ 최고</span>' : ""}</td>` +
      `<td>${f(r, r.results.train_metric)}</td><td class="acts-cell">` +
      (r.current ? `<span class="badge rec">지금 모델</span>` : !r.data_ok ? `<span class="muted">다른 데이터</span>`
        : r.weights ? `<button data-run="${r.n}">이 모델로</button>` : "") +
      ` <button class="x-small" data-run-del="${r.n}" title="이 기록 지우기">×</button></td></tr>`).join("") + `</table>`;
  renderCompare();
}
function lossText(s) {                                      // 학습 설정의 손실 → '교차 엔트로피 ×1 + 가중치 L2 ×0.0001' (없으면 기본)
  const L = lossInfo();
  if (!s || !s.loss || !L) return "기본";
  return s.loss.map((t) => `${t.head ? t.head + " · " : ""}${L.names[t.term] || t.term}${t.node ? ` (${t.node})` : ""} ×${t.w ?? 1}`).join(" + ");
}
function blockList(g) {                                     // 모델의 블록 이름들 (그래프 순서, 입력 빼고)
  return ((g && g.nodes) || []).filter((n) => n.type !== "input").map((n) => blockName(n));
}
function renderCompare() {                                  // 학습 기록 둘을 고르면: 다른 칸만 진하게
  const pick = (st.cmp || []).map((n) => st.runs.find((r) => r.n === n)).filter(Boolean);
  $("runs-compare").hidden = pick.length !== 2;
  if (pick.length !== 2) return;
  const [a, b] = pick.sort((x, y) => x.n - y.n), f = (r, v) => metricText(r.results.metric_name, v);
  const ba = blockList(a.graph), bb = blockList(b.graph);
  const only = (x, y) => x.filter((n) => !y.includes(n));
  const rows = [
    ["모델", modelName(a.graph), modelName(b.graph)],
    ["블록", only(ba, bb).map(esc).join(", ") || "같음", only(bb, ba).map(esc).join(", ") || "같음", true],
    ["Epochs", a.results.epochs_done ?? "-", b.results.epochs_done ?? "-"],
    ["Learning rate", a.settings.lr ?? "-", b.settings.lr ?? "-"],
    ["Batch size", a.settings.batch_size ?? "-", b.settings.batch_size ?? "-"],
    ["가장 좋은 epoch 남기기", a.settings.keep_best ? "예" : "아니오", b.settings.keep_best ? "예" : "아니오"],
    ["손실", esc(lossText(a.settings)), esc(lossText(b.settings))],
    ["시험", f(a, a.results.test_metric), f(b, b.results.test_metric)],
    ["학습", f(a, a.results.train_metric), f(b, b.results.train_metric)],
  ];
  $("runs-compare").innerHTML = `<table class="tab cmp"><tr><th></th><th>${a.n}번</th><th>${b.n}번</th></tr>` +
    rows.map(([k, x, y, raw]) => `<tr class="${String(x) !== String(y) ? "diff" : ""}"><th>${k}</th><td>${raw ? x : esc(x)}</td><td>${raw ? y : esc(y)}</td></tr>`).join("") + `</table>`;
}
$("runs").addEventListener("change", (e) => {
  const n = Number(e.target.dataset.cmp);
  if (!n) return;
  st.cmp = (st.cmp || []).filter((x) => x !== n).concat(e.target.checked ? [n] : []).slice(-2);   // 둘까지 (셋째를 고르면 처음 것이 빠짐)
  renderRuns();
});
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
  if (s.keep_best !== undefined) $("keep-best").checked = !!s.keep_best;
  if (s.early_stop !== undefined) $("early-stop").value = String(s.early_stop || 0);
  st.loss = s.loss || null;
  renderLoss();
}
// ------------------------------------------------------------ 손실 조립 (losses.py): 손실 = Σ 가중치 × 항, 정답 종류에 맞는 항만
const LOSS_OPT_LABEL = { smooth: "라벨 스무딩", gamma: "γ", delta: "δ", balance: "드문 클래스 무겁게" };
function lossInfo() { return st.summary && st.summary.losses; }
function lossFits(L, terms) {                              // 이 데이터에 맞는 손실인지 (정답 이름·항)
  const heads = Object.fromEntries((L.heads || []).map((h) => [h.name, h]));
  return terms.every((t) => (t.head ? heads[t.head] && heads[t.head].allowed.includes(t.term) : L.allowed.includes(t.term)));
}
function lossTerms() {
  const L = lossInfo();
  if (st.loss && L && !lossFits(L, st.loss)) st.loss = null;   // 다른 데이터의 손실 설정이 남았으면 기본으로
  return st.loss || (L ? L.default.map((t) => Object.assign({}, t)) : []);
}
function lossGroups(L) {                                    // 칩 줄: [{head, allowed}] — 정답이 하나면 한 줄
  return L.heads ? L.heads.map((h) => ({ head: h.name, allowed: h.allowed })).concat([{ head: null, allowed: L.allowed }])
    : [{ head: null, allowed: L.allowed }];
}
function renderLoss() {
  const L = lossInfo();
  if (!$("loss-terms")) return;
  if (!L) { $("loss-terms").innerHTML = ""; return; }
  const terms = lossTerms();
  document.querySelector(".loss-box").classList.toggle("multi", !!L.heads);
  if (L.heads) {
    $("loss-terms").innerHTML = lossGroups(L).map((g) => {
      const mine = terms.map((t, i) => [t, i]).filter(([t]) => (t.head ?? null) === g.head);
      return `<div class="loss-row"><span class="loss-k">${g.head === null ? "공통" : esc(g.head)}</span>` +
        mine.map(([t, i], n) => (n ? `<span class="lplus">+</span>` : "") + lossChip(L, t, i, g.allowed, g.head === null || mine.length > 1)).join("") +
        `<button type="button" class="ladd" data-add="${g.head === null ? "" : esc(g.head)}">+ 항</button></div>`;
    }).join("");
    $("loss-add").hidden = true;
    $("loss-reset").hidden = !st.loss;
    renderRecipes();
    return;
  }
  $("loss-add").hidden = false;
  $("loss-terms").innerHTML = terms.map((t, i) => (i ? `<span class="lplus">+</span>` : "") + lossChip(L, t, i, L.allowed, terms.length > 1)).join("");
  $("loss-reset").hidden = !st.loss;
  renderRecipes();
}
function lossChip(L, t, i, allowed, canDelete) {
  {
    const opts = L.options[t.term] || {};
    const fields = Object.entries(opts).map(([k, [d, lo, hi]]) => k === "node"
      ? `<label class="lopt">블록 <select data-i="${i}" data-k="node">${lossNodes().map(([id, nm]) => `<option value="${esc(id)}"${id === t.node ? " selected" : ""}>${esc(nm)}</option>`).join("")}</select></label>`
      : typeof d === "boolean"
      ? `<label class="lopt"><input type="checkbox" data-i="${i}" data-k="${k}"${t[k] ?? d ? " checked" : ""}> ${LOSS_OPT_LABEL[k] || k}</label>`
      : `<label class="lopt">${LOSS_OPT_LABEL[k] || k} <input type="number" data-i="${i}" data-k="${k}" min="${lo}" max="${hi}" step="0.1" value="${t[k] ?? d}"></label>`).join("");
    return `<span class="loss-term"><input type="number" class="lw" data-i="${i}" data-k="w" min="0" step="any" value="${t.w ?? 1}"> ×` +
      `<select data-i="${i}" data-k="term">${allowed.map((a) => `<option value="${a}"${a === t.term ? " selected" : ""}>${esc(L.names[a])}</option>`).join("")}</select>` +
      fields + (canDelete ? `<button type="button" class="ldel" data-del="${i}" aria-label="빼기">×</button>` : "") + `</span>`;
  }
}
function lossNodes() {                                       // 블록 출력 L1에 고를 블록: 입력·출력 빼고 (그래프 순서)
  const g = st.graph || { nodes: [] };
  return g.nodes.filter((n) => n.type !== "input" && n.type !== "output").map((n) => [n.id, `${blockName(n)} (${n.id})`]);
}
function lastHidden() { const ns = lossNodes(); return ns.length ? ns[ns.length - 1][0] : ""; }
// 조립 예시: 정답 종류마다 자주 쓰는 손실 (고르면 칩이 그대로 바뀜, 이어서 고칠 수 있음)
const LOSS_RECIPES = {
  class: [["드문 클래스에 무게", [{ term: "focal", w: 1, gamma: 2, balance: true }]],
    ["덜 확신하게 (라벨 스무딩)", [{ term: "ce", w: 1, smooth: 0.1, balance: false }]],
    ["가중치를 작게 (L2)", [{ term: "ce", w: 1, smooth: 0, balance: false }, { term: "l2", w: 0.0001 }]]],
  value: [["튀는 값에 덜 흔들리게 (Huber)", [{ term: "huber", w: 1, delta: 1 }]], ["평균 절대 오차 (MAE)", [{ term: "mae", w: 1 }]]],
  seq_class: [["짧은 상태도 놓치지 않게 (Dice)", [{ term: "ce", w: 1, smooth: 0, balance: true }, { term: "dice", w: 0.5 }]]],
};
const ANY_RECIPE = ["블록 출력을 드물게 (희소)", "act_l1"];
function lossKind() { const t = st.summary && st.summary.task; return t === "classification" ? "class" : t === "segmentation" ? "seq_class" : t === "detection" || t === "multi" ? null : "value"; }
function renderRecipes() {
  const k = lossKind(), list = (k && LOSS_RECIPES[k]) || [];
  const opts = list.map(([n], i) => `<option value="${i}">${esc(n)}</option>`).concat(st.graph && lossNodes().length ? [`<option value="act">${esc(ANY_RECIPE[0])}</option>`] : []);
  $("loss-recipe").innerHTML = `<option value="">조립 예시</option>` + opts.join("");
  $("loss-recipe").hidden = !opts.length;
}
$("loss-recipe").onchange = () => {
  const v = $("loss-recipe").value;
  $("loss-recipe").value = "";
  if (v === "") return;
  if (v === "act") return setLoss(lossTerms().filter((t) => t.term !== "act_l1").concat([{ term: "act_l1", w: 0.001, node: lastHidden() }]));
  setLoss(LOSS_RECIPES[lossKind()][Number(v)][1].map((t) => Object.assign({}, t)));
};
function setLoss(terms) {
  const L = lossInfo();
  st.loss = JSON.stringify(terms) === JSON.stringify(L.default) ? null : terms;
  renderLoss();
}
$("loss-terms").addEventListener("change", (e) => {
  const el = e.target, i = Number(el.dataset.i), k = el.dataset.k;
  if (k === undefined) return;
  const terms = lossTerms().map((t) => Object.assign({}, t));
  if (k === "term") terms[i] = Object.assign({ term: el.value, w: terms[i].w ?? 1 }, terms[i].head ? { head: terms[i].head } : {},
    el.value === "act_l1" ? { node: lastHidden() } : {});   // 항을 바꾸면 설정은 그 항의 기본
  else terms[i][k] = el.type === "checkbox" ? el.checked : k === "node" ? el.value : Number(el.value);
  setLoss(terms);
});
$("loss-terms").addEventListener("click", (e) => {
  const d = e.target.dataset.del, add = e.target.dataset.add;
  if (add !== undefined) {                                     // 정답 여러 개: 그 줄에 항 더하기 (공통 줄은 L2·L1)
    const g = lossGroups(lossInfo()).find((x) => (x.head ?? "") === add);
    const used = new Set(lossTerms().filter((t) => (t.head ?? "") === add).map((t) => t.term));
    const next = g.allowed.find((a) => !used.has(a)) || g.allowed[0];
    return setLoss(lossTerms().concat([Object.assign({ term: next, w: next === "l2" || next === "l1" ? 0.0001 : next === "act_l1" ? 0.001 : 1 },
      add ? { head: add } : {}, next === "act_l1" ? { node: lastHidden() } : {})]));
  }
  if (d === undefined) return;
  setLoss(lossTerms().filter((_, i) => i !== Number(d)));
});
$("loss-add").onclick = () => {
  const L = lossInfo();
  if (!L) return;
  const used = new Set(lossTerms().map((t) => t.term)), next = L.allowed.find((a) => !used.has(a)) || L.allowed[0];
  setLoss(lossTerms().concat([Object.assign({ term: next, w: next === "l2" || next === "l1" ? 0.0001 : next === "act_l1" ? 0.001 : 1 },
    next === "act_l1" ? { node: lastHidden() } : {})]));
};
$("loss-reset").onclick = () => setLoss(lossInfo().default.map((t) => Object.assign({}, t)));
function renderLossParts(progress) {                       // 항이 둘 이상이면 마지막 epoch의 항마다 값
  const last = progress[progress.length - 1], terms = last && last.terms;
  $("loss-parts").innerHTML = terms ? kv(Object.entries(terms).map(([k, v]) => [k, Number(v.toPrecision(4))])) : "";
}
function epochsText(r) {                                  // "30" · "21 (조기 종료) · 가장 좋았던 11"
  return `${r.epochs_done}${r.early_stopped ? " (조기 종료)" : ""}` + (r.best_epoch ? ` · 가장 좋았던 ${r.best_epoch}` : "");
}
function fmtLeft(sec) { return sec < 60 ? `${Math.max(1, Math.round(sec))}초` : `${Math.round(sec / 60)}분`; }
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
  csv: (s) => s.forecast ? `CSV: 한 행 = 한 시점인 긴 시계열 (학습 때처럼). '${s.forecast.column}' 열(없으면 마지막 숫자 열)의 끝 ${s.length}개로 그다음 ${s.forecast.horizon}개를 예측합니다.`
    : s.anomaly ? `CSV: 한 행 = 샘플 하나, 값 열 ${s.length}개 — 학습 CSV와 같은 순서. 샘플마다 이상 점수와 판정 (기준값은 ④ 학습 결과에서). 상태 열이 더 있으면 맞았는지 표시합니다.`
    : `CSV: 한 행 = 샘플 하나, 값 열 ${s.image ? s.image[0] * s.image[1] + "개(픽셀)" : s.length + "개"} — 학습 CSV와 같은 순서. 정답 열이 더 있으면 맞았는지 표시합니다.`,
  folder: (s) => s.task === "detection" ? `그림 폴더: 그림 파일(png·jpg·bmp)을 넣은 폴더. 흑백 ${s.image[0]}×${s.image[1]}로 맞추고 그림마다 사물을 찾아 상자로 표시합니다. CSV에는 상자마다 한 줄 (위치는 그림 크기에 대한 0–1 비율).`
    : `그림 폴더: 그림 파일(png·jpg·bmp)을 넣은 폴더. 흑백으로 바꾸고 ${s.image[0]}×${s.image[1]}로 맞춥니다. 클래스 이름 하위 폴더에 나눠 넣으면 그것을 정답으로 봅니다.`,
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
function renderPredictDet(r) {                          // 탐지: 그림마다 찾은 상자 (그림 + 표)
  const s = r.summary, names = st.summary.classes, im = st.summary.image;
  const cards = [["그림", s.n], ["찾은 상자", s.n_boxes]].concat(Object.entries(s.counts).map(([c, n]) => [`'${c}'`, n]));
  if (s.no_box) cards.push(["찾은 것 없는 그림", s.no_box]);
  $("predict-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const gt = (r.pics || []).some((p) => p.truth);
  $("predict-gallery").innerHTML = (r.pics || []).length ? (gt ? detLegend(names) : detLegend(names).replace("점선 = 정답 · ", "")) +
    `<div class="det-gallery">${r.pics.map((p, i) => `<div class="det-card">${detImage(p.values, im[0], im[1], 168, p.truth || null, r.rows[i].boxes, names, s.score)}` +
      `<div class="muted">${esc(p.name)} · ${r.rows[i].boxes.length}개</div></div>`).join("")}</div>` +
    (r.rows.length > r.pics.length ? `<p class="muted">그림은 처음 ${r.pics.length}장만 보입니다. 전부는 아래 표와 CSV로.</p>` : "") : "";
  $("predict-table").innerHTML = `<tr><th>그림</th><th>찾은 상자</th><th>내용 (클래스 점수)</th></tr>` + r.rows.map((x) =>
    `<tr><td>${esc(x.name)}</td><td><b>${x.boxes.length}</b></td><td>${x.boxes.map((b) => `${esc(names[b[0]])} ${Math.round(b[1] * 100)}%`).join(", ") || '<span class="muted">없음</span>'}</td></tr>`).join("");
  $("predict-note").textContent = r.shown < s.n ? `표에는 처음 ${r.shown}개만 보입니다. 전부는 CSV로 내려받으세요.` : "";
  $("predict-result").hidden = false;
}
function renderPredictFcMany(r) {                       // 시계열 예측 (예시 새 데이터): 창마다 지난 값·실제·예측
  const s = r.summary, num = (v) => Number(Number(v).toPrecision(4));
  $("predict-cards").innerHTML = [["창", s.n], ["평균 오차 MAE", num(s.mae)], ["맞힌 값 열", s.column || "-"]]
    .map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  $("predict-gallery").innerHTML = `<p class="muted fc-legend"><span style="color:${FC_COLORS.hist}">━</span> 지난 값 · <span style="color:${FC_COLORS.y}">━</span> 실제 · <span style="color:${FC_COLORS.pred}">━</span> 예측</p>` +
    `<div class="fc-gallery">${r.rows.map((x) => { const hist = x.history.slice(-Math.max(s.horizon * 3, 24));
      return `<div class="fc-card">${svgSeries([{ v: hist, color: FC_COLORS.hist }, { v: x.y, x0: hist.length, color: FC_COLORS.y, width: 1.8 },
        { v: x.pred, x0: hist.length, color: FC_COLORS.pred, width: 1.8 }], 280, 100, hist.length)}<div class="muted">${esc(x.name)} · MAE ${num(x.mae)}</div></div>`; }).join("")}</div>`;
  $("predict-table").innerHTML = `<tr><th>창</th><th>평균 오차 MAE</th></tr>` + r.rows.map((x) => `<tr><td>${esc(x.name)}</td><td>${num(x.mae)}</td></tr>`).join("");
  $("predict-note").textContent = "";
  $("predict-result").hidden = false;
}
function renderPredictFc(r) {                           // 시계열 예측: 지난 값 + 다음 값 그래프와 표
  if (r.summary.many) return renderPredictFcMany(r);
  const s = r.summary, num = (v) => Number(Number(v).toPrecision(4)), hist = s.history.slice(-Math.max(s.pred.length * 3, 24));
  $("predict-cards").innerHTML = [["예측한 칸", s.n], ["맞힌 값 열", s.column || "-"], ["예측 평균", num(s.mean)], ["예측 범위", `${num(s.range[0])} ~ ${num(s.range[1])}`]]
    .map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  $("predict-gallery").innerHTML = svgSeries([{ v: hist, color: FC_COLORS.hist }, { v: s.pred, x0: hist.length, color: FC_COLORS.pred, width: 2 }], 600, 140, hist.length) +
    `<p class="muted fc-legend"><span style="color:${FC_COLORS.hist}">━</span> 파일 끝의 지난 값 · <span style="color:${FC_COLORS.pred}">━</span> 예측한 다음 값</p>`;
  $("predict-table").innerHTML = `<tr><th>몇 칸 뒤</th><th>예측</th></tr>` + r.rows.map((x) => `<tr><td>${esc(x.name)}</td><td><b>${num(x.pred)}</b></td></tr>`).join("");
  $("predict-note").textContent = "";
  $("predict-result").hidden = false;
}
function renderPredictAe(r) {                           // 이상 탐지: 샘플마다 이상 점수·판정
  const s = r.summary, num = (v) => Number(Number(v).toPrecision(3));
  const cards = [["샘플", s.n], ["기준값", s.threshold === null ? "-" : num(s.threshold)]];
  if (s.counts) cards.push(["이상으로 판정", s.counts["이상"]], ["정상으로 판정", s.counts["정상"]]);
  if (s.accuracy !== undefined) cards.push(["판정이 정답과 같음", pct(s.accuracy)]);
  $("predict-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const head = ["샘플", "이상 점수", "판정"].concat(r.has_label ? ["정답", ""] : []);
  $("predict-table").innerHTML = `<tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` + r.rows.map((x) =>
    `<tr class="${x.anomaly ? "bad" : ""}"><td>${esc(x.name)}</td><td>${num(x.score)}</td><td><b>${x.anomaly === null ? "-" : x.anomaly ? "이상" : "정상"}</b></td>` +
    (r.has_label ? `<td>${esc(x.label)}</td><td>${x.ok === null ? "" : x.ok ? "✓" : "✗"}</td>` : "") + `</tr>`).join("");
  $("predict-note").textContent = r.shown < s.n ? `표에는 처음 ${r.shown}개만 보입니다. 전부는 CSV로 내려받으세요.` : "";
  $("predict-result").hidden = false;
}
function renderPredictSeg(r) {                          // 구간 분할: 창마다 예측 구간 (정답이 있으면 함께)
  const s = r.summary;
  const cards = [["창", s.n]].concat(s.accuracy === null ? [] : [["시점 정확도", pct(s.accuracy)]]);
  $("predict-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const head = ["창", "예측 구간"].concat(r.has_label ? ["정답", "맞힌 시점"] : []);
  $("predict-table").innerHTML = `<tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` + r.rows.map((x) =>
    `<tr><td>${esc(x.name)}</td><td class="seg-cell">${segStrip(x.seg_pred, 200)}<div>${esc(x.pred)}</div></td>` +
    (r.has_label ? `<td class="seg-cell">${segStrip(x.seg_truth, 200)}<div>${esc(x.label)}</div></td><td>${pct(x.acc)}</td>` : "") + `</tr>`).join("");
  $("predict-note").innerHTML = segLegend(s.classes);
  $("predict-result").hidden = false;
}
function renderPredictMulti(r) {                        // 정답 여러 개: 샘플마다 정답마다의 예측 (정답이 있으면 함께)
  const s = r.summary, cell = (v) => (typeof v === "number" ? Number(v.toPrecision(4)) : esc(v));
  const cards = [["샘플", s.n]].concat(s.score === null || s.score === undefined ? [] : [["평균 성적 (정답과 비교)", s.score.toFixed(3)]]);
  $("predict-cards").innerHTML = cards.map(([k, v]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const head = ["샘플"].concat(s.heads.map((h) => h.name));
  $("predict-table").innerHTML = `<tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr>` + r.rows.map((x) =>
    `<tr class="${x.ok === false ? "bad" : ""}"><td>${esc(x.name)}</td>` + x.multi_pred.map((v, j) =>
      `<td><b>${cell(v)}</b>${x.multi_label ? `<div class="muted">정답 ${cell(x.multi_label[j])}</div>` : ""}</td>`).join("") + `</tr>`).join("");
  $("predict-note").textContent = r.shown < s.n ? `표에는 처음 ${r.shown}개만 보입니다. 전부는 CSV로 내려받으세요.` : "";
  $("predict-result").hidden = false;
}
function renderPredict(r) {
  if (r.summary.task === "segmentation") return renderPredictSeg(r);
  if (r.summary.task === "multi") { $("predict-gallery").innerHTML = ""; return renderPredictMulti(r); }
  $("predict-gallery").innerHTML = "";
  if (r.summary.task === "detection") return renderPredictDet(r);
  if (r.summary.task === "forecast") return renderPredictFc(r);
  if (r.summary.task === "anomaly") return renderPredictAe(r);
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
$("btn-predict-example").onclick = () => runPredict({ example_new: true });
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
  const n = st.summary ? st.summary.n_test : 0;               // 데이터를 바꾸면 예전 번호가 범위를 넘을 수 있음 → 맞춤
  const i = Math.max(0, Math.min(Number($("sample-index").value) || 0, n - 1));
  $("sample-index").value = i;
  try {
    const a = await api("/api/activations", { sample_index: i });
    const num = (v) => (typeof v === "number" ? Number(v.toPrecision(4)) : v);
    $("sample-info").textContent = `정답 ${num(a.label)} · 예측 ${num(a.prediction)}`;
    const graph = st.trainedGraph || st.graph;              // 값은 학습한 모델 기준
    const im = st.summary.image;
    const inPic = im ? svgImage(a.input, im[0], im[1], Math.max(3, Math.round(96 / im[1])))
      : inputKind() === "tabular"                                // 표 데이터: 열 이름과 값 (크기가 제각각이라 그림보다 표)
      ? `<table class="tab"><tr>${st.summary.features.map((f) => `<th>${esc(f)}</th>`).join("")}</tr>` +
        `<tr>${a.input.map((v) => `<td>${num(v)}</td>`).join("")}</tr></table>` : svgChannels(a.input, st.summary.channels, 600, 60);
    let html = `<div class="fs"><div class="name">입력</div>${inPic}</div>`;
    if (a.saliency && a.saliency.length) {                  // 어디를 봤나: 입력의 어느 값이 예측을 크게 바꾸는가 (빨갈수록)
      const L = st.summary.length, C = st.summary.channels || 1;
      html += `<div class="fs sal"><div class="name">어디를 봤나</div>${attnOverlay(a.saliency, im || C < 2 ? a.input : a.input.slice(0, L))}</div>`;
    }
    for (const [id, b, depth] of orderItems(a.order, graph)) {
      const v = a.values[id];
      if (!v || b.type === "input") continue;
      const at = a.attention && a.attention[id];
      html += `<div class="fs" style="margin-left:${depth * 12}px"><div class="name">${esc(blockName(b))}</div>${valuePic(v, b)}` +
        `${at ? attnPic(at, graph, a.input) : ""}</div>`;
    }
    const ids = Object.keys(a.attention || {});
    if (ids.length) {                                       // 맨 위에 마지막 어텐션 블록의 '어디를 봤나'를 먼저
      const last = orderItems(a.order, graph).map(([id]) => id).filter((id) => ids.includes(id)).pop() || ids[ids.length - 1];
      const f = attnFocus(a.attention[last], graph);
      html = `<div class="fs attn-top"><div class="name">어텐션 열지도 — 마지막 어텐션 블록이 어디를 봤나</div>` +
        `${attnOverlay(f.v, a.input)}<div class="shape">${esc(f.what)}. 빨갈수록 많이 봄. 블록마다의 지도는 아래 각 블록에 있습니다.</div></div>` + html;
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
  if (!WEB) go("task");                                      // 웹은 첫 화면 '내 프로젝트' (projects.js)
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

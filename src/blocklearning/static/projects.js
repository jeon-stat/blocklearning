// BlockLearning 웹 버전: 첫 화면 "내 프로젝트"와 자동 저장.
// 처음 열면 첫 화면에서 프로젝트를 고르거나 새로 시작한다. 작업이 바뀌면(데이터·모델·과제·학습) 잠시 뒤 지금 프로젝트에
// 저장하고, 프로젝트가 없으면 처음 바뀔 때 데이터·과제 이름으로 만든다.
// 저장 위치: 이 브라우저(로그인 없음) 또는 구글 드라이브(로그인 시, 내 드라이브/BlockLearning/<프로젝트>/). store.js 참고.
(function () {
"use strict";
const S = () => window.DLStore;
const WEBB = () => window.DLS_WEB;
const KEY = "dlstudio-current";                       // 마지막으로 연 프로젝트 {kind, id, name} (첫 화면의 '이어서 하기')
const START = "dlstudio-start";                       // 새로 고친 뒤 할 일 (새 프로젝트 → task, 예시로 → quick)
const SAVE_DELAY = 1500;

let cur = null, dirty = false, dataDirty = false, timer = null, saving = null, tab = "local", quietUntil = 0;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function remember() { try { if (cur) localStorage.setItem(KEY, JSON.stringify(cur)); } catch (e) { /* 기억 못 해도 됨 */ } }
function forget() { try { localStorage.removeItem(KEY); } catch (e) { /* 없음 */ } }
function recall() { try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { return null; } }
const store = (kind) => (kind === "drive" ? S().drive : S().local);
const when = (t) => { const d = new Date(t); return isNaN(d) ? "" : d.toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }); };
const defaultName = () => "제목 없는 프로젝트 " + new Date().toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
function autoName() {                                    // 데이터와 과제로 이름 짓기 (예: "심전도 정상·이상 · 시계열 분류 (11. 24.)")
  try {
    let data = "";
    if (st.dataName) data = st.dataName;
    else if (st.dataSrc && st.examples) {
      const e = st.examples.find((x) => x[0] === st.dataSrc);
      data = e ? e[1].split(" · ")[0].replace(/\s*\([^)]*\)\s*$/, "") : st.dataSrc;   // "이름 (분야) · 과제" → 이름
    } else {
      const v = ["csv-path", "folder-path", "series-path"].map((id) => ($(id) || {}).value || "").find((x) => x);
      data = (v || "").replace(/\s*\(파일 \d+개\)$/, "").replace(/\.(csv|tsv|txt)$/i, "");
    }
    const base = [data, st.sub ? st.sub.name : ""].filter(Boolean).join(" · ");
    return base ? `${base} (${new Date().toLocaleDateString("ko-KR", { month: "numeric", day: "numeric" })})` : defaultName();
  } catch (e) { return defaultName(); }
}
const AUTO_SUFFIX = / \(\d{1,2}\. \d{1,2}\.\)( \d+)?$/;   // autoName·uniqueName이 붙이는 꼬리 "(11. 26.)", "(11. 26.) 2"
async function uniqueName(kind, name) {               // 같은 이름이 있으면 뒤에 번호
  try {
    const names = new Set((await store(kind).list()).map((p) => p.name));
    let n = name, k = 2;
    while (names.has(n)) n = name + ' ' + k++;
    return n;
  } catch (e) { return name; }
}
function stepsText() {                                  // 만들기 단계 진행: 과제·데이터·모델·학습(다시 해야 하면 stale)
  const train = st.trained ? (st.trainedJson !== st.graphJson ? "stale" : true) : false;
  return JSON.stringify({ task: !!st.tpl, data: !!st.summary, model: !!(st.checked && st.checked.ok), train });
}
function stepsOf(card) {                                // 카드에 적힌 진행 (예전 기록은 한 줄 요약에서 어림)
  try { if (card.dataset.steps) return JSON.parse(card.dataset.steps); } catch (e) { /* 아래 */ }
  const meta = card.dataset.meta || "";
  return { task: !!meta, data: true, model: !!meta, train: /학습 \d+번/.test(meta) };
}
function paintSteps(p) {                                // 고른 (열지 않은) 프로젝트의 진행을 왼쪽 만들기 흐름에
  const set = (name, done, stale, locked) => {
    const b = document.querySelector(`.step[data-step="${name}"]`);
    if (!b) return;
    b.classList.toggle("done", !!done && !stale);
    b.classList.toggle("stale", !!stale);
    b.classList.toggle("locked", !!locked);
  };
  set("task", p.task);
  set("data", p.data);
  set("model", p.model, false, !p.data);
  set("train", p.train === true, p.train === "stale", !p.data);
  set("predict", false, false, !p.data || !p.train);
}
function metaText() {                                   // 첫 화면 카드의 한 줄 요약: 과제 · 학습 횟수 · 최고 시험 성적
  try {
    const runs = (st.runs || []).filter((r) => r.data_ok);
    const best = runs.reduce((a, r) => (r.results.test_metric !== null && r.results.test_metric !== undefined &&
      (!a || r.results.test_metric > a.results.test_metric) ? r : a), null);
    const name = best && best.results.metric_name;
    const score = !best ? "아직 학습 안 함" : name === "r2" ? `최고 R² ${best.results.test_metric.toFixed(3)}`
      : name === "mean" ? `최고 평균 ${best.results.test_metric.toFixed(3)}`
      : name === "auc" ? `최고 AUC ${best.results.test_metric.toFixed(3)}` : name === "recon" ? "학습함 (정상만)"
      : `최고 시험 ${name === "map50" ? "mAP@0.5" : "정확도"} ${(best.results.test_metric * 100).toFixed(1)}%`;
    return [st.sub ? st.sub.name : "", runs.length ? `학습 ${runs.length}번` : "", score].filter(Boolean).join(" · ");
  } catch (e) { return ""; }
}

// ------------------------------------------------------------ 머리의 프로젝트 단추와 저장 상태
function status(text, bad = false) {
  const el = $("proj-status");
  if (el) {
    el.textContent = text;
    el.classList.toggle("bad", bad);
    el.title = cur ? `작업이 바뀌면 '${cur.name}'에 자동으로 저장합니다 (${store(cur.kind).label})` : "데이터를 불러오면 자동으로 저장을 시작합니다";
  }
}
function showCurrent() {
  const b = $("btn-projects");
  const named = cur && st.step !== "home";                // 홈에서는 '열린 프로젝트'를 따로 보이지 않음
  if (b) b.textContent = named ? cur.name : "내 프로젝트";
  if (b) b.title = named ? `${cur.name} — ${store(cur.kind).label}에 저장 (누르면 첫 화면)` : "첫 화면: 프로젝트 목록";
}

// ------------------------------------------------------------ 자동 저장
function onChange(kind) {
  if (kind === "model" && Date.now() < quietUntil) return;   // 연 직후 화면이 다시 검사하는 것은 바뀜이 아님
  dirty = true;
  if (kind === "data") dataDirty = true;
  clearTimeout(timer);
  timer = setTimeout(saveNow, SAVE_DELAY);
  status("저장 기다리는 중…");
}
async function saveNow() {
  if (saving) { await saving; }
  if (!dirty) return;
  let wantData = false;
  saving = (async () => {
    try {
      if (!WEBB().hasData()) { status(""); dirty = false; return; }     // 데이터가 없으면 저장할 것이 없음
      wantData = dataDirty || !cur;
      dirty = false;
      const kind = cur ? cur.kind : S().drive.available() && S().drive.signedIn() ? "drive" : "local";
      if (cur && wantData && AUTO_SUFFIX.test(cur.name)) {     // 지어 준 이름 그대로면 (사용자가 안 바꿨으면) 데이터가 바뀔 때 이름도 따라
        const next = autoName();
        if (next.replace(AUTO_SUFFIX, "") !== cur.name.replace(AUTO_SUFFIX, "")) {
          const nn = await uniqueName(cur.kind, next);
          await store(cur.kind).rename(cur.id, nn);
          cur.name = nn;
          remember(); showCurrent();
        }
      }
      const name = cur ? cur.name : await uniqueName(kind, autoName());
      const rec = WEBB().recordBytes(name, { dataCopy: wantData });
      if (!rec) { status(""); return; }                // 아직 모델이 없음: 데이터 사본은 다음 저장 때 (표시를 지우지 않음)
      status("저장 중…");
      if (!cur) {                                      // 처음 저장할 수 있을 때 프로젝트를 만듦 (빈 프로젝트가 생기지 않게)
        const p = await store(kind).create(name);
        cur = { kind, id: p.id, name: p.name };
        remember(); showCurrent();
      }
      await store(cur.kind).save(cur.id, Object.assign(wantData ? rec : { project: rec.project }, { meta: metaText(), steps: stepsText() }));
      if (wantData) dataDirty = false;
      status("자동 저장됨 " + new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }));
    } catch (e) {
      dirty = true;
      if (wantData) dataDirty = true;
      if (e && /데이터/.test(String(e.message)) && !cur) { status(""); return; }
      status(e.needLogin ? "저장 안 됨: 구글 로그인 필요" : "저장 안 됨: " + (e.message || e), true);
    }
  })();
  await saving;
  saving = null;
}

// ------------------------------------------------------------ 열기·새로 만들기·가져오기·내보내기
async function openProject(kind, p) {
  if (cur && cur.id === p.id && WEBB().hasData()) return go(st.trained ? "train" : "model");   // 이미 연 프로젝트
  await saveNow();
  status("여는 중…");
  try {
    const rec = await store(kind).load(p.id);
    if (!rec || !rec.project) {
      cur = { kind, id: p.id, name: p.name };
      remember(); showCurrent();
      status("아직 저장된 내용이 없음");
      return go("task");
    }
    const r = await WEBB().openRecord(rec.project, rec.data);
    cur = { kind, id: p.id, name: p.name };
    remember(); showCurrent();
    quietUntil = Date.now() + 3000;
    applyOpened(r);                                           // app.js
    dirty = false; dataDirty = false;
    if (r.notice) onChange("train");                          // 예시 데이터가 바뀌어 학습 결과를 비웠으면 그 상태를 저장 (카드 요약·진행도)
    status("열었음");
    showMessage(`'${p.name}'을(를) 열었음`, r.notice || "", "", true);
  } catch (e) { status("열지 못함", true); showError(e.message !== undefined ? e : { message: String(e) }); }
}
async function startFresh(then) {                        // 새 프로젝트: 지금 작업이 있으면 저장하고 깨끗한 화면으로 (새로 고침)
  await saveNow();
  cur = null;
  if (WEBB().hasData()) {
    try { sessionStorage.setItem(START, then); } catch (e) { /* 없음 */ }
    forget();
    location.reload();
    return;
  }
  showCurrent();
  status("");
  afterStart(then);
}
async function afterStart() {
  go("task");
}
async function importFile() {
  const path = await WEBB().pickProject();
  if (!path) return;
  try {
    await saveNow();
    const r = await WEBB().api("/api/project/open", { path });
    const kind = tab === "drive" && S().drive.signedIn() ? "drive" : "local";
    const name = WEBB().openedName() || path.split("/").pop().replace(/\.(blocklearning|dlstudio)$/i, "");
    const p = await store(kind).create(name);
    cur = { kind, id: p.id, name: p.name };
    remember(); showCurrent();
    applyOpened(r);
    dirty = true; dataDirty = true;
    await saveNow();
    showMessage(`가져왔음: '${name}' (${store(kind).label}에 저장)`, "", "", true);
  } catch (e) { showError(e.message !== undefined ? e : { message: String(e) }); }
}
function exportFile() {
  try {
    const rec = WEBB().recordBytes(cur ? cur.name : "", { withData: true });
    if (!rec) return showMessage("내보낼 것이 없음", "프로젝트를 열거나 데이터와 모델을 먼저 만드세요");
    const name = (cur ? cur.name : "blocklearning-project").replace(/[\\/:*?"<>|]/g, "_") + ".blocklearning";
    WEBB().download(name, rec.project);
    showMessage(`내려받았음: ${name}`, "데이터 사본·학습 기록까지 들어 있어, 다른 PC의 BlockLearning에서 [파일 가져오기]로 그대로 열 수 있습니다", "", true);
  } catch (e) { showError(e); }
}

// ------------------------------------------------------------ 첫 화면 "내 프로젝트"
const svgI = (d) => `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const ICON = {
  blocks: svgI('<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="8.5" y="14" width="7" height="7" rx="1.5"/><path d="M6.5 10v2h11v-2M12 12v2"/>'),
  chip: svgI('<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'),
  chart: svgI('<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/>'),
  code: svgI('<path d="m8 8-4 4 4 4M16 8l4 4-4 4M14 5l-4 14"/>'),
  folder: svgI('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
  pen: svgI('<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>'),
  trash: svgI('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>'),
};
// 첫 화면 그림: 데이터 → 이어 붙인 블록(모델) → 결과. 점이 흘러감 (움직임을 줄이는 설정이면 멈춤)
const HERO_ART = `<svg viewBox="0 0 360 260" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="ha-bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eef2ff"/><stop offset="1" stop-color="#f5f3ff"/></linearGradient>
    <linearGradient id="ha-ring" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#7c3aed"/></linearGradient>
    <pattern id="ha-dots" width="14" height="14" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r="1" fill="#c7d2fe" opacity=".55"/></pattern>
    <filter id="ha-sh" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#312e81" flood-opacity=".10"/></filter>
    <path id="ha-flow" d="M98 130 C118 130 118 58 140 58 L180 58 L180 202 L220 202 C236 202 236 130 252 130"/>
  </defs>
  <rect width="360" height="260" rx="18" fill="url(#ha-bg)"/>
  <rect width="360" height="260" rx="18" fill="url(#ha-dots)"/>

  <!-- 이어 주는 선 -->
  <path d="M98 130 C118 130 118 58 140 58" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-dasharray="4 5"/>
  <path d="M220 202 C236 202 236 130 252 130" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-dasharray="4 5"/>

  <!-- 데이터: 그림 조각 + 신호 -->
  <g filter="url(#ha-sh)">
    <rect x="18" y="66" width="80" height="128" rx="14" fill="#fff"/>
  </g>
  <g transform="translate(31 80)">
    <rect width="11" height="11" rx="2" fill="#c7d2fe"/><rect x="13" width="11" height="11" rx="2" fill="#818cf8"/><rect x="26" width="11" height="11" rx="2" fill="#e0e7ff"/><rect x="39" width="11" height="11" rx="2" fill="#a5b4fc"/>
    <rect y="13" width="11" height="11" rx="2" fill="#818cf8"/><rect x="13" y="13" width="11" height="11" rx="2" fill="#4f46e5"/><rect x="26" y="13" width="11" height="11" rx="2" fill="#a5b4fc"/><rect x="39" y="13" width="11" height="11" rx="2" fill="#e0e7ff"/>
    <rect y="26" width="11" height="11" rx="2" fill="#e0e7ff"/><rect x="13" y="26" width="11" height="11" rx="2" fill="#a5b4fc"/><rect x="26" y="26" width="11" height="11" rx="2" fill="#6366f1"/><rect x="39" y="26" width="11" height="11" rx="2" fill="#c7d2fe"/>
  </g>
  <path d="M30 152 q6 -14 12 0 t12 0 t12 0 t12 -2" fill="none" stroke="#14b8a6" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M30 174 q6 -9 12 0 t12 0 t12 0 t12 0" fill="none" stroke="#f59e0b" stroke-width="2.4" stroke-linecap="round" opacity=".85"/>

  <!-- 흐르는 데이터 (블록 뒤로 지나감) -->
  <g fill="#4f46e5">
    <circle r="3.5" opacity=".9"><animateMotion dur="3.6s" repeatCount="indefinite" calcMode="linear"><mpath href="#ha-flow"/></animateMotion></circle>
    <circle r="3" fill="#14b8a6" opacity=".9"><animateMotion dur="3.6s" begin="-1.2s" repeatCount="indefinite" calcMode="linear"><mpath href="#ha-flow"/></animateMotion></circle>
    <circle r="3" fill="#f97316" opacity=".9"><animateMotion dur="3.6s" begin="-2.4s" repeatCount="indefinite" calcMode="linear"><mpath href="#ha-flow"/></animateMotion></circle>
  </g>

  <!-- 모델: 이어 붙인 블록 -->
  <g filter="url(#ha-sh)">
    <rect x="140" y="40" width="80" height="36" rx="10" fill="#fff"/>
    <rect x="140" y="88" width="80" height="36" rx="10" fill="#fff"/>
    <rect x="140" y="136" width="80" height="36" rx="10" fill="#fff"/>
    <rect x="140" y="184" width="80" height="36" rx="10" fill="#fff"/>
  </g>
  <g>
    <rect x="140" y="40" width="80" height="36" rx="10" fill="#ccfbf1" stroke="#14b8a6" stroke-width="1.5"/>
    <rect x="140" y="88" width="80" height="36" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
    <rect x="140" y="136" width="80" height="36" rx="10" fill="#ede9fe" stroke="#8b5cf6" stroke-width="1.5"/>
    <rect x="140" y="184" width="80" height="36" rx="10" fill="#ffedd5" stroke="#f97316" stroke-width="1.5"/>
    <g fill="#fff" opacity=".9">
      <rect x="152" y="52" width="40" height="5" rx="2.5"/><rect x="152" y="61" width="26" height="5" rx="2.5"/>
      <rect x="152" y="100" width="34" height="5" rx="2.5"/><rect x="152" y="109" width="44" height="5" rx="2.5"/>
      <rect x="152" y="148" width="44" height="5" rx="2.5"/><rect x="152" y="157" width="22" height="5" rx="2.5"/>
      <rect x="152" y="196" width="30" height="5" rx="2.5"/><rect x="152" y="205" width="40" height="5" rx="2.5"/>
    </g>
    <g stroke-width="2" fill="#fff">
      <circle cx="180" cy="82" r="4" stroke="#3b82f6"/><circle cx="180" cy="130" r="4" stroke="#8b5cf6"/><circle cx="180" cy="178" r="4" stroke="#f97316"/>
    </g>
  </g>

  <!-- 결과: 다 됨 + 오르는 곡선 -->
  <g filter="url(#ha-sh)">
    <rect x="252" y="66" width="90" height="128" rx="14" fill="#fff"/>
  </g>
  <circle cx="297" cy="112" r="26" fill="none" stroke="#eef2ff" stroke-width="7"/>
  <circle cx="297" cy="112" r="26" fill="none" stroke="url(#ha-ring)" stroke-width="7" stroke-linecap="round"
          stroke-dasharray="163.4" stroke-dashoffset="18" transform="rotate(-90 297 112)">
    <animate attributeName="stroke-dashoffset" values="163.4;18;18" keyTimes="0;.45;1" dur="5s" repeatCount="indefinite"/>
  </circle>
  <path d="M286 112 l7 7 l13 -14" fill="none" stroke="#4f46e5" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M266 178 C278 176 284 166 294 162 S314 152 328 150" fill="none" stroke="#4f46e5" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M266 178 C278 176 284 166 294 162 S314 152 328 150 L328 182 L266 182 Z" fill="#4f46e5" opacity=".08"/>

</svg>`;
function ensureHome() {
  if ($("step-home")) return;
  const sec = document.createElement("section");
  sec.id = "step-home";
  sec.className = "panel";
  sec.hidden = true;
  sec.innerHTML = `
    <div class="hero">
      <div class="hero-text">
        <div class="eyebrow">설치 없이 · 브라우저 안에서 · 무료</div>
        <h1>블록을 이어 붙여<br>딥러닝 모델을 만드세요</h1>
        <p>데이터를 넣고, 블록으로 모델을 짜고, 바로 학습해 결과를 봅니다. 코드를 몰라도 되고, 데이터는 이 PC 밖으로 나가지 않습니다.</p>
        <div class="hero-cta">
          <button class="primary lg" id="home-new">＋ 새 프로젝트</button>
          <button class="lg" id="home-video">▶ 사용 영상</button>
        </div>
      </div>
      <div class="hero-art" aria-hidden="true">${HERO_ART}</div>
    </div>
    <div class="features">
      <div class="feature"><span class="fi">${ICON.blocks}</span><b>블록으로 모델</b><span>CNN·RNN·ResNet을 끌어다 놓고 선으로 이어 만듭니다</span></div>
      <div class="feature"><span class="fi">${ICON.chip}</span><b>브라우저에서 학습</b><span>설치·서버 없이 이 브라우저 안에서 바로 학습합니다</span></div>
      <div class="feature"><span class="fi">${ICON.chart}</span><b>결과를 쉬운 말로</b><span>정확도·틀린 샘플·학습 기록을 보고 다음 할 일을 알려 줍니다</span></div>
      <div class="feature"><span class="fi">${ICON.code}</span><b>PyTorch 코드로</b><span>만든 모델을 PyTorch·TensorFlow 코드와 가중치로 받습니다</span></div>
    </div>
    <div class="home-projects">
      <div class="sec-head">
        <h2>내 프로젝트</h2>
        <div class="tabs home-tabs"><button class="tab" data-kind="local">이 브라우저</button><button class="tab" data-kind="drive">구글 드라이브</button></div>
        <span class="grow"></span>
        <button id="home-import">파일 가져오기</button>
        <button id="home-export" hidden>지금 프로젝트 내려받기</button>
      </div>
      <div id="home-where" class="proj-where"></div>
      <div id="home-list" class="proj-cards"></div>
    </div>
    <p class="muted use-note">교육·실험용 도구입니다. 학습 결과를 의료 진단이나 제조 판정 같은 실제 결정에 그대로 쓰지 마세요. <a href="privacy.html" target="_blank" rel="noopener">개인정보처리방침</a></p>`;
  document.querySelector("main").prepend(sec);
  try {                                                    // 움직임을 줄이는 설정이면 그림을 멈춤
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      sec.querySelectorAll(".hero-art animate").forEach((a) => a.remove());
      sec.querySelectorAll(".hero-art animateMotion").forEach((a) => a.parentNode.remove());   // 흐르는 점은 뺌
    }
  } catch (e) { /* 없음 */ }
  sec.querySelectorAll(".tab").forEach((b) => (b.onclick = () => { tab = b.dataset.kind; render(); }));
  $("home-new").onclick = () => startFresh("task");
  $("home-video").onclick = openVideo;
  $("home-import").onclick = importFile;
  $("home-export").onclick = exportFile;
  $("home-list").onclick = async (e) => {
    const card = e.target.closest("[data-id]");
    if (!card) return;
    const b = e.target.closest("button[data-a]");
    const kind = tab, item = { id: card.dataset.id, name: card.dataset.name };
    try {
      if (!b && selId !== item.id) return selectCard(item.id);   // 처음 누르면 고르기만, 한 번 더 누르면 열기
      if (!b || b.dataset.a === "open") return openProject(kind, item);
      if (b.dataset.a === "rename") {
        const name = prompt("새 이름", item.name);
        if (!name || name.trim() === item.name) return;
        await store(kind).rename(item.id, name.trim());
        if (cur && cur.id === item.id) { cur.name = name.trim(); remember(); showCurrent(); onChange("model"); }
        return render();
      }
      if (b.dataset.a === "remove") {
        const where = kind === "drive" ? "구글 드라이브 휴지통으로 옮깁니다 (드라이브에서 되살릴 수 있음)" : "이 브라우저에서 지웁니다. 되살릴 수 없습니다";
        if (!confirm(`'${item.name}'을(를) 지울까요? ${where}.`)) return;
        await store(kind).remove(item.id);
        const last = recall();
        if (last && last.id === item.id) forget();
        if (cur && cur.id === item.id) { cur = null; showCurrent(); status(""); }
        return render();
      }
    } catch (err) { showError(err.message !== undefined ? err : { message: String(err) }); }
  };
}
// 사용 영상 창: 누를 때만 받아 재생, 닫으면 멈춤
function openVideo() {
  let d = $("video-dialog");
  if (!d) {
    d = document.createElement("div");
    d.id = "video-dialog";
    d.className = "dialog";
    d.innerHTML = `<div class="dialog-box video-box" role="dialog" aria-label="사용 영상">
      <div class="dialog-head"><b>사용 영상</b><button class="x" aria-label="닫기">×</button></div>
      <video src="src/blocklearning/static/demo.webm" poster="src/blocklearning/static/demo.jpg" controls muted playsinline></video></div>`;
    document.body.appendChild(d);
    const close = () => { d.hidden = true; d.querySelector("video").pause(); };
    d.querySelector(".x").onclick = close;
    d.onclick = (e) => { if (e.target === d) close(); };
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !d.hidden) close(); });
  }
  d.hidden = false;
  d.querySelector("video").play().catch(() => {});
}
async function render() {
  ensureHome();
  document.querySelectorAll("#step-home .tab").forEach((b) => b.classList.toggle("active", b.dataset.kind === tab));
  $("home-export").hidden = !WEBB().hasData();
  const where = $("home-where"), list = $("home-list"), last = recall();
  list.innerHTML = "";
  if (tab === "drive") {
    const dv = S().drive;
    if (!dv.available()) {
      where.innerHTML = `<p class="muted">구글 드라이브 저장은 준비 중입니다. 지금은 '이 브라우저'에 저장됩니다.</p>`;
      return;
    }
    if (!dv.signedIn()) {
      where.innerHTML = `<p>구글 계정으로 로그인하면 <b>내 드라이브/BlockLearning/</b> 아래 프로젝트마다 폴더를 만들어 저장합니다. 다른 PC에서도 같은 계정으로 이어서 할 수 있습니다.</p>
        <p class="muted">권한은 "BlockLearning이 만든 파일만"(drive.file)입니다. 다른 드라이브 파일·메일·계정 정보는 볼 수 없습니다. 데이터 사본이 본인 드라이브에 올라가니, 회사·의료 데이터는 규정을 확인하세요.</p>
        <button class="primary" id="proj-login">구글로 로그인</button> <a href="privacy.html" target="_blank" rel="noopener">개인정보처리방침</a>`;
      $("proj-login").onclick = async () => {
        try { await dv.signIn(""); render(); if (dirty) saveNow(); }
        catch (e) { showError({ message: e.message }); }
      };
      return;
    }
    where.innerHTML = `<p class="muted">구글 드라이브에 연결됨 (내 드라이브/BlockLearning/). 1시간이 지나면 다시 로그인을 묻습니다. <button id="proj-logout">로그아웃</button></p>`;
    $("proj-logout").onclick = () => { dv.signOut(); render(); };
  } else {
    where.innerHTML = `<p class="muted">이 브라우저에 저장됩니다. 사이트 데이터를 지우면 사라지니 [내려받기]나 구글 드라이브로 보관하세요.</p>`;
  }
  list.innerHTML = `<p class="muted">불러오는 중…</p>`;
  try {
    const items = await store(tab).list();
    list.innerHTML = items.length ? items.map((p) => {
      const recent = last && last.id === p.id;
      return `<div class="proj-card${selId === p.id ? " sel" : ""}" data-id="${esc(p.id)}" data-name="${esc(p.name)}" data-meta="${esc(p.meta || "")}" data-steps="${esc(p.steps || "")}">
        <div class="pc-top"><span class="pc-icon">${esc((p.name || "?").trim().charAt(0))}</span>
          ${recent ? '<span class="badge">최근</span>' : ""}</div>
        <div class="pn">${esc(p.name)}</div>
        <div class="pm">${esc(p.meta || "아직 학습 안 함")}</div>
        <div class="pc-foot"><span class="pt">${esc(when(p.updated))} 저장</span>
          <span class="pa"><button data-a="rename" class="ghost icon" title="이름 바꾸기" aria-label="이름 바꾸기">${ICON.pen}</button><button data-a="remove" class="ghost danger icon" title="지우기" aria-label="지우기">${ICON.trash}</button></span></div></div>`;
    }).join("") : `<div class="empty"><span class="fi">${ICON.folder}</span><b>아직 프로젝트가 없습니다</b>
      <span>데이터를 넣으면 자동으로 저장됩니다.</span></div>`;
  } catch (e) { list.innerHTML = `<p class="bad">${esc(e.message || e)}</p>`; }
}
// 프로젝트 카드 고르기: 처음 누르면 고르기만 (테두리), 같은 카드를 한 번 더 누르면 열기, 빈 곳을 누르면 고르기 풀기
let selId = null;
function selectCard(id) {
  selId = id;
  document.querySelectorAll("#home-list .proj-card").forEach((c) => c.classList.toggle("sel", c.dataset.id === id));
  const card = id && document.querySelector(`#home-list .proj-card[data-id="${CSS.escape(id)}"]`);
  if (card && !(cur && cur.id === id)) paintSteps(stepsOf(card));   // 열지 않은 프로젝트: 저장해 둔 진행
  else markSteps();                                                   // 지금 연 프로젝트 (또는 고르지 않음): 지금 상태
  showSteps();
}
// 왼쪽 '만들기' 흐름(과제~예측): 내 프로젝트 화면에서는 프로젝트를 골랐을 때만, 다른 단계에서는 늘
function showSteps() {
  showCurrent();
  if (st.step !== "home") markSteps();                    // 홈에서 고른 카드의 진행으로 칠한 것은 단계로 가면 지금 상태로 되돌림
  const show = st.step !== "home" || !!selId;
  document.querySelectorAll("#steps .nav-label, #steps .step:not(.home-step)").forEach((el) => (el.style.display = show ? "" : "none"));
}
// 고른 프로젝트가 지금 연 것과 다르면, 만들기 단계를 누를 때 그 프로젝트를 먼저 열고 그 단계로
document.addEventListener("click", async (e) => {
  const b = e.target.closest("#steps .step:not(.home-step)");
  if (!b || st.step !== "home" || !selId || (cur && cur.id === selId)) return;
  const card = document.querySelector(`#home-list .proj-card[data-id="${CSS.escape(selId)}"]`);
  if (!card) return;
  e.stopImmediatePropagation();
  e.preventDefault();
  await openProject(tab, { id: card.dataset.id, name: card.dataset.name });
  if (cur && cur.id === card.dataset.id) go(b.dataset.step);
}, true);
document.addEventListener("click", (e) => {
  if (selId && !e.target.closest("#home-list .proj-card")) selectCard(null);
});
function home(fromHistory = false) {
  ensureHome();
  if (cur) tab = cur.kind;
  selId = null;                                             // 홈: 아무것도 고르지 않은 채 (작업하던 프로젝트는 메모리에 남아 다시 열면 바로)
  go("home", fromHistory === true);                       // 단추로 부르면 이벤트가 들어오므로 true일 때만
  render();
}

// ------------------------------------------------------------ 시작
async function init() {
  for (const id of ["btn-open", "btn-save"]) if ($(id)) $(id).hidden = true;       // 웹은 첫 화면 '내 프로젝트'로
  const box = document.createElement("div");                       // 사이드바 위: 지금 프로젝트 (누르면 첫 화면)
  box.className = "proj-switch";
  box.innerHTML = `<div class="ps-label">프로젝트</div><button id="btn-projects" class="proj-btn"></button><span id="proj-status" class="proj-status"></span>`;
  document.querySelector(".brand").after(box);
  const b = $("btn-projects");
  b.onclick = home;
  const nav = $("steps"), hb = document.createElement("button");     // 단계 줄 맨 앞: 첫 화면
  hb.className = "step home-step";
  hb.dataset.step = "home";
  hb.innerHTML = `<span class="n"><svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true"><path d="M3 9.5 10 3.5l7 6V17H12.5v-4.5h-5V17H3z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg></span><span class="sl">내 프로젝트<small>열기·새로 만들기</small></span>`;
  hb.onclick = home;
  nav.prepend(hb);
  document.querySelector(".brand").onclick = home;
  ensureHome();
  showCurrent();
  WEBB().onChange(onChange);
  window.addEventListener("beforeunload", (e) => { if (dirty) { saveNow(); e.preventDefault(); e.returnValue = ""; } });
  let then = "";
  try { then = sessionStorage.getItem(START) || ""; sessionStorage.removeItem(START); } catch (e) { /* 없음 */ }
  if (then) { await WEBB().ready; afterStart(then); return; }
  const job = new URLSearchParams(location.search).get("label");   // 라벨링 페이지 [studio에서 쓰기]: 새 프로젝트로 그 정답을 불러옴
  if (job) {
    history.replaceState(null, "", location.pathname);
    await WEBB().ready;
    for (let i = 0; i < 80 && !(st.catalog && st.catalog.length); i++) await new Promise((r) => setTimeout(r, 250));
    go("data");
    await loadData(Object.assign(dataOptions(), { label_job: job }));
    return;
  }
  const last = recall();
  tab = last ? last.kind : "local";
  home();
}

window.DLProjects = { init, saveNow, current: () => cur, home, startNew: startFresh, showSteps };
})();

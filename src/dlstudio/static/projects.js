// DL studio 웹 버전: 첫 화면 "내 프로젝트"와 자동 저장.
// 처음 열면 첫 화면에서 프로젝트를 고르거나 새로 시작한다. 작업이 바뀌면(데이터·모델·과제·학습) 잠시 뒤 지금 프로젝트에
// 저장하고, 프로젝트가 없으면 처음 바뀔 때 데이터·과제 이름으로 만든다.
// 저장 위치: 이 브라우저(로그인 없음) 또는 구글 드라이브(로그인 시, 내 드라이브/DL studio/<프로젝트>/). store.js 참고.
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
async function uniqueName(kind, name) {               // 같은 이름이 있으면 뒤에 번호
  try {
    const names = new Set((await store(kind).list()).map((p) => p.name));
    let n = name, k = 2;
    while (names.has(n)) n = name + ' ' + k++;
    return n;
  } catch (e) { return name; }
}
function metaText() {                                   // 첫 화면 카드의 한 줄 요약: 과제 · 학습 횟수 · 최고 시험 성적
  try {
    const runs = (st.runs || []).filter((r) => r.data_ok);
    const best = runs.reduce((a, r) => (r.results.test_metric !== null && r.results.test_metric !== undefined &&
      (!a || r.results.test_metric > a.results.test_metric) ? r : a), null);
    const score = best ? (best.results.metric_name === "r2" ? `최고 R² ${best.results.test_metric.toFixed(3)}`
      : `최고 시험 ${best.results.metric_name === "map50" ? "mAP@0.5" : "정확도"} ${(best.results.test_metric * 100).toFixed(1)}%`) : "아직 학습 안 함";
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
  if (b) b.textContent = cur ? cur.name : "내 프로젝트";
  if (b) b.title = cur ? `${cur.name} — ${store(cur.kind).label}에 저장 (누르면 첫 화면)` : "첫 화면: 프로젝트 목록";
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
      const name = cur ? cur.name : await uniqueName(kind, autoName());
      const rec = WEBB().recordBytes(name, { dataCopy: wantData });
      if (!rec) { status(""); return; }                // 아직 모델이 없음: 데이터 사본은 다음 저장 때 (표시를 지우지 않음)
      status("저장 중…");
      if (!cur) {                                      // 처음 저장할 수 있을 때 프로젝트를 만듦 (빈 프로젝트가 생기지 않게)
        const p = await store(kind).create(name);
        cur = { kind, id: p.id, name: p.name };
        remember(); showCurrent();
      }
      await store(cur.kind).save(cur.id, Object.assign(wantData ? rec : { project: rec.project }, { meta: metaText() }));
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
    status("열었음");
    showMessage(`'${p.name}'을(를) 열었음`, "", "", true);
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
async function afterStart(then) {
  go("task");
  if (then !== "quick") return;
  for (let i = 0; i < 80 && !(st.catalog && st.catalog.length); i++) await new Promise((r) => setTimeout(r, 250));   // 과제 목록이 올 때까지
  $("btn-quick").click();
}
async function importFile() {
  const path = await WEBB().pickProject();
  if (!path) return;
  try {
    await saveNow();
    const r = await WEBB().api("/api/project/open", { path });
    const kind = tab === "drive" && S().drive.signedIn() ? "drive" : "local";
    const name = WEBB().openedName() || path.split("/").pop().replace(/\.dlstudio$/i, "");
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
    const name = (cur ? cur.name : "dlstudio-project").replace(/[\\/:*?"<>|]/g, "_") + ".dlstudio";
    WEBB().download(name, rec.project);
    showMessage(`내려받았음: ${name}`, "데이터 사본·학습 기록까지 들어 있어, 다른 PC의 DL studio에서 [파일 가져오기]로 그대로 열 수 있습니다", "", true);
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
// 첫 화면 그림: 블록 그래프 (Input → Conv1D → ReLU → Output, 옆에 학습 곡선)
const HERO_ART = `<svg viewBox="0 0 360 260" width="100%" height="100%">
  <defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eef2ff"/><stop offset="1" stop-color="#f5f3ff"/></linearGradient></defs>
  <rect x="0" y="0" width="360" height="260" rx="18" fill="url(#hg)"/>
  <g font-family="inherit" font-size="11" text-anchor="middle">
    <path d="M100 52v22M100 104v22M100 156v22" stroke="#94a3b8" stroke-width="1.5"/>
    <rect x="40" y="24" width="120" height="30" rx="8" fill="#f1efe8" stroke="#888780"/><text x="100" y="43" fill="#1f2937">Input</text>
    <rect x="40" y="74" width="120" height="30" rx="8" fill="#e1f5ee" stroke="#1d9e75"/><text x="100" y="93" fill="#1f2937">Conv1D · 16</text>
    <circle cx="100" cy="141" r="15" fill="#eeedfe" stroke="#7f77dd"/><text x="100" y="145" fill="#1f2937">ReLU</text>
    <rect x="40" y="178" width="120" height="30" rx="8" fill="#faece7" stroke="#d85a30"/><text x="100" y="197" fill="#1f2937">Output</text>
  </g>
  <rect x="186" y="40" width="150" height="120" rx="12" fill="#fff" stroke="#e5e7eb"/>
  <text x="200" y="62" font-size="11" fill="#64748b">시험 정확도</text>
  <text x="200" y="84" font-size="20" font-weight="700" fill="#4f46e5">93.5%</text>
  <polyline points="200,140 222,128 244,112 266,104 288,97 310,95 322,94" fill="none" stroke="#4f46e5" stroke-width="2.2"/>
  <polyline points="200,100 222,116 244,126 266,132 288,136 310,138 322,139" fill="none" stroke="#f97316" stroke-width="1.6" opacity=".7"/>
  <rect x="186" y="176" width="150" height="40" rx="10" fill="#fff" stroke="#e5e7eb"/>
  <text x="200" y="200" font-size="11" fill="#334155">PyTorch 코드로 내려받기</text>
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
          <button class="lg" id="home-quick">예시로 바로 해 보기 →</button>
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
  sec.querySelectorAll(".tab").forEach((b) => (b.onclick = () => { tab = b.dataset.kind; render(); }));
  $("home-new").onclick = () => startFresh("task");
  $("home-quick").onclick = () => startFresh("quick");
  $("home-import").onclick = importFile;
  $("home-export").onclick = exportFile;
  $("home-list").onclick = async (e) => {
    const card = e.target.closest("[data-id]");
    if (!card) return;
    const b = e.target.closest("button[data-a]");
    const kind = tab, item = { id: card.dataset.id, name: card.dataset.name };
    try {
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
      where.innerHTML = `<p>구글 계정으로 로그인하면 <b>내 드라이브/DL studio/</b> 아래 프로젝트마다 폴더를 만들어 저장합니다. 다른 PC에서도 같은 계정으로 이어서 할 수 있습니다.</p>
        <p class="muted">권한은 "DL studio가 만든 파일만"(drive.file)입니다. 다른 드라이브 파일·메일·계정 정보는 볼 수 없습니다. 데이터 사본이 본인 드라이브에 올라가니, 회사·의료 데이터는 규정을 확인하세요.</p>
        <button class="primary" id="proj-login">구글로 로그인</button> <a href="privacy.html" target="_blank" rel="noopener">개인정보처리방침</a>`;
      $("proj-login").onclick = async () => {
        try { await dv.signIn(""); render(); if (dirty) saveNow(); }
        catch (e) { showError({ message: e.message }); }
      };
      return;
    }
    where.innerHTML = `<p class="muted">구글 드라이브에 연결됨 (내 드라이브/DL studio/). 1시간이 지나면 다시 로그인을 묻습니다. <button id="proj-logout">로그아웃</button></p>`;
    $("proj-logout").onclick = () => { dv.signOut(); render(); };
  } else {
    where.innerHTML = `<p class="muted">이 PC의 이 브라우저 안에 저장합니다 (로그인 없음). 브라우저의 사이트 데이터를 지우면 사라지니, 중요한 것은 [내려받기]나 구글 드라이브로 보관하세요.</p>`;
  }
  list.innerHTML = `<p class="muted">불러오는 중…</p>`;
  try {
    const items = await store(tab).list();
    list.innerHTML = items.length ? items.map((p) => {
      const now = cur && cur.id === p.id, recent = !now && last && last.id === p.id;
      return `<div class="proj-card${now ? " cur" : ""}" data-id="${esc(p.id)}" data-name="${esc(p.name)}" title="눌러서 열기">
        <div class="pc-top"><span class="pc-icon">${esc((p.name || "?").trim().charAt(0))}</span>
          ${now ? '<span class="badge rec">지금 연 프로젝트</span>' : recent ? '<span class="badge">최근</span>' : ""}</div>
        <div class="pn">${esc(p.name)}</div>
        <div class="pm">${esc(p.meta || "아직 학습 안 함")}</div>
        <div class="pc-foot"><span class="pt">${esc(when(p.updated))} 저장</span>
          <span class="pa"><button data-a="rename" class="ghost icon" title="이름 바꾸기" aria-label="이름 바꾸기">${ICON.pen}</button><button data-a="remove" class="ghost danger icon" title="지우기" aria-label="지우기">${ICON.trash}</button></span></div></div>`;
    }).join("") : `<div class="empty"><span class="fi">${ICON.folder}</span><b>아직 프로젝트가 없습니다</b>
      <span>[새 프로젝트]나 [예시로 바로 해 보기]로 시작하세요. 데이터를 넣으면 자동으로 저장됩니다.</span></div>`;
  } catch (e) { list.innerHTML = `<p class="bad">${esc(e.message || e)}</p>`; }
}
function home(fromHistory = false) {
  ensureHome();
  if (cur) tab = cur.kind;
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

window.DLProjects = { init, saveNow, current: () => cur, home, startNew: startFresh };
})();

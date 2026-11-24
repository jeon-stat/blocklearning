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
    if (st.dataSrc && st.examples) {
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
      : `최고 시험 정확도 ${(best.results.test_metric * 100).toFixed(1)}%`) : "아직 학습 안 함";
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
  saving = (async () => {
    try {
      if (!WEBB().hasData()) { status(""); dirty = false; return; }     // 데이터가 없으면 저장할 것이 없음
      if (!cur) {
        const kind = S().drive.available() && S().drive.signedIn() ? "drive" : "local";
        const p = await store(kind).create(await uniqueName(kind, autoName()));
        cur = { kind, id: p.id, name: p.name };
        dataDirty = true;
        remember(); showCurrent();
      }
      const wantData = dataDirty;
      dirty = false; dataDirty = false;
      status("저장 중…");
      const rec = WEBB().recordBytes(cur.name, { dataCopy: wantData });
      if (!rec) { status(""); return; }
      await store(cur.kind).save(cur.id, Object.assign(wantData ? rec : { project: rec.project }, { meta: metaText() }));
      status("자동 저장됨 " + new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }));
    } catch (e) {
      dirty = true;
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
function ensureHome() {
  if ($("step-home")) return;
  const sec = document.createElement("section");
  sec.id = "step-home";
  sec.className = "panel";
  sec.hidden = true;
  sec.innerHTML = `<h2>내 프로젝트</h2>
    <p class="lead">프로젝트마다 데이터·모델·학습 기록이 함께 저장됩니다. 이어서 할 프로젝트를 고르거나 새로 시작하세요.</p>
    <div class="home-start">
      <button class="primary" id="home-new">＋ 새 프로젝트</button>
      <button id="home-quick">예시로 바로 해 보기</button>
      <button id="home-import">파일 가져오기 (.dlstudio)</button>
      <button id="home-export" hidden>지금 프로젝트 내려받기</button>
    </div>
    <div class="tabs home-tabs"><button class="tab" data-kind="local">이 브라우저</button><button class="tab" data-kind="drive">구글 드라이브</button></div>
    <div id="home-where" class="proj-where"></div>
    <div id="home-list" class="proj-cards"></div>
    <p class="muted use-note">교육·실험용 도구입니다. 학습 결과를 의료 진단이나 제조 판정 같은 실제 결정에 그대로 쓰지 마세요.</p>`;
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
        <div class="pn">${esc(p.name)}${now ? ' <span class="badge rec">지금 연 프로젝트</span>' : recent ? ' <span class="badge">최근</span>' : ""}</div>
        <div class="pm">${esc(p.meta || "")}</div>
        <div class="pt">${esc(when(p.updated))} 저장</div>
        <div class="pa"><button data-a="open" class="primary">${now ? "계속하기" : "열기"}</button>
          <button data-a="rename">이름 바꾸기</button><button data-a="remove">지우기</button></div></div>`;
    }).join("") : `<p class="muted">아직 프로젝트가 없습니다. [새 프로젝트]나 [예시로 바로 해 보기]로 시작하세요. 데이터를 넣으면 자동으로 저장됩니다.</p>`;
  } catch (e) { list.innerHTML = `<p class="bad">${esc(e.message || e)}</p>`; }
}
function home() {
  ensureHome();
  if (cur) tab = cur.kind;
  go("home");
  render();
}

// ------------------------------------------------------------ 시작
async function init() {
  const files = document.querySelector(".files");
  for (const id of ["btn-open", "btn-save"]) if ($(id)) $(id).hidden = true;       // 웹은 첫 화면 '내 프로젝트'로
  const b = document.createElement("button");
  b.id = "btn-projects";
  b.className = "proj-btn";
  b.onclick = home;
  const s = document.createElement("span");
  s.id = "proj-status";
  s.className = "proj-status";
  files.prepend(s);
  files.prepend(b);
  const nav = $("steps"), hb = document.createElement("button");     // 단계 줄 맨 앞: 첫 화면
  hb.className = "step home-step";
  hb.dataset.step = "home";
  hb.innerHTML = `<span class="n">⌂</span>내 프로젝트`;
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
  const last = recall();
  tab = last ? last.kind : "local";
  home();
}

window.DLProjects = { init, saveNow, current: () => cur, home, startNew: startFresh };
})();

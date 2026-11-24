// DL studio 웹 버전: "내 프로젝트" 창과 자동 저장.
// 작업이 바뀌면(데이터·모델·과제·학습) 잠시 뒤 지금 프로젝트에 저장한다. 프로젝트가 없으면 처음 바뀔 때 만든다.
// 저장 위치: 이 브라우저(로그인 없음) 또는 구글 드라이브(로그인 시, 내 드라이브/DL studio/<프로젝트>/). store.js 참고.
(function () {
"use strict";
const S = () => window.DLStore;
const WEBB = () => window.DLS_WEB;
const KEY = "dlstudio-current";                       // 지금 프로젝트 {kind, id, name} (이 브라우저에만 기억)
const SAVE_DELAY = 1500;

let cur = null, dirty = false, dataDirty = false, timer = null, saving = null, tab = "local", quietUntil = 0, pendingDrive = false;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function remember() { try { cur ? localStorage.setItem(KEY, JSON.stringify(cur)) : localStorage.removeItem(KEY); } catch (e) { /* 기억 못 해도 됨 */ } }
function recall() { try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { return null; } }
const store = (kind) => (kind === "drive" ? S().drive : S().local);
const when = (t) => { const d = new Date(t); return isNaN(d) ? "" : d.toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }); };
const defaultName = () => "제목 없는 프로젝트 " + new Date().toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
function autoName() {                                    // 데이터와 과제로 이름 짓기 (예: "심전도 정상·이상 · 시계열 분류 (11. 24.)")
  try {
    let data = "";
    if (st.dataSrc && st.examples) {
      const e = st.examples.find((x) => x[0] === st.dataSrc);
      data = e ? e[1].replace(/\s*\([^)]*\)\s*$/, "") : st.dataSrc;
    } else {
      const v = ["csv-path", "folder-path", "series-path"].map((id) => ($(id) || {}).value || "").find((x) => x);
      data = (v || "").replace(/\s*\(파일 \d+개\)$/, "").replace(/\.(csv|tsv|txt)$/i, "");
    }
    const base = [data, st.sub ? st.sub.name : ""].filter(Boolean).join(" · ");
    return base ? `${base} (${new Date().toLocaleDateString("ko-KR", { month: "numeric", day: "numeric" })})` : defaultName();
  } catch (e) { return defaultName(); }
}

// ------------------------------------------------------------ 머리의 "프로젝트" 단추와 저장 상태
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
  if (b) b.title = cur ? `${cur.name} — ${store(cur.kind).label}에 저장` : "프로젝트 목록 · 저장 위치";
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
        const p = await store(kind).create(autoName());
        cur = { kind, id: p.id, name: p.name };
        dataDirty = true;
        remember(); showCurrent();
      }
      const wantData = dataDirty;
      dirty = false; dataDirty = false;
      status("저장 중…");
      const rec = WEBB().recordBytes(cur.name, { dataCopy: wantData });
      if (!rec) { status(""); return; }
      await store(cur.kind).save(cur.id, wantData ? rec : { project: rec.project });
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
  status("여는 중…");
  try {
    const rec = await store(kind).load(p.id);
    cur = { kind, id: p.id, name: p.name };
    remember(); showCurrent();
    if (!rec || !rec.project) { status("아직 저장된 내용이 없음"); return; }
    const r = await WEBB().openRecord(rec.project, rec.data);
    quietUntil = Date.now() + 3000;
    applyOpened(r);                                           // app.js
    dirty = false; dataDirty = false;
    status("열었음");
    closeDialog();
    showMessage(`'${p.name}'을(를) 열었음`, "", "", true);
  } catch (e) { status("열지 못함", true); showError(e.message !== undefined ? e : { message: String(e) }); }
}
async function newProject() {
  const name = prompt("새 프로젝트 이름", autoName());
  if (!name) return;
  await saveNow();
  const kind = tab === "drive" && S().drive.signedIn() ? "drive" : "local";
  const p = await store(kind).create(name.trim());
  cur = { kind, id: p.id, name: p.name };
  remember();
  location.reload();                                          // 빈 화면에서 시작 (데이터를 넣으면 이 프로젝트에 저장)
}
async function importFile() {
  const path = await WEBB().pickProject();
  if (!path) return;
  try {
    const r = await WEBB().api("/api/project/open", { path });
    applyOpened(r);
    const kind = tab === "drive" && S().drive.signedIn() ? "drive" : "local";
    const name = WEBB().openedName() || path.split("/").pop().replace(/\.dlstudio$/i, "");
    const p = await store(kind).create(name);
    cur = { kind, id: p.id, name: p.name };
    remember(); showCurrent();
    dirty = true; dataDirty = true;
    await saveNow();
    closeDialog();
    showMessage(`가져왔음: '${name}' (${store(kind).label}에 저장)`, "", "", true);
  } catch (e) { showError(e.message !== undefined ? e : { message: String(e) }); }
}
function exportFile() {
  try {
    const rec = WEBB().recordBytes(cur ? cur.name : "", { withData: true });
    if (!rec) return showMessage("내보낼 것이 없음", "② 데이터와 ③ 모델을 먼저 만드세요");
    const name = (cur ? cur.name : "dlstudio-project").replace(/[\\/:*?"<>|]/g, "_") + ".dlstudio";
    WEBB().download(name, rec.project);
    showMessage(`내려받았음: ${name}`, "데이터 사본까지 들어 있어, 다른 PC의 DL studio에서 [파일 가져오기]로 그대로 열 수 있습니다", "", true);
  } catch (e) { showError(e); }
}

// ------------------------------------------------------------ "내 프로젝트" 창
function ensureDialog() {
  if ($("proj-dialog")) return;
  const d = document.createElement("div");
  d.id = "proj-dialog";
  d.className = "dialog";
  d.hidden = true;
  d.innerHTML = `<div class="dialog-box proj-box" role="dialog" aria-label="내 프로젝트">
    <div class="dialog-head"><b>내 프로젝트</b>
      <div class="tabs"><button class="tab" data-kind="local">이 브라우저</button><button class="tab" data-kind="drive">구글 드라이브</button></div>
      <button class="x" id="proj-close" aria-label="닫기">×</button></div>
    <div id="proj-where" class="proj-where"></div>
    <div class="row"><button class="primary" id="proj-new">새 프로젝트</button><button id="proj-import">파일 가져오기 (.dlstudio)</button>
      <button id="proj-export">지금 프로젝트 내려받기 (데이터 포함)</button></div>
    <ul id="proj-list" class="proj-list"></ul></div>`;
  document.body.appendChild(d);
  d.onclick = (e) => { if (e.target === d) closeDialog(); };
  $("proj-close").onclick = closeDialog;
  d.querySelectorAll(".tab").forEach((b) => (b.onclick = () => { tab = b.dataset.kind; render(); }));
  $("proj-new").onclick = newProject;
  $("proj-import").onclick = importFile;
  $("proj-export").onclick = exportFile;
  $("proj-list").onclick = async (e) => {
    const b = e.target.closest("button[data-a]");
    if (!b) return;
    const kind = tab, item = { id: b.dataset.id, name: b.dataset.name };
    try {
      if (b.dataset.a === "open") return openProject(kind, item);
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
        if (cur && cur.id === item.id) { cur = null; remember(); showCurrent(); status(""); }
        return render();
      }
    } catch (err) { showError(err.message !== undefined ? err : { message: String(err) }); }
  };
}
function closeDialog() { if ($("proj-dialog")) $("proj-dialog").hidden = true; }
async function render() {
  const d = $("proj-dialog");
  d.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.kind === tab));
  const where = $("proj-where"), list = $("proj-list");
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
        try {
          await dv.signIn("");                                // 처음엔 구글이 동의를 묻고, 그다음부터는 계정만 확인
          if (pendingDrive && cur && cur.kind === "drive") {      // 새로 고친 뒤: 하던 드라이브 프로젝트를 이어서
            pendingDrive = false;
            const p = (await dv.list()).find((x) => x.id === cur.id);
            if (p) return openProject("drive", p);
          }
          render();
          if (dirty) saveNow();
        } catch (e) { showError({ message: e.message }); }
      };
      return;
    }
    where.innerHTML = `<p class="muted">구글 드라이브에 연결됨 (내 드라이브/DL studio/). 1시간이 지나면 다시 로그인을 묻습니다. <button id="proj-logout">로그아웃</button></p>`;
    $("proj-logout").onclick = () => { dv.signOut(); render(); };
  } else {
    where.innerHTML = `<p class="muted">이 PC의 이 브라우저 안에 저장합니다 (로그인 없음). 브라우저의 사이트 데이터를 지우면 사라지니, 중요한 것은 [내려받기]나 구글 드라이브로 보관하세요.</p>`;
  }
  list.innerHTML = `<li class="muted">불러오는 중…</li>`;
  try {
    const items = await store(tab).list();
    list.innerHTML = items.length ? items.map((p) => `<li class="${cur && cur.id === p.id ? "cur" : ""}">
        <span class="pn">${esc(p.name)}${cur && cur.id === p.id ? ' <span class="badge rec">지금</span>' : ""}</span><span class="pt">${esc(when(p.updated))}</span>
        <button data-a="open" data-id="${esc(p.id)}" data-name="${esc(p.name)}">열기</button>
        <button data-a="rename" data-id="${esc(p.id)}" data-name="${esc(p.name)}">이름 바꾸기</button>
        <button data-a="remove" data-id="${esc(p.id)}" data-name="${esc(p.name)}">지우기</button></li>`).join("")
      : `<li class="muted">아직 프로젝트가 없습니다. 데이터를 넣고 모델을 만들면 자동으로 저장됩니다.</li>`;
  } catch (e) { list.innerHTML = `<li class="bad">${esc(e.message || e)}</li>`; }
}
function openDialog() {
  ensureDialog();
  tab = cur ? cur.kind : tab;
  $("proj-dialog").hidden = false;
  render();
}

// ------------------------------------------------------------ 시작
async function init() {
  const files = document.querySelector(".files");
  for (const id of ["btn-open", "btn-save"]) if ($(id)) $(id).hidden = true;       // 웹은 '내 프로젝트' 창으로
  const b = document.createElement("button");
  b.id = "btn-projects";
  b.className = "proj-btn";
  b.onclick = openDialog;
  const s = document.createElement("span");
  s.id = "proj-status";
  s.className = "proj-status";
  files.prepend(s);
  files.prepend(b);
  cur = recall();
  showCurrent();
  WEBB().onChange(onChange);
  window.addEventListener("beforeunload", (e) => { if (dirty) { saveNow(); e.preventDefault(); e.returnValue = ""; } });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("proj-dialog") && !$("proj-dialog").hidden) closeDialog(); });
  if (cur && cur.kind === "local") {                          // 지난번 프로젝트를 이어서
    await WEBB().ready;
    const items = await S().local.list().catch(() => []);
    const p = items.find((x) => x.id === cur.id);
    if (p) await openProject("local", p);
    else { cur = null; remember(); showCurrent(); }
  } else if (cur && cur.kind === "drive") {
    pendingDrive = true;
    status("구글 드라이브 프로젝트 — 눌러서 로그인하면 이어서 엽니다");
    const el = $("proj-status");
    el.classList.add("link");
    el.onclick = () => { el.classList.remove("link"); el.onclick = null; tab = "drive"; openDialog(); };
  }
}

window.DLProjects = { init, saveNow, current: () => cur };
})();

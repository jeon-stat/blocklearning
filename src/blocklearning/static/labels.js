// BlockLearning 웹 버전의 라벨링 페이지: 작업 목록(이 브라우저 보관함)·자동 저장·데이터 고르기·내려받기·studio로 보내기.
// 라벨링 화면과 계산은 label.js(로컬 실행과 같음)가 하고, 여기서는 웹에서만 다른 부분(파일 고르기·저장)을 맡는다.
// 작업 = 정답 JSON(work) + 데이터 사본 zip(data, 처음 한 번). store.js의 DLStore.labels.
(function () {
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ST = () => window.DLStore.labels;
const W = () => window.DLS_WEB;
const LB = () => window.DLLabel;
const when = (t) => { const d = new Date(t); return isNaN(d) ? "" : d.toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }); };
const SCHEMA = { class: "클래스", number: "숫자", segments: "구간", boxes: "도형" };
const svgI = (d) => `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const PEN = svgI('<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>');
const TRASH = svgI('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>');

let cur = null, timer = null, saving = null, dirty = false, needData = false;

// ------------------------------------------------------------ 자동 저장
function status(text, bad = false) {
  const el = $("lab-save");
  if (el) { el.textContent = text; el.classList.toggle("bad", bad); }
}
async function saveNow() {
  if (saving) await saving;
  if (!cur || !dirty) return;
  saving = (async () => {
    try {
      const { work, meta } = W().labelWork();
      const rec = { work, meta };
      if (needData) rec.data = W().labelData();
      dirty = false;
      status("저장 중…");
      await ST().save(cur.id, rec);
      needData = false;
      status("자동 저장됨 " + new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }));
    } catch (e) {
      dirty = true;
      status("저장 안 됨: " + (e.message || e), true);
    }
  })();
  await saving;
  saving = null;
}
function onChange(kind) {
  if (kind !== "label" || !cur) return;
  dirty = true;
  clearTimeout(timer);
  timer = setTimeout(saveNow, 800);
  status("저장 기다리는 중…");
}
async function createRecord(name) {                     // 새 작업·가져온 작업 → 보관함에 새로 (데이터 사본도)
  const p = await ST().create(name);
  cur = { id: p.id, name: p.name };
  dirty = true;
  needData = true;
  await saveNow();
  showName();
}
function showName() { const el = $("lab-job-name"); if (el) el.textContent = cur ? cur.name : ""; }

// ------------------------------------------------------------ 처음 화면: 내 라벨링 작업
function jobLine(m) {
  if (!m) return "";
  const kind = { series_folder: "시계열", image_folder: "이미지", table: "표" }[m.kind] || "";
  return `${kind} · ${SCHEMA[m.schema] || ""} 정답 · ${m.total}개 중 ${m.labeled}개 붙임` + (m.unconfirmed ? ` (확인 안 함 ${m.unconfirmed})` : "");
}
async function render() {
  const list = $("lab-jobs");
  if (!list) return;
  list.innerHTML = `<p class="muted">불러오는 중…</p>`;
  try {
    const items = await ST().list();
    list.innerHTML = items.length ? items.map((p) => {
      const pct = p.meta && p.meta.total ? Math.round((100 * p.meta.labeled) / p.meta.total) : 0;
      return `<div class="proj-card${cur && cur.id === p.id ? " cur" : ""}" data-id="${esc(p.id)}" data-name="${esc(p.name)}" title="눌러서 열기">
        <div class="pc-top"><span class="pc-icon lab-icon">${esc((p.name || "?").trim().charAt(0))}</span>
          ${cur && cur.id === p.id ? '<span class="badge rec">지금 연 작업</span>' : ""}</div>
        <div class="pn">${esc(p.name)}</div>
        <div class="pm">${esc(jobLine(p.meta))}</div>
        <div class="job-bar"><span style="width:${pct}%"></span></div>
        <div class="pc-foot"><span class="pt">${esc(when(p.updated))} 저장</span>
          <span class="pa"><button data-a="use" class="ghost" title="studio에서 이 정답으로 학습 데이터 불러오기">studio에서 쓰기 ↗</button>
          <button data-a="rename" class="ghost icon" title="이름 바꾸기" aria-label="이름 바꾸기">${PEN}</button>
          <button data-a="remove" class="ghost danger icon" title="지우기" aria-label="지우기">${TRASH}</button></span></div></div>`;
    }).join("") : `<div class="empty"><b>아직 라벨링 작업이 없습니다</b><span>아래 [새 작업]에서 데이터를 고르고 정답 종류를 정하면 시작합니다. 붙인 정답은 이 브라우저에 자동으로 저장됩니다.</span></div>`;
  } catch (e) { list.innerHTML = `<p class="err">${esc(e.message || e)}</p>`; }
}
async function openJob(item) {
  await saveNow();
  try {
    status("여는 중…");
    const rec = await ST().load(item.id);
    if (!rec || !rec.work) throw { message: "저장된 내용이 없는 작업", hint: "지우고 새로 만드세요" };
    const st = await W().labelOpen(rec.work, rec.data);
    cur = { id: item.id, name: item.name };
    dirty = false; needData = !rec.data;
    LB().start(st);
    showName();
    status("열었음");
  } catch (e) { status(""); LB().showError(e); }
}
function useInStudio(id) {                              // studio(새 탭)에서 이 작업의 정답으로 학습 데이터
  window.open("./?label=" + encodeURIComponent(id), "_blank");
}

// ------------------------------------------------------------ 화면 붙이기 (label.js 화면을 웹에 맞게)
function setup() {
  document.body.classList.add("web");
  const studio = document.querySelector("a.btn-link");
  if (studio) studio.href = "./";
  // 처음 화면: 작업 목록 + 데이터 고르기 (경로 칸 대신)
  const lead = document.querySelector("#lab-start .lead");
  lead.innerHTML = lead.innerHTML.replace("붙인 정답은 작업 파일에 바로바로 저장되고, 다 붙이면 studio 학습 데이터로 씁니다.",
    "붙인 정답은 이 브라우저에 자동으로 저장되고, studio ② 데이터의 <b>라벨링한 데이터</b>에서 학습 데이터로 불러 씁니다.");
  lead.insertAdjacentHTML("afterend", `<div class="sec-title"><h3>내 라벨링 작업</h3><span class="muted">이 브라우저에 저장됩니다. studio ② 데이터의 '라벨링한 데이터'에서 불러 씁니다</span>
    <span class="grow"></span><button id="lab-import-bundle">작업 묶음 가져오기 (.dllabel.zip)</button></div>
    <div id="lab-jobs" class="proj-cards"></div>`);
  for (const h of document.querySelectorAll("#lab-start h3")) if (h.textContent.trim() === "작업 파일 열기") { h.hidden = true; h.nextElementSibling.hidden = true; }
  const fileLabel = $("lab-file") && $("lab-file").closest("label");
  if (fileLabel) fileLabel.hidden = true;
  $("lab-folder").readOnly = true;
  $("lab-folder").placeholder = "[데이터 고르기]로 고르세요 (브라우저 안에서만 읽음)";
  $("lab-scan").textContent = "데이터 고르기";
  $("lab-scan").onclick = async () => {
    const table = $("lab-kind").value === "table";
    if (await W().pickInto("lab-folder", table ? { accept: ".csv,.tsv,.txt" } : { dir: true })) LB().scan();
  };
  $("lab-kind").addEventListener("change", () => { $("lab-folder").value = ""; delete $("lab-folder").dataset.path; });
  const newBtn = $("lab-new"), origNew = newBtn.onclick;
  newBtn.onclick = async () => {
    await saveNow();
    const before = LB().state();
    await origNew();
    const st = LB().state();
    if (st && st !== before && (!before || st.file !== before.file)) {
      const data = $("lab-folder").value.replace(/\s*\(파일 \d+개\)$/, "").replace(/\.(csv|tsv|txt)$/i, "");
      await createRecord(`${data} · ${SCHEMA[st.schema] || ""} 정답`);
    }
  };
  $("lab-import-bundle").onclick = async () => {
    const files = await W().pick({ accept: ".zip" });
    if (!files.length) return;
    try {
      await saveNow();
      const st = await W().labelOpenBundle(await W().writeFiles(files));
      LB().start(st);
      await createRecord(files[0].name.replace(/\.dllabel\.zip$|\.zip$/i, ""));
      LB().showMessage(`가져왔음: '${cur.name}'`, "이 브라우저에 새 작업으로 저장했습니다", "", true);
    } catch (e) { LB().showError(e); }
  };
  $("lab-jobs").onclick = async (e) => {
    const card = e.target.closest("[data-id]");
    if (!card) return;
    const b = e.target.closest("button[data-a]"), item = { id: card.dataset.id, name: card.dataset.name };
    try {
      if (!b) return openJob(item);
      if (b.dataset.a === "use") { await saveNow(); return useInStudio(item.id); }
      if (b.dataset.a === "rename") {
        const name = prompt("새 이름", item.name);
        if (!name || name.trim() === item.name) return;
        await ST().rename(item.id, name.trim());
        if (cur && cur.id === item.id) { cur.name = name.trim(); showName(); }
        return render();
      }
      if (b.dataset.a === "remove") {
        if (!confirm(`'${item.name}'을(를) 지울까요? 이 브라우저에서 지웁니다. 되살릴 수 없습니다 (필요하면 먼저 [작업 내려받기]).`)) return;
        await ST().remove(item.id);
        if (cur && cur.id === item.id) { cur = null; showName(); status(""); }
        return render();
      }
    } catch (err) { LB().showError(err); }
  };
  // 작업 화면: 파일 경로 대신 고르기·내려받기
  $("lab-use").insertAdjacentHTML("beforebegin", `<button id="lab-bundle" title="정답과 데이터 사본을 한 파일로 (다른 PC·다른 사람이 [작업 묶음 가져오기]로 엶)">작업 내려받기</button>`);
  const out = async (op, msg, extra) => {
    try {
      const r = await W().api("/api/label/" + op, Object.assign({ name: cur ? cur.name : "labels", confirmed_only: $("lab-confirmed-only").checked }, extra || {}));
      LB().showMessage(msg(r), "", "", true);
    } catch (e) { LB().showError(e); }
  };
  $("lab-bundle").onclick = () => out("bundle", () => "작업 묶음을 내려받았음 (정답 + 데이터 사본)");
  $("lab-export-file").hidden = true;
  $("lab-export-go").textContent = "정답 표 CSV 내려받기";
  $("lab-export-go").onclick = () => out("export", (r) => `정답 ${r.rows}개를 CSV로 내려받았음`);
  $("lab-yolo-dir").hidden = true; $("lab-coco-file").hidden = true;
  $("lab-yolo-go").textContent = "YOLO로 내려받기 (.zip)"; $("lab-coco-go").textContent = "COCO로 내려받기 (.json)";
  $("lab-yolo-go").onclick = () => out("export_yolo", (r) => `YOLO로 도형 ${r.boxes}개를 내려받았음 (다각형이 있으면 분할 형식, 점은 COCO로)`);
  $("lab-coco-go").onclick = () => out("export_coco", () => "COCO 형식으로 내려받았음");
  $("lab-export-normal").onclick = () => out("export_normals", (r) => `정상 ${r.rows}개 목록을 내려받았음`);
  $("lab-split-go").onclick = () => out("split", (r) => `작업을 ${r.files.length}개로 나눠 내려받았음 — 각자 라벨링 페이지의 [작업 묶음 가져오기]로 엽니다`,
    { n: Number($("lab-split-n").value) || 2 });
  const splitNote = $("lab-split-go").parentElement.querySelector(".muted");
  if (splitNote) splitNote.textContent = "나눈 묶음(.dllabel.zip)을 각자에게 보내면, 맡은 샘플만 열립니다";
  const wrapPick = (btnId, inputId, accept, label) => {
    const b = $(btnId), orig = b.onclick;
    $(inputId).readOnly = true;
    $(inputId).placeholder = "단추로 고르기";
    if (label) b.textContent = label;
    b.onclick = async () => { if (await W().pickInto(inputId, { accept })) orig(); };
  };
  wrapPick("lab-merge-go", "lab-merge-file", ".zip,.dllabel", "다른 사람 작업 골라 합치기");
  wrapPick("lab-import-go", "lab-import-file", ".csv,.tsv,.txt", "정답 표 골라 가져오기");
  $("lab-use").onclick = async () => {
    if (!cur) return;
    const sc = (LB().state() || {}).schema;
    if (sc === "segments")                                  // studio에 아직 없는 과제: 내보내기로 안내
      return LB().showMessage("구간 정답으로 학습하는 '구간 분할' 과제는 아직 studio에 없습니다", "[내보내기]로 구간 표를 받아 두세요");
    await saveNow();
    useInStudio(cur.id);
    LB().showMessage("studio를 새 탭에서 열어 이 정답으로 학습 데이터를 불러옵니다", "라벨링은 이 탭에서 계속할 수 있습니다", "", true);
  };
  $("lab-switch").addEventListener("click", () => { saveNow(); render(); $("lab-back-name").textContent = cur ? cur.name : ""; });
  // 머리: 작업 이름·저장 상태
  $("lab-progress").insertAdjacentHTML("beforebegin", `<div class="lab-job"><span id="lab-job-name" class="lab-job-name"></span><span id="lab-save" class="proj-status"></span></div>`);
  window.addEventListener("beforeunload", (e) => { if (dirty) { saveNow(); e.preventDefault(); e.returnValue = ""; } });
}

async function init() {
  setup();
  W().onChange(onChange);
  await W().ready;
  render();
}

window.DLLabels = { init, saveNow, current: () => cur };
})();

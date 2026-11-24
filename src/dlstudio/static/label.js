// DL studio 라벨링 작업실: 격자(카드) + 옆 크게 보기 + 단축키 + 규칙·가져오기·내보내기. 계산·저장은 서버(/api/label/*).
// 웹 버전(GitHub Pages)에서는 같은 요청을 브라우저 안(web.js → Pyodide)에서 처리하고, 작업 목록·저장은 labels.js가 맡는다.
(function () {
"use strict";

// ------------------------------------------------------------ 순수 함수 (node 시험)
function rangeSelect(names, anchor, target) {          // 목록에서 anchor..target (순서 무관). 기준이 없으면 target만
  const a = names.indexOf(anchor), b = names.indexOf(target);
  if (b < 0) return [];
  if (a < 0) return [target];
  return names.slice(Math.min(a, b), Math.max(a, b) + 1);
}
function keyAction(e) {                                 // 키 → 동작. 입력칸에 쓰는 중이면 null
  const tag = ((e.target && e.target.tagName) || "").toUpperCase();
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return null;
  const k = e.key, ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && (k === "z" || k === "Z")) return { type: e.shiftKey ? "redo" : "undo" };
  if (ctrl && (k === "y" || k === "Y")) return { type: "redo" };
  if (ctrl && (k === "a" || k === "A")) return { type: "all" };
  if (ctrl) return null;
  if (/^[1-9]$/.test(k)) return { type: "label", index: Number(k) - 1 };
  if (k === "0" || k === "Delete") return { type: "clear" };
  if (k === "Enter") return { type: "confirm" };
  if (k === "ArrowRight" || k === "j" || k === "J") return { type: "next" };
  if (k === "ArrowLeft" || k === "k" || k === "K") return { type: "prev" };
  if (k === "Escape") return { type: "esc" };
  if (k === "c" || k === "C") return { type: "copy_prev" };
  return null;
}
function nextAfter(before, after, cur) {             // 붙인 뒤 다음 샘플: 목록에서 빠졌어도 같은 자리의 다음 것
  const i = before.indexOf(cur);
  if (!after.length || i < 0) return null;
  if (after.includes(cur)) { const j = after.indexOf(cur); return j + 1 < after.length ? after[j + 1] : null; }
  return after[Math.min(i, after.length - 1)];
}
function pxToRows(f0, f1, a, b, nv, nrows) {           // 끈 가로 비율(0–1) → 원래 행 [시작, 끝)
  const lo = Math.min(f0, f1), hi = Math.max(f0, f1), k = nrows / nv;
  const s = Math.round(a + lo * (b - a)), e = Math.round(a + hi * (b - a)) + 1;
  const start = Math.max(0, Math.round(s * k)), end = Math.min(nrows, Math.max(Math.round(e * k), Math.round(s * k) + 1));
  return [start, end];
}
function isDrag(x0, y0, x1, y1) {                       // 몇 픽셀 이상 움직였을 때만 끌기 (눌렀다 떼기·두 번 누르기는 아님)
  return Math.abs(x1 - x0) + Math.abs(y1 - y0) >= 6;
}
function normBox(x0, y0, x1, y1) {                     // 끈 두 점(0–1) → [x, y, w, h] (그림 안으로 자름, 너무 작으면 null)
  const c = (v) => Math.min(1, Math.max(0, v));
  const x = c(Math.min(x0, x1)), y = c(Math.min(y0, y1)), w = c(Math.max(x0, x1)) - x, h = c(Math.max(y0, y1)) - y;
  return w < 0.005 || h < 0.005 ? null : [+x.toFixed(5), +y.toFixed(5), +w.toFixed(5), +h.toFixed(5)];
}
function progressText(c) {
  return `전체 ${c.total} · 정답 ${c.labeled} (사람 ${c.manual} · 규칙 ${c.rule} · 가져옴 ${c.import}` +
    (c.model ? ` · 모델 ${c.model}` : "") + `) · ` +
    `확인 안 함 ${c.unconfirmed} · 정답 없음 ${c.unlabeled}` + (c.error ? ` · 읽기 오류 ${c.error}` : "");
}
const API = { rangeSelect, keyAction, progressText, nextAfter, pxToRows, normBox, isDrag };
if (typeof module !== "undefined") module.exports = API;
if (typeof document === "undefined") return;           // node: 화면 없음

// ------------------------------------------------------------ 공통
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const WEB = !!window.DLS_WEB;
const pathOf = (id) => (($(id).dataset.path || $(id).value) || "").trim();   // 웹: 칸에는 고른 이름, 실제 위치는 data-path
async function api(path, body) {
  if (WEB) return window.DLS_WEB.api(path, body || {});
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({ error: { message: `응답을 읽을 수 없음 (${r.status})` } }));
  if (!r.ok) throw j.error || { message: `오류 ${r.status}` };
  return j;
}
const L = (op, body) => api("/api/label/" + op, body);
function showMessage(msg, hint = "", detail = "", ok = false) {
  $("message").hidden = false;
  $("message").classList.toggle("ok", ok);
  $("msg-text").textContent = msg;
  $("msg-hint").textContent = hint ? "→ " + hint : "";
  $("msg-detail-box").hidden = !detail;
  $("msg-detail").textContent = detail || "";
}
const showError = (e) => showMessage(e.message || String(e), e.hint || "", e.detail || "");
$("msg-close").onclick = () => ($("message").hidden = true);
const fmt = (v) => (typeof v === "number" ? Number(v.toPrecision(4)) : v);
function spark(vals, w, h, color = "#2f6fde") {
  if (!vals || !vals.length) return `<svg width="${w}" height="${h}"></svg>`;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  const pts = vals.map((v, i) => `${(2 + (i / Math.max(1, vals.length - 1)) * (w - 4)).toFixed(1)},${(2 + (h - 4) - ((v - lo) / (hi - lo)) * (h - 4)).toFixed(1)}`);
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="none"><polyline fill="none" stroke="${color}" stroke-width="1.2" points="${pts.join(" ")}"/></svg>`;
}

// ------------------------------------------------------------ 상태
const S = { state: null, items: [], total: 0, page: 0, pages: 1, sel: new Set(), anchor: null, cur: null, detail: null, view: null };
const colorOf = (label) => {                          // 색은 서버가 #rrggbb로 검사하지만 화면에서도 한 번 더
  const c = ((S.state && S.state.classes.find((k) => k.name === label)) || {}).color;
  return /^#[0-9a-fA-F]{6}$/.test(c || "") ? c : "#9aa3af";
};
const BY_NAME = { manual: "사람", rule: "규칙", import: "가져옴", model: "모델 제안" };
function filterBody() {
  return { filter: { label: $("lab-f-label").value || "all", by: $("lab-f-by").value || null,
    unconfirmed: $("lab-f-unconfirmed").checked, error: $("lab-f-error").checked,
    cluster: $("lab-f-cluster").value === "" ? null : Number($("lab-f-cluster").value),
    who: $("lab-f-who").value || null, conflict: $("lab-f-conflict").checked },
    sort: { key: $("lab-sort").value, desc: $("lab-desc").checked }, page: S.page, size: 60 };
}

// 요약값 이름 (데이터 종류마다 다름: 시계열·그림은 고정, 표는 열 이름 그대로)
const STAT_LABEL = { max: "최대", min: "최소", mean: "평균", std: "표준편차", range: "범위 (최대 − 최소)", length: "길이",
  first: "처음 값", last: "마지막 값", argmax_frac: "최대 위치 (0–1)", width: "너비 (픽셀)", height: "높이 (픽셀)" };
const statLabel = (k) => STAT_LABEL[k] || k;
const isNum = () => S.state && S.state.schema === "number";
const isSeg = () => S.state && S.state.schema === "segments";
const isBox = () => S.state && S.state.schema === "boxes";
const isRegion = () => isSeg() || isBox();
function segStrip(segs, n, w, h) {                       // 카드·차트 아래 구간 띠
  if (!segs || !segs.length || !n) return "";
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="lab-strip">` +
    segs.map(([a, b, c]) => `<rect x="${(Number(a) / n) * w}" y="0" width="${Math.max(1, ((Number(b) - Number(a)) / n) * w)}" height="${h}" fill="${colorOf(c)}"><title>${esc(c)} ${Number(a)}–${Number(b)}</title></rect>`).join("") + `</svg>`;
}
const kindOf = () => (S.state ? S.state.source.kind : $("lab-kind").value);
function svgGray(vals, n, px) {                          // 그림 썸네일 (n×n 흑백 0–1)
  if (!vals || vals.length !== n * n) return "";
  let s = `<svg viewBox="0 0 ${n} ${n}" width="${px}" height="${px}" shape-rendering="crispEdges" class="lab-img">`;
  for (let i = 0; i < n * n; i++) { const g = Math.round(255 * Math.min(1, Math.max(0, vals[i]))); s += `<rect x="${i % n}" y="${Math.floor(i / n)}" width="1" height="1" fill="rgb(${g},${g},${g})"/>`; }
  return s + "</svg>";
}
function bars(vals, w, h) {                              // 표 썸네일: 열마다 표준화한 값 막대
  if (!vals || !vals.length) return `<svg width="${w}" height="${h}"></svg>`;
  const bw = (w - 4) / vals.length, mid = h / 2;
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">` + vals.map((v, i) => {
    const hh = Math.min(mid - 2, Math.abs(v) * (mid - 2) / 3);
    return `<rect x="${(2 + i * bw).toFixed(1)}" y="${(v >= 0 ? mid - hh : mid).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${Math.max(0.5, hh).toFixed(1)}" fill="#7f77dd"/>`;
  }).join("") + `<line x1="0" x2="${w}" y1="${mid}" y2="${mid}" stroke="#e3e6eb"/></svg>`;
}
function thumbOf(it, w, h, color) {
  const k = kindOf();
  if (k === "image_folder") return svgGray(it.thumb, 24, h);
  if (k === "table") return bars(it.thumb, w, h);
  return spark(it.thumb, w, h, color);
}
const labelText = (v) => (typeof v === "number" ? String(fmt(v)) : String(v));

// ------------------------------------------------------------ 처음 화면
(function prefill() {                                   // studio에서 ?folder=&column= 으로 넘어옴
  const q = new URLSearchParams(location.search);
  if (q.get("folder")) { $("lab-folder").value = q.get("folder"); scan(q.get("column")); }
})();
function defaultWorkFile(folder) {
  const f = folder.replace(/[\\/]+$/, "");
  return f + "_labels.dllabel";                       // 데이터 폴더 옆
}
function kindChanged() {
  const k = $("lab-kind").value;
  $("lab-folder").placeholder = { series_folder: "C:\\data\\사이클 (샘플마다 CSV 하나)", image_folder: "C:\\data\\사진 (그림 폴더)", table: "C:\\data\\공정.csv (한 행 = 샘플)" }[k];
  $("lab-column-box").hidden = k !== "series_folder";
  $("lab-idcol-box").hidden = k !== "table";
  $("lab-classes-box").hidden = $("lab-schema").value === "number";
  $("lab-new-box").hidden = true;
}
$("lab-kind").onchange = kindChanged;
$("lab-schema").onchange = () => { $("lab-classes-box").hidden = $("lab-schema").value === "number"; };
$("lab-purpose").onchange = () => {                       // 이상 탐지: 클래스는 정상·이상 (더 넣을 수도 있음)
  if ($("lab-purpose").value === "anomaly") { $("lab-classes-input").value = "정상, 이상"; if ($("lab-schema").value === "number") $("lab-schema").value = "class"; }
};
async function scan(column) {
  const path = pathOf("lab-folder");
  if (!path) return showMessage(WEB ? "데이터를 먼저 고르세요" : "경로를 넣으세요", WEB ? "[데이터 고르기]로 폴더(또는 표 CSV)를 고르세요" : "데이터 폴더(또는 표 CSV)의 전체 경로");
  if ($("lab-kind").value !== "series_folder") {         // 그림·표는 만들 때 읽음 (살펴보기는 작업 파일 경로만 채움)
    $("lab-new-box").hidden = false;
    $("lab-scan-info").textContent = $("lab-kind").value === "table" ? "표 CSV: 숫자 열이 요약값이 됩니다" : "그림 폴더: 하위 폴더가 2개 이상이면 그 이름을 처음 정답으로 가져옵니다";
    if (!$("lab-file").value) $("lab-file").value = defaultWorkFile(path.replace(/\.csv$/i, ""));
    return;
  }
  try {
    const s = await api("/api/data/scan_folder", { path });
    $("lab-new-box").hidden = false;
    const [lo, med, hi] = s.lengths;
    $("lab-scan-info").innerHTML = `CSV <b>${s.n_files}</b>개 · 길이 <b>${lo}–${hi}</b> (중앙값 ${med})` +
      (s.classes ? ` · 하위 폴더 정답 (${s.classes.map(esc).join(", ")}): 처음 정답으로 가져옴 (확인 안 함)` : "") +
      (s.skipped.length ? `<div class="muted">${s.skipped.map(esc).join(" · ")}</div>` : "");
    $("lab-column").innerHTML = s.columns.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    $("lab-column").value = column && s.columns.includes(column) ? column : s.default_column;
    if (s.classes) $("lab-classes-input").value = s.classes.join(", ");
    if (!$("lab-file").value) $("lab-file").value = defaultWorkFile(path);
  } catch (e) { showError(e); }
}
$("lab-scan").onclick = () => scan();
$("lab-folder").oninput = () => { $("lab-new-box").hidden = true; $("lab-file").value = ""; delete $("lab-folder").dataset.path; };
$("lab-new").onclick = async () => {
  const classes = $("lab-classes-input").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean);
  try {
    const st = await L("new", { path: pathOf("lab-folder"), column: $("lab-column").value, classes, file: $("lab-file").value.trim(),
      kind: $("lab-kind").value, schema: $("lab-schema").value, id_col: $("lab-idcol").value.trim() || null, purpose: $("lab-purpose").value });
    start(st);
  } catch (e) { showError(e); }
};
$("lab-open").onclick = async () => {
  try { start(await L("open", { file: $("lab-open-file").value.trim() })); } catch (e) { showError(e); }
};
function start(st) {
  let saved = "";
  try { saved = localStorage.getItem("dlstudio-labeler") || ""; } catch (e) { /* 없음 */ }
  if (!st.who && saved) setWho(saved);                    // 이 브라우저에서 쓰던 작업자 이름
  $("lab-start").hidden = true;
  $("lab-work").hidden = false;
  $("lab-switch").hidden = false;
  S.page = 0; S.sel.clear(); S.cur = null;
  setState(st);
  if (st.missing) showMessage(`작업 파일에 정답이 있지만 폴더에서 사라진 샘플 ${st.missing}개`, "그 정답은 작업 파일에 남아 있고 내보내기에서는 빠집니다");
  loadItems();
}

// ------------------------------------------------------------ 상태 표시 (진행 막대, 클래스)
function setState(st) {
  S.state = st;
  const c = st.counts;
  const pct = (n) => (c.total ? (100 * n) / c.total : 0);
  $("lab-progress").innerHTML = `<div class="lab-bar"><span style="width:${pct(c.labeled - c.unconfirmed)}%" class="done"></span>` +
    `<span style="width:${pct(c.unconfirmed)}%" class="unconf"></span></div><div class="lab-bar-text">${esc(progressText(c))}</div>`;
  $("lab-title").textContent = `${st.source.path.split(/[\\/]/).pop()}` + (st.source.kind === "series_folder" ? ` · 값 열 ${st.source.column}` : "") +
    (st.schema === "number" ? " · 숫자 정답" : "");
  $("lab-numbar").hidden = st.schema !== "number";
  $("lab-who-show").textContent = st.who ? `작업자: ${st.who}` : "작업자 이름 없음";
  if (st.assigned != null) $("lab-title").textContent += ` · 맡은 샘플 ${st.assigned}개`;
  const kw = $("lab-f-who").value;
  $("lab-f-who").innerHTML = `<option value="">전부</option>` + (st.labelers || []).map((w) => `<option value="${esc(w)}">${esc(w)}</option>`).join("");
  $("lab-f-who").value = (st.labelers || []).includes(kw) ? kw : "";
  $("lab-rule-label-box").hidden = st.schema === "number";
  $("lab-rule-num-box").hidden = st.schema !== "number";
  $("lab-suggest").hidden = st.schema !== "class";
  $("lab-normal-cand").hidden = st.purpose !== "anomaly";
  $("lab-export-normal").hidden = st.purpose !== "anomaly" || st.schema !== "class";
  $("lab-tool-rule").hidden = st.schema === "segments" || st.schema === "boxes";
  $("lab-tool-import").hidden = st.schema === "segments" || st.schema === "boxes";
  $("lab-export-det").hidden = st.schema !== "boxes";
  if (st.schema === "boxes") {
    if (!$("lab-yolo-dir").value) $("lab-yolo-dir").value = st.file.replace(/\.dllabel$/, "") + "_yolo";
    if (!$("lab-coco-file").value) $("lab-coco-file").value = st.file.replace(/\.dllabel$/, "") + "_coco.json";
    if (S.brush === undefined) S.brush = 0;
  }
  $("lab-tool-column").hidden = st.schema !== "segments";
  if (st.schema === "segments" && S.brush === undefined) S.brush = st.classes.length ? 0 : null;
  const sk = $("lab-sort").value;                       // 정렬: 이름, (모델), (정답 값), 요약값들
  $("lab-sort").innerHTML = `<option value="name">이름</option>` + (st.schema === "class" ? `<option value="uncertainty">헷갈리는 것부터 (모델)</option>` : `<option value="label">정답 값</option>`) +
    st.stat_names.map((k) => `<option value="${esc(k)}">${esc(statLabel(k))}</option>`).join("");
  $("lab-sort").value = [...$("lab-sort").options].some((o) => o.value === sk) ? sk : "name";
  $("lab-undo").disabled = !st.can_undo;
  $("lab-redo").disabled = !st.can_redo;
  $("lab-classes").hidden = st.schema === "number";
  $("lab-classes").innerHTML = st.classes.map((k, i) => `<button class="lab-chip" data-i="${i}" style="--c:${colorOf(k.name)}" title="단축키 ${i + 1}">` +
    `<span class="key">${i + 1}</span>${esc(k.name)} <span class="n">${k.count}</span></button>`).join("") +
    (st.classes.length < 9 ? `<button id="lab-class-add" class="lab-chip add">+ 클래스</button>` : "") +
    `<span class="muted lab-class-help">클래스를 두 번 누르면 이름 바꾸기</span>`;
  const keep = $("lab-f-label").value;
  $("lab-f-label").innerHTML = `<option value="all">전부</option><option value="none">정답 없음</option><option value="any">정답 있음</option>` +
    st.classes.map((k) => `<option value="cls:${esc(k.name)}">${esc(k.name)}</option>`).join("");
  $("lab-f-label").value = [...$("lab-f-label").options].some((o) => o.value === keep) ? keep : "all";
  const rl = $("lab-rule-label").value;
  $("lab-rule-label").innerHTML = st.classes.map((k) => `<option value="${esc(k.name)}">${esc(k.name)}</option>`).join("");
  if (rl) $("lab-rule-label").value = rl;
  if (!$("lab-export-file").value) $("lab-export-file").value = st.file.replace(/\.dllabel$/, "") + "_정답.csv";
}
$("lab-classes").onclick = async (e) => {
  if (e.target.closest("#lab-class-add")) {
    const name = prompt("새 클래스 이름");
    if (name) try { setState(await L("classes", { op: "add", name })); } catch (err) { showError(err); }
    return;
  }
  const b = e.target.closest(".lab-chip[data-i]");
  if (!b) return;
  if (isRegion()) { S.brush = Number(b.dataset.i); if (S.detail) renderDetail(); return; }   // 구간·상자: 붓 고르기
  applyLabel(Number(b.dataset.i));
};
$("lab-classes").ondblclick = async (e) => {
  const b = e.target.closest(".lab-chip[data-i]");
  if (!b) return;
  const old = S.state.classes[Number(b.dataset.i)].name, nw = prompt(`'${old}'의 새 이름 (붙은 정답도 함께 바뀜)`, old);
  if (nw && nw !== old) try { setState(await L("classes", { op: "rename", name: old, new: nw })); loadItems(); } catch (err) { showError(err); }
};

// ------------------------------------------------------------ 격자
async function loadItems() {
  try {
    const r = await L("items", filterBody());
    Object.assign(S, { items: r.items, total: r.total, page: r.page, pages: r.pages });
    const shown = new Set(r.items.map((x) => x.name));       // 안 보이는 샘플은 고른 것에서 뺌 (숫자키가 숨은 샘플에 붙지 않게)
    S.sel = new Set([...S.sel].filter((n) => shown.has(n)));
    if (S.anchor && !shown.has(S.anchor)) S.anchor = null;
    renderGrid();
  } catch (e) { showError(e); }
}
function renderGrid() {
  $("lab-grid").innerHTML = S.items.length ? S.items.map((it) => {
    const has = (it.label !== null && it.label !== undefined) || (it.segments && it.segments.length > 0) || (it.boxes && it.boxes.length > 0);
    const chip = isBox() ? (has ? `<span class="lab-tag${it.confirmed ? "" : " unconf"}" style="--c:#4b5563">상자 ${it.boxes.length}</span>` : `<span class="lab-tag none">상자 없음</span>`) : isSeg() ? (has ? `<span class="lab-tag${it.confirmed ? "" : " unconf"}" style="--c:#4b5563">구간 ${it.segments.length}</span>` : `<span class="lab-tag none">구간 없음</span>`) : has ? `<span class="lab-tag${it.confirmed ? "" : " unconf"}" style="--c:${isNum() ? "#2f6fde" : colorOf(it.label)}">${esc(labelText(it.label))}</span>` : `<span class="lab-tag none">정답 없음</span>`;
    return `<div class="lab-card${S.sel.has(it.name) ? " sel" : ""}${S.cur === it.name ? " cur" : ""}${it.error ? " bad" : ""}" data-n="${esc(it.name)}">` +
      `<div class="lab-name" title="${esc(it.name)}">${esc(it.name)}</div>` +
      (it.error ? `<div class="lab-err">${esc(it.error)}</div>` : `<div class="lab-thumb">${thumbOf(it, 150, 46, has && !isNum() && !isSeg() ? colorOf(it.label) : "#6b7280")}</div>`) +
      (isSeg() && it.stats ? segStrip(it.segments, it.stats.length, 150, 6) : "") +
      `<div class="lab-meta">${chip}${it.p != null ? `<span class="lab-p" title="모델 확신도">${Math.round(it.p * 100)}%</span>` : ""}` +
      `${it.note ? `<span class="lab-note" title="${esc(it.note)}">✎</span>` : ""}${it.conflict ? `<span class="lab-conf" title="다른 작업자와 정답이 다름">⚠</span>` : ""}</div></div>`;
  }).join("") : `<p class="muted">이 조건에 맞는 샘플이 없습니다.</p>`;
  $("lab-page").textContent = `${S.total}개 · ${S.page + 1} / ${S.pages} 쪽` + (S.sel.size > 1 ? ` · ${S.sel.size}개 고름` : "");
}
const names = () => S.items.map((x) => x.name);
$("lab-grid").onclick = (e) => {
  const c = e.target.closest(".lab-card");
  if (!c) return;
  const n = c.dataset.n;
  if (e.shiftKey && S.anchor) { S.sel = new Set(rangeSelect(names(), S.anchor, n)); }
  else if (e.ctrlKey || e.metaKey) { S.sel.has(n) ? S.sel.delete(n) : S.sel.add(n); S.anchor = n; }
  else { S.sel = new Set([n]); S.anchor = n; }
  S.cur = n;
  renderGrid();
  loadDetail(n);
};
for (const id of ["lab-f-label", "lab-f-by", "lab-f-unconfirmed", "lab-f-error", "lab-sort", "lab-desc", "lab-f-who", "lab-f-conflict"]) $(id).onchange = () => { S.page = 0; loadItems(); };
$("lab-prev-page").onclick = () => { if (S.page > 0) { S.page--; loadItems(); } };
$("lab-next-page").onclick = () => { if (S.page < S.pages - 1) { S.page++; loadItems(); } };

// ------------------------------------------------------------ 크게 보기 (휠 확대, 끌어서 이동, 두 번 클릭 원래대로)
async function loadDetail(n) {
  try {
    S.detail = await L("item", { name: n });
    S.boxes = (S.detail.boxes || []).map((b) => b.slice());
    S.boxSel = null;
    S.view = [0, Math.max(1, S.detail.values.length - 1)];
    renderDetail();
  } catch (e) { showError(e); }
}
function boxOverlay() {                                  // 그림 위 상자 (0–1 비율을 viewBox 1000으로)
  const bx = S.boxes || [];
  return bx.map(([x, y, w, h, k], i) => {
    const c = colorOf(k), sel = S.boxSel === i;
    return `<g><rect data-i="${i}" x="${x * 1000}" y="${y * 1000}" width="${w * 1000}" height="${h * 1000}" fill="${c}" fill-opacity="${sel ? 0.25 : 0.12}" stroke="${c}" stroke-width="${sel ? 3 : 2}" vector-effect="non-scaling-stroke"/>` +
      `<text x="${x * 1000 + 6}" y="${y * 1000 + 28}" font-size="26" fill="${c}" style="pointer-events:none">${esc(k)}</text>` +
      (sel ? `<rect data-h="br" x="${(x + w) * 1000 - 14}" y="${(y + h) * 1000 - 14}" width="28" height="28" fill="#fff" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke"/>` : "") + `</g>`;
  }).join("") + (S.boxDraft ? `<rect x="${S.boxDraft[0] * 1000}" y="${S.boxDraft[1] * 1000}" width="${S.boxDraft[2] * 1000}" height="${S.boxDraft[3] * 1000}" fill="none" stroke="#2f6fde" stroke-dasharray="6 4" stroke-width="2" vector-effect="non-scaling-stroke"/>` : "");
}
const imgSrc = (name) => (WEB ? `data-lname="${esc(name)}"` : `src="/api/label/file?name=${encodeURIComponent(name)}"`);
function fillImages() {                                  // 웹: 그림 원본을 브라우저 안에서 읽어 넣음
  if (!WEB) return;
  document.querySelectorAll("img[data-lname]").forEach((im) => {
    window.DLS_WEB.labelFileUrl(im.dataset.lname).then((u) => { im.src = u; }).catch(() => {});
  });
}
function bigView() {                                     // 데이터 종류마다 크게 보기
  const d = S.detail;
  if (d.image && isBox()) return `<div class="lab-boxwrap"><img class="lab-boximg${(d.width || 0) < 256 ? " px" : ""}" ${imgSrc(d.name)} alt="${esc(d.name)}" draggable="false">` +
    `<svg id="lab-boxsvg" viewBox="0 0 1000 1000" preserveAspectRatio="none">${boxOverlay()}</svg></div>` +
    `<div class="muted lab-help">끌어서 상자 그리기 (붓 = 숫자키) · 상자 눌러 고르기 → 끌어 옮기기·오른쪽 아래 끌어 크기 · 숫자키로 클래스 바꾸기 · Delete 지우기 · C 이전 그림 상자 복사 · ${d.width}×${d.height}</div>`;
  if (d.image) return `<img class="lab-bigimg${(d.width || 0) < 256 ? " px" : ""}" ${imgSrc(d.name)} alt="${esc(d.name)}">` +
    `<div class="muted">${d.width}×${d.height} 픽셀</div>`;
  if (d.row) return `<div class="kv lab-row">` + Object.entries(d.row).map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("") +
    `</div><div class="muted">원본 ${d.line}행</div>`;
  return bigChart() + (isSeg() ? `<div class="muted lab-help">끌어서 구간 칠하기 (붓 = 숫자키 1–9, 0 = 지우개) · Shift+끌기 이동 · 휠 확대 · 두 번 눌러 원래 크기</div>`
    : `<div class="muted lab-help">휠로 확대 · 끌어서 이동 · 두 번 눌러 원래 크기</div>`);
}
function bigChart() {
  const v = S.detail.values, W = 400, H = 220, P = 30;
  if (!v.length) return `<p class="err">${esc(S.detail.error || "값이 없음")}</p>`;
  const [a, b] = S.view, part = v.slice(Math.floor(a), Math.ceil(b) + 1);
  let lo = Math.min(...part), hi = Math.max(...part);
  if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
  const x = (i) => P + ((i - a) / Math.max(1e-9, b - a)) * (W - P - 6), y = (t) => 6 + (H - 26) - ((t - lo) / (hi - lo)) * (H - 26);
  const pts = part.map((t, k) => `${x(Math.floor(a) + k).toFixed(1)},${y(t).toFixed(1)}`).join(" ");
  const col = S.detail.label ? colorOf(S.detail.label) : "#2f6fde";
  const scale = S.detail.n / Math.max(1, v.length);       // 줄였으면 원래 행 번호로
  return `<svg id="lab-big" viewBox="0 0 ${W} ${H}" width="100%" class="lab-big"><rect x="${P}" y="6" width="${W - P - 6}" height="${H - 26}" fill="#fbfbfc" stroke="#e3e6eb"/>` +
    `<polyline fill="none" stroke="${col}" stroke-width="1.4" points="${pts}"/>` +
    `<text x="${P - 4}" y="12" font-size="10" text-anchor="end" fill="#6b7280">${fmt(hi)}</text><text x="${P - 4}" y="${H - 20}" font-size="10" text-anchor="end" fill="#6b7280">${fmt(lo)}</text>` +
    (isSeg() ? (S.detail.segments || []).map(([s0, s1, c]) => {     // 구간 띠 (원래 행 → 보이는 값 위치)
      const xa = Math.max(P, x(s0 / scale)), xb = Math.min(W - 6, x(s1 / scale));
      return xb > xa ? `<rect x="${xa.toFixed(1)}" y="6" width="${(xb - xa).toFixed(1)}" height="${H - 26}" fill="${colorOf(c)}" fill-opacity=".18"/>` +
        `<rect x="${xa.toFixed(1)}" y="${H - 22}" width="${(xb - xa).toFixed(1)}" height="5" fill="${colorOf(c)}"/>` : "";
    }).join("") : "") +
    (S.paintSel ? `<rect x="${Math.min(S.paintSel[0], S.paintSel[1])}" y="6" width="${Math.abs(S.paintSel[1] - S.paintSel[0])}" height="${H - 26}" fill="${S.brush == null ? "#999" : colorOf(S.state.classes[S.brush].name)}" fill-opacity=".35"/>` : "") +
    `<text x="${P}" y="${H - 6}" font-size="10" fill="#6b7280">행 ${Math.round(a * scale) + 1}</text><text x="${W - 6}" y="${H - 6}" font-size="10" text-anchor="end" fill="#6b7280">${Math.round(b * scale) + 1}</text></svg>`;
}
function renderDetail() {
  const d = S.detail;
  if (!d) return;
  const st = d.stats ? Object.entries(d.stats).map(([k, v]) => `<div class="k">${esc(statLabel(k))}</div><div class="v">${fmt(v)}</div>`).join("") : "";
  const has = (d.label !== null && d.label !== undefined) || (d.segments && d.segments.length > 0);
  const btns = S.state.classes.map((k, i) => `<button class="lab-chip${d.label === k.name ? " on" : ""}" data-i="${i}" style="--c:${colorOf(k.name)}"><span class="key">${i + 1}</span>${esc(k.name)}</button>`).join("");
  $("lab-detail").innerHTML = `<div class="row lab-dhead"><button id="lab-d-prev">◀</button><b class="lab-dname">${esc(d.name)}</b><button id="lab-d-next">▶</button></div>` +
    bigView() +
    (isBox() ? `<div class="lab-dlabels"><span class="muted">붓:</span>${S.state.classes.map((k, i) => `<button class="lab-chip${S.brush === i ? " on" : ""}" data-brush="${i}" style="--c:${colorOf(k.name)}"><span class="key">${i + 1}</span>${esc(k.name)}</button>`).join("")}</div>` +
      `<div class="lab-seglist">${(S.boxes || []).map(([x, y, w, h, k], i) => `<span class="lab-tag${S.boxSel === i ? " sel" : ""}" style="--c:${colorOf(k)}">${i + 1}. ${esc(k)}</span>`).join(" ") || '<span class="muted">상자 없음</span>'}</div>`
    : isSeg() ? `<div class="lab-dlabels"><span class="muted">붓:</span>${S.state.classes.map((k, i) => `<button class="lab-chip${S.brush === i ? " on" : ""}" data-brush="${i}" style="--c:${colorOf(k.name)}"><span class="key">${i + 1}</span>${esc(k.name)}</button>`).join("")}` +
      `<button class="lab-chip${S.brush == null ? " on" : ""}" data-brush="">0 지우개</button>` + (has && !d.confirmed ? `<button id="lab-d-confirm" class="lab-chip">Enter 확인</button>` : "") + `</div>` +
      `<div class="lab-seglist">${(d.segments || []).map(([a, b, c]) => `<span class="lab-tag" style="--c:${colorOf(c)}">${esc(c)} ${Number(a)}–${Number(b)}</span>`).join(" ") || '<span class="muted">구간 없음</span>'}</div>`
    : isNum() ? `<div class="row lab-dlabels"><input id="lab-d-num" type="number" step="any" value="${has ? esc(d.label) : ""}" placeholder="숫자 정답">` +
      `<button id="lab-d-num-go" class="primary">넣기 (Enter)</button><button id="lab-d-clear" class="lab-chip">지우기</button>` +
      (has && !d.confirmed ? `<button id="lab-d-confirm" class="lab-chip">확인</button>` : "") + `</div>`
      : `<div class="lab-dlabels">${btns}<button id="lab-d-clear" class="lab-chip">0 지우기</button>` +
      (has && !d.confirmed ? `<button id="lab-d-confirm" class="lab-chip">Enter 확인</button>` : "") + `</div>`) +
    `<div class="muted">정답: ${isBox() ? ((S.boxes || []).length ? `상자 ${S.boxes.length}개` : "상자 없음") : isSeg() ? ((d.segments || []).length ? `구간 ${d.segments.length}개 (${BY_NAME[d.by] || "-"}${d.confirmed ? ", 확인됨" : ", 확인 안 함"})` : "구간 없음")
      : has ? `${esc(labelText(d.label))} (${BY_NAME[d.by] || "-"}${d.confirmed ? ", 확인됨" : ", 확인 안 함"})` : "없음"}</div>` +
    (d.conflict ? `<div class="lab-conflict">⚠ 충돌: ${esc(d.conflict.who || "다른 작업자")}의 정답은 <b>${esc(conflictText(d.conflict))}</b>` +
      `<div class="row"><button id="lab-res-mine">내 것 유지</button><button id="lab-res-theirs">그쪽 것으로</button></div></div>` : "") +
    (d.who ? `<div class="muted">작업자: ${esc(d.who)}</div>` : "") +
    `<label class="lab-notefld">메모 <textarea id="lab-d-note" rows="2">${esc(d.note || "")}</textarea></label>` +
    `<label class="check"><input id="lab-auto" type="checkbox"${S.autoNext !== false ? " checked" : ""}> 붙이면 다음 샘플로</label>` +
    `<div class="kv lab-stats">${st}</div>`;
  $("lab-d-prev").onclick = () => move(-1);
  $("lab-d-next").onclick = () => move(1);
  if ($("lab-d-clear")) $("lab-d-clear").onclick = () => clearLabel();
  if ($("lab-res-mine")) $("lab-res-mine").onclick = () => resolveConflict(d.name, "mine");
  if ($("lab-res-theirs")) $("lab-res-theirs").onclick = () => resolveConflict(d.name, "theirs");
  $("lab-detail").querySelectorAll("[data-brush]").forEach((b) => (b.onclick = () => { S.brush = b.dataset.brush === "" ? null : Number(b.dataset.brush); renderDetail(); }));
  if ($("lab-d-confirm")) $("lab-d-confirm").onclick = () => confirmSel();
  $("lab-detail").querySelectorAll(".lab-dlabels .lab-chip[data-i]").forEach((b) => (b.onclick = () => applyLabel(Number(b.dataset.i), [d.name])));
  $("lab-d-note").onchange = async (e) => { try { setState(await L("note", { name: d.name, text: e.target.value })); loadItems(); } catch (err) { showError(err); } };
  $("lab-auto").onchange = (e) => (S.autoNext = e.target.checked);
  if ($("lab-d-num")) {
    const go = async () => {
      const v = $("lab-d-num").value;
      if (v === "") return;
      try { await after(await L("set", { names: [d.name], label: Number(v) }), true); } catch (err) { showError(err); }
    };
    $("lab-d-num-go").onclick = go;
    $("lab-d-num").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); go(); } };
    $("lab-d-num").focus();
  }
  bindChart();
  bindBoxes();
  fillImages();
}
function redrawBoxes() { const s = $("lab-boxsvg"); if (s) s.innerHTML = boxOverlay(); }
async function saveBoxes() {
  try { setState(await L("set_boxes", { name: S.detail.name, boxes: S.boxes })); await loadItems(); }
  catch (e) { showError(e); await loadDetail(S.detail.name); }
}
function bindBoxes() {                                   // 상자 그리기·고르기·옮기기·크기 바꾸기
  const svg = $("lab-boxsvg");
  if (!svg) return;
  const pt = (e) => { const r = svg.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
  svg.onmousedown = (e) => {
    e.preventDefault();
    const [px, py] = pt(e), h = e.target.dataset.h, i = e.target.dataset.i;
    let mode, orig;
    if (h && S.boxSel != null) { mode = "resize"; orig = S.boxes[S.boxSel].slice(); }
    else if (i !== undefined) { S.boxSel = Number(i); mode = "move"; orig = S.boxes[S.boxSel].slice(); }
    else { S.boxSel = null; mode = "draw"; }
    redrawBoxes();
    let moved = false;
    const mv = (ev) => {
      const [qx, qy] = pt(ev);
      moved = true;
      if (mode === "draw") { S.boxDraft = normBox(px, py, qx, qy) || [Math.min(px, qx), Math.min(py, qy), 0, 0]; }
      else if (mode === "move") {
        const [x, y, w, hh, k] = orig;
        S.boxes[S.boxSel] = [Math.min(1 - w, Math.max(0, x + qx - px)), Math.min(1 - hh, Math.max(0, y + qy - py)), w, hh, k];
      } else {
        const [x, y, , , k] = orig, nb = normBox(x, y, qx, qy);
        if (nb) S.boxes[S.boxSel] = [...nb, k];
      }
      redrawBoxes();
    };
    const up = async (ev) => {
      document.removeEventListener("mousemove", mv); document.removeEventListener("mouseup", up);
      if (mode === "draw") {
        const [qx, qy] = pt(ev), nb = normBox(px, py, qx, qy);
        S.boxDraft = null;
        const k = S.state.classes[S.brush == null ? 0 : S.brush];
        if (nb && k) { S.boxes = (S.boxes || []).concat([[...nb, k.name]]); S.boxSel = S.boxes.length - 1; await saveBoxes(); }
        renderDetail();
      } else if (moved) { await saveBoxes(); renderDetail(); }
      else renderDetail();
    };
    document.addEventListener("mousemove", mv);
    document.addEventListener("mouseup", up);
  };
}
function redrawChart() {                                 // 차트만 바꿈 (메모·체크 상자는 그대로)
  const old = $("lab-big");
  if (!old) return;
  old.outerHTML = bigChart();
  bindChart();
}
function bindChart() {
  const svg = $("lab-big"), d = S.detail;
  if (!svg || !d) return;
  const n = d.values.length - 1;
  svg.onwheel = (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect(), f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const [a, b] = S.view, mid = a + f * (b - a), k = e.deltaY < 0 ? 0.7 : 1 / 0.7;
    const na = mid - (mid - a) * k, nb = mid + (b - mid) * k;
    if (nb - na < 8) return;
    S.view = [Math.max(0, na), Math.min(n, nb)];
    redrawChart();
  };
  svg.onmousedown = (e) => {
    e.preventDefault();                                  // 끄는 동안 글자가 선택되지 않게
    if (isSeg() && !e.shiftKey) return paintDrag(e, svg);
    const r = svg.getBoundingClientRect(), x0 = e.clientX, v0 = S.view.slice();
    const mv = (ev) => {
      const dx = ((ev.clientX - x0) / r.width) * (v0[1] - v0[0]), w = v0[1] - v0[0];
      const a = Math.min(Math.max(0, v0[0] - dx), n - w);
      S.view = [a, a + w];
      redrawChart();
    };
    const up = () => { document.removeEventListener("mousemove", mv); document.removeEventListener("mouseup", up); };
    document.addEventListener("mousemove", mv);
    document.addEventListener("mouseup", up);
  };
  svg.ondblclick = () => { S.view = [0, Math.max(1, n)]; redrawChart(); };
}
function paintDrag(e, svg) {                             // 구간 칠하기: 끈 범위를 붓 클래스로 (0 = 지우개)
  const r = svg.getBoundingClientRect(), W = 400, P = 30;
  const toSvg = (cx) => Math.min(W - 6, Math.max(P, ((cx - r.left) / r.width) * W));
  const frac = (sx) => (sx - P) / (W - P - 6);
  const x0 = toSvg(e.clientX), cx0 = e.clientX, cy0 = e.clientY;
  S.paintSel = [x0, x0];
  const mv = (ev) => { S.paintSel = [x0, toSvg(ev.clientX)]; redrawChart(); };
  const up = async (ev) => {
    document.removeEventListener("mousemove", mv); document.removeEventListener("mouseup", up);
    const x1 = toSvg(ev.clientX), d = S.detail;
    S.paintSel = null;
    if (!isDrag(cx0, cy0, ev.clientX, ev.clientY)) { redrawChart(); return; }   // 누르기만 했으면 칠하지 않음
    const [start, end] = pxToRows(frac(x0), frac(x1), S.view[0], S.view[1], d.values.length, d.n);
    const cls = S.brush == null ? null : S.state.classes[S.brush] && S.state.classes[S.brush].name;
    if (S.brush != null && !cls) return showMessage("붓으로 쓸 클래스가 없습니다", "클래스를 더하거나 '열에서 가져오기'를 쓰세요");
    try { setState(await L("paint", { name: d.name, start, end, cls })); await loadItems(); await loadDetailKeepView(d.name); }
    catch (err) { showError(err); redrawChart(); }
  };
  document.addEventListener("mousemove", mv);
  document.addEventListener("mouseup", up);
}
async function loadDetailKeepView(n) {                   // 칠한 뒤: 확대 상태 그대로 다시 그림
  const view = S.view;
  S.detail = await L("item", { name: n });
  S.view = view;
  renderDetail();
}
function conflictText(c) {
  if (c.segments) return `구간 ${c.segments.length}개`;
  if (c.boxes) return `상자 ${c.boxes.length}개`;
  return labelText(c.label);
}
async function resolveConflict(name, take) {
  try { await after(await L("resolve", { name, take }), true); } catch (e) { showError(e); }
}
function move(dir) {                                     // 크게 보기 이전·다음 (페이지를 넘어가면 다음 쪽)
  const ns = names(), i = ns.indexOf(S.cur);
  if (i + dir >= 0 && i + dir < ns.length) {
    S.cur = ns[i + dir]; S.sel = new Set([S.cur]); S.anchor = S.cur;
    renderGrid(); loadDetail(S.cur);
  } else if (dir > 0 && S.page < S.pages - 1) { S.page++; loadItems().then(() => { if (S.items[0]) { S.cur = S.items[0].name; S.sel = new Set([S.cur]); renderGrid(); loadDetail(S.cur); } }); }
  else if (dir < 0 && S.page > 0) { S.page--; loadItems().then(() => { const last = S.items[S.items.length - 1]; if (last) { S.cur = last.name; S.sel = new Set([S.cur]); renderGrid(); loadDetail(S.cur); } }); }
}

// ------------------------------------------------------------ 붙이기·지우기·확인·되돌리기
const targets = (given) => given || (S.sel.size ? [...S.sel] : S.cur ? [S.cur] : []);
async function after(st, moveNext) {
  setState(st);
  const cur = S.cur, before = names(), single = S.sel.size <= 1;
  await loadItems();
  const now = names();
  if (moveNext && S.autoNext !== false && single && cur) {
    const nx = nextAfter(before, now, cur);
    if (nx) { S.cur = nx; S.sel = new Set([nx]); S.anchor = nx; renderGrid(); loadDetail(nx); return; }
    if (!now.includes(cur) && S.page < S.pages - 1) { move(1); return; }
  }
  if (S.cur && now.includes(S.cur)) loadDetail(S.cur);
  else if (S.cur) { S.cur = null; S.detail = null; $("lab-detail").innerHTML = `<p class="muted">이 조건의 샘플을 다 봤습니다.</p>`; }
}
$("lab-num-all-go").onclick = async () => {
  const ns = targets(), v = $("lab-num-all").value;
  if (!ns.length || v === "") return showMessage("샘플을 고르고 숫자를 넣으세요");
  try { await after(await L("set", { names: ns, label: Number(v) }), false); } catch (e) { showError(e); }
};
async function applyLabel(i, given) {
  if (isNum()) return;                                   // 숫자 정답은 숫자키로 붙이지 않음 (입력 칸)
  const ns = targets(given), k = S.state.classes[i];
  if (!k) return;
  if (!ns.length) return showMessage("샘플을 먼저 고르세요", "카드를 누르거나 Shift·Ctrl+클릭으로 여러 개");
  try { await after(await L("set", { names: ns, label: k.name }), true); } catch (e) { showError(e); }
}
async function clearLabel() {
  const ns = targets();
  if (ns.length) try { await after(await L("set", { names: ns, label: null }), false); } catch (e) { showError(e); }
}
async function confirmSel() {
  const ns = targets();
  if (ns.length) try { await after(await L("confirm", { names: ns }), true); } catch (e) { showError(e); }
}
async function undoRedo(op) {
  try { const st = await L(op, {}); if (!st.changed) showMessage(op === "undo" ? "더 되돌릴 것이 없습니다" : "다시 할 것이 없습니다"); await after(st, false); }
  catch (e) { showError(e); }
}
$("lab-undo").onclick = () => undoRedo("undo");
$("lab-redo").onclick = () => undoRedo("redo");
document.addEventListener("keydown", (e) => {
  if ($("lab-work").hidden) return;
  const a = keyAction(e);
  if (!a) return;
  e.preventDefault();
  if (isBox()) {                                         // 상자 작업: 숫자키 = (고른 상자의) 클래스, Delete = 상자 지우기
    if (a.type === "label" && a.index < S.state.classes.length) {
      if (S.boxSel != null && S.boxes[S.boxSel]) { S.boxes[S.boxSel][4] = S.state.classes[a.index].name; saveBoxes().then(renderDetail); }
      else { S.brush = a.index; if (S.detail) renderDetail(); }
      return;
    }
    if (a.type === "clear" && S.boxSel != null) { S.boxes.splice(S.boxSel, 1); S.boxSel = null; saveBoxes().then(renderDetail); return; }
    if (a.type === "esc") { S.boxSel = null; if (S.detail) renderDetail(); return; }
    if (a.type === "copy_prev" && S.detail) {
      const ns = names(), i = ns.indexOf(S.detail.name);
      if (i > 0) L("item", { name: ns[i - 1] }).then(async (prev) => {
        if (!(prev.boxes || []).length) return showMessage("이전 그림에 상자가 없습니다");
        S.boxes = prev.boxes.map((b) => b.slice()); await saveBoxes(); renderDetail();
      }).catch(showError);
      return;
    }
    if (a.type === "label" || a.type === "clear" || a.type === "confirm") return;
  }
  if (a.type === "copy_prev") return;
  if (isSeg() && (a.type === "label" || a.type === "clear")) {   // 구간 작업: 숫자키 = 붓 고르기
    S.brush = a.type === "clear" ? null : (a.index < S.state.classes.length ? a.index : S.brush);
    if (S.detail) renderDetail();
    return;
  }
  if (a.type === "label") applyLabel(a.index);
  else if (a.type === "clear") clearLabel();
  else if (a.type === "confirm") confirmSel();
  else if (a.type === "next") move(1);
  else if (a.type === "prev") move(-1);
  else if (a.type === "undo" || a.type === "redo") undoRedo(a.type);
  else if (a.type === "all") { S.sel = new Set(names()); renderGrid(); }
  else if (a.type === "esc") { S.sel.clear(); renderGrid(); }
});

// ------------------------------------------------------------ 도구: 규칙·가져오기·내보내기·학습 데이터로 쓰기
function togglePanel(id) { for (const p of ["lab-rule", "lab-import", "lab-export", "lab-cluster", "lab-column-panel", "lab-team"]) $(p).hidden = p === id ? !$(p).hidden : true; }
$("lab-tool-column").onclick = async () => {
  togglePanel("lab-column-panel");
  try { const r = await L("columns", {}); $("lab-col-select").innerHTML = r.columns.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join(""); }
  catch (e) { showError(e); }
};
$("lab-col-go").onclick = async () => {
  try {
    const st = await L("import_column", { column: $("lab-col-select").value });
    showMessage(`${st.samples}개 샘플에 구간을 가져왔음` + (st.new_classes.length ? ` · 새 클래스 ${st.new_classes.join(", ")}` : "") +
      (st.kept_manual ? ` · 사람이 칠한 ${st.kept_manual}개는 그대로` : ""), "가져온 구간은 '확인 안 함'으로 걸러 보며 확인하세요", "", true);
    if (S.brush == null && st.classes.length) S.brush = 0;
    await after(st, false);
  } catch (e) { showError(e); }
};
$("lab-tool-cluster").onclick = () => togglePanel("lab-cluster");
$("lab-tool-team").onclick = () => { togglePanel("lab-team"); $("lab-who").value = (S.state && S.state.who) || ""; };
async function setWho(name) {
  try {
    const st = await L("who", { name });
    try { localStorage.setItem("dlstudio-labeler", name); } catch (e) { /* 저장 안 돼도 됨 */ }
    if (st.counts) setState(st);
  } catch (e) { showError(e); }
}
$("lab-who-go").onclick = () => setWho($("lab-who").value.trim());
$("lab-split-go").onclick = async () => {
  try { const r = await L("split", { n: Number($("lab-split-n").value) || 2 }); showMessage(`작업을 ${r.files.length}개로 나눴음`, r.files.join("  ·  "), "", true); }
  catch (e) { showError(e); }
};
$("lab-merge-go").onclick = async () => {
  try {
    const st = await L("merge", { file: pathOf("lab-merge-file") });
    showMessage(`합쳤음: 새로 ${st.added}개 · 같음 ${st.agreed}개 · 충돌 ${st.conflicts}개` + (st.agreement != null ? ` (일치율 ${Math.round(st.agreement * 100)}%)` : ""),
      st.conflicts ? "'충돌만'으로 걸러 보며 하나씩 정하세요 (내 것 유지 / 그쪽 것으로 / 숫자키로 새로)" : "", "", true);
    await after(st, false);
  } catch (e) { showError(e); }
};
$("lab-normal-cand").onclick = async () => {
  try {
    const r = await L("normal_candidates", { k: 4 });
    $("lab-f-label").value = "all"; S.page = 0;           // 후보 묶음(가장 큰 묶음 = 0번)만 걸러 보기
    $("lab-f-cluster").innerHTML = `<option value="">전부</option><option value="0">정상 후보 (${r.count})</option>`;
    $("lab-f-cluster").value = "0";
    await loadItems();
    S.sel = new Set(r.names.filter((n) => names().includes(n)));
    renderGrid();
    showMessage(`정상 후보 ${r.count}개 (가장 큰 묶음)` + (S.sel.size < r.count ? ` · 이 쪽에 보이는 ${S.sel.size}개를 골랐음` : " · 모두 골랐음"),
      "훑어보고 맞으면 1(정상)을 누르세요. 다르게 생긴 것은 Ctrl+클릭으로 빼세요", "", true);
  } catch (e) { showError(e); }
};
$("lab-export-normal").onclick = async () => {
  const file = $("lab-export-file").value.trim().replace(/_정답\.csv$/, "") + "_정상목록.csv";
  try { const r = await L("export_normals", { file, confirmed_only: $("lab-confirmed-only").checked }); showMessage(`정상 ${r.rows}개 목록을 내보냈음: ${file}`, "정상만으로 학습하는 이상 탐지(오토인코더) 과제에 씁니다 (studio 과제는 준비 중)", "", true); }
  catch (e) { showError(e); }
};
$("lab-cluster-go").onclick = async () => {
  try {
    const r = await L("cluster", { k: Number($("lab-cluster-k").value) || 6 });
    $("lab-f-cluster").innerHTML = `<option value="">전부</option>` + r.groups.map((g) => `<option value="${g.id}">묶음 ${g.id + 1} (${g.size})</option>`).join("");
    $("lab-cluster-groups").innerHTML = r.groups.map((g) => `<div class="lab-group" data-id="${g.id}">` +
      `<div class="lab-name">묶음 ${g.id + 1} · ${g.size}개</div><div class="lab-thumb">${thumbOf({ thumb: g.thumb }, 150, 40, "#2f6fde")}</div>` +
      `<div class="lab-gl">${Object.entries(g.labels).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}</div>` +
      `<div class="row"><button data-a="show">보기</button><button data-a="pick">전부 고르기</button></div></div>`).join("");
  } catch (e) { showError(e); }
};
$("lab-cluster-groups").onclick = async (e) => {
  const b = e.target.closest("button[data-a]"), g = e.target.closest(".lab-group");
  if (!b || !g) return;
  $("lab-f-cluster").value = g.dataset.id;
  S.page = 0;
  if (b.dataset.a === "pick") {                          // 이 묶음 전부 고르기 (여러 쪽이면 한 쪽에 다 보이도록 크게)
    const r = await L("items", Object.assign(filterBody(), { size: 2000 }));
    await loadItems();
    S.sel = new Set(r.items.map((x) => x.name));
    renderGrid();
    showMessage(`묶음 ${Number(g.dataset.id) + 1}의 ${S.sel.size}개를 골랐음`, "숫자키로 한꺼번에 정답을 붙이세요", "", true);
  } else loadItems();
};
$("lab-suggest").onclick = async () => {
  try {
    const st = await L("suggest", {});
    showMessage(`모델이 ${st.applied}개에 정답을 제안했음 (학습 ${st.trained_on}개` + (st.acc != null ? `, 예상 정확도 ${Math.round(st.acc * 100)}%` : ", 정답이 10개 미만이라 예상 정확도는 모름") + `)`,
      "정렬을 '헷갈리는 것부터'로 두고 확인하세요 (Enter 확인, 숫자키로 고치기)", "", true);
    $("lab-sort").value = "uncertainty";
    await after(st, false);
  } catch (e) { showError(e); }
};
$("lab-tool-rule").onclick = () => { togglePanel("lab-rule"); if (!$("lab-conds").children.length) addCond(); };
$("lab-tool-import").onclick = () => togglePanel("lab-import");
$("lab-tool-export").onclick = () => togglePanel("lab-export");
function addCond() {
  const div = document.createElement("div");
  div.className = "row lab-cond";
  div.innerHTML = `<select class="c-stat">${S.state.stat_names.map((k) => `<option value="${esc(k)}">${esc(statLabel(k))}</option>`).join("")}</select>` +
    `<select class="c-op"><option value=">">&gt;</option><option value=">=">≥</option><option value="<">&lt;</option><option value="<=">≤</option><option value="==">=</option></select>` +
    `<input class="c-val" type="number" step="any" placeholder="값"><button class="c-del">빼기</button>`;
  div.querySelector(".c-del").onclick = () => div.remove();
  $("lab-conds").appendChild(div);
}
$("lab-cond-add").onclick = addCond;
function conds() {
  return [...document.querySelectorAll(".lab-cond")].map((d) => ({ stat: d.querySelector(".c-stat").value, op: d.querySelector(".c-op").value,
    value: d.querySelector(".c-val").value === "" ? null : Number(d.querySelector(".c-val").value) }));
}
$("lab-rule-preview").onclick = async () => {
  try {
    const r = await L("rule_preview", { conds: conds(), overwrite: $("lab-rule-overwrite").checked,
      label: isNum() ? ($("lab-rule-num").value === "" ? null : Number($("lab-rule-num").value)) : $("lab-rule-label").value });
    $("lab-rule-result").textContent = `${r.count}개가 해당` + (r.names.length ? ` (예: ${r.names.slice(0, 5).join(", ")}${r.count > 5 ? " …" : ""})` : "");
  } catch (e) { showError(e); }
};
$("lab-rule-apply").onclick = async () => {
  try {
    const lab = isNum() ? Number($("lab-rule-num").value) : $("lab-rule-label").value;
    if (isNum() && $("lab-rule-num").value === "") return showMessage("붙일 숫자를 넣으세요");
    const st = await L("rule_apply", { conds: conds(), label: lab, overwrite: $("lab-rule-overwrite").checked });
    $("lab-rule-result").textContent = `${st.changed}개에 '${lab}'을(를) 붙였음 (확인 안 함으로)`;
    await after(st, false);
  } catch (e) { showError(e); }
};
$("lab-import-go").onclick = async () => {
  try {
    const st = await L("import", { label_file: pathOf("lab-import-file") });
    showMessage(`${st.applied}개를 가져왔음` + (st.unknown ? ` · 표에만 있고 폴더에 없는 이름 ${st.unknown}개는 건너뜀` : "") +
      (st.kept_manual ? ` · 사람이 붙인 정답 ${st.kept_manual}개는 그대로 둠` : ""), "가져온 정답은 '확인 안 함'으로 걸러 보며 확인하세요", "", true);
    await after(st, false);
  } catch (e) { showError(e); }
};
$("lab-yolo-go").onclick = async () => {
  try { const r = await L("export_yolo", { folder: $("lab-yolo-dir").value.trim(), confirmed_only: $("lab-confirmed-only").checked }); showMessage(`YOLO로 상자 ${r.boxes}개를 내보냈음: ${$("lab-yolo-dir").value.trim()}`, "", "", true); }
  catch (e) { showError(e); }
};
$("lab-coco-go").onclick = async () => {
  try { const r = await L("export_coco", { file: $("lab-coco-file").value.trim(), confirmed_only: $("lab-confirmed-only").checked }); showMessage(`COCO로 내보냈음: ${r.file}`, "", "", true); }
  catch (e) { showError(e); }
};
$("lab-export-go").onclick = async () => {
  try {
    const r = await L("export", { file: $("lab-export-file").value.trim(), confirmed_only: $("lab-confirmed-only").checked });
    showMessage(`정답 ${r.rows}개를 내보냈음: ${$("lab-export-file").value.trim()}`, "", "", true);
  } catch (e) { showError(e); }
};
$("lab-use").onclick = async () => {
  try {
    const r = await api("/api/label/use", { confirmed_only: $("lab-confirmed-only").checked });
    const s = r.summary;
    showMessage(`studio 데이터로 불러왔음: 샘플 ${s.n_train + s.n_test}개, ` + (s.task === "regression" ? "회귀" : `클래스 ${s.classes.join(" · ")}`) + ` (${r.label_file})`,
      "studio 탭을 새로 고치면 ③ 모델로 이어서 할 수 있습니다", "", true);
  } catch (e) { showError(e); }
};

// 시작 화면 ↔ 지금 작업 (열린 작업은 새 작업을 만들거나 다른 파일을 열 때까지 그대로)
function showStart() {
  $("lab-work").hidden = true;
  $("lab-start").hidden = false;
  $("lab-switch").hidden = true;
  $("lab-back-box").hidden = !S.state;
  if (S.state) $("lab-back-name").textContent = S.state.file || "";
}
$("lab-switch").onclick = showStart;
window.DLLabel = { start, showStart, setState, after, loadItems, showMessage, showError, scan, state: () => S.state };
$("lab-back").onclick = () => { if (S.state) start(S.state); };

// 새로 고쳐도 서버에 남은 작업을 이어서 (studio에서 ?folder= 로 넘어왔으면 시작 화면에 머묾)
(async () => {
  if (WEB) return;                                       // 웹: 작업 목록에서 고름 (labels.js)
  let st = null;
  try { st = await L("state", {}); } catch (e) { return; /* 작업 없음: 처음 화면 */ }
  if (new URLSearchParams(location.search).get("folder")) { S.state = st; showStart(); return; }
  start(st);
})();
})();

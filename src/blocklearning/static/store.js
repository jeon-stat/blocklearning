// BlockLearning 웹 버전의 프로젝트 저장소. 두 가지가 같은 모양으로 쓰인다:
//   local — 이 브라우저 안 보관함(IndexedDB). 로그인 없음, 이 PC·이 브라우저에만
//   drive — 사용자 본인의 구글 드라이브 "BlockLearning" 폴더 아래 프로젝트마다 하위 폴더
// 프로젝트 = project.dlstudio(그래프·설정·결과·가중치, 자주 저장) + data.zip(고른 데이터 사본, 데이터를 바꿀 때만)
// 모양: list() → [{id, name, updated}] · create(name) → {id, name} · save(id, {project, data?}) · load(id) → {project, data}
//       rename(id, name) · remove(id)
(function () {
"use strict";

// 구글 클라우드 콘솔에서 만든 OAuth 클라이언트 ID (공개돼도 되는 값, 비밀번호 아님). 비어 있으면 드라이브는 '준비 중'
const GOOGLE_CLIENT_ID = "159159621123-6hakqofb5e8o68dt3olbiucbl3q4qsq1.apps.googleusercontent.com";
// 이 앱이 만든 파일만 보는 권한. 사용자의 다른 드라이브 파일·Gmail·계정에는 닿지 않음 (넓히지 말 것)
const DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file";
const ROOT_NAME = "BlockLearning";
const OLD_ROOT_NAME = "DL studio";                      // 예전 이름: 처음 로그인할 때 이 폴더를 새 이름으로 바꿈 (안의 프로젝트는 그대로)
const FOLDER = "application/vnd.google-apps.folder";

// ------------------------------------------------------------ 이 브라우저 (IndexedDB)
function idb() {
  return new Promise((ok, no) => {
    const r = indexedDB.open("dlstudio", 2);                  // 2: 라벨링 작업(labels) 더함
    r.onupgradeneeded = () => {
      const db = r.result, has = (n) => db.objectStoreNames.contains(n);
      if (!has("projects")) db.createObjectStore("projects", { keyPath: "id" });
      if (!has("blobs")) db.createObjectStore("blobs");
      if (!has("labels")) db.createObjectStore("labels", { keyPath: "id" });
    };
    r.onsuccess = () => ok(r.result);
    r.onerror = () => no(r.error || new Error("브라우저 보관함을 열 수 없음"));
  });
}
async function tx(stores, mode, fn) {
  const db = await idb();
  return new Promise((ok, no) => {
    const t = db.transaction(stores, mode);
    let out;
    Promise.resolve(fn(t)).then((v) => { out = v; });
    t.oncomplete = () => { db.close(); ok(out); };
    t.onerror = () => { db.close(); no(t.error); };
    t.onabort = () => { db.close(); no(t.error || new Error("저장이 취소됨 (보관함 용량이 모자랄 수 있음)")); };
  });
}
const req = (r) => new Promise((ok, no) => { r.onsuccess = () => ok(r.result); r.onerror = () => no(r.error); });

const local = {
  kind: "local",
  label: "이 브라우저",
  available: () => typeof indexedDB !== "undefined",
  async list() {
    const all = await tx(["projects"], "readonly", (t) => req(t.objectStore("projects").getAll()));
    return all.sort((a, b) => b.updated - a.updated);
  },
  async create(name) {
    const p = { id: "p" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), name, updated: Date.now() };
    await tx(["projects"], "readwrite", (t) => t.objectStore("projects").put(p));
    return p;
  },
  async save(id, { project, data, meta, steps }) {
    await tx(["projects", "blobs"], "readwrite", async (t) => {
      const ps = t.objectStore("projects");
      const p = await req(ps.get(id));
      if (!p) throw new Error("프로젝트가 없음");
      p.updated = Date.now();
      if (meta !== undefined) p.meta = meta;
      if (steps !== undefined) p.steps = steps;              // 만들기 단계 진행 (첫 화면에서 고른 카드의 진행 표시)
      ps.put(p);
      t.objectStore("blobs").put(project, id + "/project");
      if (data !== undefined) t.objectStore("blobs").put(data, id + "/data");
    });
  },
  async load(id) {
    return tx(["blobs"], "readonly", async (t) => {
      const b = t.objectStore("blobs");
      return { project: await req(b.get(id + "/project")), data: await req(b.get(id + "/data")) };
    });
  },
  async rename(id, name) {
    await tx(["projects"], "readwrite", async (t) => {
      const ps = t.objectStore("projects"), p = await req(ps.get(id));
      if (p) { p.name = name; p.updated = Date.now(); ps.put(p); }
    });
  },
  async remove(id) {
    await tx(["projects", "blobs"], "readwrite", (t) => {
      t.objectStore("projects").delete(id);
      t.objectStore("blobs").delete(id + "/project");
      t.objectStore("blobs").delete(id + "/data");
    });
  },
};

// ------------------------------------------------------------ 라벨링 작업 (이 브라우저): 목록(labels) + 정답 JSON·데이터 사본(blobs의 L<id>/work·data)
// 모양: list() → [{id, name, updated, meta}] · create(name) · save(id, {work, data?, meta}) · load(id) → {work, data} · rename · remove
const labels = {
  kind: "local",
  label: "이 브라우저",
  async list() {
    const all = await tx(["labels"], "readonly", (t) => req(t.objectStore("labels").getAll()));
    return all.sort((a, b) => b.updated - a.updated);
  },
  async create(name) {
    const p = { id: "l" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), name, updated: Date.now(), meta: null };
    await tx(["labels"], "readwrite", (t) => t.objectStore("labels").put(p));
    return p;
  },
  async save(id, { work, data, meta }) {
    await tx(["labels", "blobs"], "readwrite", async (t) => {
      const ls = t.objectStore("labels");
      const p = await req(ls.get(id));
      if (!p) throw new Error("라벨링 작업이 없음");
      p.updated = Date.now();
      if (meta !== undefined) p.meta = meta;
      ls.put(p);
      if (work !== undefined) t.objectStore("blobs").put(work, "L" + id + "/work");
      if (data !== undefined) t.objectStore("blobs").put(data, "L" + id + "/data");
    });
  },
  async load(id) {
    return tx(["blobs"], "readonly", async (t) => {
      const b = t.objectStore("blobs");
      return { work: await req(b.get("L" + id + "/work")), data: await req(b.get("L" + id + "/data")) };
    });
  },
  async rename(id, name) {
    await tx(["labels"], "readwrite", async (t) => {
      const ls = t.objectStore("labels"), p = await req(ls.get(id));
      if (p) { p.name = name; p.updated = Date.now(); ls.put(p); }
    });
  },
  async remove(id) {
    await tx(["labels", "blobs"], "readwrite", (t) => {
      t.objectStore("labels").delete(id);
      t.objectStore("blobs").delete("L" + id + "/work");
      t.objectStore("blobs").delete("L" + id + "/data");
    });
  },
};

// ------------------------------------------------------------ 구글 드라이브
// 로그인은 구글 창(accounts.google.com)에서 하고, 이 페이지는 1시간짜리 출입증만 메모리에 둔다 (브라우저 저장소에 남기지 않음).
let token = null, tokenUntil = 0, tokenClient = null, rootId = null, user = "";

function loadGis() {
  if (window.google && google.accounts && google.accounts.oauth2) return Promise.resolve();
  return new Promise((ok, no) => {
    const s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client";
    s.onload = ok;
    s.onerror = () => no(new Error("구글 로그인 스크립트를 불러오지 못함"));
    document.head.appendChild(s);
  });
}
async function signIn(prompt = "") {                      // 단추를 누를 때 불러야 함 (팝업)
  if (!GOOGLE_CLIENT_ID) throw new Error("구글 드라이브 연결은 준비 중입니다");
  await loadGis();
  return new Promise((ok, no) => {
    tokenClient = google.accounts.oauth2.initTokenClient({
      client_id: GOOGLE_CLIENT_ID, scope: DRIVE_SCOPE,
      callback: (r) => {
        if (r.error) return no(new Error("구글 로그인 실패: " + r.error));
        if (!google.accounts.oauth2.hasGrantedAllScopes(r, DRIVE_SCOPE)) return no(new Error("드라이브 권한을 허용해야 저장할 수 있습니다"));
        token = r.access_token;
        tokenUntil = Date.now() + (Number(r.expires_in || 3600) - 60) * 1000;
        ok();
      },
      error_callback: (e) => no(new Error(e && e.type === "popup_closed" ? "로그인 창을 닫았습니다" : "구글 로그인 창을 열 수 없음 (팝업 차단 확인)")),
    });
    tokenClient.requestAccessToken({ prompt });
  });
}
function signOut() {
  if (token && window.google && google.accounts) google.accounts.oauth2.revoke(token, () => {});
  token = null; tokenUntil = 0; rootId = null; user = "";
}
const signedIn = () => !!token && Date.now() < tokenUntil;

async function gapi(path, opt = {}) {
  if (!signedIn()) throw Object.assign(new Error("구글 로그인이 필요합니다 (1시간이 지나면 다시 로그인)"), { needLogin: true });
  const r = await fetch(path.startsWith("http") ? path : "https://www.googleapis.com/drive/v3/" + path,
    Object.assign({}, opt, { headers: Object.assign({ Authorization: "Bearer " + token }, opt.headers || {}) }));
  if (r.status === 401) { token = null; throw Object.assign(new Error("구글 로그인이 끝났습니다. 다시 로그인하세요"), { needLogin: true }); }
  if (!r.ok) throw new Error(`구글 드라이브 오류 ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r;
}
const q = (s) => encodeURIComponent(s);
const esc = (s) => String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
async function findChildren(parent, extra = "") {
  const r = await gapi(`files?q=${q(`'${parent}' in parents and trashed=false${extra}`)}&fields=${q("files(id,name,mimeType,modifiedTime,description,appProperties)")}&pageSize=1000&spaces=drive`);
  return (await r.json()).files || [];
}
async function makeFolder(name, parent) {
  const r = await gapi("files?fields=id,name,modifiedTime", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, mimeType: FOLDER, parents: [parent] }) });
  return r.json();
}
async function root() {                                   // 내 드라이브/BlockLearning (이 앱이 만든 것만 보임)
  if (rootId) return rootId;
  const find = (name) => findChildren("root", ` and mimeType='${FOLDER}' and name='${esc(name)}'`);
  let found = await find(ROOT_NAME);
  if (!found.length) {
    found = await find(OLD_ROOT_NAME);
    if (found.length) await gapi(`files/${found[0].id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: ROOT_NAME }) });
  }
  rootId = found.length ? found[0].id : (await makeFolder(ROOT_NAME, "root")).id;
  return rootId;
}
async function putFile(folder, name, bytes) {           // 같은 이름 파일이 있으면 내용만 바꿈
  const ex = (await findChildren(folder, ` and name='${esc(name)}'`))[0];
  const type = "application/octet-stream";
  if (ex) {
    await gapi(`https://www.googleapis.com/upload/drive/v3/files/${ex.id}?uploadType=media`, { method: "PATCH", headers: { "Content-Type": type }, body: bytes });
    return;
  }
  const meta = new Blob([JSON.stringify({ name, parents: [folder] })], { type: "application/json" });
  const form = new FormData();
  form.append("metadata", meta);
  form.append("file", new Blob([bytes], { type }));
  await gapi("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id", { method: "POST", body: form });
}
async function getFile(folder, name) {
  const f = (await findChildren(folder, ` and name='${esc(name)}'`))[0];
  if (!f) return null;
  return new Uint8Array(await (await gapi(`files/${f.id}?alt=media`)).arrayBuffer());
}

const metaSent = {}, stepsSent = {};
const drive = {
  kind: "drive",
  label: "구글 드라이브",
  available: () => !!GOOGLE_CLIENT_ID,
  signedIn, signIn, signOut, user: () => user,
  async list() {
    const items = await findChildren(await root(), ` and mimeType='${FOLDER}'`);
    return items.map((f) => ({ id: f.id, name: f.name, updated: Date.parse(f.modifiedTime) || 0, meta: f.description || "", steps: (f.appProperties || {}).steps || "" })).sort((a, b) => b.updated - a.updated);
  },
  async create(name) { const f = await makeFolder(name, await root()); return { id: f.id, name: f.name, updated: Date.now() }; },
  async save(id, { project, data, meta, steps }) {
    await putFile(id, "project.dlstudio", project);
    if (data !== undefined && data !== null) await putFile(id, "data.zip", data);
    if (meta !== undefined && meta !== metaSent[id]) {        // 폴더 설명에 한 줄 요약 (첫 화면 카드에 보임)
      await gapi(`files/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ description: meta }) });
      metaSent[id] = meta;
    }
    if (steps !== undefined && steps !== stepsSent[id]) {      // 만들기 단계 진행 (폴더의 앱 속성)
      await gapi(`files/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ appProperties: { steps } }) });
      stepsSent[id] = steps;
    }
  },
  async load(id) { return { project: await getFile(id, "project.dlstudio"), data: await getFile(id, "data.zip") }; },
  async rename(id, name) {
    await gapi(`files/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  },
  async remove(id) {                                    // 휴지통으로 (드라이브에서 되살릴 수 있음)
    await gapi(`files/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trashed: true }) });
  },
};

window.DLStore = { local, drive, labels, GOOGLE_CLIENT_ID, DRIVE_SCOPE };
})();

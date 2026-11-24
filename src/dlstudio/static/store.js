// DL studio 웹 버전의 프로젝트 저장소. 두 가지가 같은 모양으로 쓰인다:
//   local — 이 브라우저 안 보관함(IndexedDB). 로그인 없음, 이 PC·이 브라우저에만
//   drive — 사용자 본인의 구글 드라이브 "DL studio" 폴더 아래 프로젝트마다 하위 폴더
// 프로젝트 = project.dlstudio(그래프·설정·결과·가중치, 자주 저장) + data.zip(고른 데이터 사본, 데이터를 바꿀 때만)
// 모양: list() → [{id, name, updated}] · create(name) → {id, name} · save(id, {project, data?}) · load(id) → {project, data}
//       rename(id, name) · remove(id)
(function () {
"use strict";

// 구글 클라우드 콘솔에서 만든 OAuth 클라이언트 ID (공개돼도 되는 값, 비밀번호 아님). 비어 있으면 드라이브는 '준비 중'
const GOOGLE_CLIENT_ID = "159159621123-6hakqofb5e8o68dt3olbiucbl3q4qsq1.apps.googleusercontent.com";
// 이 앱이 만든 파일만 보는 권한. 사용자의 다른 드라이브 파일·Gmail·계정에는 닿지 않음 (넓히지 말 것)
const DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file";
const ROOT_NAME = "DL studio";
const FOLDER = "application/vnd.google-apps.folder";

// ------------------------------------------------------------ 이 브라우저 (IndexedDB)
function idb() {
  return new Promise((ok, no) => {
    const r = indexedDB.open("dlstudio", 1);
    r.onupgradeneeded = () => { r.result.createObjectStore("projects", { keyPath: "id" }); r.result.createObjectStore("blobs"); };
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
  async save(id, { project, data }) {
    await tx(["projects", "blobs"], "readwrite", async (t) => {
      const ps = t.objectStore("projects");
      const p = await req(ps.get(id));
      if (!p) throw new Error("프로젝트가 없음");
      p.updated = Date.now();
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
  const r = await gapi(`files?q=${q(`'${parent}' in parents and trashed=false${extra}`)}&fields=${q("files(id,name,mimeType,modifiedTime)")}&pageSize=1000&spaces=drive`);
  return (await r.json()).files || [];
}
async function makeFolder(name, parent) {
  const r = await gapi("files?fields=id,name,modifiedTime", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, mimeType: FOLDER, parents: [parent] }) });
  return r.json();
}
async function root() {                                   // 내 드라이브/DL studio (이 앱이 만든 것만 보임)
  if (rootId) return rootId;
  const found = await findChildren("root", ` and mimeType='${FOLDER}' and name='${esc(ROOT_NAME)}'`);
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

const drive = {
  kind: "drive",
  label: "구글 드라이브",
  available: () => !!GOOGLE_CLIENT_ID,
  signedIn, signIn, signOut, user: () => user,
  async list() {
    const items = await findChildren(await root(), ` and mimeType='${FOLDER}'`);
    return items.map((f) => ({ id: f.id, name: f.name, updated: Date.parse(f.modifiedTime) || 0 })).sort((a, b) => b.updated - a.updated);
  },
  async create(name) { const f = await makeFolder(name, await root()); return { id: f.id, name: f.name, updated: Date.now() }; },
  async save(id, { project, data }) {
    await putFile(id, "project.dlstudio", project);
    if (data !== undefined && data !== null) await putFile(id, "data.zip", data);
  },
  async load(id) { return { project: await getFile(id, "project.dlstudio"), data: await getFile(id, "data.zip") }; },
  async rename(id, name) {
    await gapi(`files/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  },
  async remove(id) {                                    // 휴지통으로 (드라이브에서 되살릴 수 있음)
    await gapi(`files/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trashed: true }) });
  },
};

window.DLStore = { local, drive, GOOGLE_CLIENT_ID, DRIVE_SCOPE };
})();

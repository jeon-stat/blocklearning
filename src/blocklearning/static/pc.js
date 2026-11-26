// 설치형: 첫 화면 '내 프로젝트'(projects.js)가 쓰는 저장소와 바뀜 알림. 웹 버전의 DLS_WEB·DLStore와 같은 모양으로,
// 저장은 BlockLearning을 실행한 컴퓨터의 프로젝트 폴더(~/BlockLearning/projects)에 (서버의 /api/projects/…).
(function () {
"use strict";
const listeners = [];
const changed = (kind) => { for (const f of listeners) { try { f(kind); } catch (e) { console.error(e); } } };
const doneJobs = new Set();
let dir = "";

// app.js의 api()가 응답을 받을 때마다: 작업이 바뀐 요청이면 자동 저장에 알림
function after(path, body, r) {
  if (body !== undefined) {
    if (path === "/api/data/load") changed("data");
    else if (path === "/api/model/check" || path === "/api/task/select") changed("model");
    else if (path === "/api/runs/delete" || path === "/api/runs/restore") changed("train");
  } else if (/^\/api\/jobs\/[^/]+$/.test(path) && r && r.state === "done" && r.kind === "train" && !doneJobs.has(r.id)) {
    doneJobs.add(r.id);
    changed("train");
  }
}

const local = {
  kind: "local",
  label: "이 컴퓨터",
  available: () => true,
  async list() {
    const r = await api("/api/projects/list", {});
    dir = r.dir;
    return r.items;
  },
  create: (name) => api("/api/projects/create", { name }),
  save: (id, { meta, steps }) => api("/api/projects/save", { id, meta, steps }),
  async load(id) {
    const r = await api("/api/projects/open", { id });
    return r.empty ? {} : { project: r };
  },
  rename: (id, name) => api("/api/projects/rename", { id, name }),
  remove: (id) => api("/api/projects/remove", { id }),
};
const drive = { kind: "drive", label: "", available: () => false, signedIn: () => false };

window.DLS_PC = {
  after,
  store: { local, drive },
  dir: () => dir,
  ready: window.DLS_LOADED || Promise.resolve(),
  onChange: (f) => listeners.push(f),
  hasData: () => !!st.summary,
  recordBytes: () => (st.summary && st.graph ? { project: true } : null),   // 저장은 서버가 파일로 (데이터는 경로로)
  openRecord: async (project) => project,                                    // local.load가 이미 열었음
  openedName: () => "",
  pickProject: () => pickUpload({ accept: ".blocklearning,.dlstudio" }, document.getElementById("home-import")),
  api: (path, body) => api(path, body),
  reset: () => api("/api/projects/close", {}),
  current: async () => (await api("/api/projects/list", {})),
  exportFile: async (id) => {
    const p = (await local.list()).find((x) => x.id === id);
    if (p) fetchBack([p.file]);
  },
};
document.addEventListener("DOMContentLoaded", () => window.DLProjects.init());
})();

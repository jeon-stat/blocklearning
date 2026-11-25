// DL studio 웹 버전 시작: 로컬 실행과 같은 화면(static/index.html)을 불러오고, 계산에 쓸 스크립트를 차례로 불러온다.
// 외부(CDN) 스크립트는 버전을 고정하고 내용 해시(SRI)가 맞을 때만 실행한다 — CDN 파일이 바뀌면 실행하지 않음.
(async () => {
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost"))
    navigator.serviceWorker.register("sw.js").catch(() => {});   // 두 번째부터 빨리 (sw.js: 버전 박힌 CDN 파일은 받아 둠, 이 사이트 파일은 늘 네트워크 먼저)
  const STATIC = "src/dlstudio/static/";
  const CDN = [
    ["https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js", "sha384-90so5tCKvl0xs9agU29IMKlAVzhfzFX7QO//YxQkRhJG58bBZrFN+2ZTRB026X5X"],
    ["https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js", "sha384-vE8hbVJ4lezako5rlvE7bY0BVzWlFhZncPlckrqNwcUQpVtgbENTgZ8TBbnPjZre"],
  ];
  window.DL_BOOT = { STATIC, TF: CDN[1] };                        // 라벨링 페이지가 학습 엔진을 필요할 때 불러옴 (web.js)
  const LABEL = document.body.dataset.page === "label";        // 라벨링 페이지 (label.html): 학습 엔진(TF.js)은 필요 없음
  const html = await (await fetch(STATIC + (LABEL ? "label.html" : "index.html"), { cache: "no-cache" })).text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.body.querySelectorAll("script").forEach((s) => s.remove());
  document.body.className = doc.body.className;
  document.body.replaceChildren(...Array.from(doc.body.childNodes).map((n) => document.importNode(n, true)));
  if (!LABEL) {                     // 첫 화면은 '내 프로젝트' (projects.js가 띄움): 스크립트를 받는 동안 ① 과제 화면 대신 '여는 중'
    document.querySelectorAll("main > .panel").forEach((el) => (el.hidden = true));
    document.querySelectorAll(".step.active").forEach((b) => b.classList.remove("active"));
    for (const id of ["btn-open", "btn-save"]) if (document.getElementById(id)) document.getElementById(id).hidden = true;   // 웹에는 없는 단추
    const w = document.createElement("section");
    w.id = "boot-wait";
    w.className = "panel";
    w.innerHTML = '<p class="muted loading">DL studio를 여는 중… 처음 열 때는 계산 프로그램(Pyodide·TensorFlow.js)을 받느라 10~20초 걸립니다.</p>';
    document.querySelector("main").prepend(w);
  }
  const load = (src, integrity) => new Promise((ok, no) => {
    const s = document.createElement("script");
    s.src = src;
    if (integrity) { s.integrity = integrity; s.crossOrigin = "anonymous"; }
    s.onload = ok;
    s.onerror = () => no(new Error("불러오지 못함: " + src));
    document.body.appendChild(s);
  });
  for (const [src, sri] of LABEL ? CDN.slice(0, 1) : CDN) await load(src, sri);
  const v = "?v=" + Date.now();                              // 새 버전을 바로 쓰게 (브라우저 캐시 피함)
  const css = document.querySelector('link[href$="style.css"]');
  if (css) css.href = STATIC + "style.css" + v;
  const STUDIO = ["engine.js", "store.js", "web.js", "graph.js", "app.js", "projects.js"];
  const LABELS = ["store.js", "web.js", "label.js", "labels.js"];         // 라벨링 페이지
  for (const f of LABEL ? LABELS : STUDIO) await load(STATIC + f + v);
  if (LABEL) return window.DLLabels.init();
  window.DLS_WEB.afterApp();
  window.DLProjects.init();
})().catch((e) => {
  const p = document.createElement("p");
  p.style.padding = "24px";
  p.textContent = "DL studio를 열 수 없음: " + e.message + " — 인터넷 연결을 확인하고 새로 고치세요.";
  document.body.replaceChildren(p);
});

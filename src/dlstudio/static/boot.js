// DL studio 웹 버전 시작: 로컬 실행과 같은 화면(static/index.html)을 불러오고, 계산에 쓸 스크립트를 차례로 불러온다.
// 외부(CDN) 스크립트는 버전을 고정하고 내용 해시(SRI)가 맞을 때만 실행한다 — CDN 파일이 바뀌면 실행하지 않음.
(async () => {
  const STATIC = "src/dlstudio/static/";
  const CDN = [
    ["https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js", "sha384-90so5tCKvl0xs9agU29IMKlAVzhfzFX7QO//YxQkRhJG58bBZrFN+2ZTRB026X5X"],
    ["https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js", "sha384-vE8hbVJ4lezako5rlvE7bY0BVzWlFhZncPlckrqNwcUQpVtgbENTgZ8TBbnPjZre"],
  ];
  const html = await (await fetch(STATIC + "index.html", { cache: "no-cache" })).text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.body.querySelectorAll("script").forEach((s) => s.remove());
  document.body.className = doc.body.className;
  document.body.replaceChildren(...Array.from(doc.body.childNodes).map((n) => document.importNode(n, true)));
  const load = (src, integrity) => new Promise((ok, no) => {
    const s = document.createElement("script");
    s.src = src;
    if (integrity) { s.integrity = integrity; s.crossOrigin = "anonymous"; }
    s.onload = ok;
    s.onerror = () => no(new Error("불러오지 못함: " + src));
    document.body.appendChild(s);
  });
  for (const [src, sri] of CDN) await load(src, sri);
  const v = "?v=" + Date.now();                              // 새 버전을 바로 쓰게 (브라우저 캐시 피함)
  const css = document.querySelector('link[href$="style.css"]');
  if (css) css.href = STATIC + "style.css" + v;
  for (const f of ["engine.js", "store.js", "web.js", "graph.js", "app.js", "projects.js"]) await load(STATIC + f + v);
  window.DLS_WEB.afterApp();
  window.DLProjects.init();
})().catch((e) => {
  const p = document.createElement("p");
  p.style.padding = "24px";
  p.textContent = "DL studio를 열 수 없음: " + e.message + " — 인터넷 연결을 확인하고 새로 고치세요.";
  document.body.replaceChildren(p);
});

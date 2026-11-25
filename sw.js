// DL studio 웹 버전의 서비스 워커: 두 번째부터 빨리 열리게.
// - 버전이 박힌 CDN 파일 (Pyodide v0.27.7·TF.js 4.22.0·글꼴): 처음 한 번 받아 두고 그다음엔 브라우저 안에서 (바뀌지 않는 파일)
// - 이 사이트의 파일: 늘 먼저 네트워크 (새 버전이 바로 보임), 끊겼을 때만 받아 둔 것
// 데이터·프로젝트는 여기를 지나지 않음 (브라우저 저장소·사용자가 고른 파일). 다른 곳(구글 로그인 등)은 건드리지 않음.
const CDN_CACHE = "dls-cdn-v1";
const APP_CACHE = "dls-app-v1";
const CDN_VERSIONED = [/^https:\/\/cdn\.jsdelivr\.net\/pyodide\/v\d/, /^https:\/\/cdn\.jsdelivr\.net\/npm\/[^/]+@\d/, /^https:\/\/cdn\.jsdelivr\.net\/npm\/@[^/]+\/[^/]+@\d/];

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith("dls-") && k !== CDN_CACHE && k !== APP_CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || req.headers.has("range")) return;
  const url = req.url;
  if (CDN_VERSIONED.some((re) => re.test(url))) {
    e.respondWith(caches.open(CDN_CACHE).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    }));
    return;
  }
  if (new URL(url).origin !== self.location.origin) return;
  e.respondWith((async () => {
    const c = await caches.open(APP_CACHE), key = url.split("?")[0];   // ?v=시각 꼴은 한 칸에 (쌓이지 않게)
    try {
      const res = await fetch(req);
      if (res.ok) c.put(key, res.clone());
      return res;
    } catch (err) {
      const hit = await c.match(key);
      if (hit) return hit;
      throw err;
    }
  })());
});

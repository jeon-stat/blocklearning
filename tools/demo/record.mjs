// 첫 화면 사용 영상 녹화 (1단계): 화면 없는 Chrome으로 사이트를 열어 실제로 눌러 가며 화면을 받아 둠.
//   node tools/demo/record.mjs <사이트 주소> <출력 폴더> [chrome 경로]
// 출력: frames/*.jpg (화면이 바뀔 때마다) + events.json (누른 곳·자막·빨리 감을 구간). 2단계는 compose.py.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const [URL_, OUT, CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"] = process.argv.slice(2);
const W = 1280, H = 800, PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "frames"), { recursive: true });
const profile = join(OUT, "profile");

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, `--window-size=${W},${H}`,
  "--no-first-run", "--no-default-browser-check", "--enable-unsafe-swiftshader", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });

let ws, nextId = 1;
const pending = new Map(), listeners = [];
async function connect() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) { ws = new WebSocket(page.webSocketDebuggerUrl); break; }
    } catch { /* 아직 안 뜸 */ }
    await sleep(200);
  }
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) { const { ok, bad } = pending.get(msg.id); pending.delete(msg.id); msg.error ? bad(new Error(msg.error.message)) : ok(msg.result); }
    else if (msg.method) listeners.forEach((f) => f(msg));
  };
}
const send = (method, params = {}) => new Promise((ok, bad) => { const id = nextId++; pending.set(id, { ok, bad }); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
const waitFor = async (expr, ms = 120000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await evaluate(expr)) return; await sleep(150); }
  throw new Error("기다리다 시간 초과: " + expr);
};

const events = { clicks: [], captions: [], fast: [], size: [W, H] };
let frames = 0;
const now = () => Date.now() / 1000;

// 화면의 요소를 보이게 하고 가운데 좌표를 얻어 진짜 마우스로 누름 (영상에는 compose.py가 커서를 그림)
async function click(selector, { hold = 700 } = {}) {
  const pos = await evaluate(`(async () => {
    const el = ${selector};
    if (!el) return null;
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    let b = el.getBoundingClientRect(), same = 0;
    for (let i = 0; i < 60 && same < 3; i++) {             // 스크롤이 멈출 때까지 (움직이는 중에 재면 엉뚱한 곳을 누름)
      await new Promise((r) => setTimeout(r, 60));
      const c = el.getBoundingClientRect();
      same = c.top === b.top ? same + 1 : 0;
      b = c;
    }
    return [b.left + b.width / 2, b.top + b.height / 2];
  })()`);
  if (!pos) throw new Error("못 찾음: " + selector);
  const [x, y] = pos;
  events.clicks.push({ t: now() + hold / 1000, x, y });
  await sleep(hold);                                         // 이동하는 동안 (커서가 그 자리로 감)
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
}
const caption = (text) => events.captions.push({ t: now(), text });
const byText = (sel, text) => `[...document.querySelectorAll(${JSON.stringify(sel)})].find((e) => e.offsetParent && e.textContent.includes(${JSON.stringify(text)}))`;
const scrollTo = (expr) => evaluate(`(async () => {                // 위쪽 자막(약 70px)에 가리지 않게 조금 아래로
  const el = ${expr}; if (!el) return;
  window.scrollBy({ top: el.getBoundingClientRect().top - 84, behavior: "smooth" });
  await new Promise((r) => setTimeout(r, 900));
})()`);

try {
  await connect();
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  listeners.push(async (m) => {
    if (m.method !== "Page.screencastFrame") return;
    const { data, metadata, sessionId } = m.params;
    const name = String(frames++).padStart(5, "0") + ".jpg";
    writeFileSync(join(OUT, "frames", name), Buffer.from(data, "base64"));
    (events.frames ||= []).push({ t: metadata.timestamp, file: name });
    send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  });
  await send("Page.navigate", { url: URL_ });
  await waitFor(`!!document.getElementById("home-new") && !!document.querySelector("#families .tcard")`);   // 과제 목록까지 (Pyodide 준비)
  await evaluate(`document.querySelectorAll(".hero-video").forEach((e) => e.remove())`);   // 영상 안에 영상이 보이지 않게
  await sleep(800);
  await send("Page.startScreencast", { format: "jpeg", quality: 85, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
  await sleep(1500);

  caption("새 프로젝트");
  await click(`document.getElementById("home-new")`);
  await sleep(1200);
  caption("① 과제 고르기 — 분류 · 시계열");
  await click(`document.querySelector('[data-f="classification"]')`);
  await sleep(700);
  await click(`document.querySelector('[data-s="cls_ts"]')`);
  await sleep(900);
  await click(`document.querySelector('#templates [data-t]')`);
  await sleep(900);
  await click(`document.querySelector('#template-box .go')`);
  await sleep(1000);

  caption("② 데이터 — 예시 데이터로");
  await click(`document.querySelector('[data-ex="ecg"]')`);
  await waitFor(`!document.getElementById("data-summary").hidden && document.getElementById("data-cards").textContent.length > 0`);
  await sleep(1200);
  await scrollTo(`document.getElementById("data-preview")`);
  await sleep(1800);
  await click(`document.getElementById("btn-next-data")`);
  await sleep(1200);

  caption("③ 모델 — 블록으로 쌓은 추천 모델");
  await sleep(2600);
  await click(`document.querySelector('#step-model .go[data-go="train"]')`);
  await sleep(1000);

  caption("④ 학습 — Epochs 15");
  await click(`document.getElementById("epochs")`, { hold: 500 });
  await evaluate(`(() => { const e = document.getElementById("epochs"); e.select(); })()`);
  await send("Input.insertText", { text: "15" });
  await sleep(500);
  await click(`document.getElementById("btn-train")`);
  const f0 = now();
  await waitFor(`/끝남|멈춤/.test(document.getElementById("train-status").textContent)`, 600000);
  events.fast.push({ t0: f0 + 1.5, t1: now() - 0.5 });          // 학습 중간은 빨리 감기
  await sleep(1500);
  await scrollTo(`document.querySelector("#step-train .train-layout")`);
  await sleep(3000);

  caption("⑤ 예측 — 새 데이터로");
  await click(`document.querySelector('#step-train .go[data-go="predict"]')`);
  await sleep(900);
  await click(`document.getElementById("btn-predict-example")`);
  await waitFor(`!document.getElementById("predict-result").hidden`);
  await sleep(1200);
  await scrollTo(`document.getElementById("predict-result")`);
  events.poster = now();
  await sleep(3500);
  await send("Page.stopScreencast");
  await sleep(300);
  events.end = now();
} finally {
  writeFileSync(join(OUT, "events.json"), JSON.stringify(events));
  try { ws && ws.close(); } catch { /* 닫힘 */ }
  chrome.kill();
}
console.log(`frames ${frames}`);

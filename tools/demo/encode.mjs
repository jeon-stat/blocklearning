// 첫 화면 사용 영상 (3단계): compose.py가 만든 그림들을 화면 없는 Chrome의 MediaRecorder로 WebM(VP9)으로.
//   node tools/demo/encode.mjs <record 출력 폴더> <출력 .webm> [초당 비트] [chrome 경로]
// 그림을 하나씩 보내 canvas에 그리고, fps 간격으로 한 장씩 녹화함 (MediaRecorder는 실제 시간으로 녹화하므로 영상 길이만큼 걸림).
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const [DIR, OUT, BPS = "900000", CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(join(DIR, "video.json"), "utf8"));
const [W, H] = meta.size, PORT = 9334;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// MediaRecorder의 WebM에는 길이(Duration)가 없어 재생바로 옮겨 가기가 안 됨 → Segment Info에 Duration(ms, float64)을 넣음.
// (MediaRecorder 출력은 Segment 크기가 '모름'이고 SeekHead·Cues가 없어, Info 크기만 고치면 됨)
function withDuration(buf, ms) {
  const vint = (p) => { const b = buf[p]; let len = 1; while (len <= 8 && !(b & (0x80 >> (len - 1)))) len++; let v = b & (0xff >> len); for (let i = 1; i < len; i++) v = v * 256 + buf[p + i]; return [v, len]; };
  const idLen = (p) => { const b = buf[p]; return b & 0x80 ? 1 : b & 0x40 ? 2 : b & 0x20 ? 3 : 4; };
  let p = 0;
  const [ebmlSize, el] = vint(p + 4); p += 4 + el + ebmlSize;          // EBML 머리
  if (buf.readUInt32BE(p) !== 0x18538067) throw new Error("Segment 없음");
  p += 4 + vint(p + 4)[1];                                               // Segment 안으로
  while (p < buf.length) {
    const il = idLen(p), id = buf.readUIntBE(p, il), [size, sl] = vint(p + il);
    if (id === 0x1549a966) {                                             // Info
      const dur = Buffer.alloc(11); dur.writeUInt16BE(0x4489, 0); dur[2] = 0x88; dur.writeDoubleBE(ms, 3);
      const body = Buffer.concat([buf.subarray(p + il + sl, p + il + sl + size), dur]);
      const head = Buffer.alloc(8); head[0] = 0x01; head.writeUIntBE(body.length, 2, 6);   // 크기를 8바이트 vint로
      return Buffer.concat([buf.subarray(0, p + il), head, body, buf.subarray(p + il + sl + size)]);
    }
    p += il + sl + size;
  }
  throw new Error("Info 없음");
}
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "dls-enc-"))}`,
  "--no-first-run", "--autoplay-policy=no-user-gesture-required", "about:blank"], { stdio: "ignore" });

let ws, nextId = 1;
const pending = new Map();
for (let i = 0; i < 50 && !ws; i++) {
  try {
    const page = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page");
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch { /* 아직 안 뜸 */ }
  if (!ws) await sleep(200);
}
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { const { ok, bad } = pending.get(msg.id); pending.delete(msg.id); msg.error ? bad(new Error(msg.error.message)) : ok(msg.result); }
};
const send = (method, params = {}) => new Promise((ok, bad) => { const id = nextId++; pending.set(id, { ok, bad }); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};

try {
  await evaluate(`(() => {
    const c = document.createElement("canvas"); c.width = ${W}; c.height = ${H}; document.body.appendChild(c);
    const g = c.getContext("2d"), stream = c.captureStream(0), track = stream.getVideoTracks()[0];
    const rec = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9", videoBitsPerSecond: ${Number(BPS)} });
    const parts = []; rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
    window.enc = {
      start() { rec.start(1000); },
      async draw(b64) {
        const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
        const bmp = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
        g.drawImage(bmp, 0, 0); bmp.close(); track.requestFrame();
      },
      stop() { return new Promise((r) => { rec.onstop = async () => {
        const buf = new Uint8Array(await new Blob(parts, { type: "video/webm" }).arrayBuffer());
        let s = ""; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
        r(btoa(s)); }; rec.stop(); }); },
    };
  })()`);
  const step = 1000 / meta.fps;
  await evaluate(`enc.draw(${JSON.stringify(readFileSync(join(DIR, "video", "00000.jpg")).toString("base64"))})`);
  await evaluate("enc.start()");
  const t0 = Date.now();
  for (let i = 0; i < meta.count; i++) {
    const wait = t0 + i * step - Date.now();
    if (wait > 0) await sleep(wait);
    await evaluate(`enc.draw(${JSON.stringify(readFileSync(join(DIR, "video", String(i).padStart(5, "0") + ".jpg")).toString("base64"))})`);
  }
  await sleep(step);
  const b64 = await evaluate("enc.stop()");
  const webm = withDuration(Buffer.from(b64, "base64"), (meta.count / meta.fps) * 1000);
  writeFileSync(OUT, webm);
  console.log(`${OUT} ${(webm.length / 1e6).toFixed(2)} MB`);
} finally {
  ws.close();
  chrome.kill();
}

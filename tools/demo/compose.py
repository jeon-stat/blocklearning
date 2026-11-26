"""첫 화면 사용 영상 (2단계): record.mjs가 받은 화면에 커서·누른 표시·단계 자막을 그려 일정한 간격(fps)의 그림으로.

    python tools/demo/compose.py <record 출력 폴더> [fps] [빨리 감기 배수]

출력: <폴더>/video/*.jpg, video.json (fps·장 수), poster.jpg (④ 학습 결과 화면). 3단계는 encode.mjs (WebM으로).
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/malgunbd.ttf"


def main(folder: str, fps: int = 15, fast: float = 6.0) -> None:
    d = Path(folder)
    ev = json.loads((d / "events.json").read_text(encoding="utf-8"))
    frames = sorted(ev["frames"], key=lambda f: f["t"])
    W, H = ev["size"]
    t_start, t_end = frames[0]["t"], ev.get("end", frames[-1]["t"] + 1)
    segs = [(f["t0"], f["t1"]) for f in ev.get("fast", [])]

    # 실제 시각 → 영상 시각 (빨리 감는 구간은 fast배 빠르게): 영상 시각마다 실제 시각을 거꾸로 구함
    def speed(r):
        return fast if any(a <= r < b for a, b in segs) else 1.0
    real_at, r, v, dt = [], t_start, 0.0, 1.0 / fps
    while r < t_end:
        real_at.append(r)
        step = dt
        while step > 1e-9 and r < t_end:                      # 구간 경계를 넘는 걸음은 나눠서
            s = speed(r)
            nxt = min([b for a, b in segs if a <= r < b] + [a for a, b in segs if a > r] + [t_end])
            use = min(step, (nxt - r) / s) if nxt > r else step
            r += use * s
            step -= use
        v += dt

    font = ImageFont.truetype(FONT, 22)
    clicks = ev["clicks"]
    caps = sorted(ev["captions"], key=lambda c: c["t"])

    def cursor_at(t):
        prev = (W * 0.55, H * 0.6)
        for c in clicks:
            if t < c["t"] - 0.6:
                return prev
            if t < c["t"]:
                k = (t - (c["t"] - 0.6)) / 0.6
                k = k * k * (3 - 2 * k)                        # 천천히 출발·도착
                return (prev[0] + (c["x"] - prev[0]) * k, prev[1] + (c["y"] - prev[1]) * k)
            prev = (c["x"], c["y"])
        return prev

    def draw_cursor(dr, x, y):
        pts = [(0, 0), (0, 22), (6, 17), (10, 26), (14, 24), (10, 15), (17, 15)]
        dr.polygon([(x + a, y + b) for a, b in pts], fill="white", outline="black", width=2)

    out = d / "video"
    out.mkdir(exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    fi, cur, poster_t = 0, None, None
    if len(caps) >= 2:
        poster_t = caps[-1]["t"] - 0.2                          # ⑤로 넘어가기 바로 전 (④ 학습 결과): 커서·자막 없는 화면
    for n, t in enumerate(real_at):
        while fi + 1 < len(frames) and frames[fi + 1]["t"] <= t:
            fi += 1
        if cur is None or cur[0] != fi:
            cur = (fi, Image.open(d / "frames" / frames[fi]["file"]).convert("RGB"))
        im = cur[1].copy()
        if im.size != (W, H):
            im = im.resize((W, H))
        dr = ImageDraw.Draw(im, "RGBA")
        for c in clicks:                                        # 누른 표시: 퍼지며 옅어지는 동그라미
            a = t - c["t"]
            if 0 <= a < 0.45:
                rr = 10 + 26 * (a / 0.45)
                dr.ellipse([c["x"] - rr, c["y"] - rr, c["x"] + rr, c["y"] + rr], outline=(79, 70, 229, int(220 * (1 - a / 0.45))), width=4)
        cap = next((c for c in reversed(caps) if c["t"] <= t), None)
        if cap:
            text = cap["text"] + ("   ▶▶ 빨리 감기" if speed(t) > 1 else "")
            tw = dr.textlength(text, font=font)
            x0, y0 = (W - tw) / 2 - 22, 18
            alpha = min(1.0, (t - cap["t"]) / 0.3)
            dr.rounded_rectangle([x0, y0, x0 + tw + 44, y0 + 46], radius=23, fill=(15, 23, 42, int(215 * alpha)))
            dr.text((x0 + 22, y0 + 9), text, font=font, fill=(255, 255, 255, int(255 * alpha)))
        cx, cy = cursor_at(t)
        draw_cursor(dr, cx, cy)
        im.save(out / f"{n:05d}.jpg", quality=88)
        if poster_t is not None and t <= poster_t:
            poster = cur[1]
    poster.save(d / "poster.jpg", quality=82)
    (d / "video.json").write_text(json.dumps({"fps": fps, "count": len(real_at), "size": [W, H]}), encoding="utf-8")
    print(f"{len(real_at)}장 · {len(real_at) / fps:.1f}초")


if __name__ == "__main__":
    main(sys.argv[1], *(int(sys.argv[2]),) if len(sys.argv) > 2 else (), *(float(sys.argv[3]),) if len(sys.argv) > 3 else ())

"""예시 데이터 만들기: 공개 실제 데이터의 원본 → 작게 줄인 npz (src/dlstudio/data/). 원본은 저장소에 넣지 않는다.

    python tools/build_examples.py <원본 폴더> [이름 …]

원본 폴더 구조 (내려받은 그대로 풀어 둠)
    mitdb/        PhysioNet MIT-BIH Arrhythmia Database 1.0.0 (*.hea, *.dat, *.atr)          ODC-By 1.0
    bearing/      Univ. of Ottawa bearing vibration, Mendeley v43hmbwxpm v2 (?-?-1.mat)        CC BY 4.0
    steel_plates/ UCI Steel Plates Faults (Faults.NNA)                                        CC BY 4.0
    hydraulic/    UCI Condition monitoring of hydraulic systems (PS1.txt …, profile.txt)       CC BY 4.0
    concrete/     UCI Concrete Compressive Strength (data.csv)                                CC BY 4.0
    steel_energy/ UCI Steel Industry Energy Consumption (Steel_industry_data.csv)             CC BY 4.0
    steelblast/   SteelBlastQC, DataverseNL doi:10.34894/EKZNN0 (*.png, annotations.csv)      CC BY 4.0
    pcb/          PCB-Defect, Mendeley vdj74sngvn v2 (PCB_Defect.zip)                          CC BY 4.0
    optdigits/    UCI Optical Recognition of Handwritten Digits (optdigits-orig.* 풀어 둔 것)    CC BY 4.0
출처·바꾼 내용은 이름마다 <이름>.SOURCE.txt로 함께 씀.
"""
from __future__ import annotations

import csv
import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "dlstudio" / "data"
RNG = np.random.default_rng(0)


def save(name: str, source: str, **arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    (OUT / f"{name}.SOURCE.txt").write_text(source.strip() + "\n", encoding="utf-8")
    size = (OUT / f"{name}.npz").stat().st_size
    print(f"{name}.npz  {size / 1e6:.2f} MB  " + ", ".join(f"{k} {v.shape} {v.dtype}" for k, v in arrays.items()))


# ---------------------------------------------------------------- 심전도 (MIT-BIH)

BEAT_NORMAL = {1}                                    # N
BEAT_ABNORMAL = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}      # L R a V F J A S E j (페이스메이커 박동 /·f·Q는 뺌)


def _mit_signal(rec: Path) -> tuple:
    head = rec.with_suffix(".hea").read_text().split("\n")
    fs = float(head[0].split()[2])
    parts = head[1].split()
    gain = float(parts[2].split("(")[0].split("/")[0]) or 200.0
    raw = np.frombuffer(rec.with_suffix(".dat").read_bytes(), dtype=np.uint8)
    raw = raw[: len(raw) // 3 * 3].reshape(-1, 3).astype(np.int32)
    s1 = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    s1 = np.where(s1 > 2047, s1 - 4096, s1)              # 형식 212: 3바이트에 12비트 두 개 — 첫 채널(MLII)만
    return s1.astype(np.float64) / gain, fs


def _mit_ann(rec: Path) -> list:
    b = rec.with_suffix(".atr").read_bytes()
    out, t, i = [], 0, 0
    while i + 1 < len(b):
        w = b[i] | (b[i + 1] << 8)
        a, n = w >> 10, w & 0x3FF
        i += 2
        if a == 0 and n == 0:
            break
        if a == 59:                                      # SKIP: 다음 4바이트가 간격
            t += (b[i + 2] | (b[i + 3] << 8) | (b[i] << 16) | (b[i + 1] << 24))
            i += 4
        elif a == 63:                                    # AUX 글: n바이트 (짝수로 맞춤)
            i += n + (n & 1)
        elif a in (60, 61, 62):
            pass
        else:
            t += n
            out.append((t, a))
    return out


def build_ecg(raw: Path, per_class: int = 2500, length: int = 140):
    beats = {0: [], 1: []}
    for hea in sorted((raw / "mitdb").glob("*.hea")):
        rec = hea.with_suffix("")
        x, fs = _mit_signal(rec)
        pre, post = int(0.25 * fs), int(0.45 * fs)
        for t, a in _mit_ann(rec):
            k = 0 if a in BEAT_NORMAL else 1 if a in BEAT_ABNORMAL else None
            if k is None or t - pre < 0 or t + post >= len(x):
                continue
            seg = x[t - pre:t + post]
            seg = np.interp(np.linspace(0, len(seg) - 1, length), np.arange(len(seg)), seg)
            beats[k].append(seg - np.median(seg))        # 기준선(직류 성분) 빼기
    X, y = [], []
    for k in (0, 1):
        b = np.asarray(beats[k])
        pick = RNG.choice(len(b), size=min(per_class, len(b)), replace=False)
        X.append(b[pick])
        y += [k] * len(pick)
    X, y = np.concatenate(X), np.asarray(y, np.int8)
    order = RNG.permutation(len(y))
    save("ecg_mitdb", """
심전도 박동 (정상·이상) — MIT-BIH Arrhythmia Database (PhysioNet), Open Data Commons Attribution License v1.0
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (2001).
Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220 (2000). https://physionet.org/content/mitdb/1.0.0/
바꾼 내용: 기록 31개의 첫 채널(MLII)에서 전문가 주석 박동마다 R파 0.25초 앞 ~ 0.45초 뒤를 잘라 140점으로 다시 뽑고 중앙값을 뺌(mV).
정상 = N, 이상 = L·R·a·V·F·J·A·S·E·j (페이스메이커 박동 제외). 각 2,500개를 무작위로 고름. 진단용 아님.
""", X=X[order].astype(np.float16), y=y[order])


# ---------------------------------------------------------------- 베어링 진동 (오타와대)

BEARING = {"H": "정상", "I": "내륜 결함", "O": "외륜 결함", "B": "볼 결함", "C": "복합 결함"}


def build_bearing(raw: Path, per_class: int = 240, length: int = 512, decim: int = 20):
    from scipy.io import loadmat
    from scipy.signal import decimate
    X, y = [], []
    for k, code in enumerate(BEARING):
        files = sorted((raw / "bearing").glob(f"{code}-*-1.mat"))
        per_file = per_class // len(files)
        for f in files:
            m = loadmat(f)
            v = np.asarray(m["Channel_1"], dtype=np.float64).ravel()
            v = decimate(decimate(v, 4), decim // 4)    # 200 kHz → 10 kHz (거르고 줄임)
            starts = RNG.choice(len(v) - length, size=per_file, replace=False)
            X += [v[s:s + length] for s in starts]
            y += [k] * per_file
    X, y = np.asarray(X), np.asarray(y, np.int8)
    order = RNG.permutation(len(y))
    save("bearing_uottawa", """
베어링 진동 (정상·내륜·외륜·볼·복합 결함) — Huang H, Baddour N. Bearing Vibration Data under Time-varying Rotational Speed
Conditions. Mendeley Data, V2, doi:10.17632/v43hmbwxpm.2 (University of Ottawa), CC BY 4.0
Data in Brief 21:1745-1749 (2018), doi:10.1016/j.dib.2018.11.019
바꾼 내용: 상태 5종 × 속도 조건 4종의 첫 측정(?-?-1.mat)에서 가속도계 신호(Channel_1, 200 kHz)를 10 kHz로 줄이고(거름 후 솎음)
512점(0.05초) 구간을 무작위로 잘라 상태마다 240개.
""", X=X[order].astype(np.float16), y=y[order])


# ---------------------------------------------------------------- 유압 시스템

HYD_SENSORS = [("PS1", "압력 PS1 (bar)"), ("PS2", "압력 PS2 (bar)"), ("PS3", "압력 PS3 (bar)"), ("PS4", "압력 PS4 (bar)"),
               ("PS5", "압력 PS5 (bar)"), ("PS6", "압력 PS6 (bar)"), ("EPS1", "모터 전력 (W)"), ("FS1", "유량 FS1 (l/min)"),
               ("FS2", "유량 FS2 (l/min)"), ("TS1", "온도 TS1 (°C)"), ("TS2", "온도 TS2 (°C)"), ("TS3", "온도 TS3 (°C)"),
               ("TS4", "온도 TS4 (°C)"), ("VS1", "진동 (mm/s)"), ("CE", "냉각 효율 (%)"), ("CP", "냉각 동력 (kW)"),
               ("SE", "효율 계수 (%)")]


def build_hydraulic(raw: Path, ts_len: int = 200):
    d = raw / "hydraulic"
    sig = {k: np.loadtxt(d / f"{k}.txt") for k, _ in HYD_SENSORS}
    prof = np.loadtxt(d / "profile.txt")                  # 냉각기 %, 밸브 %, 펌프 누설 0–2, 어큐뮬레이터 bar, 안정 여부
    feats = np.stack([sig[k].mean(1) for k, _ in HYD_SENSORS], 1)
    ps2 = sig["PS2"][:, :2000].reshape(len(prof), ts_len, -1).mean(2)   # 100 Hz 앞 20초 2,000점 → 200점 (10점씩 평균)
    src = """
유압 시스템 상태 감시 — Helwig N, Pignanelli E, Schütze A. Condition monitoring of hydraulic systems.
UCI Machine Learning Repository (2015), doi:10.24432/C5CW21, CC BY 4.0
Helwig N, et al. Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics. I2MTC 2015.
바꾼 내용: 60초 작업 사이클 2,205개. 표: 센서 17개의 사이클 평균. 시계열: 압력 PS2 (100 Hz)의 앞 20초 2,000점을 10점씩 평균해 200점.
상태: profile.txt의 냉각기 효율(%)·밸브 상태(%)·펌프 누설(0–2)·어큐뮬레이터 압력(bar) 그대로.
"""
    save("hydraulic", src, X=feats.astype(np.float32), ps2=ps2.astype(np.float32), cooler=prof[:, 0].astype(np.float32),
         valve=prof[:, 1].astype(np.float32), pump=prof[:, 2].astype(np.float32), acc=prof[:, 3].astype(np.float32))


# ---------------------------------------------------------------- 콘크리트

def build_concrete(raw: Path):
    rows = list(csv.reader((raw / "concrete" / "data.csv").read_text(encoding="utf-8-sig").splitlines()))[1:]
    a = np.asarray([[float(v) for v in r] for r in rows if r])
    save("concrete", """
콘크리트 배합 → 압축강도 (MPa) — Yeh I-C. Concrete Compressive Strength. UCI Machine Learning Repository (1998),
doi:10.24432/C5PK67, CC BY 4.0. Yeh I-C. Cement and Concrete Research 28(12):1797-1808 (1998).
바꾼 내용: 없음 (열 이름만 한국어로: 시멘트·고로슬래그·플라이애시·물·감수제·굵은 골재·잔골재 kg/m³, 재령 일).
""", X=a[:, :8].astype(np.float32), y=a[:, 8].astype(np.float32))


# ---------------------------------------------------------------- 철강 공장 전력

def build_steel_energy(raw: Path):
    rows = list(csv.DictReader(io.StringIO((raw / "steel_energy" / "Steel_industry_data.csv").read_text(encoding="utf-8-sig"))))
    v = np.asarray([float(r["Usage_kWh"]) for r in rows])
    hourly = v[: len(v) // 4 * 4].reshape(-1, 4).sum(1)    # 15분 → 1시간 합
    save("steel_power", """
철강 공장 전력 사용량 (kWh, 1시간마다) — Sathishkumar V E, Shin C, Cho Y. Steel Industry Energy Consumption.
UCI Machine Learning Repository (2021), doi:10.24432/C52G8C, CC BY 4.0 (DAEWOO Steel Co. Ltd, 광양, 2018년)
바꾼 내용: 15분 간격 Usage_kWh를 1시간씩 더해 8,760점.
""", series=hourly.astype(np.float32))


# ---------------------------------------------------------------- 손글씨 숫자 32×32

def build_digits(raw: Path):
    X, y = [], []
    for f in ("optdigits-orig.tra", "optdigits-orig.cv", "optdigits-orig.windep"):
        lines = (raw / "optdigits" / f).read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if len(l.strip()) == 32 and set(l.strip()) <= {"0", "1"})
        while i + 32 < len(lines):
            block = lines[i:i + 32]
            if not all(len(l.strip()) == 32 for l in block):
                break
            X.append([[int(c) for c in l.strip()] for l in block])
            y.append(int(lines[i + 32].strip()))
            i += 33
    X, y = np.asarray(X, np.uint8), np.asarray(y, np.int8)
    save("digits32", """
손글씨 숫자 0–9 (32×32 흑백) — Alpaydin E, Kaynak C. Optical Recognition of Handwritten Digits.
UCI Machine Learning Repository (1998), doi:10.24432/C50P49, CC BY 4.0
바꾼 내용: optdigits-orig.tra·cv·windep의 32×32 비트맵(0·1)을 그대로 모음 (wdep는 같은 사람 글씨라 뺌).
""", X=X, y=y)


# ---------------------------------------------------------------- 강판 표면 (SteelBlastQC)

def build_steelblast(raw: Path, size: int = 64):
    from PIL import Image
    d = raw / "steelblast"
    base = next(p.parent for p in d.rglob("annotations.csv"))
    ann = list(csv.DictReader((base / "annotations.csv").read_text(encoding="utf-8-sig").splitlines()))
    X, y = [], []
    for r in ann:                                         # file_path: train/not-good/slice035_011.png, label: 0 양호 · 1 불량
        f = base / r["file_path"]
        if not f.exists():                                # annotations.csv에는 있지만 공개 파일에 없는 그림 (96장)은 건너뜀
            continue
        with Image.open(f) as im:
            X.append(np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), np.uint8))
        y.append(int(r["label"]))
    save("steelblast", f"""
쇼트 블라스팅한 강판 표면 (양호·불량) — SteelBlastQC Dataset, DataverseNL doi:10.34894/EKZNN0, CC BY 4.0
Ruzavina I, et al. SteelBlastQC: Shot-blasted Steel Surface Dataset with Interpretable Detection of Surface Defects. IJCNN 2025 (arXiv:2504.20510).
바꾼 내용: 공개된 512×512 컬러 {len(X):,}장을 흑백 {size}×{size}로 줄임. 정답: good → 양호, not-good(용접선·긁힘·부식·변색) → 불량.
""", X=np.asarray(X, np.uint8), y=np.asarray(y, np.int8))


# ---------------------------------------------------------------- 강판 결함 7종 (UCI Steel Plates Faults)

PLATE_FEATURES = ["X 최소", "X 최대", "Y 최소", "Y 최대", "면적 (픽셀)", "X 둘레", "Y 둘레", "밝기 합", "최소 밝기", "최대 밝기",
                  "컨베이어 길이", "강종 A300", "강종 A400", "판 두께", "가장자리 지수", "빈 공간 지수", "정사각 지수",
                  "바깥 X 지수", "가장자리 X 지수", "가장자리 Y 지수", "바깥 전체 지수", "면적 로그", "X 로그 지수",
                  "Y 로그 지수", "방향 지수", "밝기 지수", "면적 시그모이드"]
PLATE_FAULTS = ["페이스트리", "Z 긁힘", "K 긁힘", "얼룩", "오염", "돌기", "기타 결함"]


def build_steel_plates(raw: Path):
    a = np.loadtxt(raw / "steel_plates" / "Faults.NNA")
    X, onehot = a[:, :27], a[:, 27:34]
    save("steel_plates", """
강판 결함 7종 (결함 영역의 모양·밝기 특징 27개) — Buscema M, Terzi S, Tastle W. Steel Plates Faults.
UCI Machine Learning Repository (2010), doi:10.24432/C5J88N, CC BY 4.0 (Semeion, Research Center of Sciences of Communication)
바꾼 내용: 없음 (열 이름만 한국어로). 정답: Pastry·Z_Scratch·K_Scatch·Stains·Dirtiness·Bumps·Other_Faults →
페이스트리·Z 긁힘·K 긁힘·얼룩·오염·돌기·기타 결함.
""", X=X.astype(np.float32), y=onehot.argmax(1).astype(np.int8))


# ---------------------------------------------------------------- PCB 결함 탐지

PCB_NAMES = ["패드 누락", "쥐 물림", "회로 끊김", "단락", "돌기", "잔여 구리"]


def build_pcb(raw: Path, size: int = 64, negatives: int = 300):
    import json
    from PIL import Image
    z = zipfile.ZipFile(raw / "pcb" / "PCB_Defect.zip")
    coco = json.loads(z.read("PCB_Defect/annotation/_annotations.coco.json"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    X, B = [], []

    def crop(im, bxs, x0, y0, S):
        g = im.crop((x0, y0, x0 + S, y0 + S)).resize((size, size), Image.LANCZOS)
        out = []
        for a in bxs:                                       # 자른 칸 안에 반 이상 들어온 결함만 (밖은 잘라냄)
            bx, by, bw, bh = a["bbox"]
            cx0, cy0, cx1, cy1 = max(bx, x0), max(by, y0), min(bx + bw, x0 + S), min(by + bh, y0 + S)
            if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.5 * bw * bh:
                continue
            out.append([a["category_id"] - 1, (cx0 - x0) / S, (cy0 - y0) / S, (cx1 - cx0) / S, (cy1 - cy0) / S])
        return np.asarray(g, np.uint8), out

    for info in coco["images"]:
        with Image.open(io.BytesIO(z.read("PCB_Defect/images/" + info["file_name"]))) as im0:
            im = im0.convert("L")
        W, H = im.size
        anns = by_img.get(info["id"], [])
        for a in anns:                                      # 결함마다 하나: 결함 크기의 2.5–4배 칸을 결함 둘레에서 무작위로
            bx, by, bw, bh = a["bbox"]
            S = int(np.clip(max(bw, bh) * RNG.uniform(2.5, 4.0), 160, min(W, H)))
            x0 = int(np.clip(bx + bw / 2 - S * RNG.uniform(0.25, 0.75), 0, W - S))
            y0 = int(np.clip(by + bh / 2 - S * RNG.uniform(0.25, 0.75), 0, H - S))
            img, bxs = crop(im, anns, x0, y0, S)
            if bxs:
                X.append(img)
                B.append(bxs)
        for _ in range(negatives // len(coco["images"]) + 1):   # 결함 없는 칸 (찾을 것이 없는 그림도 배움)
            S = int(RNG.uniform(160, 400))
            x0, y0 = int(RNG.uniform(0, W - S)), int(RNG.uniform(0, H - S))
            if not any(a["bbox"][0] < x0 + S and a["bbox"][0] + a["bbox"][2] > x0 and a["bbox"][1] < y0 + S
                       and a["bbox"][1] + a["bbox"][3] > y0 for a in anns):
                X.append(crop(im, [], x0, y0, S)[0])
                B.append([])
    order = RNG.permutation(len(X))
    X = np.asarray(X, np.uint8)[order]
    boxes = np.asarray([[i] + b for i, k in enumerate(order) for b in B[k]], np.float32)
    save("pcb_det", f"""
PCB 결함 위치 (6종) — PCB-Defect: An Annotated Dataset for Surface Defect Detection in Printed Circuit Boards.
Mendeley Data V2, doi:10.17632/vdj74sngvn.2, CC BY 4.0. Data in Brief (2025), doi:10.1016/j.dib.2025.112296
바꾼 내용: 기판 사진 230장(Version 1, 가로세로 상자)에서 결함마다 결함 둘레를 결함 크기의 2.5–4배(160픽셀 이상) 정사각형으로 잘라
흑백 {size}×{size}로 줄이고, 그 안에 반 이상 들어온 결함의 상자를 0–1 비율로 옮김. 결함 없는 칸도 섞음.
클래스: missing_pad·mouse_bite·open_circuit·short·spur·spurious_copper → 패드 누락·쥐 물림·회로 끊김·단락·돌기·잔여 구리.
""", X=X, boxes=boxes, names=np.asarray(PCB_NAMES))
    print("  상자", len(boxes), "결함 없는 그림", sum(1 for b in B if not b))


BUILD = {"ecg": build_ecg, "bearing": build_bearing, "hydraulic": build_hydraulic,
         "concrete": build_concrete, "steel_energy": build_steel_energy, "digits": build_digits, "steelblast": build_steelblast,
         "steel_plates": build_steel_plates, "pcb": build_pcb}

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for name in sys.argv[2:] or BUILD:
        BUILD[name](root)

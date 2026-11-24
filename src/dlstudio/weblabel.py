"""웹 버전의 라벨링 작업: 브라우저 보관함(IndexedDB)에 '작업'(정답 JSON)과 '데이터 사본'(zip)으로 두고,
열 때 Pyodide 파일 시스템에 다시 풀어 LabelProject로 연다. 라벨링 페이지와 studio(② 라벨링한 데이터)가 함께 쓴다.

파일 자리: 고른 데이터는 /data/u<번호>/... (web.js의 writeFiles), 다시 푼 작업은 /data/L<번호>/...
작업 파일은 그 뿌리 바로 아래 _work.dllabel (데이터 폴더 밖이라 데이터로 읽히지 않음).
작업 JSON의 데이터 경로는 옮길 수 있게 "@data/<뿌리>/..."로 적어 둔다.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile

import numpy as np

from dlstudio.core import Studio, UserError, _clean
from dlstudio import web
from dlstudio.web import _wrap

WORK = "_work.dllabel"
OUT = "/out"                             # 내려받을 파일을 잠시 두는 곳 (시험은 바꿔 씀)
_MARK = "@data/"
_n = [0]


def _root(path: str) -> str:
    m = re.match(re.escape(web.DATA_DIR) + r"/([uL]\d+)(/|$)", str(path or ""))
    if not m:
        raise UserError("데이터 자리를 알 수 없음", "데이터를 다시 고르세요")
    return m.group(1)


def _new_root() -> str:
    while True:
        _n[0] += 1
        r = f"L{_n[0]}"
        if not os.path.exists(f"{web.DATA_DIR}/{r}"):
            return r


@_wrap
def new_job(studio: Studio, body_json: str) -> dict:
    """새 라벨링 작업: 고른 데이터(/data/u<번호>/...) 옆에 작업 파일을 두고 만듦."""
    b = json.loads(body_json)
    root = _root(b.get("path"))
    b["file"] = f"{web.DATA_DIR}/{root}/{WORK}"
    code, r = studio.call("POST", "/api/label/new", b)
    if code != 200:
        e = r["error"]
        raise UserError(e["message"], e.get("hint", ""), e.get("issues"))
    return r


def work_text(studio: Studio) -> str:
    """지금 작업의 정답 JSON (데이터 경로는 @data/...로)."""
    p = studio.label
    if p is None:
        raise UserError("라벨링 작업이 없음", "작업을 새로 만들거나 여세요")
    return json.dumps(_portable(p.meta, p.file), ensure_ascii=False)


def data_bytes(studio: Studio) -> bytes:
    """지금 작업의 데이터 사본 (뿌리 아래 파일 전부, 작업 파일·내보낸 파일 빼고) → zip 바이트. 데이터를 바꾸지 않으니 처음 한 번만."""
    p = studio.label
    root = _root(p.file)
    base = f"{web.DATA_DIR}/{root}"
    src = os.path.normpath(p.meta["source"]["path"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for dirpath, _, files in os.walk(base):
            for f in files:
                full = os.path.normpath(os.path.join(dirpath, f))
                if full == src or full.startswith(src + os.sep):        # 데이터 폴더(또는 표 파일)만
                    z.write(full, os.path.relpath(full, base).replace(os.sep, "/"))
    return buf.getvalue()


def summary(studio: Studio) -> str:
    """보관함 목록에 보일 한 줄 요약."""
    st = studio.label.state()
    c = st["counts"]
    kind = {"series_folder": "시계열", "image_folder": "이미지", "table": "표"}.get(st["source"]["kind"], "")
    schema = {"class": "클래스", "number": "숫자", "segments": "구간", "boxes": "상자"}.get(st["schema"], "")
    return json.dumps({"kind": st["source"]["kind"], "schema": st["schema"], "text": f"{kind} · {schema} 정답",
                       "total": c["total"], "labeled": c["labeled"], "unconfirmed": c["unconfirmed"],
                       "classes": [k["name"] for k in st["classes"]]}, ensure_ascii=False)


def _unpack(studio: Studio, data_path: str, work: str) -> str:
    """데이터 사본 zip과 정답 JSON을 새 자리(/data/L<번호>/)에 풀고 작업 파일 경로를 돌려줌."""
    root = _new_root()
    base = f"{web.DATA_DIR}/{root}"
    base_n = os.path.normpath(base)
    os.makedirs(base, exist_ok=True)
    if data_path:
        with zipfile.ZipFile(data_path) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                target = os.path.normpath(os.path.join(base, n))
                if not target.startswith(base_n + os.sep):
                    continue                                    # zip 안의 ../ 같은 경로는 무시
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(z.read(n))
    try:
        meta = json.loads(work)
        path = meta["source"]["path"]
    except (ValueError, KeyError, TypeError):
        raise UserError("라벨링 작업을 읽을 수 없음", "작업이 손상되었을 수 있습니다") from None
    m = re.match(re.escape(_MARK) + r"([uL]\d+)(/.*)?$", path)
    if not m:
        raise UserError("라벨링 작업의 데이터 자리를 알 수 없음", "작업을 다시 만드세요")
    meta["source"]["path"] = base + (m.group(2) or "")
    file = f"{base}/{WORK}"
    with open(file, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False))
    return file


@_wrap
def open_job(studio: Studio, data_path: str, work: str) -> dict:
    """보관함의 작업을 라벨링 페이지에서 엶."""
    file = _unpack(studio, data_path, work)
    code, r = studio.call("POST", "/api/label/open", {"file": file})
    if code != 200:
        e = r["error"]
        raise UserError(e["message"], e.get("hint", ""), e.get("issues"))
    return r


@_wrap
def use_job(studio: Studio, data_path: str, work: str, confirmed_only: bool = False) -> dict:
    """보관함의 작업을 studio 학습 데이터로: 풀고 → 정답 표를 만들어 → 데이터로 불러옴 (② 라벨링한 데이터)."""
    from dlstudio.labeling import LabelProject
    file = _unpack(studio, data_path, work)
    p = LabelProject.open(file)
    r = studio._label_use(p, bool(confirmed_only), None)
    return _clean({"summary": r["summary"], "source": r["source"]})


@_wrap
def export_file(studio: Studio, op: str, body_json: str) -> dict:
    """내보내기(정답 표·정상 목록·COCO·YOLO·나누기)를 /out/ 아래에 쓰고, 내려받을 파일 [(이름, 경로)]을 돌려줌."""
    b = json.loads(body_json or "{}")
    p = studio.label
    if p is None:
        raise UserError("라벨링 작업이 없음", "작업을 새로 만들거나 여세요")
    out = f"{OUT}/{_new_root()}"
    os.makedirs(out, exist_ok=True)
    name = str(b.get("name") or "labels").replace("/", "_")
    co = bool(b.get("confirmed_only"))
    if op == "export":
        f = f"{out}/{name}_정답.csv"
        n = p.export(f, co)
        return {"files": [[os.path.basename(f), f]], "rows": n}
    if op == "export_normals":
        f = f"{out}/{name}_정상목록.csv"
        return {"files": [[os.path.basename(f), f]], "rows": p.export_normals(f, co)}
    if op == "export_coco":
        f = f"{out}/{name}_coco.json"
        p.export_coco(f, co)
        return {"files": [[os.path.basename(f), f]]}
    if op == "export_yolo":
        d = f"{out}/{name}_yolo"
        n = p.export_yolo(d, co)
        z = f"{out}/{name}_yolo.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for dirpath, _, files in os.walk(d):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    zf.write(full, os.path.relpath(full, d).replace(os.sep, "/"))
        return {"files": [[os.path.basename(z), z]], "boxes": n}
    if op == "split":                                    # 나눈 작업마다 데이터 사본까지 든 묶음 (받는 사람이 그대로 열게)
        parts = p.split(int(b.get("n", 2)))
        data = data_bytes(studio)
        files = []
        for i, f in enumerate(parts):
            meta = json.loads(open(f, encoding="utf-8").read())
            os.remove(f)
            z = f"{out}/{name}_part{i + 1}.dllabel.zip"
            with open(z, "wb") as fh:
                fh.write(_bundle(_portable(meta, p.file), data))
            files.append([os.path.basename(z), z])
        return {"files": files}
    if op == "bundle":                                   # 작업 내려받기: 정답 + 데이터 사본 (다른 PC·다른 사람이 그대로 엶)
        z = f"{out}/{name}.dllabel.zip"
        with open(z, "wb") as fh:
            fh.write(bundle_bytes(studio))
        return {"files": [[os.path.basename(z), z]]}
    raise UserError(f"내보내기 '{op}'을(를) 모름", "")


def _portable(meta: dict, file: str) -> dict:
    root = _root(file)
    meta = json.loads(json.dumps(meta, ensure_ascii=False))
    meta["source"]["path"] = meta["source"]["path"].replace(f"{web.DATA_DIR}/{root}", _MARK + root, 1)
    return meta


def _bundle(meta: dict, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("work.json", json.dumps(meta, ensure_ascii=False))
        with zipfile.ZipFile(io.BytesIO(data)) as d:
            for n in d.namelist():
                z.writestr("data/" + n, d.read(n))
    return buf.getvalue()


def bundle_bytes(studio: Studio) -> bytes:
    """작업 묶음 (.dllabel.zip): work.json(정답, 옮길 수 있는 경로) + data/(데이터 사본)."""
    return _bundle(json.loads(work_text(studio)), data_bytes(studio))


def _read_bundle(path: str) -> tuple:
    """묶음 zip → (정답 JSON 글자, data/ 아래를 뽑은 zip 경로)."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise UserError("라벨링 작업 묶음(.dllabel.zip)이 아님", "라벨링 페이지에서 [작업 내려받기]로 받은 파일을 고르세요") from None
    with z:
        if "work.json" not in z.namelist():
            raise UserError("라벨링 작업 묶음이 아님 (work.json 없음)", "라벨링 페이지에서 [작업 내려받기]로 받은 파일을 고르세요")
        work = z.read("work.json").decode("utf-8")
        data = path + ".data.zip"
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_STORED) as d:
            for n in z.namelist():
                if n.startswith("data/") and not n.endswith("/"):
                    d.writestr(n[5:], z.read(n))
    return work, data


@_wrap
def open_bundle(studio: Studio, path: str) -> dict:
    """작업 묶음 가져오기 → 라벨링 페이지에서 엶 (보관함에는 부르는 쪽이 새로 저장)."""
    work, data = _read_bundle(path)
    file = _unpack(studio, data, work)
    code, r = studio.call("POST", "/api/label/open", {"file": file})
    if code != 200:
        e = r["error"]
        raise UserError(e["message"], e.get("hint", ""), e.get("issues"))
    return r


@_wrap
def merge_file(studio: Studio, path: str) -> dict:
    """다른 사람의 작업(묶음 .dllabel.zip 또는 .dllabel) 합치기: 데이터 자리를 지금 작업으로 맞춘 뒤 합침."""
    p = studio.label
    if p is None:
        raise UserError("라벨링 작업이 없음", "작업을 새로 만들거나 여세요")
    try:
        work = _read_bundle(path)[0] if zipfile.is_zipfile(path) else open(path, encoding="utf-8").read()
        other = json.loads(work)
        other["source"]["path"] = p.meta["source"]["path"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        raise UserError("다른 사람의 작업을 읽을 수 없음", "라벨링 작업 묶음(.dllabel.zip)이나 작업 파일(.dllabel)을 고르세요") from None
    tmp = f"{OUT}/merge_{_new_root()}.dllabel"
    os.makedirs(OUT, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(other, ensure_ascii=False))
    r = p.merge(tmp)
    return _clean(dict(p.state(), **r))


def file_path(studio: Studio, name: str) -> str:
    """그림 작업의 원본 그림 경로 (크게 보기: 브라우저가 이 파일을 읽어 보여 줌)."""
    if studio.label is None:
        raise UserError("라벨링 작업이 없음", "")
    return str(studio.label.file_path(name))


# ------------------------------------------------------------ 딥러닝 제안 (웹: 학습은 화면의 엔진이, 데이터 준비·적용은 여기서)
_deep = {}


@_wrap
def deep_prepare(studio: Studio) -> dict:
    """정답 붙은 샘플로 학습할 데이터와 계획을 준비 (엔진이 deep_payload로 값을 가져감)."""
    from dlstudio import codegen
    from dlstudio.core import deep_graph, graph_name
    p = studio.label
    if p is None:
        raise UserError("라벨링 작업이 없음", "작업을 새로 만들거나 여세요")
    prep = p.deep_data()
    d = prep["data"]
    _deep.clear()
    _deep.update(prep)
    plan = codegen.plan_dict(deep_graph(prep["kind"]), d.length, d.n_out, d.image)
    return {"plan": plan, "model": graph_name(prep["kind"]), "n_pred": len(prep["targets"]),
            "classes": [str(c) for c in d.classes]}


def deep_payload() -> dict:
    """엔진 학습용 값: web.data_payload와 같은 모양 + 제안할 값 X_pred."""
    d = _deep["data"]
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).tobytes()
    return {"X_train": f32(d.X_train), "X_test": f32(d.X_test), "y_train": f32(d.y_train), "y_test": f32(d.y_test),
            "X_pred": f32(_deep["X_pred"]),
            "meta": json.dumps({"n_train": int(len(d.y_train)), "n_test": int(len(d.y_test)), "n_features": int(d.X_train.shape[1]),
                                "task": d.task, "n_out": int(d.n_out), "length": int(d.length),
                                "image": list(d.image) if d.image else None, "n_pred": len(_deep["targets"])})}


@_wrap
def deep_apply(studio: Studio, probs_bytes, acc) -> dict:
    """엔진이 낸 확률 (제안할 샘플 수 × 클래스 수) → 모델 제안으로 붙임."""
    raw = probs_bytes.to_bytes() if hasattr(probs_bytes, "to_bytes") else bytes(probs_bytes)
    d = _deep["data"]
    P = np.frombuffer(raw, dtype=np.float32).reshape(len(_deep["targets"]), -1)
    r = studio.label.apply_model(_deep["targets"], list(d.classes), P)
    trained = int(len(d.y_train) + len(d.y_test))
    _deep.clear()
    return _clean(dict(studio.label.state(), **r, acc=None if acc is None else float(acc), trained_on=trained))

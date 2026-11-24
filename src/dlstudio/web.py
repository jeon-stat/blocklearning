"""웹 버전(GitHub Pages)에서 Pyodide로 부르는 Python 쪽.

화면의 다리(static/web.js)가 이 모듈을 부른다. API는 core.Studio(web=True)를 그대로 쓰고, torch가 필요한 일
(학습·샘플 따라가기·모델 정보)은 화면의 TensorFlow.js 엔진이 맡는다. 여기서는 엔진에 넘길 계획·데이터와
가중치를 담은 파일(.npz, 프로젝트)을 만든다. 데이터는 브라우저 안에서만 쓰고 밖으로 보내지 않는다.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile

import numpy as np

from dlstudio import codegen
from dlstudio.core import Studio, UserError, _clean, _graph_from, _jsonable
from dlstudio.graph import check_graph


class WebStudio(Studio):
    def __init__(self):
        super().__init__(web=True)

    def api_model_check(self, b: dict) -> dict:
        """torch 없이: 모양·문제는 그대로, '모델 정보'는 codegen.plan_dict의 파라미터 수에서."""
        d = self.need_data()
        graph = _graph_from(b)
        c = check_graph(graph, d.length, d.n_out, image=d.image)
        out = dict(c.to_dict(), graph=graph, panel=None)
        if isinstance(graph, dict):
            self.graph = graph
        if c.ok:
            plan = codegen.plan_dict(graph, d.length, d.n_out, d.image)
            out["panel"] = {"ok": True, "params": plan["params"], "mb": plan["params"] * 4 / 1e6, "message": "", "warnings": []}
        return out

    def api_data_load(self, b: dict) -> dict:
        out = super().api_data_load(b)
        self.model_graph = None                                 # 새 데이터: 화면의 학습 결과는 버림
        return out


def call_json(studio: Studio, method: str, path: str, body_json: str) -> str:
    """화면의 api(path, body) 한 번: JSON 글자로 받고 JSON 글자 {"status", "body"}로 돌려줌."""
    code, obj = studio.call(method, path, json.loads(body_json) if body_json else None)
    return json.dumps({"status": code, "body": _clean(obj)}, ensure_ascii=False, default=_jsonable)


def _wrap(fn):
    """UserError·DataError를 화면 오류 모양 JSON으로 (다리가 status를 보고 throw)."""
    def run(*a):
        from dlstudio.data import DataError
        try:
            return json.dumps({"status": 200, "body": fn(*a)}, ensure_ascii=False, default=_jsonable)
        except UserError as e:
            return json.dumps({"status": 400, "body": {"error": {"message": e.message, "hint": e.hint, "issues": e.issues}}},
                              ensure_ascii=False, default=_jsonable)
        except DataError as e:
            i = e.issues[0]
            return json.dumps({"status": 400, "body": {"error": {"message": i.message, "hint": i.hint,
                                                                 "issues": [x.to_dict() for x in e.issues]}}}, ensure_ascii=False)
    return run


@_wrap
def plan(studio: Studio, graph_json: str) -> dict:
    """학습할 그래프 → 엔진용 계획 (문제가 있으면 오류)."""
    d = studio.need_data()
    graph = json.loads(graph_json)
    c = check_graph(graph, d.length, d.n_out, image=d.image)
    if not c.ok:
        i = c.issues[0]
        raise UserError(i.message, i.hint, [x.to_dict() for x in c.issues])
    return codegen.plan_dict(graph, d.length, d.n_out, d.image)


def data_payload(studio: Studio) -> dict:
    """엔진에 넘길 데이터: 정규화한 X(float32 바이트), 정답, 모양·과제 정보."""
    d = studio.need_data()
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).tobytes()
    return {"X_train": f32(d.X_train), "X_test": f32(d.X_test), "y_train": f32(d.y_train), "y_test": f32(d.y_test),
            "meta": json.dumps({"n_train": int(len(d.y_train)), "n_test": int(len(d.y_test)), "n_features": int(d.X_train.shape[1]),
                                "task": d.task, "n_out": int(d.n_out), "length": int(d.length),
                                "image": list(d.image) if d.image else None,
                                "classes": [str(c) for c in d.classes] if d.task == "classification" else []})}


def set_trained(studio: Studio, graph_json: str, settings_json: str, results_json: str) -> None:
    """화면에서 학습이 끝났음을 기록 (코드·프로젝트 저장이 학습한 그래프를 알게)."""
    studio.model_graph = json.loads(graph_json)
    studio.train_settings = json.loads(settings_json)
    studio.results = json.loads(results_json)


@_wrap
def sample(studio: Studio, i: int) -> dict:
    """샘플 따라가기의 데이터 쪽: 시험 샘플 i의 정규화한 값, 원래 값, 정답, 노드 순서."""
    d = studio.need_data()
    if studio.model_graph is None:
        raise UserError("학습된 모델이 없음", "④ 학습 단계에서 먼저 학습하세요")
    if not 0 <= i < len(d.y_test):
        raise UserError(f"시험 샘플 번호 {i}이(가) 없음", f"0–{len(d.y_test) - 1} 사이로 고르세요")
    label = d.classes[int(d.y_test[i])] if d.task == "classification" else float(d.y_test[i])
    order = check_graph(studio.model_graph, d.length, d.n_out, image=d.image).order
    return {"x": d.X_test[i].astype(float).tolist(), "input": np.round(d.raw(d.X_test[i]), 5).tolist(),
            "label": label, "order": order, "classes": [str(c) for c in d.classes]}


def _arrays(state) -> dict:
    """화면에서 온 가중치 {이름: {shape, data(TypedArray), int}} → numpy."""
    state = state.to_py() if hasattr(state, "to_py") else state
    out = {}
    for k, v in state.items():
        raw = v["data"]
        raw = raw.to_bytes() if hasattr(raw, "to_bytes") else bytes(raw)       # Pyodide: Uint8Array → JsProxy 또는 memoryview
        src = np.frombuffer(raw, dtype=np.int32 if v.get("int") else np.float32)
        out[k] = src.astype(np.int64 if v.get("int") else np.float32).reshape(v["shape"])
    return out


def weights_npz(studio: Studio, graph_json: str, state) -> bytes:
    """학습한 가중치 → 생성한 PyTorch 코드의 이름으로 .npz 바이트."""
    d = studio.need_data()
    buf = io.BytesIO()
    codegen.save_npz(codegen.code_state(_arrays(state), json.loads(graph_json), d.length, d.n_out, d.image), buf)
    return buf.getvalue()


def project_bytes(studio: Studio, state) -> bytes:
    """프로젝트(.dlstudio) 바이트: 그래프·데이터 출처·설정·결과 + 학습한 가중치(있으면)."""
    from dlstudio.project import save_project
    d = studio.need_data()
    graph = studio.model_graph if state is not None and studio.model_graph else studio.graph
    if not graph:
        raise UserError("저장할 모델이 없음", "③ 모델 단계에서 층을 놓으세요")
    buf = io.BytesIO()
    save_project(buf, graph, d.source, studio.train_settings, _arrays(state) if state is not None else None,
                 studio.results, studio.selection)
    return buf.getvalue()


@_wrap
def open_project(studio: Studio, path: str) -> dict:
    """프로젝트 파일 열기. 예시 데이터면 다시 불러오고, 아니면 지금 불러온 데이터를 씀 (웹은 파일 위치를 기억 못 함).
    돌려줌: 화면용 응답 + 가중치 {이름: {shape, data(list), int}} + 엔진용 계획."""
    from dlstudio.project import open_project as _open
    try:
        pr = _open(path)
    except ValueError as e:
        raise UserError(str(e), "DL studio에서 저장한 .dlstudio 파일을 고르세요") from None
    src = dict(pr["data_source"])
    if src.get("example"):
        load = {k: src[k] for k in ("normalize", "test_ratio", "seed") if k in src}
        studio.api_data_load(dict(load, example=src["example"]))
    elif studio.data is None:
        raise UserError("이 프로젝트의 데이터를 먼저 불러오세요",
                        "웹 버전은 PC의 파일 위치를 기억하지 못합니다. ② 데이터에서 같은 데이터를 고른 뒤 다시 여세요")
    d = studio.data
    studio.graph = pr["graph"]
    studio.train_settings, studio.results, studio.selection = pr["train_settings"], pr["results"], pr["selection"]
    state, plan_ = None, None
    if pr["state_dict"] is not None:
        plan_ = codegen.plan_dict(pr["graph"], d.length, d.n_out, d.image)
        want = {p[0]: p[1] for st in plan_["steps"] for p in st["params"]}
        if {k: list(v.shape) for k, v in pr["state_dict"].items()} != want:
            raise UserError("프로젝트의 모델이 지금 데이터와 맞지 않음", "프로젝트를 저장할 때와 같은 데이터를 불러오세요")
        state = {k: {"shape": list(v.shape), "data": np.asarray(v).ravel().tolist(), "int": bool(np.issubdtype(v.dtype, np.integer))}
                 for k, v in pr["state_dict"].items()}
        studio.model_graph = pr["graph"]
    else:
        studio.model_graph = None
    res = {"graph": pr["graph"], "summary": d.summary(), "train_settings": pr["train_settings"], "results": pr["results"],
           "trained": state is not None, "selection": studio.selection}
    return {"response": _clean(res), "state": state, "plan": plan_}


# ------------------------------------------------------------ 프로젝트 기록 (브라우저 보관함·구글 드라이브)
# 기록 = 프로젝트 파일(project.dlstudio: 그래프·설정·결과·가중치, 자주 저장) + 데이터 사본(data.zip: 고른 파일들, 데이터를 바꿀 때만).
# 고른 파일은 Pyodide 안의 /data/u<번호>/...에 있다 → 데이터 출처의 경로는 "@data/u<번호>/..."로 적어 두고, 열 때 새 자리로 바꾼다.
DATA_DIR = "/data"                                       # Pyodide 안에서 고른 파일을 두는 곳 (시험은 바꿔 씀)
_MARK = "@data/"


def _root_re():
    return re.compile("^" + re.escape(DATA_DIR) + r"/(u\d+)(/|$)")


def _walk_strings(obj, fn):
    if isinstance(obj, dict):
        return {k: _walk_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_strings(v, fn) for v in obj]
    return fn(obj) if isinstance(obj, str) else obj


def _data_roots(source: dict) -> list:
    roots = set()
    rx = _root_re()
    _walk_strings(source, lambda v: roots.add(rx.match(v).group(1)) if rx.match(v) else v)
    return sorted(roots)


def _portable_source(source: dict) -> dict:
    rx = _root_re()
    return _walk_strings(source, lambda v: rx.sub(lambda m: _MARK + m.group(1) + m.group(2), v))


def data_zip(studio: Studio) -> bytes:
    """지금 데이터의 파일들(고른 파일·폴더) → zip 바이트 (예시 데이터면 빈 바이트)."""
    d = studio.need_data()
    roots = _data_roots(d.source)
    if not roots:
        return b""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:     # 그림·CSV는 대개 이미 압축이 잘 안 됨
        for r in roots:
            base = f"{DATA_DIR}/{r}"
            for dirpath, _, files in os.walk(base):
                for f in files:
                    full = os.path.join(dirpath, f)
                    z.write(full, f"{r}/{os.path.relpath(full, base)}".replace(os.sep, "/"))
    return buf.getvalue()


def record_project(studio: Studio, state, name: str = "", with_data: bool = False) -> bytes:
    """프로젝트 파일 바이트. 데이터 경로는 옮길 수 있게(@data/...). with_data면 데이터 사본도 zip 안 data/에 (내보내기용)."""
    from dlstudio.project import save_project
    d = studio.need_data()
    graph = studio.model_graph if state is not None and studio.model_graph else studio.graph
    if not graph:
        raise UserError("저장할 모델이 없음", "③ 모델 단계에서 층을 놓으세요")
    extra = {}
    if with_data:
        raw = data_zip(studio)
        if raw:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                extra = {f"data/{n}": z.read(n) for n in z.namelist()}
    buf = io.BytesIO()
    save_project(buf, graph, _portable_source(d.source), studio.train_settings,
                 _arrays(state) if state is not None else None, studio.results, studio.selection, extra, name)
    return buf.getvalue()


_n_open = [0]


@_wrap
def record_open(studio: Studio, project_path: str, data_path: str = "") -> dict:
    """프로젝트 기록 열기: 데이터 사본(따로 받은 data.zip 또는 프로젝트 zip 안 data/)을 새 자리에 풀고, 출처를 그 자리로 바꿔 불러옴.
    돌려줌: 화면용 응답 + 가중치 + 엔진용 계획 + 프로젝트 이름."""
    from dlstudio.project import open_project as _open
    try:
        pr = _open(project_path)
    except (ValueError, KeyError) as e:
        raise UserError(f"프로젝트 파일을 읽을 수 없음: {e}", "DL studio에서 저장한 .dlstudio 파일을 고르세요") from None
    _n_open[0] += 1
    root = f"{DATA_DIR}/r{_n_open[0]}"
    root_n = os.path.normpath(root)
    files = 0
    sources = []
    if data_path:
        sources.append(("", zipfile.ZipFile(data_path)))
    with zipfile.ZipFile(project_path) as pz:
        if any(n.startswith("data/") for n in pz.namelist()):
            sources.append(("data/", zipfile.ZipFile(io.BytesIO(open(project_path, "rb").read()))))
    for prefix, z in sources:
        with z:
            for n in z.namelist():
                if not n.startswith(prefix) or n.endswith("/") or n == prefix:
                    continue
                rel = n[len(prefix):]
                target = os.path.normpath(os.path.join(root, rel))
                if not target.startswith(root_n + os.sep):
                    continue                                    # zip 안의 ../ 같은 경로는 무시
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(z.read(n))
                files += 1
    uses_copy = []                                          # 출처가 데이터 사본(@data/...)을 가리키는지
    src = _walk_strings(pr["data_source"], lambda v: (uses_copy.append(1), root + "/" + v[len(_MARK):])[1] if v.startswith(_MARK) else v)
    if src.get("example"):
        studio.load_source(src)
    elif files or uses_copy:
        if not files:
            raise UserError("이 프로젝트의 데이터 사본이 없음", "② 데이터에서 같은 데이터를 고른 뒤 다시 저장하세요")
        studio.load_source(src)
    elif studio.data is None:
        raise UserError("이 프로젝트의 데이터를 먼저 불러오세요", "② 데이터에서 같은 데이터를 고른 뒤 다시 여세요")
    d = studio.data
    studio.graph = pr["graph"]
    studio.train_settings, studio.results, studio.selection = pr["train_settings"], pr["results"], pr["selection"]
    state, plan_ = None, None
    if pr["state_dict"] is not None:
        plan_ = codegen.plan_dict(pr["graph"], d.length, d.n_out, d.image)
        want = {p[0]: p[1] for st in plan_["steps"] for p in st["params"]}
        if {k: list(v.shape) for k, v in pr["state_dict"].items()} != want:
            raise UserError("프로젝트의 모델이 지금 데이터와 맞지 않음", "프로젝트를 저장할 때와 같은 데이터를 불러오세요")
        state = {k: {"shape": list(v.shape), "data": np.asarray(v).ravel().tolist(), "int": bool(np.issubdtype(v.dtype, np.integer))}
                 for k, v in pr["state_dict"].items()}
        studio.model_graph = pr["graph"]
    else:
        studio.model_graph = None
    res = {"graph": pr["graph"], "summary": d.summary(), "train_settings": pr["train_settings"], "results": pr["results"],
           "trained": state is not None, "selection": studio.selection}
    return {"response": _clean(res), "state": state, "plan": plan_, "name": pr.get("name", "")}

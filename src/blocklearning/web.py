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

from blocklearning import codegen
from blocklearning.core import Studio, UserError, _clean, _graph_from, _jsonable
from blocklearning.graph import check_graph


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
            out["panel"] = {"ok": True, "params": plan["params"], "frozen": plan["frozen"],
                            "mb": (plan["params"] + plan["frozen"]) * 4 / 1e6, "message": "", "warnings": []}
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
        from blocklearning.data import DataError
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
                                "task": d.task, "n_out": int(d.n_out), "length": int(d.length), "channels": getattr(d.length, "channels", 1),
                                "image": list(d.image) if d.image else None,
                                "classes": [str(c) for c in d.classes] if d.task in ("classification", "detection") else [],
                                "det": None if d.task != "detection" else
                                {"grid": d.n_out.grid, "per_cell": d.n_out.per_cell, "n_cls": d.n_out.n_cls,
                                 "train": d.det["train"], "test": d.det["test"]},
                                "seg": None if d.task != "segmentation" else {"n_cls": d.n_out.n_cls, "length": d.n_out.length},
                                "multi": None if d.task != "multi" else {"heads": d.n_out.heads}})}


@_wrap
def task_result(studio: Studio, out_bytes, out_tr_bytes) -> dict:
    """예측·이상 탐지: 화면이 학습한 모델의 시험·학습 출력(float32) → 결과 화면 값 (로컬 학습과 같은 jobs.task_result)."""
    from blocklearning.jobs import task_result as tr
    d = studio.need_data()
    raw = lambda b: b.to_bytes() if hasattr(b, "to_bytes") else bytes(b)
    out = np.frombuffer(raw(out_bytes), dtype=np.float32).reshape(len(d.X_test), -1).astype(np.float64)
    out_tr = np.frombuffer(raw(out_tr_bytes), dtype=np.float32).reshape(len(d.X_train), -1).astype(np.float64)
    return _clean(tr(d, out, out_tr))


def set_trained(studio: Studio, graph_json: str, settings_json: str, results_json: str, state=None) -> None:
    """화면에서 학습이 끝났음을 기록 (코드·프로젝트 저장이 학습한 그래프를 알게). state가 있으면 학습 기록에도 더함."""
    studio.model_graph = json.loads(graph_json)
    studio.train_settings = json.loads(settings_json)
    studio.results = json.loads(results_json)
    if state is not None:
        studio.add_run(studio.model_graph, studio.train_settings, studio.results, _arrays(state))


def _state_lists(sd: dict) -> dict:
    """가중치 dict(이름 → 배열) → 화면 엔진이 읽는 {이름: {shape, data(list), int}}."""
    return {k: {"shape": list(np.shape(v)), "data": np.asarray(v).ravel().tolist(),
                "int": bool(np.issubdtype(np.asarray(v).dtype, np.integer))} for k, v in sd.items()}


def _opened(studio: Studio, pr: dict) -> dict:
    """열린 프로젝트 → 상태에 반영하고 화면용 응답 + 가중치 + 엔진용 계획."""
    d = studio.data
    studio.graph = pr["graph"]
    studio.train_settings, studio.results, studio.selection = pr["train_settings"], pr["results"], pr["selection"]
    state, plan_, notice = None, None, ""
    if pr["state_dict"] is not None:
        try:
            plan_ = codegen.plan_dict(pr["graph"], d.length, d.n_out, d.image)
            want = {p[0]: p[1] for st in plan_["steps"] for p in st["params"] if p[2] != "frozen"}   # 고정 가중치는 저장 안 함
            fits = {k: list(v.shape) for k, v in pr["state_dict"].items()} == want
        except Exception:                                   # 그래프가 지금 데이터 모양에서 검사를 통과하지 못함
            fits = False
        if fits:
            state = _state_lists(pr["state_dict"])
            studio.model_graph = pr["graph"]
        elif dict(pr.get("data_source") or {}).get("example"):
            # 예시 데이터가 바뀜 (예전 합성 예시 → 실제 데이터): 모델 구조는 두고 학습 결과만 비움
            plan_, studio.model_graph, studio.results = None, None, {}
            pr = dict(pr, results={})
            notice = "예시 데이터가 공개된 실제 데이터로 바뀌어 예전 학습 결과는 비웠습니다. ④에서 다시 학습하세요."
        else:
            raise UserError("프로젝트의 모델이 지금 데이터와 맞지 않음", "프로젝트를 저장할 때와 같은 데이터를 불러오세요")
    else:
        studio.model_graph = None
    studio.set_runs(pr)
    res = {"graph": pr["graph"], "summary": d.summary(), "train_settings": pr["train_settings"], "results": pr["results"],
           "trained": state is not None, "selection": studio.selection, "runs": studio.runs_view(),
           "predict_kinds": studio.predict_kinds(), "notice": notice}
    return {"response": _clean(res), "state": state, "plan": plan_}


@_wrap
def run_restore(studio: Studio, n: int) -> dict:
    """학습 기록 n을 지금 모델로: 화면용 응답 + 가중치 + 엔진용 계획."""
    run = studio.pick_run(n)
    d = studio.data
    plan_ = codegen.plan_dict(run["graph"], d.length, d.n_out, d.image)
    return {"response": _clean(studio.restored()), "state": _state_lists(studio.run_weights[run["n"]]), "plan": plan_}


@_wrap
def predict_prepare(studio: Studio, body_json: str) -> dict:
    """새 데이터로 예측 1단계: 읽고 정규화해 둠 (엔진이 predict_x로 가져감)."""
    from blocklearning.core import _paths
    from blocklearning.predict import read_new
    d = studio.need_data()
    if studio.model_graph is None:
        raise UserError("학습된 모델이 없음", "④ 학습 단계에서 먼저 학습하세요")
    studio.pending = read_new(d, _paths(json.loads(body_json)))
    return {"n": len(studio.pending["names"])}


def predict_x(studio: Studio) -> bytes:
    return np.ascontiguousarray(studio.pending["X"], dtype=np.float32).tobytes()


@_wrap
def predict_finish(studio: Studio, out_bytes) -> dict:
    """새 데이터로 예측 2단계: 엔진의 출력 → 표·요약·CSV."""
    from blocklearning.predict import table
    raw = out_bytes.to_bytes() if hasattr(out_bytes, "to_bytes") else bytes(out_bytes)
    p, studio.pending = studio.pending, None
    out = np.frombuffer(raw, dtype=np.float32).reshape(len(p["names"]), -1)
    return _clean(table(studio.data, p["names"], out, p["labels"], p["X"], studio.results, p.get("truth")))


@_wrap
def sample(studio: Studio, i: int) -> dict:
    """샘플 따라가기의 데이터 쪽: 시험 샘플 i의 정규화한 값, 원래 값, 정답, 노드 순서."""
    d = studio.need_data()
    if studio.model_graph is None:
        raise UserError("학습된 모델이 없음", "④ 학습 단계에서 먼저 학습하세요")
    if not 0 <= i < len(d.y_test):
        raise UserError(f"시험 샘플 번호 {i}이(가) 없음", f"0–{len(d.y_test) - 1} 사이로 고르세요")
    label = d.label_text("test", i)
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
    codegen.save_npz(codegen.code_state(_arrays(state), json.loads(graph_json), d.length, d.n_out, d.image), buf, data=d)
    return buf.getvalue()


def project_bytes(studio: Studio, state) -> bytes:
    """프로젝트(.blocklearning) 바이트: 그래프·데이터 출처·설정·결과 + 학습한 가중치(있으면)."""
    from blocklearning.project import save_project
    d = studio.need_data()
    graph = studio.model_graph if state is not None and studio.model_graph else studio.graph
    if not graph:
        raise UserError("저장할 모델이 없음", "③ 모델 단계에서 층을 놓으세요")
    buf = io.BytesIO()
    save_project(buf, graph, d.source, studio.train_settings, studio.strip_frozen(graph, _arrays(state)) if state is not None else None,
                 studio.results, studio.selection, runs=studio.runs, run_weights=studio.run_weights,
                 current_run=studio.current_run)
    return buf.getvalue()


def example_files_json(kind: str) -> str:
    """예시 하나에 필요한 데이터 파일 (화면이 누를 때만 내려받아 /app/blocklearning/data/에 둠)."""
    from blocklearning.data import example_files
    return json.dumps(example_files(str(kind or "")))


def project_example(path: str) -> str:
    """프로젝트 파일의 데이터가 예시면 그 이름 (열기 전에 예시 파일을 받아 두게). 아니면 빈 글자."""
    from blocklearning.project import open_project as _open
    try:
        return str(dict(_open(path)["data_source"]).get("example") or "")
    except Exception:                                    # 못 읽으면 여는 단계가 알맞은 오류를 냄
        return ""


@_wrap
def open_project(studio: Studio, path: str) -> dict:
    """프로젝트 파일 열기. 예시 데이터면 다시 불러오고, 아니면 지금 불러온 데이터를 씀 (웹은 파일 위치를 기억 못 함).
    돌려줌: 화면용 응답 + 가중치 {이름: {shape, data(list), int}} + 엔진용 계획."""
    from blocklearning.project import open_project as _open
    try:
        pr = _open(path)
    except ValueError as e:
        raise UserError(str(e), "BlockLearning에서 저장한 .blocklearning 파일을 고르세요") from None
    src = dict(pr["data_source"])
    if src.get("example"):
        load = {k: src[k] for k in ("normalize", "test_ratio", "seed") if k in src}
        studio.api_data_load(dict(load, example=src["example"]))
    elif studio.data is None:
        raise UserError("이 프로젝트의 데이터를 먼저 불러오세요",
                        "웹 버전은 PC의 파일 위치를 기억하지 못합니다. ② 데이터에서 같은 데이터를 고른 뒤 다시 여세요")
    return _opened(studio, pr)


# ------------------------------------------------------------ 프로젝트 기록 (브라우저 보관함·구글 드라이브)
# 기록 = 프로젝트 파일(project.dlstudio: 그래프·설정·결과·가중치, 자주 저장) + 데이터 사본(data.zip: 고른 파일들, 데이터를 바꿀 때만).
# 고른 파일은 Pyodide 안의 /data/u<번호>/...에 있다 → 데이터 출처의 경로는 "@data/u<번호>/..."로 적어 두고, 열 때 새 자리로 바꾼다.
DATA_DIR = "/data"                                       # Pyodide 안에서 고른 파일을 두는 곳 (시험은 바꿔 씀)
_MARK = "@data/"


def _root_re():
    return re.compile("^" + re.escape(DATA_DIR) + r"/([uL]\d+)(/|$)")      # u: 고른 파일, L: 다시 푼 라벨링 작업


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
    from blocklearning.project import save_project
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
                 studio.strip_frozen(graph, _arrays(state)) if state is not None else None, studio.results, studio.selection, extra, name,
                 runs=studio.runs, run_weights=studio.run_weights, current_run=studio.current_run)
    return buf.getvalue()


_n_open = [0]


@_wrap
def record_open(studio: Studio, project_path: str, data_path: str = "") -> dict:
    """프로젝트 기록 열기: 데이터 사본(따로 받은 data.zip 또는 프로젝트 zip 안 data/)을 새 자리에 풀고, 출처를 그 자리로 바꿔 불러옴.
    돌려줌: 화면용 응답 + 가중치 + 엔진용 계획 + 프로젝트 이름."""
    from blocklearning.project import open_project as _open
    try:
        pr = _open(project_path)
    except (ValueError, KeyError) as e:
        raise UserError(f"프로젝트 파일을 읽을 수 없음: {e}", "BlockLearning에서 저장한 .blocklearning 파일을 고르세요") from None
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
    return dict(_opened(studio, pr), name=pr.get("name", ""))


def fit_distance(studio: Studio, feats, n: int, c: int, k: int) -> dict:
    """이상 탐지 '정상과의 거리': 엔진이 뽑은 정상 특징(float32 바이트, n × c) → 넣을 값 {이름: float32 바이트}."""
    from blocklearning.anomaly import fit_distance as fit
    raw = feats.to_bytes() if hasattr(feats, "to_bytes") else bytes(feats)
    F = np.frombuffer(raw, dtype=np.float32).reshape(int(n), int(c))
    return {name: np.ascontiguousarray(v, dtype=np.float32).tobytes() for name, v in fit(F, int(k)).items()}


@_wrap
def loss_setup(studio: Studio, settings_json: str) -> dict:
    """학습 설정의 손실 → 엔진이 쓸 {spec (기본이면 None), class_weights}. 잘못이면 오류."""
    from blocklearning import losses
    d = studio.need_data()
    heads = d.n_out.heads if d.task == "multi" else None
    try:
        spec = losses.normalize(json.loads(settings_json or "{}").get("loss"), d.task, heads)
    except ValueError as e:
        raise UserError(str(e), "④ 학습 → 더 보기의 손실을 고치세요") from None
    if heads:                                               # 정답 여러 개: 엔진에 기본 손실이 없으니 늘 항 목록으로
        cw = {k: v.tolist() for k, v in losses.multi_class_weights(d.y_train, heads, spec).items()}
        return {"spec": [dict(t, label=losses.label(t)) for t in spec], "class_weights": cw or None}
    if losses.is_default(spec, d.task):
        return {"spec": None, "class_weights": None}
    K = len(d.classes) if d.task in ("classification", "segmentation") else 0
    cw = losses.class_weights(d.y_train, K).tolist() if K and any(t.get("balance") for t in spec) else None
    return {"spec": [dict(t, label=losses.label(t)) for t in spec], "class_weights": cw}

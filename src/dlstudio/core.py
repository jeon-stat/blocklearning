"""DL studio의 핵심: 상태(데이터, 모델 그래프, 학습된 모델, 작업)와 화면이 부르는 API.

HTTP와 상관없다. 로컬 서버(server.py)는 이것을 HTTP로 열고, 웹 버전(GitHub Pages)은 브라우저 안의 Pyodide에서
Studio().call(...)로 바로 부른다 (그때 torch가 필요한 일 — 학습·샘플 따라가기·모델 정보 — 은 화면의 TensorFlow.js가 맡음).
"""
from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from dlstudio import tasks
from dlstudio.blocks import to_model_input
from dlstudio.data import (EXAMPLES, DataError, _read_table, load_csv, load_example, load_image_folder,
                           load_series_folder, scan_series_folder)
from dlstudio.graph import build_graph_model, check_graph, from_blocks
from dlstudio.jobs import JobManager, TrainSettings, confusion, regression_view, train_job


class UserError(Exception):
    """화면에 보여 줄 입력 오류 (400)."""

    def __init__(self, message: str, hint: str = "", issues: Optional[list] = None):
        super().__init__(message)
        self.message, self.hint, self.issues = message, hint, issues or []


class Studio:
    """한 사람이 쓰는 도구의 상태와 API. web=True면 torch 없이 (모델 검사의 '모델 정보'는 화면이 계산)."""

    def __init__(self, web: bool = False):
        self.web = web
        self.jobs = JobManager()
        self.data = None
        self.graph: Optional[dict] = None       # 마지막으로 검사한 모델 그래프
        self.model = None                   # 학습된 GraphNet
        self.model_graph: Optional[dict] = None
        self.train_settings: dict = {}
        self.results: dict = {}
        self.selection: Optional[dict] = None   # 고른 과제 {family, subtask, template}
        self.label = None                       # 라벨링 작업 (LabelProject)
        self.label_who = ""                     # 라벨링 작업자 이름 (화면에서 받음)
        self.label_lock = threading.Lock()      # 라벨링 요청은 하나씩 (작업 파일 저장이 겹치지 않게)

    # ------------------------------------------------------------ API
    def api_task_select(self, b: dict) -> dict:
        sel = {k: str(b.get(k, "")) for k in ("family", "subtask", "template")}
        try:
            tasks.find(sel["family"], sel["subtask"], sel["template"])
        except KeyError:
            raise UserError("고를 수 없는 과제", "목록에 있고 '준비 중'이 아닌 과제를 고르세요")
        self.selection = sel
        return {"selection": sel}

    def cancel_training(self) -> None:
        """데이터·프로젝트를 바꾸면 돌고 있는 학습을 멈춤 (옛 데이터로 학습한 모델이 새 데이터에 붙지 않게)."""
        self.jobs.cancel_kind("train")

    def need_data(self):
        if self.data is None:
            raise UserError("데이터가 없음", "② 데이터 단계에서 CSV나 예시 데이터를 먼저 불러오세요")
        return self.data

    def need_model(self):
        if self.model is None:
            raise UserError("학습된 모델이 없음", "④ 학습 단계에서 먼저 학습하세요")
        return self.model

    def model_input(self, X):
        return to_model_input(self.model.kind, X, image=self.data.image)

    def api_data_load(self, b: dict) -> dict:
        opts = {k: b[k] for k in ("normalize", "test_ratio", "seed") if k in b}
        self.cancel_training()
        if b.get("example"):
            kind = b["example"] if isinstance(b["example"], str) else "ecg"
            if kind not in EXAMPLES:
                raise UserError(f"예시 데이터 종류를 모름: {kind}", ", ".join(EXAMPLES))
            d = load_example(kind, **opts)
        elif b.get("series_folder"):
            lf = _path(b.get("label_file")) or None
            d = load_series_folder(_path(b["series_folder"]), str(b.get("column", "")), int(b.get("length", 0)),
                                   resample=b.get("resample", "interp"), label_file=lf, task=b.get("task", "auto"),
                                   only_labeled=bool(b.get("only_labeled")), **opts)
        elif b.get("folder"):
            folder = _path(b.get("folder"))
            if not os.path.isdir(folder):
                raise UserError(f"폴더를 찾을 수 없음: {folder}", "클래스별 하위 폴더가 든 폴더의 전체 경로를 넣으세요")
            d = load_image_folder(folder, size=int(b.get("size", 28)), label_file=_path(b.get("label_file")) or None,
                                  only_labeled=bool(b.get("only_labeled")), task=b.get("task", "auto"), **opts)
        else:
            path = _path(b.get("path"))
            if not os.path.isfile(path):
                raise UserError(f"파일을 찾을 수 없음: {path}", "CSV 파일의 전체 경로를 넣으세요")
            iw = b.get("image_width")
            d = load_csv(path, label_col=b.get("label_col"), task=b.get("task", "auto"),
                         image_width=None if iw is None or iw == "" else int(iw), **opts)
        self.data, self.model, self.model_graph, self.results = d, None, None, {}
        return {"summary": d.summary(), "source": d.source}

    # ------------------------------------------------------------ 라벨링 작업실
    def api_label(self, op: str, b: dict) -> dict:
        with self.label_lock:
            return self._api_label(op, b)

    def _api_label(self, op: str, b: dict) -> dict:
        from dlstudio.labeling import LabelProject
        if op == "who":                                    # 작업자 이름 (작업을 바꿔도 유지)
            self.label_who = str(b.get("name", "")).strip()[:40]
            if self.label is not None:
                self.label.who = self.label_who
                return self.label.state()
            return {"who": self.label_who}
        if op == "new":
            self.label = LabelProject.new(_path(b.get("path")), str(b.get("column", "")), list(b.get("classes") or []),
                                          _path(b.get("file")), kind=str(b.get("kind") or "series_folder"),
                                          schema=str(b.get("schema") or "class"), id_col=b.get("id_col") or None,
                                          purpose=str(b.get("purpose") or "general"))
            self.label.who = self.label_who
            return self.label.state()
        if op == "open":
            self.label = LabelProject.open(_path(b.get("file")))
            self.label.who = self.label_who
            return self.label.state()
        p = self.label
        if p is None:
            raise UserError("라벨링 작업이 없음", "작업을 새로 만들거나 작업 파일을 여세요")
        if op == "state":
            return p.state()
        if op == "items":
            return p.items(b.get("filter"), b.get("sort"), int(b.get("page", 0)), int(b.get("size", 60)))
        if op == "item":
            return p.item(str(b.get("name", "")))
        if op == "set":
            n = p.set_labels(list(b.get("names") or []), b.get("label"))
            return dict(p.state(), changed=n)
        if op == "confirm":
            n = p.confirm(list(b.get("names") or []))           # 먼저 확인하고 상태를 (dict 인자 순서 주의)
            return dict(p.state(), changed=n)
        if op == "note":
            p.note(str(b.get("name", "")), str(b.get("text", "")))
            return p.state()
        if op in ("undo", "redo"):
            ok = p.undo() if op == "undo" else p.redo()
            return dict(p.state(), changed=int(ok))
        if op == "classes":
            p.classes_op(str(b.get("op", "")), name=b.get("name", ""), new=b.get("new", ""), color=b.get("color", ""))
            return p.state()
        if op == "rule_preview":
            names = p.rule_matches(list(b.get("conds") or []), bool(b.get("overwrite")), b.get("label"))
            return {"count": len(names), "names": names[:20]}
        if op == "rule_apply":
            n = p.rule_apply(list(b.get("conds") or []), b.get("label"), bool(b.get("overwrite")))
            return dict(p.state(), changed=n)
        if op == "cluster":
            return p.cluster(int(b.get("k", 6)))
        if op == "normal_candidates":
            names = p.normal_candidates(int(b.get("k", 4)))
            return {"names": names, "count": len(names)}
        if op == "export_normals":
            return {"rows": p.export_normals(_path(b.get("file")), bool(b.get("confirmed_only")))}
        if op == "split":
            return {"files": p.split(int(b.get("n", 2)))}
        if op == "merge":
            r = p.merge(_path(b.get("file")))
            return dict(p.state(), **r)
        if op == "resolve":
            p.resolve(str(b.get("name", "")), str(b.get("take", "")))
            return p.state()
        if op == "set_boxes":
            p.set_boxes(str(b.get("name", "")), b.get("boxes") or [])
            return p.state()
        if op == "export_yolo":
            return {"boxes": p.export_yolo(_path(b.get("folder")), bool(b.get("confirmed_only")))}
        if op == "export_coco":
            return {"file": p.export_coco(_path(b.get("file")), bool(b.get("confirmed_only")))}
        if op == "paint":
            p.paint(str(b.get("name", "")), b.get("start"), b.get("end"), b.get("cls"))
            return p.state()
        if op == "import_column":
            r = p.import_column(str(b.get("column", "")))
            return dict(p.state(), **r)
        if op == "columns":                                # 첫 샘플 파일의 열 이름 (구간을 가져올 열 고르기)
            first = next((s for s in p.samples.values() if s.get("path")), None)
            return {"columns": _read_table(first["path"])[0] if first else []}
        if op == "suggest":
            r = p.suggest()
            return dict(p.state(), **r)
        if op == "import":
            r = p.import_table(_path(b.get("label_file")))
            return dict(p.state(), **r)
        if op == "export":
            n = p.export(_path(b.get("file")), bool(b.get("confirmed_only")))
            return {"rows": n}
        if op == "use":
            return self._label_use(p, bool(b.get("confirmed_only")), _path(b.get("file")) or None)
        raise UserError(f"라벨링 동작 '{op}'을(를) 모름", "")

    def _label_use(self, p, confirmed_only: bool, file) -> dict:
        """정답을 내보내고 studio 데이터로 불러옴. 시계열·그림 폴더는 정답 표로, 표는 정답 열을 붙인 학습용 CSV로."""
        if p.schema == "boxes":
            raise UserError("상자 정답으로 학습하는 '객체 탐지' 과제는 아직 DL studio에 없음",
                            "[내보내기]에서 YOLO·COCO 형식으로 저장해 다른 탐지 도구에서 쓸 수 있습니다")
        if p.schema == "segments":
            raise UserError("구간 정답으로 학습하는 '구간 분할' 과제는 아직 studio에 없음",
                            "[내보내기]로 구간 표(file,start,end,label)를 저장해 두세요. 구간 분할 과제가 생기면 그대로 씁니다")
        lab = p.labels(confirmed_only)
        number = p.schema == "number"
        kinds = len(set(lab.values()))
        if len(lab) < 10 or (not number and kinds < 2):
            what = f"정답이 있는 샘플 {len(lab)}개" + ("" if number else f", 클래스 {kinds}종류")
            raise UserError(f"{what}라 학습할 수 없음",
                            "정답 있는 샘플이 10개 이상" + ("" if number else "이고 클래스가 2종류 이상") + "이어야 합니다"
                            + (" (확인된 것만 쓰는 중: 확인을 더 하거나 '확인된 것만'을 끄세요)" if confirmed_only else ""))
        work = Path(p.file)
        stem = work.name.replace(".dllabel", "")
        src = p.meta["source"]
        task = "regression" if number else "auto"
        if src["kind"] == "table":
            file = file or str(work.with_name(stem + "_학습.csv"))
            p.export_training(file, confirmed_only)
            return dict(self.api_data_load({"path": file, "task": task}), label_file=file)
        file = file or str(work.with_name(stem + "_정답.csv"))
        p.export(file, confirmed_only)
        if src["kind"] == "image_folder":
            d = self.api_data_load({"folder": src["path"], "size": 28, "label_file": file, "only_labeled": True, "task": task})
            return dict(d, label_file=file)
        length = scan_series_folder(src["path"], label_file=file)["default_length"]
        d = self.api_data_load({"series_folder": src["path"], "column": src["column"], "length": length, "label_file": file,
                                "only_labeled": True, "task": task})
        return dict(d, label_file=file)

    def api_data_scan_folder(self, b: dict) -> dict:
        """샘플마다 CSV 하나인 폴더 살펴보기 (불러오기 전에 파일 수·길이·열·정답 방식)."""
        return scan_series_folder(_path(b.get("path")), label_file=_path(b.get("label_file")) or None)

    def api_model_check(self, b: dict) -> dict:
        d = self.need_data()
        graph = _graph_from(b)
        c = check_graph(graph, d.length, d.n_out, image=d.image)
        out = dict(c.to_dict(), graph=graph, panel=None)
        if isinstance(graph, dict):
            self.graph = graph                           # 하던 작업은 문제가 있어도 기억 (새로 고쳐도 남게)
        if c.ok:
            from dlstudio.analysis import model_panel
            out["panel"] = model_panel(build_graph_model(graph, d.length, d.n_out, image=d.image))
        return out

    def api_train(self, b: dict) -> dict:
        d = self.need_data()
        graph = _graph_from(b)
        c = check_graph(graph, d.length, d.n_out, image=d.image)
        if not c.ok:
            i = c.issues[0]
            raise UserError(i.message, i.hint, [x.to_dict() for x in c.issues])
        settings = TrainSettings.from_dict(b.get("settings"))

        def run(job):
            r = train_job(job, graph, d, settings)
            if not job.cancelled and self.data is d:              # 그사이 데이터가 바뀌었으면 버림
                self.model, self.model_graph, self.train_settings = r["model"], graph, dict(b.get("settings") or {})
                self.results = {k: v for k, v in r.items() if k != "model"}
            return r

        return {"id": self.jobs.start("train", run)}

    def api_activations(self, b: dict) -> dict:
        from dlstudio.analysis import activations
        import torch
        d, m = self.need_data(), self.need_model()
        i = int(b.get("sample_index", 0))
        if not 0 <= i < len(d.y_test):
            raise UserError(f"시험 샘플 번호 {i}이(가) 없음", f"0–{len(d.y_test) - 1} 사이로 고르세요")
        x = self.model_input(d.X_test[i:i + 1])[0]
        with torch.no_grad():
            out = m.eval()(torch.as_tensor(x[None], dtype=torch.float32))[0].numpy()
        label = d.classes[int(d.y_test[i])] if d.task == "classification" else float(d.y_test[i])
        pred = d.classes[int(out.argmax())] if d.task == "classification" else float(out[0])
        order = check_graph(self.model_graph, d.length, d.n_out, image=d.image).order
        return {"index": i, "label": label, "prediction": pred, "input": np.round(d.raw(d.X_test[i]), 5).tolist(),
                "values": activations(m, x), "order": order}

    def api_project_save(self, b: dict) -> dict:
        from dlstudio.project import save_project
        b = dict(b, path=_path(b.get("path")))
        d = self.need_data()
        graph = self.model_graph if self.model is not None else self.graph   # 학습한 모델이 있으면 가중치와 같은 그래프
        if not graph:
            raise UserError("저장할 모델이 없음", "③ 모델 단계에서 층을 놓으세요")
        save_project(b["path"], graph, d.source, self.train_settings, self.model, self.results, self.selection)
        return {"ok": True}

    def api_project_open(self, b: dict) -> dict:
        from dlstudio.project import open_project
        b = dict(b, path=_path(b.get("path")))
        self.cancel_training()
        if not os.path.isfile(b["path"]):
            raise UserError(f"파일을 찾을 수 없음: {b.get('path')}", "프로젝트 파일(.dlstudio)의 전체 경로를 넣으세요")
        pr = open_project(b["path"])
        summary = self.load_source(pr["data_source"])
        from dlstudio.graph import model_from_saved
        self.graph = pr["graph"]
        self.train_settings, self.results, self.selection = pr["train_settings"], pr["results"], pr["selection"]
        m = model_from_saved(pr, self.data.length, self.data.n_out, image=self.data.image)
        if m is not None:
            self.model, self.model_graph = m.eval(), pr["graph"]
        return {"graph": pr["graph"], "summary": summary, "train_settings": pr["train_settings"],
                "results": pr["results"], "trained": self.model is not None, "selection": self.selection}

    def load_source(self, source: dict) -> dict:
        """프로젝트에 적힌 데이터 출처(예시·CSV·이미지 폴더·시계열 폴더)를 다시 불러와 요약을 돌려줌."""
        src = dict(source)
        load = {k: src[k] for k in ("normalize", "test_ratio", "seed") if k in src}
        if src.get("example"):
            return self.api_data_load(dict(load, example=src["example"]))["summary"]
        if src.get("folder"):
            return self.api_data_load(dict(load, folder=src["folder"], size=src.get("size", 28), task=src.get("task", "auto"),
                                           label_file=src.get("label_file"), only_labeled=src.get("only_labeled")))["summary"]
        if src.get("series_folder"):
            return self.api_data_load(dict(load, task=src.get("task", "auto"),
                                           **{k: src.get(k) for k in ("series_folder", "column", "length",
                                                                      "resample", "label_file", "only_labeled")}))["summary"]
        return self.api_data_load(dict(load, path=src.get("path"), label_col=src.get("label_col"),
                                       image_width=src.get("image_width"), task=src.get("task", "auto")))["summary"]

    def api_project_export(self, b: dict) -> dict:
        from dlstudio.project import export_torch
        b = dict(b, path=_path(b.get("path")))
        m = self.need_model()                            # torch만으로 읽는 TorchScript (.pt)
        return {"ok": True, "info": export_torch(m, self.data, b["path"])}

    def _code(self, b: dict, weights_file: Optional[str] = None) -> tuple:
        """요청의 그래프 → (코드, 학습한 모델과 같은 그래프인지, framework)."""
        from dlstudio import codegen
        d = self.need_data()
        graph = _graph_from(b)
        fw = b.get("framework", "pytorch")
        if fw not in codegen.FRAMEWORKS:
            raise UserError(f"코드 종류를 모름: {fw!r}", "pytorch 또는 tensorflow")
        c = check_graph(graph, d.length, d.n_out, image=d.image)
        if not c.ok:
            i = c.issues[0]
            raise UserError(i.message, i.hint or "③ 모델의 문제를 먼저 고치세요", [x.to_dict() for x in c.issues])
        trained = (self.model is not None or self.web) and self.model_graph is not None and _same_model(graph, self.model_graph)
        info = {"task": d.task, "classes": list(d.classes) if d.task == "classification" else []}
        settings = b.get("settings") or self.train_settings or {}
        train = {k: settings[k] for k in ("epochs", "lr", "batch_size") if k in settings}
        code = codegen.generate(fw, graph, d.length, d.n_out, d.image, info, train,
                                weights_file=weights_file if trained and fw == "pytorch" else None)
        return code, trained, fw, graph

    def api_code(self, b: dict) -> dict:
        """③·④ [코드로 보기]: PyTorch 또는 TensorFlow Keras 코드."""
        code, trained, fw, _ = self._code(b, weights_file=b.get("weights_file"))
        return {"code": code, "framework": fw, "trained": trained,
                "filename": "model_pytorch.py" if fw == "pytorch" else "model_tensorflow.py"}

    def api_code_save(self, b: dict) -> dict:
        """코드를 .py로 저장. PyTorch이고 학습한 모델이면 가중치(<이름>_weights.npz)도 같은 폴더에."""
        from dlstudio import codegen
        path = Path(_path(b.get("path")))
        if path.suffix.lower() != ".py":
            raise UserError("파이썬 파일(.py) 경로를 넣으세요", "예: model_pytorch.py (폴더까지 넣은 전체 경로)")
        weights = path.with_name(path.stem + "_weights.npz")
        code, trained, fw, graph = self._code(b, weights_file=weights.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        files = [str(path)]
        if trained and fw == "pytorch":
            d = self.data
            codegen.save_npz(codegen.torch_state_dict(self.model, graph, d.length, d.n_out, d.image), str(weights), data=d)
            files.append(str(weights))
        path.write_text(code, encoding="utf-8")
        return {"files": files, "trained": trained}

    def api_state(self) -> dict:
        """화면을 새로 고쳐도 이어서 쓰도록 지금 상태 (데이터 요약, 블록, 학습 여부·결과)."""
        return {"summary": None if self.data is None else self.data.summary(),
                "graph": self.graph or self.model_graph, "trained_graph": self.model_graph,
                "trained": self.model is not None,
                "results": self.results, "train_settings": self.train_settings, "selection": self.selection}

    def job_view(self, job_id: str) -> dict:
        st = self.jobs.get(job_id)
        if isinstance(st["result"], dict):
            st["result"] = {k: v for k, v in st["result"].items() if k not in ("model", "traceback")}
        return st

    # ------------------------------------------------------------ 부르기 (HTTP·웹 공통)
    def call(self, method: str, path: str, body=None) -> tuple:
        """API 하나를 불러 (상태 코드, JSON으로 보낼 값). GET은 상태·과제 목록·작업 진행, POST는 ROUTES."""
        body = body if isinstance(body, dict) else {}
        try:
            if method == "GET":
                if path == "/api/state":
                    return 200, _clean(self.api_state())
                if path == "/api/tasks":
                    return 200, tasks.catalog_view()
                if path.startswith("/api/jobs/"):
                    try:
                        return 200, _clean(self.job_view(path.split("/")[3]))
                    except KeyError:
                        return 404, _err("작업을 찾을 수 없음")
                return 404, _err("없는 주소")
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                self.jobs.cancel(path.split("/")[3])
                return 200, {"ok": True}
            if path.startswith("/api/label/"):
                return 200, _clean(self.api_label(path[len("/api/label/"):], body))
            name = ROUTES.get(path)
            if name is None:
                return 404, _err("없는 주소")
            return 200, _clean(getattr(self, name)(body))
        except UserError as e:
            return 400, _err(e.message, e.hint, issues=e.issues)
        except DataError as e:
            first = e.issues[0]
            return 400, _err(first.message, first.hint, issues=[i.to_dict() for i in e.issues])
        except (ValueError, KeyError, TypeError) as e:
            return 400, _err(f"입력이 맞지 않음: {e}", "값을 확인하세요", traceback.format_exc()[-1500:])
        except Exception as e:                                 # 화면에는 한 줄, 자세히는 detail
            return 500, _err(f"처리 중 오류: {type(e).__name__}: {str(e)[:200]}", "", traceback.format_exc()[-3000:])


def _err(message: str, hint: str = "", detail: str = "", issues=None) -> dict:
    return {"error": {"message": message, "hint": hint, "detail": detail, "issues": issues or []}}


ROUTES = {
    "/api/data/load": "api_data_load", "/api/model/check": "api_model_check", "/api/train": "api_train",
    "/api/activations": "api_activations",
    "/api/project/save": "api_project_save", "/api/project/open": "api_project_open",
    "/api/project/export": "api_project_export", "/api/task/select": "api_task_select",
    "/api/data/scan_folder": "api_data_scan_folder", "/api/code": "api_code", "/api/code/save": "api_code_save",
}


VIEW_KEYS = ("pos", "size", "brk", "shift")      # 화면용 값 (모델에는 영향 없음)


def _same_model(a, b) -> bool:
    """두 그래프가 같은 모델인지 (화면용 값은 빼고 비교)."""
    def strip(g):
        if not isinstance(g, dict):
            return g
        def node(n):
            n = {k: v for k, v in n.items() if k not in VIEW_KEYS}
            if isinstance(n.get("body"), dict):
                n["body"] = dict(n["body"], nodes=[node(x) for x in n["body"].get("nodes", [])])
            return n
        return dict(g, nodes=[node(n) for n in g.get("nodes", [])])
    return strip(a) == strip(b)


def _graph_from(b: dict):
    """요청의 모델: 그래프(v2), 또는 예전 블록 목록이 오면 그래프로 옮김."""
    if "graph" in b:
        return b.get("graph")
    blocks = b.get("blocks")
    return from_blocks(blocks) if isinstance(blocks, list) else blocks


def _path(v) -> str:
    """화면에서 넣은 경로: 앞뒤 공백과 따옴표 제거 (Windows '경로로 복사'는 큰따옴표를 붙임)."""
    return str(v or "").strip().strip('"').strip("'").strip()


def _clean(o):
    """JSON에 못 넣는 NaN·Infinity는 None으로 (브라우저 JSON.parse가 깨지지 않게)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

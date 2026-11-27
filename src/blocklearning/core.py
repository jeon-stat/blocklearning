"""BlockLearning의 핵심: 상태(데이터, 모델 그래프, 학습된 모델, 작업)와 화면이 부르는 API.

HTTP와 상관없다. 로컬 서버(server.py)는 이것을 HTTP로 열고, 웹 버전(GitHub Pages)은 브라우저 안의 Pyodide에서
Studio().call(...)로 바로 부른다 (그때 torch가 필요한 일 — 학습·샘플 따라가기·모델 정보 — 은 화면의 TensorFlow.js가 맡음).
"""
from __future__ import annotations

import datetime
import os
import threading
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from blocklearning import tasks
from blocklearning.blocks import to_model_input
from blocklearning.data import (EXAMPLE_KINDS, ALIASES, DataError, _read_table, load_csv, load_example, load_image_folder,
                           load_series_folder, scan_series_folder)
from blocklearning.graph import build_graph_model, check_graph, from_blocks
from blocklearning.data import channels_of
from blocklearning.jobs import JobManager, TrainSettings, confusion, regression_view, train_job


class UserError(Exception):
    """화면에 보여 줄 입력 오류 (400)."""

    def __init__(self, message: str, hint: str = "", issues: Optional[list] = None):
        super().__init__(message)
        self.message, self.hint, self.issues = message, hint, issues or []


class Studio:
    """한 사람이 쓰는 도구의 상태와 API. web=True면 torch 없이 (모델 검사의 '모델 정보'는 화면이 계산)."""

    EXTRA_ROUTES: dict = {}                     # 설치형 서버만 더하는 API (server.py)

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
        self.runs: list = []                    # 학습 기록 [{n, time, graph, settings, results, data_key}]
        self.run_weights: dict = {}             # 기록 번호 → 가중치 {이름: 배열} (GraphNet 이름)
        self.current_run: Optional[int] = None  # 지금 학습된 모델이 몇 번 기록인지
        self.pending = None                     # 웹 예측: 엔진에 넘길 값을 잠시 둠
        self.new_data = None                    # ⑤ 마지막으로 추론한 새 데이터 (read_new 결과: 샘플마다 추론 과정을 볼 때)
        self.outputs: set = set()               # 저장·내보내기로 쓴 파일 (설치형: 이것만 브라우저로 내려받을 수 있음)

    def wrote(self, *paths) -> list:
        """저장·내보내기로 쓴 파일을 기억하고 전체 경로로 돌려줌."""
        full = [str(Path(p).resolve()) for p in paths]
        self.outputs.update(full)
        return full

    def reset_work(self) -> None:
        """새 프로젝트로: 데이터·모델·학습 결과를 비움 (라벨링 작업과 저장한 파일 목록은 둠)."""
        self.cancel_training()
        self.data = self.graph = self.model = self.model_graph = self.selection = self.new_data = None
        self.train_settings, self.results = {}, {}
        self.runs, self.run_weights, self.current_run = [], {}, None

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
        return to_model_input(self.model.kind, X, image=self.data.image, channels=channels_of(self.data.length))

    def api_data_load(self, b: dict) -> dict:
        opts = {k: b[k] for k in ("normalize", "test_ratio", "seed") if k in b}
        self.cancel_training()
        if b.get("forecast_reload"):                        # 예측: 지금 데이터를 값 열·입력·예측 길이만 바꿔 다시
            src = dict(self.need_data().source or {})
            if src.get("task") != "forecast":
                raise UserError("지금 데이터는 시계열 예측 데이터가 아님", "② 데이터에서 시계열 예측 데이터를 불러오세요")
            src.update({k: b["forecast_reload"].get(k) for k in ("column", "L", "H", "stride") if b["forecast_reload"].get(k)})
            if "calendar" in b["forecast_reload"]:
                src["calendar"] = bool(b["forecast_reload"]["calendar"])
            return {"summary": self.load_source(src), "source": self.data.source}
        if b.get("roles_reload"):                           # ② 역할 표: 지금 CSV를 열 역할·채널 수만 바꿔 다시
            src = dict(self.need_data().source or {})
            if not src.get("path") or src.get("task") == "forecast":
                raise UserError("역할 표는 CSV로 불러온 데이터에만 씀", "② 데이터에서 CSV를 불러오세요")
            r = b["roles_reload"]
            src.update(targets=[int(j) for j in r.get("targets") or []], skip=[int(j) for j in r.get("skip") or []],
                       channels=max(1, int(r.get("channels") or 1)))
            if r.get("length"):
                src["length"] = int(r["length"])
            if not src["targets"]:
                raise UserError("정답 열이 없음", "역할 표에서 정답 열을 하나 이상 고르세요")
            src["label_col"] = src["targets"][0]
            return {"summary": self.load_source(src), "source": self.data.source}
        if b.get("task") == "segmentation" and b.get("path") and not b.get("example"):   # 구간 분할: 한 행 = 한 시점 + 상태 열
            from blocklearning.segment import load_segment_csv
            path = _path(b.get("path"))
            if not os.path.isfile(path):
                raise UserError(f"파일을 찾을 수 없음: {path}", "CSV 파일의 전체 경로를 넣으세요")
            tg = b.get("targets") or ([b["label_col"]] if b.get("label_col") is not None else None)
            d = load_segment_csv(path, state_col=tg[0] if tg else None, skip=b.get("skip"), length=b.get("length") or None, **opts)
            self.data, self.model, self.model_graph, self.results, self.current_run, self.new_data = d, None, None, {}, None, None
            return {"summary": d.summary(), "source": d.source}
        if b.get("task") == "forecast" and not b.get("example"):   # 긴 시계열 CSV 하나 (한 행 = 한 시점)
            from blocklearning.forecast import load_forecast_csv
            path = _path(b.get("path"))
            if not os.path.isfile(path):
                raise UserError(f"파일을 찾을 수 없음: {path}", "CSV 파일의 전체 경로를 넣으세요")
            d = load_forecast_csv(path, b.get("column"), b.get("L"), b.get("H"), b.get("stride"), calendar=b.get("calendar", True), **opts)
            self.data, self.model, self.model_graph, self.results, self.current_run, self.new_data = d, None, None, {}, None, None
            return {"summary": d.summary(), "source": d.source}
        if b.get("example"):
            kind = b["example"] if isinstance(b["example"], str) else "ecg"
            if kind not in EXAMPLE_KINDS and kind not in ALIASES:
                raise UserError(f"예시 데이터 종류를 모름: {kind}", ", ".join(EXAMPLE_KINDS))
            d = load_example(kind, **opts, **{k: b[k] for k in ("L", "H", "stride") if b.get(k)}, calendar=b.get("calendar", True))
        elif b.get("series_folder"):
            lf = _path(b.get("label_file")) or None
            col = b.get("column", "")
            d = load_series_folder(_path(b["series_folder"]), [str(c) for c in col] if isinstance(col, list) else str(col), int(b.get("length", 0)),
                                   resample=b.get("resample", "interp"), label_file=lf,
                                   task="auto" if b.get("task") == "generation" else b.get("task", "auto"),
                                   only_labeled=bool(b.get("only_labeled")), **opts)
        elif b.get("folder") and b.get("det_file"):                # 탐지: 그림 폴더 + 탐지 정답 파일 (라벨링에서 만듦)
            folder = _path(b.get("folder"))
            if not os.path.isdir(folder):
                raise UserError(f"폴더를 찾을 수 없음: {folder}", "라벨링할 때와 같은 그림 폴더를 고르세요")
            from blocklearning.data import load_detection_folder
            d = load_detection_folder(folder, _path(b["det_file"]), size=int(b.get("size", 64)), **opts)
        elif b.get("folder"):
            folder = _path(b.get("folder"))
            if not os.path.isdir(folder):
                raise UserError(f"폴더를 찾을 수 없음: {folder}", "클래스별 하위 폴더가 든 폴더의 전체 경로를 넣으세요")
            d = load_image_folder(folder, size=int(b.get("size", 28)), label_file=_path(b.get("label_file")) or None,
                                  only_labeled=bool(b.get("only_labeled")),
                                  task="anomaly" if b.get("task") == "generation" and not b.get("label_file") else b.get("task", "auto"),
                                  color=bool(b.get("color")), **opts)
        else:
            path = _path(b.get("path"))
            if not os.path.isfile(path):
                raise UserError(f"파일을 찾을 수 없음: {path}", "CSV 파일의 전체 경로를 넣으세요")
            iw = b.get("image_width")
            d = load_csv(path, label_col=b.get("label_col"), task="auto" if b.get("task") in ("multi", "generation") else b.get("task", "auto"),
                         image_width=None if iw is None or iw == "" else int(iw), targets=b.get("targets") or None,
                         skip=b.get("skip") or None, channels=int(b.get("channels") or 1), **opts)
        if b.get("task") == "anomaly" and d.task != "anomaly":       # 분류처럼 읽은 것 → 정상만 학습하는 이상 탐지로
            from blocklearning.anomaly import to_anomaly
            d = to_anomaly(d, b.get("normal"))
        if b.get("task") == "generation" and d.task != "generation":   # 읽은 것의 입력만으로 새 샘플 만들기 (정답은 쓰지 않음)
            from blocklearning.generate import to_generation
            d = to_generation(d)
        self.data, self.model, self.model_graph, self.results, self.current_run, self.new_data = d, None, None, {}, None, None
        return {"summary": d.summary(), "source": d.source}

    # ------------------------------------------------------------ 라벨링 작업실
    def api_label(self, op: str, b: dict) -> dict:
        with self.label_lock:
            return self._api_label(op, b)

    def _api_label(self, op: str, b: dict) -> dict:
        from blocklearning.labeling import LabelProject
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
                                          purpose=str(b.get("purpose") or "general"),
                                          columns=[str(c) for c in (b.get("columns") or []) if isinstance(c, str)])
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
            return {"boxes": p.export_yolo(_path(b.get("folder")), bool(b.get("confirmed_only")))}         # 도형 수
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
        if op == "deep_suggest":
            r = self._deep_suggest(p)
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

    def _deep_suggest(self, p) -> dict:
        """PC판 딥러닝 제안: 정답 붙은 샘플로 대표 모델(시계열 1D CNN·그림 작은 CNN·표 MLP)을 학습해 나머지에 제안."""
        import torch
        from blocklearning.jobs import fit, metric
        prep = p.deep_data()
        d = prep["data"]
        graph = deep_graph(prep["kind"])
        net = build_graph_model(graph, d.length, d.n_out, seed=0, image=d.image)
        xin = lambda X: torch.as_tensor(to_model_input(net.kind, X, image=d.image), dtype=torch.float32)
        net = fit(net, to_model_input(net.kind, d.X_train, image=d.image), d.y_train, "classification", DEEP_EPOCHS, DEEP_LR, 32, seed=0)
        net.eval()
        with torch.no_grad():
            acc = metric(net(xin(d.X_test)).numpy(), d.y_test, "classification") if len(d.y_test) else None
            probs = torch.softmax(net(xin(prep["X_pred"])), 1).numpy()
        r = p.apply_model(prep["targets"], list(d.classes), probs)
        return dict(r, acc=acc, trained_on=int(len(d.y_train) + len(d.y_test)), model=graph_name(prep["kind"]))

    def _label_use(self, p, confirmed_only: bool, file) -> dict:
        """정답을 내보내고 studio 데이터로 불러옴. 시계열·그림 폴더는 정답 표로, 표는 정답 열을 붙인 학습용 CSV로."""
        if p.schema == "boxes":
            return self._label_use_det(p, confirmed_only, file)
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

    def _label_use_det(self, p, confirmed_only: bool, file) -> dict:
        """도형 정답 → 탐지 정답 파일(JSON: 클래스, 그림마다 상자) → 객체 탐지 데이터. 다각형·원은 감싸는 상자로, 점은 뺌.
        상자가 하나도 없는 클래스는 뺌 (배울 것이 없음)."""
        import json
        from blocklearning.detect import DEFAULT_SIZE, shape_box
        names = [c["name"] for c in p.classes]
        items = []
        for n, shapes in p._boxed(confirmed_only):
            W, H = p.samples[n].get("width") or 1, p.samples[n].get("height") or 1
            bs = []
            for s in shapes:
                b = shape_box(s, W, H) if s[-1] in names else None
                if b:
                    bs.append([s[-1]] + [round(float(v), 6) for v in b])
            if bs:
                items.append({"name": n, "boxes": bs})
        used = [c for c in names if any(b[0] == c for it in items for b in it["boxes"])]
        if len(items) < 10:
            raise UserError(f"상자를 그린 그림이 {len(items)}장이라 학습할 수 없음",
                            "상자·다각형·원을 그린 그림이 10장 이상이어야 합니다 (점은 탐지에 쓰지 않음)"
                            + (" (확인된 것만 쓰는 중: 확인을 더 하거나 '확인된 것만'을 끄세요)" if confirmed_only else ""))
        idx = {c: i for i, c in enumerate(used)}
        for it in items:
            it["boxes"] = [[idx[b[0]]] + b[1:] for b in it["boxes"]]
        work = Path(p.file)
        file = file or str(work.with_name(work.name.replace(".dllabel", "") + "_탐지.json"))
        Path(file).write_text(json.dumps({"classes": used, "items": items}, ensure_ascii=False), encoding="utf-8")
        d = self.api_data_load({"folder": p.meta["source"]["path"], "det_file": file, "size": DEFAULT_SIZE})
        return dict(d, label_file=file)

    def api_data_columns(self, b: dict) -> dict:
        """CSV 열 고르기 (불러오기 전): 열마다 이름·종류·값 몇 개·짐작한 역할."""
        from blocklearning.data import csv_columns
        path = _path(b.get("path"))
        if not os.path.isfile(path):
            raise UserError(f"파일을 찾을 수 없음: {path}", "CSV 파일의 전체 경로를 넣으세요")
        return csv_columns(path)

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
            from blocklearning.analysis import model_panel
            out["panel"] = model_panel(build_graph_model(graph, d.length, d.n_out, image=d.image))
        return out

    def api_train(self, b: dict) -> dict:
        d = self.need_data()
        graph = _graph_from(b)
        c = check_graph(graph, d.length, d.n_out, image=d.image)
        if not c.ok:
            i = c.issues[0]
            raise UserError(i.message, i.hint, [x.to_dict() for x in c.issues])
        from blocklearning.generate import graph_issue
        gi = graph_issue(d, graph)
        if gi:
            raise UserError(*gi)
        settings = TrainSettings.from_dict(b.get("settings"))
        from blocklearning import losses
        try:
            spec = losses.normalize(settings.loss, d.task, d.n_out.heads if d.task == "multi" else None)
            nodes = set(check_graph(graph, d.length, d.n_out, image=d.image).shapes)
            bad = [t["node"] for t in spec if t["term"] == "act_l1" and t["node"] not in nodes]
            if bad:
                raise ValueError(f"블록 출력 L1의 블록 '{bad[0]}'이(가) 모델에 없음")
        except ValueError as e:
            raise UserError(str(e), "④ 학습 → 더 보기의 손실을 고치세요") from None
        from blocklearning.optim import OptimError, normalize as optim_check
        try:
            optim_check(settings.optim())
        except OptimError as e:
            raise UserError(str(e), "④ 학습의 학습 방법 설정을 고치세요") from None
        init = self._resume_state(graph, d) if b.get("resume") else None

        def run(job):
            r = train_job(job, graph, d, settings, init_state=init)
            r["stopped"] = job.stopped                            # [멈추기]: 거기까지 학습한 모델을 남김
            if (not job.cancelled or job.stopped) and self.data is d:   # 그사이 데이터가 바뀌었거나 다시 시작했으면 버림
                self.model, self.model_graph, self.new_data = r["model"], graph, None
                self.train_settings = dict(b.get("settings") or {}, **({"resume": True} if init is not None else {}))
                self.results = {k: v for k, v in r.items() if k != "model"}
                self.add_run(graph, self.train_settings, self.results,
                             {k: v.detach().cpu().numpy() for k, v in r["model"].state_dict().items()})
            return r

        return {"id": self.jobs.start("train", run)}

    def api_generate(self, b: dict) -> dict:
        """생성: 학습한 모델로 새 샘플 n개 (VAE는 디코더로, 확산은 잡음을 걷어 내며). seed가 같으면 같은 샘플, scale은 뽑는 폭,
        label은 클래스 조건 블록이 있을 때 만들 클래스 번호 (없으면 클래스를 돌아가며)."""
        from blocklearning import generate
        d, m = self.need_data(), self.need_model()
        if d.task != "generation":
            raise UserError("생성 과제로 학습한 모델이 아님", "① 과제에서 '생성'을 고르고 학습하세요")
        seed = int(b.get("seed", 0))
        try:
            X, y, Z = generate.sample(m, d, int(b.get("n", 16)), seed, float(b.get("scale", 1.0)), b.get("label"))
        except ValueError as e:
            raise UserError(str(e), "모델에 'VAE 잠재 공간'이나 '확산 시점' 블록을 두고, VAE면 잠재 공간 뒤 블록이 앞쪽 블록을 쓰지 않게 하세요") from None
        return generate.view(d, X, seed, y, generate.judge(d, Z, y))

    def _resume_state(self, graph: dict, d) -> dict:
        """이어 학습: 지금 모델의 가중치 (같은 데이터·같은 모양의 모델일 때만)."""
        from blocklearning.graph import build_graph_model
        if self.model is None:
            raise UserError("이어서 학습할 모델이 없음", "먼저 [학습 시작]으로 학습하세요")
        want = {k: tuple(v.shape) for k, v in build_graph_model(graph, d.length, d.n_out, image=d.image).state_dict().items()}
        have = self.model.state_dict()
        if want != {k: tuple(v.shape) for k, v in have.items()}:
            raise UserError("지금 모델과 블록 구성이 달라 이어서 학습할 수 없음", "블록을 바꿨다면 [학습 시작]으로 처음부터 학습하세요")
        return {k: v.detach().clone() for k, v in have.items()}

    def picked_sample(self, b: dict) -> dict:
        """샘플 따라가기에 넣을 샘플 하나: 시험 샘플(sample_index) · ⑤에서 추론한 새 데이터(new_index) · 직접 넣은 값(rows, 원래 단위 한 줄).
        돌려줌 {"x": 정규화한 값, "label": 정답 (없으면 None), "name", "index"}."""
        d = self.need_data()
        if b.get("rows") is not None:
            from blocklearning.predict import read_new
            new = read_new(d, {"rows": [b["rows"]]})
            return {"x": new["X"][0], "label": None, "name": new["names"][0], "index": None}
        if b.get("new_index") is not None:
            from blocklearning.predict import new_label
            nd = self.new_data
            if nd is None:
                raise UserError("추론한 새 데이터가 없음", "⑤ 추론에서 새 데이터로 먼저 추론하세요")
            i = int(b["new_index"])
            if not 0 <= i < len(nd["names"]):
                raise UserError(f"새 데이터 번호 {i}이(가) 없음", f"0–{len(nd['names']) - 1} 사이로 고르세요")
            return {"x": nd["X"][i], "label": new_label(d, nd, i), "name": nd["names"][i], "index": i}
        i = int(b.get("sample_index", 0))
        if not 0 <= i < len(d.y_test):
            raise UserError(f"시험 샘플 번호 {i}이(가) 없음", f"0–{len(d.y_test) - 1} 사이로 고르세요")
        return {"x": d.X_test[i], "label": d.label_text("test", i), "name": f"시험 샘플 {i}", "index": i}

    def api_activations(self, b: dict) -> dict:
        """샘플 하나가 블록마다 지난 값·어디를 봤나·어텐션. 시험 샘플·새 데이터·직접 넣은 값 (picked_sample)."""
        from blocklearning.analysis import activations, attention
        import torch
        d, m = self.need_data(), self.need_model()
        p = self.picked_sample(b)
        xn = np.asarray(p["x"], dtype=np.float32)
        x = self.model_input(xn[None])[0]
        with torch.no_grad():
            out = m.eval()(torch.as_tensor(x[None], dtype=torch.float32))[0].numpy()
        probs = None
        if d.task == "detection":
            from blocklearning.detect import SHOW_SCORE, decode
            pred = f"상자 {len(decode(out, d.n_out.grid, d.n_out.n_cls, SHOW_SCORE))}개 찾음"
        elif d.task == "forecast":
            pred = f"다음 {len(out)}개 (첫 값 {float(out[0]):.4g})"
        elif d.task == "segmentation":
            from blocklearning.segment import predicted, states_text
            pred = states_text(predicted(out[None], 1, len(d.classes))[0], d.classes)
        elif d.task == "multi":
            from blocklearning.multi import pred_values
            pred = " · ".join(f"{h['name']} {v:g}" if isinstance(v, float) else f"{h['name']} {v}"
                              for h, v in zip(d.n_out.heads, pred_values(out, d.n_out.heads)))
        elif d.task == "anomaly":
            from blocklearning.anomaly import scores
            s, th = float(scores(out[None], xn[None].astype(np.float64))[0]), (self.results or {}).get("threshold")
            pred = f"이상 점수 {s:.3g}" + ("" if th is None else (" → 이상" if s > th else " → 정상"))
        elif d.task == "classification":
            from blocklearning.predict import top_probs
            pred, probs = d.classes[int(out.argmax())], top_probs(out, d.classes, 5)
        else:
            pred = float(out[0])
        order = check_graph(self.model_graph, d.length, d.n_out, image=d.image).order
        from blocklearning.analysis import saliency
        try:
            sal = saliency(m, x, d)
        except Exception:                                     # 기울기를 못 내는 블록이 있으면 보이지 않음
            sal = None
        r = {"index": p["index"], "name": p["name"], "label": p["label"], "prediction": pred,
             "input": np.round(d.raw(xn.astype(np.float64)), 5).tolist(),
             "values": activations(m, x), "attention": attention(m, x), "order": order, "saliency": sal}
        if probs is not None:
            r["probs"] = probs
        return r

    # ------------------------------------------------------------ 학습 기록
    def data_key(self) -> str:
        """같은 데이터인지 가리는 열쇠 (과제·클래스·모양·샘플 수). 다른 데이터의 기록은 되돌릴 수 없음."""
        d = self.data
        if d is None:
            return ""
        import json
        key = [d.task, [str(c) for c in d.classes], int(d.length), int(len(d.y_train)), int(len(d.y_test)),
               list(d.image) if d.image else None]
        if channels_of(d.length) > 1:
            key.append(channels_of(d.length))
        return json.dumps(key, ensure_ascii=False)

    def add_run(self, graph: dict, settings: dict, results: dict, weights: Optional[dict]) -> int:
        n = max([r["n"] for r in self.runs], default=0) + 1
        self.runs.append({"n": n, "time": datetime.datetime.now().isoformat(timespec="seconds"), "graph": graph,
                          "settings": dict(settings or {}), "results": dict(results or {}), "data_key": self.data_key()})
        if weights is not None:
            self.run_weights[n] = self.strip_frozen(graph, weights)
        while len(self.runs) > MAX_RUNS:                    # 오래된 기록부터 지움 (프로젝트 파일이 너무 커지지 않게)
            old = self.runs.pop(0)
            self.run_weights.pop(old["n"], None)
        self.current_run = n
        return n

    def frozen_names(self, graph) -> set:
        """그래프의 고정(사전학습) 가중치 이름: 프로젝트·학습 기록에 넣지 않음 (열 때 pretrained/에서 다시 채움)."""
        from blocklearning.pretrained import frozen_names
        d = self.data
        if d is None or not graph:
            return set()
        try:
            return frozen_names(graph, d.length, d.n_out, d.image)
        except Exception:                                       # 지금 데이터 모양에서 검사를 통과하지 못하는 그래프
            return set()

    def strip_frozen(self, graph, weights: dict) -> dict:
        frozen = self.frozen_names(graph)
        return {k: v for k, v in weights.items() if k not in frozen} if frozen else weights

    def runs_view(self) -> list:
        key = self.data_key()
        keep = ("metric_name", "test_metric", "train_metric", "epochs_done", "mae")
        return [{"n": r["n"], "time": r["time"], "graph": r["graph"], "settings": r["settings"],
                 "results": {k: r["results"].get(k) for k in keep}, "data_ok": r.get("data_key") == key,
                 "weights": r["n"] in self.run_weights, "current": r["n"] == self.current_run} for r in self.runs]

    def api_runs(self, b: Optional[dict] = None) -> dict:
        return {"runs": self.runs_view()}

    def pick_run(self, n) -> dict:
        """기록 n을 지금 모델로 (그래프·설정·결과). 가중치는 돌려주고, 모델을 만드는 일은 부르는 쪽 (torch 또는 웹 엔진)."""
        run = next((r for r in self.runs if r["n"] == int(n)), None)
        if run is None:
            raise UserError(f"{n}번 학습 기록이 없음", "학습 기록 표를 새로 고치세요")
        self.need_data()
        if run.get("data_key") != self.data_key():
            raise UserError("다른 데이터로 학습한 기록임", "그 기록을 만들 때와 같은 데이터를 불러오세요")
        if run["n"] not in self.run_weights:
            raise UserError("이 기록에는 가중치가 없음", "같은 모델로 다시 학습하세요")
        self.graph = run["graph"]
        self.model_graph, self.train_settings, self.results = run["graph"], dict(run["settings"]), dict(run["results"])
        self.current_run, self.new_data = run["n"], None
        return run

    def api_runs_restore(self, b: dict) -> dict:
        run = self.pick_run(b.get("n"))
        from blocklearning.pretrained import load_state
        d = self.data
        m = build_graph_model(run["graph"], d.length, d.n_out, image=d.image)
        load_state(m, self.run_weights[run["n"]])
        self.model = m.eval()
        return self.restored()

    def restored(self) -> dict:
        return {"graph": self.model_graph, "results": self.results, "train_settings": self.train_settings,
                "trained": True, "runs": self.runs_view()}

    def api_runs_delete(self, b: dict) -> dict:
        n = int(b.get("n"))
        self.runs = [r for r in self.runs if r["n"] != n]
        self.run_weights.pop(n, None)
        if self.current_run == n:
            self.current_run = None
        return {"runs": self.runs_view()}

    # ------------------------------------------------------------ 새 데이터로 예측 · 틀린 샘플 · 블록 미리보기
    def api_predict(self, b: dict) -> dict:
        from blocklearning import predict
        import torch
        d, m = self.need_data(), self.need_model()
        new = predict.read_new(d, _paths(b))
        self.new_data = new
        X = to_model_input(m.kind, new["X"], image=d.image, channels=channels_of(d.length))
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                outs.append(m.eval()(torch.as_tensor(X[i:i + 512], dtype=torch.float32)).numpy())
        return predict.table(d, new["names"], np.concatenate(outs), new["labels"], new["X"], self.results, new.get("truth"))

    def api_anomaly_threshold(self, b: dict) -> dict:
        """이상 탐지 기준값 바꾸기 (④ 결과의 분포에서 고른 값 → ⑤ 예측·샘플 따라가기의 판정)."""
        d = self.need_data()
        if d.task != "anomaly" or not self.results:
            raise UserError("이상 탐지로 학습한 결과가 없음", "④ 학습을 먼저 하세요")
        try:
            th = float(b.get("threshold"))
        except (TypeError, ValueError):
            raise UserError("기준값이 숫자가 아님", "분포 아래 막대를 움직여 고르세요") from None
        self.results["threshold"] = th
        for r in self.runs:                                # 지금 기록에도 (다시 열 때 같은 기준값)
            if r["n"] == self.current_run:
                r["results"]["threshold"] = th
        return {"threshold": th}

    def api_samples(self, b: dict) -> dict:
        """시험 샘플 여럿의 원래 값·정답 (틀린 샘플 모아 보기). cell=[정답, 예측]이면 그 칸의 샘플들."""
        d = self.need_data()
        if b.get("cell") is not None:
            pred = (self.results or {}).get("pred")
            if pred is None or len(pred) != len(d.y_test):
                raise UserError("샘플마다의 예측이 없음", "다시 학습하면 볼 수 있습니다")
            t, p = (int(v) for v in b["cell"])
            idx = [int(i) for i in np.flatnonzero((np.asarray(d.y_test, dtype=int) == t) & (np.asarray(pred) == p))]
        else:
            idx = [int(i) for i in b.get("indices", [])]
        idx = [i for i in idx if 0 <= i < len(d.y_test)]
        show = idx[:MAX_SAMPLES]
        lab = lambda i: d.det["test"][i] if d.task == "detection" else d.label_text("test", i)
        from blocklearning.analysis import _thin
        raw = lambda i: np.round(d.raw(d.X_test[i]), 4).tolist() if d.image else _thin(d.raw(d.X_test[i])[None])[0]
        return {"indices": idx, "items": [{"index": i, "label": lab(i), "input": raw(i)} for i in show]}

    def api_preview(self, b: dict) -> dict:
        """③ 블록 미리보기: 학습 샘플 하나가 그 블록을 지난 값. 학습한 모델과 같으면 학습한 가중치, 아니면 처음 값(무작위)."""
        from blocklearning.analysis import activations
        d = self.need_data()
        graph, node = _graph_from(b), str(b.get("node", ""))
        c = check_graph(graph, d.length, d.n_out, image=d.image)
        if not c.ok:
            i = c.issues[0]
            raise UserError(i.message, i.hint or "③ 모델의 문제를 먼저 고치세요")
        trained = self.model is not None and self.model_graph is not None and _same_model(graph, self.model_graph)
        m = self.model if trained else build_graph_model(graph, d.length, d.n_out, image=d.image, seed=0)
        vals = activations(m, to_model_input(m.kind, d.X_train[:1], image=d.image, channels=channels_of(d.length))[0])
        if node not in vals:
            raise UserError("이 블록의 값을 찾을 수 없음", "블록을 다시 고르세요")
        return {"node": node, "value": vals[node], "trained": trained, "label": _label_of(d, 0)}

    def api_project_save(self, b: dict) -> dict:
        from blocklearning.project import save_project
        b = dict(b, path=_path(b.get("path")))
        d = self.need_data()
        graph = self.model_graph if self.model is not None else self.graph   # 학습한 모델이 있으면 가중치와 같은 그래프
        if not graph:
            raise UserError("저장할 모델이 없음", "③ 모델 단계에서 층을 놓으세요")
        save_project(b["path"], graph, d.source, self.train_settings,
                     None if self.model is None else self.strip_frozen(graph, self.model.state_dict()), self.results, self.selection,
                     runs=self.runs, run_weights=self.run_weights, current_run=self.current_run)
        return {"ok": True, "files": self.wrote(b["path"])}

    def api_project_open(self, b: dict) -> dict:
        from blocklearning.project import open_project
        b = dict(b, path=_path(b.get("path")))
        self.cancel_training()
        if not os.path.isfile(b["path"]):
            raise UserError(f"파일을 찾을 수 없음: {b.get('path')}", "프로젝트 파일(.blocklearning)의 전체 경로를 넣으세요")
        pr = open_project(b["path"])
        summary = self.load_source(pr["data_source"])
        from blocklearning.graph import model_from_saved
        self.graph = pr["graph"]
        self.train_settings, self.results, self.selection = pr["train_settings"], pr["results"], pr["selection"]
        m = model_from_saved(pr, self.data.length, self.data.n_out, image=self.data.image)
        if m is not None:
            self.model, self.model_graph = m.eval(), pr["graph"]
        self.set_runs(pr)
        return {"graph": pr["graph"], "summary": summary, "train_settings": pr["train_settings"],
                "results": pr["results"], "trained": self.model is not None, "selection": self.selection,
                "runs": self.runs_view(), "predict_kinds": self.predict_kinds()}

    def set_runs(self, pr: dict) -> None:
        """연 프로젝트의 학습 기록 (예전 파일은 없음)."""
        self.runs, self.run_weights = list(pr.get("runs") or []), dict(pr.get("run_weights") or {})
        self.current_run = pr.get("current_run")

    def load_source(self, source: dict) -> dict:
        """프로젝트에 적힌 데이터 출처(예시·CSV·이미지 폴더·시계열 폴더)를 다시 불러와 요약을 돌려줌."""
        src = dict(source)
        load = {k: src[k] for k in ("normalize", "test_ratio", "seed", "normal") if k in src}
        if src.get("task") == "forecast" and not src.get("example"):
            return self.api_data_load(dict(load, task="forecast", path=src.get("path"), calendar=src.get("calendar", True),
                                           **{k: src.get(k) for k in ("column", "L", "H", "stride")}))["summary"]
        if src.get("example"):
            return self.api_data_load(dict(load, example=src["example"], calendar=src.get("calendar", True),
                                           **{k: src[k] for k in ("L", "H", "stride") if src.get(k)}))["summary"]
        if src.get("folder") and src.get("det_file"):
            return self.api_data_load(dict(load, folder=src["folder"], size=src.get("size", 64), det_file=src["det_file"]))["summary"]
        if src.get("folder"):
            return self.api_data_load(dict(load, folder=src["folder"], size=src.get("size", 28), task=src.get("task", "auto"), color=src.get("color", False),
                                           label_file=src.get("label_file"), only_labeled=src.get("only_labeled")))["summary"]
        if src.get("series_folder"):
            return self.api_data_load(dict(load, task=src.get("task", "auto"),
                                           **{k: src.get(k) for k in ("series_folder", "column", "length",
                                                                      "resample", "label_file", "only_labeled")}))["summary"]
        return self.api_data_load(dict(load, path=src.get("path"), label_col=src.get("label_col"),
                                       image_width=src.get("image_width"), task=src.get("task", "auto"),
                                       targets=src.get("targets"), skip=src.get("skip"), channels=src.get("channels"),
                                       length=src.get("length")))["summary"]

    def api_project_export(self, b: dict) -> dict:
        from blocklearning.project import export_torch
        b = dict(b, path=_path(b.get("path")))
        m = self.need_model()                            # torch만으로 읽는 TorchScript (.pt)
        info = export_torch(m, self.data, b["path"])
        return {"ok": True, "info": info, "files": self.wrote(b["path"])}

    def _code(self, b: dict, weights_file: Optional[str] = None) -> tuple:
        """요청의 그래프 → (코드, 학습한 모델과 같은 그래프인지, framework)."""
        from blocklearning import codegen
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
        heads = d.n_out.heads if d.task == "multi" else None
        info = {"task": d.task, "classes": list(d.classes) if d.task in ("classification", "detection") else []}
        if d.task == "forecast" and (d.extra or {}).get("calendar"):
            info["calendar"] = list(d.extra["calendar"])
        if heads:
            info["heads"] = heads
        if d.task == "generation" and (d.extra or {}).get("labels"):
            info["gen_labels"] = list(d.extra["labels"])
        settings = b.get("settings") or self.train_settings or {}
        train = {k: settings[k] for k in ("epochs", "lr", "batch_size") if k in settings}
        from blocklearning import losses
        try:
            spec = losses.normalize(settings.get("loss"), d.task, heads)
        except ValueError:
            spec = losses.normalize(None, d.task, heads)
        if (heads or not losses.is_default(spec, d.task, heads)) and d.task != "detection":   # 조립한 손실도 코드로 (탐지는 기본 손실)
            train["loss"] = spec
            K = len(d.classes) if d.task in ("classification", "segmentation") else 0
            if heads:
                cw = losses.multi_class_weights(d.y_train, heads, spec)
                if cw:
                    train["class_weights"] = {k: v.tolist() for k, v in cw.items()}
            elif K and any(t.get("balance") for t in spec):
                train["class_weights"] = losses.class_weights(d.y_train, K).tolist()
        try:
            code = codegen.generate(fw, graph, d.length, d.n_out, d.image, info, train,
                                    weights_file=weights_file if trained and fw == "pytorch" else None)
        except ValueError as e:                              # Keras로 못 바꾸는 블록 등
            raise UserError(str(e), "PyTorch 코드는 모든 블록을 내보낼 수 있습니다") from None
        return code, trained, fw, graph

    def api_code(self, b: dict) -> dict:
        """③·④ [코드로 보기]: PyTorch 또는 TensorFlow Keras 코드."""
        code, trained, fw, _ = self._code(b, weights_file=b.get("weights_file"))
        return {"code": code, "framework": fw, "trained": trained,
                "filename": "model_pytorch.py" if fw == "pytorch" else "model_tensorflow.py"}

    def api_code_save(self, b: dict) -> dict:
        """코드를 .py로 저장. PyTorch이고 학습한 모델이면 가중치(<이름>_weights.npz)도 같은 폴더에."""
        from blocklearning import codegen
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
        return {"files": self.wrote(*files), "trained": trained}

    def api_state(self) -> dict:
        """화면을 새로 고쳐도 이어서 쓰도록 지금 상태 (데이터 요약, 블록, 학습 여부·결과)."""
        return {"summary": None if self.data is None else self.data.summary(),
                "graph": self.graph or self.model_graph, "trained_graph": self.model_graph,
                "trained": self.model is not None,
                "results": self.results, "train_settings": self.train_settings, "selection": self.selection,
                "runs": self.runs_view(), "predict_kinds": self.predict_kinds()}

    def predict_kinds(self) -> list:
        from blocklearning.predict import new_data_kinds
        return [] if self.data is None else new_data_kinds(self.data)

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
                if path == "/api/runs":
                    return 200, _clean(self.api_runs())
                if path == "/api/device":
                    return 200, device_info()
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
            name = ROUTES.get(path) or self.EXTRA_ROUTES.get(path)
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


def device_info() -> dict:
    """학습에 쓸 장치 (화면의 '장치' 칸): NVIDIA GPU가 있으면 그 이름, 없으면 CPU.
    remote: SSH로 접속한 서버에서 실행 중 (저장한 파일을 브라우저로도 내려받게).
    gpus: 번호·이름·메모리(사용 중/전체 GB), auto: '자동'이 고를 GPU 번호 (메모리를 가장 적게 쓰는 것)."""
    from blocklearning.cli import remote_kind
    from blocklearning.devices import gpus, least_used
    gs = gpus()
    return {"gpu": bool(gs), "name": gs[least_used(gs)]["name"] if gs else None, "count": len(gs), "gpus": gs,
            "auto": least_used(gs) if gs else None, "remote": bool(remote_kind())}


def _err(message: str, hint: str = "", detail: str = "", issues=None) -> dict:
    return {"error": {"message": message, "hint": hint, "detail": detail, "issues": issues or []}}


ROUTES = {
    "/api/data/load": "api_data_load", "/api/model/check": "api_model_check", "/api/train": "api_train",
    "/api/activations": "api_activations",
    "/api/anomaly/threshold": "api_anomaly_threshold",
    "/api/project/save": "api_project_save", "/api/project/open": "api_project_open",
    "/api/project/export": "api_project_export", "/api/task/select": "api_task_select",
    "/api/data/scan_folder": "api_data_scan_folder", "/api/data/columns": "api_data_columns", "/api/code": "api_code", "/api/code/save": "api_code_save",
    "/api/runs/restore": "api_runs_restore", "/api/runs/delete": "api_runs_delete", "/api/predict": "api_predict", "/api/generate": "api_generate",
    "/api/samples": "api_samples", "/api/preview": "api_preview",
}
MAX_RUNS = 20             # 학습 기록은 최근 이만큼 (가중치까지 프로젝트 파일에 들어감)
DEEP_EPOCHS, DEEP_LR = 40, 0.003                 # 라벨링 딥러닝 제안: 적은 정답으로 빠르게
DEEP_TEMPLATE = {"series_folder": "cnn", "image_folder": "cnn2d", "table": "mlp"}


def deep_graph(kind: str) -> dict:
    """라벨링 딥러닝 제안에 쓰는 대표 모델 그래프 (데이터 종류별 첫 추천 모델)."""
    want = DEEP_TEMPLATE[kind]
    for f in tasks.catalog_view()["families"]:
        for s in f["subtasks"]:
            for t in s.get("templates", []):
                if t["id"] == want:
                    return t["graph"]
    raise UserError("대표 모델을 찾을 수 없음", "")


def graph_name(kind: str) -> str:
    return {"series_folder": "1D CNN", "image_folder": "작은 CNN", "table": "MLP"}[kind]
MAX_SAMPLES = 24          # 틀린 샘플 모아 보기에서 한 번에 보이는 수


def _label_of(d, i: int):
    return d.label_text("train", i)


def _paths(b: dict) -> dict:
    """예측 요청의 경로들에서 따옴표·공백 빼기."""
    return {k: (_path(v) if k in ("path", "folder", "series_folder") else v) for k, v in b.items()}


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

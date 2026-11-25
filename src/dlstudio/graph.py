"""그래프: 화면에서 층을 선으로 이은 모델(JSON v2)을 검사하고, 노드마다 출력 모양을 계산하고, torch 모델로 만든다.

그래프 = {"version": 2, "nodes": [{"id", "type", ...설정, "inputs": [id...]}]}.
입력 노드 id는 "input", Repeat 묶음 안에서 묶음의 입력은 "in". 합치기는 add(모양이 모두 같음)·concat(채널 방향).
층마다의 모양 규칙은 blocks._Walker.block을 그대로 쓴다. 검사를 통과한 그래프는 모델 만들기가 실패하지 않는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from dlstudio import layers as X
from dlstudio.blocks import BLOCK_TYPES, NAMES, RNN_TYPES, _Walker

MERGE_TYPES = ("add", "concat") + X.NEW_MERGES             # 입력 2개 이상을 합침 (가중치 없음)
SAME_SHAPE_MERGES = ("add",) + X.NEW_MERGES                 # 모양이 모두 같아야 함
LAYER_TYPES = tuple(t for t in BLOCK_TYPES if t not in ("branch", "residual", "repeat")) + MERGE_TYPES + ("repeat", "input")
ID_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
INPUT, BODY_IN = "input", "in"
_CONFIG_SKIP = ("id", "inputs", "body")
GRAPH_NAMES = dict(NAMES, add="Add", concat="Concat", input="Input", repeat="Repeat", **{k: X.EXTRA_NAMES[k] for k in X.NEW_MERGES})


@dataclass
class GraphIssue:
    node: str
    message: str
    hint: str

    def to_dict(self) -> dict:
        return {"node": self.node, "message": self.message, "hint": self.hint}


@dataclass
class GraphChecked:
    issues: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)        # 노드 id → 출력 모양 (배치 제외)
    order: list = field(default_factory=list)         # 위상 순서. Repeat는 [id, 안쪽 순서]
    kind: str = "cnn"

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "issues": [i.to_dict() for i in self.issues],
                "shapes": {k: list(v) for k, v in self.shapes.items()}, "order": self.order}


@dataclass
class _Scope:
    """그래프 한 층위 (맨 위 또는 Repeat 몸체): 노드, 위상 순서, 문제 있는 노드, 출구."""
    nodes: dict                       # id → 노드 (입력 노드 제외)
    entry: str                        # "input" 또는 "in"
    order: list = field(default_factory=list)
    bad: set = field(default_factory=set)
    exit: Optional[str] = None
    bodies: dict = field(default_factory=dict)        # repeat id → _Scope


def _config(node: dict) -> dict:
    return {k: v for k, v in node.items() if k not in _CONFIG_SKIP}


class _Checker:
    def __init__(self, length: int, n_out: int, image):
        self.length, self.n_out, self.image = length, n_out, image
        self.issues: list = []
        self.shapes: dict = {}

    def issue(self, node, message, hint=""):
        self.issues.append(GraphIssue(str(node), message, hint))

    # ------------------------------------------------------------ 1. 형식
    def form(self, graph) -> Optional[list]:
        """형식이 맞으면 맨 위 노드 목록, 아니면 None (문제를 남김)."""
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not graph["nodes"]:
            self.issue("", "모델 그래프 형식이 아님", "대표 모델을 불러오거나 층을 놓으세요")
            return None
        seen: set = set()
        ok = self.form_list(graph["nodes"], top=True, seen=seen)
        top = graph["nodes"]
        if ok:
            ins = [n for n in top if n["type"] == "input"]
            outs = [n for n in top if n["type"] == "output"]
            if len(ins) != 1 or ins[0]["id"] != INPUT:
                self.issue("", "입력(Input) 노드가 하나 있어야 함", "대표 모델을 불러와 다시 시작하세요")
                ok = False
            if not outs:
                self.issue("", "출력 노드가 없음", "맨 아래에 Output을 놓으세요")
                ok = False
            for n in outs[1:]:
                self.issue(n["id"], "출력 노드는 하나만", "출력 노드를 하나만 남기세요")
                ok = False
        return top if ok else None

    def form_list(self, nodes, top: bool, seen: set) -> bool:
        ok = True
        for n in nodes:
            if not isinstance(n, dict):
                self.issue("", f"층 형식이 아님: {str(n)[:40]!r}", "")
                ok = False
                continue
            nid, t = n.get("id"), n.get("type")
            if not isinstance(nid, str) or not ID_RE.match(nid) or nid == BODY_IN or (nid == INPUT) != (t == "input"):
                self.issue(str(nid)[:40], f"층 이름(id) {str(nid)[:40]!r}을(를) 쓸 수 없음", "영문·숫자·밑줄 1–40자")
                ok = False
                continue
            if nid in seen:
                self.issue(nid, f"층 이름(id) {nid}이(가) 겹침", "층마다 다른 id를 쓰세요")
                ok = False
                continue
            seen.add(nid)
            if t not in LAYER_TYPES:
                self.issue(nid, f"알 수 없는 층 {t!r}", "팔레트에 있는 층만 쓸 수 있습니다")
                ok = False
                continue
            if t == "input":
                if not top:
                    self.issue(nid, "묶음 안에는 Input을 둘 수 없음", "")
                    ok = False
                continue
            ins = n.get("inputs")
            if not isinstance(ins, list) or not all(isinstance(x, str) for x in ins):
                self.issue(nid, "입력 목록(inputs) 형식이 아님", "")
                ok = False
            if t == "repeat":
                body = n.get("body")
                if not isinstance(body, dict) or not isinstance(body.get("nodes"), list) or not body["nodes"]:
                    self.issue(nid, "Repeat 안이 비어 있음", "안에 층을 넣으세요")
                    ok = False
                    continue
                if not self.form_list(body["nodes"], top=False, seen=seen):
                    ok = False
                    continue
                inner = [x["type"] for x in body["nodes"]]
                if any(x in RNN_TYPES + ("output",) + X.AUG_TYPES for x in inner):
                    self.issue(nid, "Repeat 안에는 LSTM·GRU·SimpleRNN·증강·Output을 넣을 수 없음", "그 층을 Repeat 밖으로 옮기세요")
                    ok = False
        return ok

    # ------------------------------------------------------------ 2. 연결
    def scope(self, nodes: list, entry: str) -> _Scope:
        sc = _Scope({n["id"]: n for n in nodes if n["type"] != "input"}, entry)
        known = set(sc.nodes) | {entry}
        users: dict = {k: [] for k in known}
        for nid, n in sc.nodes.items():
            t, ins = n["type"], n.get("inputs") or []
            for x in ins:
                if x not in known:
                    self.issue(nid, f"없는 층 {x!r}을(를) 입력으로 가리킴", "선을 다시 이으세요")
                    sc.bad.add(nid)
                elif x in sc.nodes and sc.nodes[x]["type"] == "output":
                    self.issue(nid, "출력 뒤에는 층을 이을 수 없음", "출력은 맨 아래에 하나만")
                    sc.bad.add(nid)
                else:
                    users[x].append(nid)
            if t in MERGE_TYPES:
                if len(ins) < 2:
                    self.issue(nid, f"{GRAPH_NAMES[t]}: 합칠 입력이 2개 이상이어야 함", "다른 층의 출력 손잡이를 끌어 이 상자에 놓으세요")
                    sc.bad.add(nid)
            elif t in X.TWO_INPUT:
                if len(ins) != 2:
                    self.issue(nid, f"{GRAPH_NAMES[t]}: 입력이 2개여야 함 (지금 {len(ins)}개) — 첫째 = 질문, 둘째 = 참고",
                               "다른 층의 출력 손잡이를 끌어 이 상자에 놓으세요")
                    sc.bad.add(nid)
            elif not ins:
                self.issue(nid, "입력이 연결되지 않음", "위 층의 출력 손잡이를 끌어 이 상자에 놓으세요")
                sc.bad.add(nid)
            elif len(ins) > 1:
                self.issue(nid, "입력은 하나만 (합치려면 Add·Concat)", "Add나 Concat 상자를 넣어 합치세요")
                sc.bad.add(nid)
        # 위상 순서 (Kahn). 남는 노드 = 순환
        deg = {k: sum(1 for x in sc.nodes[k].get("inputs") or [] if x in sc.nodes) for k in sc.nodes}
        ready = [k for k in sc.nodes if deg[k] == 0]
        ready.sort(key=list(sc.nodes).index)
        while ready:
            k = ready.pop(0)
            sc.order.append(k)
            for u in sorted(set(users[k]), key=list(sc.nodes).index):
                deg[u] -= sum(1 for x in sc.nodes[u]["inputs"] if x == k)
                if deg[u] == 0:
                    ready.append(u)
        for k in sc.nodes:
            if k not in sc.order:
                self.issue(k, "선이 돌아서 제자리로 옴 (순환)", "아래에서 위로 잇는 선을 지우세요")
                sc.bad.add(k)
        sinks = [k for k in sc.order if not users[k]]
        if entry == INPUT:
            outs = [k for k in sc.nodes if sc.nodes[k]["type"] == "output"]
            sc.exit = outs[0] if outs else None
            reach = self.ancestors(sc, sc.exit) if sc.exit else set()
            for k in sc.order:
                if k not in reach and k != sc.exit:
                    self.issue(k, "출력까지 이어지지 않음", "이 층의 출력을 다음 층에 잇거나 지우세요")
                    sc.bad.add(k)
            self.rnn_rule(sc, users)
        else:
            sc.exit = sinks[0] if len(sinks) == 1 else None
        for k in sc.order:
            if sc.nodes[k]["type"] == "repeat" and k not in sc.bad:
                body = self.scope(sc.nodes[k]["body"]["nodes"], BODY_IN)
                sc.bodies[k] = body
                if body.exit is None:
                    self.issue(k, "묶음의 출구가 하나여야 함", "묶음 안의 마지막 층 하나만 밖으로 나가게 하세요")
                    sc.bad.add(k)
        return sc

    @staticmethod
    def ancestors(sc: _Scope, nid: str) -> set:
        out, stack = set(), [nid]
        while stack:
            k = stack.pop()
            if k in out or k not in sc.nodes:
                continue
            out.add(k)
            stack += sc.nodes[k].get("inputs") or []
        return out

    def rnn_rule(self, sc: _Scope, users: dict) -> None:
        """증강(학습 때만 그림·신호를 바꿈)은 Input 바로 뒤나 다른 증강 뒤에만 (배운 값을 지난 뒤에는 뜻이 없음)."""
        for k in sc.order:
            n = sc.nodes[k]
            if n["type"] in X.AUG_TYPES and not all(x == INPUT or (x in sc.nodes and sc.nodes[x]["type"] in X.AUG_TYPES)
                                                    for x in n.get("inputs") or []):
                self.issue(k, f"{NAMES[n['type']]}은(는) Input 바로 뒤(다른 증강 블록 뒤)에만 쓸 수 있음",
                           "증강 블록을 Input 바로 아래로 옮기세요")
                sc.bad.add(k)

    # ------------------------------------------------------------ 3. 모양
    def shape_pass(self, sc: _Scope, in_shape) -> Optional[tuple]:
        """위상 순서로 모양을 계산해 self.shapes에 남기고 출구 모양을 돌려줌."""
        vals = {sc.entry: in_shape}
        for k in sc.order:
            n = sc.nodes[k]
            ins = [vals.get(x) for x in n.get("inputs") or []]
            if k in sc.bad or not ins or any(s is None for s in ins):
                vals[k] = None
                continue
            t = n["type"]
            if t in SAME_SHAPE_MERGES:
                s = self.add_shape(k, ins, GRAPH_NAMES[t])
            elif t in X.TWO_INPUT:
                w = _Walker(self.n_out, image=bool(self.image))
                s = X.two_shape(lambda m, h: self.issue(k, m, h), n, ins[0], ins[1]) if w.values_ok(_config(n), (k,)) else None
                for i in w.issues:
                    self.issue(k, i.message, i.hint)
            elif t == "concat":
                s = self.concat_shape(k, ins)
            elif t == "repeat":
                s = self.repeat_shape(k, n, sc.bodies[k], ins[0])
            else:
                w = _Walker(self.n_out, image=bool(self.image))
                s = w.block(_config(n), ins[0], (k,), "top", first=True, last=True)
                for i in w.issues:
                    self.issue(k, i.message, i.hint)
                if w.issues:
                    s = None
            vals[k] = None if s is None else tuple(s)
            if s is not None:
                self.shapes[k] = tuple(s)
        return vals.get(sc.exit) if sc.exit else None

    def add_shape(self, k, ins, name="Add"):
        if any(tuple(s) != tuple(ins[0]) for s in ins[1:]):
            shown = "과(와) ".join(str(tuple(s)) for s in ins)
            self.issue(k, f"{name}: 합칠 값들의 모양이 다름 {shown}",
                       "모양을 맞추세요: 채널이나 길이가 다르면 지름길에 1×1 Conv(필요하면 stride 2)를 넣으세요")
            return None
        return tuple(ins[0])

    def concat_shape(self, k, ins):
        if all(len(s) == 1 for s in ins):                      # 숫자 묶음끼리: 이어 붙임
            return (sum(s[0] for s in ins),)
        if len({len(s) for s in ins}) != 1 or len({tuple(s[1:]) for s in ins}) != 1:
            shown = "과(와) ".join(str(tuple(s)) for s in ins)
            self.issue(k, f"Concat: 채널 말고 크기가 다름 {shown}", "길이(크기)가 같은 값끼리 합치세요 (AvgPool·stride를 맞추기)")
            return None
        return (sum(s[0] for s in ins),) + tuple(ins[0][1:])

    def repeat_shape(self, k, n, body: _Scope, shape):
        w = _Walker(self.n_out, image=bool(self.image))
        if not w.values_ok(_config(n), (k,)):
            for i in w.issues:
                self.issue(k, i.message, i.hint)
            return None
        start = len(self.issues)
        for _ in range(n["times"]):                     # 펼친 것과 같은 규칙. 모양은 마지막 반복 기준
            del self.issues[start:]
            shape = self.shape_pass(body, shape)
            if shape is None:
                break
        return shape


def _order_out(sc: _Scope) -> list:
    return [[k, _order_out(sc.bodies[k])] if k in sc.bodies else k for k in sc.order]


def check_graph(graph, length: int, n_out: int, image: tuple = None, rules: bool = True) -> GraphChecked:
    """그래프 검사와 노드마다 출력 모양. length: 데이터 길이(이미지면 H·W), n_out: 클래스 수(회귀면 1), image: (H, W)."""
    c = _Checker(length, n_out, image)
    kind = "image" if image else "cnn"
    top = c.form(graph)
    if top is None:
        return GraphChecked(c.issues, {}, [], kind)
    sc = c.scope(top, INPUT)
    if not image and any(sc.nodes[k]["type"] in RNN_TYPES and INPUT in (sc.nodes[k].get("inputs") or []) for k in sc.nodes):
        kind = "rnn"
    start = (1,) + tuple(image) if image else (1, length)
    c.shapes[INPUT] = start
    c.shape_pass(sc, start)
    if rules and not c.issues:                             # 모양이 다 맞을 때만 (계획을 만들어 봄)
        _distance_rule(c, sc, graph, length, n_out, image)
    return GraphChecked(c.issues, c.shapes, [INPUT] + _order_out(sc), kind)


def _distance_rule(c, sc, graph, length, n_out, image) -> None:
    """출력 '정상과의 거리'는 정상 특징의 분포를 그대로 씀 → 그 앞에 학습하는 층이 있으면 안 됨 (학습할 손실이 없음)."""
    out = next((k for k, n in sc.nodes.items() if n["type"] == "output" and n.get("head") == "distance"), None)
    if out is None:
        return
    from dlstudio.codegen import make_plan, param_specs
    plan = make_plan(graph, length, n_out, image)
    if any(p[2] != "frozen" for s in plan.steps if s.type != "output" for p in param_specs(s)):
        c.issue(out, "'정상과의 거리' 출력 앞에는 학습하는 층을 둘 수 없음",
                "사전학습 블록(MobileNetV3-Small) 바로 뒤에 두세요 (학습하는 층을 쓰려면 출력을 보통 출력으로)")


# ---------------------------------------------------------------- torch 모델 만들기

FROZEN_MODULES = ("MobileNetV3Small",)                     # 학습하지 않는 사전학습 모듈 (nn_extra)


def _key(nid: str) -> str:
    """ModuleDict 키 (nn.Module 속성 이름과 겹치지 않게 앞에 m_)."""
    return "m_" + nid


def _graph_net_class():
    import torch
    from torch import nn

    class GraphNet(nn.Module):
        """그래프를 위상 순서대로 실행하는 모델. plan: [("layer"|"add"|"concat", id, 입력들) 또는 ("repeat", id, 입력들, 출구, [반복마다 plan])].
        mods: id → 모듈 (Repeat는 반복마다 몸체 ModuleDict의 ModuleList). block_modules: id → 모듈 (등록 안 한 dict)."""

        def __init__(self, plan, mods, kind: str, output_id: str, block_modules: dict):
            super().__init__()
            self.plan = plan
            self.mods = mods
            self.kind = kind
            self.output_id = output_id
            self.block_modules = block_modules
            self.start = None                                  # 특징을 미리 뽑아 둔 학습: 입력 대신 이 노드의 출력을 받음

        def distance_head(self):
            """출력이 '정상과의 거리'면 (그 모듈, 들어가는 노드 id), 아니면 None."""
            m = self.mods[_key(self.output_id)] if _key(self.output_id) in self.mods else None
            if type(m).__name__ != "NormalDistance":
                return None
            step = next(st for st in self.plan if st[1] == self.output_id)
            return m, step[2][0]

        def frozen_head(self):
            """입력 바로 뒤의 고정(사전학습) 블록 하나만 입력을 쓰면 그 노드 id (학습 때 특징을 한 번만 뽑아 둠), 아니면 None."""
            if not self.plan or self.plan[0][0] != "layer" or self.plan[0][2] != [INPUT]:
                return None
            nid = self.plan[0][1]
            if type(self.mods[_key(nid)]).__name__ not in FROZEN_MODULES:
                return None
            return nid if not any(INPUT in st[2] for st in self.plan[1:]) else None

        @staticmethod
        def _exec(plan, vals: dict, mods, rec):
            for step in plan:
                kind, nid, ins = step[0], step[1], step[2]
                if nid in vals:                                # 미리 뽑아 둔 특징 (start)
                    continue
                if kind == "layer":
                    v = mods[_key(nid)](vals[ins[0]])
                elif kind == "layer2":
                    v = mods[_key(nid)](vals[ins[0]], vals[ins[1]])
                elif kind in ("add", "multiply", "average", "maximum"):
                    v = vals[ins[0]]
                    for j in ins[1:]:
                        v = v + vals[j] if kind in ("add", "average") else v * vals[j] if kind == "multiply" else torch.maximum(v, vals[j])
                    if kind == "average":
                        v = v / len(ins)
                elif kind == "concat":
                    v = torch.cat([vals[j] for j in ins], dim=1)
                else:                                          # repeat
                    v = vals[ins[0]]
                    for body_mods, body_plan in zip(mods[_key(nid)], step[4]):
                        inner = {BODY_IN: v}
                        GraphNet._exec(body_plan, inner, body_mods, rec)
                        v = inner[step[3]]
                vals[nid] = v
                if rec is not None:
                    rec[nid] = v
            return vals

        def forward(self, x):
            if self.start is not None:
                return self._exec(self.plan, {self.start: x}, self.mods, None)[self.output_id]
            if self.kind == "rnn":                             # 순환 모델의 입력 (N, L, 1) → 층들이 쓰는 (N, 1, L)
                x = x.transpose(1, 2)
            return self._exec(self.plan, {INPUT: x}, self.mods, None)[self.output_id]

        def trace(self, x) -> dict:
            """모든 노드의 출력 (Repeat 안은 마지막 반복). 샘플 따라가기용."""
            rec: dict = {INPUT: x}
            self._exec(self.plan, {INPUT: x.transpose(1, 2) if self.kind == "rnn" else x}, self.mods, rec)
            return rec

    return GraphNet


def build_graph_model(graph: dict, length: int, n_out: int, seed: int = 0, image: tuple = None):
    """그래프 JSON → GraphNet (float32). 검사를 통과하지 못하면 ValueError(첫 문제)."""
    import torch
    from torch import nn
    from dlstudio.blocks import extra_layer, make_layer

    checked = check_graph(graph, length, n_out, image=image)
    if not checked.ok:
        raise ValueError(checked.issues[0].message)
    c = _Checker(length, n_out, image)
    sc = c.scope(c.form(graph), INPUT)
    torch.manual_seed(seed)
    block_modules: dict = {}

    def build(scope: _Scope, in_shape):
        """이 층위를 in_shape로 다시 계산해 (plan, ModuleDict)."""
        c.shape_pass(scope, in_shape)
        shape_of = lambda x: in_shape if x == scope.entry else c.shapes[x]
        plan, mods = [], nn.ModuleDict()
        for k in scope.order:
            n = scope.nodes[k]
            t, ins = n["type"], list(n.get("inputs") or [])
            if t in MERGE_TYPES:
                plan.append((t, k, ins))
                continue
            if t == "repeat":
                reps, steps, s = nn.ModuleList(), [], shape_of(ins[0])
                body = scope.bodies[k]
                for _ in range(n["times"]):                     # 가중치는 반복마다 따로, 모양도 반복마다 다시
                    p, m = build(body, s)
                    reps.append(m)
                    steps.append(p)
                    s = c.shapes[body.exit]
                mods[_key(k)] = reps
                plan.append(("repeat", k, ins, body.exit, steps))
                c.shapes[k] = tuple(s)
                continue
            if t in X.TWO_INPUT:
                m = extra_layer(_config(n), shape_of(ins[0]), shape_of(ins[1]))
            else:
                m = make_layer(_config(n), shape_of(ins[0]), n_out)
            mods[_key(k)] = m
            block_modules[k] = m
            plan.append(("layer2" if t in X.TWO_INPUT else "layer", k, ins))
        return plan, mods

    plan, mods = build(sc, (1,) + tuple(image) if image else (1, length))
    return _graph_net_class()(plan, mods, checked.kind, sc.exit, block_modules).float()


def output_linear(net):
    """출력 노드의 Linear (회귀 정답 되돌림을 접어 넣을 곳)."""
    m = net.mods[_key(net.output_id)]
    return m[-1] if hasattr(m, "__getitem__") else m


# ---------------------------------------------------------------- 예전 블록 목록(v1) 옮기기

def _bid(path: tuple) -> str:
    return "b" + "_".join(map(str, path))


def _times(b: dict) -> int:
    """예전 Repeat의 반복 수. 규칙(1–10 정수) 밖이면 1 (옮긴 그래프의 검사가 값 문제를 알림, 큰 값으로 멈추지 않게)."""
    t = b.get("times")
    return t if isinstance(t, int) and not isinstance(t, bool) and 1 <= t <= 10 else 1


def _lanes(b: dict) -> list:
    lanes = b.get("branches")
    return lanes if isinstance(lanes, list) else []


def _channels(seq: list, C):
    """블록 목록을 지난 뒤의 채널(또는 너비) 수. 모르면 None (옮긴 그래프의 검사가 문제를 찾음)."""
    for b in seq if isinstance(seq, list) else []:
        t = b.get("type") if isinstance(b, dict) else None
        if t in ("conv", "conv2d"):
            C = b.get("channels")
        elif t in ("dense",):
            C = b.get("units")
        elif t in RNN_TYPES:
            C = b.get("hidden")
        elif t == "branch":
            lanes = [_channels(lane, C) for lane in _lanes(b)]
            C = sum(lanes) if lanes and None not in lanes else None
        elif t == "residual":
            C = _channels(b.get("body") or [], C)
        elif t == "repeat":
            for _ in range(_times(b)):
                C = _channels(b.get("body") or [], C)
        elif t == "flatten":
            C = None
    return C


def from_blocks(blocks: list) -> dict:
    """블록 목록(v1) → 그래프(v2). 노드 id는 블록 위치에서: (2, 0, 1) → b2_0_1.
    Residual → 몸체 + <id>_add(채널이 바뀌면 지름길 1×1 conv <id>_sc), Concat branches → 갈래 + <id>_cat, Repeat → 몸체를 재귀로."""
    image = any(t in json_types(blocks) for t in ("conv2d", "pool2d", "global_avg2d"))

    def lower(seq: list, entry: str, prefix: tuple, out: list, C) -> tuple:
        cur = entry
        for i, b in enumerate(seq if isinstance(seq, list) else []):
            path = prefix + (i,)
            nid = _bid(path)
            if not isinstance(b, dict):
                out.append({"id": nid, "type": str(b), "inputs": [cur]})
                cur = nid
                continue
            t = b.get("type")
            if t == "residual":
                ex, C2 = lower(b.get("body") or [], cur, path, out, C)
                sc = cur
                if C2 is not None and C is not None and C2 != C:
                    out.append({"id": nid + "_sc", "type": "conv2d" if image else "conv", "channels": C2, "kernel": 1,
                                "stride": 1, "inputs": [cur]})
                    sc = nid + "_sc"
                out.append({"id": nid + "_add", "type": "add", "inputs": [ex, sc]})
                cur, C = nid + "_add", C2
            elif t == "branch":
                ends, total = [], 0
                for j, lane in enumerate(_lanes(b)):
                    ex, Cj = lower(lane, cur, path + (j,), out, C)
                    ends.append(ex)
                    total = None if total is None or Cj is None else total + Cj
                out.append({"id": nid + "_cat", "type": "concat", "inputs": ends})
                cur, C = nid + "_cat", total
            elif t == "repeat" and _unroll(b, C):              # 반복마다 구조가 다름 (첫 반복만 지름길) → 펼침
                for k in range(_times(b)):
                    cur, C = lower(b.get("body") or [], cur, path + (f"r{k}",), out, C)
            elif t == "repeat":
                body: list = []
                lower(b.get("body") or [], BODY_IN, path, body, C)
                out.append({"id": nid, "type": "repeat", "times": b.get("times"), "inputs": [cur], "body": {"nodes": body}})
                cur, C = nid, _channels([b], C)
            else:
                out.append(dict({k: v for k, v in b.items() if k not in _CONFIG_SKIP}, id=nid, inputs=[cur]))
                cur, C = nid, _channels([b], C)
        return cur, C

    nodes: list = []
    lower(blocks, INPUT, (), nodes, 1)
    return {"version": 2, "nodes": [{"id": INPUT, "type": "input"}] + nodes}


def _unroll(b: dict, C) -> bool:
    """예전 Repeat의 몸체가 채널 수를 바꾸고 Residual이 있으면, 반복마다 지름길 유무가 달라 묶음 하나로 못 씀."""
    if _times(b) < 2 or "residual" not in json_types(b.get("body")):
        return False
    C1 = _channels(b.get("body") or [], C)
    return C1 is not None and C is not None and C1 != C


def json_types(blocks) -> set:
    """블록 목록 안(묶음·갈래 포함)의 모든 type."""
    out: set = set()
    for b in blocks if isinstance(blocks, list) else []:
        if isinstance(b, dict):
            out.add(b.get("type"))
            out |= json_types(b.get("body"))
            for lane in _lanes(b):
                out |= json_types(lane)
    return out


def model_from_saved(project: dict, length: int, n_out: int, image: tuple = None):
    """open_project 결과 → 가중치를 읽은 GraphNet (가중치가 없으면 None)."""
    state = project.get("state_dict")
    if state is None:
        return None
    from dlstudio.pretrained import load_state
    net = build_graph_model(project["graph"], length, n_out, image=image)
    load_state(net, state)
    return net

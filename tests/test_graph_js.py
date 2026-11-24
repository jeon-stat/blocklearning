"""그래프 편집기 화면(static/graph.js)의 순수 함수: 자동 배치와 편집 동작. node가 있으면 실행."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import dlstudio

STATIC = Path(dlstudio.__file__).parent / "static"
GRAPH = STATIC / "graph.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")

HELPERS = r"""
const N = (id, type, inputs, kw) => Object.assign({ id, type, inputs }, kw || {});
const G = (...nodes) => ({ version: 2, nodes: [{ id: "input", type: "input" }].concat(nodes) });
const byId = (g, id) => { let f = null; const walk = (ns) => ns.forEach((n) => { if (n.id === id) f = n; if (n.body) walk(n.body.nodes); }); walk(g.nodes); return f; };
const ids = (g) => { const out = []; const walk = (ns) => ns.forEach((n) => { out.push(n.id); if (n.body) walk(n.body.nodes); }); walk(g.nodes); return out.sort(); };
const chain = () => G(N("a", "conv", ["input"], { channels: 8, kernel: 3 }), N("b", "act", ["a"], { kind: "relu" }),
  N("c", "conv", ["b"], { channels: 8, kernel: 3 }), N("g", "global_avg", ["c"]), N("o", "output", ["g"]));
const res = () => G(N("a", "conv", ["input"], { channels: 8, kernel: 3 }), N("b", "act", ["a"], { kind: "relu" }),
  N("s", "add", ["b", "input"]), N("g", "global_avg", ["s"]), N("o", "output", ["g"]));
let seed = 1; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
"""


def _run(js: str):
    code = (f"const L = require({json.dumps(str(GRAPH))});\n{HELPERS}\n"
            f"const out = (() => {{ {js} }})();\nprocess.stdout.write(JSON.stringify(out));")
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_layout_chain_is_one_column():
    r = _run("const l = L.layoutGraph(chain()); return l.nodes.map((n) => [n.id, n.x + n.w / 2, n.y]);")
    xs = {round(x, 3) for _, x, _ in r}
    assert len(xs) == 1
    ys = [y for _, _, y in r]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    assert [i for i, _, _ in r][0] == "input"


def test_layout_skip_edge_goes_around():
    r = _run("""const l = L.layoutGraph(res());
      const right = Math.max(...l.nodes.filter((n) => ["a", "b"].includes(n.id)).map((n) => n.x + n.w));
      const skip = l.edges.find((e) => e.from === "input" && e.to === "s");
      return { right, xs: skip.points.map((p) => p[0]), n: l.edges.length };""")
    assert max(r["xs"]) > r["right"]
    assert r["n"] == 6                                  # input→a, a→b, b→s, input→s, s→g, g→o


def test_layout_no_overlap_random():
    r = _run("""const bad = [];
      for (let t = 0; t < 30; t++) {
        const nodes = [];
        for (let i = 0; i < 20; i++) {
          const prev = ["input"].concat(nodes.map((n) => n.id));
          const pick = () => prev[Math.floor(rnd() * prev.length)];
          if (rnd() < 0.25 && prev.length > 2) nodes.push(N("m" + i, rnd() < 0.5 ? "add" : "concat", [pick(), pick()]));
          else if (rnd() < 0.1) nodes.push(N("r" + i, "repeat", [pick()], { times: 2, body: { nodes: [
            N("r" + i + "a", "conv", ["in"], { channels: 4, kernel: 3 }), N("r" + i + "b", "add", ["r" + i + "a", "in"])] } }));
          else nodes.push(N("n" + i, "conv", [prev[prev.length - 1 - Math.floor(rnd() * 2)] || "input"], { channels: 4, kernel: 3 }));
        }
        nodes.push(N("o", "output", [nodes[nodes.length - 1].id]));
        const l = L.layoutGraph(G(...nodes));
        const tops = l.nodes.filter((n) => !n.inGroup);
        for (const a of tops) for (const b of tops) {
          if (a === b || a.layer !== b.layer) continue;
          if (a.x < b.x + b.w && b.x < a.x + a.w) bad.push([t, a.id, b.id]);
        }
        for (const gr of l.groups) for (const n of l.nodes.filter((n) => n.inGroup === gr.id))
          if (n.x < gr.x || n.y < gr.y || n.x + n.w > gr.x + gr.w || n.y + n.h > gr.y + gr.h) bad.push([t, "group", gr.id, n.id]);
        if (!(l.width > 0 && l.height > 0)) bad.push([t, "size"]);
      }
      return bad;""")
    assert r == []


def test_insert_on_edge():
    r = _run("""const g = L.insertOnEdge(chain(), "b", "c", { id: "x", type: "bn" });
      return [byId(g, "x").inputs, byId(g, "c").inputs, byId(chain(), "c").inputs];""")
    assert r == [["b"], ["x"], ["b"]]                   # 원본은 그대로


def test_connect_replaces_existing_input():
    """보통 층에 선을 이으면 기존 선을 바꿈 (Add를 만들지 않음). 합치기 상자(Add·Concat)에 이으면 입력을 더함."""
    r = _run("""const { graph, error } = L.connect(chain(), "a", "c");
      return [error, byId(graph, "c").inputs, graph.nodes.some((n) => n.type === "add")];""")
    assert r == [None, ["a"], False]
    r = _run("""const { graph } = L.connect(res(), "a", "s"); return byId(graph, "s").inputs;""")
    assert r == ["b", "input", "a"]
    r = _run("""return L.connect(chain(), "b", "c").error;""")
    assert r                                            # 이미 이어져 있음


def test_add_connected_and_feeding():
    """선을 빈 곳에 놓아 층을 고르면: 출력에서 끌었으면 그 뒤에 이어진 새 층, 입력에서 끌었으면 그 앞을 채우는 새 층."""
    r = _run("""const a = L.addConnected(chain(), "b", { type: "bn" }, [400, 200]);
      const n = byId(a.graph, a.id);
      const f = L.addFeeding(chain(), "c", { type: "pool", size: 2 }, [400, 200]);
      const m = byId(f.graph, f.id);
      const fm = L.addFeeding(res(), "s", { type: "bn" }, [0, 0]);
      return [n.type, n.inputs, n.pos, m.inputs, byId(f.graph, "c").inputs[0] === f.id, byId(fm.graph, "s").inputs];""")
    assert r[0] == "bn" and r[1] == ["b"] and r[2] == [400, 200] and r[3] == [] and r[4] is True
    assert r[5][:2] == ["b", "input"] and len(r[5]) == 3


def test_connect_rejects_cycle():
    r = _run("""return [L.connect(chain(), "c", "a").error, L.connect(chain(), "c", "c").error, L.connect(chain(), "a", "input").error];""")
    assert all(isinstance(e, str) and e for e in r)


def test_delete_bridges_inputs():
    r = _run("""const g0 = G(N("a", "conv", ["input"], { channels: 4, kernel: 3 }), N("b", "conv", ["a"], { channels: 4, kernel: 3 }),
        N("c", "act", ["b"], { kind: "relu" }), N("s", "add", ["b", "a"]), N("t", "add", ["c", "s"]), N("o", "output", ["t"]));
      const g = L.removeNode(g0, "b");
      const all = new Set(ids(g).concat(["input"]));
      const dangling = g.nodes.flatMap((n) => (n.inputs || []).filter((x) => !all.has(x)));
      return [byId(g, "b"), byId(g, "c").inputs, byId(g, "s").inputs, dangling, L.removeNode(g0, "o") === g0 || ids(L.removeNode(g0, "o")).includes("o")];""")
    assert r == [None, ["a"], ["a"], [], True]


def test_remove_edge():
    r = _run("""const g = L.removeEdge(res(), "input", "s"); return byId(g, "s").inputs;""")
    assert r == ["b"]


def test_wrap_residual_adds_shortcut():
    r = _run("""const g = L.wrapResidual(chain(), ["b", "c"], { a: [16, 70], b: [16, 70], c: [32, 35] });
      const gn = byId(g, "g"), add = byId(g, gn.inputs[0]);
      const sc = byId(g, add.inputs[1]);
      const same = L.wrapResidual(chain(), ["b", "c"], { a: [8, 70], b: [8, 70], c: [8, 70] });
      const add2 = byId(same, byId(same, "g").inputs[0]);
      return [add.type, add.inputs[0], sc.type, sc.kernel, sc.stride, sc.channels, sc.inputs, add2.inputs];""")
    assert r == ["add", "c", "conv", 1, 2, 32, ["a"], ["c", "a"]]
    r = _run("""const g = L.wrapResidual(chain(), ["b", "c"], { a: [4, 8, 8], b: [4, 8, 8], c: [8, 4, 4] });
      const add = byId(g, byId(g, "g").inputs[0]); return byId(g, add.inputs[1]).type;""")
    assert r == "conv2d"


def test_wrap_repeat_and_unwrap():
    r = _run("""const g = L.wrapRepeat(chain(), ["a", "b"]);
      const rep = g.nodes.find((n) => n.type === "repeat");
      const back = L.unwrap(g, rep.id);
      const norm = (x) => JSON.stringify(x.nodes.slice().sort((p, q) => (p.id < q.id ? -1 : 1)));
      return [rep.inputs, rep.body.nodes.map((n) => [n.id, n.inputs]), byId(g, "c").inputs[0] === rep.id, norm(back) === norm(chain())];""")
    assert r == [["input"], [["a", ["in"]], ["b", ["a"]]], True, True]


def test_split_branch():
    r = _run("""const g = L.splitBranch(chain(), ["c"], { type: "conv", channels: 4, kernel: 5 });
      const cat = byId(g, byId(g, "g").inputs[0]);
      const lane = byId(g, cat.inputs[1]);
      return [cat.type, cat.inputs[0], lane.inputs, lane.kernel];""")
    assert r == ["concat", "c", ["b"], 5]


def test_copy_paste_requires_single_entry_exit():
    r = _run("""return [L.copyNodes(res(), ["a", "s"]), L.copyNodes(chain(), ["a", "c"]), L.copyNodes(chain(), ["o"]),
      L.copyNodes(chain(), ["a", "b"]) !== null, L.singleEntryExit(res(), ["a", "b"])];""")
    assert r[:4] == [None, None, None, True]
    assert r[4]["source"] == "input" and r[4]["exit"] == "b"


def test_paste_new_ids():
    r = _run("""const g0 = chain();
      const clip = L.copyNodes(g0, ["a", "b"]);
      const { graph, ids: added } = L.pasteAfter(g0, "b", clip);
      const first = byId(graph, added[0]), last = byId(graph, added[added.length - 1]);
      return [added.length, ids(graph).length, first.inputs, byId(graph, "c").inputs, last.id, new Set(ids(graph)).size === ids(graph).length];""")
    assert r[0] == 2 and r[1] == 8 and r[2] == ["b"] and r[3] == [r[4]] and r[5] is True


def test_new_id_and_set_param_and_label():
    r = _run("""const g = chain();
      return [L.newId(g), L.newId(G(N("n7", "bn", ["input"]))), byId(L.setParam(g, "a", "channels", 32), "a").channels, byId(g, "a").channels,
        L.label(byId(g, "a")), L.label({ type: "add" }), L.label({ type: "repeat", times: 3 })];""")
    assert r[0] == "n1" and r[1] == "n8" and r[2] == 32 and r[3] == 8
    assert r[4].startswith("Conv1D") and r[5].startswith("Add") and r[6] == "Repeat × 3"


def test_unwrap_empty_body_keeps_merge_inputs_unique():
    """몸체가 빈 묶음을 풀면 Add 입력이 같은 값 둘이 되지 않음 (Repeat → Residual → 안쪽 지우기 → 풀기)."""
    r = _run("""let g = L.wrapRepeat(chain(), ["c"]);
      const rep = g.nodes.find((n) => n.type === "repeat").id;
      g = L.wrapResidual(g, [rep], { b: [8, 20], [rep]: [8, 20] });
      g = L.removeNode(g, "c");
      g = L.unwrap(g, rep);
      return g.nodes.filter((n) => n.type === "add").map((n) => n.inputs);""")
    assert r and all(len(set(x)) == len(x) for x in r)


def test_settings_values_are_escaped():
    """설정 칸의 값(공유받은 프로젝트에서 올 수 있음)은 이스케이프해서 넣음."""
    import re
    js = GRAPH.read_text(encoding="utf-8")
    assert not re.search(r'value="\$\{(?!esc\()', js.split("function mount(")[1].split("function drawPalette")[0].replace("option value", "")), \
        "설정 칸 value에 esc 없이 값이 들어감"


# ---------------- 옮기기·자유 배치·선 잡기·화살표·활성화 원 (UX)

def test_move_node_onto_edge():
    """놓인 상자를 다른 선 위로 옮기면: 원래 자리는 이어 붙고, 새 선 사이에 들어감 (id·설정 유지)."""
    r = _run("""const { graph, error } = L.moveNode(chain(), "b", "c", "g");
      return [error, byId(graph, "c").inputs, byId(graph, "b").inputs, byId(graph, "g").inputs, byId(graph, "b").kind];""")
    assert r == [None, ["a"], ["c"], ["b"], "relu"]
    r = _run("""return [L.moveNode(chain(), "b", "a", "b").graph === undefined, L.moveNode(chain(), "o", "a", "b").error,
      L.moveNode(chain(), "input", "a", "b").error, L.moveNode(chain(), "b", "b", "c").error];""")
    assert r[1] and r[2] and r[3]


def test_move_node_keeps_pos_off():
    """선 위로 옮긴 상자는 자유 위치를 지워 흐름에 맞춰 놓임."""
    r = _run("""const g = L.setPos(chain(), "b", [400, 20]);
      return [byId(g, "b").pos, byId(L.moveNode(g, "b", "c", "g").graph, "b").pos];""")
    assert r == [[400, 20], None]


def test_set_clear_strip_pos():
    r = _run("""const g = L.setPos(L.setPos(chain(), "a", [10.4, -5]), "c", [30, 40]);
      return [byId(g, "a").pos, L.clearPos(g).nodes.some((n) => n.pos), JSON.stringify(L.stripPos(g)) === JSON.stringify(chain()),
        byId(chain(), "a").pos];""")
    assert r == [[10, -5], False, True, None]


def test_layout_honors_pos_x_and_keeps_rows():
    """옮긴 위치는 희망: 가로는 그대로 따르고(겹치지 않으면), 세로는 층에 맞춰 붙음 (선이 짧게)."""
    r = _run("""const auto = L.layoutGraph(chain()).nodes.find((n) => n.id === "c");
      const l = L.layoutGraph(L.setPos(chain(), "c", [600, 40])), c = l.nodes.find((n) => n.id === "c");
      const e1 = l.edges.find((e) => e.from === "b" && e.to === "c"), last = e1.points[e1.points.length - 1];
      return [c.x - c.ox, c.y === auto.y, Math.abs(last[0] - (c.x + c.ax)) < 0.6, Math.abs(last[1] - c.y) < 0.6, l.width >= c.x + c.w];""")
    assert r == [600, True, True, True, True]


def test_pos_y_only_pushes_down():
    """세로로 끌면 그만큼 아래 층으로 내려가지만, 위 상자보다 위로는 못 감."""
    r = _run("""const y = (g, id) => L.layoutGraph(g).nodes.find((n) => n.id === id).y;
      const nat = y(chain(), "c"), b = y(chain(), "b");
      return [y(L.setPos(chain(), "c", [0, 0]), "c") === nat, y(L.setPos(chain(), "c", [0, 900]), "c") > nat + 200,
        y(L.setPos(chain(), "c", [0, 900]), "g") > y(L.setPos(chain(), "c", [0, 900]), "c"), b < nat];""")
    assert r == [True, True, True, True]


def test_layout_with_hints_never_overlaps_and_edges_go_down():
    """아무렇게나 옮겨 둬도: 같은 층위 상자끼리 겹치지 않고, 묶음은 안쪽을 품고, 선은 모두 아래로 감."""
    r = _run("""const bad = [];
      for (let t = 0; t < 40; t++) {
        const nodes = [];
        for (let i = 0; i < 16; i++) {
          const prev = ["input"].concat(nodes.map((n) => n.id));
          const pick = () => prev[Math.floor(rnd() * prev.length)];
          if (rnd() < 0.25 && prev.length > 2) nodes.push(N("m" + i, rnd() < 0.5 ? "add" : "concat", [pick(), pick()]));
          else if (rnd() < 0.12) nodes.push(N("r" + i, "repeat", [pick()], { times: 2, body: { nodes: [
            N("r" + i + "a", "conv", ["in"], { channels: 4, kernel: 3 }), N("r" + i + "c", "act", ["r" + i + "a"], { kind: "relu" }),
            N("r" + i + "b", "add", ["r" + i + "c", "in"])] } }));
          else nodes.push(N("n" + i, rnd() < 0.3 ? "act" : "conv", [prev[prev.length - 1 - Math.floor(rnd() * 2)] || "input"], { channels: 4, kernel: 3, kind: "relu" }));
        }
        nodes.push(N("o", "output", [nodes[nodes.length - 1].id]));
        let g = G(...nodes);
        const all = []; const walk = (ns) => ns.forEach((n) => { all.push(n.id); if (n.body) walk(n.body.nodes); }); walk(g.nodes);
        for (const id of all) if (rnd() < 0.5) g = L.setPos(g, id, [rnd() * 700, rnd() * 900]);
        const l = L.layoutGraph(g);
        const scope = (n) => (n.anc || []).join("/");
        for (const a of l.nodes) for (const b of l.nodes) {
          if (a.id >= b.id || scope(a) !== scope(b)) continue;
          if (a.x < b.x + b.w - 0.5 && b.x < a.x + a.w - 0.5 && a.y < b.y + b.h - 0.5 && b.y < a.y + a.h - 0.5) bad.push([t, "overlap", a.id, b.id]);
        }
        for (const gr of l.groups) for (const n of l.nodes.filter((n) => (n.anc || []).includes(gr.id)))
          if (n.x < gr.x || n.y < gr.y || n.x + n.w > gr.x + gr.w + 0.5 || n.y + n.h > gr.y + gr.h + 0.5) bad.push([t, "group", gr.id, n.id]);
        for (const e of l.edges) {
          const p = e.points;
          if (p[p.length - 1][1] <= p[0][1]) bad.push([t, "up", e.from, e.to]);
        }
      }
      return bad.slice(0, 5);""")
    assert r == []


def test_layout_repeat_edges_connect_inside():
    """묶음으로 들어가는 선은 묶음 안 입구까지 이어지고, 묶음 출구에서 묶음 아래까지 선이 있음 (끊긴 화살표 없음)."""
    r = _run("""const g = L.wrapRepeat(chain(), ["a", "b"]);
      const rep = g.nodes.find((n) => n.type === "repeat").id;
      const l = L.layoutGraph(g), gr = l.groups.find((x) => x.id === rep), en = l.entries[0];
      const into = l.edges.find((e) => e.to === rep && e.kind !== "exit"), out = l.edges.find((e) => e.kind === "exit" && e.to === rep);
      const end = into.points[into.points.length - 1];
      return [Math.abs(end[1] - (en.y + en.h / 2)) < 1.5, end[1] > gr.y, !!out,
        out && Math.abs(out.points[out.points.length - 1][1] - (gr.y + gr.h)) < 1.5];""")
    assert r == [True, True, True, True]


def test_activation_is_circle():
    r = _run("""const l = L.layoutGraph(chain()), b = l.nodes.find((n) => n.id === "b"), a = l.nodes.find((n) => n.id === "a");
      return [b.w === b.h, b.w < a.w, Math.abs((b.x + b.w / 2) - (a.x + a.w / 2)) < 1, L.shortLabel({ type: "act", kind: "relu" })];""")
    assert r[:3] == [True, True, True] and r[3] == "ReLU"


def test_nearest_edge_is_forgiving():
    """선에서 조금 떨어져(40px) 놓아도 그 선을 잡음. 아주 멀면 없음."""
    r = _run("""const l = L.layoutGraph(chain()), e = l.edges.find((x) => x.from === "a" && x.to === "b");
      const p = e.points[0], q = e.points[e.points.length - 1], mid = [(p[0] + q[0]) / 2 + 40, (p[1] + q[1]) / 2];
      const near = L.nearestEdge(l.edges, mid), far = L.nearestEdge(l.edges, [mid[0] + 400, mid[1]]);
      return [near && near.from + ">" + near.to, far];""")
    assert r == ["a>b", None]


def test_add_free_node_at_position():
    """선과 먼 곳에 놓으면 그 자리에 아직 안 이어진 상자로."""
    r = _run("""const g = L.addFree(chain(), { type: "bn" }, [500, 300]); const n = g.nodes[g.nodes.length - 1];
      return [n.type, n.inputs, n.pos, n.id];""")
    assert r == ["bn", [], [500, 300], "n1"]


def test_free_position_inside_repeat():
    """묶음 안 상자도 가로로 옮김 (묶음 안 좌표). 묶음은 안쪽을 품도록 커짐."""
    r = _run("""let g = L.wrapRepeat(chain(), ["a", "b"]);
      const rep = g.nodes.find((n) => n.type === "repeat").id;
      const before = L.layoutGraph(g).groups.find((x) => x.id === rep);
      g = L.setPos(g, "b", [300, 0]);
      const l = L.layoutGraph(g), gr = l.groups.find((x) => x.id === rep), b = l.nodes.find((n) => n.id === "b");
      return [b.x - b.ox, b.x + b.w <= gr.x + gr.w, b.y + b.h <= gr.y + gr.h, gr.w > before.w, b.anc];""")
    assert r[:4] == [300, True, True, True] and len(r[4]) == 1 and r[4][0].startswith("n")


def test_edge_path_styles():
    """꺾인선은 직선 조각만, 곡선은 꺾이는 곳을 둥글게 (Q)."""
    r = _run("""const pts = [[0, 0], [0, 20], [80, 20], [80, 60]];
      return [L.edgePath(pts, "ortho"), L.edgePath(pts, "curve")];""")
    assert "Q" not in r[0] and r[0].startswith("M0 0") and "Q" in r[1]


def test_align_to_parent_makes_vertical_short_arrow():
    """고른 상자를 '자동 정렬'하면 바로 위 상자와 가운데가 맞고, 기본 간격(짧은 화살표)으로 붙음."""
    r = _run("""let g = L.setPos(L.setPos(L.setPos(chain(), "a", [300, 0]), "b", [500, 600]), "c", [90, 900]);
      g = L.alignToParent(g, ["c", "b", "a"]);
      const l = L.layoutGraph(g), n = (id) => l.nodes.find((x) => x.id === id), cx = (id) => n(id).x + n(id).ax;
      const straight = ["input>a", "a>b", "b>c"].map((k) => { const [f, t] = k.split(">"); const e = l.edges.find((x) => x.from === f && x.to === t);
        return e.points.length === 2 && Math.abs(e.points[0][0] - e.points[1][0]) < 0.6 && e.points[1][1] - e.points[0][1] < 40; });
      return [Math.abs(cx("a") - cx("input")) < 0.6, Math.abs(cx("b") - cx("a")) < 0.6, Math.abs(cx("c") - cx("b")) < 0.6, straight];""")
    assert r == [True, True, True, [True, True, True]]


def test_align_with_repeat_below_stays_centered():
    """정렬한 상자 아래에 묶음이 있어 왼쪽 끝 보정이 생겨도, 옮기지 않은 위 상자와 가운데가 맞음."""
    r = _run("""const rep = (src) => N("R", "repeat", [src], { times: 2, body: { nodes: [
        N("ra", "conv", ["in"], { channels: 8, kernel: 3 }), N("rb", "act", ["ra"], { kind: "relu" }), N("rs", "add", ["rb", "in"])] } });
      let g = G(N("a", "conv", ["input"], { channels: 8, kernel: 3 }), N("b", "bn", ["a"], { pos: [900, 0] }),
        N("c", "act", ["b"], { kind: "poly2", pos: [0, 0] }), rep("c"), N("o", "output", ["R"]));
      g = L.alignToParent(g, ["b", "c"]);
      const l = L.layoutGraph(g), cx = (id) => { const n = l.nodes.find((x) => x.id === id); return n.x + n.ax; };
      return [Math.abs(cx("b") - cx("a")) < 0.6, Math.abs(cx("c") - cx("b")) < 0.6, Math.abs(cx("a") - cx("input")) < 0.6];""")
    assert r == [True, True, True]


def test_skip_lane_keeps_its_side_after_moving_chain():
    """잔차 줄을 옆으로 옮겨도 건너뛰는 선이 줄마다 같은 쪽에 있어 다른 선과 엇갈리지 않음."""
    r = _run("""let g = G(N("a", "conv", ["input"], { channels: 8, kernel: 3 }), N("b", "bn", ["a"]), N("c", "act", ["b"], { kind: "relu" }),
        N("s", "add", ["c", "input"]), N("o", "output", ["s"]));
      g = L.setPos(g, "a", [500, 0]);
      const l = L.layoutGraph(g), e = l.edges.find((x) => x.from === "input" && x.to === "s");
      const n = (id) => l.nodes.find((x) => x.id === id);
      const xs = e.points.slice(0, -2).map((p) => p[0]);                 // 목표로 들어가는 마지막 꺾임 전까지
      return [new Set(xs).size, xs.every((x) => x < n("a").x || x > n("a").x + n("a").w)];""")
    assert r == [1, True]


# ---------------- 가로 방향, 상자 크기

def test_horizontal_chain_is_one_row():
    r = _run("""const l = L.layoutGraph(chain(), undefined, { dir: "h" });
      const ys = l.nodes.map((n) => Math.round(n.y + n.h / 2)), xs = l.nodes.map((n) => n.x);
      const e = l.edges.find((x) => x.from === "a" && x.to === "b");
      return [l.dir, new Set(ys).size, xs.every((x, i) => i === 0 || x > xs[i - 1]), e.points[e.points.length - 1][0] > e.points[0][0],
        Math.abs(e.points[0][0] - (l.nodes.find((n) => n.id === "a").x + l.nodes.find((n) => n.id === "a").w)) < 0.6];""")
    assert r == ["h", 1, True, True, True]


def test_horizontal_with_hints_never_overlaps_and_edges_go_right():
    r = _run("""const bad = [];
      for (let t = 0; t < 25; t++) {
        const nodes = [];
        for (let i = 0; i < 14; i++) {
          const prev = ["input"].concat(nodes.map((n) => n.id));
          const pick = () => prev[Math.floor(rnd() * prev.length)];
          if (rnd() < 0.25 && prev.length > 2) nodes.push(N("m" + i, "add", [pick(), pick()]));
          else if (rnd() < 0.15) nodes.push(N("r" + i, "repeat", [pick()], { times: 2, body: { nodes: [
            N("r" + i + "a", "conv", ["in"], { channels: 4, kernel: 3 }), N("r" + i + "b", "add", ["r" + i + "a", "in"])] } }));
          else nodes.push(N("n" + i, rnd() < 0.3 ? "act" : "conv", [prev[prev.length - 1]], { kind: "relu", channels: 4, kernel: 3 }));
        }
        nodes.push(N("o", "output", [nodes[nodes.length - 1].id]));
        let g = G(...nodes);
        for (const n of g.nodes) if (rnd() < 0.4) g = L.setPos(g, n.id, [rnd() * 900, rnd() * 500]);
        const l = L.layoutGraph(g, undefined, { dir: "h" });
        const scope = (n) => (n.anc || []).join("/");
        for (const a of l.nodes) for (const b of l.nodes) {
          if (a.id >= b.id || scope(a) !== scope(b)) continue;
          if (a.x < b.x + b.w - 0.5 && b.x < a.x + a.w - 0.5 && a.y < b.y + b.h - 0.5 && b.y < a.y + a.h - 0.5) bad.push([t, a.id, b.id]);
        }
        for (const gr of l.groups) for (const n of l.nodes.filter((n) => (n.anc || []).includes(gr.id)))
          if (n.y < gr.y + 24 || n.x < gr.x || n.x + n.w > gr.x + gr.w + 0.5 || n.y + n.h > gr.y + gr.h + 0.5) bad.push([t, "group", gr.id, n.id]);
        for (const e of l.edges) if (e.points[e.points.length - 1][0] <= e.points[0][0]) bad.push([t, "left", e.from, e.to]);
      }
      return bad.slice(0, 5);""")
    assert r == []


def test_horizontal_pos_and_align():
    """가로일 때도 옮긴 위치는 화면 좌표 그대로(세로 = 같은 열 안 자리), 정렬은 왼쪽 상자와 가운데(가로선)."""
    r = _run("""let g = L.setPos(chain(), "b", [0, 400]);
      let l = L.layoutGraph(g, undefined, { dir: "h" }), b = l.nodes.find((n) => n.id === "b");
      const yHint = b.y - b.oy;
      g = L.alignToParent(g, ["b", "c"], { dir: "h" });
      l = L.layoutGraph(g, undefined, { dir: "h" });
      const cy = (id) => { const n = l.nodes.find((x) => x.id === id); return n.y + n.ax; };
      return [yHint, Math.abs(cy("b") - cy("a")) < 0.6, Math.abs(cy("c") - cy("b")) < 0.6];""")
    assert r == [400, True, True]


def test_box_size_for_selected():
    """고른 상자들의 크기를 한 번에 (보기용, 모델과 상관없음). 활성화 원은 짧은 쪽 지름. null이면 기본 크기."""
    r = _run("""let g = L.setSize(chain(), ["a", "b", "c"], [300, 80]);
      const l = L.layoutGraph(g), n = (id) => l.nodes.find((x) => x.id === id);
      const back = L.layoutGraph(L.setSize(g, ["a"], null)).nodes.find((x) => x.id === "a");
      const small = L.setSize(chain(), ["a"], [5, 5000]);
      return [n("a").w, n("a").h, n("c").w, n("b").w, n("b").h, back.w, back.h, byId(small, "a").size,
        JSON.stringify(L.stripView(g)) === JSON.stringify(chain())];""")
    assert r == [300, 80, 300, 80, 80, 236, 50, [60, 400], True]


def test_horizontal_boxes_are_narrow():
    """가로 방향의 기본 상자는 좁게 (모델이 옆으로 덜 길어짐). 사용자가 정한 크기는 그대로."""
    r = _run("""const v = L.layoutGraph(chain()), h = L.layoutGraph(chain(), undefined, { dir: "h" });
      const w = (l, id) => l.nodes.find((n) => n.id === id).w;
      const custom = L.layoutGraph(L.setSize(chain(), ["a"], [300, 80]), undefined, { dir: "h" });
      const wide = L.layoutGraph(chain(), (n) => [236, 50], { dir: "h" });
      return [w(h, "a") <= 140, w(h, "input") <= 140, h.width < wide.width * 0.7, w(custom, "a")];""")
    assert r == [True, True, True, 300]


def test_label_lines_wrap_in_narrow_boxes():
    """좁은 상자는 이름과 설정을 두 줄로, 그래도 길면 '…'로 줄임. 넓은 상자는 한 줄."""
    r = _run("""const n = { type: "conv", channels: 32, kernel: 7, stride: 1 };
      return [L.labelLines(n, null, 236), L.labelLines(n, null, 132),
        L.labelLines({ type: "output" }, { task: "classification", classes: ["정상", "이상", "주의", "위험", "고장"] }, 132)];""")
    assert r[0] == ["Conv1D · 32 filters · k=7"]
    assert r[1] == ["Conv1D", "32 filters · k=7"]
    assert r[2][0] == "Output" and len(r[2]) == 2 and r[2][1].endswith("…") and len(r[2][1]) <= 18


# ---------------- 줄바꿈 (가로면 아래 줄로, 세로면 오른쪽 칸으로)

const_res3 = """
const res3 = () => G(N("c1", "conv", ["input"], { channels: 8, kernel: 3 }), N("p1", "act", ["c1"], { kind: "relu" }), N("s1", "add", ["p1", "input"]),
  N("c2", "conv", ["s1"], { channels: 8, kernel: 3 }), N("p2", "act", ["c2"], { kind: "relu" }), N("s2", "add", ["p2", "s1"]),
  N("c3", "conv", ["s2"], { channels: 8, kernel: 3 }), N("p3", "act", ["c3"], { kind: "relu" }), N("s3", "add", ["p3", "s2"]),
  N("g", "global_avg", ["s3"]), N("o", "output", ["g"]));
const overlaps = (l) => { const bad = []; for (const a of l.nodes) for (const b of l.nodes) {
  if (a.id >= b.id || (a.anc || []).join() !== (b.anc || []).join()) continue;
  if (a.x < b.x + b.w - 0.5 && b.x < a.x + a.w - 0.5 && a.y < b.y + b.h - 0.5 && b.y < a.y + a.h - 0.5) bad.push([a.id, b.id]); } return bad; };
"""


def test_manual_break_horizontal_goes_to_next_row():
    """가로 방향에서 '여기서 줄바꿈'한 상자부터는 아래 줄로, 줄 첫머리(왼쪽)부터. 줄 사이는 되돌아가는 선."""
    r = _run(const_res3 + """let g = L.setBreak(L.setBreak(res3(), "c2", true), "c3", true);
      const l = L.layoutGraph(g, undefined, { dir: "h" }), n = (id) => l.nodes.find((x) => x.id === id);
      const e = l.edges.find((x) => x.from === "s1" && x.to === "c2");
      const end = e.points[e.points.length - 1], dips = e.points.some((p) => p[1] > n("s1").y + n("s1").h && p[1] < n("c2").y);
      return [n("c2").y > n("s1").y + n("s1").h, n("c3").y > n("c2").y + n("c2").h, Math.abs(n("c2").x - n("input").x) < 1,
        Math.abs(end[0] - n("c2").x) < 0.6, dips, overlaps(l).length, byId(L.setBreak(g, "c2", false), "c2").brk];""")
    assert r == [True, True, True, True, True, 0, None]


def test_wrapped_row_can_move_along_flow():
    """줄바꿈한 줄은 손잡이로 흐름 방향(가로면 좌우)으로 옮길 수 있음: 그 줄만 움직이고, 0이면 맨 앞, 겹치지 않음."""
    r = _run(const_res3 + """const g = L.setBreak(res3(), "c2", true), opt = { dir: "h" };
      const l0 = L.layoutGraph(g, undefined, opt), x0 = (id) => l0.nodes.find((x) => x.id === id).x;
      const g1 = L.setShift(g, "c2", 120), l1 = L.layoutGraph(g1, undefined, opt), x1 = (id) => l1.nodes.find((x) => x.id === id).x;
      const back = L.setShift(g1, "c2", -50);
      return [l0.rows.length, l0.rows[1].anchor, Math.round(x1("c2") - x0("c2")), Math.round(x1("s3") - x0("s3")),
        Math.round(x1("c1") - x0("c1")), l1.rows[1].shift, overlaps(l1).length, "shift" in L.byIdOf(back, "c2"),
        "shift" in L.stripView(g1).nodes.find((n) => n.id === "c2"), "shift" in L.clearPos(g1).nodes.find((n) => n.id === "c2"),
        Math.round(l1.rows[1].x - x1("c2"))];""".replace("L.byIdOf", "((g, id) => byId(g, id))"))
    assert r == [2, "c2", 120, 120, 0, 120, 0, False, False, False, 0], r


def test_manual_break_vertical_goes_to_next_column():
    r = _run(const_res3 + """const l = L.layoutGraph(L.setBreak(res3(), "c2", true)), n = (id) => l.nodes.find((x) => x.id === id);
      return [n("c2").x > n("c1").x + n("c1").w, Math.abs(n("c2").y - n("input").y) < 1, overlaps(l).length];""")
    assert r == [True, True, 0]


def test_auto_wrap_keeps_rows_within_width():
    """자동 줄바꿈: 정한 폭을 넘으면 다음 줄로. 모든 줄이 그 폭 안 (상자 하나가 더 넓은 경우 말고)."""
    r = _run(const_res3 + """const l = L.layoutGraph(res3(), undefined, { dir: "h", wrap: 700 });
      const rows = new Set(l.nodes.map((n) => Math.round(n.y / 40)));
      return [Math.max(...l.nodes.map((n) => n.x + n.w)) <= 700 + 24 + 1, rows.size > 1, overlaps(l).length,
        l.edges.every((e) => e.points.length >= 2)];""")
    assert r == [True, True, 0, True]


def test_breaks_with_random_graphs_never_overlap():
    r = _run(const_res3 + """const bad = [];
      for (let t = 0; t < 25; t++) {
        const nodes = [];
        for (let i = 0; i < 14; i++) {
          const prev = ["input"].concat(nodes.map((n) => n.id));
          const pick = () => prev[Math.floor(rnd() * prev.length)];
          if (rnd() < 0.25 && prev.length > 2) nodes.push(N("m" + i, "add", [pick(), pick()]));
          else if (rnd() < 0.12) nodes.push(N("r" + i, "repeat", [pick()], { times: 2, body: { nodes: [
            N("r" + i + "a", "conv", ["in"], { channels: 4, kernel: 3 }), N("r" + i + "b", "add", ["r" + i + "a", "in"])] } }));
          else nodes.push(N("n" + i, "conv", [prev[prev.length - 1]], { channels: 4, kernel: 3 }));
        }
        nodes.push(N("o", "output", [nodes[nodes.length - 1].id]));
        let g = G(...nodes);
        for (const n of g.nodes) if (rnd() < 0.25) g = L.setBreak(g, n.id, true);
        for (const dir of ["h", "v"]) {
          const l = L.layoutGraph(g, undefined, { dir, wrap: rnd() < 0.5 ? 800 : undefined });
          const o = overlaps(l);
          if (o.length) bad.push([t, dir, o[0]]);
          for (const gr of l.groups) for (const n of l.nodes.filter((n) => (n.anc || []).includes(gr.id)))
            if (n.x < gr.x - 0.5 || n.y < gr.y - 0.5 || n.x + n.w > gr.x + gr.w + 0.5 || n.y + n.h > gr.y + gr.h + 0.5) bad.push([t, dir, "group", gr.id]);
        }
      }
      return bad.slice(0, 5);""")
    assert r == []


def test_activation_default_is_relu_without_he_words():
    r = _run("""return [L.TYPES.act.make().kind, L.TYPES.se.make().act, L.label({ type: "act", kind: "gelu" }),
      Object.values(L.TYPES).map((T) => T.desc).join(" ")];""")
    assert r[:3] == ["relu", "relu", "GELU"]
    assert "암호문" not in r[3] and "Poly2" not in r[3] and "근사" not in r[3]

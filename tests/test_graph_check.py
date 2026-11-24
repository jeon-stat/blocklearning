"""그래프 검사(check_graph): 형식·연결·모양·합치기·RNN·Tanh·Repeat 규칙."""
import pytest

from dlstudio import blocks as B
from dlstudio.graph import check_graph


def G(*nodes):
    return {"version": 2, "nodes": [{"id": "input", "type": "input"}, *nodes]}


def N(id, type, *inputs, **kw):
    return dict(id=id, type=type, inputs=list(inputs), **kw)


def conv(id, src, ch=16, k=3, **kw):
    return N(id, "conv", src, channels=ch, kernel=k, stride=kw.pop("stride", 1), **kw)


def msgs(c, node):
    return [i.message for i in c.issues if i.node == node]


def test_chain_shapes():
    g = G(conv("c", "input", 16, 5), N("a", "act", "c", kind="relu"), N("g", "global_avg", "a"), N("o", "output", "g"))
    c = check_graph(g, 70, 2)
    assert c.ok, c.issues
    assert c.shapes == {"input": (1, 70), "c": (16, 70), "a": (16, 70), "g": (16,), "o": (2,)}
    assert c.kind == "cnn"
    assert c.order == ["input", "c", "a", "g", "o"]


def downsample_res():
    return G(conv("c0", "input", 8), N("r0", "act", "c0", kind="relu"),
             conv("a", "r0", 16, stride=2), N("ra", "act", "a", kind="relu"), conv("b", "ra", 16),
             conv("sc", "r0", 16, 1, stride=2), N("s", "add", "b", "sc"),
             N("g", "global_avg", "s"), N("o", "output", "g"))


def dense_res():
    return G(N("d0", "dense", "input", units=32), N("a0", "act", "d0", kind="relu"),
             N("d1", "dense", "a0", units=32), N("a1", "act", "d1", kind="relu"), N("d2", "dense", "a1", units=32),
             N("s", "add", "d2", "a0"), N("o", "output", "s"))


def far_concat():
    return G(conv("a", "input", 8), conv("b", "a", 8), conv("c", "b", 8), N("cat", "concat", "a", "c"),
             N("g", "global_avg", "cat"), N("o", "output", "g"))


def test_three_target_structures_pass():
    for g in (downsample_res(), dense_res(), far_concat()):
        c = check_graph(g, 40, 3)
        assert c.ok, c.issues
    assert check_graph(downsample_res(), 40, 3).shapes["s"] == (16, 20)
    assert check_graph(far_concat(), 40, 3).shapes["cat"] == (16, 40)
    assert check_graph(dense_res(), 40, 3).shapes["s"] == (32,)


def test_add_shape_mismatch():
    g = G(conv("a", "input", 16), conv("b", "a", 16, stride=2), N("s", "add", "a", "b"), N("g", "global_avg", "s"),
          N("o", "output", "g"))
    c = check_graph(g, 70, 2)
    assert any("모양이 다름" in m for m in msgs(c, "s"))
    assert any("1×1" in i.hint for i in c.issues if i.node == "s")


def test_concat_rules():
    g = G(conv("a", "input", 8), conv("b", "a", 8, stride=2), N("cat", "concat", "a", "b"), N("g", "global_avg", "cat"),
          N("o", "output", "g"))
    assert msgs(check_graph(g, 70, 2), "cat")
    g = G(N("d", "dense", "input", units=16), N("e", "dense", "d", units=16), N("cat", "concat", "d", "e"), N("o", "output", "cat"))
    c = check_graph(g, 20, 2)                                     # 숫자 묶음끼리는 이어 붙임
    assert c.ok and c.shapes["cat"] == (32,)
    g = G(N("a", "conv2d", "input", channels=8, kernel=3), N("b", "conv2d", "a", channels=4, kernel=3),
          N("cat", "concat", "a", "b"), N("g", "global_avg2d", "cat"), N("o", "output", "g"))
    c = check_graph(g, 64, 10, image=(8, 8))
    assert c.ok, c.issues
    assert c.shapes["cat"] == (12, 8, 8)


def test_connection_rules():
    base = [N("g", "global_avg", "c"), N("o", "output", "g")]
    c = check_graph(G(N("c", "conv", channels=8, kernel=3), *base), 20, 2)
    assert any("입력이 연결되지 않음" in m for m in msgs(c, "c"))
    c = check_graph(G(conv("x", "input", 8), N("c", "conv", "input", "x", channels=8, kernel=3), *base), 20, 2)
    assert any("입력은 하나만" in m for m in msgs(c, "c"))
    c = check_graph(G(conv("c", "input", 8), N("s", "add", "c"), N("g", "global_avg", "s"), N("o", "output", "g")), 20, 2)
    assert any("2개 이상" in m for m in msgs(c, "s"))
    c = check_graph(G(conv("c", "input", 8), conv("lost", "c", 8), *base), 20, 2)
    assert any("출력까지 이어지지 않음" in m for m in msgs(c, "lost"))
    c = check_graph(G(conv("a", "b", 8), conv("b", "a", 8), conv("c", "input", 8), *base), 20, 2)
    assert any("제자리로" in m for m in msgs(c, "a")) and any("제자리로" in m for m in msgs(c, "b"))
    c = check_graph(G(conv("c", "input", 8), N("g", "global_avg", "c"), N("o", "output", "g"), N("o2", "output", "g")), 20, 2)
    assert not c.ok
    c = check_graph(G(conv("c", "input", 8), N("g", "global_avg", "c"), N("o", "output", "g"), N("d", "dense", "o", units=2)), 20, 2)
    assert not c.ok


def test_rnn_rules():
    c = check_graph(G(N("r", "lstm", "input", hidden=16, layers=1), N("o", "output", "r")), 30, 2)
    assert c.ok and c.kind == "rnn" and c.shapes["r"] == (16,)
    c = check_graph(G(conv("c", "input", 4), N("r", "lstm", "c", hidden=16, layers=1), N("o", "output", "r")), 30, 2)
    assert c.ok and c.shapes["r"] == (16,)                        # 순환 층은 (채널, 길이) 뒤 어디에나
    c = check_graph(G(N("r", "lstm", "input", hidden=8, layers=1), conv("c", "input", 8), N("g", "global_avg", "c"),
                      N("s", "add", "r", "g"), N("o", "output", "s")), 30, 2)
    assert c.ok and c.kind == "rnn"                                # 입력을 순환 층과 나눠 써도 됨
    c = check_graph(G(N("d", "dense", "input", units=4), N("r", "lstm", "d", hidden=8, layers=1), N("o", "output", "r")), 30, 2)
    assert msgs(c, "r")                                            # 숫자 묶음 뒤에는 안 됨
    c = check_graph(G(conv("c", "input", 4), N("f", "rand_flip", "c"), N("g", "global_avg", "f"), N("o", "output", "g")), 30, 2)
    assert msgs(c, "f")                                            # 증강은 Input 바로 뒤에만


def test_tanh_anywhere():
    g = G(N("d", "dense", "input", units=8), N("r", "act", "d", kind="relu"), N("t", "act", "r", kind="tanh"),
          N("o", "output", "t"))
    assert check_graph(g, 10, 2).ok


def rep(times, *body, id="R", src="input"):
    return dict(id=id, type="repeat", times=times, inputs=[src], body={"nodes": list(body)})


def test_repeat():
    body = [conv("rc", "in", 16), N("ra", "act", "rc", kind="relu"), N("rs", "add", "ra", "in")]
    g = G(conv("c", "input", 16), rep(2, *body, src="c"), N("g", "global_avg", "R"), N("o", "output", "g"))
    c = check_graph(g, 70, 2)
    assert c.ok, c.issues
    assert c.shapes["R"] == (16, 70) and c.shapes["rs"] == (16, 70)
    assert c.order == ["input", "c", ["R", ["rc", "ra", "rs"]], "g", "o"]
    two_exits = [conv("rc", "in", 16), conv("rd", "in", 16)]
    c = check_graph(G(conv("c", "input", 16), rep(2, *two_exits, src="c"), N("g", "global_avg", "R"), N("o", "output", "g")), 70, 2)
    assert any("출구가 하나" in m for m in msgs(c, "R"))
    c = check_graph(G(rep(1, N("rl", "lstm", "in", hidden=4, layers=1)), N("o", "output", "R")), 30, 2)
    assert not c.ok
    g = G(conv("c", "input", 8), rep(2, conv("rc", "in", 8, stride=2), src="c"), N("g", "global_avg", "R"), N("o", "output", "g"))
    c = check_graph(g, 70, 2)
    assert c.ok and c.shapes["R"] == (8, 18)


@pytest.mark.parametrize("g", [
    None, {}, {"nodes": "x"}, {"nodes": ["conv"]}, G(N("c", "nope", "input")),
    G(conv("c", "input"), conv("c", "c"), N("o", "output", "c")), G(conv("a.b", "input"), N("o", "output", "a.b")),
    G(dict(id="c", type="conv", channels=8, kernel=3, inputs="input"), N("o", "output", "c")),
    G(conv("c", "ghost"), N("g", "global_avg", "c"), N("o", "output", "g")),
    {"nodes": [conv("c", "input"), N("o", "output", "c")]},
    G(rep(2, src="input"), N("o", "output", "R")), G(dict(id="R", type="repeat", times=2, inputs=["input"], body="x"), N("o", "output", "R")),
])
def test_malformed_graphs_give_issues(g):
    c = check_graph(g, 20, 2)
    assert not c.ok and c.issues
    assert c.to_dict()["ok"] is False


@pytest.mark.parametrize("blocks,graph,L,image", [
    ([{"type": "conv", "channels": 6, "kernel": 3, "groups": 4}, {"type": "global_avg"}, {"type": "output"}],
     G(N("x", "conv", "input", channels=6, kernel=3, groups=4), N("g", "global_avg", "x"), N("o", "output", "g")), 20, None),
    ([{"type": "pool", "size": 4}, {"type": "pool", "size": 4}, {"type": "global_avg"}, {"type": "output"}],
     G(N("p", "pool", "input", size=4), N("x", "pool", "p", size=4), N("g", "global_avg", "x"), N("o", "output", "g")), 10, None),
    ([{"type": "dense", "units": 3000}, {"type": "output"}],
     G(N("x", "dense", "input", units=3000), N("o", "output", "x")), 10, None),
    ([{"type": "conv", "channels": 4, "kernel": 3}, {"type": "output"}],
     G(N("x", "conv", "input", channels=4, kernel=3), N("o", "output", "x")), 64, (8, 8)),
])
def test_layer_rules_carried_over(blocks, graph, L, image):
    old = B.check(blocks, L, 2, image=image)
    new = check_graph(graph, L, 2, image=image)
    assert [i.message for i in old.issues] and [i.message for i in old.issues][0] in [i.message for i in new.issues if i.node == "x"]


def test_add_same_input_twice_is_fine():
    g = G(conv("a", "input", 8), N("s", "add", "a", "a"), N("g", "global_avg", "s"), N("o", "output", "g"))
    assert check_graph(g, 20, 2).ok
    g = G(conv("a", "input", 8), N("s", "concat", "a", "a"), N("g", "global_avg", "s"), N("o", "output", "g"))
    assert check_graph(g, 20, 2).ok


def test_pos_is_ignored_by_check_and_build():
    """화면 위치(pos)는 모델에 영향 없음."""
    from dlstudio.graph import build_graph_model
    g = G(dict(conv("c", "input", 8), pos=[300, 20]), N("g", "global_avg", "c", pos=[1, 2]), N("o", "output", "g"))
    c = check_graph(g, 20, 2)
    assert c.ok, c.issues
    build_graph_model(g, 20, 2)


def test_view_only_fields_are_ignored():
    """화면용 값(pos, size)은 검사·모델에 영향 없음."""
    from dlstudio.graph import build_graph_model
    g = G(dict(conv("c", "input", 8), size=[300, 80]), N("g", "global_avg", "c", size=[100, 40]), N("o", "output", "g"))
    assert check_graph(g, 20, 2).ok
    build_graph_model(g, 20, 2)

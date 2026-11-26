"""라벨링 작업실 화면(static/label.js)의 순수 함수: 범위 고르기, 단축키 해석, 진행 문구. node가 있으면 실행."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import blocklearning

STATIC = Path(blocklearning.__file__).parent / "static"
LABEL = STATIC / "label.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")


def _run(js: str):
    code = f"const L = require({json.dumps(str(LABEL))});\nconst out = (() => {{ {js} }})();\nprocess.stdout.write(JSON.stringify(out));"
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_range_select():
    names = ["a", "b", "c", "d", "e"]
    assert _run(f"return L.rangeSelect({json.dumps(names)}, 'b', 'd');") == ["b", "c", "d"]
    assert _run(f"return L.rangeSelect({json.dumps(names)}, 'd', 'b');") == ["b", "c", "d"]
    assert _run(f"return L.rangeSelect({json.dumps(names)}, 'x', 'c');") == ["c"]          # 기준이 없으면 그것만


def test_key_action():
    k = lambda **e: _run(f"return L.keyAction({json.dumps(e)});")
    assert k(key="3", target={"tagName": "DIV"}) == {"type": "label", "index": 2}
    assert k(key="0", target={"tagName": "DIV"}) == {"type": "clear"}
    assert k(key="Delete", target={"tagName": "BODY"}) == {"type": "clear"}
    assert k(key="Enter", target={"tagName": "DIV"}) == {"type": "confirm"}
    assert k(key="ArrowRight", target={"tagName": "DIV"}) == {"type": "next"} and k(key="j", target={"tagName": "DIV"}) == {"type": "next"}
    assert k(key="ArrowLeft", target={"tagName": "DIV"}) == {"type": "prev"} and k(key="k", target={"tagName": "DIV"}) == {"type": "prev"}
    assert k(key="z", ctrlKey=True, target={"tagName": "DIV"}) == {"type": "undo"}
    assert k(key="y", ctrlKey=True, target={"tagName": "DIV"}) == {"type": "redo"}
    assert k(key="Z", ctrlKey=True, shiftKey=True, target={"tagName": "DIV"}) == {"type": "redo"}
    assert k(key="Escape", target={"tagName": "DIV"}) == {"type": "esc"}
    for tag in ("INPUT", "TEXTAREA", "SELECT"):                           # 입력칸에 쓰는 중에는 단축키 없음
        assert k(key="3", target={"tagName": tag}) is None


def test_progress_text():
    c = {"total": 12, "labeled": 8, "manual": 3, "rule": 5, "import": 0, "unconfirmed": 5, "unlabeled": 4, "error": 0}
    assert _run(f"return L.progressText({json.dumps(c)});") == "전체 12 · 정답 8 (사람 3 · 규칙 5 · 가져옴 0) · 확인 안 함 5 · 정답 없음 4"
    c["error"] = 2
    assert _run(f"return L.progressText({json.dumps(c)});").endswith(" · 읽기 오류 2")


def test_label_page_files_and_link():
    html = (STATIC / "label.html").read_text(encoding="utf-8")
    assert "/static/label.js" in html and "/static/style.css" in html and "http://" not in html and "https://" not in html
    assert 'id="btn-label"' in (STATIC / "index.html").read_text(encoding="utf-8")


def test_label_js_compiles_with_other_scripts():
    """label.js도 IIFE: graph.js·app.js와 이어 붙여 컴파일해도 이름이 겹치지 않음."""
    srcs = [STATIC / "graph.js", STATIC / "app.js", LABEL]
    code = ("const vm = require('vm'); const fs = require('fs');"
            f"const src = {json.dumps([str(s) for s in srcs])}.map((p) => fs.readFileSync(p, 'utf8')).join(String.fromCharCode(10));"
            "try { new vm.Script(src); } catch (e) { console.error(e.message); process.exit(3); }")
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr


def test_next_after_relabel_and_backspace():
    """정답을 붙여 카드가 목록에서 빠져도(걸러 보기) 같은 자리의 다음 샘플로. Backspace는 지우기가 아님."""
    assert _run('return L.nextAfter(["a","b","c","d"], ["a","c","d"], "b");') == "c"
    assert _run('return L.nextAfter(["a","b","c","d"], ["a","b","c","d"], "b");') == "c"
    assert _run('return L.nextAfter(["a","b"], ["a"], "b");') == "a"
    assert _run('return L.nextAfter(["a"], [], "a");') is None
    assert _run('return L.keyAction({key: "Backspace", target: {tagName: "DIV"}});') is None


def test_progress_text_with_model():
    c = {"total": 20, "labeled": 20, "manual": 6, "rule": 0, "import": 0, "model": 14, "unconfirmed": 14, "unlabeled": 0, "error": 0}
    assert _run(f"return L.progressText({json.dumps(c)});") == \
        "전체 20 · 정답 20 (사람 6 · 규칙 0 · 가져옴 0 · 모델 14) · 확인 안 함 14 · 정답 없음 0"


def test_px_to_rows():
    """크게 보기에서 끈 가로 위치(차트 안 비율 0–1) → 원래 행 번호 [시작, 끝). 확대(보이는 범위 a–b)와 줄인 값(점 수 ≠ 행 수)을 반영."""
    assert _run("return L.pxToRows(0, 1, 0, 99, 100, 100);") == [0, 100]
    assert _run("return L.pxToRows(0.5, 0.25, 0, 99, 100, 100);") == [25, 51]          # 거꾸로 끌어도
    assert _run("return L.pxToRows(0, 0.5, 50, 99, 100, 100);") == [50, 76]            # 확대된 화면
    assert _run("return L.pxToRows(0, 1, 0, 1999, 2000, 10000);") == [0, 10000]        # 줄인 값 → 원래 행
    assert _run("return L.pxToRows(0.3, 0.3, 0, 99, 100, 100);") == [30, 31]           # 클릭만 해도 한 칸


def test_norm_box_and_copy_key():
    assert _run("return L.normBox(0.6, 0.7, 0.2, 0.1);") == [0.2, 0.1, 0.4, 0.6]        # 거꾸로 끌어도
    assert _run("return L.normBox(-0.2, 0.5, 0.3, 1.4);") == [0, 0.5, 0.3, 0.5]         # 그림 밖은 잘라냄
    assert _run("return L.normBox(0.5, 0.5, 0.501, 0.502);") is None                    # 너무 작으면 상자 아님
    assert _run('return L.keyAction({key: "c", target: {tagName: "DIV"}});') == {"type": "copy_prev"}


def test_click_without_drag_does_not_paint():
    assert _run("return L.isDrag(100, 100, 102, 101);") is False
    assert _run("return L.isDrag(100, 100, 108, 100);") is True


def test_shape_helpers():
    """도형: 통째로 옮겨도 그림 안, 다각형 닫기는 겹친 점을 하나로·3개 이상만."""
    r = _run("""return [L.shapeKind([0.1, 0.1, 0.2, 0.2, "a"]), L.shapeKind(["poly", [[0, 0], [1, 0], [0, 1]], "a"]), L.shapeKind(["pt", 0.5, 0.5, "a"]),
      L.moveShape(["poly", [[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]], "a"], -0.5, 0.7),
      L.moveShape(["pt", 0.9, 0.2, "a"], 0.3, -0.1), L.moveShape([0.7, 0.7, 0.2, 0.2, "a"], 0.5, 0),
      L.closePoly([[0.1, 0.1], [0.5, 0.1], [0.5, 0.1], [0.3, 0.6], [0.1, 0.1]]), L.closePoly([[0.1, 0.1], [0.5, 0.1]])];""")
    assert r[:3] == ["box", "poly", "pt"]
    xs = [p[0] for p in r[3][1]]; ys = [p[1] for p in r[3][1]]
    assert min(xs) == 0 and abs(max(ys) - 1) < 1e-9                 # 왼쪽·아래 끝에 붙음, 모양은 그대로
    assert r[4][1] == 1 and abs(r[4][2] - 0.1) < 1e-9                  # 점도 그림 안으로
    assert abs(r[5][0] - 0.8) < 1e-9
    assert r[6] == [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]] and r[7] is None


def test_circle_helpers():
    """원: 반지름은 너비 비율 (세로로 끈 거리는 너비/높이로 나눔), 옮겨도 가운데는 그림 안."""
    r = _run("""return [L.shapeKind(["circle", 0.5, 0.5, 0.1, "a"]), L.circleR(0.5, 0.5, 0.6, 0.5, 2), L.circleR(0.5, 0.5, 0.5, 0.6, 2),
      L.circleR(0, 0, 1, 1, 1), L.moveShape(["circle", 0.9, 0.2, 0.1, "a"], 0.3, -0.5)];""")
    assert r[0] == "circle" and abs(r[1] - 0.1) < 1e-9 and abs(r[2] - 0.05) < 1e-9 and r[3] == 1
    assert r[4] == ["circle", 1, 0, 0.1, "a"]

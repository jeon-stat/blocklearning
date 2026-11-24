"""화면 파일: 필요한 파일이 있고, index.html이 레포 안의 Blockly와 화면 스크립트를 부르며, 외부 주소를 쓰지 않는다."""
import re
from pathlib import Path

import dlstudio

STATIC = Path(dlstudio.__file__).parent / "static"
FILES = ["index.html", "app.js", "graph.js", "style.css"]


def test_files_exist():
    for f in FILES:
        assert (STATIC / f).is_file(), f


def test_index_loads_local_scripts_in_order():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    order = [html.find(s) for s in ("/static/graph.js", "/static/app.js")]
    assert "flow.js" not in html and not (STATIC / "flow.js").exists()
    assert all(i >= 0 for i in order) and order == sorted(order)
    assert "/static/style.css" in html


def test_no_external_urls():
    for f in ("index.html", "app.js", "graph.js", "style.css"):
        text = (STATIC / f).read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), f


def test_all_layer_types_in_palette():
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    for t in ("conv", "pool", "global_avg", "bn", "act", "dense", "lstm", "gru", "add", "concat", "output"):
        assert f'make: () => ({{ type: "{t}"' in js, t


def test_model_lead_explains_graph_editing():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    lead = html[html.find('id="step-model"'):]
    lead = lead[lead.find('<p class="lead">'):lead.find("</p>")]
    assert "선 위" in lead and "동그라미" in lead and "파란 선" not in lead


def test_graph_editor_has_ports_and_edge_picking():
    """선 잇기(상자 아래 출력 손잡이)와 선 고르기(data-edge)가 화면에 있음."""
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    assert 'class="port"' in js and "data-edge=" in js and "function mount(" in js
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "StudioGraph" in app and "StudioFlow" not in app and "blocks" not in app


def test_no_third_party_editor():
    """흐름도 편집기는 직접 그림 (Blockly 등 외부 라이브러리 없음)."""
    assert not (STATIC / "vendor").exists()
    assert "Blockly" not in (STATIC / "app.js").read_text(encoding="utf-8")


def test_four_steps_start_with_task():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="step-remote"' not in html
    pos = [html.find(f'id="step-{s}"') for s in ("task", "data", "model", "train")]
    assert all(i >= 0 for i in pos) and pos == sorted(pos)


def test_image_inputs_exist():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for i in ("folder-path", "folder-size", "btn-folder", "image-width"):
        assert f'id="{i}"' in html, i


def test_hidden_attribute_always_hides():
    """display: flex 같은 스타일이 hidden 속성을 덮어쓰면 숨긴 칸(예: 이미지 폴더 칸)이 다른 과제에서도 보임."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important", css)


def test_series_folder_inputs_exist():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for i in ("series-row", "series-path", "btn-series-scan", "series-box", "series-column", "series-length",
              "series-resample", "series-label-file", "btn-series-load", "series-guide"):
        assert f'id="{i}"' in html, i


def test_series_label_file_is_reachable_before_scan():
    """정답 표 칸이 살펴보기 결과 칸(처음엔 숨김) 안에 있으면, 살펴보기가 실패할 때 넣을 방법이 없음."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    row = html[html.find('id="series-row"'):]
    assert row.find('id="series-label-file"') < row.find('id="series-box"')


def test_label_page_can_go_back_to_start():
    """열린 작업이 있어도 시작 화면(데이터 종류·정답 종류 고르기)으로 돌아갈 수 있어야 한다."""
    html = (STATIC / "label.html").read_text(encoding="utf-8")
    js = (STATIC / "label.js").read_text(encoding="utf-8")
    header = html[html.find('<header'):html.find('</header>')]
    assert 'id="lab-switch"' in header                       # 위쪽 줄에 늘 보이는 버튼
    assert 'id="lab-back"' in html                           # 시작 화면에서 지금 작업으로 돌아가기
    assert '$("lab-switch").onclick' in js and '$("lab-back").onclick' in js


def test_label_start_text_names_every_data_kind():
    html = (STATIC / "label.html").read_text(encoding="utf-8")
    lead = html[html.find('<p class="lead">'):]
    lead = lead[:lead.find("</p>")]
    for word in ("시계열", "이미지", "표"):
        assert word in lead, word


def test_label_page_stays_on_start_when_coming_from_studio():
    """studio에서 ?folder= 로 넘어오면 남은 작업으로 건너뛰지 않는다."""
    js = (STATIC / "label.js").read_text(encoding="utf-8")
    tail = js[js.rfind("// 새로 고쳐도"):]
    assert "folder" in tail


def test_drag_overlays_do_not_catch_the_mouse():
    """끄는 동안 보이는 선(wire)·놓을 자리 표시·네모가 마우스를 가로채면, 놓는 순간 상자 대신 그것이 눌려 선이 안 이어짐."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    sel = " ".join(re.findall(r"([^{}]*)\{\s*pointer-events:\s*none", css))
    for name in (".wire", ".drop", ".band"):
        assert name in sel, name


def test_studio_page_fits_window_and_scrolls_inside():
    """studio 화면은 창에 딱 맞고(페이지 스크롤 없음), 안쪽 칸만 스크롤. 판은 남는 공간을 다 씀."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert '<body class="app">' in html
    assert re.search(r"body\.app\s*\{[^}]*height:\s*100vh[^}]*overflow:\s*hidden", css)
    assert re.search(r"body\.app\s*>\s*main\s*\{[^}]*overflow:\s*auto", css)
    assert "max-height: 680px" not in css.split("@media")[0]
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    assert "clientWidth" in js and "clientHeight" in js          # 판(SVG)이 보이는 칸을 다 채움


def test_edge_style_control_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="edge-style"' in html and 'value="ortho"' in html and 'value="curve"' in html


def test_messages_float_and_do_not_push_layout():
    """안내 메시지가 뜰 때 화면이 아래로 밀리면 누르려던 곳이 바뀜: 떠 있는 알림으로, 성공 알림은 잠시 뒤 사라짐."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert re.search(r"body\.app\s*>\s*\.message\s*\{[^}]*position:\s*fixed", css)
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "setTimeout" in app.split("function showMessage")[1].split("\n}\n")[0]


def test_click_after_drag_is_not_swallowed_later():
    """끌기 뒤 따라오는 click을 버리는 표시는 바로 풀려야 함 (안 그러면 다음 진짜 클릭이 무시됨)."""
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    assert js.count("skipClick = true;") == 1 and js.count("ignoreNextClick()") >= 3   # 표시는 도우미 한 곳에서만 켬


def test_late_check_result_is_dropped():
    """바뀌기 전 그래프의 검사 결과가 늦게 와도 그리지 않음: 바꿀 때마다 검사 번호를 올림."""
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    body = app.split("function scheduleCheck()")[1].split("\n}\n")[0]
    assert "st.checkSeq++" in body


def test_mouse_first_editing():
    """키보드 없이: 입력·출력 소켓을 늘 보여 주고, 오른쪽 클릭 메뉴와 층 고르기 메뉴가 있음."""
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "data-inport=" in js and '"contextmenu"' in js and "ctx-menu" in js
    assert not re.search(r"\.port\s*\{[^}]*opacity:\s*0", css)       # 소켓이 숨어 있지 않음


def test_unconnected_box_is_gentle_not_error():
    """막 놓아서 아직 안 이은 상자는 빨간 오류가 아니라 '선을 이어 주세요' (회색 점선)."""
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "loose" in js and "선을 이어 주세요" in js
    assert re.search(r"\.fb\.loose[^{]*\{[^}]*stroke-dasharray", css)


def test_canvas_scrolls_instead_of_shrinking():
    """그림이 판보다 넓어지면 줄어들지 않고 스크롤 (판이 flex면 SVG가 줄어듦)."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    canvas = re.search(r"\n\.canvas\s*\{([^}]*)\}", css).group(1)
    assert "display: flex" not in canvas
    assert re.search(r"\.flow-svg\s*\{[^}]*flex:\s*none", css)


def test_direction_and_size_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    assert 'id="flow-dir"' in html and 'value="v"' in html and 'value="h"' in html
    assert "data-size=" in js and 'class="rsz"' in js and "stripView" in (STATIC / "app.js").read_text(encoding="utf-8")


def test_wrap_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    assert 'id="auto-wrap"' in html and "여기서 줄바꿈" in js and "줄바꿈 풀기" in js and "setWrap" in js


def test_no_he_words_on_screen():
    """DL studio는 동형암호를 다루지 않음: 화면 파일에 암호문·HE·Poly2·모드 전환이 없음."""
    import re as _re
    for f in ("index.html", "app.js", "graph.js", "label.html", "label.js"):
        text = (STATIC / f).read_text(encoding="utf-8")
        for w in ("암호문", "동형암호", "Poly2", "poly2", "heml", "/api/remote", "/api/mode\""):
            assert w not in text, (f, w)
        assert not _re.search(r"HE", text), f


def test_code_dialog_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for i in ('id="btn-code"', 'id="btn-code-train"', 'id="code-dialog"', 'data-fw="pytorch"', 'data-fw="tensorflow"',
              'id="code-copy"', 'id="code-save"'):
        assert i in html, i
    assert '"/api/code"' in js and '"/api/code/save"' in js


def test_web_version_files():
    """웹 버전(GitHub Pages): 저장소 맨 위 index.html이 화면을 불러 Pyodide·TF.js·엔진·다리·화면 순서로 부름."""
    root = STATIC.parents[2]
    html = (root / "index.html").read_text(encoding="utf-8")
    assert (root / ".nojekyll").exists()
    assert 'src/dlstudio/static/index.html' in html and "pyodide" in html and "tfjs@" in html
    order = [html.find(f'"{f}"') for f in ("engine.js", "web.js", "graph.js", "app.js")]
    assert all(i >= 0 for i in order) and order == sorted(order)
    web = (STATIC / "web.js").read_text(encoding="utf-8")
    for f in ("core.py", "web.py", "codegen.py", "data.py", "graph.py", "blocks.py", "tasks.py"):
        assert f'"{f}"' in web, f
    src = STATIC.parent
    listed = set(re.findall(r'"(\w+\.py)"', web))
    assert {p.name for p in src.glob("*.py")} - {"server.py", "cli.py", "__main__.py"} <= listed   # 웹에서 쓰는 모듈은 모두 불러옴
    assert "window.DLS_WEB" in (STATIC / "app.js").read_text(encoding="utf-8")

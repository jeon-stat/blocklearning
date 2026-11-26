"""화면 파일: 필요한 파일이 있고, index.html이 레포 안의 Blockly와 화면 스크립트를 부르며, 외부 주소를 쓰지 않는다."""
import re
from pathlib import Path

import blocklearning

STATIC = Path(blocklearning.__file__).parent / "static"
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
    """검사가 아는 층은 모두 왼쪽 목록에 (Input·Repeat·옛 블록 묶음 빼고)."""
    from blocklearning.graph import LAYER_TYPES
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    for t in LAYER_TYPES:
        if t in ("input", "repeat", "branch", "residual"):
            continue
        assert f'type: "{t}"' in js, t


def test_graph_editing_help_is_folded():
    """조작 설명은 단계 첫머리 줄글이 아니라 오른쪽 [조작 방법] (접힘) 안에만."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    model = html[html.find('id="step-model"'):html.find('id="step-train"')]
    assert '<p class="lead">' not in model
    js = (STATIC / "graph.js").read_text(encoding="utf-8")
    assert '<details class="keys"><summary>조작 방법</summary>' in js and "동그라미" in js


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
    for i in ("series-row", "series-path", "btn-series-scan", "series-box", "series-columns", "series-length",
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
    """BlockLearning은 동형암호를 다루지 않음: 화면 파일에 암호문·HE·Poly2·모드 전환이 없음."""
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
    assert (root / ".nojekyll").exists()
    assert 'src/blocklearning/static/boot.js' in (root / "index.html").read_text(encoding="utf-8")
    html = (STATIC / "boot.js").read_text(encoding="utf-8")
    assert 'index.html' in html and "pyodide" in html and "tfjs@" in html
    order = [html.find(f'"{f}"') for f in ("engine.js", "web.js", "graph.js", "app.js")]
    assert all(i >= 0 for i in order) and order == sorted(order)
    web = (STATIC / "web.js").read_text(encoding="utf-8")
    for f in ("core.py", "web.py", "codegen.py", "data.py", "graph.py", "blocks.py", "tasks.py"):
        assert f'"{f}"' in web, f
    src = STATIC.parent
    listed = set(re.findall(r'"(\w+\.py)"', web))
    assert {p.name for p in src.glob("*.py")} - {"server.py", "cli.py", "__main__.py", "devices.py", "myprojects.py"} <= listed   # 웹에서 쓰는 모듈은 모두 불러옴
    assert "window.DLS_WEB" in (STATIC / "app.js").read_text(encoding="utf-8")


def test_web_storage_and_security():
    """웹 버전 보관·보안: drive.file 권한만, 출입증은 저장소에 안 남김, CSP·SRI, 개인정보처리방침."""
    root = STATIC.parents[2]
    store = (STATIC / "store.js").read_text(encoding="utf-8")
    scopes = set(re.findall(r"https://www\.googleapis\.com/auth/[\w.]+", store))
    assert scopes == {"https://www.googleapis.com/auth/drive.file"}
    assert "localStorage" not in store and "sessionStorage" not in store          # 토큰을 남기지 않음
    html = (root / "index.html").read_text(encoding="utf-8")
    csp = re.search(r'Content-Security-Policy" content="([^"]+)"', html).group(1)
    connect = re.search(r"connect-src ([^;]+)", csp).group(1).split()
    assert set(connect) <= {"'self'", "https://cdn.jsdelivr.net", "https://www.googleapis.com", "https://oauth2.googleapis.com",
                            "https://accounts.google.com"}
    assert "<script>" not in html                                                # 인라인 스크립트 없음
    boot = (STATIC / "boot.js").read_text(encoding="utf-8")
    assert boot.count("sha384-") == 2 and "integrity" in boot
    for f in ("engine.js", "store.js", "web.js", "graph.js", "app.js", "projects.js"):
        assert f'"{f}"' in boot, f
    assert (root / "privacy.html").exists() and "drive.file" in (root / "privacy.html").read_text(encoding="utf-8")


def test_web_training_runs_in_worker():
    """웹 학습은 학습 일꾼(train-worker.js)에서: 화면 스레드가 버벅이지 않게. TF.js는 일꾼도 내용 해시(SRI)를 확인하고 받음."""
    web = (STATIC / "web.js").read_text(encoding="utf-8")
    worker = (STATIC / "train-worker.js").read_text(encoding="utf-8")
    assert "E().train(" not in web.replace("return E().train(plan, d, s, onEpoch, isCancelled)", "")   # 일꾼을 못 쓸 때만 화면 스레드
    assert web.count("trainOffMain(") == 3                                         # 정의 + ④ 학습 + 라벨링 딥러닝 제안
    assert "integrity: tf[1]" in worker and "net.state()" in worker
    assert "globalThis.DLEngine = API" in (STATIC / "engine.js").read_text(encoding="utf-8")


def test_stop_skips_slow_final_checks():
    """[멈추기] 뒤에는 CPU로 맞춰 보기(무거운 모델은 몇 분)를 건너뛰고, 사전학습 특징을 뽑는 중에도 바로 멈춤."""
    eng = (STATIC / "engine.js").read_text(encoding="utf-8")
    assert "const chk = (n) => (halted ? 0 : n);" in eng and "checkedPredict(net, Fte, chk(d.n_test))" in eng
    assert "headFeatures(net, Xtr, MNV3S_B, isCancelled)" in eng and "throw stopped();" in eng


def test_old_name_data_still_opens():
    """이름을 BlockLearning으로 바꿔도 예전(DL studio) 데이터가 그대로: 브라우저 보관함·설정 이름, 드라이브 폴더, .dlstudio 파일."""
    store = (STATIC / "store.js").read_text(encoding="utf-8")
    assert 'indexedDB.open("dlstudio"' in store and '"project.dlstudio"' in store
    assert 'OLD_ROOT_NAME = "DL studio"' in store and "find(OLD_ROOT_NAME)" in store       # 예전 드라이브 폴더는 새 이름으로 바꿔 씀
    projects, web = (STATIC / "projects.js").read_text(encoding="utf-8"), (STATIC / "web.js").read_text(encoding="utf-8")
    assert '"dlstudio-current"' in projects and "(blocklearning|dlstudio)" in projects
    assert '".blocklearning,.dlstudio,.zip"' in web


def test_web_label_page_and_python_files():
    """웹 라벨링 페이지: 첫 페이지와 같은 보안 정책, 라벨링 스크립트, 그리고 Pyodide가 받는 파이썬 파일이 패키지와 맞음."""
    root = STATIC.parents[2]
    index, label = (root / "index.html").read_text(encoding="utf-8"), (root / "label.html").read_text(encoding="utf-8")
    csp = lambda h: re.search(r'Content-Security-Policy" content="([^"]+)"', h).group(1)
    assert csp(index) == csp(label) and 'data-page="label"' in label and "<script>" not in label
    boot = (STATIC / "boot.js").read_text(encoding="utf-8")
    assert '"labels.js"' in boot and '"label.js"' in boot
    web = (STATIC / "web.js").read_text(encoding="utf-8")
    listed = set(re.findall(r'"(\w+\.py)"', web[web.index("PY_FILES"):web.index("];", web.index("PY_FILES"))]))
    have = {p.name for p in STATIC.parent.glob("*.py")} - {"__main__.py", "cli.py", "server.py", "devices.py", "myprojects.py"}
    assert listed == have, (have - listed, listed - have)


def test_every_step_has_previous_button_and_history():
    """② 데이터부터 단계 맨 아래에 [← 이전] (바로 앞 단계로), 단계 이동은 브라우저 기록에도 (뒤로 가기가 사이트를 떠나지 않게)."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    prev = {"data": "task", "model": "data", "train": "model", "predict": "train"}
    for step, back in prev.items():
        sec = html[html.index(f'<section id="step-{step}"'):]
        sec = sec[:sec.index("</section>")]
        assert f'class="primary go back" data-go="{back}"' in sec, step   # 색은 [다음]과 같게
    assert 'go back"' not in html[html.index('<section id="step-task"'):html.index('<section id="step-data"')]
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "history.pushState({ dls: step }" in js and 'addEventListener("popstate"' in js

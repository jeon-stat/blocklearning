"""라벨링 작업실 (L0): 샘플마다 CSV 하나인 폴더에 클래스 정답 붙이기. 작업 파일, 목록, 편집, 되돌리기, 규칙, 가져오기·내보내기."""
import json
import os

import numpy as np
import pytest

from pathlib import Path

from dlstudio.data import DataError, load_series_folder
from dlstudio.labeling import LabelProject


def _folder(tmp, n=12, sub=None, length=30, peak=None):
    """v 열 CSV n개. sub가 있으면 하위 폴더 sub[i % len(sub)]에 나눠 담음. peak(i): i번 샘플의 최댓값."""
    root = tmp / "data"
    for i in range(n):
        d = root / sub[i % len(sub)] if sub else root
        d.mkdir(parents=True, exist_ok=True)
        top = peak(i) if peak else 1.0 + i * 0.1
        vals = [np.sin(j / 3) * 0.1 for j in range(length)]
        vals[length // 2] = top
        (d / f"c{i:02d}.csv").write_text("t,v\n" + "\n".join(f"{j},{x:.4f}" for j, x in enumerate(vals)), encoding="utf-8")
    return root


def _new(tmp, **kw):
    root = _folder(tmp, **kw)
    return LabelProject.new(str(root), "v", ["정상", "이상"], str(tmp / "work.dllabel")), root


# ---------------------------------------------------------------- Task 1: 작업 파일·목록·요약값

def test_new_open_roundtrip_and_counts(tmp_path):
    p, _ = _new(tmp_path)
    c = p.state()["counts"]
    assert c["total"] == 12 and c["unlabeled"] == 12 and c["labeled"] == 0
    q = LabelProject.open(str(tmp_path / "work.dllabel"))
    assert [x["name"] for x in q.state()["classes"]] == ["정상", "이상"]
    meta = json.loads((tmp_path / "work.dllabel").read_text(encoding="utf-8"))
    assert meta["format"] == "dlstudio-label" and meta["version"] == 1 and meta["schema"]["kind"] == "class"


def test_subfolder_labels_imported(tmp_path):
    p, _ = _new(tmp_path, sub=["정상", "이상"])
    it = p.items(size=60)["items"]
    assert all(x["by"] == "import" and x["confirmed"] is False for x in it)
    assert {x["label"] for x in it} == {"정상", "이상"} and p.state()["counts"]["import"] == 12


def test_items_filter_sort_page(tmp_path):
    p, _ = _new(tmp_path, n=25, peak=lambda i: float(i))
    r = p.items(sort={"key": "max", "desc": True}, size=10)
    assert r["items"][0]["name"] == "c24.csv" and r["pages"] == 3 and r["total"] == 25
    assert len(r["items"][0]["thumb"]) == 120 and r["items"][0]["stats"]["max"] == pytest.approx(24)
    assert p.items(filter={"label": "none"})["total"] == 25
    assert p.items(page=2, size=10)["items"][-1]["name"] == "c24.csv"


def test_item_detail_and_downsample(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    for i in range(3):
        v = np.zeros(5000)
        v[1234], v[4321] = 9.0, -7.0
        (root / f"long{i}.csv").write_text("v\n" + "\n".join(f"{x}" for x in v), encoding="utf-8")
    p = LabelProject.new(str(root), "v", ["a", "b"], str(tmp_path / "w.dllabel"))
    d = p.item("long0.csv")
    assert d["n"] == 5000 and len(d["values"]) <= 2000
    assert max(d["values"]) == 9.0 and min(d["values"]) == -7.0          # 줄여도 최대·최소는 남음


def test_files_added_and_removed_after_save(tmp_path):
    p, root = _new(tmp_path)
    p.set_labels(["c00.csv"], "정상")
    (root / "c00.csv").unlink()
    (root / "new.csv").write_text("t,v\n0,1\n1,2\n2,3", encoding="utf-8")
    q = LabelProject.open(str(tmp_path / "work.dllabel"))
    s = q.state()
    assert s["missing"] == 1 and s["counts"]["total"] == 12 and s["counts"]["labeled"] == 0
    assert q.items(filter={"label": "none"})["total"] == 12


def test_bad_sample_marked_not_fatal(tmp_path):
    p, root = _new(tmp_path)
    (root / "c03.csv").write_text("t,w\n0,1\n1,2", encoding="utf-8")            # 열 v 없음
    q = LabelProject.open(str(tmp_path / "work.dllabel"))
    bad = [x for x in q.items(size=60)["items"] if x["error"]]
    assert [x["name"] for x in bad] == ["c03.csv"] and q.state()["counts"]["error"] == 1


def test_unknown_schema_rejected(tmp_path):
    p, _ = _new(tmp_path)
    f = tmp_path / "work.dllabel"
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["schema"]["kind"] = "boxes"
    f.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(DataError):
        LabelProject.open(str(f))


# ---------------------------------------------------------------- Task 2: 편집·되돌리기·규칙·가져오기·내보내기

def _label_of(p, name):
    return next(x for x in p.items(size=999)["items"] if x["name"] == name)


def test_set_confirm_note_undo_redo(tmp_path):
    p, _ = _new(tmp_path)
    assert p.set_labels(["c00.csv", "c01.csv"], "정상") == 2
    x = _label_of(p, "c00.csv")
    assert x["label"] == "정상" and x["by"] == "manual" and x["confirmed"] is True
    assert p.set_labels(["c00.csv", "c01.csv"], "정상") == 0                  # 그대로면 바뀐 것 없음
    p.set_labels(["c00.csv"], None)
    assert _label_of(p, "c00.csv")["label"] is None
    p.note("c01.csv", "모양이 이상함")
    assert _label_of(p, "c01.csv")["note"] == "모양이 이상함"
    assert p.undo() and p.undo()                                              # 메모, 지우기 되돌림
    assert _label_of(p, "c00.csv")["label"] == "정상" and _label_of(p, "c01.csv")["note"] == ""
    assert p.redo() and _label_of(p, "c00.csv")["label"] is None
    q = LabelProject.open(str(tmp_path / "work.dllabel"))                   # 저장도 됨
    assert _label_of(q, "c00.csv")["label"] is None and _label_of(q, "c01.csv")["label"] == "정상"


def test_confirm_only_labeled(tmp_path):
    p, _ = _new(tmp_path, sub=["정상", "이상"])
    assert p.confirm(["정상/c00.csv", "이상/c01.csv"]) == 2
    assert p.state()["counts"]["unconfirmed"] == 10
    p.set_labels(["정상/c02.csv"], None)
    assert p.confirm(["정상/c02.csv"]) == 0                                    # 정답 없는 것은 확인 안 됨


def test_rename_class_and_undo(tmp_path):
    p, _ = _new(tmp_path)
    p.set_labels(["c00.csv"], "이상")
    p.classes_op("rename", name="이상", new="결함")
    assert [c["name"] for c in p.state()["classes"]] == ["정상", "결함"] and _label_of(p, "c00.csv")["label"] == "결함"
    p.undo()
    assert [c["name"] for c in p.state()["classes"]] == ["정상", "이상"] and _label_of(p, "c00.csv")["label"] == "이상"
    p.classes_op("add", name="애매함")
    p.classes_op("color", name="애매함", color="#000000")
    assert p.state()["classes"][2] == {"name": "애매함", "color": "#000000", "count": 0}
    with pytest.raises(DataError):
        p.classes_op("remove", name="이상")                                   # 정답이 남은 클래스
    p.classes_op("remove", name="애매함")
    assert len(p.state()["classes"]) == 2


def test_rule_preview_equals_apply(tmp_path):
    p, _ = _new(tmp_path, n=12, peak=lambda i: float(i))
    p.set_labels(["c09.csv"], "정상")
    conds = [{"stat": "max", "op": ">", "value": 5}]
    m = p.rule_matches(conds)
    assert sorted(m) == ["c06.csv", "c07.csv", "c08.csv", "c10.csv", "c11.csv"]   # 사람이 붙인 c09는 건너뜀
    assert p.rule_apply(conds, "이상") == len(m)
    x = _label_of(p, "c06.csv")
    assert x["by"] == "rule" and x["confirmed"] is False and _label_of(p, "c09.csv")["label"] == "정상"
    assert len(p.rule_matches(conds, overwrite=True)) == 6 and p.rule_apply(conds, "이상", overwrite=True) == 1
    assert p.rule_matches([{"stat": "max", "op": ">", "value": 5}, {"stat": "max", "op": "<=", "value": 7}], overwrite=True) \
        == ["c06.csv", "c07.csv"]


def test_import_and_export_roundtrip(tmp_path):
    p, root = _new(tmp_path)
    lab = tmp_path / "정답.csv"
    lab.write_text("\n".join([f"c{i:02d}{'.csv' if i % 2 else ''},{'정상' if i < 6 else '이상'}" for i in range(12)] + ["zz,정상"]),
                   encoding="utf-8")
    r = p.import_table(str(lab))
    assert r == {"applied": 12, "unknown": 1, "kept_manual": 0} and p.state()["counts"]["import"] == 12
    out = tmp_path / "out.csv"
    assert p.export(str(out)) == 12
    d = load_series_folder(str(root), "v", 16, label_file=str(out))
    assert d.classes == ["이상", "정상"] and len(d.y_train) + len(d.y_test) == 12
    assert p.export(str(tmp_path / "out2.csv"), confirmed_only=True) == 0


def test_save_failure_rolls_back(tmp_path, monkeypatch):
    p, _ = _new(tmp_path)
    def boom(*a, **k):
        raise PermissionError("read-only")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(DataError):
        p.set_labels(["c00.csv"], "정상")
    assert _label_of(p, "c00.csv")["label"] is None and not p.state()["can_undo"]


def test_limits(tmp_path):
    p, _ = _new(tmp_path)
    for k in range(7):
        p.classes_op("add", name=f"k{k}")
    with pytest.raises(DataError):
        p.classes_op("add", name="열번째")
    with pytest.raises(DataError):
        p.set_labels(["c00.csv"], "없는클래스")
    with pytest.raises(DataError):
        p.rule_apply([{"stat": "nope", "op": ">", "value": 1}], "정상")


def test_export_from_subfolder_data_loads(tmp_path):
    """하위 폴더 데이터에서 정답을 고쳐 내보낸 표(이름 = 폴더/파일)도 studio가 읽음."""
    p, root = _new(tmp_path, sub=["정상", "이상"])
    p.set_labels(["정상/c00.csv"], "이상")                                   # 가져온 정답 하나를 고침
    out = tmp_path / "out.csv"
    assert p.export(str(out)) == 12
    d = load_series_folder(str(root), "v", 16, label_file=str(out))
    assert len(d.y_train) + len(d.y_test) == 12 and sorted(np.bincount(np.concatenate([d.y_train, d.y_test]))) == [5, 7]


def test_import_keeps_manual_labels(tmp_path):
    """정답 표 가져오기는 사람이 붙인 정답을 덮지 않고, 건너뛴 수를 알려 줌."""
    p, _ = _new(tmp_path)
    p.set_labels(["c00.csv"], "이상")
    lab = tmp_path / "정답.csv"
    lab.write_text("\n".join(f"c{i:02d},정상" for i in range(12)), encoding="utf-8")
    r = p.import_table(str(lab))
    assert r == {"applied": 11, "unknown": 0, "kept_manual": 1}
    x = _label_of(p, "c00.csv")
    assert x["label"] == "이상" and x["by"] == "manual"


# ---------------------------------------------------------------- 최종 검토 지적
def test_use_with_partial_labels_and_bad_samples(tmp_path):
    """정답을 일부만 붙여도(읽기 오류 샘플이 있어도) 정답 있는 샘플로 학습 데이터를 만듦."""
    p, root = _new(tmp_path, n=20)
    (root / "c19.csv").write_text("t,w\n0,1\n1,2", encoding="utf-8")         # 읽기 오류 샘플
    p = LabelProject.open(str(tmp_path / "work.dllabel"))
    p.set_labels([f"c{i:02d}.csv" for i in range(6)], "정상")
    p.set_labels([f"c{i:02d}.csv" for i in range(6, 12)], "이상")
    out = tmp_path / "out.csv"
    p.export(str(out))
    d = load_series_folder(str(root), "v", 16, label_file=str(out), only_labeled=True)
    assert len(d.y_train) + len(d.y_test) == 12


def test_new_refuses_existing_work_file(tmp_path):
    p, root = _new(tmp_path)
    p.set_labels(["c00.csv"], "정상")
    with pytest.raises(DataError) as e:
        LabelProject.new(str(root), "v", ["정상", "이상"], str(tmp_path / "work.dllabel"))
    assert "열기" in e.value.issues[0].hint
    assert _label_of(LabelProject.open(str(tmp_path / "work.dllabel")), "c00.csv")["label"] == "정상"


def test_open_sanitizes_colors_and_by(tmp_path):
    p, _ = _new(tmp_path)
    p.set_labels(["c00.csv"], "정상")
    f = tmp_path / "work.dllabel"
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["schema"]["classes"][0]["color"] = '"><img src=x onerror=alert(1)>'
    meta["items"]["c00.csv"]["by"] = "<script>"
    f.write_text(json.dumps(meta), encoding="utf-8")
    q = LabelProject.open(str(f))
    import re
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", c["color"]) for c in q.state()["classes"])
    assert _label_of(q, "c00.csv")["by"] in (None, "manual", "rule", "import")


def test_rule_preview_with_label_equals_apply_when_overwriting(tmp_path):
    p, _ = _new(tmp_path, n=12, peak=lambda i: float(i))
    p.set_labels(["c08.csv", "c09.csv"], "이상")
    conds = [{"stat": "max", "op": ">", "value": 5}]
    n = len(p.rule_matches(conds, overwrite=True, label="이상"))
    assert n == p.rule_apply(conds, "이상", overwrite=True) == 4


def test_filter_class_named_like_keyword(tmp_path):
    root = _folder(tmp_path)
    p = LabelProject.new(str(root), "v", ["none", "any"], str(tmp_path / "w.dllabel"))
    p.set_labels(["c00.csv"], "none")
    assert p.items(filter={"label": "cls:none"})["total"] == 1
    assert p.items(filter={"label": "none"})["total"] == 11                  # 정답 없음


def test_export_refuses_paths_inside_data_folder(tmp_path):
    p, root = _new(tmp_path)
    p.set_labels(["c00.csv"], "정상")
    with pytest.raises(DataError):
        p.export(str(root / "c01.csv"))                                     # 데이터 파일 덮어쓰기 금지
    with pytest.raises(DataError):
        p.export(str(root / "정답.csv"))                                    # 데이터 폴더 안도 금지 (다음에 샘플로 읽힘)
    with pytest.raises(DataError):
        LabelProject.new(str(root), "v", ["a", "b"], str(root / "w.dllabel"))


# ---------------------------------------------------------------- L1: 묶기·모델 제안
def _two_shapes(tmp, n_each=10):
    """사인파 n_each개 + 네모파 n_each개 (잡음 조금). 이름 s##(사인), q##(네모)."""
    root = tmp / "shapes"
    root.mkdir()
    rng = np.random.default_rng(0)
    t = np.arange(60)
    for i in range(n_each):
        for name, base in (("s", np.sin(t / 5)), ("q", np.sign(np.sin(t / 5 + 1.0)) * (t > 30))):
            v = base + rng.normal(0, 0.05, len(t))
            (root / f"{name}{i:02d}.csv").write_text("v\n" + "\n".join(f"{x:.4f}" for x in v), encoding="utf-8")
    return LabelProject.new(str(root), "v", ["사인", "네모"], str(tmp / "s.dllabel"))


def test_cluster_separates_shapes(tmp_path):
    p = _two_shapes(tmp_path)
    r = p.cluster(k=2)
    assert r["k"] == 2 and sorted(g["size"] for g in r["groups"]) == [10, 10]
    ids = {x["name"][0]: set() for x in p.items(size=99)["items"]}
    for x in p.items(size=99)["items"]:
        ids[x["name"][0]].add(x["cluster"])
    assert len(ids["s"]) == 1 and len(ids["q"]) == 1 and ids["s"] != ids["q"]   # 모양마다 한 묶음
    g = r["groups"][0]
    assert p.items(filter={"cluster": g["id"]}, size=99)["total"] == g["size"] and g["rep"] in [x["name"] for x in p.items(size=99)["items"]]
    with pytest.raises(DataError):
        p.cluster(k=1)


def test_suggest_labels_rest_with_confidence(tmp_path):
    p = _two_shapes(tmp_path)
    p.set_labels([f"s{i:02d}.csv" for i in range(3)], "사인")
    p.set_labels([f"q{i:02d}.csv" for i in range(3)], "네모")
    r = p.suggest()
    assert r["applied"] == 14 and r["trained_on"] == 6
    rows = {x["name"]: x for x in p.items(size=99)["items"]}
    for n, x in rows.items():
        if x["by"] == "model":
            assert x["label"] == ("사인" if n[0] == "s" else "네모") and x["confirmed"] is False and 0.5 < x["p"] <= 1
    assert rows["s00.csv"]["by"] == "manual"                                  # 사람 정답은 그대로
    first = p.items(sort={"key": "uncertainty"}, size=99)["items"][0]
    assert first["by"] == "model" and first["p"] == min(x["p"] for x in rows.values() if x["by"] == "model")
    assert p.state()["counts"]["model"] == 14
    p.undo()
    assert p.state()["counts"]["model"] == 0


def test_suggest_needs_two_labeled_classes(tmp_path):
    p = _two_shapes(tmp_path)
    p.set_labels(["s00.csv", "s01.csv"], "사인")
    with pytest.raises(DataError):
        p.suggest()


def test_suggest_accuracy_unknown_with_few_labels(tmp_path):
    """학습 정답이 10개 미만이면 떼어 낸 샘플이 한두 개라 정확도 숫자가 0%/100%로 튐 → 모름(None)."""
    p = _two_shapes(tmp_path)
    p.set_labels([f"s{i:02d}.csv" for i in range(4)], "사인")
    p.set_labels([f"q{i:02d}.csv" for i in range(4)], "네모")
    assert p.suggest()["acc"] is None
    p.set_labels([f"s{i:02d}.csv" for i in range(4, 8)], "사인")
    p.set_labels([f"q{i:02d}.csv" for i in range(4, 8)], "네모")
    assert p.suggest()["acc"] == 1.0                                          # 16개: 숫자가 나옴 (모양이 뚜렷해 다 맞음)


# ---------------------------------------------------------------- L2: 이미지·표 데이터, 숫자 정답
def _images(tmp, sub=("밝음", "어두움"), n=6):
    from PIL import Image
    root = tmp / "imgs"
    for k, c in enumerate(sub):
        (root / c).mkdir(parents=True)
        for i in range(n):
            Image.new("L", (20 + i, 16), color=200 - 150 * k + i).save(root / c / f"{i}.png")
    return root


def test_image_folder_project(tmp_path):
    pytest.importorskip("PIL")
    from dlstudio.data import load_image_folder
    root = _images(tmp_path)
    p = LabelProject.new(str(root), "", [], str(tmp_path / "i.dllabel"), kind="image_folder")
    s = p.state()
    assert s["counts"]["total"] == 12 and s["counts"]["import"] == 12 and [c["name"] for c in s["classes"]] == ["밝음", "어두움"]
    x = p.items(sort={"key": "mean", "desc": True}, size=60)["items"][0]
    assert len(x["thumb"]) == 24 * 24 and x["stats"]["width"] >= 20 and x["name"].startswith("밝음/")
    assert p.rule_matches([{"stat": "mean", "op": ">", "value": 0.5}], overwrite=True) == sorted(n for n in p.samples if n.startswith("밝음/"))
    d = p.item("밝음/0.png")
    assert d["image"] is True and d["width"] == 20
    out = tmp_path / "out.csv"
    p.export(str(out))
    ds = load_image_folder(str(root), size=8, label_file=str(out), only_labeled=True)
    assert ds.classes == ["밝음", "어두움"] and len(ds.y_train) + len(ds.y_test) == 12


def test_table_project_and_training_csv(tmp_path):
    from dlstudio.data import load_csv
    f = tmp_path / "공정.csv"
    rows = [f"id{i},{20 + i},{1.0 + (i % 3) * 0.1},메모{i}" for i in range(20)]
    f.write_text("lot,온도,압력,비고\n" + "\n".join(rows), encoding="utf-8")
    p = LabelProject.new(str(f), "", ["양품", "불량"], str(tmp_path / "t.dllabel"), kind="table", id_col="lot")
    assert p.state()["counts"]["total"] == 20 and "id3" in p.samples
    x = p.items(sort={"key": "온도", "desc": True}, size=3)["items"][0]
    assert x["name"] == "id19" and set(x["stats"]) == {"온도", "압력"}
    assert p.rule_apply([{"stat": "온도", "op": ">=", "value": 30}], "불량") == 10
    p.set_labels([f"id{i}" for i in range(10)], "양품")
    assert p.item("id3")["row"] == {"lot": "id3", "온도": "23", "압력": "1.0", "비고": "메모3"}
    out = tmp_path / "학습.csv"
    assert p.export_training(str(out)) == 20
    d = load_csv(str(out))
    assert d.classes == ["불량", "양품"] and d.length == 2                     # 숫자 열(온도·압력) + 정답


def test_number_schema(tmp_path):
    root = _folder(tmp_path, n=12, peak=lambda i: float(i))
    p = LabelProject.new(str(root), "v", [], str(tmp_path / "n.dllabel"), schema="number")
    assert p.state()["schema"] == "number" and p.state()["classes"] == []
    assert p.set_labels(["c00.csv", "c01.csv"], 3.5) == 2
    assert _label_of(p, "c00.csv")["label"] == 3.5
    with pytest.raises(DataError):
        p.set_labels(["c00.csv"], "abc")
    assert p.rule_apply([{"stat": "max", "op": ">", "value": 5}], 7.0) == 6
    lab = tmp_path / "y.csv"
    lab.write_text("\n".join(f"c{i:02d},{i * 1.5}" for i in range(12)), encoding="utf-8")
    assert p.import_table(str(lab))["applied"] == 10                           # 사람이 붙인 c00·c01은 그대로
    assert p.items(sort={"key": "label", "desc": True}, size=1)["items"][0]["name"] == "c11.csv"
    out = tmp_path / "o.csv"
    p.export(str(out))
    assert load_series_folder(str(root), "v", 16, label_file=str(out), only_labeled=True).task == "regression"
    with pytest.raises(DataError):
        p.suggest()
    bad = tmp_path / "bad.csv"
    bad.write_text("c00,abc\n", encoding="utf-8")
    with pytest.raises(DataError):
        p.import_table(str(bad))


# ---------------------------------------------------------------- L3: 시계열 구간
def _stage_folder(tmp, n=12):
    """값 열 v + 단계 열 stage (A 10행, B 15행, C 5행)."""
    root = tmp / "stg"
    root.mkdir()
    for i in range(n):
        stages = ["A"] * 10 + ["B"] * 15 + ["C"] * 5
        (root / f"c{i:02d}.csv").write_text("v,stage\n" + "\n".join(f"{np.sin(j / 3) + i:.3f},{s}" for j, s in enumerate(stages)),
                                            encoding="utf-8")
    return root


def test_segments_paint_erase_undo(tmp_path):
    root = _stage_folder(tmp_path)
    p = LabelProject.new(str(root), "v", ["가열", "유지"], str(tmp_path / "g.dllabel"), schema="segments")
    assert p.state()["schema"] == "segments"
    p.paint("c00.csv", 0, 10, "가열")
    p.paint("c00.csv", 8, 20, "유지")                                        # 겹치면 새 구간이 이김
    assert p.item("c00.csv")["segments"] == [[0, 8, "가열"], [8, 20, "유지"]]
    p.paint("c00.csv", 12, 14, None)                                          # 지우개
    assert p.item("c00.csv")["segments"] == [[0, 8, "가열"], [8, 12, "유지"], [14, 20, "유지"]]
    c = p.state()["counts"]
    assert c["labeled"] == 1 and c["manual"] == 1 and p.state()["classes"][1]["count"] == 1
    assert p.items(filter={"label": "cls:가열"})["total"] == 1
    p.undo()
    assert p.item("c00.csv")["segments"] == [[0, 8, "가열"], [8, 20, "유지"]]
    with pytest.raises(DataError):
        p.paint("c00.csv", 5, 99, "가열")                                    # 길이(30)를 넘음
    with pytest.raises(DataError):
        p.set_labels(["c00.csv"], "가열")                                     # 구간 작업에서는 샘플 정답을 쓰지 않음


def test_segments_import_column_and_export(tmp_path):
    root = _stage_folder(tmp_path)
    p = LabelProject.new(str(root), "v", [], str(tmp_path / "g.dllabel"), schema="segments")
    r = p.import_column("stage")
    assert r["samples"] == 12 and [c["name"] for c in p.state()["classes"]] == ["A", "B", "C"]
    x = p.item("c03.csv")
    assert x["segments"] == [[0, 10, "A"], [10, 25, "B"], [25, 30, "C"]] and x["by"] == "import" and not x["confirmed"]
    out = tmp_path / "seg.csv"
    assert p.export_segments(str(out)) == 36
    lines = out.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "file,start,end,label" and lines[1] == "c00.csv,0,10,A"
    with pytest.raises(DataError):
        p.import_column("없는열")


# ---------------------------------------------------------------- L4: 이상 탐지
def test_anomaly_purpose_candidates_and_normal_export(tmp_path):
    root = _folder(tmp_path, n=12, peak=lambda i: 9.0 if i in (3, 7) else 1.0)
    p = LabelProject.new(str(root), "v", [], str(tmp_path / "a.dllabel"), purpose="anomaly")
    s = p.state()
    assert s["purpose"] == "anomaly" and [c["name"] for c in s["classes"]] == ["정상", "이상"]
    cand = p.normal_candidates(k=2)
    assert len(cand) == 10 and "c03.csv" not in cand and "c07.csv" not in cand    # 가장 큰 묶음 = 정상 후보 (붙이지는 않음)
    assert p.state()["counts"]["labeled"] == 0
    p.set_labels(cand, "정상")
    p.set_labels(["c03.csv", "c07.csv"], "이상")
    out = tmp_path / "normal.csv"
    assert p.export_normals(str(out)) == 10
    lines = out.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "file" and len(lines) == 11 and "c03.csv" not in lines
    assert LabelProject.open(str(tmp_path / "a.dllabel")).state()["purpose"] == "anomaly"


# ---------------------------------------------------------------- L5: 이미지 상자 (객체 탐지)
def test_boxes_set_counts_undo_and_exports(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    root = tmp_path / "det"
    (root / "a").mkdir(parents=True)
    for i in range(3):
        Image.new("L", (100, 50), color=128).save(root / ("a" if i == 2 else ".") / f"{i}.png") if i == 2 else \
            Image.new("L", (100, 50), color=128).save(root / f"{i}.png")
    p = LabelProject.new(str(root), "", ["결함", "글자"], str(tmp_path / "d.dllabel"), kind="image_folder", schema="boxes")
    assert p.state()["schema"] == "boxes" and p.state()["counts"]["import"] == 0        # 상자 작업은 하위 폴더를 정답으로 안 씀
    p.set_boxes("0.png", [[0.1, 0.2, 0.3, 0.4, "결함"], [0.5, 0.5, 0.2, 0.2, "글자"]])
    p.set_boxes("a/2.png", [[0.0, 0.0, 1.0, 1.0, "결함"]])
    s = p.state()
    assert s["counts"]["labeled"] == 2 and [c["count"] for c in s["classes"]] == [2, 1]
    assert p.items(filter={"label": "cls:글자"})["total"] == 1
    assert p.item("0.png")["boxes"][1] == [0.5, 0.5, 0.2, 0.2, "글자"]
    p.undo()
    assert p.state()["counts"]["labeled"] == 1
    p.redo()
    with pytest.raises(DataError):
        p.set_boxes("1.png", [[0.8, 0.1, 0.5, 0.1, "결함"]])                          # 그림 밖으로 나감
    with pytest.raises(DataError):
        p.set_boxes("1.png", [[0.1, 0.1, 0.1, 0.1, "없는클래스"]])
    with pytest.raises(DataError):
        p.set_labels(["1.png"], "결함")
    yolo = tmp_path / "yolo"
    assert p.export_yolo(str(yolo)) == 3
    assert (yolo / "classes.txt").read_text(encoding="utf-8").split() == ["결함", "글자"]
    assert (yolo / "labels" / "0.txt").read_text(encoding="utf-8").splitlines() == ["0 0.25 0.4 0.3 0.4", "1 0.6 0.6 0.2 0.2"]
    assert (yolo / "labels" / "a" / "2.txt").exists()
    coco = json.loads(Path(p.export_coco(str(tmp_path / "coco.json"))).read_text(encoding="utf-8"))
    assert [c["name"] for c in coco["categories"]] == ["결함", "글자"] and len(coco["images"]) == 2 and len(coco["annotations"]) == 3
    a0 = coco["annotations"][0]
    assert a0["bbox"] == [10.0, 10.0, 30.0, 20.0] and a0["area"] == 600.0              # 픽셀 (그림 100×50)


# ---------------------------------------------------------------- L6: 여러 사람이 나눠 하기 (파일로)
def test_who_recorded_and_filter(tmp_path):
    p, _ = _new(tmp_path)
    p.who = "김연구"
    p.set_labels(["c00.csv", "c01.csv"], "정상")
    p.who = "이현장"
    p.set_labels(["c02.csv"], "이상")
    assert _label_of(p, "c00.csv")["who"] == "김연구" and _label_of(p, "c02.csv")["who"] == "이현장"
    assert p.items(filter={"who": "김연구"})["total"] == 2 and sorted(p.state()["labelers"]) == ["김연구", "이현장"]


def test_split_into_assigned_parts(tmp_path):
    p, _ = _new(tmp_path)
    p.set_labels(["c05.csv"], "이상")
    files = p.split(3)
    assert len(files) == 3 and all(Path(f).exists() for f in files)
    parts = [LabelProject.open(f) for f in files]
    assert sorted(sum((list(q.samples) for q in parts), [])) == sorted(p.samples)          # 겹치지 않고 전부
    assert [q.state()["counts"]["total"] for q in parts] == [4, 4, 4]
    has5 = next(q for q in parts if "c05.csv" in q.samples)
    assert _label_of(has5, "c05.csv")["label"] == "이상"                                     # 이미 붙인 정답은 함께 감


def test_merge_adds_agrees_conflicts_and_resolve(tmp_path):
    p, root = _new(tmp_path)
    q = LabelProject.new(str(root), "v", ["정상", "이상"], str(tmp_path / "other.dllabel"))
    p.who, q.who = "나", "동료"
    p.set_labels(["c00.csv", "c01.csv"], "정상")
    q.set_labels(["c00.csv"], "정상")                                          # 일치
    q.set_labels(["c01.csv"], "이상")                                          # 충돌
    q.set_labels(["c02.csv", "c03.csv"], "이상")                               # 내 쪽엔 없음 → 가져옴
    r = p.merge(str(tmp_path / "other.dllabel"))
    assert r == {"added": 2, "agreed": 1, "conflicts": 1, "agreement": 0.5}
    assert _label_of(p, "c02.csv")["who"] == "동료" and p.state()["counts"]["conflict"] == 1
    assert p.items(filter={"conflict": True})["total"] == 1
    c = p.item("c01.csv")["conflict"]
    assert c == {"label": "이상", "who": "동료"}
    p.resolve("c01.csv", "theirs")
    x = _label_of(p, "c01.csv")
    assert x["label"] == "이상" and p.state()["counts"]["conflict"] == 0
    p.undo()
    assert p.state()["counts"]["conflict"] == 1
    other = tmp_path / "other_data"
    _folder(other)
    z = LabelProject.new(str(other / "data"), "v", ["정상", "이상"], str(tmp_path / "z.dllabel"))
    with pytest.raises(DataError):
        p.merge(str(tmp_path / "z.dllabel"))                                 # 다른 데이터의 작업은 합칠 수 없음


# ---------------------------------------------------------------- L1–L6 최종 검토 지적
def test_rename_and_remove_class_in_region_schemas(tmp_path):
    root = _stage_folder(tmp_path)
    p = LabelProject.new(str(root), "v", ["가열", "유지"], str(tmp_path / "g.dllabel"), schema="segments")
    p.paint("c00.csv", 0, 10, "가열")
    p.classes_op("rename", name="가열", new="heat")
    assert p.item("c00.csv")["segments"] == [[0, 10, "heat"]] and p.state()["classes"][0]["count"] == 1
    with pytest.raises(DataError):
        p.classes_op("remove", name="heat")                                     # 구간에 쓰는 클래스는 못 지움
    p.undo()
    assert p.item("c00.csv")["segments"] == [[0, 10, "가열"]]


def test_split_merge_back_three_way(tmp_path):
    p, root = _new(tmp_path)
    p.rule_apply([{"stat": "max", "op": ">", "value": 0}], "정상")              # 전부 규칙 정답 (확인 전)
    files = p.split(2)
    q = LabelProject.open(files[0])
    mine = sorted(q.samples)
    q.who = "동료"
    q.confirm(mine[:2])                                                        # 확인
    q.set_labels([mine[2]], "이상")                                            # 고침
    q.set_labels([mine[3]], None)                                              # 지움
    r = p.merge(files[0])
    assert r["conflicts"] == 0
    assert all(_label_of(p, n)["confirmed"] for n in mine[:2])
    assert _label_of(p, mine[2])["label"] == "이상" and _label_of(p, mine[3])["label"] is None


def test_second_suggest_keeps_confirmed(tmp_path):
    p = _two_shapes(tmp_path)
    p.set_labels([f"s{i:02d}.csv" for i in range(3)], "사인")
    p.set_labels([f"q{i:02d}.csv" for i in range(3)], "네모")
    p.suggest()
    p.confirm(["s05.csv"])
    p.suggest()
    assert _label_of(p, "s05.csv")["confirmed"] is True


def test_conflict_values_sanitized_and_resolve_validated(tmp_path):
    root = _folder(tmp_path)
    p = LabelProject.new(str(root), "v", [], str(tmp_path / "n.dllabel"), schema="number")
    p.set_labels(["c00.csv"], 1.0)
    f = tmp_path / "n.dllabel"
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["items"]["c00.csv"]["conflict"] = {"label": '"><img src=x>', "who": "x"}
    f.write_text(json.dumps(meta), encoding="utf-8")
    q = LabelProject.open(str(f))
    assert q.item("c00.csv")["conflict"] is None                               # 숫자 작업에 글자 충돌 값 → 버림


def test_table_training_skips_error_rows(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("lot,a,b\n" + "\n".join(f"id{i},{i},{i % 3}" for i in range(12)) + "\nidx,x,1\nidy,5\n", encoding="utf-8")
    p = LabelProject.new(str(f), "", ["양품", "불량"], str(tmp_path / "t.dllabel"), kind="table", id_col="lot")
    p.set_labels(["idx", "idy", "id0", "id1"], "양품")
    out = tmp_path / "train.csv"
    assert p.export_training(str(out)) == 2                                    # 오류 행(idx, idy)은 뺌


def test_yolo_export_refuses_same_stem(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    root = tmp_path / "im"
    root.mkdir()
    Image.new("L", (10, 10)).save(root / "a.png")
    Image.new("L", (10, 10)).save(root / "a.jpg")
    p = LabelProject.new(str(root), "", ["k"], str(tmp_path / "b.dllabel"), kind="image_folder", schema="boxes")
    p.set_boxes("a.png", [[0.1, 0.1, 0.2, 0.2, "k"]])
    p.set_boxes("a.jpg", [[0.1, 0.1, 0.2, 0.2, "k"], [0.5, 0.5, 0.2, 0.2, "k"]])
    with pytest.raises(DataError):
        p.export_yolo(str(tmp_path / "y"))
    with pytest.raises(DataError):
        p.export(str(tmp_path / "x.csv"))                                     # 상자 작업은 정답 표 CSV가 아님 (YOLO·COCO로)


def test_boxes_sanitized_on_open(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    root = tmp_path / "im"
    root.mkdir()
    Image.new("L", (10, 10)).save(root / "a.png")
    p = LabelProject.new(str(root), "", ["k"], str(tmp_path / "b.dllabel"), kind="image_folder", schema="boxes")
    p.set_boxes("a.png", [[0.1, 0.1, 0.2, 0.2, "k"]])
    f = tmp_path / "b.dllabel"
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["items"]["a.png"]["boxes"] += [[float("nan"), 0, 0.1, 0.1, "k"], [0.1, 0.1, -0.2, 0.1, "k"], [0.1, 0.1, 0.1, 0.1, "유령"]]
    f.write_text(json.dumps(meta), encoding="utf-8")
    assert LabelProject.open(str(f)).item("a.png")["boxes"] == [[0.1, 0.1, 0.2, 0.2, "k"]]


def test_relabel_same_class_clears_conflict(tmp_path):
    p, root = _new(tmp_path)
    q = LabelProject.new(str(root), "v", ["정상", "이상"], str(tmp_path / "o.dllabel"))
    p.set_labels(["c00.csv"], "정상")
    q.set_labels(["c00.csv"], "이상")
    p.merge(str(tmp_path / "o.dllabel"))
    p.set_labels(["c00.csv"], "정상")                                          # 내 것으로 다시 붙임 = 결정
    assert p.state()["counts"]["conflict"] == 0

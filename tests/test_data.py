"""DL studio 데이터: CSV(한 행 = 샘플, 정답 열 하나)와 합성 예시 데이터 → Dataset."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import dlstudio.data as data_mod
from tests.synth import example_arrays
from dlstudio.data import (DataError, load_csv, load_example, load_image_folder, load_series_folder,
                              make_dataset, read_series_folder, scan_series_folder)


def _write(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ([",".join(header)] if header else []) + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _rows(n=20, L=8, labels=None, seed=0):
    rng = np.random.default_rng(seed)
    labels = labels if labels is not None else [i % 2 for i in range(n)]
    return [list(np.round(rng.normal(size=L), 4)) + [labels[i]] for i in range(n)]


def test_basic_classification(tmp_path):
    d = load_csv(_write(tmp_path / "a.csv", _rows()))
    assert d.task == "classification" and d.length == 8
    assert len(d.y_train) + len(d.y_test) == 20
    assert set(d.y_test.tolist()) == {0, 1}                      # 클래스 비율 유지 → 시험에 둘 다


def test_header_row_detected(tmp_path):
    d = load_csv(_write(tmp_path / "a.csv", _rows(), header=[f"t{i}" for i in range(1, 9)] + ["label"]))
    assert len(d.y_train) + len(d.y_test) == 20


def test_string_labels_mapped(tmp_path):
    rows = _rows(labels=["normal" if i % 2 else "abnormal" for i in range(20)])
    d = load_csv(_write(tmp_path / "a.csv", rows))
    assert d.classes == ["abnormal", "normal"]
    assert set(np.concatenate([d.y_train, d.y_test]).tolist()) == {0, 1}


def test_regression_auto(tmp_path):
    rows = _rows(labels=[0.5 + i * 0.37 for i in range(20)])
    d = load_csv(_write(tmp_path / "a.csv", rows))
    assert d.task == "regression" and d.classes == []


def test_non_numeric_cell_reports_location(tmp_path):
    rows = _rows()
    rows[2][4] = "abc"
    with pytest.raises(DataError) as e:
        load_csv(_write(tmp_path / "a.csv", rows))
    assert any(i.row == 3 and i.col == 5 for i in e.value.issues)


def test_empty_cell_reports_location(tmp_path):
    rows = _rows()
    rows[4][1] = ""
    with pytest.raises(DataError) as e:
        load_csv(_write(tmp_path / "a.csv", rows))
    assert any(i.row == 5 and i.col == 2 for i in e.value.issues)


def test_single_class_rejected(tmp_path):
    with pytest.raises(DataError):
        load_csv(_write(tmp_path / "a.csv", _rows(labels=[1] * 20)))


def test_too_few_samples(tmp_path):
    with pytest.raises(DataError):
        load_csv(_write(tmp_path / "a.csv", _rows(n=5)))


def test_ragged_rows(tmp_path):
    rows = _rows()
    rows[7] = rows[7][:5]
    with pytest.raises(DataError) as e:
        load_csv(_write(tmp_path / "a.csv", rows))
    assert any(i.row == 8 for i in e.value.issues)


def test_normalize_uses_train_stats(tmp_path):
    d = load_csv(_write(tmp_path / "a.csv", _rows(n=40)))
    assert np.allclose(d.X_train.mean(0), 0, atol=1e-9)
    assert np.allclose(d.X_train.std(0), 1, atol=1e-9)


def test_split_reproducible(tmp_path):
    p = _write(tmp_path / "a.csv", _rows(n=40))
    assert np.array_equal(load_csv(p, seed=3).X_test, load_csv(p, seed=3).X_test)


def test_non_ascii_path(tmp_path):
    d = load_csv(_write(tmp_path / "한글폴더" / "데이터.csv", _rows()))
    assert d.length == 8


def test_example_data():
    d = load_example()
    assert d.length == 140 and len(d.y_train) + len(d.y_test) == 4990          # MIT-BIH 박동 (정상·이상 각 2,500, 예측 예시용 10개 뺌)
    y = np.concatenate([d.y_train, d.y_test])
    assert (y == 0).sum() == 2495 and (y == 1).sum() == 2495
    assert d.classes == ["정상", "이상"]
    json.dumps(d.summary(), ensure_ascii=False)


def test_make_dataset_from_arrays():
    X, y = example_arrays(n=200, length=32)
    d = make_dataset(X, y)
    assert d.task == "classification" and d.length == 32
    s = d.summary()
    assert s["n_train"] + s["n_test"] == 200 and len(s["preview"]) <= 6




def test_example_kind_unknown():
    with pytest.raises(ValueError):
        load_example("nope")


def test_pandas_style_numeric_header_detected(tmp_path):
    """pandas to_csv 기본 머리글 (0,1,…,label 또는 0,1,…,8)도 머리글로."""
    rows = _rows()
    d = load_csv(_write(tmp_path / "a.csv", rows, header=[str(i) for i in range(8)] + ["label"]))
    assert d.classes == ["0", "1"]
    d = load_csv(_write(tmp_path / "b.csv", rows, header=[str(i) for i in range(9)]))
    assert d.classes == ["0", "1"]


def test_korean_excel_cp949_csv(tmp_path):
    p = tmp_path / "excel.csv"
    rows = _rows(labels=["정상" if i % 2 else "이상" for i in range(20)])
    p.write_bytes(("\n".join(",".join(map(str, r)) for r in rows) + "\n").encode("cp949"))
    d = load_csv(str(p))
    assert d.classes == ["이상", "정상"]


# ---------------------------------------------------------------- 과제별 예시 데이터







def test_csv_header_becomes_feature_names(tmp_path):
    d = load_csv(_write(tmp_path / "a.csv", _rows(), header=[f"t{i}" for i in range(1, 9)] + ["label"]))
    assert d.summary()["features"] == [f"t{i}" for i in range(1, 9)]
    d = load_csv(_write(tmp_path / "b.csv", _rows()))
    assert d.summary()["features"] == [f"열 {i}" for i in range(1, 9)]




def test_image_normalization_uses_one_mean_for_all_pixels():
    X = np.zeros((40, 16))
    X[:, 5] = 1.0                                                        # 늘 0인 테두리 픽셀이 있어도
    d = make_dataset(X, np.arange(40) % 2, image=(4, 4))
    assert np.allclose(d.mean, d.mean[0]) and np.allclose(d.std, d.std[0])
    assert np.isfinite(d.X_train).all()


def test_non_image_dataset_has_no_image():
    d = load_example("ecg")
    assert d.image is None and d.summary()["image"] is None


def _img_folder(tmp_path, classes=("양품", "불량"), n=6):
    from PIL import Image
    for c in classes:
        (tmp_path / c).mkdir()
        for i in range(n):
            mode = ["L", "RGB", "RGBA", "P"][i % 4]
            color = (i * 30) % 255 if mode in ("L", "P") else (i * 30, 0, 0) + ((255,) if mode == "RGBA" else ())
            Image.new(mode, (20 + i, 12), color=color).save(tmp_path / c / f"사진 {i}.png")
    (tmp_path / classes[0] / "Thumbs.db").write_bytes(b"x")
    (tmp_path / "desktop.ini").write_text("x")
    return tmp_path


def test_image_folder_mixed_modes_korean_names_and_junk(tmp_path):
    pytest.importorskip("PIL")
    d = load_image_folder(str(_img_folder(tmp_path)), size=16)
    assert d.classes == sorted(["양품", "불량"]) and d.image == (16, 16) and d.length == 256
    assert len(d.y_train) + len(d.y_test) == 12
    assert 0 <= d.raw(d.X_train).min() + 1e-9 and d.raw(d.X_train).max() <= 1 + 1e-9
    assert d.source["folder"] == str(tmp_path) and d.source["size"] == 16


def test_image_folder_errors(tmp_path):
    pytest.importorskip("PIL")
    one = tmp_path / "one"
    one.mkdir()
    with pytest.raises(DataError):                                       # 클래스 폴더 하나뿐
        load_image_folder(str(_img_folder(one, classes=("a",))))
    (tmp_path / "x").mkdir()
    bad = _img_folder(tmp_path / "x", classes=("a", "b"), n=2)
    (bad / "a" / "1.png").write_bytes(b"not png")                        # 정상 그림들 사이에 깨진 파일 하나
    with pytest.raises(DataError) as e:
        load_image_folder(str(bad))
    assert "1.png" in e.value.issues[0].message
    with pytest.raises(DataError):
        load_image_folder(str(one), size=20)                             # 8·16·28·32만


def test_image_folder_without_pillow_and_too_many(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    folder = str(_img_folder(tmp_path))
    monkeypatch.setattr(data_mod, "MAX_IMAGES", 5)                       # 12장 > 5장 → 거절
    with pytest.raises(DataError):
        load_image_folder(folder)
    monkeypatch.setattr(data_mod, "MAX_IMAGES", 20000)
    monkeypatch.setitem(sys.modules, "PIL", None)                        # Pillow 없음 → 설치 안내
    with pytest.raises(DataError) as e:
        load_image_folder(folder)
    assert "pillow" in (e.value.issues[0].message + e.value.issues[0].hint).lower()


def test_pixel_csv_square_and_scaled(tmp_path):
    text = "\n".join(",".join(str((i * 7 + j) % 256) for j in range(16)) + f",{'ab'[i % 2]}" for i in range(30))
    p = tmp_path / "px.csv"
    p.write_text(text, encoding="utf-8")
    d = load_csv(str(p), image_width=0)
    assert d.image == (4, 4) and d.raw(d.X_train).max() <= 1 + 1e-9    # 0–255 → 0–1
    d2 = load_csv(str(p), image_width=8)
    assert d2.image == (2, 8)
    with pytest.raises(DataError):
        load_csv(str(p), image_width=5)                                  # 16을 5로 못 나눔
    assert load_csv(str(p)).image is None                                # 너비를 안 주면 이미지가 아님


def test_image_folder_16bit_transparent_and_mac_junk(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
        for i in range(3):                                                      # 샘플 최소 10개를 넘게
            g = (np.tile(np.arange(16), (16, 1)) * 256).astype(np.uint16)      # 16비트 그라데이션 0–3840
            Image.fromarray(g).save(tmp_path / c / f"g16_{i}.png")
            rgba = np.zeros((16, 16, 4), np.uint8)
            rgba[4:12, 6:10, 3] = 255                                           # 투명 배경 위 검은 막대
            Image.fromarray(rgba, "RGBA").save(tmp_path / c / f"icon_{i}.png")
        (tmp_path / c / "._icon.png").write_bytes(b"\x00\x05\x16\x07 mac resource fork")   # macOS AppleDouble
    (tmp_path / "__MACOSX" / "a").mkdir(parents=True)
    (tmp_path / "__MACOSX" / "a" / "._g16.png").write_bytes(b"junk")
    d = load_image_folder(str(tmp_path), size=16)
    assert d.classes == ["a", "b"]
    raw = d.raw(np.concatenate([d.X_train, d.X_test]))
    kinds = [len(np.unique(np.round(r, 3))) for r in raw]
    assert min(kinds) > 1                      # 투명 아이콘이 통째로 검게 되지 않음 (흰 배경 + 검은 막대)
    assert max(kinds) > 2                      # 16비트 그라데이션이 0/1 두 값으로 뭉개지지 않음




def _series(tmp, files):          # files: {"상대경로": [값...]} → 머리글 t,value,stage의 CSV
    for rel, vals in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("t,value,stage\n" + "\n".join(f"{i},{v},S{i % 3}" for i, v in enumerate(vals)), encoding="utf-8")
    return tmp


def test_scan_series_folder_modes(tmp_path):
    a = _series(tmp_path / "a", {**{f"정상/c{i}.csv": [1.0 + j % 3 for j in range(20 + i)] for i in range(6)},
                                 **{f"이상/c{i}.csv": [2.0 + j % 2 for j in range(30)] for i in range(6)}})
    (a / "그림.png").write_bytes(b"x")
    (a / "__MACOSX").mkdir()
    (a / ".x.csv").write_text("t,value\n1,2")
    s = scan_series_folder(str(a))
    assert s["mode"] == "folders" and s["classes"] == ["이상", "정상"] and s["n_files"] == 12
    assert s["lengths"][0] == 20 and s["lengths"][2] == 30 and s["columns"] == ["t", "value"] and s["default_column"] == "value"
    assert s["default_length"] == 24                                       # 중앙값 27.5 → 8의 배수로 내림
    b = _series(tmp_path / "b", {f"c{i}.csv": [float(i)] * 16 for i in range(12)})
    assert scan_series_folder(str(b))["mode"] == "none"
    (b / "labels.csv").write_text("file,label\n" + "\n".join(f"c{i},{'ab'[i % 2]}" for i in range(12)), encoding="utf-8")
    s = scan_series_folder(str(b), label_file=str(b / "labels.csv"))
    assert s["mode"] == "file" and s["n_files"] == 12                     # 정답 표 자신은 샘플이 아님


def test_read_series_resample_and_crop(tmp_path):
    f = _series(tmp_path, {"x/a.csv": [0, 10], "x/b.csv": [5], "y/c.csv": [0, 1, 2, 3, 4, 5, 6, 7]})
    names, X, y, mode = read_series_folder(str(f), "value", 9)             # 길이는 8 이상 (설계 4절)
    assert mode == "folders" and X.shape == (3, 9)
    assert np.allclose(X[names.index("x/a.csv")], np.linspace(0, 10, 9)) and np.allclose(X[names.index("x/b.csv")], 5)
    _, Xc, _, _ = read_series_folder(str(f), "value", 9, resample="crop")
    assert np.allclose(Xc[names.index("y/c.csv")], [0, 1, 2, 3, 4, 5, 6, 7, 7])
    assert np.allclose(Xc[names.index("x/a.csv")], [0] + [10] * 8)


def test_series_label_file_matching_and_errors(tmp_path):
    f = _series(tmp_path / "d", {f"Cycle_{i}.csv": [float(i % 2)] * 12 for i in range(12)})
    lab = tmp_path / "정답.csv"
    names = [" cycle_0.CSV ", "Cycle_1.csv"] + [f"Cycle_{i}" for i in range(2, 12)]
    lab.write_text("name,y\n" + "\n".join(f"{n},{i % 2}" for i, n in enumerate(names)), encoding="utf-8")
    d = load_series_folder(str(f), "value", 8, label_file=str(lab))
    assert d.task == "classification" and len(d.y_train) + len(d.y_test) == 12 and d.source["series_folder"] == str(f)
    lab.write_text("name,y\nCycle_0,a\n", encoding="utf-8")
    with pytest.raises(DataError) as e:
        load_series_folder(str(f), "value", 8, label_file=str(lab))
    assert "Cycle_1" in e.value.issues[0].message                         # 정답 표에 없는 파일 이름
    with pytest.raises(DataError) as e:
        load_series_folder(str(f), "value", 8)                            # 정답 없음
    assert "정답" in e.value.issues[0].message


def test_series_mixed_encodings_and_bad_values(tmp_path):
    f = tmp_path / "m"
    (f / "a").mkdir(parents=True)
    (f / "b").mkdir()
    for i in range(6):
        (f / "a" / f"{i}.csv").write_bytes(("시간;값\n" + "\n".join(f"{j};{j}" for j in range(10))).encode("cp949"))
        (f / "b" / f"{i}.csv").write_text("시간,값\n" + "\n".join(f"{j},{-j}" for j in range(10)), encoding="utf-8")
    assert load_series_folder(str(f), "값", 8).length == 8
    (f / "b" / "0.csv").write_text("시간,값\n0,1\n1,abc\n2,3", encoding="utf-8")
    with pytest.raises(DataError) as e:
        load_series_folder(str(f), "값", 8)
    assert "0.csv" in e.value.issues[0].message and "abc" in e.value.issues[0].message


def test_series_default_column_skips_constant_ids(tmp_path):
    """사이클 번호처럼 파일 안에서 늘 같은 숫자 열은 기본 값 열로 고르지 않음 (값이 변하는 숫자 열 중 마지막)."""
    for i in range(12):
        (tmp_path / f"c{i}.csv").write_text("time,power,cycle_no\n" + "\n".join(f"t{j},{np.sin(j)},{i}" for j in range(30)),
                                            encoding="utf-8")
    s = scan_series_folder(str(tmp_path))
    assert s["columns"] == ["power", "cycle_no"] and s["default_column"] == "power"


def test_label_table_without_header_keeps_first_row(tmp_path):
    f = _series(tmp_path / "d", {f"c{i}.csv": [float(j % 3) for j in range(12)] for i in range(12)})
    lab = tmp_path / "정답.csv"
    lab.write_text("\n".join(f"c{i},{'ab'[i % 2]}" for i in range(12)), encoding="utf-8")   # 화면 안내처럼 머리글 없이
    d = load_series_folder(str(f), "value", 8, label_file=str(lab))
    assert len(d.y_train) + len(d.y_test) == 12


def test_label_table_blank_and_conflicting_labels(tmp_path):
    f = _series(tmp_path / "d", {f"c{i}.csv": [float(j % 3) for j in range(12)] for i in range(12)})
    lab = tmp_path / "정답.csv"
    lab.write_text("file,y\n" + "\n".join(f"c{i},{'' if i == 3 else 'ab'[i % 2]}" for i in range(12)), encoding="utf-8")
    with pytest.raises(DataError) as e:
        load_series_folder(str(f), "value", 8, label_file=str(lab))
    assert "5행" in e.value.issues[0].message                             # c3의 정답 칸이 빔 (머리글 포함 5행)
    lab.write_text("file,y\n" + "\n".join(f"c{i},{'ab'[i % 2]}" for i in range(12)) + "\nc0,b", encoding="utf-8")
    with pytest.raises(DataError) as e:
        load_series_folder(str(f), "value", 8, label_file=str(lab))
    assert "c0" in e.value.issues[0].message                              # 같은 파일에 다른 정답 두 개


def test_headerless_sample_files_with_text_time_column(tmp_path):
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
        for i in range(6):
            (tmp_path / c / f"{i}.csv").write_text(
                "\n".join(f"2024-01-01 00:00:{j:02d},{np.sin(j) + i * 0.1:.4f}" for j in range(20)), encoding="utf-8")
    s = scan_series_folder(str(tmp_path))
    assert s["columns"] == ["열 2"] and s["lengths"][0] == 20               # 첫 행도 값
    assert load_series_folder(str(tmp_path), "열 2", 16).length == 16


def test_scan_uses_first_file_with_numbers_and_caps_default_length(tmp_path, monkeypatch):
    f = _series(tmp_path / "d", {f"cycle_{i}.csv": [float(j % 5) for j in range(40)] for i in range(12)})
    (f / "answers.csv").write_text("file,label\n" + "\n".join(f"cycle_{i},a" for i in range(12)), encoding="utf-8")
    s = scan_series_folder(str(f))
    assert s["default_column"] == "value" and any("answers.csv" in x for x in s["skipped"])
    monkeypatch.setattr(data_mod, "SERIES_MAX_LEN", 16)
    assert scan_series_folder(str(f))["default_length"] == 16              # 길이 중앙값 40 → 한도 16


def test_label_table_mode_ignores_unmatched_subfolders(tmp_path):
    """맨 위 CSV + 관계없는 하위 폴더(backup/): 표에 없는 하위 폴더 CSV는 쓰지 않음 (예전 동작 유지)."""
    f = _series(tmp_path / "d", {**{f"c{i}.csv": [float(j % 3) for j in range(12)] for i in range(12)},
                                 "backup/old1.csv": [1.0] * 12, "backup/old2.csv": [1.0] * 12})
    lab = tmp_path / "y.csv"
    lab.write_text("file,y\n" + "\n".join(f"c{i},{'ab'[i % 2]}" for i in range(12)), encoding="utf-8")
    assert len(load_series_folder(str(f), "value", 8, label_file=str(lab)).y_test) > 0



# ---------------------------------------------------------------- 예시 데이터: 모두 공개된 실제 데이터 (출처 파일과 함께)

REAL = [  # 종류, 과제, 클래스 수(회귀 0), 길이, 그림, 샘플 수
    ("ecg", "classification", 2, 140, None, 4990),                 # 예측 예시용 10개는 빼 둠 (*_new)
    ("bearing", "classification", 5, 512, None, 1190),
    ("steel_plates", "classification", 7, 27, None, 1931),
    ("concrete", "regression", 0, 8, None, 1020),
    ("hydraulic_valve", "regression", 0, 200, None, 2195),
    ("steelblast", "classification", 2, 64 * 64, (64, 64), 1644),
    ("pneumonia", "classification", 2, 28 * 28, (28, 28), 1190),
    ("digits", "classification", 10, 32 * 32, (32, 32), 4667),
]


@pytest.mark.parametrize("kind,task,k,length,image,n", REAL, ids=[r[0] for r in REAL])
def test_real_examples(kind, task, k, length, image, n):
    from dlstudio.data import EXAMPLES
    d = load_example(kind)
    assert d.task == task and len(d.classes) == k and d.length == length and d.image == image
    assert len(d.X_train) + len(d.X_test) == n and np.isfinite(d.X_train).all()
    json.dumps(d.summary(), ensure_ascii=False)
    src = Path(__file__).resolve().parents[1] / "src" / "dlstudio" / "data" / f"{EXAMPLES[kind][0]}.SOURCE.txt"
    assert "CC BY 4.0" in src.read_text(encoding="utf-8") or "Open Data Commons Attribution" in src.read_text(encoding="utf-8")
    if image:
        raw = d.raw(d.X_train)
        assert raw.min() >= -1e-6 and raw.max() <= 1 + 1e-6            # 0–1 픽셀로 되돌아옴


def test_old_synthetic_names_load_real_data():
    """예전(합성) 예시 이름으로 저장한 프로젝트도 열림 → 바꾼 실제 데이터."""
    from dlstudio.data import ALIASES, EXAMPLE_KINDS
    assert set(ALIASES.values()) <= set(EXAMPLE_KINDS)
    assert load_example("vibration").classes == load_example("bearing").classes
    assert load_example("process_reg").features[0] == "시멘트 (kg/m³)"


def test_example_files_and_no_synthetic_generators():
    import dlstudio.data as D
    from dlstudio.data import EXAMPLE_KINDS, example_files
    for kind in EXAMPLE_KINDS:
        fs = example_files(kind)
        assert fs and all((Path(D.__file__).parent / "data" / f).exists() for f in fs), kind
    for name in ("example_arrays", "vibration_arrays", "surface_arrays", "parts_det_arrays", "wear_arrays"):
        assert not hasattr(D, name)                                        # 화면 예시에 합성 데이터 없음

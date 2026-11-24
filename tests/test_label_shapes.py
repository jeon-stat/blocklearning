"""라벨링 도형 작업: 상자·다각형·점을 섞어 저장, 다시 열기, YOLO(탐지·분할)·COCO(bbox·segmentation·keypoints)로 내보내기."""
import json

import pytest

from dlstudio.core import Studio

PIL = pytest.importorskip("PIL.Image")


def _job(tmp_path):
    root = tmp_path / "그림"
    root.mkdir()
    for i in range(3):
        PIL.new("L", (40, 20), 255).save(root / f"g{i}.png")
    s = Studio()
    c, st = s.call("POST", "/api/label/new", {"path": str(root), "classes": ["흠", "구멍"], "file": str(tmp_path / "w.dllabel"),
                                              "kind": "image_folder", "schema": "boxes"})
    assert c == 200, st
    return s


def test_mixed_shapes_save_and_export(tmp_path):
    s = _job(tmp_path)
    shapes = [[0.1, 0.1, 0.2, 0.3, "흠"], ["poly", [[0.5, 0.1], [0.9, 0.1], [0.7, 0.8]], "구멍"], ["pt", 0.25, 0.75, "흠"]]
    c, st = s.call("POST", "/api/label/set_boxes", {"name": "g0.png", "boxes": shapes})
    assert c == 200 and st["counts"]["labeled"] == 1 and {k["name"]: k["count"] for k in st["classes"]} == {"흠": 1, "구멍": 1}
    assert s.call("POST", "/api/label/item", {"name": "g0.png"})[1]["boxes"] == shapes
    bad = [["poly", [[0.1, 0.1], [0.2, 0.2]], "흠"]], [["pt", 1.5, 0.2, "흠"]], [["poly", [[0, 0], [1, 0], [0, 1]], "없는클래스"]]
    for b in bad:
        assert s.call("POST", "/api/label/set_boxes", {"name": "g1.png", "boxes": b})[0] == 400
    s2 = Studio()                                                          # 다시 열어도 그대로 (작업 파일 정리 규칙 통과)
    assert s2.call("POST", "/api/label/open", {"file": str(tmp_path / "w.dllabel")})[0] == 200
    assert s2.label.meta["items"]["g0.png"]["boxes"] == shapes
    s.call("POST", "/api/label/export_yolo", {"folder": str(tmp_path / "yolo")})
    line = (tmp_path / "yolo" / "labels" / "g0.txt").read_text(encoding="utf-8").splitlines()
    assert len(line) == 2 and len(line[0].split()) == 9 and line[1].startswith("1 0.5 0.1 0.9 0.1 0.7 0.8")   # 분할: 상자도 네 꼭짓점, 점은 뺌
    s.call("POST", "/api/label/export_coco", {"file": str(tmp_path / "coco.json")})
    coco = json.loads((tmp_path / "coco.json").read_text(encoding="utf-8"))
    anns = coco["annotations"]
    assert anns[0]["bbox"] == [4.0, 2.0, 8.0, 6.0]
    assert anns[1]["segmentation"] == [[20.0, 2.0, 36.0, 2.0, 28.0, 16.0]] and anns[1]["area"] == 112.0
    assert anns[2]["keypoints"] == [10.0, 15.0, 2] and coco["categories"][0]["keypoints"] == ["흠"]


def test_boxes_only_stays_detection_format(tmp_path):
    s = _job(tmp_path)
    s.call("POST", "/api/label/set_boxes", {"name": "g0.png", "boxes": [[0.1, 0.1, 0.2, 0.3, "흠"]]})
    s.call("POST", "/api/label/export_yolo", {"folder": str(tmp_path / "yolo")})
    assert (tmp_path / "yolo" / "labels" / "g0.txt").read_text(encoding="utf-8").split() == ["0", "0.2", "0.25", "0.2", "0.3"]

"""프로젝트 파일(.dlstudio, zip): 모델 그래프, 데이터 출처·설정, 학습 설정, 결과 요약, 학습된 가중치.
내보내기: torch만으로 읽는 모델 파일(TorchScript .pt)."""
from __future__ import annotations

import io
import json
import warnings
import zipfile
from pathlib import Path
from typing import Optional

FORMAT, VERSION = "dlstudio", 1


def save_project(path, graph: dict, data_source: dict, train_settings: dict, model, results: dict,
                 selection: dict = None, extra_files: Optional[dict] = None, name: str = "") -> None:
    """path: 파일 경로 또는 쓸 수 있는 파일(BytesIO). model: torch 모델, 또는 가중치 dict(이름 → 배열, 웹 버전).
    extra_files: zip에 함께 넣을 {이름: 바이트} (웹 버전의 데이터 사본 data/...). name: 프로젝트 이름."""
    if isinstance(path, (str, Path)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": FORMAT, "version": VERSION, "name": name, "graph": graph, "data_source": data_source,
            "train_settings": train_settings, "results": results, "selection": selection}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(meta, ensure_ascii=False, indent=1))
        for arc, raw in (extra_files or {}).items():
            z.writestr(arc, raw)
        if model is not None:                        # 가중치: PyTorch state_dict 이름 그대로 .npz (torch 없이도 읽음)
            from dlstudio.codegen import save_npz
            buf = io.BytesIO()
            save_npz(model if isinstance(model, dict) else model.state_dict(), buf)
            z.writestr("model.npz", buf.getvalue())


def open_project(path: str) -> dict:
    try:
        with zipfile.ZipFile(Path(path)) as z:
            names = set(z.namelist())
            meta = json.loads(z.read("project.json")) if "project.json" in names else {}
            if meta.get("format") != FORMAT or meta.get("version") != VERSION:
                raise ValueError("DL studio 프로젝트 파일이 아님")
            state: Optional[dict] = None
            if "model.npz" in names:
                import numpy as np
                with np.load(io.BytesIO(z.read("model.npz"))) as npz:
                    state = {k: npz[k] for k in npz.files}
    except zipfile.BadZipFile:
        raise ValueError("DL studio 프로젝트 파일이 아님 (zip이 아님)") from None
    return {"graph": meta["graph"], "data_source": meta["data_source"], "train_settings": meta["train_settings"],
            "name": meta.get("name", ""),
            "results": meta.get("results", {}), "selection": meta.get("selection"), "state_dict": state}


def _export_module(model, data):
    """원래 값 한 행(CSV 그대로) → 정규화 → 모델 입력 모양 → 모델. studio 없이 torch만으로 돌아가게 TorchScript로 굳힐 모듈."""
    import numpy as np
    import torch
    from torch import nn

    class StudioExport(nn.Module):
        def __init__(self, net, mean, std, kind: str, image):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(np.asarray(std), dtype=torch.float32))
            self.kind, self.image = kind, image

        def forward(self, x):                                  # x: (샘플 수, 특징 수) 원래 값
            z = (x - self.mean) / self.std
            if self.kind == "image":
                return self.net(z.reshape(-1, 1, self.image[0], self.image[1]))
            return self.net(z.unsqueeze(1) if self.kind == "cnn" else z.unsqueeze(2))

    return StudioExport(model, data.mean, data.std, model.kind, data.image).eval()


def export_torch(model, data, path: str) -> dict:
    """학습된 모델 → 평문 모델 파일(TorchScript .pt). 받는 쪽은 DL studio 없이 torch.jit.load로 읽어
    원래 값 행(샘플 수, 특징 수)을 넣으면 (샘플 수, 클래스 수) 점수 또는 (샘플 수, 1) 예측값이 나온다.
    과제·클래스 이름·입력 설명은 파일 안의 studio.json (torch.jit.load(..., _extra_files={"studio.json": ""}))."""
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m = _export_module(model.eval(), data)
    example = torch.as_tensor(data.raw(data.X_train[:2]), dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(m, example, check_trace=False)
    cls = data.task == "classification"
    info = {"format": "dlstudio-model", "task": data.task, "classes": [str(c) for c in data.classes] if cls else [],
            "n_features": int(data.X_train.shape[1]), "features": list(data.features or []),
            "image": list(data.image) if data.image else None,
            "input": ("행 하나 = 그림 하나를 흑백 0–1 값으로 펼친 픽셀 (높이×너비)" if data.image
                      else "행 하나 = 샘플 하나. 학습 CSV와 같은 열 순서의 원래 값 (정답 열 빼고)"),
            "output": "클래스마다 점수 (가장 큰 것이 예측, 순서는 classes)" if cls else "예측값 (원래 단위)"}
    with warnings.catch_warnings():                          # 새 torch는 TorchScript를 낡았다고 알리지만 가장 널리 읽힘
        warnings.simplefilter("ignore")
        torch.jit.save(traced, str(path), _extra_files={"studio.json": json.dumps(info, ensure_ascii=False)})
    return info

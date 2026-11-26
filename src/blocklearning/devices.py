"""학습 장치: NVIDIA GPU 목록과 지금 쓰이는 메모리 (공용 서버에서 남이 쓰는 GPU를 피해 고르게).

번호는 torch(CUDA) 번호. CUDA_VISIBLE_DEVICES로 일부만 보이게 해도 맞도록 nvidia-smi와는 UUID로 맞춘다.
GPU 정보만 읽고 GPU에 아무것도 올리지 않는다 (남의 GPU에 메모리를 잡지 않게).
"""
from __future__ import annotations

import subprocess


def gpus() -> list:
    """[{index, name, total_gb, used_gb}]. used_gb는 nvidia-smi로 읽은 사용 중 메모리 (못 읽으면 None)."""
    import torch
    if not torch.cuda.is_available():
        return []
    used = smi_used()
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        u = used.get(str(getattr(p, "uuid", "") or ""))
        out.append({"index": i, "name": p.name, "total_gb": round(p.total_memory / 1024 ** 3, 1),
                    "used_gb": None if u is None else round(u / 1024, 1)})
    return out


def smi_used(text: str = None) -> dict:
    """nvidia-smi → {UUID(앞의 GPU- 뺌): 사용 중 메모리 MiB}. nvidia-smi가 없거나 실패하면 {}."""
    if text is None:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return {}
        if r.returncode:
            return {}
        text = r.stdout
    d = {}
    for line in text.splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            d[parts[0].removeprefix("GPU-")] = float(parts[1])
        except ValueError:
            pass
    return d


def least_used(gs: list) -> int:
    """'자동'이 고를 GPU: 메모리를 가장 적게 쓰는 것 (같으면 번호가 작은 것). 사용량을 모르면 0번."""
    known = [g for g in gs if g.get("used_gb") is not None]
    return min(known, key=lambda g: (g["used_gb"], g["index"]))["index"] if known else 0

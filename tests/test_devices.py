"""학습 장치: GPU 목록·사용량 읽기, '자동'이 덜 쓰는 GPU를 고르는지, 없는 GPU 번호는 알려 주는지."""
import pytest

from blocklearning import devices
from blocklearning.jobs import _device

SMI = """GPU-6e4a4e6a-69dc-4d12-f7a2-899dbed48480, 3548
GPU-1111aaaa-0000-0000-0000-000000000000, 1
GPU-2c7f81ae-bdce-498a-5d6f-925eeb5e0cf9, 1
[N/A], x
"""


def test_smi_used_parses_uuid_and_memory():
    d = devices.smi_used(SMI)
    assert d == {"6e4a4e6a-69dc-4d12-f7a2-899dbed48480": 3548.0, "1111aaaa-0000-0000-0000-000000000000": 1.0,
                 "2c7f81ae-bdce-498a-5d6f-925eeb5e0cf9": 1.0}


def test_least_used_skips_busy_gpu():
    gs = [{"index": 0, "used_gb": 3.5}, {"index": 1, "used_gb": 0.0}, {"index": 2, "used_gb": 0.0}]
    assert devices.least_used(gs) == 1                       # 0번은 남이 쓰는 중 → 비어 있는 것 중 번호가 작은 것
    assert devices.least_used([{"index": 0, "used_gb": None}, {"index": 1, "used_gb": None}]) == 0
    assert devices.least_used([]) == 0


def test_device_names():
    import torch
    assert _device("cpu") is None and _device("") is None
    if not torch.cuda.is_available():
        assert _device("auto") is None
    with pytest.raises(ValueError, match="GPU 99번이 없음"):
        _device("cuda:99")


def test_cpu_torch_hint_only_when_nvidia_gpu_but_cpu_torch(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devices, "smi_used", lambda text=None: {"6e4a": 1.0})      # nvidia-smi는 GPU를 봄
    h = devices.cpu_torch_hint()
    assert "CPU용" in h and "pip uninstall -y torch" in h and "download.pytorch.org/whl/cu" in h
    monkeypatch.setattr(devices, "smi_used", lambda text=None: {})                # NVIDIA GPU 없음
    assert devices.cpu_torch_hint() == ""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)                  # 이미 GPU용
    monkeypatch.setattr(devices, "smi_used", lambda text=None: {"6e4a": 1.0})
    assert devices.cpu_torch_hint() == ""

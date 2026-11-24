"""과제 고르기: 큰 틀(분류·회귀 …) → 세부 과제(시계열·표 …) → 대표 모델(템플릿).

딥러닝을 모르는 사람이 '무엇을 하고 싶은지'부터 고르고, 그 과제에 맞는 대표 모델을 받아 블록으로 고치게 한다.
템플릿 블록은 ReLU 활성화를 쓴다. 모든 템플릿은 자기 예시 데이터에서 검사·학습된다 (시험으로 확인).
"""


def _conv(ch, k, d=1):
    b = {"type": "conv", "channels": ch, "kernel": k, "stride": 1}
    if d > 1:
        b["dilation"] = d
    return b


P2 = {"type": "act", "kind": "relu"}          # 활성화 (이름은 예전 그대로 P2)
BN = {"type": "bn"}
GAP = {"type": "global_avg"}
OUT = {"type": "output"}


def _res(*body):
    return {"type": "residual", "body": list(body)}


def _dense(n):
    return {"type": "dense", "units": n}


def _drop(r):
    return {"type": "dropout", "rate": r}


T_CNN = {"id": "cnn", "name": "1D CNN",
         "desc": "짧은 구간의 모양(파형)을 찾는 필터를 겹겹이 쌓은 기본 모델",
         "when": "처음 시작할 때. 대부분의 신호에서 무난함",
         "blocks": [_conv(16, 5), P2, {"type": "repeat", "times": 2, "body": [_res(_conv(16, 3), P2)]}, GAP, OUT]}

T_RESNET = {"id": "resnet1d", "name": "ResNet1D",
            "desc": "Residual block을 여러 번 쌓은 깊은 CNN. 시계열 분류에서 자주 쓰는 기준 모델",
            "when": "1D CNN보다 더 복잡한 패턴이 있을 때",
            "blocks": [_conv(32, 7), BN, P2,
                       {"type": "repeat", "times": 3, "body": [_res(_conv(32, 3), BN, P2, _conv(32, 3), BN)]},
                       P2, GAP, OUT]}

T_INCEPTION = {"id": "inception", "name": "InceptionTime",
               "desc": "짧은 필터와 긴 필터를 나란히 두고 결과를 이어 붙임 (여러 길이의 패턴을 한 번에)",
               "when": "짧은 패턴과 긴 패턴이 섞여 있을 때",
               "blocks": [_conv(16, 1), {"type": "branch", "branches": [[_conv(8, 3)], [_conv(8, 9)], []]},
                          BN, P2, GAP, OUT]}

T_TCN = {"id": "tcn", "name": "TCN",
         "desc": "칸을 띄워 보는 Conv1D(dilation 2·4·8)로 먼 과거까지 보고 마지막 시점으로 판단",
         "when": "긴 시계열에서 오래전 값이 중요할 때",
         "blocks": [_conv(16, 3), P2, _res(_conv(16, 3, 2), P2), _res(_conv(16, 3, 4), P2),
                    _res(_conv(16, 3, 8), P2), {"type": "last_step"}, OUT]}

T_LSTM = {"id": "lstm", "name": "LSTM",
          "desc": "값을 하나씩 순서대로 읽으며 기억을 갱신하는 순환 모델",
          "when": "순서·흐름 자체가 중요할 때. CNN보다 학습이 느림",
          "blocks": [{"type": "lstm", "hidden": 16, "layers": 1}, OUT]}

T_MLP = {"id": "mlp", "name": "MLP",
         "desc": "Dense 층 몇 개로 된 가장 기본 모델. 표(열마다 다른 값) 데이터에 씀",
         "when": "처음 시작할 때. 열이 수십 개 이하일 때",
         "blocks": [_dense(32), P2, _dense(16), P2, OUT]}

T_DEEP_MLP = {"id": "deep_mlp", "name": "깊은 MLP",
              "desc": "Dense 층을 더 쌓고 BatchNorm·Dropout으로 학습을 안정시킨 모델",
              "when": "데이터가 많고 MLP로 부족할 때",
              "blocks": [_dense(64), BN, P2, _drop(0.2), _dense(32), BN, P2, _drop(0.2), _dense(16), P2, OUT]}


def _c2(ch, k=3):
    return {"type": "conv2d", "channels": ch, "kernel": k, "stride": 1}


P2D = {"type": "pool2d", "size": 2}
GAP2 = {"type": "global_avg2d"}

T_CNN2D = {"id": "cnn2d", "name": "작은 CNN",
           "desc": "3×3 필터로 획·모서리를 찾고, 줄여 가며 모양을 모으는 기본 이미지 모델",
           "when": "처음 시작할 때. 작은 흑백 이미지에서 무난함",
           "blocks": [_c2(8), P2, P2D, _c2(16), P2, GAP2, OUT]}

T_LENET = {"id": "lenet", "name": "LeNet식",
           "desc": "Conv → Pool을 두 번 하고 Dense로 판단하는 고전 손글씨 모델 (LeNet-5를 작게)",
           "when": "위치가 중요한 그림 (어디에 획이 있는지)",
           "blocks": [_c2(6), P2, P2D, _c2(16), P2, P2D, {"type": "flatten"}, _dense(32), P2, OUT]}

T_RESNET2D = {"id": "resnet2d", "name": "작은 ResNet",
              "desc": "Residual block을 쌓은 깊은 이미지 모델. 깊어도 학습이 잘 됨",
              "when": "작은 CNN으로 부족할 때",
              "blocks": [_c2(16), BN, P2, {"type": "repeat", "times": 2, "body": [_res(_c2(16), BN, P2)]}, GAP2, OUT]}


def _soon(id_, name, input_, examples_text, desc):
    return {"id": id_, "name": name, "input": input_, "available": False, "desc": desc,
            "examples_text": examples_text}


CATALOG = [
    {"id": "classification", "name": "분류", "icon": "🏷️",
     "desc": "데이터가 어느 종류(클래스)에 속하는지 맞힙니다",
     "subtasks": [
         {"id": "cls_ts", "name": "시계열 분류", "input": "timeseries", "available": True, "task": "classification",
          "desc": "시간에 따라 측정한 신호 한 구간을 보고 종류를 맞힘",
          "examples_text": "심전도 정상·부정맥, 설비 진동 정상·결함, 센서 파형으로 공정 상태 판별",
          "example_data": [["ecg", "심전도 정상·이상 (의료)"], ["vibration", "설비 진동 정상·결함 (제조)"]],
          "templates": [T_CNN, T_RESNET, T_INCEPTION, T_TCN, T_LSTM]},
         {"id": "cls_tab", "name": "표 데이터 분류", "input": "tabular", "available": True, "task": "classification",
          "desc": "한 행(여러 측정값)을 보고 종류를 맞힘",
          "examples_text": "공정 조건으로 불량 판정, 검사 수치로 질환 여부 판별",
          "example_data": [["process_cls", "공정 조건 → 양품·불량 (제조)"]],
          "templates": [T_MLP, T_DEEP_MLP]},
         {"id": "cls_img", "name": "이미지 분류", "input": "image", "available": True, "task": "classification",
          "desc": "흑백 그림 한 장을 보고 종류를 맞힘",
          "examples_text": "손글씨 숫자 읽기, 제품 사진으로 양품·불량 판정, 현미경 사진 분류",
          "example_data": [["surface", "금속 표면 정상·스크래치·얼룩·구멍 (제조, 32×32)"],
                           ["pneumonia", "흉부 X-ray 정상·폐렴 (의료, 28×28)"], ["digits", "손글씨 숫자 0–9 (8×8)"]],
          "templates": [T_CNN2D, T_LENET, T_RESNET2D]},
     ]},
    {"id": "regression", "name": "회귀", "icon": "📈",
     "desc": "숫자 값(수율, 수명, 수치)을 예측합니다",
     "subtasks": [
         {"id": "reg_ts", "name": "시계열 회귀", "input": "timeseries", "available": True, "task": "regression",
          "desc": "신호 한 구간을 보고 숫자 하나를 예측",
          "examples_text": "진동 신호로 남은 수명(RUL) 예측, 생체 신호로 수치 추정",
          "example_data": [["wear_rul", "진동 → 남은 수명 (제조)"]],
          "templates": [T_CNN, T_TCN, T_LSTM]},
         {"id": "reg_tab", "name": "표 데이터 회귀", "input": "tabular", "available": True, "task": "regression",
          "desc": "한 행(여러 측정값)을 보고 숫자 하나를 예측",
          "examples_text": "공정 조건으로 수율 예측, 검사 수치로 위험 점수 추정",
          "example_data": [["process_reg", "공정 조건 → 수율 (제조)"]],
          "templates": [T_MLP]},
         _soon("reg_multi", "여러 값 동시 예측", "tabular", "공정 조건으로 두께·강도·수율을 한 번에 예측",
               "숫자 여러 개를 한 모델로 예측"),
     ]},
    {"id": "anomaly", "name": "이상 탐지", "icon": "🚨",
     "desc": "정상 데이터만으로 학습해, 평소와 다른 데이터를 찾아냅니다",
     "subtasks": [
         _soon("ae_ts", "시계열 이상 탐지 (오토인코더)", "timeseries",
               "정상 진동만 학습해 고장 전조 찾기, 정상 심전도와 다른 박동 찾기",
               "정상 신호를 복원하도록 학습하고, 복원이 잘 안 되는 구간을 이상으로 봄"),
         _soon("ae_tab", "표 데이터 이상 탐지", "tabular", "평소와 다른 공정 조건 조합 찾기",
               "정상 행을 복원하도록 학습하고, 복원 오차가 큰 행을 이상으로 봄"),
     ]},
    {"id": "forecast", "name": "예측 (미래 값)", "icon": "🔮",
     "desc": "지금까지의 흐름으로 앞으로의 값을 예측합니다",
     "subtasks": [
         _soon("fc_ts", "시계열 예측", "timeseries", "다음 1시간 설비 온도, 다음 날 생산량",
               "과거 구간을 보고 다음 몇 시점의 값을 예측"),
     ]},
    {"id": "segmentation", "name": "구간 분할", "icon": "✂️",
     "desc": "긴 신호의 시점마다 어떤 상태인지 표시합니다",
     "subtasks": [
         _soon("seg_ts", "시계열 구간 분할", "timeseries", "심전도에서 박동 구간 표시, 공정 단계별 구간 나누기",
               "시점 하나하나에 종류를 붙임"),
     ]},
]


DEFAULT_TEMPLATES = [T_CNN, T_LSTM, T_INCEPTION, T_MLP]      # 과제를 고르지 않았을 때 ③ 모델의 예시 버튼


def _view(t: dict) -> dict:
    """화면에 보내는 템플릿: 블록 목록 대신 그래프."""
    from dlstudio.graph import from_blocks
    return dict({k: v for k, v in t.items() if k != "blocks"}, graph=from_blocks(t["blocks"]))


def catalog_view() -> dict:
    """/api/tasks 응답: 과제 목록(템플릿은 그래프로)과 기본 예시."""
    fams = [dict(f, subtasks=[dict(s, templates=[_view(t) for t in s["templates"]]) if "templates" in s else dict(s)
                              for s in f["subtasks"]]) for f in CATALOG]
    return {"families": fams, "defaults": [_view(t) for t in DEFAULT_TEMPLATES]}


def find(family, subtask, template):
    """(큰 틀, 세부 과제, 템플릿)을 돌려준다. 없거나 준비 중이면 KeyError."""
    fam = next((f for f in CATALOG if f["id"] == family), None)
    sub = fam and next((s for s in fam["subtasks"] if s["id"] == subtask and s["available"]), None)
    t = sub and next((t for t in sub["templates"] if t["id"] == template), None)
    if t is None:
        raise KeyError(f"{family}/{subtask}/{template}")
    return fam, sub, t

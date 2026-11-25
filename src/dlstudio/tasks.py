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
         "desc": "신호에서 짧은 모양(봉우리·떨림 같은 파형)을 찾는 필터를 몇 겹 쌓은 기본 모델",
         "when": "처음 시작할 때. 대부분의 신호에서 무난하고 빠름",
         "blocks": [_conv(16, 5), P2, {"type": "repeat", "times": 2, "body": [_res(_conv(16, 3), P2)]}, GAP, OUT]}

T_RESNET = {"id": "resnet1d", "name": "ResNet1D",
            "desc": "1D CNN을 더 깊게 쌓고, 입력을 출력에 바로 더해 주는 지름길(Residual)을 둬서 깊어도 잘 배우는 모델",
            "when": "1D CNN으로 부족할 때. 배우는 값이 10배쯤 많아 학습이 더 오래 걸림",
            "blocks": [_conv(32, 7), BN, P2,
                       {"type": "repeat", "times": 3, "body": [_res(_conv(32, 3), BN, P2, _conv(32, 3), BN)]},
                       P2, GAP, OUT]}

T_INCEPTION = {"id": "inception", "name": "InceptionTime",
               "desc": "짧은 필터와 긴 필터를 나란히 써서, 짧은 모양과 긴 모양을 한 번에 찾아 이어 붙이는 모델",
               "when": "짧은 패턴과 긴 패턴이 섞여 있을 때. 크기는 1D CNN과 비슷",
               "blocks": [_conv(16, 1), {"type": "branch", "branches": [[_conv(8, 3)], [_conv(8, 9)], []]},
                          BN, P2, GAP, OUT]}

T_TCN = {"id": "tcn", "name": "TCN",
         "desc": "칸을 띄엄띄엄 건너뛰며 보는 필터(dilation 2·4·8)를 겹쳐 먼 과거까지 보고, 마지막 시점에서 판단하는 모델",
         "when": "긴 신호에서 오래전 값이 중요할 때. 크기는 1D CNN과 비슷",
         "blocks": [_conv(16, 3), P2, _res(_conv(16, 3, 2), P2), _res(_conv(16, 3, 4), P2),
                    _res(_conv(16, 3, 8), P2), {"type": "last_step"}, OUT]}

T_LSTM = {"id": "lstm", "name": "LSTM",
          "desc": "값을 앞에서부터 하나씩 읽으며 기억을 고쳐 나가는 순환 모델 (순서를 그대로 따라감)",
          "when": "순서·흐름 자체가 중요할 때. 하나씩 차례로 계산해 CNN보다 학습이 느림",
          "blocks": [{"type": "lstm", "hidden": 16, "layers": 1}, OUT]}

T_MLP = {"id": "mlp", "name": "MLP",
         "desc": "모든 열의 값을 섞어 판단하는 층(Dense)을 몇 개 쌓은 가장 기본 모델",
         "when": "처음 시작할 때. 열이 수십 개 이하일 때",
         "blocks": [_dense(32), P2, _dense(16), P2, OUT]}

T_DEEP_MLP = {"id": "deep_mlp", "name": "깊은 MLP",
              "desc": "MLP를 더 깊게 쌓고, 값 크기를 고르게 맞추는 BatchNorm과 외우기(과적합)를 막는 Dropout을 더한 모델",
              "when": "데이터가 많고 MLP로 부족할 때. 배우는 값이 4배쯤 많음",
              "blocks": [_dense(64), BN, P2, _drop(0.2), _dense(32), BN, P2, _drop(0.2), _dense(16), P2, OUT]}


def _c2(ch, k=3):
    return {"type": "conv2d", "channels": ch, "kernel": k, "stride": 1}


P2D = {"type": "pool2d", "size": 2}
GAP2 = {"type": "global_avg2d"}

T_CNN2D = {"id": "cnn2d", "name": "작은 CNN",
           "desc": "3×3 필터로 획·모서리를 찾고, 그림을 줄여 가며 모양을 모으는 기본 이미지 모델",
           "when": "처음 시작할 때. 작은 흑백 그림에서 무난하고 빠름",
           "blocks": [_c2(8), P2, P2D, _c2(16), P2, GAP2, OUT]}

T_LENET = {"id": "lenet", "name": "LeNet식",
           "desc": "필터로 찾고 그림을 줄이기를 두 번 한 뒤 Dense로 판단하는 고전 손글씨 모델 (LeNet-5를 작게)",
           "when": "어디에 획이 있는지(위치)가 중요한 그림. Dense 때문에 배우는 값이 많음",
           "blocks": [_c2(6), P2, P2D, _c2(16), P2, P2D, {"type": "flatten"}, _dense(32), P2, OUT]}

T_RESNET2D = {"id": "resnet2d", "name": "작은 ResNet",
              "desc": "지름길(Residual)을 둔 깊은 이미지 모델. 깊어도 잘 배움",
              "when": "작은 CNN으로 부족할 때. 학습이 더 오래 걸림",
              "blocks": [_c2(16), BN, P2, {"type": "repeat", "times": 2, "body": [_res(_c2(16), BN, P2)]}, GAP2, OUT]}

T_TRANSFER = {"id": "mnv3_transfer", "name": "전이 학습 (MobileNetV3)",
              "desc": "Google이 ImageNet 사진으로 미리 학습한 MobileNetV3-Small로 특징 1024개를 뽑고, 끝 층만 학습",
              "when": "그림이 적거나 학습이 오래 걸릴 때. 아주 작은 그림(28×28 등)에서는 효과가 적을 수 있음",
              "blocks": [{"type": "mnv3s", "size": 128}, {"type": "dropout", "rate": 0.2}, OUT]}


T_DET_SMALL = {"id": "det_small", "name": "작은 탐지 모델",
               "desc": "3×3 필터로 모양을 찾으며 그림을 세 번 줄여 8칸 격자로 만든 뒤, 칸마다 '사물이 있나·상자·종류'를 맞히는 "
                       "YOLO 첫 버전식 모델",
               "when": "처음 시작할 때. 배경이 단순한 작은 그림(부품·결함 위치)에서 빠르게 학습",
               "blocks": [_c2(16), P2, P2D, _c2(32), P2, P2D, _c2(32), P2, P2D, _c2(32), P2, OUT]}

T_DET_RES = {"id": "det_res", "name": "깊은 탐지 모델",
             "desc": "작은 탐지 모델에 BatchNorm과 지름길(Residual)을 더해 더 깊게 쌓아, 더 크고 복잡한 모양도 보는 모델",
             "when": "작은 탐지 모델로 부족할 때. 학습이 2–3배 오래 걸림",
             "blocks": [_c2(16), BN, P2, P2D, _c2(32), BN, P2, P2D, _res(_c2(32), BN, P2), _c2(64), BN, P2, P2D,
                        _res(_c2(64), BN, P2), OUT]}


# Transformer 계열: 위치(토큰)마다 값 c개인 줄 (c, T)에 Encoder 블록을 쌓음.
# Encoder 블록 = LayerNorm → Self-attention + 지름길 → LayerNorm → FFN(4배·GELU, 1×1) + 지름길 (Pre-LN, ViT·GPT식)
LN = {"type": "layernorm"}
GELU = {"type": "act", "kind": "gelu"}


def _encoder(c, heads):
    return [_res(LN, {"type": "mha", "heads": heads}), _res(LN, _conv(c * 2, 1), GELU, _conv(c, 1))]


T_TRANSFORMER = {"id": "transformer", "name": "Transformer",
                 "desc": "필터로 짧은 모양을 찾아 줄인 뒤, 위치마다 '다른 어느 위치를 볼지'를 스스로 정하는 Self-attention을 두 번 "
                         "쌓은 모델. 학습 뒤 샘플 따라가기에서 어디를 봤는지(어텐션 열지도)가 보임",
                 "when": "먼 위치끼리의 관계가 중요할 때. 데이터가 적으면 CNN보다 덜 배울 수 있음",
                 "blocks": [_conv(32, 5), P2, {"type": "max_pool", "size": 2}, {"type": "pos_enc", "mode": "sin"},
                            *_encoder(32, 4), *_encoder(32, 4), LN, GAP, OUT]}

T_FT_TRANSFORMER = {"id": "ft_transformer", "name": "FT-Transformer식",
                    "desc": "열 하나를 토큰 하나로 보고(값 → 16개 값, 열마다 다른 위치 값을 더함), 열끼리 서로 보며 섞는 "
                            "Self-attention을 쌓은 표 데이터 Transformer. 어느 열이 어느 열을 봤는지가 보임",
                    "when": "열끼리 얽힌 관계가 중요할 때. MLP보다 느리고, 행이 적으면 MLP가 나을 수 있음",
                    "blocks": [{"type": "reshape", "channels": 1}, _conv(16, 1), {"type": "pos_enc", "mode": "learned"},
                               *_encoder(16, 2), *_encoder(16, 2), LN, GAP, OUT]}

T_VIT = {"id": "vit", "name": "ViT (Vision Transformer)",
         "desc": "그림을 4×4 조각으로 잘라 조각마다 토큰으로 만들고, 맨 앞 요약 자리(CLS)와 함께 Self-attention을 쌓은 뒤 "
                 "CLS 자리로 판단하는 모델. 요약 자리가 그림의 어느 조각을 봤는지가 보임",
         "when": "Transformer를 그림에 써 보고 싶을 때. 작은 데이터에서는 CNN보다 정확도가 낮고 학습이 느림",
         "blocks": [{"type": "patch_embed", "patch": 4, "dim": 32}, {"type": "cls_token"}, {"type": "pos_enc", "mode": "learned"},
                    *_encoder(32, 4), *_encoder(32, 4), LN, {"type": "first_step"}, OUT]}

T_HYBRID = {"id": "cnn_transformer", "name": "CNN + Transformer",
            "desc": "CNN으로 그림을 줄여 특징 지도를 만든 뒤, 지도의 칸들을 한 줄(토큰)로 펴서 Self-attention으로 칸끼리 "
                    "보게 하는 모델 (DETR·하이브리드 ViT식). 어느 칸을 봤는지가 보임",
            "when": "ViT가 잘 안 배울 때. CNN이 먼저 모양을 찾아 줘서 작은 데이터에서도 무난",
            "blocks": [_c2(16), P2, P2D, _c2(32), P2, P2D, {"type": "to_seq"}, {"type": "pos_enc", "mode": "learned"},
                       *_encoder(32, 4), LN, GAP, OUT]}


# 시계열 예측: 출력은 '다음 H개' (출력층 Dense가 H개를 냄)
T_FC_LINEAR = {"id": "fc_linear", "name": "선형 (DLinear식)",
               "desc": "지난 값들에 가중치를 곱해 더하는 것만으로 다음 값들을 내는 가장 단순한 모델. 단순해도 주기가 뚜렷한 "
                       "신호에서 깊은 모델만큼 잘 맞히는 경우가 많아 기준으로 좋음",
               "when": "처음 시작할 때. 아주 빠름",
               "blocks": [OUT]}
T_FC_CNN = {"id": "fc_cnn", "name": "1D CNN",
            "desc": "필터로 신호의 짧은 모양을 찾은 뒤, 위치마다의 결과를 그대로 펼쳐 다음 값들을 냄 (어느 때의 모양인지가 남음)",
            "when": "모양(패턴)이 반복되는 신호",
            "blocks": [_conv(16, 5), P2, _conv(16, 5), P2, {"type": "max_pool", "size": 2}, {"type": "flatten"}, OUT]}
T_FC_MLP = {"id": "fc_mlp", "name": "MLP",
            "desc": "지난 값들을 Dense 층 몇 개로 섞어 다음 값들을 내는 모델. 선형보다 굽은 관계도 배움",
            "when": "선형으로 부족할 때",
            "blocks": [{"type": "flatten"}, _dense(64), P2, _dense(64), P2, OUT]}

# 이상 탐지 (오토인코더): 좁은 가운데(병목)로 줄였다가 출력층에서 입력 값 전부를 되살림
T_AE_DENSE = {"id": "ae_dense", "name": "Dense 오토인코더",
              "desc": "신호를 Dense 층으로 16개 값까지 줄였다가(요약) 다시 원래 길이로 되살리는 모델. 정상의 모양만 배워서, "
                      "평소와 다른 신호는 잘 못 되살림 → 되살린 오차가 이상 점수",
              "when": "처음 시작할 때. 빠름",
              "blocks": [{"type": "flatten"}, _dense(64), P2, _dense(16), P2, _dense(64), P2, OUT]}
T_AE_CONV = {"id": "ae_conv", "name": "Conv 오토인코더",
             "desc": "필터로 짧은 모양을 찾으며 줄인 뒤 16개 값으로 요약하고, 출력층에서 신호 전체를 되살리는 모델. "
                     "충격·떨림 같은 짧은 모양의 이상에 강함",
             "when": "Dense 오토인코더로 부족할 때",
             "blocks": [_conv(16, 7), P2, {"type": "max_pool", "size": 2}, _conv(8, 5), P2, {"type": "max_pool", "size": 2},
                        {"type": "flatten"}, _dense(16), P2, OUT]}
T_AE_TAB = {"id": "ae_tab_dense", "name": "오토인코더",
            "desc": "측정 항목들을 4개 값까지 줄였다가 다시 모든 항목을 되살리는 모델. 정상의 항목끼리 관계만 배워서, "
                    "평소와 다른 조합은 잘 못 되살림",
            "when": "열이 수십 개 이하인 표 데이터",
            "blocks": [_dense(16), P2, _dense(4), P2, _dense(16), P2, OUT]}


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
          "example_data": [["bearing", "베어링 진동 — 정상·내륜·외륜·볼·복합 결함 (제조)"], ["ecg", "심전도 박동 정상·이상 (의료)"]],
          "templates": [T_CNN, T_RESNET, T_INCEPTION, T_TCN, T_LSTM, T_TRANSFORMER]},
         {"id": "cls_tab", "name": "표 데이터 분류", "input": "tabular", "available": True, "task": "classification",
          "desc": "한 행(여러 측정값)을 보고 종류를 맞힘",
          "examples_text": "공정 조건으로 불량 판정, 검사 수치로 질환 여부 판별",
          "example_data": [["steel_plates", "강판 결함 7종 — 결함 영역의 모양·밝기 27가지 (제조)"]],
          "templates": [T_MLP, T_DEEP_MLP, T_FT_TRANSFORMER]},
         {"id": "cls_img", "name": "이미지 분류", "input": "image", "available": True, "task": "classification",
          "desc": "흑백 그림 한 장을 보고 종류를 맞힘",
          "examples_text": "손글씨 숫자 읽기, 제품 사진으로 양품·불량 판정, 현미경 사진 분류",
          "example_data": [["steelblast", "강판 표면 양호·불량 (제조, 64×64)"],
                           ["pneumonia", "흉부 X-ray 정상·폐렴 (의료, 28×28)"], ["digits", "손글씨 숫자 0–9 (32×32)"]],
          "templates": [T_CNN2D, T_LENET, T_RESNET2D, T_TRANSFER, T_HYBRID, T_VIT]},
     ]},
    {"id": "regression", "name": "회귀", "icon": "📈",
     "desc": "숫자 값(수율, 수명, 수치)을 예측합니다",
     "subtasks": [
         {"id": "reg_ts", "name": "시계열 회귀", "input": "timeseries", "available": True, "task": "regression",
          "desc": "신호 한 구간을 보고 숫자 하나를 예측",
          "examples_text": "진동 신호로 남은 수명(RUL) 예측, 생체 신호로 수치 추정",
          "example_data": [["hydraulic_valve", "유압 압력 신호 → 밸브 상태 % (제조)"]],
          "templates": [T_RESNET, T_CNN, T_TCN, T_LSTM, T_TRANSFORMER]},
         {"id": "reg_tab", "name": "표 데이터 회귀", "input": "tabular", "available": True, "task": "regression",
          "desc": "한 행(여러 측정값)을 보고 숫자 하나를 예측",
          "examples_text": "공정 조건으로 수율 예측, 검사 수치로 위험 점수 추정",
          "example_data": [["concrete", "콘크리트 배합 → 압축강도 MPa (제조)"]],
          "templates": [T_MLP, T_FT_TRANSFORMER]},
         _soon("reg_multi", "여러 값 동시 예측", "tabular", "공정 조건으로 두께·강도·수율을 한 번에 예측",
               "숫자 여러 개를 한 모델로 예측"),
     ]},
    {"id": "detection", "name": "객체 탐지", "icon": "🎯",
     "desc": "그림 속 사물이 어디에 있는지 상자로 찾고 종류를 맞힙니다",
     "subtasks": [
         {"id": "det_img", "name": "이미지 객체 탐지", "input": "image", "available": True, "task": "detection",
          "desc": "그림 한 장에서 사물마다 위치(상자)와 종류를 찾음. 정답은 라벨링에서 상자·다각형·원으로 그림",
          "examples_text": "부품 위치 찾기·개수 세기, 표면 결함 위치 표시, 현미경 사진에서 세포 찾기",
          "example_data": [["pcb_det", "PCB 결함 위치 찾기 — 6종 (제조, 64×64)"]],
          "templates": [T_DET_SMALL, T_DET_RES]},
     ]},
    {"id": "anomaly", "name": "이상 탐지", "icon": "🚨",
     "desc": "정상 데이터만으로 학습해, 평소와 다른 데이터를 찾아냅니다",
     "subtasks": [
         {"id": "ae_ts", "name": "시계열 이상 탐지 (오토인코더)", "input": "timeseries", "available": True, "task": "anomaly",
          "desc": "정상 신호만 되살리도록 학습하고, 잘 못 되살리는 신호를 이상으로 봄. 이상 샘플이 적거나 없어도 됨",
          "examples_text": "정상 진동만 학습해 고장 전조 찾기, 정상 심전도와 다른 박동 찾기",
          "example_data": [["bearing_ae", "베어링 진동 — 정상만 학습해 결함 찾기 (제조)"]],
          "templates": [T_AE_DENSE, T_AE_CONV]},
         {"id": "ae_tab", "name": "표 데이터 이상 탐지", "input": "tabular", "available": True, "task": "anomaly",
          "desc": "정상 행만 되살리도록 학습하고, 되살린 오차가 큰 행(평소와 다른 측정값 조합)을 이상으로 봄",
          "examples_text": "평소와 다른 공정 조건 조합 찾기, 센서 값 이상 조합 찾기",
          "example_data": [["hydraulic_ae", "유압 시스템 — 정상만 학습해 냉각기·밸브·펌프 고장 찾기 (제조)"]],
          "templates": [T_AE_TAB]},
     ]},
    {"id": "forecast", "name": "예측 (미래 값)", "icon": "🔮",
     "desc": "지금까지의 흐름으로 앞으로의 값을 예측합니다",
     "subtasks": [
         {"id": "fc_ts", "name": "시계열 예측", "input": "timeseries", "available": True, "task": "forecast",
          "desc": "긴 신호 하나의 지난 구간을 보고 다음 몇 시점의 값을 예측. 시험은 신호의 뒤쪽 구간 (미래를 미리 보지 않게)",
          "examples_text": "다음 날 공장 전력 사용량, 다음 1시간 설비 온도, 다음 주 생산량",
          "example_data": [["steel_power", "철강 공장 전력 사용량 — 지난 4일로 다음 하루 (제조, 1시간마다)"]],
          "templates": [T_FC_LINEAR, T_FC_MLP, T_FC_CNN, T_LSTM, T_TRANSFORMER]},
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

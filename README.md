# DL studio

파이썬·딥러닝 코드를 몰라도 브라우저에서 **층(블록)을 선으로 이어** 모델을 만들고, 학습하고, 학습한 모델을 파일로 내보내 쓰는 도구입니다.
제조(센서·설비)·의료(심전도 등) 시계열·표 데이터와 흑백 이미지를 대상으로 합니다.
모든 계산은 내 PC 안에서 하고, 화면은 `127.0.0.1`에만 열립니다.

## 실행

```bash
pip install -e .            # 처음 한 번 (torch가 이미 있으면 그대로 씀). 이미지 폴더를 쓰려면 pip install -e ".[image]"
python -m dlstudio          # 또는 dlstudio
```

브라우저에 `http://127.0.0.1:8765/`가 열립니다. `--port 8800`으로 포트를, `--no-browser`로 브라우저 자동 열기를 바꿀 수 있습니다.
애플리케이션 제어 정책이 실행 파일(`dlstudio.exe`)을 막는 PC에서는 `python -m dlstudio`를 쓰세요.

## 단계

| 단계 | 하는 일 |
|---|---|
| ① 과제 | 하고 싶은 일부터 고름: 큰 틀(분류·회귀·이상 탐지·예측·구간 분할) → 데이터 종류(시계열·표·이미지) → 대표 모델. 대표 모델마다 설명과 쓰는 때. 지금 쓸 수 있는 과제는 아래 표, 나머지는 "준비 중" |
| ② 데이터 | CSV(한 행 = 샘플, 값 열 + 마지막 열 정답), 샘플마다 CSV 하나인 시계열 폴더(정답은 클래스별 하위 폴더 또는 `파일이름,정답` 표), 이미지 폴더(클래스별 하위 폴더, 흑백 8·16·28·32 크기로 맞춤, Pillow 필요) 또는 픽셀 CSV. 오류는 행·열 위치와 함께 쉬운 말로 |
| ③ 모델 | 층을 선으로 잇는 그래프 편집기. 왼쪽 목록의 층(Keras 용어)을 끌어 선 위에 놓으면 끼워 넣고, 상자 아래 동그라미를 끌어 다른 상자에 놓으면 선을 이음(Add·Concat). 잔차·Repeat 묶음·갈래, 방향(위→아래·왼쪽→오른쪽)·줄바꿈, 되돌리기·복사·붙여넣기. 층: Conv1D·AvgPool1D·GlobalAvgPool1D·LastTimestep·Conv2D·AvgPool2D·GlobalAvgPool2D·BatchNorm·Activation(ReLU·LeakyReLU·GELU·SiLU·ELU·Mish·Softplus·Sigmoid·Tanh)·Dropout·Flatten·Dense·LSTM·GRU·SimpleRNN·SE block·Add·Concat·Repeat. 상자마다 지난 데이터 모양, 옆에 파라미터 수·크기. 안 되는 조립은 그 상자를 빨갛게 하고 이유·고칠 방법 표시 |
| ④ 학습 | 이 PC에서 학습 (GPU 있으면 사용, Adam, 분류 교차 엔트로피·회귀 MSE). 학습 곡선, 분류는 정확도·혼동 행렬·클래스별 재현율, 회귀는 R²·평균 오차(MAE)·"예측 vs 실제" 산점도, 샘플 하나가 층을 지나는 모습. 회귀 정답은 정규화해 학습하고 출력층에서 되돌려, 모델은 원래 단위로 냄 |

지금 쓸 수 있는 과제와 대표 모델 (모두 ReLU, 예시 데이터에서 검사·학습까지 시험으로 확인):

| 과제 | 예시 데이터 | 대표 모델 |
|---|---|---|
| 분류 · 시계열 | 심전도 정상·이상 (의료, 합성), 설비 진동 정상·결함 (제조, 합성) | 1D CNN, ResNet1D, InceptionTime, TCN, LSTM |
| 분류 · 표 | 공정 조건 → 양품·불량 (제조, 합성) | MLP, 깊은 MLP |
| 회귀 · 시계열 | 진동 → 남은 수명 (제조, 합성) | 1D CNN, TCN, LSTM |
| 회귀 · 표 | 공정 조건 → 수율 (제조, 합성) | MLP |
| 분류 · 이미지 | 금속 표면 결함 32×32 (합성) / 흉부 X-ray 정상·폐렴 28×28 ([PneumoniaMNIST](https://medmnist.com/) 1,200장, CC BY 4.0, 진단용 아님) / 손글씨 숫자 8×8 ([UCI](https://archive.ics.uci.edu/dataset/80), CC BY 4.0) | 작은 CNN, LeNet식, 작은 ResNet |

이미지 예시 데이터의 출처·라이선스는 `src/dlstudio/data/*.SOURCE.txt`에 있습니다.

## 저장과 내보내기

- **프로젝트:** 모델 그래프·데이터 출처·설정·학습된 가중치를 `.dlstudio` 파일로 저장·열기
- **모델 내보내기 (`.pt`):** ④ 아래 [학습한 모델 내보내기] 또는 위 [내보내기]. 정규화와 입력 모양 맞추기가 파일 안에 들어 있어서,
  받는 쪽은 DL studio 없이 **torch만으로** 학습 CSV와 같은 열 순서의 원래 값(정답 열 빼고)을 넣으면 예측합니다.

```python
import json, torch
extra = {"studio.json": ""}
m = torch.jit.load("model.pt", _extra_files=extra)       # TorchScript
info = json.loads(extra["studio.json"])                  # task, classes, n_features, features, input, output
scores = m(torch.tensor(rows, dtype=torch.float32))      # rows: (샘플 수, 특징 수)
pred = [info["classes"][i] for i in scores.argmax(1)]    # 분류. 회귀면 scores[:, 0]이 예측값 (원래 단위)
```

## 코드로 보기 (PyTorch · TensorFlow)

③ 모델·④ 학습의 [코드로 보기]: 지금 모델을 사람이 읽고 고칠 수 있는 파이썬 코드로 바꿉니다. 층 하나가 코드 한 줄이고, 줄마다 화면 이름과 출력 모양이 주석으로 달립니다. Repeat 묶음은 펼쳐서 `층이름_r0`, `층이름_r1`처럼 씁니다.

| | PyTorch | TensorFlow (Keras) |
|---|---|---|
| 만드는 코드 | `StudioModel(nn.Module)` + `train()` + 실행 예 | `build_model()` (함수형 API) + `train()` + 실행 예 |
| 입력 모양 | 채널 먼저 `(샘플 수, 1, 길이)` · 이미지 `(샘플 수, 1, 높이, 너비)` · 순환 층 `(샘플 수, 길이, 1)` | 채널 마지막 `(샘플 수, 길이, 1)` · 이미지 `(샘플 수, 높이, 너비, 1)` |
| 학습한 가중치 | 학습한 모델이면 [파일로 저장] 때 `<이름>_weights.npz`도 저장, 코드가 불러옴 (studio와 출력이 같음, 시험으로 확인) | 들어 있지 않음 → `train()`으로 새로 학습 |

TensorFlow 코드는 문법과 구조(층 이름·모양 계산)를 시험으로 확인했지만, 이 PC에서 TensorFlow를 실행해 보지는 못했습니다 (`tests/test_codegen.py`의 Keras 시험은 TensorFlow가 설치된 환경에서만 돎).

## 라벨링 작업실

상단 [라벨링 작업실] (새 탭 `/label`): 정답이 없는 데이터에 정답을 붙입니다. 데이터 파일은 바꾸지 않고 작업 파일(`.dllabel`)에 바로바로 저장합니다.

- 데이터: 시계열(샘플마다 CSV 하나인 폴더) · 이미지 폴더 · 표(CSV). 정답: 클래스 · 숫자 · 구간(시계열 시점별) · 상자(객체 탐지)
- 단축키(숫자 1–9 클래스, Enter 확인, ←/→), 규칙·정답 표 가져오기·**비슷한 것끼리 묶기**·**모델 제안**, 여럿이 나눠 하고 합치기
- [학습 데이터로 쓰기]: 클래스·숫자 정답을 바로 ② 데이터로 불러옴. 구간·상자는 표·YOLO·COCO로 내보내기

## 개발

```bash
pip install -e . pytest
python -m pytest            # 화면 그래프 편집기 시험은 node가 있으면 함께 돎
```

| 파일 (`src/dlstudio/`) | 하는 일 |
|---|---|
| `graph.py` · `blocks.py` | 그래프 검사(모양·연결·규칙), 층 → torch 모델 |
| `data.py` | CSV·폴더·이미지 읽기, 과제 판단, 정규화, 학습/시험 나누기, 예시 데이터 |
| `jobs.py` | 학습 루프(`fit`)와 작업 스레드(진행·멈추기) |
| `project.py` | 프로젝트 저장·열기, `.pt` 내보내기 |
| `codegen.py` | 그래프 → PyTorch·TensorFlow Keras 코드, 가중치 이름 옮기기 |
| `tasks.py` | 과제·대표 모델 목록 |
| `labeling*.py` | 라벨링 작업실 |
| `server.py` · `cli.py` | 로컬 HTTP 서버(표준 라이브러리만)와 실행 명령 |
| `static/` | 화면 (외부 라이브러리 없음) |

## 출처

[heml](https://github.com/cryptolab-jiminjeon/heml)의 `heml studio`(동형암호 추론용)에서 2026-11-24에 떼어 낸 프로젝트입니다.
동형암호 관련 부분(암호문 변환·비용·원격 서버 실행, Poly2 활성화, HE 때문에 있던 조립 규칙)을 모두 빼고 평문 딥러닝만 남겼습니다.

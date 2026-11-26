# BlockLearning

파이썬·딥러닝 코드를 몰라도 브라우저에서 **층(블록)을 선으로 이어** 모델을 만들고, 학습하고, 학습한 모델을 파일로 내보내 쓰는 도구입니다.
제조(센서·설비)·의료(심전도 등) 시계열·표 데이터와 흑백 이미지를 대상으로 합니다.

> 교육·실험용 도구입니다. 학습 결과를 의료 진단이나 제조 판정 같은 실제 결정에 그대로 쓰지 마세요.

**웹 버전:** https://jeon-stat.github.io/blocklearning/ — 설치 없이 브라우저에서 바로 씁니다. 계산은 모두 그 브라우저 안에서 하고, 고른 데이터 파일은 PC 밖으로 나가지 않습니다.
**로컬 실행:** 아래처럼 내 PC에서 띄우면 `127.0.0.1`에만 열리고, 학습은 PyTorch로 합니다.

## 웹 버전 (GitHub Pages)

저장소 맨 위의 `index.html`이 화면(`src/blocklearning/static/index.html`)을 불러 쓰고, 서버 대신 브라우저 안에서 계산합니다.

| 하는 일 | 로컬 실행 | 웹 버전 |
|---|---|---|
| 검사·데이터 읽기·대표 모델·코드 생성·프로젝트 파일 | Python (PyTorch) | **같은 Python 코드**를 [Pyodide](https://pyodide.org)로 브라우저에서 |
| 학습·샘플 따라가기·모델 정보 | PyTorch | [TensorFlow.js](https://www.tensorflow.org/js) (`static/engine.js`, 그래픽카드가 있으면 WebGL로 빠르게) |
| 데이터 넣기 | 경로 입력 | 파일·폴더 고르기 (브라우저 안에만 둠) |
| 내보내기 | TorchScript `.pt` | PyTorch 코드 + 가중치 `.npz` 내려받기 |
| 저장·열기 | `.blocklearning` 파일 | **내 프로젝트**: 바뀌면 자동 저장, 다시 열면 데이터·학습 결과째 이어서. 저장 위치는 이 브라우저(IndexedDB) 또는 **구글 드라이브**(내 드라이브/BlockLearning/<프로젝트>/). `.blocklearning`(데이터 포함) 내려받기·가져오기 |
| 라벨링 작업실 | 있음 | 아직 없음 |

- 엔진은 PyTorch 식을 TF.js 연산으로 그대로 옮겨서, 같은 가중치면 출력·기울기·BatchNorm 갱신이 PyTorch와 같습니다 (`tests/test_engine.py`, 46개 경우).
  그래서 웹에서 학습한 가중치를 PyTorch 코드로 그대로 내보냅니다.
- 처음 열 때 Pyodide·TF.js를 받느라 5–10초쯤 걸립니다. 그다음은 서비스 워커(`sw.js`)가 받아 둔 것을 씀: 버전이 박힌 CDN 파일만
  브라우저에 두고, 이 사이트의 파일은 늘 네트워크에서 먼저 받아 새 버전이 바로 보입니다 (데이터·프로젝트는 지나지 않음).
- 구글 드라이브: 권한은 `drive.file`(BlockLearning이 만든 파일만)뿐이고, 출입증은 메모리에만 둡니다 (새로 고치면 로그인 한 번).
  OAuth 클라이언트 ID는 `static/store.js`의 `GOOGLE_CLIENT_ID` (공개 값). 개인정보처리방침: [privacy.html](privacy.html)
- 보안: `index.html`의 CSP로 통신은 구글 API·jsDelivr만 허용(조작된 코드가 데이터를 낯선 서버로 못 보냄), CDN 스크립트는 SRI 해시로 확인 (`static/boot.js`)
- 배포: 저장소 Settings → Pages → Source를 **Deploy from a branch**, 가지 `main`, 폴더 `/ (root)`로. 무료 계정은 공개 저장소에서만 Pages를 쓸 수 있습니다.
- 내 PC에서 웹 버전을 미리 보려면 저장소 맨 위에서 `python -m http.server 8000` → http://127.0.0.1:8000/

## 실행 (로컬)

```bash
pip install blocklearning   # 처음 한 번 (torch가 이미 있으면 그대로 씀)
blocklearning               # 또는 python -m blocklearning
```

NVIDIA GPU로 학습하려면 [pytorch.org](https://pytorch.org/get-started/locally/)에서 고른 GPU용 `pip install torch ...`를 먼저 실행하세요
(그냥 설치하면 Windows에서는 CPU용 torch가 깔립니다). 소스에서 설치하려면 저장소에서 `pip install -e .`.

브라우저에 `http://127.0.0.1:8765/?token=…`이 열립니다. `--port 8800`으로 포트를, `--no-browser`로 브라우저 자동 열기를 바꿀 수 있습니다.
애플리케이션 제어 정책이 실행 파일(`blocklearning.exe`)을 막는 PC에서는 `python -m blocklearning`을 쓰세요.
학습 화면의 '장치' 칸에 학습에 쓸 GPU 이름(없으면 CPU)이 나옵니다. 데이터는 경로를 넣거나 [파일 고르기]·[폴더 고르기]로 브라우저에서 고릅니다
(고른 파일은 BlockLearning을 실행한 컴퓨터의 `~/.blocklearning/uploads/`에 복사됨. 다 쓴 뒤 지워도 됨).
SSH로 접속한 서버에서 실행하면 저장·내보낸 파일(프로젝트, `.pt`, 코드)이 서버에 저장되고 브라우저로도 내려받아집니다.

### GPU 서버에서 쓰기

서버에 SSH로 접속해 `pip install blocklearning` → `blocklearning`을 실행하면, 명령창에 내 PC에서 할 일이 나옵니다.

- **터미널(SSH)로 접속했을 때:** 내 PC에서 새 터미널을 열고 안내된 `ssh -L 8765:127.0.0.1:8765 사용자@서버` 를 실행한 뒤, 안내된 주소를 내 PC 브라우저로 엽니다.
- **VS Code(Remote-SSH)로 접속했을 때:** 안내된 주소를 Ctrl+클릭하면 VS Code가 포트를 이어 내 PC 브라우저로 엽니다.
- 터미널을 닫아도 학습이 계속되게 하려면 `tmux new -s bl` 안에서 실행하고 `Ctrl+B`, `D`로 빠져나옵니다 (다시 볼 때 `tmux attach -t bl`).

### 보안

- 서버는 그 컴퓨터 안(`127.0.0.1`)에서만 열립니다. 다른 컴퓨터에서 보려면 위처럼 SSH로 포트를 이어 오세요. **서버를 바깥(`0.0.0.0`)에 열지 마세요.**
- 실행할 때마다 새 **접속 열쇠**(주소의 `token`)를 만들고, 열쇠가 든 주소로 연 브라우저만 들어옵니다 (Jupyter와 같은 방식).
  같은 서버를 쓰는 다른 사용자도 열쇠 없이는 못 들어오니, **그 주소를 남에게 보여 주거나 공유하지 마세요.** 끄면(Ctrl+C) 열쇠도 버려집니다.
- 다른 웹사이트가 몰래 부르는 요청(CSRF·DNS rebinding)은 Host·Origin 검사로 거절하고, 프로젝트 파일은 코드 실행 없이(pickle 없이) 읽습니다.
- 학습·데이터는 그 컴퓨터 밖으로 보내지 않습니다. 보안 문제를 찾으면 [SECURITY.md](SECURITY.md)를 보세요.

## 단계

| 단계 | 하는 일 |
|---|---|
| ① 과제 | (고르지 않아도 됨: [과제는 나중에]로 ② 데이터부터 넣으면 데이터를 보고 분류·회귀와 시계열·표·이미지를 정해 과제·추천 모델을 고름, 시계열↔표는 한 번에 바꿀 수 있음) 하고 싶은 일부터 고름: 큰 틀(분류·회귀·객체 탐지·이상 탐지·예측·구간 분할) → 데이터 종류(시계열·표·이미지) → 대표 모델. 대표 모델마다 설명과 쓰는 때. 지금 쓸 수 있는 과제는 아래 표, 나머지는 "준비 중" |
| ② 데이터 | CSV(한 행 = 샘플, 값 열 + 마지막 열 정답), 샘플마다 CSV 하나인 시계열 폴더(정답은 클래스별 하위 폴더 또는 `파일이름,정답` 표, 값 열을 여러 개 고르면 **여러 채널**), 이미지 폴더(클래스별 하위 폴더, 흑백 8·16·28·32 크기로 맞춤, Pillow 필요) 또는 픽셀 CSV. CSV는 고르면 먼저 **열 고르기**: 열마다 짐작한 역할(마지막 열 = 정답, 글자뿐인 열·0이나 1부터 1씩 느는 번호 열 = 안 씀)을 채워 보여 주고, 열 이름을 눌러 입력 → 정답 → 안 씀을 바꾼 뒤 [불러오기] (같은 역할의 숫자 열이 많으면 칩 하나로 묶음, 정답이 둘 이상이면 정답 여러 개, 값 열이 많으면 채널 수로 나눔). 불러온 뒤에도 같은 표로 바꿔 다시 읽음. 오류는 행·열 위치와 함께 쉬운 말로 |
| ③ 모델 | 층을 선으로 잇는 그래프 편집기. 왼쪽 목록의 층(Keras 용어)을 끌어 선 위에 놓으면 끼워 넣고, 상자 아래 동그라미를 끌어 다른 상자에 놓으면 선을 이음(Add·Concat). 잔차·Repeat 묶음·갈래, 방향(위→아래·왼쪽→오른쪽)·줄바꿈, 되돌리기·복사·붙여넣기. 층 90여 개를 16개 범주로 (왼쪽 목록에서 범주를 누르면 펼침, 위 칸으로 찾기): 완전연결(Dense·GLU·SwiGLU) · 합성곱(Conv1D·Causal·Conv2D·Depthwise·Separable) · 줄이기·모으기(Avg·Max·GlobalAvg·GlobalMax·Adaptive·AttentionPool·First/LastTimestep) · 키우기(Upsample·PixelShuffle·ConvTranspose) · 순환(LSTM·GRU·SimpleRNN, 양방향·줄 그대로 내보내기, 어디에나) · 어텐션(SE·ECA·CBAM·Spatial·MultiHead·Causal·Cross·Linear·NonLocal) · Transformer 부품(PositionalEncoding sin·학습형, RoPE, PatchEmbedding, CLS token) · 정규화(BatchNorm·LayerNorm·RMSNorm·GroupNorm·InstanceNorm) · 활성화 15종(ReLU·GELU·SiLU·SELU·ReLU6·HardSwish·Softmax·PReLU 등) · 과적합 막기(Dropout·SpatialDropout·DropPath·GaussianNoise) · 합치기·나누기(Add·Concat·Multiply·Average·Maximum·Split) · 모양 바꾸기(Flatten·Reshape·Transpose·ToSequence·Crop·Pad) · 입력 변환(Embedding·Spectrogram·Mel·MFCC) · 증강(Flip·Rotation·Shift·Zoom·Scale·Stretch, 학습 때만) · 사전학습(MobileNetV3-Small, 아래) · **묶음** 14종(Conv–BN–ReLU, ResNet·Bottleneck·SE-ResNet, Inception, MobileNet, DenseNet, ConvNeXt, U-Net 한 단계, Transformer Encoder·Decoder, MLP-Mixer, ViT 입구, Residual MLP: 넣는 자리의 채널에 맞춰 만들어짐) · Repeat. 1D·2D 층은 들어오는 모양으로 가림 (그림도 ToSequence 뒤에는 어텐션·Conv1D, 시계열도 Spectrogram 뒤에는 Conv2D). 모든 층은 PyTorch·웹 엔진이 같은 값을 냄(시험으로 확인), PyTorch 코드로 내보내짐 (Keras 코드는 일부 층 빼고) 상자마다 지난 데이터 모양, 옆에 파라미터 수·크기. 안 되는 조립은 그 상자를 빨갛게 하고 이유·고칠 방법 표시 |
| ④ 학습 | 이 PC에서 학습 (GPU 있으면 사용, Adam, 분류 교차 엔트로피·회귀 MSE·탐지 YOLO식 손실). 학습 곡선, 분류는 정확도·혼동 행렬·클래스별 재현율, 회귀는 R²·평균 오차(MAE)·"예측 vs 실제" 산점도, 탐지는 mAP@0.5·클래스별 AP·점수 50%에서 정밀도·재현율과 시험 그림 위 정답(점선)·찾은 상자(실선, 점수 슬라이더), 샘플 하나가 층을 지나는 모습(어텐션 블록이 있으면 **어텐션 열지도**: 그 샘플에서 어디를 봤는지를 그림·신호·표 위에 빨갛게 겹치고, 질문 위치 × 본 위치 지도도). 샘플 따라가기의 **어디를 봤나**: 입력 값마다 \|∂ 예측 / ∂ 입력\|를 입력 위에 빨갛게 (어텐션이 없어도). 회귀 정답은 정규화해 학습하고 출력층에서 되돌려, 모델은 원래 단위로 냄. [더 보기]: **가장 좋은 epoch 남기기**(학습 몫의 10%를 검증 몫으로 떼어 검증 손실(탐지는 검증 mAP)이 가장 좋았던 epoch의 모델을 남김)·**조기 종료**·**손실 조립**(아래). **학습 기록**: 학습할 때마다 모델·설정·성적이 남고(최근 20번, 프로젝트에 저장), [이 모델로]로 그때 모델로 돌아감, 둘을 골라 **비교**(다른 칸만 진하게). 혼동 행렬 칸을 누르면 그 칸(틀린 샘플 등)의 샘플들을 봄 |
| ⑤ 예측 | 학습한 모델로 새 데이터를 예측(추론). 예시 데이터면 [예시 새 데이터 10개로 예측]으로 학습·시험에 쓰지 않은 10개를 바로 예측하고 정답과 비교: 학습 때와 같은 모양의 CSV(정답 열은 있어도 없어도 됨), 그림 폴더, 시계열 폴더, 표 데이터는 값 직접 넣기. 같은 정규화를 해서 예측하고, 정답이 있으면 맞았는지 표시. 결과는 CSV로 내려받기 |

웹 버전은 처음 열면 **내 프로젝트** 첫 화면에서 프로젝트를 고르거나 새로 시작합니다. ③ 모델에서 블록을 고르고 [미리보기]를 누르면 샘플 하나가 그 층을 지난 값과 모양을 봅니다 (누를 때만).

지금 쓸 수 있는 과제와 대표 모델 (CNN·RNN 계열은 ReLU, Transformer 계열은 GELU·LayerNorm, 예시 데이터에서 검사·학습까지 시험으로 확인):

| 과제 | 예시 데이터 | 대표 모델 |
|---|---|---|
| 분류 · 시계열 | 베어링 진동 정상·내륜·외륜·볼·복합 결함 ([오타와대](https://doi.org/10.17632/v43hmbwxpm.2), CC BY 4.0) / 심전도 박동 정상·이상 ([MIT-BIH](https://physionet.org/content/mitdb/1.0.0/), ODC-By 1.0, 진단용 아님) | 1D CNN, ResNet1D, InceptionTime, TCN, LSTM, Transformer |
| 분류 · 표 | 강판 결함 7종 — 결함 영역의 모양·밝기 27가지 ([UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198), CC BY 4.0) | MLP, 깊은 MLP, FT-Transformer식 |
| 회귀 · 시계열 | 유압 압력 신호 → 밸브 상태 % ([UCI 유압 시스템](https://archive.ics.uci.edu/dataset/447), CC BY 4.0) | ResNet1D, 1D CNN, TCN, LSTM, Transformer |
| 회귀 · 표 | 콘크리트 배합 → 압축강도 MPa ([UCI](https://archive.ics.uci.edu/dataset/165), CC BY 4.0) | MLP, FT-Transformer식 |
| 객체 탐지 · 이미지 | PCB 결함 위치 6종 64×64 ([PCB-Defect](https://doi.org/10.17632/vdj74sngvn.2), CC BY 4.0) | 작은 탐지 모델, 깊은 탐지 모델, 전이 학습 탐지 (MobileNetV3) |
| 이상 탐지 · 시계열 | 베어링 진동 — 정상만 학습해 결함 찾기 (오타와대, CC BY 4.0) | Dense 오토인코더, Conv 오토인코더 |
| 이상 탐지 · 표 | 유압 시스템 — 정상만 학습해 냉각기·밸브·펌프 고장 찾기 (UCI, CC BY 4.0) | 오토인코더 |
| 이상 탐지 · 이미지 | 강판 표면 — 양호만 학습해 불량 찾기 (SteelBlastQC) / 흉부 X-ray — 정상만 학습해 폐렴 찾기 (PneumoniaMNIST) | 사전학습 특징 + 정상과의 거리 |
| 예측 · 시계열 | 철강 공장 전력 사용량 1시간마다 — 지난 일주일과 요일·시각으로 다음 하루 ([UCI, 광양 철강 공장](https://archive.ics.uci.edu/dataset/851), CC BY 4.0) | 선형 (DLinear식), MLP, 1D CNN, LSTM, Transformer |
| 분류 · 이미지 | 강판 표면 양호·불량 64×64 ([SteelBlastQC](https://doi.org/10.34894/EKZNN0), CC BY 4.0) / 흉부 X-ray 정상·폐렴 28×28 ([PneumoniaMNIST](https://medmnist.com/), CC BY 4.0, 진단용 아님) / 손글씨 숫자 32×32 ([UCI](https://archive.ics.uci.edu/dataset/80), CC BY 4.0) | 작은 CNN, LeNet식, 작은 ResNet, 전이 학습 (MobileNetV3), CNN + Transformer, ViT |
| 구간 분할 · 시계열 | 출하 검사 기계 전류 100Hz — 시점마다 공정 단계 5묶음 / 같은 기계 전류 + 공압 압력 두 채널 ([Mendeley ypzswhhzh9](https://doi.org/10.17632/ypzswhhzh9.3), CC BY 4.0) | TCN (팽창 합성곱), U-Net식, 양방향 GRU |
| 여러 정답 · 표 | 유압 시스템 — 냉각기·펌프 상태(분류)와 밸브 %·축압기 압력(숫자)을 한 번에 (UCI, CC BY 4.0) | MLP (몸통 함께 쓰기), 깊은 MLP (몸통 함께 쓰기) |

예시 데이터는 모두 공개된 실제 데이터입니다 (합성 데이터 없음). 원본을 작게 줄인 파일과 출처·라이선스·바꾼 내용은 `src/blocklearning/data/*.npz`·`*.SOURCE.txt`, 원본에서 다시 만드는 법은 `tools/build_examples.py`. 웹 버전은 예시를 누를 때만 그 파일을 받습니다.

**객체 탐지**는 YOLO 첫 버전과 같은 격자 방식입니다: 그림을 2×2 AvgPool로 세 번 줄여 8×8 격자(64×64 그림 기준)로 만들고, 출력 블록이 격자 칸마다 1×1 필터로 '사물의 가운데가 이 칸에 있나 · 칸 안 위치 · 너비·높이 · 클래스'를 냅니다 (출력 앞이 격자가 아니면 모델 검사가 막음). 손실은 있음(BCE, 빈 칸은 무게 0.5) + 위치·크기 제곱 오차 × 5 + 클래스 교차 엔트로피, 찾은 상자는 같은 클래스끼리 50% 넘게 겹치면 점수 높은 것만 남깁니다. 같은 칸에 사물 가운데가 둘이면 큰 것만 배웁니다 (작은 사물이 빽빽하면 그림 크기를 키우세요).

## 저장과 내보내기

- **프로젝트:** 모델 그래프·데이터 출처·설정·학습된 가중치·학습 기록을 `.blocklearning` 파일로 저장·열기
- **모델 내보내기 (`.pt`):** ④ 아래 [학습한 모델 내보내기] 또는 위 [내보내기]. 정규화와 입력 모양 맞추기가 파일 안에 들어 있어서,
  받는 쪽은 BlockLearning 없이 **torch만으로** 학습 CSV와 같은 열 순서의 원래 값(정답 열 빼고)을 넣으면 예측합니다.

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

## 라벨링

[라벨링 ↗] (새 탭, 웹은 `label.html`, 로컬은 `/label`): 정답이 없는 데이터에 정답을 붙입니다. 데이터 파일은 바꾸지 않습니다.

- **웹 버전:** 라벨링 페이지에서 데이터(폴더·CSV)를 고르고 정답을 붙이면, 정답과 데이터 사본이 이 브라우저 보관함에 자동 저장됩니다 ('내 라벨링 작업').
  studio ② 데이터의 **라벨링한 데이터**에서 골라 학습 데이터로 불러오거나, 라벨링 페이지의 [학습 데이터로 쓰기]로 studio를 새 탭에 엽니다.
  다른 PC·다른 사람과는 [작업 내려받기] → [작업 묶음 가져오기] (`.dllabel.zip`: 정답 + 데이터 사본). 라벨링 작업의 구글 드라이브 저장은 아직 없습니다.
- **로컬 실행:** 작업 파일(`.dllabel`)에 바로바로 저장합니다.

- 데이터: 시계열(샘플마다 CSV 하나인 폴더) · 이미지 폴더 · 표(CSV). 정답: 클래스 · 숫자 · 구간(시계열 시점별) · **도형**(이미지: 상자·다각형·원·점, 도구 B·P·R·O, 꼭짓점 끌기·변 가운데로 추가·Alt+누르기로 지우기, 원은 테두리 ●로 반지름)
- 시계열은 기준 열 + **함께 볼 열**(센서 여러 개, 7개까지)을 위아래로 함께 보며 붙임. 함께 볼 열의 요약값(평균·최대 등)도 규칙·정렬·묶기에 씀 (studio 학습은 기준 열 하나)
- 단축키(숫자 1–9 클래스, Enter 확인, ←/→), 규칙·정답 표 가져오기·**비슷한 것끼리 묶기**, 여럿이 나눠 하고 합치기
- **빠른 제안**(요약값으로 작은 모델) · **딥러닝 제안**(정답 10개 이상이면 시계열 1D CNN·그림 작은 CNN·표 MLP를 학습해 나머지에 제안, 웹은 브라우저 안에서, PC는 PyTorch)
- [학습 데이터로 쓰기]: 클래스·숫자 정답을 바로 ② 데이터로 불러옴. 도형(이미지)은 **객체 탐지** 데이터로 (다각형·원은 감싸는 상자로, 점은 뺌, 흑백 64×64). 구간은 표로, 도형은 YOLO(상자만이면 탐지, 다각형·원이 있으면 분할 — 원은 32각형)·COCO(bbox·segmentation·keypoints)로 내보내기

## 개발

```bash
pip install -e . pytest
npm install --prefix tests/js   # 웹 엔진 시험용 TF.js (node가 있으면)
python -m pytest                # 화면 그래프 편집기·웹 엔진 시험은 node가 있으면 함께 돎
```

### PyPI 배포

`pyproject.toml`의 `version`을 올려 push한 뒤, GitHub에서 태그 `v<버전>`(예: `v0.1.1`)으로 Release를 게시하면
`.github/workflows/publish.yml`이 패키지를 만들어 PyPI에 올립니다 (Trusted Publishing, 토큰 없음). PyPI 소개 글은 `docs/PYPI.md`.
처음 한 번: PyPI → Account settings → Publishing → **Add a new pending publisher**에서
프로젝트 `blocklearning`, 소유자 `jeon-stat`, 저장소 `blocklearning`, 워크플로 `publish.yml`, 환경 `pypi`를 등록.

| 파일 (`src/blocklearning/`) | 하는 일 |
|---|---|
| `graph.py` · `blocks.py` | 그래프 검사(모양·연결·규칙), 층 → torch 모델 |
| `data.py` | CSV·폴더·이미지 읽기, 과제 판단, 정규화, 학습/시험 나누기, 예시 데이터 |
| `jobs.py` | 학습 루프(`fit`)와 작업 스레드(진행·멈추기) |
| `project.py` | 프로젝트 저장·열기, `.pt` 내보내기 |
| `codegen.py` | 그래프 → PyTorch·TensorFlow Keras 코드, 가중치 이름 옮기기 |
| `pretrained.py` · `pretrained/` | 사전학습 가중치 (고정) 읽기·채우기 |
| `tasks.py` | 과제·대표 모델 목록 |
| `labeling*.py` | 라벨링 작업실 |
| `core.py` | 상태와 API (HTTP와 상관없음: 로컬 서버와 웹 버전이 함께 씀) |
| `server.py` · `cli.py` | 로컬 HTTP 서버(표준 라이브러리만)와 실행 명령 |
| `web.py` · `static/web.js` · `static/engine.js` · `static/train-worker.js` | 웹 버전: Pyodide 쪽 함수, 화면과 잇는 다리, TF.js 학습 엔진, 학습 일꾼(화면과 다른 스레드) |
| `static/store.js` · `static/projects.js` · `static/boot.js` | 웹 버전: 저장소(브라우저·구글 드라이브), 내 프로젝트·자동 저장, 시작 |
| `static/` | 화면 (외부 라이브러리 없음) |

### 전이 학습 (사전학습 MobileNetV3-Small)

이미지 분류의 대표 모델 [전이 학습 (MobileNetV3)]과 블록 목록의 '사전학습' 범주. Google이 ImageNet 사진으로 미리 학습한 MobileNetV3-Small 특징 추출기(값 151만 개, 6MB)가
그림 하나를 특징 1024개로 바꾸고, 그 뒤의 층만 학습합니다. 사전학습 부분은 학습하지 않습니다(고정).

- 블록 안에서: 그림마다 0~1로 → 정한 크기(64–224, 기본 128)로 늘이거나 줄임(bilinear) → 흑백이면 3채널로 복사 → −1~1. 가장자리 채움은 원래 모델(TensorFlow)과 같게.
- 학습 때 특징을 한 번만 뽑아 두고 뒷부분만 반복 학습합니다 (블록이 입력 바로 뒤에 있고, 앞에 증강이 없을 때).
- 탐지: 블록 설정 [출력]을 '특징 지도'로 하면 그림을 2배로 넣어 앞 블록 8개까지만 써서, 원래 그림 8픽셀마다 특징 48개(탐지 격자와 같은 칸)를 냅니다. 대표 모델 [전이 학습 탐지 (MobileNetV3)] = 특징 지도 → Conv2D 64 → ReLU → 출력. PCB 결함 예시: 시험 mAP@0.5 0.23 (PyTorch 58초), 처음부터 학습하는 깊은 탐지 모델 0.23 (약 400초)·작은 탐지 모델 0.09.
- 미세조정: 블록 설정 [함께 학습할 끝 블록]을 1–3으로 하면 끝 블록과 마지막 1×1 합성곱 둘을 함께 학습합니다 (작은 학습률 권장, 예: 0.0003). 15 epoch에서 강판 93.3% → 96.0% (끝 2개), 흉부 X-ray 90.8% → 93.3% (끝 1개)로 올랐지만, 특징을 매번 다시 계산해 3–10배 오래 걸리고 프로젝트에 학습한 값(수 MB)이 저장됩니다. 아주 작은 그림(28×28 흉부 X-ray)에서는 처음부터 학습하는 모델보다 나아지지 않을 수 있습니다.
- 가중치: Google의 [MobileNetV3 small-100-224 feature vector](https://www.kaggle.com/models/google/mobilenet-v3/tfJs/small-100-224-feature-vector/1) (TF Hub `mobilenet_v3_small_100_224/feature_vector/5`, **Apache License 2.0**). `tools/build_pretrained.py`가 TF.js 파일을 `src/blocklearning/pretrained/mnv3s.bin`·`.json`으로 옮기고(축 순서만 바꿈), 원래 모델과 출력이 같은지 확인합니다. 출처·바꾼 내용은 `mnv3s.SOURCE.txt`, 라이선스 원문은 `LICENSE-mnv3s.txt`.
- 웹 버전은 이 블록을 처음 쓸 때 같은 사이트에서 가중치를 받습니다. 프로젝트·학습 기록에는 고정 가중치를 넣지 않고(열 때 다시 채움), 내보내는 PyTorch 코드·가중치에는 모두 넣습니다. Keras 코드로는 내보내지 않습니다.

### 구간 분할 (시점마다 상태)

긴 신호의 시점 하나하나에 상태(공정 단계 등)를 붙입니다. 신호를 길이 L의 창으로 자르고, 모델은 길이를 줄이지 않은 채 끝까지 가서 출력 블록이 시점마다 1×1 합성곱으로 상태 점수를 냅니다 (출력 = 상태 수 × L, 상태가 먼저). 손실은 시점마다 교차 엔트로피, 성적은 시점 정확도입니다.

- 출력 블록 바로 앞은 길이 L인 줄이어야 합니다. 길이를 줄이는 Pool·stride 2·Flatten·Dense를 쓰면 빨간 상자로 알려 줍니다 (줄였으면 Upsample로 되돌리기, 순환 층은 return_sequences).
- 결과: 시점 정확도(시험·학습), 시점마다 센 혼동 행렬·상태별 재현율, 상태가 많이 바뀌는 시험 창의 신호와 정답·예측 색 띠. ⑤ 예측은 예시 새 데이터 10개의 창마다 예측 구간과 정답, CSV(시점마다 한 줄).
- 예시: 소형 전기 구동부 출하 검사 기계의 전류(100Hz)를 256칸(2.56초)씩 자르고, 기계 PLC 상태를 다섯 묶음(대기 · 동작 1–10 · 동작 11–16 · 긴 단계 17 · 동작 18–25)으로 붙였습니다. 별도 실험(겹치지 않는 창 전부, 팽창 합성곱 6층, 30 epoch)에서 압력만으로는 대기와 긴 단계를 가르지 못해 시점 정확도 58%, 전류만으로 98%, 둘 다 쓰면 99.7%였습니다. BlockLearning 예시(창 일부, TCN 대표 모델)는 웹에서 10 epoch에 86%입니다.
- 여러 채널: 예시 '전류 + 공압 압력'은 두 신호를 채널로 넣습니다 (입력 모양 (2, 256), 채널마다 정규화).
- 내 CSV: 한 행 = 한 시점 (시간 순서), 값 열들 + 마지막 열 = 그 시점의 상태. 같은 길이의 창으로 겹치지 않게 자르고, 값 열이 여럿이면 채널. 역할 표에서 상태 열·안 쓸 열·창 길이를 바꿉니다.

### 손실 조립

과제를 고르면 손실이 정해지지만 (분류 교차 엔트로피 · 회귀 MSE · 구간 분할 시점마다 교차 엔트로피 · 탐지 YOLO식), ④ 학습 → [더 보기]의 **손실 칩**에서
`손실 = Σ 가중치 × 항`으로 바꿀 수 있습니다. 정답 종류에 맞는 항만 고를 수 있습니다 (설계: `docs/design-recipes.md`).

| 정답 | 항 (설정) |
|---|---|
| 클래스 | 교차 엔트로피 (라벨 스무딩, 드문 클래스 무겁게) · Focal (γ, 드문 클래스 무겁게) |
| 숫자 | MSE · MAE · Huber (δ) |
| 시점마다 상태 | 교차 엔트로피 · Dice (짧은 상태도 놓치지 않게) |
| 모두 | 가중치 L2 · 가중치 L1 · 블록 출력 L1 (고른 블록의 출력을 드물게) |

- 항마다의 값이 학습 곡선 아래에 나옵니다. [조립 예시]로 자주 쓰는 조합을 한 번에, [기본으로]로 되돌림.
- PyTorch(`losses.py`)·웹 엔진(`engine.js buildLoss`)·내보내는 PyTorch 코드(`studio_loss`)가 같은 식 (값과 기울기를 시험으로 비교).

### 정답 여러 개 (분류·숫자 섞어서)

한 입력으로 여러 정답을 한 모델이 함께 맞힙니다. 출력 블록 하나가 정답마다의 머리(분류는 클래스 수, 숫자는 1)를 이어 붙여 내고, 손실은 정답마다 항을 더합니다 (손실 칩도 정답마다 한 줄).
성적은 정답마다 정확도·R², 전체는 그 평균. 결과는 정답 탭마다 혼동 행렬·산점도, ⑤ 예측은 정답마다 한 열.
예시: 유압 시스템 냉각기 상태(3)·펌프 누설(3) + 밸브 %·축압기 bar (깊은 MLP 30 epoch: 냉각기 100%, 펌프 99%, 밸브 R² 0.93, 축압기 R² 0.81).
내 CSV는 역할 표에서 정답 열을 둘 이상 고르면 됩니다 (글자이거나 정수 20종류 이하면 분류, 아니면 숫자).

### 이상 탐지 · 시계열 예측

- **이상 탐지 (오토인코더)**: 정답 열(정상·결함 등)이 있는 데이터를 넣으면 정상 클래스(이름이 정상·normal·양품이면 그것, 아니면 가장 많은 것)만 학습하고, 나머지는 '이상'으로 시험에 씁니다. 정상만 있어도 됩니다. 모델은 입력을 좁게 줄였다가 되살리고, 되살린 값과 입력의 차이(제곱 평균)가 이상 점수입니다. 결과: AUC, 이상 점수 분포(정상·이상)와 기준값 막대(처음은 학습 정상의 95% 지점, [이 기준값을 예측에 쓰기]로 ⑤ 예측·샘플 따라가기에 반영), 점수 높은 샘플의 입력·되살린 값과 차이가 큰 곳(표는 열마다).
- **이미지 이상 탐지 (정상과의 거리)**: 출력 블록을 '정상과의 거리'로 하면 되살리지 않고, 사전학습 MobileNetV3-Small이 뽑은 특징 1024개가 정상 그림들의 특징 분포에서 얼마나 떨어졌는지를 점수로 씁니다 (Mahalanobis 거리를 분산이 큰 방향 128개 + 나머지로 줄인 것). 경사 학습 없이 학습 몫의 정상 특징에서 평균·주성분을 한 번 계산하므로 몇 초면 끝납니다 (웹은 특징을 엔진이 뽑고, 계산은 브라우저 안 Python이 함). 그 앞에는 학습하는 층을 둘 수 없습니다. 같은 예시에서 처음부터 학습한 Conv 오토인코더보다 훨씬 잘 찾습니다 (시험 AUC: 강판 0.96 vs 0.44, 흉부 X-ray 0.83 vs 0.64).
- **시계열 예측**: 한 행 = 한 시점인 긴 시계열 CSV 하나에서 맞힐 열을 고르고, '지난 L개 → 다음 H개' 창으로 잘라 학습합니다 (불러온 뒤 열·L·H·간격을 바꿔 다시 만들기). 시험은 신호의 뒤쪽 구간 (시간 순서 그대로 나눠 미래를 미리 보지 않음). 결과: MAE·RMSE(원래 단위), 단순 예측(지난 H개 되풀이 — 하루 주기면 '어제 같은 시각')과 비교, 몇 칸 뒤까지 맞히는지, 시험 구간 예시. ⑤ 예측은 새 CSV의 끝 L개로 그다음 H개.

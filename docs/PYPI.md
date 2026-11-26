# BlockLearning

파이썬·딥러닝 코드를 몰라도 브라우저 화면에서 **층(블록)을 선으로 이어** 모델을 만들고, 학습하고, 학습한 모델을 파일로 내보내는 도구입니다.
시계열(센서·심전도)·표 데이터와 흑백 이미지를 다룹니다.

- 설치 없이 써 보기: https://jeon-stat.github.io/blocklearning/ (계산은 브라우저 안에서)
- 이 패키지: 같은 화면을 내 PC에 띄우고, 학습은 **PyTorch로 내 PC에서** 합니다. NVIDIA 그래픽카드가 있으면 GPU로 학습합니다.

> 교육·실험용 도구입니다. 학습 결과를 의료 진단이나 제조 판정 같은 실제 결정에 그대로 쓰지 마세요.

## 설치

Python 3.10 이상이 필요합니다.

```
pip install blocklearning
```

**NVIDIA GPU로 학습하려면** 먼저 GPU용 PyTorch를 깔고 나서 위 명령을 실행하세요.
[pytorch.org](https://pytorch.org/get-started/locally/)에서 내 PC(운영체제·CUDA)에 맞는 `pip install torch ...` 명령을 골라 실행하면 됩니다.
그냥 `pip install blocklearning`만 하면 (Windows에서는) CPU용 PyTorch가 깔려 CPU로 학습합니다.

이미 설치했는데 학습 화면의 '장치'에 GPU가 없다고 나오면 (실행할 때 명령창에도 안내가 나옵니다), BlockLearning을 끄고 PyTorch만 GPU용으로 바꾸세요.
GPU용 PyTorch로도 '장치'에서 CPU를 고르면 CPU로 학습합니다.

```
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

마지막 줄이 `True`면 됩니다. `False`면 NVIDIA 드라이버를 업데이트하거나 pytorch.org에서 내 PC에 맞는 명령을 고르세요.
NVIDIA 그래픽카드만 쓸 수 있습니다 (AMD·Intel 내장 그래픽, Mac은 CPU로 학습).

## 실행

```
blocklearning
```

브라우저에 `http://127.0.0.1:8765/?token=…`이 열립니다. 그 컴퓨터 안에서만 열리고(`127.0.0.1`), 데이터는 밖으로 나가지 않습니다.
주소의 `token`은 실행할 때마다 새로 만드는 접속 열쇠라 남에게 보여 주지 마세요.
끝내려면 명령창에서 Ctrl+C. 실행 파일이 회사 정책 등으로 막히면 `python -m blocklearning`.

첫 화면 **내 프로젝트**에 프로젝트가 모입니다. 작업이 바뀌면 `~/BlockLearning/projects/`에 자동 저장되고, 카드를 눌러 이어서 합니다.

**GPU 서버에서:** SSH로 접속한 서버에서 실행하면 명령창에 내 PC에서 할 일(`ssh -L …` 명령, VS Code면 주소 Ctrl+클릭)이 나옵니다.
GPU가 여럿이면 학습 화면의 '장치'에서 번호를 고릅니다('자동'은 가장 덜 쓰는 GPU).
데이터는 화면의 [파일 고르기]·[폴더 고르기]로 내 PC에서 골라 서버로 올릴 수 있습니다.
저장·내보낸 파일은 서버에 저장되고 브라우저로도 내려받아집니다 (프로젝트는 첫 화면의 [지금 프로젝트 내려받기]).

- `--port 8800`: 다른 포트로
- `--no-browser`: 브라우저를 자동으로 열지 않음

## 할 수 있는 것

① 과제 고르기 → ② 데이터 (CSV·이미지 폴더·예시 데이터) → ③ 블록으로 모델 → ④ 학습 → ⑤ 새 데이터로 예측.
분류·회귀·이상 탐지·시계열 예측·구간 분할·물체 찾기·여러 정답 동시에. 학습한 모델은 PyTorch 코드나 TorchScript 파일로 내보낼 수 있습니다.

소스·자세한 설명: https://github.com/jeon-stat/blocklearning · 라이선스: MIT (예시 데이터와 사전학습 가중치는 각자의 라이선스, 패키지 안 `*.SOURCE.txt`·`LICENSE-mnv3s.txt`)

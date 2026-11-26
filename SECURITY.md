# 보안 문제 알리기 (Security Policy)

BlockLearning에서 보안 문제를 찾으면, 공개 이슈에 자세한 내용을 쓰지 말고
GitHub의 **[비공개 보안 알림](https://github.com/jeon-stat/blocklearning/security/advisories/new)**
(저장소 → Security → Report a vulnerability)으로 알려 주세요.

Please report security issues privately via GitHub's
[private vulnerability reporting](https://github.com/jeon-stat/blocklearning/security/advisories/new),
not in public issues.

알려 주실 것: 어떤 버전·어떤 방식(웹 / 설치형)인지, 재현 방법, 생길 수 있는 피해.
확인하는 대로 답하고, 고친 버전을 내면 알림에 적어 두겠습니다. 고치는 동안에는 내용을 공개하지 말아 주세요.

## 대상

- 웹 버전 (https://jeon-stat.github.io/blocklearning/)
- 설치형 (`pip install blocklearning`): 최신 버전만 고칩니다.

## 설계상 지키는 것

- 설치형 서버는 `127.0.0.1`에만 열고, 실행할 때마다 만드는 접속 열쇠(token)가 있어야 들어옵니다.
- 다른 출처(Origin)·다른 Host에서 온 요청은 거절합니다.
- 프로젝트·가중치 파일은 pickle 없이 읽습니다 (남이 준 파일을 열어도 코드가 실행되지 않음).
- 데이터·학습 결과를 운영자에게 보내지 않습니다. 웹 버전은 서버가 없고 모든 계산을 브라우저 안에서 합니다.

BlockLearning은 MIT 라이선스로 "있는 그대로" 제공되며, 사용에 따른 책임은 지지 않습니다 ([LICENSE](LICENSE)).

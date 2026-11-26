"""명령: `blocklearning [--port 8765] [--no-browser]` (또는 `python -m blocklearning`) → PC에 BlockLearning을 띄우고 브라우저를 연다."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from typing import Optional

PORT_TRIES = 20


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="blocklearning", description="BlockLearning: 블록으로 딥러닝 모델 만들기 (브라우저 화면)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    from blocklearning.server import make_server
    server, err = None, None
    for k in range(PORT_TRIES if args.port else 1):          # 포트가 쓰이고 있으면 다음 포트
        try:
            server = make_server(args.port + k if args.port else 0)
            break
        except OSError as e:
            err = e
    if server is None:
        print(f"포트를 열 수 없음: {err}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"BlockLearning: {url}  (끝내려면 Ctrl+C)", flush=True)
    if args.dry_run:
        server.server_close()
        return 0
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

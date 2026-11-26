"""명령: `blocklearning [--port 8765] [--no-browser]` (또는 `python -m blocklearning`) → PC에 BlockLearning을 띄우고 브라우저를 연다.

SSH로 접속한 서버에서 실행하면 브라우저를 열지 않고, 내 PC에서 화면을 여는 방법(SSH 포트 잇기·VS Code)을 알려 준다.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import webbrowser
from typing import Optional

PORT_TRIES = 20


def remote_kind(env=None) -> str:
    """이 명령을 어디서 실행했나: "vscode"(VS Code 원격 터미널) · "ssh"(SSH 접속) · ""(이 PC)."""
    env = os.environ if env is None else env
    if not (env.get("SSH_CONNECTION") or env.get("SSH_CLIENT")):
        return ""
    return "vscode" if env.get("TERM_PROGRAM") == "vscode" or env.get("VSCODE_IPC_HOOK_CLI") else "ssh"


def ssh_command(port: int, env=None) -> str:
    """내 PC에서 실행할 SSH 포트 잇기 명령 (SSH_CONNECTION: 내 PC 주소·포트, 서버 주소·포트)."""
    env = os.environ if env is None else env
    parts = (env.get("SSH_CONNECTION") or "").split()
    host = parts[2] if len(parts) >= 4 else "서버주소"
    sport = parts[3] if len(parts) >= 4 else "22"
    try:
        user = getpass.getuser()
    except Exception:
        user = "사용자"
    return f"ssh -L {port}:127.0.0.1:{port} {'-p ' + sport + ' ' if sport != '22' else ''}{user}@{host}"


def guide(url: str, port: int, env=None) -> str:
    """실행할 때 명령창에 보여 줄 안내."""
    kind = remote_kind(env)
    lines = [f"BlockLearning: {url}", "  끝내려면 Ctrl+C · 주소의 token은 접속 열쇠라 남에게 보여 주지 마세요"]
    if kind == "vscode":
        lines += ["", "VS Code로 서버에 접속해 있습니다. 위 주소를 Ctrl+클릭하면 VS Code가 포트를 이어 내 PC 브라우저로 엽니다."]
    elif kind == "ssh":
        lines += ["", "서버에서 실행 중입니다. 내 PC 브라우저로 화면을 보려면:",
                  "  1) 내 PC에서 새 터미널을 열고 (평소 접속하던 서버 주소로):",
                  f"       {ssh_command(port, env)}",
                  f"  2) 내 PC 브라우저에서 위 주소 열기",
                  "  터미널을 닫아도 계속 돌게 하려면 tmux 안에서 실행하세요 (tmux new -s bl → 실행 → Ctrl+B, D)"]
    return "\n".join(lines)


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
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/?token={server.token}"
    print(guide(url, port), flush=True)
    if args.dry_run:
        server.server_close()
        return 0
    if not args.no_browser and not remote_kind():           # 서버에는 브라우저가 없음
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

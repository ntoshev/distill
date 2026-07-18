#!/usr/bin/env python3
"""Serve one MLX adapter and evaluate it on the fixed ten-example GSM8K run."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def wait_for_port(port: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"MLX server did not open port {port} within {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    server_command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        args.model,
        "--adapter-path",
        str(args.adapter),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-tokens",
        "1024",
        "--chat-template-args",
        '{"enable_thinking":false}',
        "--log-level",
        "WARNING",
    ]
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "local")
    server = subprocess.Popen(server_command, env=environment)
    try:
        wait_for_port(args.port)
        command = [
            "prime",
            "eval",
            "run",
            "gsm8k",
            "--model",
            args.model,
            "--api-base-url",
            f"http://127.0.0.1:{args.port}/v1",
            "--api-key-var",
            "OPENAI_API_KEY",
            "--num-examples",
            "10",
            "--rollouts-per-example",
            "1",
            "--max-tokens",
            "1024",
            "--temperature",
            "0",
            "--output-dir",
            str(args.output),
            "--save-results",
            "--disable-tui",
            "--disable-env-server",
        ]
        return subprocess.run(command, env=environment, check=False).returncode
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    raise SystemExit(main())

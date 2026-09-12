"""
Manages the lifecycle of a local llama-server (llama.cpp) subprocess.

The agent starts this server itself before investigating any folders and
stops it (even on error / Ctrl-C) when done. Nothing here writes to the
riddims database or to any source folder -- this module only launches and
supervises a local HTTP inference server.
"""

import atexit
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

import config


class LlamaServerError(RuntimeError):
    pass


class LlamaServerManager:
    def __init__(
        self,
        server_bin: str = config.LLAMA_SERVER_BIN,
        model_path: str = config.LLAMA_MODEL_PATH,
        host: str = config.LLAMA_HOST,
        port: int = config.LLAMA_PORT,
        ctx_size: int = config.LLAMA_CTX_SIZE,
        n_gpu_layers: int = config.LLAMA_N_GPU_LAYERS,
        parallel_slots: int = config.LLAMA_PARALLEL_SLOTS,
        extra_args: list[str] | None = None,
        startup_timeout_s: int = config.LLAMA_STARTUP_TIMEOUT_S,
    ):
        self.server_bin = server_bin
        self.model_path = model_path
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.parallel_slots = parallel_slots
        self.extra_args = extra_args if extra_args is not None else list(config.LLAMA_EXTRA_ARGS)
        self.startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _preflight(self) -> None:
        if shutil.which(self.server_bin) is None and not Path(self.server_bin).exists():
            raise LlamaServerError(
                f"llama-server executable not found: {self.server_bin!r}. "
                "Set LLAMA_SERVER_BIN in config.py (or the environment) to the "
                "full path of llama-server(.exe), or put it on PATH."
            )
        if not Path(self.model_path).exists():
            raise LlamaServerError(
                f"model file not found: {self.model_path!r}. "
                "Set LLAMA_MODEL_PATH in config.py (or the environment)."
            )

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        self._preflight()

        cmd = [
            self.server_bin,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers),
            "--parallel", str(self.parallel_slots),
            *self.extra_args,
        ]

        print(f"[llama_server_manager] starting: {' '.join(cmd)}", file=sys.stderr)
        # Do not use subprocess.PIPE here: llama-server continually writes
        # request/timing logs, and an unread pipe eventually fills and blocks
        # the server itself.  Forward its combined output to the terminal so
        # the process can never deadlock on logging during a long batch.
        self._proc = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atexit.register(self.stop)

        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        health_url = f"{self.base_url}/health"
        last_err = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                out = self._drain_output()
                raise LlamaServerError(
                    f"llama-server exited during startup (code={self._proc.returncode}).\n"
                    f"--- process output ---\n{out}"
                )
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    print("[llama_server_manager] server healthy.", file=sys.stderr)
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(1)

        self.stop()
        raise LlamaServerError(
            f"llama-server did not become healthy within {self.startup_timeout_s}s "
            f"(last error: {last_err})"
        )

    def _drain_output(self, max_chars: int = 4000) -> str:
        if not self._proc or not self._proc.stdout:
            return ""
        try:
            out = self._proc.stdout.read()
            return (out or "")[-max_chars:]
        except Exception:
            return ""

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            print("[llama_server_manager] stopping llama-server...", file=sys.stderr)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=15)
        self._proc = None

    # Context-manager sugar so callers can do:
    #   with LlamaServerManager() as srv: ...
    def __enter__(self) -> "LlamaServerManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

#!/usr/bin/env python3
"""Run independent benchmark commands under one Hunyuan GPU handoff.

The Hunyuan idle keepalive uses a single pause marker, not a reference count.
Consequently, independently wrapped benchmark processes are unsafe: the first
process to finish could resume Hunyuan while another benchmark is still using
a GPU.  This supervisor is the sole owner of the handoff for a whole group:

1. verify that the keepalive is live and not already paused;
2. pause it and wait for its bounded request wave to drain;
3. run up to four commands concurrently, with each command pinned to one
   physical GPU through ``CUDA_VISIBLE_DEVICES``;
4. wait for every child, even when another child fails; and
5. resume the keepalive exactly once from the outermost ``finally`` block.

Commands are passed as JSON argument arrays and are never interpreted by a
shell.  A group file has this form:

.. code-block:: json

    {
      "schema": "claimforge_benchmark_gpu_group_v1",
      "group_id": "balanced250-wave-1",
      "max_parallel": 4,
      "jobs": [
        {
          "id": "effort-formal",
          "gpu": 4,
          "command": ["/path/to/python", "-m", "module", "--device", "cuda:0"],
          "cwd": ".",
          "env": {"PYTHONHASHSEED": "0"}
        }
      ]
    }

Multiple jobs may name the same physical GPU; they run sequentially on that
GPU.  Jobs assigned to different GPUs may run concurrently.  The default
physical-GPU allowlist is ``4,5,6,7`` for this host.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, BinaryIO, Mapping, Protocol, Sequence
import urllib.request


REPO = Path(__file__).resolve().parents[1]
KEEPALIVE_SCRIPT = REPO / "scripts" / "hunyuan_idle_keepalive.py"
KEEPALIVE_PYTHON = Path("/usr/bin/python")
DEFAULT_STATE_ROOT = REPO / "tmp" / "benchmark_gpu_groups"
GLOBAL_SUPERVISOR_LOCK = DEFAULT_STATE_ROOT / "supervisor.lock"
DEFAULT_METRICS_URL = "http://127.0.0.1:8001/metrics"
GROUP_SCHEMA = "claimforge_benchmark_gpu_group_v1"
DEFAULT_ALLOWED_GPUS = (4, 5, 6, 7)
MAX_PARALLEL_LIMIT = 4
REQUIRED_QUEUE_METRICS = {
    "vllm_omni:num_requests_running",
    "vllm_omni:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SAFE_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
FORBIDDEN_JOB_ENV = {
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
}


class SupervisorError(RuntimeError):
    """A fail-closed group-supervision error."""


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    gpu: int
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str]


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    max_parallel: int
    jobs: tuple[JobSpec, ...]


class ChildProcess(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class Backend(Protocol):
    def keepalive_status(self) -> dict[str, Any]: ...

    def pause_keepalive(self) -> None: ...

    def resume_keepalive(self) -> None: ...

    def fetch_metrics(self) -> str: ...

    def spawn(
        self,
        job: JobSpec,
        env: Mapping[str, str],
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> ChildProcess: ...

    def terminate(self, process: ChildProcess) -> None: ...

    def kill(self, process: ChildProcess) -> None: ...

    def sleep(self, seconds: float) -> None: ...

    def monotonic(self) -> float: ...


class SystemBackend:
    """Production backend.  Tests replace this class with a pure fake."""

    def __init__(
        self,
        *,
        metrics_url: str,
        probe_timeout: float,
        logger: logging.Logger,
    ) -> None:
        self.metrics_url = metrics_url
        self.probe_timeout = probe_timeout
        self.logger = logger

    def _keepalive_command(self, command: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                str(KEEPALIVE_PYTHON),
                str(KEEPALIVE_SCRIPT),
                command,
            ],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            self.logger.info("keepalive %s stdout: %s", command, result.stdout.strip())
        if result.stderr.strip():
            self.logger.warning("keepalive %s stderr: %s", command, result.stderr.strip())
        return result

    def keepalive_status(self) -> dict[str, Any]:
        result = self._keepalive_command("status")
        if result.returncode != 0:
            raise SupervisorError(
                f"keepalive status failed with exit {result.returncode}"
            )
        try:
            status = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SupervisorError("keepalive status did not return valid JSON") from exc
        if not isinstance(status, dict):
            raise SupervisorError("keepalive status root is not an object")
        return status

    def pause_keepalive(self) -> None:
        result = self._keepalive_command("pause")
        if result.returncode != 0:
            raise SupervisorError(
                f"keepalive pause failed with exit {result.returncode}"
            )

    def resume_keepalive(self) -> None:
        result = self._keepalive_command("resume")
        if result.returncode != 0:
            raise SupervisorError(
                f"keepalive resume failed with exit {result.returncode}"
            )

    def fetch_metrics(self) -> str:
        request = urllib.request.Request(self.metrics_url, method="GET")
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.probe_timeout,
            ) as response:
                payload = response.read()
        except Exception as exc:
            raise SupervisorError(
                f"could not read service metrics from {self.metrics_url}: {exc!r}"
            ) from exc
        return payload.decode("utf-8")

    def spawn(
        self,
        job: JobSpec,
        env: Mapping[str, str],
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> ChildProcess:
        return subprocess.Popen(
            list(job.command),
            cwd=job.cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

    def terminate(self, process: ChildProcess) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def kill(self, process: ChildProcess) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass
class ActiveJob:
    job: JobSpec
    process: ChildProcess
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    started_monotonic: float

    def close_logs(self) -> None:
        self.stdout_handle.close()
        self.stderr_handle.close()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_allowed_gpus(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("allowed GPUs must be a comma-separated list")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("allowed GPUs must be integers") from exc
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("allowed GPU indices must be non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("allowed GPU indices must be unique")
    if len(values) > MAX_PARALLEL_LIMIT:
        raise argparse.ArgumentTypeError(
            f"at most {MAX_PARALLEL_LIMIT} GPUs may be allowed"
        )
    return values


def _validate_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise SupervisorError(
            f"{field} must match {SAFE_ID.pattern!r}"
        )
    return value


def _resolve_job_cwd(raw: Any) -> Path:
    if raw is None:
        return REPO
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise SupervisorError("job cwd must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    try:
        path.relative_to(REPO)
    except ValueError as exc:
        raise SupervisorError(f"job cwd escapes repository root: {path}") from exc
    if not path.is_dir():
        raise SupervisorError(f"job cwd is not a directory: {path}")
    return path


def load_group_spec(
    path: Path,
    *,
    allowed_gpus: Sequence[int] = DEFAULT_ALLOWED_GPUS,
) -> GroupSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SupervisorError(f"group file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SupervisorError(f"group file is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SupervisorError("group file root must be an object")
    if raw.get("schema") != GROUP_SCHEMA:
        raise SupervisorError(f"group schema must be {GROUP_SCHEMA!r}")

    group_id = _validate_id(raw.get("group_id"), field="group_id")
    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise SupervisorError("jobs must be a non-empty array")

    allowed = set(allowed_gpus)
    if not allowed:
        raise SupervisorError("allowed GPU set must not be empty")
    jobs: list[JobSpec] = []
    seen_ids: set[str] = set()
    for index, job_raw in enumerate(jobs_raw):
        prefix = f"jobs[{index}]"
        if not isinstance(job_raw, dict):
            raise SupervisorError(f"{prefix} must be an object")
        job_id = _validate_id(job_raw.get("id"), field=f"{prefix}.id")
        if job_id in seen_ids:
            raise SupervisorError(f"duplicate job id: {job_id}")
        seen_ids.add(job_id)

        gpu = job_raw.get("gpu")
        if isinstance(gpu, bool) or not isinstance(gpu, int):
            raise SupervisorError(f"{prefix}.gpu must be an integer")
        if gpu not in allowed:
            raise SupervisorError(
                f"{prefix}.gpu {gpu} is outside allowed GPUs {sorted(allowed)}"
            )

        command_raw = job_raw.get("command")
        if (
            not isinstance(command_raw, list)
            or not command_raw
            or any(
                not isinstance(argument, str)
                or not argument
                or "\0" in argument
                for argument in command_raw
            )
        ):
            raise SupervisorError(
                f"{prefix}.command must be a non-empty array of non-empty strings"
            )

        env_raw = job_raw.get("env", {})
        if not isinstance(env_raw, dict):
            raise SupervisorError(f"{prefix}.env must be an object")
        env: dict[str, str] = {}
        for key, value in env_raw.items():
            if not isinstance(key, str) or not SAFE_ENV_KEY.fullmatch(key):
                raise SupervisorError(f"{prefix}.env has invalid key {key!r}")
            if key in FORBIDDEN_JOB_ENV:
                raise SupervisorError(
                    f"{prefix}.env may not override supervisor-owned {key}"
                )
            if not isinstance(value, str) or "\0" in value:
                raise SupervisorError(
                    f"{prefix}.env[{key!r}] must be a string without NUL"
                )
            env[key] = value

        jobs.append(
            JobSpec(
                job_id=job_id,
                gpu=gpu,
                command=tuple(command_raw),
                cwd=_resolve_job_cwd(job_raw.get("cwd")),
                env=env,
            )
        )

    max_parallel_raw = raw.get("max_parallel", len({job.gpu for job in jobs}))
    if (
        isinstance(max_parallel_raw, bool)
        or not isinstance(max_parallel_raw, int)
        or not 1 <= max_parallel_raw <= MAX_PARALLEL_LIMIT
    ):
        raise SupervisorError(
            f"max_parallel must be an integer in [1, {MAX_PARALLEL_LIMIT}]"
        )
    if max_parallel_raw > len(allowed):
        raise SupervisorError(
            "max_parallel may not exceed the number of allowed GPUs"
        )
    return GroupSpec(
        group_id=group_id,
        max_parallel=max_parallel_raw,
        jobs=tuple(jobs),
    )


def parse_queue_metrics(text: str) -> dict[str, float]:
    totals = {name: 0.0 for name in REQUIRED_QUEUE_METRICS}
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        metric_name = fields[0].split("{", 1)[0]
        if metric_name not in REQUIRED_QUEUE_METRICS:
            continue
        try:
            value = float(fields[1])
        except ValueError as exc:
            raise SupervisorError(
                f"invalid value for service metric {metric_name}: {fields[1]!r}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise SupervisorError(
                f"invalid value for service metric {metric_name}: {value!r}"
            )
        totals[metric_name] += value
        seen.add(metric_name)
    missing = REQUIRED_QUEUE_METRICS - seen
    if missing:
        raise SupervisorError(
            f"required service metrics are missing: {sorted(missing)}"
        )
    return totals


def wait_for_service_drain(
    backend: Backend,
    *,
    timeout: float,
    poll_interval: float,
    zero_samples: int,
    logger: logging.Logger,
) -> None:
    deadline = backend.monotonic() + timeout
    consecutive_zero = 0
    while True:
        metrics = parse_queue_metrics(backend.fetch_metrics())
        running = (
            metrics["vllm_omni:num_requests_running"]
            + metrics["vllm:num_requests_running"]
        )
        waiting = (
            metrics["vllm_omni:num_requests_waiting"]
            + metrics["vllm:num_requests_waiting"]
        )
        if running == 0 and waiting == 0:
            consecutive_zero += 1
            if consecutive_zero >= zero_samples:
                logger.info(
                    "service queue drained across %s consecutive probes",
                    zero_samples,
                )
                return
        else:
            consecutive_zero = 0
            logger.info(
                "waiting for service queue: running=%.0f waiting=%.0f",
                running,
                waiting,
            )
        if backend.monotonic() >= deadline:
            raise SupervisorError(
                "timed out waiting for Hunyuan service queue to drain"
            )
        backend.sleep(poll_interval)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(
        f"benchmark_gpu_group.{log_path.parent.name}.{os.getpid()}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def initial_job_state(job: JobSpec, state_dir: Path) -> dict[str, Any]:
    return {
        "id": job.job_id,
        "gpu": job.gpu,
        "logical_device": "cuda:0",
        "command": list(job.command),
        "cwd": str(job.cwd),
        "env_keys": sorted(job.env),
        "stdout_log": str(state_dir / f"{job.job_id}.stdout.log"),
        "stderr_log": str(state_dir / f"{job.job_id}.stderr.log"),
        "state": "pending",
        "pid": None,
        "returncode": None,
        "started_at": None,
        "completed_at": None,
        "elapsed_s": None,
        "error": None,
    }


def _set_job_completed(
    row: dict[str, Any],
    *,
    returncode: int,
    state: str,
    elapsed_s: float,
) -> None:
    row["state"] = state
    row["returncode"] = returncode
    row["completed_at"] = utc_now()
    row["elapsed_s"] = round(max(0.0, elapsed_s), 3)


def _base_child_env(job: JobSpec) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CUDA_DEVICE_ORDER", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    env.update(job.env)
    # Match CUDA ordinals to nvidia-smi's PCI-bus ordering so the physical
    # GPU recorded in the group file is the one exposed as logical cuda:0.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    return env


def _stop_active_jobs(
    active: dict[str, ActiveJob],
    state_by_id: dict[str, dict[str, Any]],
    *,
    backend: Backend,
    terminate_timeout: float,
    logger: logging.Logger,
) -> None:
    if not active:
        return
    logger.warning("terminating %s active benchmark job(s)", len(active))
    for item in active.values():
        try:
            backend.terminate(item.process)
        except Exception as exc:
            logger.error("could not terminate %s: %r", item.job.job_id, exc)

    deadline = backend.monotonic() + terminate_timeout
    for job_id, item in list(active.items()):
        remaining = max(0.0, deadline - backend.monotonic())
        try:
            returncode = item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.error("job %s ignored SIGTERM; sending SIGKILL", job_id)
            try:
                backend.kill(item.process)
            finally:
                returncode = item.process.wait()
        except Exception as exc:
            logger.error("wait failed for %s: %r", job_id, exc)
            returncode = -1
        _set_job_completed(
            state_by_id[job_id],
            returncode=returncode,
            state="terminated",
            elapsed_s=backend.monotonic() - item.started_monotonic,
        )
        item.close_logs()
        active.pop(job_id, None)


def execute_jobs(
    spec: GroupSpec,
    *,
    backend: Backend,
    state: dict[str, Any],
    status_path: Path,
    state_dir: Path,
    logger: logging.Logger,
    poll_interval: float,
    terminate_timeout: float,
    interrupted_signal: list[int | None],
) -> bool:
    state_by_id = {row["id"]: row for row in state["jobs"]}
    pending = list(spec.jobs)
    active: dict[str, ActiveJob] = {}
    all_succeeded = True
    try:
        while pending or active:
            if interrupted_signal[0] is not None:
                all_succeeded = False
                for job in pending:
                    row = state_by_id[job.job_id]
                    row["state"] = "skipped"
                    row["completed_at"] = utc_now()
                    row["error"] = (
                        f"supervisor interrupted by signal {interrupted_signal[0]}"
                    )
                pending.clear()
                _stop_active_jobs(
                    active,
                    state_by_id,
                    backend=backend,
                    terminate_timeout=terminate_timeout,
                    logger=logger,
                )
                state["updated_at"] = utc_now()
                atomic_json(status_path, state)
                break

            launched_or_completed = False
            while len(active) < spec.max_parallel:
                active_gpus = {item.job.gpu for item in active.values()}
                runnable_index = next(
                    (
                        index
                        for index, job in enumerate(pending)
                        if job.gpu not in active_gpus
                    ),
                    None,
                )
                if runnable_index is None:
                    break
                job = pending.pop(runnable_index)
                row = state_by_id[job.job_id]
                stdout_path = Path(row["stdout_log"])
                stderr_path = Path(row["stderr_log"])
                stdout_handle = stdout_path.open("xb")
                stderr_handle = stderr_path.open("xb")
                try:
                    process = backend.spawn(
                        job,
                        _base_child_env(job),
                        stdout_handle,
                        stderr_handle,
                    )
                except Exception as exc:
                    stdout_handle.close()
                    stderr_handle.close()
                    all_succeeded = False
                    row["state"] = "spawn_failed"
                    row["completed_at"] = utc_now()
                    row["error"] = repr(exc)
                    logger.exception("could not start job %s", job.job_id)
                    launched_or_completed = True
                    state["updated_at"] = utc_now()
                    atomic_json(status_path, state)
                    continue
                row["state"] = "running"
                row["pid"] = process.pid
                row["started_at"] = utc_now()
                active[job.job_id] = ActiveJob(
                    job=job,
                    process=process,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    started_monotonic=backend.monotonic(),
                )
                logger.info(
                    "started job=%s pid=%s physical_gpu=%s logical_device=cuda:0",
                    job.job_id,
                    process.pid,
                    job.gpu,
                )
                launched_or_completed = True
                state["updated_at"] = utc_now()
                atomic_json(status_path, state)

            for job_id, item in list(active.items()):
                returncode = item.process.poll()
                if returncode is None:
                    continue
                succeeded = returncode == 0
                all_succeeded = all_succeeded and succeeded
                row = state_by_id[job_id]
                _set_job_completed(
                    row,
                    returncode=returncode,
                    state="succeeded" if succeeded else "failed",
                    elapsed_s=backend.monotonic() - item.started_monotonic,
                )
                item.close_logs()
                active.pop(job_id)
                logger.info(
                    "completed job=%s returncode=%s",
                    job_id,
                    returncode,
                )
                launched_or_completed = True
                state["updated_at"] = utc_now()
                atomic_json(status_path, state)

            if (pending or active) and not launched_or_completed:
                backend.sleep(poll_interval)
    except BaseException:
        _stop_active_jobs(
            active,
            state_by_id,
            backend=backend,
            terminate_timeout=terminate_timeout,
            logger=logger,
        )
        raise
    return all_succeeded


def run_group(
    spec: GroupSpec,
    *,
    state_root: Path,
    lock_path: Path = GLOBAL_SUPERVISOR_LOCK,
    backend_factory: Any = SystemBackend,
    metrics_url: str = DEFAULT_METRICS_URL,
    probe_timeout: float = 5.0,
    drain_timeout: float = 300.0,
    drain_poll_interval: float = 0.25,
    drain_zero_samples: int = 3,
    child_poll_interval: float = 0.2,
    terminate_timeout: float = 30.0,
    install_signal_handlers: bool = True,
) -> int:
    state_root.mkdir(parents=True, exist_ok=True)
    # This lock is intentionally independent of ``state_root``. Otherwise two
    # callers could select different log roots and both believe they own the
    # keepalive's process-global pause marker.
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise SupervisorError(
            f"another GPU benchmark supervisor holds {lock_path}"
        )

    state_dir = state_root / spec.group_id
    try:
        state_dir.mkdir()
    except FileExistsError as exc:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        raise SupervisorError(
            f"group state directory already exists: {state_dir}"
        ) from exc

    logger = setup_logger(state_dir / "supervisor.log")
    status_path = state_dir / "status.json"
    state: dict[str, Any] = {
        "schema": GROUP_SCHEMA,
        "group_id": spec.group_id,
        "state": "preflight",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_at": None,
        "max_parallel": spec.max_parallel,
        "keepalive": {
            "initially_running": None,
            "initially_paused": None,
            "pause_requested": False,
            "pause_confirmed": False,
            "queue_drained": False,
            "resume_attempted": False,
            "resume_succeeded": None,
            "resume_error": None,
        },
        "interrupted_signal": None,
        "error": None,
        "jobs": [
            initial_job_state(job, state_dir)
            for job in spec.jobs
        ],
    }
    atomic_json(status_path, state)
    backend: Backend = backend_factory(
        metrics_url=metrics_url,
        probe_timeout=probe_timeout,
        logger=logger,
    )
    pause_requested = False
    infrastructure_error: BaseException | None = None
    jobs_succeeded = False
    interrupted_signal: list[int | None] = [None]
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        if interrupted_signal[0] is None:
            interrupted_signal[0] = signum
            logger.warning("received signal %s; stopping benchmark children", signum)

    if install_signal_handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    try:
        initial_status = backend.keepalive_status()
        initially_running = initial_status.get("running") is True
        initially_paused = initial_status.get("paused") is True
        state["keepalive"]["initially_running"] = initially_running
        state["keepalive"]["initially_paused"] = initially_paused
        state["updated_at"] = utc_now()
        atomic_json(status_path, state)
        if not initially_running:
            raise SupervisorError(
                "Hunyuan keepalive is not running; refusing a GPU handoff"
            )
        if initially_paused:
            raise SupervisorError(
                "Hunyuan keepalive was already paused; ownership is ambiguous"
            )

        state["state"] = "pausing"
        state["keepalive"]["pause_requested"] = True
        state["updated_at"] = utc_now()
        atomic_json(status_path, state)
        # Once this transition is requested, the outer finally owns exactly one
        # resume attempt, even when pause confirmation or drain later fails.
        pause_requested = True
        backend.pause_keepalive()

        paused_status = backend.keepalive_status()
        if paused_status.get("running") is not True:
            raise SupervisorError(
                "keepalive stopped while the benchmark group was acquiring it"
            )
        if paused_status.get("paused") is not True:
            raise SupervisorError("keepalive pause marker was not confirmed")
        state["keepalive"]["pause_confirmed"] = True
        state["state"] = "draining"
        state["updated_at"] = utc_now()
        atomic_json(status_path, state)

        wait_for_service_drain(
            backend,
            timeout=drain_timeout,
            poll_interval=drain_poll_interval,
            zero_samples=drain_zero_samples,
            logger=logger,
        )
        state["keepalive"]["queue_drained"] = True
        state["state"] = "running"
        state["updated_at"] = utc_now()
        atomic_json(status_path, state)

        jobs_succeeded = execute_jobs(
            spec,
            backend=backend,
            state=state,
            status_path=status_path,
            state_dir=state_dir,
            logger=logger,
            poll_interval=child_poll_interval,
            terminate_timeout=terminate_timeout,
            interrupted_signal=interrupted_signal,
        )
    except BaseException as exc:
        infrastructure_error = exc
        state["error"] = repr(exc)
        logger.exception("GPU benchmark group failed")
    finally:
        state["interrupted_signal"] = interrupted_signal[0]
        # This is the only resume call site. execute_jobs does not return while
        # a child is active, and its exception path terminates and waits first.
        if pause_requested:
            state["keepalive"]["resume_attempted"] = True
            state["updated_at"] = utc_now()
            atomic_json(status_path, state)
            try:
                backend.resume_keepalive()
            except BaseException as exc:
                state["keepalive"]["resume_succeeded"] = False
                state["keepalive"]["resume_error"] = repr(exc)
                logger.exception("could not resume Hunyuan keepalive")
                if infrastructure_error is None:
                    infrastructure_error = exc
            else:
                state["keepalive"]["resume_succeeded"] = True
                logger.info("Hunyuan keepalive resume requested")

        # Keep our non-terminating handlers installed until after the resume
        # attempt. Restoring SIGTERM's default action earlier would create a
        # small but real window in which this supervisor could die paused.
        if install_signal_handlers:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)

        if infrastructure_error is not None:
            state["state"] = "supervisor_failed"
        elif interrupted_signal[0] is not None:
            state["state"] = "interrupted"
        elif jobs_succeeded:
            state["state"] = "succeeded"
        else:
            state["state"] = "failed"
        state["completed_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_json(status_path, state)
        close_logger(logger)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    if infrastructure_error is not None:
        return 2
    if interrupted_signal[0] is not None:
        return 128 + int(interrupted_signal[0])
    return 0 if jobs_succeeded else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-file",
        type=Path,
        required=True,
        help="JSON group definition; commands are argument arrays, never shell text",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_STATE_ROOT,
        help="gitignored directory for immutable group status and child logs",
    )
    parser.add_argument(
        "--allowed-gpus",
        type=parse_allowed_gpus,
        default=DEFAULT_ALLOWED_GPUS,
        help="comma-separated physical GPU allowlist (default: 4,5,6,7)",
    )
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL)
    parser.add_argument("--probe-timeout", type=float, default=5.0)
    parser.add_argument("--drain-timeout", type=float, default=300.0)
    parser.add_argument("--drain-poll-interval", type=float, default=0.25)
    parser.add_argument("--drain-zero-samples", type=int, default=3)
    parser.add_argument("--child-poll-interval", type=float, default=0.2)
    parser.add_argument("--terminate-timeout", type=float, default=30.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "probe_timeout": args.probe_timeout,
        "drain_timeout": args.drain_timeout,
        "drain_poll_interval": args.drain_poll_interval,
        "child_poll_interval": args.child_poll_interval,
        "terminate_timeout": args.terminate_timeout,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise SupervisorError(f"{name} must be finite and positive")
    if args.drain_zero_samples < 1:
        raise SupervisorError("drain_zero_samples must be at least one")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        spec = load_group_spec(
            args.group_file.resolve(),
            allowed_gpus=args.allowed_gpus,
        )
        return run_group(
            spec,
            state_root=args.state_root.resolve(),
            metrics_url=args.metrics_url,
            probe_timeout=args.probe_timeout,
            drain_timeout=args.drain_timeout,
            drain_poll_interval=args.drain_poll_interval,
            drain_zero_samples=args.drain_zero_samples,
            child_poll_interval=args.child_poll_interval,
            terminate_timeout=args.terminate_timeout,
        )
    except SupervisorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Keep HunyuanImage busy with one low-priority image request at a time.

The worker submits a request only after the vLLM-Omni aggregate and per-stage
running/waiting metrics have all stayed at zero for a short grace period.  It
also yields while the repository's foreground batch client is running.

Only the most recent image is retained under ``tmp/hunyuan_idle_keepalive``.
Runtime state lives under the same gitignored directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any

import requests
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_hunyuan_generation import (  # noqa: E402
    call_edit_omni,
    make_prompt,
    position_phrase,
    upscale_size,
)


STATE_DIR = REPO / "tmp" / "hunyuan_idle_keepalive"
LOCK_PATH = STATE_DIR / "worker.lock"
CONTROL_LOCK_PATH = STATE_DIR / "control.lock"
ADMISSION_LOCK_PATH = STATE_DIR / "admission.lock"
PID_PATH = STATE_DIR / "worker.json"
STATUS_PATH = STATE_DIR / "status.json"
PAUSE_PATH = STATE_DIR / "paused"
LATEST_PATH = STATE_DIR / "latest.png"
LATEST_META_PATH = STATE_DIR / "latest.json"
LATEST_PART_PATH = STATE_DIR / "latest.part"
LOG_PATH = STATE_DIR / "keepalive.log"

TMUX_SESSION = "hunyuan-idle"
VENV_PYTHON = Path("/root/HunyuanImage-3.0/.venv/bin/python")
API_ROOT = "http://127.0.0.1:8001"
EDIT_URL = f"{API_ROOT}/v1/images/edits"
METRICS_URL = f"{API_ROOT}/metrics"
HEALTH_URL = f"{API_ROOT}/health"
MODEL_NAME = "vllm_hunyuan_image3"

METRIC_NAMES = {
    "vllm_omni:num_requests_running",
    "vllm_omni:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}
FOREGROUND_CLIENTS = {"run_hunyuan_generation.py"}
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def proc_starttime(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    # Fields after the final ')' start at field 3. starttime is field 22.
    fields_after_comm = text.rsplit(")", 1)[1].split()
    return fields_after_comm[19] if len(fields_after_comm) > 19 else None


def proc_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def worker_identity() -> dict[str, Any] | None:
    try:
        info = json.loads(PID_PATH.read_text(encoding="utf-8"))
        pid = int(info["pid"])
        expected_starttime = str(info["starttime"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    argv = proc_argv(pid)
    if proc_starttime(pid) != expected_starttime:
        return None
    if not any(Path(arg).name == Path(__file__).name for arg in argv):
        return None
    if "run" not in argv:
        return None
    return {**info, "argv": argv}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("hunyuan_idle_keepalive")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    rotating = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)
    return logger


def write_status(state: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(extra)
    atomic_json(STATUS_PATH, payload)


def parse_metrics(text: str) -> dict[str, float]:
    totals = {name: 0.0 for name in METRIC_NAMES}
    seen: set[str] = set()
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            metric_name = sample.name
            if metric_name not in METRIC_NAMES:
                continue
            value = float(sample.value)
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"invalid value for metric {metric_name}: {value!r}")
            totals[metric_name] += value
            seen.add(metric_name)
    missing = METRIC_NAMES - seen
    if missing:
        raise RuntimeError(f"required metrics missing: {sorted(missing)}")
    return totals


def foreground_clients() -> list[dict[str, Any]]:
    clients: list[dict[str, Any]] = []
    self_pid = os.getpid()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == self_pid:
            continue
        argv = proc_argv(pid)
        if any(Path(arg).name in FOREGROUND_CLIENTS for arg in argv):
            clients.append({"pid": pid, "argv": argv})
    return clients


def interruptible_sleep(stop_requested: list[bool], seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while not stop_requested[0]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def load_keepalive_input(task_index: int) -> tuple[Image.Image, str, dict[str, Any]]:
    tasks_path = REPO / "annotations" / "generation_tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line]
    try:
        task = tasks[task_index]
    except IndexError as exc:
        raise RuntimeError(f"task index {task_index} is out of range for {tasks_path}") from exc

    crop = Image.open(REPO / task["context_crop"]).convert("RGB")
    width, height = crop.size
    target_width, target_height = upscale_size(width, height)
    upscaled = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)
    box = [int(value) for value in task["edit_region_in_context_xyxy"]]
    prompt = make_prompt(task, position_phrase(box, crop.size), "object")
    input_meta = {
        "task_id": task["task_id"],
        "input_context_crop": task["context_crop"],
        "input_size": [width, height],
        "request_size": [target_width, target_height],
        "prompt": prompt,
    }
    return upscaled, prompt, input_meta


def run_worker(args: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)

    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another Hunyuan idle keepalive worker already holds the lock", file=sys.stderr)
        return 2
    LATEST_PART_PATH.unlink(missing_ok=True)

    starttime = proc_starttime(os.getpid())
    if starttime is None:
        raise RuntimeError("could not read worker process starttime")
    logger = setup_logging()
    stop_requested = [False]

    def request_stop(signum: int, _frame: Any) -> None:
        stop_requested[0] = True
        logger.info("stop requested by signal %s; finishing the current request", signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    session = requests.Session()
    session.trust_env = False
    image, prompt, input_meta = load_keepalive_input(args.task_index)
    atomic_json(
        PID_PATH,
        {
            "pid": os.getpid(),
            "starttime": starttime,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    successes = 0
    failures = 0
    idle_since: float | None = None
    backoff = args.error_backoff
    last_reported_state: str | None = None
    last_status_state: str | None = None
    last_status_write = 0.0

    def publish_status(state: str, *, force: bool = False, **extra: Any) -> None:
        nonlocal last_status_state, last_status_write
        now = time.monotonic()
        if not force and state == last_status_state and now - last_status_write < 10.0:
            return
        write_status(state, **extra)
        last_status_state = state
        last_status_write = now

    logger.info(
        "worker started pid=%s task=%s request_size=%sx%s steps=%s idle_grace=%.1fs",
        os.getpid(),
        input_meta["task_id"],
        *input_meta["request_size"],
        args.steps,
        args.idle_grace,
    )

    try:
        while not stop_requested[0]:
            if PAUSE_PATH.exists():
                idle_since = None
                if last_reported_state != "paused":
                    logger.info("paused by marker %s", PAUSE_PATH)
                    last_reported_state = "paused"
                publish_status("paused", successes=successes, failures=failures)
                interruptible_sleep(stop_requested, args.poll_interval)
                continue

            clients = foreground_clients()
            if clients:
                idle_since = None
                state = "foreground_client"
                if last_reported_state != state:
                    logger.info("yielding to foreground generation client(s): %s", [c["pid"] for c in clients])
                    last_reported_state = state
                publish_status(
                    "busy",
                    reason=state,
                    foreground_pids=[client["pid"] for client in clients],
                    successes=successes,
                    failures=failures,
                )
                interruptible_sleep(stop_requested, args.poll_interval)
                continue

            try:
                health = session.get(HEALTH_URL, timeout=args.probe_timeout)
                health.raise_for_status()
                metrics_response = session.get(METRICS_URL, timeout=args.probe_timeout)
                metrics_response.raise_for_status()
                metrics = parse_metrics(metrics_response.text)
            except Exception as exc:
                idle_since = None
                failures += 1
                logger.warning("service probe failed: %r; retrying in %.1fs", exc, backoff)
                publish_status(
                    "backoff",
                    force=True,
                    reason="probe_failed",
                    error=repr(exc),
                    retry_in_s=backoff,
                    successes=successes,
                    failures=failures,
                )
                interruptible_sleep(stop_requested, backoff)
                backoff = min(args.max_backoff, max(args.error_backoff, backoff * 2))
                continue

            running = metrics["vllm_omni:num_requests_running"] + metrics["vllm:num_requests_running"]
            waiting = metrics["vllm_omni:num_requests_waiting"] + metrics["vllm:num_requests_waiting"]
            if running > 0 or waiting > 0:
                idle_since = None
                state = "service_busy"
                if last_reported_state != state:
                    logger.info("yielding to service queue: running=%.0f waiting=%.0f", running, waiting)
                    last_reported_state = state
                publish_status(
                    "busy",
                    reason=state,
                    running=running,
                    waiting=waiting,
                    successes=successes,
                    failures=failures,
                )
                interruptible_sleep(stop_requested, args.poll_interval)
                continue

            now = time.monotonic()
            if idle_since is None:
                idle_since = now
            idle_for = now - idle_since
            if idle_for < args.idle_grace:
                state = "idle_grace"
                if last_reported_state != state:
                    logger.info("service idle; waiting %.1fs grace period", args.idle_grace)
                    last_reported_state = state
                publish_status(
                    "waiting",
                    reason=state,
                    idle_for_s=round(idle_for, 3),
                    successes=successes,
                    failures=failures,
                )
                interruptible_sleep(stop_requested, args.poll_interval)
                continue

            # Keep the admission lock for the whole image. A pause command takes
            # the same lock before creating its marker, so when pause returns no
            # keepalive image is in flight and none can slip through afterward.
            generation_backoff: float | None = None
            with exclusive_file_lock(ADMISSION_LOCK_PATH):
                # Recheck immediately before admission. Metrics still have a
                # small unavoidable check-to-POST race with unrelated clients,
                # but at most one keepalive image can run and none is pre-queued.
                clients = foreground_clients()
                if clients or stop_requested[0] or PAUSE_PATH.exists():
                    idle_since = None
                    continue

                iteration = successes + failures + 1
                seed = random.SystemRandom().randint(1, 9_000_000)
                publish_status(
                    "generating",
                    force=True,
                    iteration=iteration,
                    seed=seed,
                    successes=successes,
                    failures=failures,
                )
                logger.info("generation %s started seed=%s", iteration, seed)
                started = time.monotonic()
                try:
                    output = call_edit_omni(
                        EDIT_URL,
                        MODEL_NAME,
                        image,
                        prompt,
                        image.width,
                        image.height,
                        args.steps,
                        seed,
                        bot_task="think",
                        sys_type="en_unified",
                        timeout=args.request_timeout,
                    )
                    output.save(LATEST_PART_PATH, format="PNG")
                    with Image.open(LATEST_PART_PATH) as check:
                        check.verify()
                    os.replace(LATEST_PART_PATH, LATEST_PATH)
                    latency = time.monotonic() - started
                    successes += 1
                    backoff = args.error_backoff
                    idle_since = None
                    last_reported_state = "generated"
                    latest_meta = {
                        **input_meta,
                        "iteration": iteration,
                        "seed": seed,
                        "steps": args.steps,
                        "latency_s": round(latency, 3),
                        "output": str(LATEST_PATH.relative_to(REPO)),
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    atomic_json(LATEST_META_PATH, latest_meta)
                    publish_status(
                        "generated",
                        force=True,
                        iteration=iteration,
                        seed=seed,
                        latency_s=round(latency, 3),
                        latest=str(LATEST_PATH),
                        successes=successes,
                        failures=failures,
                    )
                    logger.info(
                        "generation %s completed in %.2fs; latest=%s",
                        iteration,
                        latency,
                        LATEST_PATH,
                    )
                except Exception as exc:
                    LATEST_PART_PATH.unlink(missing_ok=True)
                    failures += 1
                    idle_since = None
                    logger.exception("generation %s failed; retrying in %.1fs", iteration, backoff)
                    publish_status(
                        "backoff",
                        force=True,
                        reason="generation_failed",
                        error=repr(exc),
                        retry_in_s=backoff,
                        successes=successes,
                        failures=failures,
                    )
                    generation_backoff = backoff
                    backoff = min(args.max_backoff, max(args.error_backoff, backoff * 2))
            if generation_backoff is not None:
                interruptible_sleep(stop_requested, generation_backoff)
    finally:
        try:
            LATEST_PART_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            logger.info("worker stopped; successes=%s failures=%s", successes, failures)
        except Exception:
            pass
        try:
            write_status("stopped", successes=successes, failures=failures)
        except OSError:
            pass
        try:
            try:
                info = json.loads(PID_PATH.read_text(encoding="utf-8"))
                if int(info.get("pid", -1)) == os.getpid() and str(info.get("starttime")) == starttime:
                    PID_PATH.unlink(missing_ok=True)
            except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
                pass
        except OSError:
            pass
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()
    return 0


def tmux_exists() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def tmux_session_dead() -> bool:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", TMUX_SESSION, "-F", "#{pane_dead}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return result.returncode == 0 and bool(states) and all(state == "1" for state in states)


def remove_tmux_session() -> None:
    if tmux_exists():
        subprocess.run(
            ["tmux", "kill-session", "-t", TMUX_SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def command_supervise(_args: argparse.Namespace) -> int:
    """Restart an unexpectedly failed worker with bounded backoff."""
    python = VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable)
    command = [str(python), str(Path(__file__).resolve()), "run"]
    stop_requested = [False]
    child: subprocess.Popen[Any] | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        stop_requested[0] = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    backoff = 5.0
    while not stop_requested[0]:
        child = subprocess.Popen(command)
        returncode = child.wait()
        child = None
        if stop_requested[0] or returncode == 0:
            return 0
        print(
            f"idle keepalive worker exited with status {returncode}; "
            f"restarting in {backoff:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        deadline = time.monotonic() + backoff
        while not stop_requested[0] and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
        backoff = min(300.0, backoff * 2)
    return 0


def _command_start_locked(_args: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if worker_identity():
        print("Hunyuan idle keepalive is already running")
        return 0
    if tmux_exists() and tmux_session_dead():
        remove_tmux_session()
    if tmux_exists():
        print(
            f"tmux session {TMUX_SESSION!r} exists without a live worker; "
            f"inspect or remove it before starting",
            file=sys.stderr,
        )
        return 2
    python = VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable)
    command = f"{python} {Path(__file__).resolve()} supervise"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", str(REPO), command],
        check=True,
    )
    subprocess.run(["tmux", "set-option", "-t", TMUX_SESSION, "remain-on-exit", "on"], check=True)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if worker_identity():
            print(f"started Hunyuan idle keepalive in tmux session {TMUX_SESSION}")
            return 0
        time.sleep(0.2)
    remove_tmux_session()
    print("tmux session started but worker did not publish a live PID", file=sys.stderr)
    return 1


def command_start(args: argparse.Namespace) -> int:
    with exclusive_file_lock(CONTROL_LOCK_PATH):
        return _command_start_locked(args)


def command_pause(_args: argparse.Namespace) -> int:
    with exclusive_file_lock(CONTROL_LOCK_PATH):
        with exclusive_file_lock(ADMISSION_LOCK_PATH):
            PAUSE_PATH.touch(exist_ok=True)
    print(f"paused via {PAUSE_PATH}; no keepalive image is in flight")
    return 0


def command_resume(_args: argparse.Namespace) -> int:
    with exclusive_file_lock(CONTROL_LOCK_PATH):
        PAUSE_PATH.unlink(missing_ok=True)
    print("resume requested")
    return 0


def _command_stop_locked(args: argparse.Namespace) -> int:
    # Serialize against image admission. If an image is active, this waits for
    # it to finish; after the lock is acquired the worker cannot start another
    # image before it receives SIGTERM.
    with exclusive_file_lock(ADMISSION_LOCK_PATH):
        identity = worker_identity()
        if not identity:
            if tmux_exists():
                remove_tmux_session()
                print("stopped idle keepalive supervisor/session")
            else:
                print("Hunyuan idle keepalive is not running")
            return 0
        pid = int(identity["pid"])
        starttime = str(identity["starttime"])
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            remove_tmux_session()
            print("Hunyuan idle keepalive exited before the stop signal was sent")
            return 0
        try:
            # Revalidate after acquiring a stable reference to this process.
            current = worker_identity()
            if (
                not current
                or int(current["pid"]) != pid
                or str(current["starttime"]) != starttime
                or proc_starttime(pid) != starttime
            ):
                remove_tmux_session()
                print("Hunyuan idle keepalive exited before the stop signal was sent")
                return 0
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            except ProcessLookupError:
                remove_tmux_session()
                print("Hunyuan idle keepalive exited before the stop signal was sent")
                return 0
        finally:
            os.close(pidfd)
    print(f"stop requested for worker pid {pid}; waiting for the current image to finish")
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        if not worker_identity():
            remove_tmux_session()
            print("worker stopped")
            return 0
        time.sleep(0.5)
    print(f"worker pid {pid} is still stopping; inspect tmux session {TMUX_SESSION}", file=sys.stderr)
    return 1


def command_stop(args: argparse.Namespace) -> int:
    with exclusive_file_lock(CONTROL_LOCK_PATH):
        return _command_stop_locked(args)


def command_status(_args: argparse.Namespace) -> int:
    identity = worker_identity()
    status: dict[str, Any] = {}
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    payload = {
        "running": bool(identity),
        "paused": PAUSE_PATH.exists(),
        "tmux_session": tmux_exists(),
        "worker": identity,
        "status": status,
        "latest": str(LATEST_PATH) if LATEST_PATH.is_file() else None,
        "log": str(LOG_PATH),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if identity else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the foreground worker")
    run.add_argument("--task-index", type=int, default=0)
    run.add_argument("--steps", type=int, default=8)
    run.add_argument("--idle-grace", type=float, default=2.0)
    run.add_argument("--poll-interval", type=float, default=0.5)
    run.add_argument("--probe-timeout", type=float, default=5.0)
    run.add_argument("--request-timeout", type=float, default=900.0)
    run.add_argument("--error-backoff", type=float, default=5.0)
    run.add_argument("--max-backoff", type=float, default=300.0)
    run.set_defaults(func=run_worker)

    supervise = subparsers.add_parser("supervise", help=argparse.SUPPRESS)
    supervise.set_defaults(func=command_supervise)

    start = subparsers.add_parser("start", help="start the worker in a persistent tmux session")
    start.set_defaults(func=command_start)

    pause = subparsers.add_parser("pause", help="pause after any in-flight image finishes")
    pause.set_defaults(func=command_pause)

    resume = subparsers.add_parser("resume", help="remove the pause marker")
    resume.set_defaults(func=command_resume)

    stop = subparsers.add_parser("stop", help="gracefully stop the worker")
    stop.add_argument("--wait", type=float, default=60.0)
    stop.set_defaults(func=command_stop)

    status = subparsers.add_parser("status", help="show worker and latest-image state")
    status.set_defaults(func=command_status)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

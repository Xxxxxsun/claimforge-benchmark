from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, BinaryIO, Mapping

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_benchmark_gpu_group as group_runner


def metrics_text(*, running: int = 0, waiting: int = 0) -> str:
    return "\n".join(
        [
            "# HELP test synthetic metrics",
            f'vllm_omni:num_requests_running{{stage="0"}} {running}',
            'vllm_omni:num_requests_running{stage="1"} 0',
            f"vllm_omni:num_requests_waiting {waiting}",
            "vllm:num_requests_running 0",
            "vllm:num_requests_waiting 0",
            "",
        ]
    )


class FakeProcess:
    next_pid = 41000

    def __init__(
        self,
        *,
        job_id: str,
        returncode: int,
        polls: int,
        events: list[tuple[Any, ...]],
        raise_on_poll: bool = False,
    ) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.job_id = job_id
        self.desired_returncode = returncode
        self.remaining_polls = polls
        self.returncode: int | None = None
        self.events = events
        self.raise_on_poll = raise_on_poll
        self.terminate_requested = False
        self.kill_requested = False

    def poll(self) -> int | None:
        if self.raise_on_poll:
            self.raise_on_poll = False
            raise RuntimeError(f"poll failed for {self.job_id}")
        if self.returncode is not None:
            return self.returncode
        if self.remaining_polls > 0:
            self.remaining_polls -= 1
            return None
        self.returncode = self.desired_returncode
        self.events.append(("exit", self.job_id, self.returncode))
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            if self.kill_requested:
                self.returncode = -9
            elif self.terminate_requested:
                self.returncode = -15
            elif timeout == 0:
                raise subprocess.TimeoutExpired(self.job_id, timeout)
            else:
                self.returncode = self.desired_returncode
            self.events.append(("wait", self.job_id, self.returncode))
        return self.returncode


class FakeBackend:
    def __init__(
        self,
        *,
        initial_paused: bool = False,
        metric_payloads: list[str] | None = None,
        metrics_error: Exception | None = None,
        processes: Mapping[str, tuple[int, int]] | None = None,
        poll_error_job: str | None = None,
    ) -> None:
        self.initial_paused = initial_paused
        self.metric_payloads = list(metric_payloads or [metrics_text()])
        self.metrics_error = metrics_error
        self.processes = dict(processes or {})
        self.poll_error_job = poll_error_job
        self.events: list[tuple[Any, ...]] = []
        self.spawn_envs: dict[str, dict[str, str]] = {}
        self.spawn_alive_counts: dict[str, int] = {}
        self.children: list[FakeProcess] = []
        self.status_calls = 0
        self.clock = 100.0

    def keepalive_status(self) -> dict[str, Any]:
        self.status_calls += 1
        paused = self.initial_paused if self.status_calls == 1 else True
        self.events.append(("status", paused))
        return {"running": True, "paused": paused}

    def pause_keepalive(self) -> None:
        self.events.append(("pause",))

    def resume_keepalive(self) -> None:
        assert all(child.returncode is not None for child in self.children)
        self.events.append(("resume",))

    def fetch_metrics(self) -> str:
        self.events.append(("metrics",))
        if self.metrics_error is not None:
            raise self.metrics_error
        if len(self.metric_payloads) > 1:
            return self.metric_payloads.pop(0)
        return self.metric_payloads[0]

    def spawn(
        self,
        job: group_runner.JobSpec,
        env: Mapping[str, str],
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> FakeProcess:
        alive = sum(child.returncode is None for child in self.children)
        self.spawn_alive_counts[job.job_id] = alive
        self.spawn_envs[job.job_id] = dict(env)
        returncode, polls = self.processes.get(job.job_id, (0, 0))
        process = FakeProcess(
            job_id=job.job_id,
            returncode=returncode,
            polls=polls,
            events=self.events,
            raise_on_poll=job.job_id == self.poll_error_job,
        )
        self.children.append(process)
        stdout.write(f"stdout {job.job_id}\n".encode())
        stderr.write(f"stderr {job.job_id}\n".encode())
        self.events.append(("spawn", job.job_id, job.gpu))
        return process

    def terminate(self, process: FakeProcess) -> None:
        process.terminate_requested = True
        self.events.append(("terminate", process.job_id))

    def kill(self, process: FakeProcess) -> None:
        process.kill_requested = True
        self.events.append(("kill", process.job_id))

    def sleep(self, seconds: float) -> None:
        self.clock += seconds
        self.events.append(("sleep", seconds))

    def monotonic(self) -> float:
        return self.clock


def make_job(job_id: str, gpu: int) -> group_runner.JobSpec:
    return group_runner.JobSpec(
        job_id=job_id,
        gpu=gpu,
        command=(f"/fake/{job_id}", "--device", "cuda:0"),
        cwd=REPO,
        env={"PYTHONHASHSEED": "0"},
    )


def run_with_fake(
    tmp_path: Path,
    spec: group_runner.GroupSpec,
    backend: FakeBackend,
) -> int:
    return group_runner.run_group(
        spec,
        state_root=tmp_path / "state",
        lock_path=tmp_path / "state" / "supervisor.lock",
        backend_factory=lambda **_kwargs: backend,
        drain_timeout=5.0,
        drain_poll_interval=0.01,
        drain_zero_samples=2,
        child_poll_interval=0.01,
        terminate_timeout=1.0,
        install_signal_handlers=False,
    )


def test_parse_queue_metrics_sums_labeled_series_and_fails_closed():
    parsed = group_runner.parse_queue_metrics(
        metrics_text(running=2, waiting=3)
    )
    assert parsed["vllm_omni:num_requests_running"] == 2
    assert parsed["vllm_omni:num_requests_waiting"] == 3

    with pytest.raises(group_runner.SupervisorError, match="missing"):
        group_runner.parse_queue_metrics("vllm:num_requests_running 0\n")
    with pytest.raises(group_runner.SupervisorError, match="invalid value"):
        group_runner.parse_queue_metrics(
            metrics_text().replace(
                "vllm:num_requests_waiting 0",
                "vllm:num_requests_waiting NaN",
            )
        )


def test_load_group_spec_validates_gpu_and_supervisor_owned_environment(
    tmp_path: Path,
):
    path = tmp_path / "group.json"
    base = {
        "schema": group_runner.GROUP_SCHEMA,
        "group_id": "wave-1",
        "max_parallel": 2,
        "jobs": [
            {
                "id": "a",
                "gpu": 4,
                "command": ["/fake/python", "-m", "fake"],
                "cwd": ".",
            }
        ],
    }
    path.write_text(json.dumps(base), encoding="utf-8")
    spec = group_runner.load_group_spec(path)
    assert spec.jobs[0].gpu == 4
    assert spec.jobs[0].cwd == REPO

    base["jobs"][0]["gpu"] = 0
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(group_runner.SupervisorError, match="outside allowed"):
        group_runner.load_group_spec(path)

    base["jobs"][0]["gpu"] = 4
    base["jobs"][0]["env"] = {"CUDA_VISIBLE_DEVICES": "7"}
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(group_runner.SupervisorError, match="may not override"):
        group_runner.load_group_spec(path)


def test_group_runs_distinct_gpus_concurrently_continues_after_failure_and_resumes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    spec = group_runner.GroupSpec(
        group_id="parallel-failure",
        max_parallel=2,
        jobs=(
            make_job("slow-a", 4),
            make_job("fail-b", 5),
            make_job("after-b", 5),
        ),
    )
    backend = FakeBackend(
        metric_payloads=[
            metrics_text(running=1),
            metrics_text(),
            metrics_text(),
        ],
        processes={
            "slow-a": (0, 4),
            "fail-b": (7, 0),
            "after-b": (0, 1),
        },
    )

    returncode = run_with_fake(tmp_path, spec, backend)

    assert returncode == 1
    assert [event for event in backend.events if event[0] == "pause"] == [
        ("pause",)
    ]
    assert [event for event in backend.events if event[0] == "resume"] == [
        ("resume",)
    ]
    assert backend.spawn_alive_counts["slow-a"] == 0
    assert backend.spawn_alive_counts["fail-b"] == 1
    # fail-b freed GPU 5 while slow-a was still active; after-b filled it.
    assert backend.spawn_alive_counts["after-b"] == 1
    assert backend.spawn_envs["slow-a"]["CUDA_VISIBLE_DEVICES"] == "4"
    assert backend.spawn_envs["slow-a"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert backend.spawn_envs["fail-b"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert backend.spawn_envs["after-b"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert "NVIDIA_VISIBLE_DEVICES" not in backend.spawn_envs["slow-a"]
    assert backend.events.index(("exit", "slow-a", 0)) < backend.events.index(
        ("resume",)
    )
    assert backend.events.index(("exit", "after-b", 0)) < backend.events.index(
        ("resume",)
    )

    state_dir = tmp_path / "state" / spec.group_id
    status = json.loads((state_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["keepalive"]["queue_drained"] is True
    assert status["keepalive"]["resume_attempted"] is True
    assert status["keepalive"]["resume_succeeded"] is True
    assert {row["id"]: row["state"] for row in status["jobs"]} == {
        "slow-a": "succeeded",
        "fail-b": "failed",
        "after-b": "succeeded",
    }
    for job in spec.jobs:
        assert (state_dir / f"{job.job_id}.stdout.log").is_file()
        assert (state_dir / f"{job.job_id}.stderr.log").is_file()
    assert (state_dir / "supervisor.log").is_file()


def test_drain_failure_launches_no_job_but_resumes_once(tmp_path: Path):
    spec = group_runner.GroupSpec(
        group_id="drain-failure",
        max_parallel=1,
        jobs=(make_job("never-started", 4),),
    )
    backend = FakeBackend(metrics_error=RuntimeError("metrics unavailable"))

    returncode = run_with_fake(tmp_path, spec, backend)

    assert returncode == 2
    assert not backend.children
    assert [event for event in backend.events if event[0] == "pause"] == [
        ("pause",)
    ]
    assert [event for event in backend.events if event[0] == "resume"] == [
        ("resume",)
    ]
    status = json.loads(
        (
            tmp_path
            / "state"
            / spec.group_id
            / "status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["state"] == "supervisor_failed"
    assert status["jobs"][0]["state"] == "pending"
    assert status["keepalive"]["resume_succeeded"] is True


def test_child_supervision_exception_terminates_and_waits_before_resume(
    tmp_path: Path,
):
    spec = group_runner.GroupSpec(
        group_id="poll-failure",
        max_parallel=2,
        jobs=(make_job("broken-poll", 4), make_job("companion", 5)),
    )
    backend = FakeBackend(
        processes={
            "broken-poll": (0, 10),
            "companion": (0, 10),
        },
        poll_error_job="broken-poll",
    )

    returncode = run_with_fake(tmp_path, spec, backend)

    assert returncode == 2
    resume_index = backend.events.index(("resume",))
    assert backend.events.index(("terminate", "broken-poll")) < resume_index
    assert backend.events.index(("terminate", "companion")) < resume_index
    assert backend.events.index(("wait", "broken-poll", -15)) < resume_index
    assert backend.events.index(("wait", "companion", -15)) < resume_index
    assert [event for event in backend.events if event[0] == "resume"] == [
        ("resume",)
    ]


def test_preexisting_pause_is_not_stolen_or_resumed(tmp_path: Path):
    spec = group_runner.GroupSpec(
        group_id="already-paused",
        max_parallel=1,
        jobs=(make_job("never-started", 4),),
    )
    backend = FakeBackend(initial_paused=True)

    returncode = run_with_fake(tmp_path, spec, backend)

    assert returncode == 2
    assert not backend.children
    assert not [event for event in backend.events if event[0] == "pause"]
    assert not [event for event in backend.events if event[0] == "resume"]
    status = json.loads(
        (
            tmp_path
            / "state"
            / spec.group_id
            / "status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["keepalive"]["initially_paused"] is True
    assert status["keepalive"]["resume_attempted"] is False

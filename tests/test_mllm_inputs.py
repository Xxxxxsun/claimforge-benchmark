from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.mllm.inputs import (
    BENCHMARK1000_DATASET_ID,
    from_benchmark1000,
)


def _write_fixture(
    root: Path,
) -> tuple[Path, Path, list[dict[str, object]]]:
    image = root / "image.png"
    image.write_bytes(b"fixture")
    ledger = root / "ledger.jsonl"
    manifest = root / "manifest.json"
    rows: list[dict[str, object]] = []
    for candidate in ("mouse", "cat", "trash_can", "real"):
        for index in range(250):
            label = "real" if candidate == "real" else "forged"
            rows.append(
                {
                    "id": f"{candidate}-{index:03d}",
                    "task_id": f"{candidate}-{index:03d}",
                    "image_path": "image.png",
                    "label": label,
                    "candidate": candidate,
                    "metadata": {
                        "dataset_id": BENCHMARK1000_DATASET_ID,
                    },
                }
            )
    ledger.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": BENCHMARK1000_DATASET_ID,
                "release_ready": True,
                "counts": {
                    "total": 1000,
                    "forged": 750,
                    "real": 250,
                    "mouse": 250,
                    "cat": 250,
                    "trash_can": 250,
                },
                "formal_ledger": {
                    "path": "ledger.jsonl",
                    "rows": 1000,
                    "sha256": hashlib.sha256(
                        ledger.read_bytes()
                    ).hexdigest(),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, ledger, rows


def _refresh_binding(manifest: Path, ledger: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["formal_ledger"]["sha256"] = hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_benchmark1000_accepts_exact_750_forged_plus_250_real(
    tmp_path: Path,
) -> None:
    manifest, ledger, _ = _write_fixture(tmp_path)

    items = from_benchmark1000(manifest, ledger, tmp_path)

    assert len(items) == 1000
    assert len({item.id for item in items}) == 1000


def test_benchmark1000_rejects_candidate_count_drift(
    tmp_path: Path,
) -> None:
    manifest, ledger, rows = _write_fixture(tmp_path)
    rows[-1]["candidate"] = "mouse"
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _refresh_binding(manifest, ledger)

    with pytest.raises(ValueError, match="candidate counts mismatch"):
        from_benchmark1000(manifest, ledger, tmp_path)


def test_benchmark1000_rejects_ledger_hash_drift(
    tmp_path: Path,
) -> None:
    manifest, ledger, _ = _write_fixture(tmp_path)
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        from_benchmark1000(manifest, ledger, tmp_path)

#!/usr/bin/env python3
"""Retry selected failed MLLM raw rows without discarding valid calls.

By default this retains its original bbox-only behavior. ``--all-errors``
targets every latest failed replicate; ``--exclude-id`` keeps known,
deterministic failures out of a recovery loop. Results are appended to the
original raw JSONL, so prior successful replicas remain usable for aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mllm.client import VisionClient
from eval.mllm.config import load_config
from eval.mllm.inputs import ImageItem
from eval.mllm.results import append_jsonl
from eval.mllm.run_mllm import _one_replicate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--concurrency", type=int, default=7)
    parser.add_argument("--all-errors", action="store_true", help="retry every latest failed replicate")
    parser.add_argument("--exclude-id", action="append", default=[], help="image ID to leave untouched; repeat for multiple IDs")
    parser.add_argument("--retry-until-complete", action="store_true", help="continue retry waves until selected targets succeed")
    parser.add_argument("--recovery-backoff-seconds", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_config(args.config, {args.model_slug})
    model = cfg["models"][0]
    client = VisionClient(model, float(cfg["api"]["timeout"]), cfg["image"])
    latest: dict[tuple[str, str, int], dict] = {}
    for line in args.raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[(row["id"], row["protocol_key"], int(row["replicate_index"]))] = row
    excluded = set(args.exclude_id)

    def selected_targets() -> list[tuple[dict, ImageItem]]:
        selected = []
        for row in latest.values():
            error = str((row.get("attempts") or [{}])[-1].get("error", ""))
            is_target = args.all_errors or (row.get("protocol_key") == "localization" and "bbox_1000" in error)
            if row.get("status") != "ok" and is_target and row["id"] not in excluded:
                selected.append((row, ImageItem(row["id"], Path(row["image_path"]), None, row.get("task_id"))))
        return selected

    recovery_id = "raw_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wave = 0
    while True:
        targets = selected_targets()
        print(json.dumps({"recovery_id": recovery_id, "wave": wave, "target_replicates": len(targets), "excluded_ids": len(excluded), "concurrency": args.concurrency}, ensure_ascii=False), flush=True)
        if not targets:
            break
        succeeded = failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    _one_replicate,
                    client,
                    item,
                    row["protocol_key"],
                    int(row["replicate_index"]),
                    cfg["retry"],
                    False,
                    tuple(row["image_size"]) if row.get("image_size") else None,
                ): row
                for row, item in targets
            }
            for future in as_completed(futures):
                previous = futures[future]
                result = future.result()
                row = {key: value for key, value in previous.items() if key not in {"attempts", "parsed", "raw_response", "latency_ms", "status"}}
                row.update(result)
                row["recovery_id"] = recovery_id
                row["recovery_reason"] = "all_errors" if args.all_errors else "invalid_bbox_schema"
                append_jsonl(args.raw_path, row)
                latest[(row["id"], row["protocol_key"], int(row["replicate_index"]))] = row
                if result["status"] == "ok":
                    succeeded += 1
                else:
                    failed += 1
                print(json.dumps({"wave": wave, "completed": succeeded + failed, "succeeded": succeeded, "failed": failed}, ensure_ascii=False), flush=True)
        if not args.retry_until_complete:
            break
        wave += 1
        delay = min(60.0, args.recovery_backoff_seconds * wave)
        print(json.dumps({"recovery_id": recovery_id, "wave": wave, "backoff_seconds": delay}, ensure_ascii=False), flush=True)
        time.sleep(delay)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a local-only signed HTTPS URL map for the CLAIMFORGE good275 pilot.

The output contains bearer-style signed URLs. Keep it outside version control.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-export", type=Path, default=Path("claimforge_generation_review_labels.json"))
    parser.add_argument("--output", type=Path, default=Path("config/pilot_good275_oss_urls.local.jsonl"))
    parser.add_argument("--oss-prefix", default="oss://quark-llm/primus/datasets/xuyue/claimforge")
    parser.add_argument("--endpoint", default="https://oss-cn-zhangjiakou.aliyuncs.com")
    parser.add_argument("--timeout-seconds", type=int, default=604800)
    args = parser.parse_args()

    payload = json.loads(args.review_export.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for row in payload.get("records", []):
        if row.get("status") == "good":
            paths.add(str(row["spliced_image"]))
            paths.add(str(row["source_image"]))
    if len(paths) != 550:
        raise SystemExit(f"expected 550 unique good275 image paths, found {len(paths)}")

    signed_rows: list[dict[str, str]] = []
    for relative_path in sorted(paths):
        cloud_url = args.oss_prefix.rstrip("/") + "/" + relative_path
        completed = subprocess.run(
            ["ossutil", "sign", cloud_url, "--timeout", str(args.timeout_seconds), "--endpoint", args.endpoint],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"https?://\S+", completed.stdout)
        if not match:
            raise RuntimeError(f"ossutil did not return an HTTPS URL for {relative_path}")
        signed_rows.append({"relative_path": relative_path, "url": match.group(0)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in signed_rows), encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps({"output": str(args.output), "url_count": len(signed_rows), "timeout_seconds": args.timeout_seconds}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze commercial API results for the mouse crop-scale probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_PROBE = Path("results/analysis/mouse_crop_scale_probe_v1")
DEFAULT_RESULTS = Path("results/commercial/crop_scale_probe_v1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def wilson_interval(positive: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = positive / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def parse_hive(row: dict[str, Any]) -> tuple[bool, float | None, str | None]:
    score = as_float(row.get("ai_probability"))
    return bool(row.get("detected", score is not None and score >= 0.9)), score, None


def parse_sightengine(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    score = as_float(row.get("ai_probability"))
    return bool(score is not None and score >= 0.5), score, None


def parse_resemble(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    label = str(row.get("provider_label") or "")
    return label in {"Fake", "Likely fake"}, as_float(row.get("provider_score")), label


def parse_alibaba(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    risks = {
        name: bool(row.get(f"{name}_detected"))
        for name in ("risk_edit", "risk_fake", "risk_aigc")
    }
    scores = [
        score
        for name in risks
        if (score := as_float(row.get(f"{name}_confidence"))) is not None
    ]
    label = "+".join(name for name, present in risks.items() if present) or "nonLabel"
    return any(risks.values()), max(scores) if scores else 0.0, label


def parse_aiornot(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    return bool(row.get("ai_detected")), as_float(row.get("ai_confidence")), None


def parse_copyleaks(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    return bool(row.get("is_ai_detected")), as_float(row.get("ai_score")), None


def error_type(row: dict[str, Any]) -> str:
    payload = json.dumps(row.get("attempts") or [], ensure_ascii=True).lower()
    markers = (
        ("insufficient_balance", "insufficient_balance"),
        ("image_blurry", "image_blurry"),
        ("insufficient_colors", "insufficient_colors"),
        ("low_dynamic_range", "low_dynamic_range"),
        ("too many requests", "too_many_requests"),
        ("usage_limit", "usage_limit"),
        ("usage limit", "usage_limit"),
    )
    for marker, name in markers:
        if marker in payload:
            return name
    attempts = row.get("attempts") or []
    if attempts and isinstance(attempts[-1], dict):
        status = attempts[-1].get("http_status")
        if status is not None:
            return f"http_{status}"
    return "other"


PARSERS: dict[
    str,
    Callable[[dict[str, Any]], tuple[bool, float | None, str | None]],
] = {
    "hive": parse_hive,
    "sightengine": parse_sightengine,
    "resemble": parse_resemble,
    "alibaba": parse_alibaba,
    "aiornot": parse_aiornot,
    "copyleaks": parse_copyleaks,
}


FILES = {
    "hive": "hive_mouse_oracle_vs_real_8x4x2_20260727.jsonl",
    "sightengine": "sightengine_mouse_oracle_vs_real_8x4x2_20260727.jsonl",
    "resemble": "resemble_mouse_oracle_vs_real_8x4x2_20260727.jsonl",
    "alibaba": "alibaba_mouse_oracle_vs_real_8x4x2_20260727.jsonl",
    "aiornot": "aiornot_mouse_oracle_vs_real_8x4x2_20260727.jsonl",
    "copyleaks": "copyleaks_mouse_oracle_vs_real_8x4x2_20260727.jsonl",
}


def latest_by_task(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["task_id"])] = row
    return latest


def normalized_rows(
    manifest: list[dict[str, Any]],
    result_dir: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for service, filename in FILES.items():
        result_path = result_dir / filename
        latest = latest_by_task(result_path)
        parser = PARSERS[service]
        for probe in manifest:
            compat_task_id = str(probe["compat_task_id"])
            raw = latest.get(compat_task_id)
            status = "missing" if raw is None else str(raw.get("status") or "error")
            positive: bool | None = None
            score: float | None = None
            provider_label: str | None = None
            failure: str | None = None
            if status == "ok" and raw is not None:
                positive, score, provider_label = parser(raw)
            elif status == "error" and raw is not None:
                failure = error_type(raw)
            output.append(
                {
                    "schema_version": "claimforge_mouse_crop_scale_result_v1",
                    "service": service,
                    "task_id": probe["task_id"],
                    "compat_task_id": compat_task_id,
                    "domain": probe["domain"],
                    "region_kind": probe["region_kind"],
                    "crop_factor": int(probe["crop_factor"]),
                    "tight_side": int(probe["tight_side"]),
                    "native_crop_side": int(probe["native_crop_size"][0]),
                    "resize_scale": float(probe["resize_scale"]),
                    "modified_fraction_native": float(
                        probe["modified_fraction_native"]
                    ),
                    "image": probe["image"],
                    "status": status,
                    "positive": positive,
                    "score": score,
                    "provider_label": provider_label,
                    "error_type": failure,
                    "result_id": raw.get("id") if raw else None,
                }
            )
    return output


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row["status"] != "missing"]
    valid = [row for row in rows if row["status"] == "ok"]
    positive = [row for row in valid if row["positive"]]
    scores = [float(row["score"]) for row in valid if row["score"] is not None]
    errors = [row for row in attempted if row["status"] == "error"]
    return {
        "expected": len(rows),
        "attempted": len(attempted),
        "valid": len(valid),
        "errors": len(errors),
        "missing": len(rows) - len(attempted),
        "coverage": len(valid) / len(rows) if rows else None,
        "positive": len(positive),
        "positive_rate_on_valid": len(positive) / len(valid) if valid else None,
        "positive_rate_wilson_95": wilson_interval(len(positive), len(valid)),
        "score_mean": mean(scores),
        "score_median": median(scores),
        "error_types": dict(Counter(row["error_type"] for row in errors)),
    }


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["task_id"]), int(row["crop_factor"]))][
            str(row["region_kind"])
        ] = row
    pairs = [
        value
        for value in by_key.values()
        if value.get("suspicious", {}).get("status") == "ok"
        and value.get("real_control", {}).get("status") == "ok"
    ]
    suspicious_positive = sum(bool(pair["suspicious"]["positive"]) for pair in pairs)
    control_positive = sum(bool(pair["real_control"]["positive"]) for pair in pairs)
    score_deltas = [
        float(pair["suspicious"]["score"]) - float(pair["real_control"]["score"])
        for pair in pairs
        if pair["suspicious"]["score"] is not None
        and pair["real_control"]["score"] is not None
    ]
    return {
        "pairs": len(pairs),
        "suspicious_positive": suspicious_positive,
        "real_control_positive": control_positive,
        "suspicious_only_positive": sum(
            bool(pair["suspicious"]["positive"])
            and not bool(pair["real_control"]["positive"])
            for pair in pairs
        ),
        "real_control_only_positive": sum(
            bool(pair["real_control"]["positive"])
            and not bool(pair["suspicious"]["positive"])
            for pair in pairs
        ),
        "both_positive": sum(
            bool(pair["real_control"]["positive"])
            and bool(pair["suspicious"]["positive"])
            for pair in pairs
        ),
        "paired_positive_rate_gap": (
            (suspicious_positive - control_positive) / len(pairs)
            if pairs
            else None
        ),
        "score_delta_mean": mean(score_deltas),
        "score_delta_median": median(score_deltas),
    }


def curve_patterns(rows: list[dict[str, Any]], factors: list[int]) -> dict[str, Any]:
    suspicious = [row for row in rows if row["region_kind"] == "suspicious"]
    by_task: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in suspicious:
        by_task[str(row["task_id"])][int(row["crop_factor"])] = row
    complete = {
        task_id: values
        for task_id, values in by_task.items()
        if all(values.get(factor, {}).get("status") == "ok" for factor in factors)
    }
    patterns: Counter[str] = Counter()
    first_loss: Counter[str] = Counter()
    non_monotonic = 0
    for values in complete.values():
        decisions = [bool(values[factor]["positive"]) for factor in factors]
        patterns["".join("1" if value else "0" for value in decisions)] += 1
        if decisions[0]:
            loss = next(
                (factor for factor, value in zip(factors[1:], decisions[1:]) if not value),
                None,
            )
            first_loss[str(loss) if loss is not None else "never"] += 1
        if any(
            not decisions[index] and decisions[index + 1]
            for index in range(len(decisions) - 1)
        ):
            non_monotonic += 1
    return {
        "complete_task_curves": len(complete),
        "decision_patterns_1x_2x_4x_8x": dict(patterns),
        "first_negative_after_positive_1x": dict(first_loss),
        "non_monotonic_curves": non_monotonic,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    factors = sorted({int(row["crop_factor"]) for row in rows})
    services: dict[str, Any] = {}
    for service in FILES:
        service_rows = [row for row in rows if row["service"] == service]
        by_factor: dict[str, Any] = {}
        for factor in factors:
            factor_rows = [
                row for row in service_rows if row["crop_factor"] == factor
            ]
            by_factor[str(factor)] = {
                "suspicious": summarize_group(
                    [row for row in factor_rows if row["region_kind"] == "suspicious"]
                ),
                "real_control": summarize_group(
                    [
                        row
                        for row in factor_rows
                        if row["region_kind"] == "real_control"
                    ]
                ),
                "paired": paired_summary(factor_rows),
            }
        overall = summarize_group(service_rows)
        if overall["valid"] == overall["expected"]:
            status = "complete"
        elif overall["attempted"] == overall["expected"]:
            status = "complete_requests_partial_applicability"
        else:
            status = "partial"
        services[service] = {
            "status": status,
            "overall": overall,
            "by_factor": by_factor,
            "paired_overall": paired_summary(service_rows),
            "suspicious_curve_patterns": curve_patterns(service_rows, factors),
        }
    return {
        "schema_version": "claimforge_mouse_crop_scale_analysis_v1",
        "expected_images_per_service": len(rows) // len(FILES),
        "factors": factors,
        "services": services,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "service",
        "factor",
        "region_kind",
        "expected",
        "attempted",
        "valid",
        "errors",
        "missing",
        "coverage",
        "positive",
        "positive_rate_on_valid",
        "score_mean",
        "score_median",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for service, service_summary in summary["services"].items():
            for factor, factor_summary in service_summary["by_factor"].items():
                for region_kind in ("suspicious", "real_control"):
                    row = factor_summary[region_kind]
                    writer.writerow(
                        {
                            "service": service,
                            "factor": factor,
                            "region_kind": region_kind,
                            **{name: row[name] for name in fieldnames[3:]},
                        }
                    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    probe_dir = (
        args.probe_dir if args.probe_dir.is_absolute() else repo_root / args.probe_dir
    ).resolve()
    result_dir = (
        args.result_dir
        if args.result_dir.is_absolute()
        else repo_root / args.result_dir
    ).resolve()
    manifest = read_jsonl(probe_dir / "manifest.jsonl")
    rows = normalized_rows(manifest, result_dir)
    summary = build_summary(rows)
    write_jsonl(probe_dir / "commercial_joined.jsonl", rows)
    write_json(probe_dir / "commercial_summary.json", summary)
    write_csv(probe_dir / "commercial_by_factor.csv", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

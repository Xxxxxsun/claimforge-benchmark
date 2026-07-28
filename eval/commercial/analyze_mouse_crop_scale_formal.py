#!/usr/bin/env python3
"""Analyze the formal commercial-API mouse crop-scale experiment."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "claimforge_mouse_crop_scale_formal_analysis_v1"
DEFAULT_PROBE = Path("results/analysis/mouse_crop_scale_formal_v1")
DEFAULT_RESULTS = Path("results/commercial/crop_scale_formal_v1")
FILES = {
    "aiornot": "aiornot_mouse_crop_scale_formal50_20260727.jsonl",
    "alibaba": "alibaba_mouse_crop_scale_formal50_20260727.jsonl",
    "copyleaks": "copyleaks_mouse_crop_scale_formal50_resize512_20260727.jsonl",
}
APPLICABLE_RENDER_MODES = {
    "aiornot": {"resize512", "native"},
    "alibaba": {"resize512", "native"},
    "copyleaks": {"resize512"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in rows
        ),
    )


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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


def mcnemar_exact(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def parse_aiornot(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    return bool(row.get("ai_detected")), as_float(row.get("ai_confidence")), None


def parse_alibaba(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    labels = ("risk_edit", "risk_fake", "risk_aigc")
    detected = {name: bool(row.get(f"{name}_detected")) for name in labels}
    scores = [
        score
        for name in labels
        if (score := as_float(row.get(f"{name}_confidence"))) is not None
    ]
    provider_label = "+".join(name for name in labels if detected[name]) or "nonLabel"
    return any(detected.values()), max(scores) if scores else 0.0, provider_label


def parse_copyleaks(
    row: dict[str, Any],
) -> tuple[bool, float | None, str | None]:
    return bool(row.get("is_ai_detected")), as_float(row.get("ai_score")), None


PARSERS: dict[
    str,
    Callable[[dict[str, Any]], tuple[bool, float | None, str | None]],
] = {
    "aiornot": parse_aiornot,
    "alibaba": parse_alibaba,
    "copyleaks": parse_copyleaks,
}


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
        ("parameter size error", "parameter_size"),
    )
    for marker, name in markers:
        if marker in payload:
            return name
    attempts = row.get("attempts") or []
    if attempts and isinstance(attempts[-1], dict):
        status = attempts[-1].get("http_status")
        if status is not None:
            return f"http_{status}"
        kind = attempts[-1].get("error_type")
        if kind:
            return str(kind)
    return "other"


def latest_by_task(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in read_jsonl(path)}


def normalized_rows(
    manifest: list[dict[str, Any]],
    result_dir: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for service, filename in FILES.items():
        latest = latest_by_task(result_dir / filename)
        parser = PARSERS[service]
        for probe in manifest:
            compat_task_id = str(probe["compat_task_id"])
            raw = latest.get(compat_task_id)
            applicable = (
                str(probe["render_mode"]) in APPLICABLE_RENDER_MODES[service]
            )
            if not applicable:
                status = "not_applicable"
            else:
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
                    "schema_version": SCHEMA_VERSION,
                    "service": service,
                    "task_id": probe["task_id"],
                    "compat_task_id": compat_task_id,
                    "task_rank": int(probe["task_rank"]),
                    "request_rank": int(probe["rank"]),
                    "domain": probe["domain"],
                    "region_kind": probe["region_kind"],
                    "render_mode": probe["render_mode"],
                    "crop_factor": float(probe["crop_factor"]),
                    "tight_side": int(probe["tight_side"]),
                    "native_crop_side": int(probe["native_crop_size"][0]),
                    "resize_scale": float(probe["resize_scale"]),
                    "modified_fraction_native": float(
                        probe["modified_fraction_native"]
                    ),
                    "image": probe["image"],
                    "status": status,
                    "applicable": applicable,
                    "positive": positive,
                    "score": score,
                    "provider_label": provider_label,
                    "error_type": failure,
                    "result_id": raw.get("id") if raw else None,
                }
            )
    return output


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if row["status"] != "not_applicable"]
    attempted = [row for row in applicable if row["status"] != "missing"]
    valid = [row for row in applicable if row["status"] == "ok"]
    positive = [row for row in valid if row["positive"]]
    scores = [float(row["score"]) for row in valid if row["score"] is not None]
    errors = [row for row in attempted if row["status"] == "error"]
    return {
        "protocol_rows": len(rows),
        "expected": len(applicable),
        "not_applicable": len(rows) - len(applicable),
        "attempted": len(attempted),
        "valid": len(valid),
        "errors": len(errors),
        "missing": len(applicable) - len(attempted),
        "coverage": len(valid) / len(applicable) if applicable else None,
        "acceptance_rate_on_attempted": (
            len(valid) / len(attempted) if attempted else None
        ),
        "positive": len(positive),
        "positive_rate_on_valid": len(positive) / len(valid) if valid else None,
        "positive_rate_wilson_95": wilson_interval(len(positive), len(valid)),
        "score_mean": mean(scores),
        "score_median": median(scores),
        "error_types": dict(Counter(row["error_type"] for row in errors)),
        "provider_label_counts": dict(
            Counter(
                str(row["provider_label"])
                for row in valid
                if row["provider_label"] is not None
            )
        ),
    }


def paired_arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_task[str(row["task_id"])][str(row["region_kind"])] = row
    pairs = [
        pair
        for pair in by_task.values()
        if pair.get("suspicious", {}).get("status") == "ok"
        and pair.get("real_control", {}).get("status") == "ok"
    ]
    suspicious_positive = sum(bool(pair["suspicious"]["positive"]) for pair in pairs)
    control_positive = sum(bool(pair["real_control"]["positive"]) for pair in pairs)
    suspicious_only = sum(
        bool(pair["suspicious"]["positive"])
        and not bool(pair["real_control"]["positive"])
        for pair in pairs
    )
    control_only = sum(
        bool(pair["real_control"]["positive"])
        and not bool(pair["suspicious"]["positive"])
        for pair in pairs
    )
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
        "suspicious_only_positive": suspicious_only,
        "real_control_only_positive": control_only,
        "both_positive": sum(
            bool(pair["suspicious"]["positive"])
            and bool(pair["real_control"]["positive"])
            for pair in pairs
        ),
        "neither_positive": sum(
            not bool(pair["suspicious"]["positive"])
            and not bool(pair["real_control"]["positive"])
            for pair in pairs
        ),
        "paired_positive_rate_gap": (
            (suspicious_positive - control_positive) / len(pairs)
            if pairs
            else None
        ),
        "mcnemar_exact_p": mcnemar_exact(suspicious_only, control_only),
        "score_delta_mean": mean(score_deltas),
        "score_delta_median": median(score_deltas),
    }


def curve_summary(
    rows: list[dict[str, Any]],
    factors: list[float],
) -> dict[str, Any]:
    suspicious = [
        row
        for row in rows
        if row["render_mode"] == "resize512"
        and row["region_kind"] == "suspicious"
    ]
    by_task: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in suspicious:
        by_task[str(row["task_id"])][float(row["crop_factor"])] = row
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
        patterns["".join("1" if decision else "0" for decision in decisions)] += 1
        if decisions[0]:
            loss = next(
                (
                    factor
                    for factor, decision in zip(factors[1:], decisions[1:])
                    if not decision
                ),
                None,
            )
            first_loss[f"{loss:g}x" if loss is not None else "never"] += 1
        if any(
            not decisions[index] and decisions[index + 1]
            for index in range(len(decisions) - 1)
        ):
            non_monotonic += 1

    retention: dict[str, Any] = {}
    for factor in factors:
        eligible = [
            values
            for values in by_task.values()
            if values.get(1.0, {}).get("status") == "ok"
            and bool(values[1.0]["positive"])
            and values.get(factor, {}).get("status") == "ok"
        ]
        retained = sum(bool(values[factor]["positive"]) for values in eligible)
        retention[f"{factor:g}"] = {
            "eligible_detected_at_1x": len(eligible),
            "still_positive": retained,
            "retention_rate": retained / len(eligible) if eligible else None,
        }
    return {
        "complete_task_curves": len(complete),
        "factor_order": factors,
        "decision_patterns": dict(patterns),
        "first_negative_after_positive_1x": dict(first_loss),
        "non_monotonic_curves": non_monotonic,
        "detection_retention_given_positive_at_1x": retention,
    }


def native_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ("suspicious", "real_control"):
        arm_rows = [row for row in rows if row["region_kind"] == arm]
        by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in arm_rows:
            if row["crop_factor"] != 1.0:
                continue
            by_task[str(row["task_id"])][str(row["render_mode"])] = row
        pairs = [
            pair
            for pair in by_task.values()
            if pair.get("resize512", {}).get("status") == "ok"
            and pair.get("native", {}).get("status") == "ok"
        ]
        resize_only = sum(
            bool(pair["resize512"]["positive"])
            and not bool(pair["native"]["positive"])
            for pair in pairs
        )
        native_only = sum(
            not bool(pair["resize512"]["positive"])
            and bool(pair["native"]["positive"])
            for pair in pairs
        )
        score_deltas = [
            float(pair["resize512"]["score"]) - float(pair["native"]["score"])
            for pair in pairs
            if pair["resize512"]["score"] is not None
            and pair["native"]["score"] is not None
        ]
        resize_positive = sum(bool(pair["resize512"]["positive"]) for pair in pairs)
        native_positive = sum(bool(pair["native"]["positive"]) for pair in pairs)
        output[arm] = {
            "pairs": len(pairs),
            "resize512_positive": resize_positive,
            "native_positive": native_positive,
            "resize512_only_positive": resize_only,
            "native_only_positive": native_only,
            "both_positive": sum(
                bool(pair["resize512"]["positive"])
                and bool(pair["native"]["positive"])
                for pair in pairs
            ),
            "neither_positive": sum(
                not bool(pair["resize512"]["positive"])
                and not bool(pair["native"]["positive"])
                for pair in pairs
            ),
            "paired_positive_rate_gap_resize_minus_native": (
                (resize_positive - native_positive) / len(pairs) if pairs else None
            ),
            "mcnemar_exact_p": mcnemar_exact(resize_only, native_only),
            "score_delta_resize_minus_native_mean": mean(score_deltas),
            "score_delta_resize_minus_native_median": median(score_deltas),
        }
    return output


def condition_key(render_mode: str, factor: float) -> str:
    return f"{render_mode}:{factor:g}x"


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    factors = sorted(
        {
            float(row["crop_factor"])
            for row in rows
            if row["render_mode"] == "resize512"
        }
    )
    services: dict[str, Any] = {}
    for service in FILES:
        service_rows = [row for row in rows if row["service"] == service]
        conditions = sorted(
            {
                (str(row["render_mode"]), float(row["crop_factor"]))
                for row in service_rows
            },
            key=lambda value: (value[0] == "native", value[1]),
        )
        by_condition: dict[str, Any] = {}
        for render_mode, factor in conditions:
            condition_rows = [
                row
                for row in service_rows
                if row["render_mode"] == render_mode
                and row["crop_factor"] == factor
            ]
            by_condition[condition_key(render_mode, factor)] = {
                "render_mode": render_mode,
                "crop_factor": factor,
                "suspicious": summarize_group(
                    [
                        row
                        for row in condition_rows
                        if row["region_kind"] == "suspicious"
                    ]
                ),
                "real_control": summarize_group(
                    [
                        row
                        for row in condition_rows
                        if row["region_kind"] == "real_control"
                    ]
                ),
                "paired": paired_arm_summary(condition_rows),
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
            "applicable_render_modes": sorted(APPLICABLE_RENDER_MODES[service]),
            "not_applicable_reason": (
                "Copyleaks requires both image dimensions to be at least 512 px"
                if service == "copyleaks"
                else None
            ),
            "positive_definition": (
                "provider ai_detected"
                if service == "aiornot"
                else (
                    "any of risk_edit, risk_fake, or risk_aigc"
                    if service == "alibaba"
                    else "provider isAiDetected"
                )
            ),
            "score_unit": "probability_0_to_1"
            if service in {"aiornot", "copyleaks"}
            else "provider_confidence_0_to_100",
            "overall": overall,
            "by_condition": by_condition,
            "resized_suspicious_curves": curve_summary(service_rows, factors),
            "native_1x_ablation": native_ablation(service_rows),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_images_per_service": len(rows) // len(FILES),
        "applicable_images_per_service": {
            service: services[service]["overall"]["expected"] for service in services
        },
        "resized_factors": factors,
        "services": services,
    }


def write_condition_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "service",
        "render_mode",
        "factor",
        "region_kind",
        "expected",
        "not_applicable",
        "attempted",
        "valid",
        "errors",
        "missing",
        "coverage",
        "acceptance_rate_on_attempted",
        "positive",
        "positive_rate_on_valid",
        "wilson_95_low",
        "wilson_95_high",
        "score_mean",
        "score_median",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for service, service_summary in summary["services"].items():
            for condition in service_summary["by_condition"].values():
                for arm in ("suspicious", "real_control"):
                    group = condition[arm]
                    interval = group["positive_rate_wilson_95"] or [None, None]
                    writer.writerow(
                        {
                            "service": service,
                            "render_mode": condition["render_mode"],
                            "factor": condition["crop_factor"],
                            "region_kind": arm,
                            "expected": group["expected"],
                            "not_applicable": group["not_applicable"],
                            "attempted": group["attempted"],
                            "valid": group["valid"],
                            "errors": group["errors"],
                            "missing": group["missing"],
                            "coverage": group["coverage"],
                            "acceptance_rate_on_attempted": group[
                                "acceptance_rate_on_attempted"
                            ],
                            "positive": group["positive"],
                            "positive_rate_on_valid": group[
                                "positive_rate_on_valid"
                            ],
                            "wilson_95_low": interval[0],
                            "wilson_95_high": interval[1],
                            "score_mean": group["score_mean"],
                            "score_median": group["score_median"],
                        }
                    )
    temporary.replace(path)


def write_paired_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "service",
        "render_mode",
        "factor",
        "pairs",
        "suspicious_positive",
        "real_control_positive",
        "suspicious_only_positive",
        "real_control_only_positive",
        "both_positive",
        "neither_positive",
        "paired_positive_rate_gap",
        "mcnemar_exact_p",
        "score_delta_mean",
        "score_delta_median",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for service, service_summary in summary["services"].items():
            for condition in service_summary["by_condition"].values():
                writer.writerow(
                    {
                        "service": service,
                        "render_mode": condition["render_mode"],
                        "factor": condition["crop_factor"],
                        **condition["paired"],
                    }
                )
    temporary.replace(path)


def svg_point(
    x: float,
    y: float,
    color: str,
    radius: float = 4,
) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
        f'fill="{color}" stroke="white" stroke-width="1.5"/>'
    )


def write_detection_svg(path: Path, summary: dict[str, Any]) -> None:
    services = list(summary["services"])
    factors = [float(value) for value in summary["resized_factors"]]
    panel_width = 430
    panel_height = 350
    margin_left = 56
    margin_right = 22
    margin_top = 82
    margin_bottom = 50
    width = panel_width * len(services)
    height = panel_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;letter-spacing:0}"
        ".tick{font-size:11px;fill:#4b5563}.title{font-size:15px;font-weight:700;"
        "fill:#111827}.note{font-size:11px;fill:#4b5563}</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    colors = {"suspicious": "#d55e00", "real_control": "#0072b2"}
    labels = {"suspicious": "Edit crop", "real_control": "Real control"}
    for panel_index, service in enumerate(services):
        panel_x = panel_index * panel_width
        plot_left = panel_x + margin_left
        plot_right = panel_x + panel_width - margin_right
        plot_top = margin_top
        plot_bottom = panel_height - margin_bottom
        service_summary = summary["services"][service]
        valid = service_summary["overall"]["valid"]
        expected = service_summary["overall"]["expected"]
        parts.append(
            f'<text class="title" x="{panel_x + 16}" y="22">'
            f"{html.escape(service)}: resized 512 inputs</text>"
        )
        parts.append(
            f'<text class="note" x="{panel_x + 16}" y="42">'
            f"valid {valid}/{expected}; points show provider decisions</text>"
        )
        for tick in range(6):
            rate = tick / 5
            y = plot_bottom - rate * (plot_bottom - plot_top)
            parts.append(
                f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
                f'y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text class="tick" x="{plot_left - 9}" y="{y + 4:.1f}" '
                f'text-anchor="end">{rate:.1f}</text>'
            )
        x_positions = [
            plot_left + index * (plot_right - plot_left) / (len(factors) - 1)
            for index in range(len(factors))
        ]
        for x, factor in zip(x_positions, factors):
            parts.append(
                f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 21}" '
                f'text-anchor="middle">{factor:g}x</text>'
            )
        parts.append(
            f'<text class="tick" x="{(plot_left + plot_right) / 2:.1f}" '
            f'y="{plot_bottom + 41}" text-anchor="middle">field-of-view factor</text>'
        )
        parts.append(
            f'<text class="tick" transform="translate({panel_x + 15},'
            f'{(plot_top + plot_bottom) / 2}) rotate(-90)" text-anchor="middle">'
            "positive rate</text>"
        )
        for arm in ("suspicious", "real_control"):
            points: list[tuple[float, float, list[float] | None]] = []
            for x, factor in zip(x_positions, factors):
                group = service_summary["by_condition"][
                    condition_key("resize512", factor)
                ][arm]
                rate = group["positive_rate_on_valid"]
                if rate is not None:
                    points.append((x, float(rate), group["positive_rate_wilson_95"]))
            if points:
                polyline = " ".join(
                    f"{x:.1f},{plot_bottom - rate * (plot_bottom - plot_top):.1f}"
                    for x, rate, _ in points
                )
                parts.append(
                    f'<polyline points="{polyline}" fill="none" '
                    f'stroke="{colors[arm]}" stroke-width="2.5"/>'
                )
                for x, rate, interval in points:
                    y = plot_bottom - rate * (plot_bottom - plot_top)
                    if interval is not None:
                        high_y = plot_bottom - interval[1] * (
                            plot_bottom - plot_top
                        )
                        low_y = plot_bottom - interval[0] * (
                            plot_bottom - plot_top
                        )
                        parts.append(
                            f'<line x1="{x:.1f}" y1="{high_y:.1f}" '
                            f'x2="{x:.1f}" y2="{low_y:.1f}" '
                            f'stroke="{colors[arm]}" stroke-width="1"/>'
                        )
                    parts.append(svg_point(x, y, colors[arm]))
            legend_x = plot_left + (0 if arm == "suspicious" else 115)
            parts.append(
                f'<line x1="{legend_x}" y1="{plot_top - 20}" '
                f'x2="{legend_x + 22}" y2="{plot_top - 20}" '
                f'stroke="{colors[arm]}" stroke-width="2.5"/>'
            )
            parts.append(
                f'<text class="tick" x="{legend_x + 28}" y="{plot_top - 16}">'
                f"{labels[arm]}</text>"
            )
    parts.append("</svg>")
    atomic_write(path, "\n".join(parts) + "\n")


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
    if len(manifest) != 700:
        raise ValueError(f"expected 700 formal inputs, found {len(manifest)}")
    rows = normalized_rows(manifest, result_dir)
    summary = build_summary(rows)
    write_jsonl(probe_dir / "commercial_joined.jsonl", rows)
    write_json(probe_dir / "commercial_summary.json", summary)
    write_condition_csv(probe_dir / "commercial_by_condition.csv", summary)
    write_paired_csv(probe_dir / "commercial_paired_by_condition.csv", summary)
    write_detection_svg(probe_dir / "commercial_detection_curves.svg", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

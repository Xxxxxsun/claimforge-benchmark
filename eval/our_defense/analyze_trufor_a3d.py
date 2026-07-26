#!/usr/bin/env python3
"""Produce paired-bootstrap and slice analysis for an A3D JSONL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from eval.opensource.common import atomic_write_json, read_jsonl, utc_now
from eval.opensource.maskclip_metrics import image_detection_metrics
from eval.our_defense.run_trufor_a3d import _logit_mean_score


SCHEMA_VERSION = "claimforge_trufor_a3d_analysis_v2"
METHODS = {
    "TruFor full": "full_score",
    "A3D local": "a3d_score",
    "A3D fused": "a3d_fused_score",
}
DETECTION_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
)


def _pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(str(row["task_id"]), {})[str(row["kind"])] = row
    pairs = []
    for task_id, kinds in grouped.items():
        if set(kinds) != {"real", "forged"}:
            raise ValueError(f"incomplete pair: {task_id}")
        real = kinds["real"]
        forged = kinds["forged"]
        if float(real["gt_fraction"]) != float(forged["gt_fraction"]):
            raise ValueError(f"GT fraction mismatch: {task_id}")
        pairs.append(
            {
                "task_id": task_id,
                "domain": forged["domain"],
                "gt_fraction": float(forged["gt_fraction"]),
                "split": forged["split"],
                "real": real,
                "forged": forged,
            }
        )
    return sorted(pairs, key=lambda pair: int(pair["forged"]["pair_rank"]))


def _add_quintiles(pairs: list[dict[str, Any]]) -> None:
    ordered = sorted(
        pairs,
        key=lambda pair: (
            float(pair["gt_fraction"]),
            int(pair["forged"]["pair_rank"]),
        ),
    )
    for index, pair in enumerate(ordered):
        pair["size_quintile"] = min(5, index * 5 // len(ordered) + 1)


def _detection(
    pairs: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    rows = []
    deltas = []
    for pair in pairs:
        def score(row: dict[str, Any]) -> float:
            value = row.get(score_key)
            if value is not None:
                return float(value)
            if score_key == "a3d_fused_score":
                return _logit_mean_score(
                    float(row["full_score"]),
                    float(row["a3d_score"]),
                )
            raise KeyError(score_key)

        real_score = score(pair["real"])
        forged_score = score(pair["forged"])
        rows.extend(
            [
                {"status": "ok", "label": 0, "score": real_score},
                {"status": "ok", "label": 1, "score": forged_score},
            ]
        )
        deltas.append(forged_score - real_score)
    metrics = image_detection_metrics(rows, 0.5)
    metrics["paired_ranking_accuracy"] = float(
        np.mean(np.asarray(deltas) > 0)
    )
    metrics["paired_delta_mean"] = float(np.mean(deltas))
    return metrics


def _localization(
    pairs: list[dict[str, Any]],
    strategy: str,
) -> dict[str, float]:
    result = {}
    for metric in ("pixel_ap", "f1", "iou", "mcc"):
        values = [
            pair["forged"]["localization"][strategy].get(metric)
            for pair in pairs
        ]
        finite = [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ]
        result[metric] = float(np.mean(finite)) if finite else float("nan")
    return result


def _proposal(
    pairs: list[dict[str, Any]],
    strategy: str,
) -> dict[str, float]:
    values = np.asarray(
        [
            float(
                pair["forged"]["localization"][strategy][
                    "proposal_target_recall"
                ]
            )
            for pair in pairs
        ],
        dtype=np.float64,
    )
    return {
        "mean_target_recall": float(np.mean(values)),
        "any_hit_rate": float(np.mean(values > 0)),
        "full_cover_rate": float(np.mean(values >= 1.0)),
    }


def _point_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(pairs),
        "gt_fraction_median": float(
            np.median([pair["gt_fraction"] for pair in pairs])
        ),
        "detection": {
            method: _detection(pairs, score_key)
            for method, score_key in METHODS.items()
        },
        "localization": {
            "TruFor full": _localization(pairs, "full"),
            "A3D top1": _localization(pairs, "a3d_top1"),
            "A3D top2": _localization(pairs, "a3d_top2"),
            "A3D all4": _localization(pairs, "a3d_all4"),
        },
        "proposal": {
            "top1": _proposal(pairs, "a3d_top1"),
            "top2": _proposal(pairs, "a3d_top2"),
            "all4": _proposal(pairs, "a3d_all4"),
        },
    }


def _percentile_interval(values: list[float]) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _bootstrap(
    pairs: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    method_samples = {
        method: {metric: [] for metric in DETECTION_METRICS}
        for method in METHODS
    }
    detection_delta = {
        method: {metric: [] for metric in DETECTION_METRICS}
        for method in ("A3D local", "A3D fused")
    }
    localization_samples = {
        method: {metric: [] for metric in ("pixel_ap", "f1")}
        for method in ("TruFor full", "A3D top1")
    }
    localization_delta = {metric: [] for metric in ("pixel_ap", "f1")}

    for _ in range(replicates):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        sample = [pairs[int(index)] for index in indices]
        detections = {
            method: _detection(sample, score_key)
            for method, score_key in METHODS.items()
        }
        for method in METHODS:
            for metric in DETECTION_METRICS:
                method_samples[method][metric].append(
                    float(detections[method][metric])
                )
        for method in detection_delta:
            for metric in DETECTION_METRICS:
                detection_delta[method][metric].append(
                    float(detections[method][metric])
                    - float(detections["TruFor full"][metric])
                )

        localizations = {
            "TruFor full": _localization(sample, "full"),
            "A3D top1": _localization(sample, "a3d_top1"),
        }
        for method in localizations:
            for metric in ("pixel_ap", "f1"):
                localization_samples[method][metric].append(
                    float(localizations[method][metric])
                )
        for metric in ("pixel_ap", "f1"):
            localization_delta[metric].append(
                float(localizations["A3D top1"][metric])
                - float(localizations["TruFor full"][metric])
            )

    return {
        "replicates": replicates,
        "seed": seed,
        "detection_95ci": {
            method: {
                metric: _percentile_interval(values)
                for metric, values in metrics.items()
            }
            for method, metrics in method_samples.items()
        },
        "detection_delta_vs_full_95ci": {
            method: {
                metric: _percentile_interval(values)
                for metric, values in metrics.items()
            }
            for method, metrics in detection_delta.items()
        },
        "localization_95ci": {
            method: {
                metric: _percentile_interval(values)
                for metric, values in metrics.items()
            }
            for method, metrics in localization_samples.items()
        },
        "localization_delta_a3d_minus_full_95ci": {
            metric: _percentile_interval(values)
            for metric, values in localization_delta.items()
        },
    }


def _scope(
    pairs: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    selected = [pair for pair in pairs if predicate(pair)]
    if not selected:
        raise ValueError("analysis scope is empty")
    return selected


def _fused_score(row: dict[str, Any]) -> float:
    value = row.get("a3d_fused_score")
    if value is not None:
        return float(value)
    return _logit_mean_score(
        float(row["full_score"]),
        float(row["a3d_score"]),
    )


def _fixed_operating_point(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    real = [row for row in ok if row["kind"] == "real"]
    forged = [row for row in ok if row["kind"] == "forged"]
    return {
        "images": len(ok),
        "real_images": len(real),
        "forged_images": len(forged),
        "threshold": threshold,
        "fpr": (
            sum(_fused_score(row) >= threshold for row in real) / len(real)
            if real
            else None
        ),
        "tpr": (
            sum(_fused_score(row) >= threshold for row in forged) / len(forged)
            if forged
            else None
        ),
    }


def _cross_object_calibration(
    calibration_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    real_dev = np.asarray(
        [
            _fused_score(row)
            for row in calibration_rows
            if row.get("status") == "ok"
            and row["kind"] == "real"
            and row["split"] == "dev"
        ],
        dtype=np.float64,
    )
    if real_dev.size == 0:
        raise ValueError("cross-object calibration has no dev-real rows")
    quantile = float(np.quantile(real_dev, 1.0 - alpha, method="higher"))
    threshold = float(np.nextafter(quantile, np.inf))
    return {
        "score": "equal_weight_mean_of_full_and_local_logits",
        "alpha": alpha,
        "calibration_dev_real_images": int(real_dev.size),
        "threshold": threshold,
        "calibration_empirical_fpr": float(np.mean(real_dev >= threshold)),
        "evaluation": {
            "all": _fixed_operating_point(evaluation_rows, threshold),
            "hash_test": _fixed_operating_point(
                [
                    row
                    for row in evaluation_rows
                    if row.get("split") == "test"
                ],
                threshold,
            ),
        },
    }


def _hard_cases(pairs: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    result = []
    for pair in pairs:
        real = float(pair["real"]["a3d_score"])
        forged = float(pair["forged"]["a3d_score"])
        recall = float(
            pair["forged"]["localization"]["a3d_all4"][
                "proposal_target_recall"
            ]
        )
        result.append(
            {
                "task_id": pair["task_id"],
                "size_quintile": pair["size_quintile"],
                "domain": pair["domain"],
                "gt_fraction": pair["gt_fraction"],
                "a3d_real_score": real,
                "a3d_forged_score": forged,
                "a3d_pair_delta": forged - real,
                "full_pair_delta": (
                    float(pair["forged"]["full_score"])
                    - float(pair["real"]["full_score"])
                ),
                "proposal_all4_target_recall": recall,
                "failure_type": (
                    "proposal_miss"
                    if recall == 0
                    else "crop_score_nonpositive_delta"
                    if forged <= real
                    else "low_margin"
                ),
            }
        )
    return sorted(result, key=lambda item: item["a3d_pair_delta"])[:limit]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# A3D evaluation",
        "",
        "Q1 is the smallest-edit quintile and Q2-Q5 contains the remaining "
        "larger edits. The analyzer does not fit or tune any A3D parameter.",
        "",
        "## Detection by size scope",
        "",
        "| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | "
        "Full TPR@5% | Local TPR@5% | Fused TPR@5% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "q1_smallest", "q2_q5_larger"):
        item = analysis["scopes"][name]
        full = item["detection"]["TruFor full"]
        local = item["detection"]["A3D local"]
        fused = item["detection"]["A3D fused"]
        lines.append(
            f"| {name} | {item['pairs']} | {_fmt(full['auroc'])} | "
            f"{_fmt(local['auroc'])} | {_fmt(fused['auroc'])} | "
            f"{_fmt(full['tpr_at_fpr_5_percent'])} | "
            f"{_fmt(local['tpr_at_fpr_5_percent'])} | "
            f"{_fmt(fused['tpr_at_fpr_5_percent'])} |"
        )
    lines.extend(
        [
            "",
            "## Size quintiles",
            "",
            "| Quintile | Median edit % | Full AUROC | A3D AUROC | "
            "Full pixel F1 | A3D pixel F1 | Proposal hit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for quintile in range(1, 6):
        item = analysis["scopes"][f"q{quintile}"]
        lines.append(
            f"| Q{quintile} | "
            f"{100 * item['gt_fraction_median']:.5f} | "
            f"{_fmt(item['detection']['TruFor full']['auroc'])} | "
            f"{_fmt(item['detection']['A3D fused']['auroc'])} | "
            f"{_fmt(item['localization']['TruFor full']['f1'])} | "
            f"{_fmt(item['localization']['A3D top1']['f1'])} | "
            f"{_fmt(item['proposal']['all4']['any_hit_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap delta, A3D minus full",
            "",
            "| Scope | Metric | 95% CI |",
            "|---|---|---:|",
        ]
    )
    for name, bootstrap in analysis["bootstrap"].items():
        for method, metrics in bootstrap[
            "detection_delta_vs_full_95ci"
        ].items():
            for metric, interval in metrics.items():
                lines.append(
                    f"| {name} | {method} {metric} | "
                    f"[{_fmt(interval[0])}, {_fmt(interval[1])}] |"
                )
        for metric, interval in bootstrap[
            "localization_delta_a3d_minus_full_95ci"
        ].items():
            lines.append(
                f"| {name} | pixel {metric} | "
                f"[{_fmt(interval[0])}, {_fmt(interval[1])}] |"
                )
    calibration = analysis.get("cross_object_calibration")
    if calibration is not None:
        lines.extend(
            [
                "",
                "## Fixed cross-object operating point",
                "",
                f"Threshold `{calibration['threshold']:.9f}` was calibrated "
                f"from {calibration['calibration_dev_real_images']} dev-real "
                "images in a separate result set.",
                "",
                "| Evaluation scope | FPR | TPR |",
                "|---|---:|---:|",
            ]
        )
        for name, point in calibration["evaluation"].items():
            lines.append(
                f"| {name} | {_fmt(point['fpr'])} | {_fmt(point['tpr'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        row
        for results_path in args.results
        for row in read_jsonl(results_path)
    ]
    pairs = _pair_rows(rows)
    _add_quintiles(pairs)
    scopes = {
        "all": pairs,
        "q1_smallest": _scope(
            pairs, lambda pair: pair["size_quintile"] == 1
        ),
        "q2_q5_larger": _scope(
            pairs, lambda pair: pair["size_quintile"] >= 2
        ),
        "hash_test": _scope(pairs, lambda pair: pair["split"] == "test"),
        "lodging": _scope(pairs, lambda pair: pair["domain"] == "lodging"),
        "restaurant": _scope(
            pairs, lambda pair: pair["domain"] == "restaurant"
        ),
    }
    for quintile in range(1, 6):
        scopes[f"q{quintile}"] = _scope(
            pairs,
            lambda pair, q=quintile: pair["size_quintile"] == q,
        )
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "source_results": [str(path) for path in args.results],
        "pairs": len(pairs),
        "scopes": {
            name: _point_summary(selected) for name, selected in scopes.items()
        },
        "bootstrap": {
            name: _bootstrap(
                scopes[name],
                args.bootstrap_replicates,
                args.seed + index,
            )
            for index, name in enumerate(
                ("all", "q1_smallest", "q2_q5_larger")
            )
        },
        "hard_cases": _hard_cases(pairs),
        "completed_at": utc_now(),
    }
    if args.calibration_results is not None:
        calibration_rows = read_jsonl(args.calibration_results)
        analysis["cross_object_calibration"] = _cross_object_calibration(
            calibration_rows,
            rows,
            args.calibration_alpha,
        )
        analysis["cross_object_calibration"]["source_results"] = str(
            args.calibration_results
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_json, analysis)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(analysis), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "pairs": len(pairs),
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
        )
    )
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--calibration-results", type=Path)
    parser.add_argument("--calibration-alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

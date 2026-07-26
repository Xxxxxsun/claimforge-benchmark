# Balanced250 generic evaluation contract

**Frozen:** 2026-07-26
**Status:** shared v2 implementation complete
**Dataset:** `claimforge-balanced250-independent-panel-jpeg-q95-v1`

## 1. Preserve the audited Mouse v1 path

The existing Mouse runners, analyzers, `common.py`, and paired metrics are
immutable provenance dependencies. Historical run manifests record their
file hashes, and several analyzers re-hash the live files during replay.
Balanced250 support therefore uses new v2 modules and thin method adapters;
it does not add schema branches to the audited v1 files.

The generic release reader has two explicit adapters:

- legacy `claimforge_mouse_canonical_v1`, with the original flat inputs and
  pair ledgers;
- `claimforge_balanced250_canonical_v1`, with nested inputs, panel, and
  source-pair ledgers.

The two pair concepts are not interchangeable. A legacy Mouse `pair_rank`
identifies one real/forged local-insertion pair. A Balanced250
`source_pairs.jsonl` row explicitly links two sample IDs for a secondary
analysis. No Balanced250 input receives a synthetic `pair_rank`.

## 2. Capability-correct input selection

Every selected row is scored at most once per method run. Method capability,
not the presence of a dense map, controls the formal input set:

| Capability | Conditions | Formal inputs | T1 | T2 |
|---|---|---:|---|---|
| whole-image detector | all seven | 1,775 | all panel conditions | N/A |
| native T1+T2 local method | all seven | 1,775 | all panel conditions | three local conditions |
| native T2-only local method | real + three local | 1,025 | N/A | three local conditions plus real FP area |

Smoke selection is deterministic by condition: panel members are preferred,
then `selection_rank`, then frozen input rank. `per_condition_limit=5`
therefore produces the first five panel rows from each applicable condition.
The formal run uses no limit or condition override. Legacy `pair_limit`
remains available only to the Mouse adapter and is rejected for Balanced250.

## 3. Three-state localization contract

Ground truth is selected from explicit row semantics:

| Row scope | `gt_mask_kind` | Loader result | Formal use |
|---|---|---|---|
| authentic real | `all_zero` | zero mask | T2 false-positive area |
| local insertion | `exact_diff` | verified binary mask | T2 localization |
| full-frame conditional edit | `not_applicable` | `None` | T1 only |

The loader verifies canonical input path, hash, byte count, dimensions, JPEG
properties, and identity. Exact-difference masks additionally require a safe
path, frozen hash, native `L` mode, binary values, matching dimensions, and
the frozen positive-pixel count. A forged label alone never implies that a
mask exists.

Full-frame dense outputs may be retained as diagnostics by a native T1+T2
method, but they cannot be scored against the orange conditioning box, a
fabricated full-image mask, or any other T2 target.

## 4. v2 result and resume identity

Every physical result row propagates at least:

```text
sample_id
rank
dataset_id
condition
condition_family
manipulation_scope
normalized_task_id
task_id
kind
label
domain
gt_mask_kind
input_path
input_sha256
input_width
input_height
run_manifest_fingerprint
status
valid_for_metrics
```

`sample_id` is the result identity. Multiple physical attempts may exist in
an append-only result file; the last physical row is authoritative. Resume
accepts an earlier success only when all frozen identity fields and the run
fingerprint still match. Identity or configuration drift fails closed rather
than silently reusing a score.

An `ok` row must set `valid_for_metrics=true`; an `error` row must set it to
false. Aggregation accepts exactly one post-resume `ok` row per selected input
and independently rechecks the v2 schema, canonical input identity, run ID,
run-manifest fingerprint, and declared score.

The run dataset contract binds:

- release manifest path and SHA-256;
- schema, dataset ID, and deterministic contract SHA-256;
- all available ledger paths, hashes, and row counts;
- capability and selection specification;
- ordered selected-sample-ID hash;
- selected row count by condition;
- score key, direction, fixed threshold, and threshold operator.

Method adapters add their own source, checkpoint, preprocessing, runtime,
and code-file hashes. They also hash `common.py`, all three shared v2 modules,
the new method adapter/analyzer, and every imported frozen v1 inference source.
The normalized T1 score key is declared in the run manifest; method-native
aliases remain available as diagnostics.

## 5. Primary and secondary statistics

The 1,775-row score cache feeds two explicitly separate analyses.

### Primary selection-unpaired panel

Only `panel.jsonl` enters the primary analysis. The 250 real rows are compared
separately with each independently selected 250-row forged condition:

```text
real250 vs local_mouse250
real250 vs local_cat250
real250 vs local_trash_can250
real250 vs fullframe_mouse250
real250 vs fullframe_cat250
real250 vs fullframe_trash_can250
```

Each comparison reports AUROC, AP, TPR at 5% real FPR, the released fixed
threshold and operator, coverage, domains, and source-cluster confidence
intervals.
The fixed operator may be strict or inclusive as frozen by the method; the
TPR-at-FPR quantile uses its own separately reported strict operator. Local and
full-frame macro averages are computed from condition-level point estimates.

“Selection-unpaired” describes the point estimate: no pair IDs, matched
deltas, sign tests, or pair-derived ranking enter the primary statistics. It
does not mean source content is independent. In the frozen panel, the real
source clusters overlap the six forged conditions by 232, 228, 231, 230, 229,
and 228 clusters respectively. Confidence intervals therefore use shared
source-cluster Poisson bootstrap weights across real/forged labels and across
conditions. Macro replicates reuse those aligned cluster weights, preserving
the observed covariance without turning the point estimate into a matched
analysis.

### Secondary source-matched analysis

Only the 1,500 explicit rows in `source_pairs.jsonl` define source matches.
Results are joined by `real_sample_id` and `forged_sample_id`; `task_id` is
never used to infer a pair. Repeated source content is clustered by
`source_content_cluster` during bootstrap, with one shared Poisson cluster
weight plan across conditions so macro intervals retain covariance. The extra
25 genuine rows outside the primary panel exist only to complete this
analysis.

## 6. Migration and execution

CNNDetection is the first whole-image v2 canary. MaskCLIP follows as the
native T1+T2 canary. A method-specific v2 runner and analyzer may import
frozen inference helpers from its v1 module, but must not edit that module.
Its adapter contract records both the new files and the imported v1 source
hash.

For each method:

```text
CPU asset and dataset preflight
-> deterministic smoke A
-> deterministic smoke B
-> A/B byte and score comparison
-> capability-correct formal run
-> primary and secondary analysis
-> complete fresh model replay
-> report
-> commit and push
```

The Hunyuan keepalive remains active during CPU work, analysis, documentation,
pushes, and after the queue ends. It is paused and drained immediately before
a CUDA benchmark window, then resumed after that window even if the method
fails.

## 7. Shared implementation

The frozen implementation is split into three method-independent modules:

- `canonical_release.py`: strict legacy-Mouse and Balanced250 release loading,
  capability selection, three-state GT, and file verification;
- `balanced_run_contract.py`: dataset/run binding, v2 identity, append-only
  last-attempt indexing, and fail-closed coverage;
- `balanced250_metrics.py`: selection-unpaired T1 point statistics with
  source-cluster confidence intervals, plus explicit source-matched secondary
  statistics.

Unit tests cover schema and provenance drift, task-capability boundaries,
panel-priority smoke selection, q95 4:4:4 JPEG enforcement, result identity,
resume history, complete coverage, primary metrics, and source-cluster
bootstrap across labels and conditions. Both frozen releases are additionally
loaded with physical file verification before the first method adapter is run.

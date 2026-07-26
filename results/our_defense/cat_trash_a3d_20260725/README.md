# Cat/trash-can A3D results

## Contents

This directory is the durable result bundle for the final cat 251-pair and
trash-can 250-pair evaluation pulled on 2026-07-25.

- `cat_final_251_clean_a3d_v1.jsonl`: per-image cat predictions and diagnostics.
- `trash_can_final_250_clean_a3d_v1.jsonl`: per-image trash-can predictions and
  diagnostics.
- `*.summary.json`: runner summaries.
- `*.analysis.json` and `*.analysis.md`: 2,000-replicate paired-bootstrap
  analyses.
- `cat_trash_combined_501_clean_a3d_v1.*`: combined 501-pair analysis.
- `canonical/*/manifest.json` and `canonical/*/pairs.jsonl`: immutable
  provenance snapshots for the evaluated inputs. Canonical JPEGs and masks
  remain under `outputs/our_defense/canonical/`.

## Current detector

For each image the detector:

1. Canonicalizes both real and forged benchmark variants with identical
   JPEG-Q95 settings.
2. Runs frozen TruFor once on the complete image.
3. Scores a 512x512, stride-384 grid from the full-pass manipulation and
   reliability maps.
4. Runs TruFor on the four strongest proposed crops.
5. Defines `local_score` as the maximum of the four crop probabilities.
6. Defines the primary score as
   `sigmoid((logit(full_score) + logit(local_score)) / 2)`.
7. Applies a single threshold calibrated from known-clean reference images.

Ground truth does not affect proposals, crop selection, scores, or the fixed
threshold. The stable localization output is the full-image TruFor map. Local
maps are retained as small-edit diagnostics because they degrade localization
on some larger edits.

## Why the fusion result is exploratory

The first frozen A3D rule reported the local maximum directly. After evaluating
cat and trash-can, local-only was found to help small cat edits but reduce
trash-can overall AUROC. Several fixed combinations of the already-computed
full/local scores were then compared on mouse, cat, and trash-can. Equal logit
fusion was selected because it was the most stable of those candidates.

Therefore cat/trash labels indirectly influenced selection of the fusion rule.
The reported fused metrics are valid measurements of these files, but they are
not a pristine blind generalization estimate. The rule must now be frozen and
tested on a future unseen object category.

## Recommended reference-to-blind protocol

Use a known-clean reference pool, not an arbitrary unlabeled real/fake mixture:

1. Before any blind inference, freeze the checkpoint, crop geometry, proposal
   budget, equal-logit fusion, and all preprocessing.
2. Choose reference IDs by a deterministic task-ID hash. Use about 80-100
   known-real images; a 20% split of a 250-image category is only 50 negatives
   and gives a noisy estimate of a 5% tail.
3. Reveal no forged labels in the reference pool. Use reference-real scores
   only to set the target-FPR quantile and, if needed, robust normalization.
4. Anonymize and shuffle the remaining same-category images plus every image
   from other categories. Keep their labels in a separate evaluator-owned key.
5. Produce and hash `predictions.jsonl` before the key is opened.
6. Open the key once and report the same frozen score at AUROC, AP, TPR@1%,
   TPR@5%, and the fixed reference-derived operating point.
7. Do not change the method after blind results. Any change creates a new
   version that requires a new unseen category.

If the reference pool is a completely unlabeled mixture, it may support
transductive score normalization under explicit contamination assumptions, but
it cannot identify a trustworthy 5% real-image FPR threshold or choose a
supervised fusion rule.

## Fixed threshold in this bundle

The threshold `0.6353510120379108` used only 80 dev-real mouse images:

| Evaluation | FPR | TPR |
|---|---:|---:|
| Cat, all | 3.98% | 94.82% |
| Trash-can, all | 4.80% | 78.80% |
| Combined, all | 4.39% | 86.83% |

This threshold transfer is cross-object. The equal-logit fusion choice is the
part that remains post-selection and needs a future blind category.

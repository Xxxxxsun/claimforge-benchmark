# A3D: adaptive anomaly-aware defense

## Outcome

A3D is a test-time coarse-to-fine wrapper around a frozen forensic model. It
targets CLAIMFORGE's main failure mode: a generated object can occupy less than
0.1% of an otherwise real image, so its evidence is diluted in a whole-image
forward pass.

The current stable detection score is the equal-weight mean of the full-image
and strongest-local-crop **logits**. This retains the global branch on large
edits while adding local evidence for small edits. The local-only score is
still emitted as a diagnostic.

On the 275-pair canonical mouse set, the frozen TruFor baseline and A3D use the
same checkpoint. Current fused A3D improves image AUROC from 0.8180 to 0.8796
and TPR@5%FPR from 0.4436 to 0.6655. Local-only is a slightly stronger mouse
detection ablation (0.8798 / 0.6727), but is less stable on trash-can. The
stable localization output remains the full-image map; local top-1 improves
mouse macro pixel F1 from 0.5001 to 0.5599 only as a diagnostic.

This is a strong native-input pilot, not a JPEG-robust solution. After JPEG
quality-90 recompression, fused A3D AUROC falls to 0.6101 (local-only 0.6166)
and pixel F1 remains about 0.20.

The complete method, implementation audit, all metrics, and conclusions are in
`docs/A3D_ADAPTIVE_DEFENSE_FULL_REPORT_2026-07-27.md`.

## Cross-object result: cat and trash-can

The final 2026-07-25 cat and trash-can sets are evaluated separately and
together. Both real and forged variants are canonicalized to JPEG-Q95 with
identical settings before inference.

| Set | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cat | 251 | 0.9693 | 0.9791 | **0.9808** | 0.9004 | 0.9402 | **0.9522** |
| Trash-can | 250 | 0.9291 | 0.9051 | **0.9318** | 0.7280 | 0.7040 | **0.7880** |
| Combined | 501 | 0.9493 | 0.9427 | **0.9567** | 0.8184 | 0.8244 | **0.8743** |

The local-only branch helps the smallest-edit quintile but can hurt larger
trash-can edits. Logit fusion removes that instability. On the combined set,
paired bootstrap with 2,000 resamples gives fused-minus-full 95% intervals of
[-0.0000, 0.0156] AUROC, [0.0005, 0.0130] average precision, and
[0.0100, 0.0978] TPR@5%FPR. In the smallest-edit quintile, the AUROC interval
is [0.0053, 0.0524].

For an actual fixed operating point, a threshold of `0.635351012` is calibrated
only from the 80 dev-real images in the earlier mouse set. Without looking at
cat/trash labels during calibration, it gives:

| Set | FPR | TPR |
|---|---:|---:|
| Cat | 3.98% | 94.82% |
| Trash-can | 4.80% | 78.80% |

The fusion rule was selected after inspecting these new cross-object results,
so this section is an exploratory validation rather than a pristine held-out
claim. Freeze it before evaluating the next object category.

## Generated-full candidate result

The manifest-only runner `run_a3d_generated_full.py` applies the same frozen
full/local equal-logit detector to every image in the three final
`generated_full_images/*` manifests without requiring a real counterpart or
localization mask. The verified 2026-07-26 run covers all 807 final-manifest
images: 275 mouse, 272 cat, and 260 trash-can. The current recursive tree also
contains 86 QC retry/backup intermediates that are not part of those final
manifests and are not included in this result.

At the fixed mouse-reference threshold `0.6353510120379108`, equal-logit fusion
detects 114/807 (14.13%) generated-full images. The local-only branch detects
197/807 (24.41%). These are detected fractions, not AUROC/AP, because the
generated-full manifests contain only generated images. The durable bundle is
under `results/our_defense/generated_full_images_a3d_20260726/`.

## Method

For each input image:

1. Run one full-image TruFor pass and retain its manipulation and reliability
   maps.
2. Cover the native image with 512x512 windows at stride 384. This defines
   candidates without running the model on every crop.
3. Score each candidate using the mean of the top 0.1% full-pass evidence
   values, weighted by `0.25 + 0.75 * reliability`.
4. Re-run frozen TruFor on only the four strongest candidates.
5. Average the full-image and maximum-crop logits with equal weight and apply a
   sigmoid. This fused value is the primary image-detection score.
6. Keep the full-image map as the stable localization output. The
   highest-scoring crop, top-two, and all-four maps are retained as small-edit
   diagnostics; they are not universal replacements for the global map.
7. Calibrate the deployment threshold once on a held-out real-image
   calibration split. Do not update model weights or thresholds per test image.

Ground truth is used only after inference for metrics and proposal-recall
diagnostics. It does not affect proposal generation, crop ranking, or output
scores.

## Main results

Q1 was used to develop the crop proposal/local-only rule. Equal-logit fusion
was later selected after aggregate cat/trash inspection and is therefore a
post-selection rule that requires a future unseen category for a strict blind
claim.

| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 275 | 0.8180 | 0.8798 | 0.8796 | 0.4436 | 0.6727 | 0.6655 |
| Q1 smallest | 55 | 0.7226 | 0.8600 | 0.8387 | 0.2182 | 0.6000 | 0.4545 |
| Q2-Q5 larger | 220 | 0.8537 | 0.8855 | 0.8946 | 0.5182 | 0.6955 | 0.7500 |

Paired bootstrap with 2,000 resamples gives the following fused-minus-full
detection 95% confidence intervals:

| Scope | AUROC | AP | TPR@5%FPR |
|---|---:|---:|---:|
| All | [0.0448, 0.0788] | [0.0462, 0.0784] | [0.1345, 0.3236] |
| Q2-Q5 larger | [0.0238, 0.0606] | [0.0286, 0.0636] | [0.1227, 0.3136] |

For the mouse-only top-1 crop-map diagnostic, local-minus-full 95% intervals
are [0.1003, 0.1449] pixel AP and [0.0392, 0.0835] pixel F1. This localization
rule does not generalize to larger cat/trash edits, so it is not the primary
map.

The improvement is concentrated where the benchmark is hardest:

| Edit-size quintile | Median edit fraction | Full AUROC | Local AUROC | Fused AUROC | Full pixel F1 | Top-1 pixel F1 | Four-proposal hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.05347% | 0.7226 | 0.8600 | 0.8387 | 0.3468 | 0.4860 | 0.9091 |
| Q2 | 0.07780% | 0.8155 | 0.8762 | 0.8889 | 0.4774 | 0.5502 | 0.9091 |
| Q3 | 0.11264% | 0.8575 | 0.8855 | 0.8926 | 0.5196 | 0.5770 | 0.9818 |
| Q4 | 0.16293% | 0.8489 | 0.8504 | 0.8648 | 0.5306 | 0.5680 | 1.0000 |
| Q5 | 0.37441% | 0.9412 | 0.9383 | 0.9438 | 0.6260 | 0.6181 | 1.0000 |

Q5 is already near saturation and does not benefit from zooming. A production
defense should retain both the global and local outputs instead of claiming
that the local branch universally replaces the global branch.

The actual four-crop implementation has median end-to-end latency 0.509 seconds
per image in this environment. The exhaustive diagnostic scan takes about
1.28 seconds per difficult Q1 image.

## Calibration result

The canonical set has a deterministic task-ID hash split with 80 calibration
pairs and 195 test pairs. A threshold set just above the empirical 95th
percentile of calibration-real fused scores is `0.6353510120379108` and gives:

- test FPR: 5.64%
- test TPR: 72.31%
- test AUROC: 0.8737
- test TPR@5%FPR: 66.67%

The threshold is not adapted on individual test inputs.

## JPEG-90 stress test

The same images were recompressed in memory at JPEG quality 90 before all
inference. This changes the threat condition but preserves image geometry.

| Scope | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|
| All | 0.5662 | 0.6166 | 0.6101 | 0.1164 | 0.1745 | 0.1782 |
| Q2-Q5 larger | 0.5795 | 0.6221 | 0.6187 | 0.1318 | 0.2045 | 0.2000 |

Fused A3D still improves detection ranking relative to full-image TruFor, but
the absolute detector is weak and localization gains are not statistically
stable.
Two failures occur:

- proposal failure: JPEG suppresses the full-pass evidence at the edit;
- crop-classifier failure: a selected crop covers the edit, but its score is
  indistinguishable from the real-image crop.

The next model iteration should train proposal and crop heads with matched JPEG,
resize, blur, and screenshot augmentations. A complementary proposal source
that is not derived from TruFor would also reduce correlated proposal failures.

## Reproduction

The verified checkpoint is:

```text
/root/.cache/claimforge/checkpoints/trufor/weights/trufor.pth.tar
SHA-256 ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844
```

The pinned source is:

```text
/root/.cache/claimforge/third_party/TruFor
commit ae54475df6f41a491d7615100feb19263dec13f7
```

TruFor's repository license is informational and nonprofit only; check it
before using this implementation beyond research evaluation.

Run the deployable clean condition:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.run_trufor_a3d \
  --run-id claim_a3d_deployable_all_20260725 \
  --scope all \
  --output-dir results/our_defense/mouse_a3d_20260725/clean \
  --device cuda:0
```

Run the JPEG stress test by adding `--jpeg-quality 90`.

Build the provenance-aware final cat/trash canonical sets:

```bash
/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.build_final_canonical \
  --category cat \
  --output-dir outputs/our_defense/canonical/cat_final_251_v1

/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.build_final_canonical \
  --category trash_can \
  --output-dir outputs/our_defense/canonical/trash_can_final_250_v1
```

Pass the corresponding `pairs.jsonl` to `run_trufor_a3d`. For example:

```bash
/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.run_trufor_a3d \
  --pairs outputs/our_defense/canonical/cat_final_251_v1/pairs.jsonl \
  --run-id cat_final_251_clean_a3d_v1 \
  --output-dir results/our_defense/cat_trash_a3d_20260725 \
  --device cuda:0
```

Generate paired-bootstrap and slice analysis:

```bash
/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.analyze_trufor_a3d \
  results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.jsonl \
  --output-json results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.analysis.json \
  --output-markdown results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.analysis.md \
  --bootstrap-replicates 2000 \
  --calibration-results results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.jsonl
```

## Artifacts

- `eval/our_defense/run_trufor_adaptive_zoom.py`: GT-centered oracle scale
  diagnostic; never report it as a deployable result.
- `eval/our_defense/run_trufor_adaptive_scan.py`: exhaustive and proposal-budget
  diagnostic.
- `eval/our_defense/build_final_canonical.py`: provenance-aware cat/trash input
  and ground-truth builder.
- `eval/our_defense/run_trufor_a3d.py`: deployable four-crop A3D runner.
- `eval/our_defense/analyze_trufor_a3d.py`: slice, failure, and paired-bootstrap
  analysis.
- `results/our_defense/mouse_a3d_20260725/`: durable clean, JPEG-90, and Q1
  diagnostics.
- `results/our_defense/a3d_aggregate_20260727/`: unified metrics and checksums.
- `docs/A3D_ADAPTIVE_DEFENSE_FULL_REPORT_2026-07-27.md`: complete report.

The `outputs/` tree remains a scratch area. All report-critical per-image
predictions, summaries, analyses, and checksums are now materialized under
`results/our_defense/`.

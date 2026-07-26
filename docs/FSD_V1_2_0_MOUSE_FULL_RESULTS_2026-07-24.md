# FSD v1.2.0 on the canonical mouse set (2026-07-24)

## 1. Status and headline

The first publicly reproducible whole-image AIGC baseline is complete.
The official FSD v1.2.0 inference release ran on all 275 matched tasks, or
550 canonical JPEG images, with no errors.

FSD is essentially unable to distinguish CLAIMFORGE's small local diffusion
insertions from their matched real images:

- image AUROC is **0.500350**, with task-paired bootstrap 95% CI
  **[0.497771, 0.502552]**;
- average precision is **0.502708**, CI **[0.501560, 0.508887]**;
- TPR at a real-only 5% FPR operating point is **4.7273%**, CI
  **[4.0000%, 5.4545%]**;
- accuracy at the released strict threshold `ai_score > 2` is
  **50.3636%**, CI **[50.0000%, 50.9091%]**;
- the released threshold detects 74/275 forged images but also flags
  72/275 matched real images;
- a forged image scores strictly above its matched real image in only
  79/275 pairs, while 150 score lower and 46 tie; and
- the mean paired score change is only **0.001289**, CI
  **[-0.006270, 0.010929]**.

The failure is not explained only by FSD's center crop. AUROC is
`0.500136` on the 192 pairs whose entire exact-difference GT enters the
effective crop, `0.498270` on the 34 partially visible pairs, and
`0.500208` on the 49 pairs with no positive GT pixel center inside it.

The independent analyzer validated the complete physical history, source,
weights, inputs, manifests, and all 550 stored 960-dimensional float64
descriptors. It independently replayed all 20 released transforms, the tied
GMM, z-normalization, score sign, and strict decision. The maximum raw-score
and final-score replay errors are both exactly `0.0`.

FSD is a whole-image classifier. It has no native dense manipulation output,
so T2 and the joint score are **N/A**. Crop visibility is an input-condition
diagnostic, not a predicted localization result.

This completes FSD's **local-splice condition only**. The required
same-domain fully synthetic control has not yet been built, so this result
does not establish that the released model is generally weak at its intended
whole-image task.

## 2. Pinned paper, source, release, and license

The method is described in the
[CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Nguyen_Forensic_Self-Descriptions_Are_All_You_Need_for_Zero-Shot_Detection_Open-Set_CVPR_2025_paper.html),
“Forensic Self-Descriptions Are All You Need for Zero-Shot Detection,
Open-Set Source Attribution, and Clustering of AI-generated Images.”

The run uses the authors'
[official repository](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25)
at commit `50f2eae06efdac2e5a33f407ca9a27a2295133ac`, together with the
[official v1.2.0 release assets](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/releases/tag/v1.2.0).

The selected artifacts were frozen before the full result:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 634 | `7cc34433045adb998762e00de7de25c50f9c1e10dbac1c18899c6c63c4cfafe4` |
| `fre.pt` | 9,861 | `d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb` |
| `gmm.pt` | 14,786,229 | `0f9fa030a3d5816266d0329fd0fb614b65e322d4bda6d083613c713bfe9bc829` |
| `fsd_transforms.pt` | 42,177,409 | `1e87d792b413101e58d9de71551182a1fab8b879ca6f6ba9780b6adcb9a5a699` |

The registered four-file bundle SHA-256 is
`3c1959f0092fdbe681e41c96f12c1b6d3762e46f21b37b08d4a7e617d1acdfce`.
All three PT files report an empty unsafe-global list and are loaded with
`weights_only=True`. The adapter requires an explicit local weights
directory and never invokes the release's automatic downloader.

The pinned source is one commit ahead of the `v1.2.0` tag
`5b317a00251988b5ec5a47317f4d82e5bdfd009d`. The changed files are
`fsd/__init__.py`, `fsd/attribution.py`, and `fsd/weights.py`; none is in
the detection-math path used by `FSDDetector.score`.

The repository license is
[CC BY-NC-SA 4.0](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/blob/50f2eae06efdac2e5a33f407ca9a27a2295133ac/LICENSE).
It is non-commercial and share-alike, so “publicly reproducible” must not be
misread as unrestricted commercial use.

## 3. What FSD does and why the idea is strong

FSD is based on a useful forensic intuition: instead of learning the visual
semantics of every generator, describe the image's own low-level statistical
regularities and ask whether they resemble real-image regularities.

The released detector:

1. converts the image to grayscale;
2. applies eight learned forensic residual filters;
3. at three scales, solves a constrained local linear-prediction problem
   directly on the test image;
4. concatenates the resulting coefficients into a 960-dimensional
   “forensic self-description”;
5. passes that vector through 20 released residual MLP transforms; and
6. evaluates it under a five-component tied-covariance GMM calibrated on
   real-image statistics.

An AI-generated image should receive unusually low likelihood under that real
distribution. The release normalizes the raw GMM likelihood using

```text
z = (raw_likelihood - (-42.25325127289017)) / 706.0556010649537
```

and declares fake only when `z < -2`. CLAIMFORGE stores the monotonic
orientation `ai_score = -z`, so the exactly equivalent frozen rule is the
strict comparison `ai_score > 2`.

This approach is attractive for open-set detection because it does not need
to name the generator and is much less dependent on object semantics than a
standard classifier. The paper reports an average AUROC of about 0.960 across
its four whole-image benchmarks. That paper number is context only; it is not
a result reproduced by this run.

The same global-statistics design explains the CLAIMFORGE failure. A typical
mouse insertion changes only a tiny fraction of the image. The global
self-description remains dominated by the untouched camera photograph, and
the forged and real score distributions become almost identical. In 49
pairs the exact edit is outside the effective crop altogether; even among the
192 fully visible pairs, however, AUROC remains 0.5001.

## 4. Material paper/release drift

This run must be labeled **FSD — official v1.2 inference release**, not a
strict reproduction of the CVPR implementation.

The public release differs materially from the method described in the paper:

| Component | CVPR paper | v1.2 inference release |
|---|---|---|
| FRE neighborhood | 11×11 description | stored FRE weight shape `8×1×15×15` |
| Per-image solver | AdamW with scheduling, up to 10,000 iterations | closed-form float64 KKT constrained least squares |
| GMM input | paper self-description directly | 20 residual MLP transforms before GMM |
| Learned pre-GMM transform size | not in the paper chain | 5,267,200 parameters |
| Public training/evaluation scripts | paper protocol described | not included in the release repository |
| Frozen decision | older paper normalization/operating discussion | release `config.json`: strict `z < -2` |

The source and weights are official and fully pinned, but the release does not
provide enough training machinery to independently prove that these exact
weights were trained with only real images. The valid claim is therefore that
CLAIMFORGE reproduces and audits the official v1.2 inference API—not that it
reconstructs the paper's training or its reported 0.960 AUROC.

## 5. Exact inference and crop-visibility contract

For every canonical JPEG, the frozen runner:

1. opens the exact path with Pillow and converts it to mode `L`;
2. does not apply EXIF transpose or ICC color conversion;
3. keeps the grayscale input as uint8 values in `[0, 255]`;
4. applies the released 15×15 FRE and removes its seven-pixel border;
5. bilinearly resizes the residual so its short side is 1024, using
   `align_corners=False`, `antialias=False`, and Python `round`;
6. takes the centered 1024×1024 crop;
7. constructs three bilinear scales;
8. solves the released float64 KKT system to obtain one 960-vector; and
9. calls the official `FSDDetector.score` once.

The runner captures the raw 960-vector during that one official call. It then
independently applies the same released transforms and GMM once more and
requires exact equality with the high-level API before writing a successful
row. It stores raw likelihood, released z-score, sign-inverted AI score, both
strict decisions, the descriptor, preprocessing geometry, and all artifact
hashes.

Crop visibility is computed from the forged exact-difference GT and copied to
both members of the matched pair. For every positive native pixel center:

```text
d = (native_index - 7 + 0.5) * resized_size / (native_size - 14) - 0.5
```

The center is visible only if it falls inside the official center crop. The
pair category is:

- `none` if zero positive GT pixel centers enter the crop;
- `full` if all enter it; and
- `partial` otherwise.

This is deliberately described as pixel-center visibility. A `none` edit
close to a boundary can still have a tiny numerical influence through the
FRE support or bilinear interpolation.

## 6. Preflight, smoke, resume, and prefix checks

The final-code one-forged-image preflight completed without error:

- `ai_score = 0.349775234168`, below the fake threshold;
- the edit category is `none`;
- forward latency is 378.316 ms; and
- the independent analyzer replays the raw and AI scores with zero error.

The five-pair smoke completed 10/10 images:

- AUROC `0.48`;
- average precision `0.514286`;
- released-threshold accuracy `0.50`;
- one paired win, two losses, and two exact ties; and
- three `full` and two `none` crop-visibility pairs.

Running the exact smoke command again reported ten selected and zero pending.
The JSONL remained exactly ten physical rows, proving the success-resume path
does not reload the model, re-run inference, or append duplicates.

The smoke and full runs use distinct run IDs, manifest fingerprints, result
paths, and descriptor paths. The independent prefix audit compared all ten
ordered shared images and proved exact descriptor hashes, values, and scores.
It also explicitly rejects copied full-run rows.

## 7. Primary T1 result

Confidence intervals use 1,000 task-paired bootstrap resamples with seed
`20260724`. Each replicate reselects the 5% FPR threshold using real scores
only and NumPy's `method="higher"` quantile rule.

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| AUROC | **0.500350** | [0.497771, 0.502552] |
| Average precision | **0.502708** | [0.501560, 0.508887] |
| TPR at real-only 5% FPR | **0.047273** | [0.040000, 0.054545] |
| Accuracy at strict `ai_score > 2` | **0.503636** | [0.500000, 0.509091] |
| Balanced accuracy at strict `> 2` | **0.503636** | [0.500000, 0.509091] |
| Paired strict ranking accuracy | **0.287273** | [0.236364, 0.341818] |
| Mean forged-minus-real score | **0.001289** | [-0.006270, 0.010929] |

The fixed-threshold confusion matrix is:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 74 | 72 | 201 | 203 |

This gives TPR `0.269091`, FPR `0.261818`, precision `0.506849`,
specificity `0.738182`, and F1 `0.351544`.

The real-only 95th-percentile threshold is `4.474811`. Using the same strict
`>` rule yields 13/275 forged detections and 13/275 real false positives:
TPR and actual FPR are both `0.047273`.

The paired comparison has 79 wins, 150 losses, and 46 exact ties. The
two-sided exact sign-test p-value is `3.1424e-6`, but its direction is
opposite the desired detector behavior: among non-ties, the local insertion
more often lowers the AI score than raises it. The mean delta nevertheless
crosses zero because a small number of larger positive changes offset many
small negative changes.

## 8. Domain and crop-visibility diagnostics

| Slice | Pairs | AUROC | AP | TPR at 5% FPR | Paired strict ranking |
|---|---:|---:|---:|---:|---:|
| Lodging | 147 | 0.500972 | 0.504026 | 0.047619 | 0.306122 |
| Restaurant | 128 | 0.499298 | 0.503800 | 0.046875 | 0.265625 |
| Edit fully visible | 192 | 0.500136 | 0.503710 | 0.046875 | 0.328125 |
| Edit partially visible | 34 | 0.498270 | 0.526242 | 0.058824 | 0.411765 |
| Edit not visible | 49 | 0.500208 | 0.500817 | 0.040816 | 0.040816 |

Neither domain separates. Full edit visibility also does not rescue the
detector, which is important evidence that center cropping is a real
limitation but not the primary explanation for the near-random aggregate
result.

The score distributions nearly overlap:

| Kind | Mean | Median | P05 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Real | 1.507691 | 1.076832 | -0.140498 | 4.395614 | 11.755709 |
| Forged | 1.508980 | 1.071410 | -0.142457 | 4.351354 | 11.648400 |

## 9. What this result does and does not prove

The supported conclusion is:

> The official FSD v1.2 whole-image inference release does not transfer to
> CLAIMFORGE's small local diffusion insertion threat model; matched real and
> forged scores are effectively indistinguishable.

It does not yet support:

- “FSD is a bad whole-image AIGC detector”;
- “the CVPR paper result is wrong”;
- “the release exactly implements the paper”; or
- “crop exclusion alone caused the failure.”

The required next control is approximately 150 same-domain fully synthetic
lodging/restaurant scenes with matched, identically encoded real images. The
same released score and frozen metrics must be applied without tuning. Only
that contrast can show whether the detector remains strong for full synthesis
but fails when the synthetic fraction becomes tiny.

## 10. Runtime, tests, and artifacts

The recorded environment is CPython 3.12.3, PyTorch `2.10.0+cu128`,
CUDA 12.8, NumPy 2.4.3, and Pillow 12.1.1 on an NVIDIA L20Z using
`cuda:4`. cuDNN is deterministic; benchmarking and TF32 are disabled; the
float32 matmul precision is `highest`.

Forward-only timing over 550 images:

| Statistic | Milliseconds |
|---|---:|
| Median | 115.047 |
| Mean | 126.617 |
| P95 | 318.040 |
| Maximum | 392.574 |

Timing encloses the official high-level score call plus CUDA synchronization.
It excludes image decode, static provenance checks, descriptor writes,
bootstrap metrics, and independent analysis.

Maximum allocated CUDA memory is 11,273,897,984 bytes, approximately
10.50 GiB. The full descriptor directory contains 550 lossless `.npy`
vectors and occupies 4,294,400 bytes.

The focused FSD suite reports **52 passed**:

- `tests/test_whole_image_metrics.py`
- `tests/test_run_fsd.py`
- `tests/test_analyze_fsd_run.py`

Primary machine-readable files:

- `results/opensource/fsd/fsd_v1_2_0_mouse_canonical_v1_full275_20260724.jsonl`
- `results/opensource/fsd/fsd_v1_2_0_mouse_canonical_v1_full275_20260724.run_manifest.json`
- `results/opensource/fsd/fsd_v1_2_0_mouse_canonical_v1_full275_20260724.summary.json`
- `results/opensource/fsd/fsd_v1_2_0_mouse_canonical_v1_full275_20260724.analysis.json`
- `outputs/opensource/fsd/fsd_v1_2_0_mouse_canonical_v1_full275_20260724/raw_descriptors/`

Their result, manifest, summary, and analysis SHA-256 values are,
respectively:

```text
6eb5877ea28145fe85800625d26a10e1b343176de0e527bdbc35aa0ef20f665f
f0123259a7e14cf25c834eecbe4922dc9b6e784cc220bd23380ad3b45b6fdf02
fe93b562c77c174eaa20afb4f40b43ccbec02bceddfbc3031b54538f2f584809
57e88d4735799048d1082ef9fde360343405ebd490f1f7bcc7f732c26347c455
```

Implementation:

- `eval/opensource/run_fsd.py`
- `eval/opensource/whole_image_metrics.py`
- `eval/opensource/analyze_fsd_run.py`

## 11. Reproduction

Full inference or artifact-validating resume:

```bash
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/fsd-v1.2.0/bin/python \
  -m eval.opensource.run_fsd \
  --repo-root /root/claimforge-benchmark \
  --weights-dir /root/.cache/claimforge/checkpoints/fsd-v1.2.0 \
  --run-id fsd_v1_2_0_mouse_canonical_v1_full275_20260724 \
  --device cuda:4 \
  --bootstrap-samples 1000 \
  --fail-fast
```

Independent analysis and prefix audit:

```bash
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/fsd-v1.2.0/bin/python \
  -m eval.opensource.analyze_fsd_run \
  --repo-root /root/claimforge-benchmark \
  --run-id fsd_v1_2_0_mouse_canonical_v1_full275_20260724 \
  --results-dir results/opensource/fsd \
  --inputs outputs/opensource/mouse_canonical_v1/inputs.jsonl \
  --fsd-root \
    /root/.cache/claimforge/third_party/Forensic-Self-Descriptions-CVPR25-50f2eae \
  --prefix-run-id fsd_v1_2_0_mouse_canonical_v1_smoke5_20260724 \
  --prefix-results-dir results/opensource/fsd
```

The tools must be invoked as modules from the repository environment. Direct
`python eval/opensource/run_fsd.py` execution is not the supported entry
point because the repository package root would not be on `sys.path`.

## 12. Bottom line and next method

FSD provides the clean first result for the whole-image track: an audited,
official whole-image detector with a strong paper motivation produces
near-perfect real/forged score overlap on tiny local AI insertions.

For the CLAIMFORGE table, report it as:

> **FSD official v1.2 inference release; T1 only; local-splice AUROC
> 0.5004; no T2; non-commercial CC BY-NC-SA license; public release differs
> materially from the CVPR implementation; fully synthetic control pending.**

The next frozen whole-image method is **UniversalFakeDetect**, the canonical
CLIP linear-probe baseline. The entire Track B result remains incomplete
until FSD and the other selected detectors are also run on the same-domain
fully synthetic control.

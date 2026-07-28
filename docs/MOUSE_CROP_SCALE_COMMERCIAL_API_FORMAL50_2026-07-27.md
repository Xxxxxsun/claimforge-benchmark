# Mouse crop-scale commercial API experiment (formal N=50)

**Date:** 2026-07-27

## Research question

This experiment evaluates the strongest simple crop-based response to
CLAIMFORGE:

1. Assume an oracle already knows the exact edited region.
2. Crop that region and submit it to a commercial detector.
3. Expand the crop field of view and measure when the recovered signal is
   lost.
4. Apply the same crop-and-resize procedure to an unchanged region from the
   same image and measure false alarms.
5. Compare a tight crop resized to 512x512 with the same crop at native size
   to isolate the effect of magnification.

The experiment reports detector positives, unchanged-region false positives,
and API input rejection separately.

## Formal protocol

- Base cell: frozen `local_splice/mouse` benchmark.
- Eligible tasks: 236 of 250. An eligible image supports an 8x crop around
  the exact edit and an unchanged 8x control region.
- Selected tasks: 50, with 25 restaurant and 25 lodging images, selected at
  deterministic per-domain quantiles of exact-difference region size.
- Tight crop side:
  - lodging: 26-104 px, median 51 px;
  - restaurant: 26-132 px, median 53 px.
- Suspicious crop: minimum square enclosing the exact RGB-difference bbox.
- Field-of-view factors: 1x, 1.5x, 2x, 3x, 4x, and 8x.
- Real control: concentric crops inside a deterministic, far-away 8x square
  containing zero exact-difference pixels in the same image.
- Main rendering: bicubic resize to 512x512, metadata-free JPEG Q95, 4:4:4
  subsampling.
- Native ablation: the 1x crop is encoded at its original pixel dimensions,
  with no interpolation.
- Request order: cyclically interleaved across domains, scales, and arms so a
  quota-limited prefix is not concentrated in one condition.

The main experiment contains:

- 50 tasks x 6 factors x 2 arms = 600 resized inputs;
- 50 tasks x 1 native condition x 2 arms = 100 native inputs;
- 700 inputs per API when the provider accepts native dimensions.

## Provider decision definitions

- **AI or Not:** provider `ai_detected` decision.
- **Alibaba Ultra:** any returned `risk_edit`, `risk_fake`, or `risk_aigc`
  label. This is a broad commercial risk flag, not a pure AI-generation
  decision.
- **Copyleaks Ultra:** provider `isAiDetected` decision. Copyleaks requires
  both dimensions to be at least 512 px, so the native-size condition is not
  applicable and was not submitted.

## Execution status

| Service | Applicable | Valid | Rejected/error | Missing | Status |
|---|---:|---:|---:|---:|---|
| Alibaba Ultra | 700 | 700 | 0 | 0 | Complete |
| Copyleaks Ultra | 600 | 312 | 288 | 0 | All applicable requests completed |
| AI or Not | 700 | 700 | 0 | 0 | Complete |

AI or Not completed all 700 applicable inputs. Its append-only raw log retains
two historical HTTP 402 `INSUFFICIENT_BALANCE` attempts from interrupted
runs; both tasks were retried successfully. The normalized analysis uses the
latest response per task and contains 700 valid verdicts.

Copyleaks completed all 600 resized requests. It rejected 255 inputs as
`image_blurry` and 33 as `insufficient_colors`. Rejections are abstentions,
not negative detections.

Hive, Sightengine, and Resemble were not rerun on the N=50 manifest because
the available accounts had already reached quota or balance limits during the
pilot. Their N=8 results remain in the pilot report and must not be presented
as N=50 results.

## Raw resized-crop results

Each cell is `positive / valid`. These rates condition on the API returning a
valid verdict. Real-control positives are false alarms for this experiment.

| Service / crop | 1x | 1.5x | 2x | 3x | 4x | 8x |
|---|---:|---:|---:|---:|---:|---:|
| Alibaba suspicious | 17/50 | 11/50 | 14/50 | 24/50 | 31/50 | 19/50 |
| Alibaba real control | 12/50 | 14/50 | 10/50 | 15/50 | 16/50 | 9/50 |
| AI or Not suspicious | 31/50 | 13/50 | 9/50 | 6/50 | 1/50 | 0/50 |
| AI or Not real control | 15/50 | 13/50 | 7/50 | 3/50 | 1/50 | 1/50 |
| Copyleaks suspicious | 2/2 | 9/11 | 17/23 | 22/39 | 24/47 | 7/50 |
| Copyleaks real control | 0/3 | 0/10 | 0/17 | 0/29 | 0/34 | 0/47 |

For Alibaba, the paired suspicious-minus-control positive-rate gaps are:

| Factor | Gap | Exact McNemar p |
|---|---:|---:|
| 1x | +10 pp | 0.405 |
| 1.5x | -6 pp | 0.607 |
| 2x | +8 pp | 0.454 |
| 3x | +18 pp | 0.093 |
| 4x | +30 pp | 0.0081 |
| 8x | +20 pp | 0.0309 |

The 4x gap remains below a six-factor Bonferroni threshold of 0.00833. This
does not make the full curve monotonic or turn Alibaba into a calibrated
binary AI detector.

For AI or Not, the paired suspicious-minus-control positive-rate gaps are:

| Factor | Gap | Exact McNemar p |
|---|---:|---:|
| 1x | +32 pp | 0.00154 |
| 1.5x | 0 pp | 1.000 |
| 2x | +4 pp | 0.791 |
| 3x | +6 pp | 0.508 |
| 4x | 0 pp | 1.000 |
| 8x | -2 pp | 1.000 |

Only the 1x condition clearly separates suspicious and unchanged crops. The
gap disappears after adding modest surrounding context.

## Copyleaks applicability

Copyleaks acceptance changes strongly with field of view. Each cell reports
`valid / attempted`; the second percentage is the positive rate among valid
inputs.

| Factor | Suspicious acceptance | Suspicious positive | Control acceptance | Control positive |
|---|---:|---:|---:|---:|
| 1x | 2/50 (4%) | 2/2 (100%) | 3/50 (6%) | 0/3 |
| 1.5x | 11/50 (22%) | 9/11 (82%) | 10/50 (20%) | 0/10 |
| 2x | 23/50 (46%) | 17/23 (74%) | 17/50 (34%) | 0/17 |
| 3x | 39/50 (78%) | 22/39 (56%) | 29/50 (58%) | 0/29 |
| 4x | 47/50 (94%) | 24/47 (51%) | 34/50 (68%) | 0/34 |
| 8x | 50/50 (100%) | 7/50 (14%) | 47/50 (94%) | 0/47 |

Copyleaks therefore has a clear operational trade-off: tight crops preserve
the strongest local signal but are usually rejected, while larger crops are
accepted more often but dilute the detection rate. The 1x rate of 2/2 must
not be read as 100% recall because 48/50 suspicious inputs received no
verdict.

## Native-size ablation

Alibaba and AI or Not have complete native and resized pairs.

| Service / arm | Native 1x | Same crop resized to 512 | Resize-only positives | Paired p |
|---|---:|---:|---:|---:|
| Alibaba suspicious | 1/50 (2%) | 17/50 (34%) | 16 | 0.000031 |
| Alibaba real control | 9/50 (18%) | 12/50 (24%) | 5 | 0.453 |
| AI or Not suspicious | 0/50 (0%) | 31/50 (62%) | 31 | <0.000000001 |
| AI or Not real control | 3/50 (6%) | 15/50 (30%) | 15 | 0.00754 |

For suspicious crops, resizing changes the decision on 16 tasks from negative
to positive and never changes a positive native decision to negative. For
real controls, five are positive only after resizing and two are positive
only at native size, giving a non-significant net increase of 6 percentage
points.

For AI or Not, no native suspicious crop is positive, while 31 of the same
crops become positive after resizing. Real-control positives also rise from
3/50 to 15/50. Thus magnification creates detector signal in both arms, not
only in the edited pixels.

Copyleaks is excluded from this ablation because its contract requires a
minimum 512x512 input. Automatically enlarging the native files would erase
the intended no-resize condition.

## Findings

### 1. There is no stable expansion threshold

**Observation.** Alibaba's suspicious positive rate is 34%, 22%, 28%, 48%,
62%, and 38% from 1x through 8x. Thirty-nine of its 50 complete task curves
contain at least one negative-to-positive reversal. Of the 17 tasks detected
at 1x, only 5 remain positive at 1.5x, but 10 are positive at 3x and 9 at 4x.

AI or Not follows a different aggregate curve: 62%, 26%, 18%, 12%, 2%, and
0%. Of the 31 tasks detected at 1x, 10, 8, 5, 1, and 0 remain positive at the
larger factors. Six of 50 task-level curves are nevertheless non-monotonic.

**Interpretation.** The provider decision is not a monotonic function of edit
dilution. Image content, magnification, and the amount of surrounding context
all affect the risk label. AI or Not shows a strong dilution trend, while
Alibaba does not, so there is no provider-independent expansion threshold.

**Implication.** The paper should not claim that an edit becomes undetectable
at one universal factor such as 2x or 4x. A defensible claim is that the
crop-defense outcome is provider- and preprocessing-dependent.

**Next test.** Complete Sightengine, Hive, and Resemble on the same fixed
manifest before making a broader cross-provider population claim.

### 2. Magnification is part of the detector signal

**Observation.** Alibaba detects only 1/50 native suspicious crops but 17/50
of the exact same crops after bicubic resizing to 512x512. AI or Not changes
even more sharply, from 0/50 to 31/50. Both paired changes are statistically
clear.

**Interpretation.** The commercial detector is not simply recovering a
property that is invariant in the edited pixels. Enlarging the generated
mouse texture makes the signal much more visible to both providers.

**Implication.** A crop-defense evaluation must specify its resize policy.
Submitting the native crop and submitting a 512x512 enlargement are different
detector interventions.

**Next test.** Add a no-resample padded-512 condition. It would hold canvas
size constant while separating interpolation from provider behavior on small
input dimensions.

### 3. Unchanged crops can receive substantial commercial risk

**Observation.** Alibaba flags 18%-32% of resized real controls at every
factor. At 1x, suspicious and real-control rates are 34% and 24%; at 1.5x the
real-control rate is higher, 28% versus 22%. AI or Not flags 30% of resized
1x real controls and 26% at 1.5x, compared with suspicious rates of 62% and
26%. Native controls are flagged more often than native suspicious crops for
both Alibaba (18% versus 2%) and AI or Not (6% versus 0%).

**Interpretation.** Cropping a real region can move it outside the detector's
training distribution or expose texture/content that the broad Alibaba risk
service labels as edited or fake. The native result shows that bicubic
interpolation alone does not explain all control positives.

**Implication.** A system that searches many candidate windows and trusts any
positive verdict can accumulate false alarms even when the image region is
unchanged.

**Next test.** Sample multiple unchanged windows per image and report
image-level family-wise false-positive rates, not only per-window rates.

### 4. Input rejection is a first-class defense failure mode

**Observation.** Copyleaks rejects 95% of 1x inputs, 79% of 1.5x inputs, and
60% of 2x inputs across the two arms. Acceptance rises with context, while
the suspicious positive rate among accepted inputs declines to 14% at 8x.

**Interpretation.** The crop must be large and detailed enough to pass the
provider's quality gate, but that enlargement can weaken the localized AI
signal.

**Implication.** Reporting only positives among valid responses overstates
small-crop defense performance. Coverage, rejection reasons, and conditional
detection must be shown together.

**Next test.** Treat provider abstention explicitly in any downstream
decision rule and compare reject-as-negative, reject-as-suspicious, and
human-review policies.

## Answer to the crop-defense objection

Oracle crop-and-query is useful as an additional analysis, but the formal
experiment does not support it as a complete countermeasure:

1. The smallest resized crop is detected in only 34% of cases by Alibaba and
   62% by AI or Not, even with perfect localization.
2. Enlarging the crop does not produce a stable, monotonic evasion boundary.
3. The same procedure assigns substantial risk to unchanged regions for
   Alibaba and AI or Not.
4. Copyleaks often abstains on the crops that should preserve the strongest
   localized signal.
5. Decisions change materially under a resize operation that contains no new
   semantic evidence.

The benchmark should report a three-part crop-defense result:

- suspicious-crop recovery;
- matched real-crop false alarms;
- API applicability or abstention.

## Scope and claim boundary

- Alibaba and AI or Not each supply a complete N=50 provider result.
- Copyleaks supplies a complete request set but only 312 valid verdicts; its
  conditional detection rates are subject to provider quality-gate selection.
- The selected 50 tasks cover restaurant and lodging scenes and a broad edit
  size range, but they are all mouse edits from the current local-splice
  generation pipeline.
- Exact McNemar tests are paired within task. Wilson intervals and all raw
  denominators are stored in the machine-readable summary.

## Artifacts

- Formal input manifest:
  `results/analysis/mouse_crop_scale_formal_v1/manifest.jsonl`
- Dataset/protocol summary:
  `results/analysis/mouse_crop_scale_formal_v1/summary.json`
- Resized visual audit:
  `results/analysis/mouse_crop_scale_formal_v1/contact_sheet_resized.jpg`
- Native ablation visual audit:
  `results/analysis/mouse_crop_scale_formal_v1/contact_sheet_native_ablation.jpg`
- Normalized per-input results:
  `results/analysis/mouse_crop_scale_formal_v1/commercial_joined.jsonl`
- Machine-readable statistics:
  `results/analysis/mouse_crop_scale_formal_v1/commercial_summary.json`
- Aggregate condition table:
  `results/analysis/mouse_crop_scale_formal_v1/commercial_by_condition.csv`
- Paired condition table:
  `results/analysis/mouse_crop_scale_formal_v1/commercial_paired_by_condition.csv`
- Detection curves:
  `results/analysis/mouse_crop_scale_formal_v1/commercial_detection_curves.svg`
  and `commercial_detection_curves.png`
- Raw provider responses:
  `results/commercial/crop_scale_formal_v1/`

## Reproduction

Build the frozen inputs:

```bash
.venv/bin/python -m eval.commercial.build_mouse_crop_scale_formal
```

Recompute all local statistics:

```bash
.venv/bin/python -m eval.commercial.analyze_mouse_crop_scale_formal
```

Provider credentials are read only from environment variables. No API key is
stored in the manifests, raw result files, summaries, or this report.

# Mouse crop-scale commercial API pilot

**Date:** 2026-07-27

## Question

This pilot tests the strongest simple version of the proposed crop defense:

1. Assume an oracle already knows exactly where the local edit is.
2. Crop the smallest region containing the edit and send only that crop to a
   commercial detector.
3. Expand the crop field of view until the detector loses the signal.
4. Apply the identical crop-and-resize pipeline to unchanged regions from the
   same images to measure false alarms and input rejection caused by cropping.

The experiment therefore measures both detection recovery on an oracle
suspicious crop and the cost of applying that defense to real image regions.

## Protocol

- Base data: frozen `local_splice/mouse` benchmark cell.
- Tasks: 8 stratified tasks, with 4 restaurant and 4 lodging images.
- Tight-region sizes: 33, 45, 56, and 84 pixels for lodging; 35, 48, 63,
  and 94 pixels for restaurant.
- Suspicious region: minimum square enclosing the exact RGB-difference bbox.
- Field-of-view factors: 1x, 2x, 4x, and 8x relative to the tight square.
- Real control: a same-size region from the same source image containing zero
  exact-difference pixels, selected deterministically far from the edit.
- Probe artifact: every crop is bicubic-resized to 512x512 and encoded as
  metadata-free JPEG Q95 with 4:4:4 subsampling.
- API upload: each service uses its existing benchmark adapter. Some adapters
  re-encode the stored JPEG, but the suspicious and real-control arms always
  receive the same service-specific transformation.
- Total target per service: 8 tasks x 4 factors x 2 crop types = 64 images.

The factor changes how much native image context is included. Output
resolution and encoding remain fixed. A 1x crop therefore undergoes the most
upsampling, while an 8x crop includes much more real context and needs less
upsampling.

## Execution coverage

| Service | Valid | Rejected/error | Missing | Status |
|---|---:|---:|---:|---|
| Sightengine | 64/64 | 0 | 0 | Complete |
| Alibaba Ultra | 64/64 | 0 | 0 | Complete |
| AI or Not | 64/64 | 0 | 0 | Complete |
| Copyleaks Ultra | 30/64 | 34 | 0 | All requests completed; partial applicability |
| Hive V3 | 50/64 | 1 quota error | 13 | Partial |
| Resemble Detect | 36/64 | 1 balance error | 27 | Partial |

Hive stopped at HTTP 429. Resemble stopped when the wallet balance was below
the per-image price. These partial rows are reported with their actual
denominators and are not silently treated as negatives.

The legacy Sightengine adapter labels the multipart part as `image/png` even
when the preserved input bytes are JPEG. Sightengine accepted and decoded all
64 files. The runner now derives the MIME type from the file extension; the
final paper-scale run should repeat this condition with the corrected adapter.

## Raw result table

Each cell is `positive / valid`. Parentheses show API rejections where they
matter. Real-control positives are false alarms for this experiment.

| Service / crop | 1x | 2x | 4x | 8x |
|---|---:|---:|---:|---:|
| Hive suspicious | 0/7 | 1/6 | 0/6 | 0/6 |
| Hive real control | 0/7 | 0/6 | 0/6 | 0/6 |
| Sightengine suspicious | 4/8 | 0/8 | 0/8 | 0/8 |
| Sightengine real control | 1/8 | 0/8 | 0/8 | 0/8 |
| Resemble suspicious | 4/5 | 5/5 | 4/4 | 2/4 |
| Resemble real control | 1/5 | 3/5 | 1/4 | 0/4 |
| Alibaba suspicious | 2/8 | 1/8 | 5/8 | 4/8 |
| Alibaba real control | 2/8 | 2/8 | 2/8 | 3/8 |
| AI or Not suspicious | 4/8 | 1/8 | 0/8 | 0/8 |
| AI or Not real control | 4/8 | 2/8 | 0/8 | 0/8 |
| Copyleaks suspicious | 0/0 (8 rejected) | 1/2 (6 rejected) | 4/8 | 1/8 |
| Copyleaks real control | 0/0 (8 rejected) | 0/2 (6 rejected) | 0/4 (4 rejected) | 0/6 (2 rejected) |

## Findings

### 1. Oracle cropping can recover signal, but only for some detectors

Sightengine detects 4/8 tight suspicious crops, even though its earlier
whole-image local-splice run detected 0 matched examples. However, all four
positives disappear as soon as the crop side expands to 2x.

AI or Not follows a similarly clear decay: 4/8 at 1x, 1/8 at 2x, and 0/8 at
4x and 8x. Three of its four 1x-positive task curves fail at 2x; the remaining
curve fails at 4x.

Copyleaks cannot evaluate any 1x suspicious crop. Once the crops become
acceptable, it detects 4/8 at 4x but only 1/8 at 8x. Hive recovers only one
positive, at 2x. Resemble retains signal farther out, but its sample is
partial. Alibaba is strongly non-monotonic, so it does not provide a clean
single expansion breakpoint.

There is therefore no detector-independent answer such as "the edit becomes
undetectable at 4x." The useful statement is service-specific: 2x already
removes all recovered Sightengine detections, while 4x removes all AI or Not
detections in this pilot.

### 2. Real crops acquire substantial false-risk scores

The objection assumes that only the suspicious crop will trigger the API.
That assumption fails for several services:

- AI or Not marks 4/8 real 1x crops positive, exactly the same count as the
  suspicious arm. At 2x it marks 2/8 real crops positive versus 1/8 suspicious.
- Alibaba flags 2/8, 2/8, 2/8, and 3/8 real controls across the four factors.
- Resemble flags 1/5 real controls at 1x and 3/5 at 2x.
- Sightengine produces one real-control false alarm at 1x.

Hive produces no real-control positives in its available prefix. Copyleaks
produces no positive valid controls, but frequently refuses to evaluate them.

These results support the narrower claim that unchanged regions can receive
commercial-detector risk after the crop-and-resize pipeline. Without an
unresized control, this pilot does not separate interpolation artifacts from
low-texture content or other crop-distribution effects, and it does not show
that every detector has this failure mode.

### 3. Input rejection is part of the defense cost

Copyleaks rejects all sixteen 1x inputs:

- all 8 suspicious crops are rejected as `image_blurry`;
- the 8 real controls are rejected for blur, insufficient colors, or low
  dynamic range.

At 2x it still accepts only 2/8 crops in each arm. Acceptance improves with
larger fields of view, but the recovered suspicious detection rate then falls.
This creates a practical trade-off: the crop must be enlarged enough for the
API to accept it, but that same enlargement dilutes the local signal.

## Answer to the defense objection

"Locate the suspicious area, crop it, and call an AI detector" is a useful
additional defense, but this pilot does not support it as a complete solution:

1. Even with oracle localization, tight-crop recall ranges from 0% to 80%
   across the services with valid outputs.
2. For Sightengine and AI or Not, a small increase in field of view removes
   most or all recovered detections.
3. Some services assign comparable risk to unchanged crops.
4. One service rejects most small crops instead of returning a verdict.

The benchmark should therefore report three quantities for any crop defense:
oracle suspicious-crop recovery, matched real-crop false alarms, and API
applicability/abstention. Reporting only suspicious-crop recall would
overstate the defense.

## Scope and next validation

This is an exploratory `N=8` cross-provider pilot selected to fit the
remaining commercial quotas. Wilson intervals are stored in the machine
summary and are wide at this sample size. The result validates the mechanism
and justifies a larger analysis, but it is not yet a paper-level population
estimate.

A final version should use at least 50 tasks, add 1.5x and 3x around the
Sightengine/AI-or-Not transition, and include both a same-location source crop
and multiple other-location real crops. A no-resize or padded condition would
separate interpolation artifacts from the effect of changing field of view.

## Artifacts

- Probe manifest:
  `results/analysis/mouse_crop_scale_probe_v1/manifest.jsonl`
- Visual audit sheet:
  `results/analysis/mouse_crop_scale_probe_v1/contact_sheet.jpg`
- Joined normalized results:
  `results/analysis/mouse_crop_scale_probe_v1/commercial_joined.jsonl`
- Machine-readable summary:
  `results/analysis/mouse_crop_scale_probe_v1/commercial_summary.json`
- Factor table:
  `results/analysis/mouse_crop_scale_probe_v1/commercial_by_factor.csv`
- Raw provider responses:
  `results/commercial/crop_scale_probe_v1/`

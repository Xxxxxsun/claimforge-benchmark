# SAM 3 hosted API options for CLAIMFORGE (2026-07-21)

## Conclusion

SAM 3 can be used through production-style hosted APIs without deploying the model locally. For CLAIMFORGE, the most direct option is fal's latest SAM 3.1 RLE endpoint:

- Endpoint: `fal-ai/sam-3-1/image-rle`
- Price shown by fal on 2026-07-21: USD 0.01 per request.
- Prompts: text, foreground/background points, and boxes.
- Output: one or more RLE masks, per-mask scores, and boxes.
- Estimated cost: USD 0.10 for a 10-image pilot, USD 2.72 for the current 272-image cat set, or USD 5.94 for 594 images.

The older `fal-ai/sam-3/image-rle` endpoint costs USD 0.005 per request, but the SAM 3.1 endpoint is preferable for a new reproducible pipeline unless the pilot shows no practical quality gain.

Meta itself publishes the SAM 3/3.1 code, gated checkpoints, notebooks, and an interactive Playground. The official release surfaces do not currently advertise a server-to-server hosted inference API, so a batch pipeline needs either a third-party provider or self-hosting.

## Hosted choices

| Option | What is available | Cost/access | Assessment |
|---|---|---|---|
| fal SAM 3.1 | `image` and `image-rle`; text/point/box prompts; masks, scores, and boxes | USD 0.01/request; API key | **Recommended.** Small adapter, explicit RLE output, latest named release. |
| fal SAM 3 | Same image and RLE interfaces | USD 0.005/request; API key | Good low-cost fallback and useful as an ablation against 3.1. |
| Roboflow Serverless | PCS text/exemplar endpoint and PVS point/box endpoint; polygon/RLE/JSON output | Roboflow key and credit billing | Mature fallback, but the public model ID is `sam3/sam3_final` rather than an explicit SAM 3.1 version. |
| Replicate community models | Several public SAM 3 deployments with text/box/point prompts | Usage-priced; model owner/version varies | Usable for exploration, but weaker for a benchmark because these are community-owned deployments and must be version-pinned carefully. |
| Self-hosted Meta code | Full control over weights, batching, and artifacts | Python 3.12+, PyTorch 2.7+, CUDA 12.6+ GPU; gated checkpoint access | Best control, highest setup and maintenance cost. Not needed for the first pilot. |

Primary endpoints:

- fal SAM 3.1 RLE: https://fal.ai/models/fal-ai/sam-3-1/image-rle/api
- fal SAM 3 RLE: https://fal.ai/models/fal-ai/sam-3/image-rle/api
- Roboflow SAM 3: https://docs.roboflow.com/deploy/supported-models/sam3
- Meta repository: https://github.com/facebookresearch/sam3
- Meta release: https://ai.meta.com/blog/segment-anything-model-3/

## How it should be used in the splice pipeline

SAM 3 should segment the generated context crop, not the final full-resolution image. The target is much larger in the crop, and the existing manifest already gives the expected edit region in crop coordinates.

For each image:

1. Upload the generated context crop.
2. Send the semantic prompt (`cat`, `mouse`, `rat`, `cockroach`, or `stain`) together with the known edit-region box or a positive center point.
3. Request up to three masks with scores and boxes.
4. Rank candidates by semantic score and overlap with the expected edit region; reject masks with implausible area or geometry.
5. Use the SAM mask as a semantic seed, not blindly as the final alpha mask.

The recommended final mask is hybrid:

```text
SAM semantic object mask
  + generated-vs-source crop differences connected to a small dilation of that mask
  - disconnected background/color-shift components
  = object plus local contact-shadow mask
```

This keeps useful contact shadows that a pure object mask may omit while avoiding the current global color-threshold behavior. Alpha compositing and any feathering must remain inside the context region so pixels outside that region stay exactly equal to the source image.

Suggested request fields:

```json
{
  "image_url": "<uploaded generated crop>",
  "prompt": "cat",
  "box_prompts": [
    {"x_min": 147, "y_min": 123, "x_max": 321, "y_max": 285, "object_id": 1}
  ],
  "apply_mask": false,
  "return_multiple_masks": true,
  "max_masks": 3,
  "include_scores": true,
  "include_boxes": true
}
```

The coordinates above are an illustrative pixel-coordinate box. The runner must use each manifest's own crop-relative box and verify the endpoint convention in the pilot.

## Pilot protocol

Run 10-20 representative crops before processing the full set:

- Both restaurant and lodging domains.
- Small, medium, and large target instances.
- Difficult fur/background color combinations.
- Cases where the current threshold-30 and threshold-40 masks visibly disagree.

Save the raw RLE, provider score/box, input SHA-256, prompt, transformed box, endpoint ID, request ID, and timestamp. Compare three outputs side by side: current threshold mask, pure SAM mask, and hybrid SAM-plus-difference mask. The selection criterion should be edge quality and preservation of target shadows without unrelated background changes, not merely mask size.

## Reproducibility and data handling

fal accepts a URL, data URI, or SDK-uploaded local image. By default, request input/output payloads are stored for 30 days. For unpublished benchmark inputs, set `X-Fal-Store-IO: 0`, configure a short `X-Fal-Object-Lifecycle-Preference`, download results immediately, and avoid committing any API key. fal notes that returned CDN media URLs are public until expiration, so the runner should store masks locally and omit temporary URLs from committed artifacts.

Data-retention references:

- https://fal.ai/docs/documentation/model-apis/inference/payloads
- https://fal.ai/docs/documentation/model-apis/common-parameters

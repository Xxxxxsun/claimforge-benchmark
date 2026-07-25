# Selected final splice images

These directories materialize the image selected for each accepted task in the
manual splice-method review. Entries marked `reject_both` are excluded.

Each final directory contains:

- one regular PNG file per accepted task;
- `manifest.jsonl`, mapping every output image to its selected method and
  original candidate path;
- `summary.json`, recording counts and the source selection file.

## Current complete sets

- `claimforge_cat_selected_251_20260725`: 232 accepted base-round images plus
  19 accepted relabel images.
- `claimforge_trash_can_selected_250_20260725`: 199 accepted base-round images
  plus 51 accepted relabel images.

Their normalized selection manifests are:

- `annotations/claimforge_cat_final_251_selections.json`
- `annotations/claimforge_trash_can_final_250_selections.json`

The older `selected_232` and `selected_199` directories are base-round
snapshots retained for provenance.

Rebuild a directory with:

```bash
python3 scripts/materialize_selected_spliced_images.py \
  --selection-json <selection.json> \
  --output-dir <final-directory>
```

The combined manifests contain accepted entries only, so rebuild them with
`--reject-selection __none__`.

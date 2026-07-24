# Selected final splice images

These directories materialize the image selected for each accepted task in the
manual splice-method review. Entries marked `reject_both` are excluded.

Each final directory contains:

- one regular PNG file per accepted task;
- `manifest.jsonl`, mapping every output image to its selected method and
  original candidate path;
- `summary.json`, recording counts and the source selection file.

Rebuild a directory with:

```bash
python3 scripts/materialize_selected_spliced_images.py \
  --selection-json <selection.json> \
  --output-dir <final-directory>
```

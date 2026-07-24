# Reject-Both Relabel Handoff

This batch contains only tasks that were manually marked `reject_both` in the
cat or trash-can splice-method comparison. The relabel pages load the original
source photographs. Generated crops, spliced images, and previous boxes are not
preloaded.

## Batch counts

- Cat: 40 images (`21 lodging`, `19 restaurant`)
- Trash can: 61 images (`36 lodging`, `25 restaurant`)
- Overlap: 9 source images occur in both batches; they remain separate because
  the requested inserted object is different.

The machine-readable batch summary is:

`annotations/reject_both_relabel_batches_20260724.json`

## Relabel pages

Start the repository label server:

```bash
python3 scripts/claimforge_label_server.py --host 127.0.0.1 --port 8000
```

Open:

- Cat: `http://localhost:8000/tools/cat-reject-both-relabel.html`
- Trash can: `http://localhost:8000/tools/trash-can-reject-both-relabel.html`

Draw a new orange insertion box and a new blue context box. The default
candidate is already fixed to `cat` or `trash can`. Progress and slots are
saved automatically to:

- `annotations/claimforge_cat_reject_both_40_relabel_slots.json`
- `annotations/claimforge_trash_can_reject_both_61_relabel_slots.json`

Do not hand the batch to generation until the page reports `Done 40/40` or
`Done 61/61`.

## Export replacement tasks

Cat:

```bash
python3 scripts/export_cat_generation_tasks.py \
  --slots-json annotations/claimforge_cat_reject_both_40_relabel_slots.json \
  --tasks annotations/cat_generation_tasks_relabel_reject_both_40.jsonl \
  --crop-dir crops/context_cat_relabel_reject_both_40 \
  --task-prefix cat \
  --default-candidate cat
```

Trash can:

```bash
python3 scripts/export_cat_generation_tasks.py \
  --slots-json annotations/claimforge_trash_can_reject_both_61_relabel_slots.json \
  --tasks annotations/trash_can_generation_tasks_relabel_reject_both_61.jsonl \
  --crop-dir crops/context_trash_can_relabel_reject_both_61 \
  --task-prefix trash_can \
  --default-candidate "trash can"
```

The replacement tasks intentionally retain their original task IDs. Generation
must use a new output directory so the previous crops and review evidence are
not overwritten.

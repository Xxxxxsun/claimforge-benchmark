# A3D unified aggregate

This directory is the machine-readable index for the complete A3D report.

- `aggregate_metrics.json`: mouse, cat, trash-can, combined, JPEG-90,
  localization, proposal, Q1 diagnostic, generated-full, coverage, and TPR@1%
  metrics.
- `artifact_checksums.sha256`: SHA-256 checksums of every source result consumed
  by the aggregate.

Regenerate both files with:

```bash
/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.aggregate_a3d_results
```

The narrative report is
`docs/A3D_ADAPTIVE_DEFENSE_FULL_REPORT_2026-07-27.md`.

#!/usr/bin/env python3
"""Run and audit official DINOv3-IML checkpoint 48 on Balanced250."""

from eval.opensource.localizer_balanced import main


if __name__ == "__main__":
    raise SystemExit(main("dinov3_iml"))

#!/usr/bin/env python3
"""Run and audit official Mesorch checkpoint 98 on Balanced250."""

from eval.opensource.localizer_balanced import main


if __name__ == "__main__":
    raise SystemExit(main("mesorch"))

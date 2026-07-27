#!/usr/bin/env python3
"""Run and audit official RelayFormer checkpoint 164 on Balanced250."""

from eval.opensource.localizer_balanced import main


if __name__ == "__main__":
    raise SystemExit(main("relayformer"))

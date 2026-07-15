"""Small dependency-free binary PNG writer for bbox-to-mask disclosure."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path


def boxes_to_pixels(boxes: list[dict], width: int, height: int) -> list[list[int]]:
    output = []
    for region in boxes:
        x1, y1, x2, y2 = region["bbox_1000"]
        output.append([max(0, min(width, round(x1 * width / 1000))), max(0, min(height, round(y1 * height / 1000))), max(0, min(width, round(x2 * width / 1000))), max(0, min(height, round(y2 * height / 1000)))])
    return [box for box in output if box[0] < box[2] and box[1] < box[3]]


def write_union_mask(path: Path, width: int, height: int, boxes: list[list[int]]) -> None:
    pixels = bytearray(width * height)
    for x1, y1, x2, y2 in boxes:
        for y in range(y1, y2):
            pixels[y * width + x1:y * width + x2] = b"\xff" * (x2 - x1)
    raw = b"".join(b"\x00" + pixels[row * width:(row + 1) * width] for row in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)

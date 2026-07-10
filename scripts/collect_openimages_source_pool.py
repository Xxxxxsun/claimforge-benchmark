#!/usr/bin/env python3
"""Collect a restaurant/lodging source-image pool from Open Images V7.

The output is intended for human screening and later CLAIMFORGE slot annotation.
It downloads licensed Open Images validation/test photos, normalizes them to
JPEG with a bounded max side, and writes manifests plus contact sheets.
"""
from __future__ import annotations

import csv
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "source_pool" / "openimages_v7_600"
CACHE = OUT / "_cache"
TARGET_PER_CATEGORY = 300
CONTACT_SHEET_PAGE_SIZE = 50

HEADERS = {
    "User-Agent": "ClaimForgeBenchmark/0.1 source pool collection",
}

URLS = {
    "class_desc": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv",
    "val_labels": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-machine-imagelabels.csv",
    "test_labels": "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-machine-imagelabels.csv",
    "val_meta": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test_meta": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}

RESTAURANT_LABELS = {
    "/m/06l8d": "restaurant",
    "/m/03bwyyr": "fast_food_restaurant",
    "/m/0gx1rv4": "chinese_restaurant",
    "/m/03sx2d": "food_court",
    "/m/03d2wd": "dining_room",
    "/m/0h8n5zk": "kitchen_dining_table",
    "/m/04bcr3": "table",
    "/m/04brg2": "tableware",
    "/m/02wbm": "food",
}

LODGING_LABELS = {
    "/m/03pty": "hotel",
    "/m/05j0rg": "boutique_hotel",
    "/m/0594v": "motel",
    "/m/02_58j": "bedroom",
    "/m/01j2bj": "bathroom",
    "/m/041x_j": "restroom",
    "/m/03f6tq": "living_room",
    "/m/0d4wf": "kitchen",
    "/m/03ssj5": "bed",
    "/m/03mcr3": "bedding",
    "/m/04mygm": "bed_sheet",
}

PEOPLE_LABELS = {
    "/m/01g317": "person",
    "/m/04yx4": "man",
    "/m/03bt1vf": "woman",
    "/m/0dzct": "human_face",
    "/m/01bl7v": "boy",
    "/m/05r655": "girl",
}

BAD_TITLE_RE = re.compile(
    r"\b(map|diagram|chart|graph|logo|poster|flyer|screenshot|scan|book|menu|"
    r"badge|icon|drawing|illustration|certificate|document)\b",
    re.IGNORECASE,
)


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cached(name: str, url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"download metadata {name}", flush=True)
        path.write_bytes(fetch(url, timeout=180))
    return path


def read_class_names() -> dict[str, str]:
    path = cached("oidv7-class-descriptions.csv", URLS["class_desc"])
    with path.open(newline="", encoding="utf-8") as f:
        return {row[0]: row[1] for row in csv.reader(f)}


def collect_labels(split: str) -> dict[str, dict]:
    label_url = URLS[f"{split}_labels"]
    label_path = cached(f"oidv7-{split}-annotations-machine-imagelabels.csv", label_url)
    all_targets = set(RESTAURANT_LABELS) | set(LODGING_LABELS) | set(PEOPLE_LABELS)
    data: dict[str, dict] = defaultdict(lambda: {"restaurant": {}, "lodging": {}, "people": {}})
    with label_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["LabelName"]
            if label not in all_targets:
                continue
            conf = float(row["Confidence"])
            image_id = row["ImageID"]
            if label in RESTAURANT_LABELS and conf >= 0.35:
                data[image_id]["restaurant"][RESTAURANT_LABELS[label]] = conf
            elif label in LODGING_LABELS and conf >= 0.35:
                data[image_id]["lodging"][LODGING_LABELS[label]] = conf
            elif label in PEOPLE_LABELS:
                data[image_id]["people"][PEOPLE_LABELS[label]] = conf
    return data


def iter_metadata(split: str):
    meta_path = cached(f"{split}-images-with-rotation.csv", URLS[f"{split}_meta"])
    with meta_path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def category_scores(labels: dict) -> tuple[float, float]:
    r = labels["restaurant"]
    l = labels["lodging"]
    r_scene = max([r.get(k, 0.0) for k in ("restaurant", "fast_food_restaurant", "chinese_restaurant", "food_court")])
    r_table = max([r.get(k, 0.0) for k in ("dining_room", "kitchen_dining_table", "table", "tableware")])
    r_food = r.get("food", 0.0)
    restaurant_score = max(
        3.0 * r_scene,
        1.7 * min(r_table, max(r_food, r_scene)),
        1.2 * r.get("dining_room", 0.0),
    )

    l_scene = max([l.get(k, 0.0) for k in ("hotel", "boutique_hotel", "motel")])
    l_room = max([l.get(k, 0.0) for k in ("bedroom", "bathroom", "restroom", "living_room", "kitchen")])
    l_bed = max([l.get(k, 0.0) for k in ("bed", "bedding", "bed_sheet")])
    lodging_score = max(3.0 * l_scene, 2.0 * l_room, 1.4 * l_bed)
    return restaurant_score, lodging_score


def build_candidates(splits=("val", "test")) -> dict[str, list[dict]]:
    candidates = {"restaurant": [], "lodging": []}
    seen = set()
    for split in splits:
        print(f"parse labels {split}", flush=True)
        labels_by_image = collect_labels(split)
        print(f"parse metadata {split}", flush=True)
        for meta in iter_metadata(split):
            image_id = meta["ImageID"]
            if image_id not in labels_by_image or image_id in seen:
                continue
            labels = labels_by_image[image_id]
            people_score = max(labels["people"].values(), default=0.0)
            if people_score >= 0.80:
                continue
            title = meta.get("Title") or ""
            if BAD_TITLE_RE.search(title):
                continue
            if not (meta.get("OriginalURL") or meta.get("Thumbnail300KURL")):
                continue
            if not meta.get("License"):
                continue

            r_score, l_score = category_scores(labels)
            if r_score < 0.65 and l_score < 0.65:
                continue
            if r_score >= l_score and r_score >= 0.65:
                category = "restaurant"
                score = r_score
            elif l_score >= 0.65:
                category = "lodging"
                score = l_score
            else:
                continue

            seen.add(image_id)
            candidates[category].append({
                "source_image_id": image_id,
                "split": split,
                "score": round(score, 4),
                "people_score": round(people_score, 4),
                "restaurant_labels": labels["restaurant"],
                "lodging_labels": labels["lodging"],
                "original_url": meta.get("OriginalURL"),
                "thumbnail_url": meta.get("Thumbnail300KURL"),
                "landing_url": meta.get("OriginalLandingURL"),
                "license": meta.get("License"),
                "author": meta.get("Author"),
                "author_profile_url": meta.get("AuthorProfileURL"),
                "title": title,
                "original_size": meta.get("OriginalSize"),
                "original_md5": meta.get("OriginalMD5"),
            })

    rng = random.Random(20260629)
    for category in candidates:
        rng.shuffle(candidates[category])
        candidates[category].sort(key=lambda x: x["score"], reverse=True)
        print(category, "candidates", len(candidates[category]), flush=True)
    return candidates


def download_image(item: dict, max_side: int) -> Image.Image | None:
    for key in ("original_url", "thumbnail_url"):
        url = item.get(key)
        if not url:
            continue
        try:
            raw = fetch(url, timeout=30)
            img = Image.open(BytesIO(raw))
            img = ImageOps.exif_transpose(img).convert("RGB")
            if min(img.size) < 360:
                continue
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            item["downloaded_from"] = key
            item["downloaded_url"] = url
            return img
        except Exception as exc:
            item.setdefault("download_errors", []).append(f"{key}: {type(exc).__name__}: {exc}")
    return None


def render_contact_sheet(items: list[dict], out_path: Path):
    thumb_w, thumb_h = 220, 165
    cols = 5
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 40)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, item in enumerate(items):
        img = Image.open(OUT / item["path"]).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w + (thumb_w - img.width) // 2
        y = (idx // cols) * (thumb_h + 40) + (thumb_h - img.height) // 2
        sheet.paste(img, (x, y))
        tx = (idx % cols) * thumb_w + 4
        ty = (idx // cols) * (thumb_h + 40) + thumb_h + 2
        draw.text((tx, ty), f"{idx:03d} {item['id']} score={item['score']}", fill=(0, 0, 0))
        draw.text((tx, ty + 15), item.get("title", "")[:32], fill=(70, 70, 70))
        draw.text((tx, ty + 29), item.get("license", "").replace("https://creativecommons.org/licenses/", "cc/")[:34], fill=(100, 100, 100))
    sheet.save(out_path, "JPEG", quality=92)


def write_contact_sheet(items: list[dict], category: str):
    sheet_dir = OUT / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(items, sheet_dir / f"{category}_contact_sheet.jpg")

    page_dir = sheet_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    for stale in page_dir.glob(f"{category}_*.jpg"):
        stale.unlink()
    for page_idx, start in enumerate(range(0, len(items), CONTACT_SHEET_PAGE_SIZE), start=1):
        page_items = items[start:start + CONTACT_SHEET_PAGE_SIZE]
        render_contact_sheet(page_items, page_dir / f"{category}_{page_idx:02d}.jpg")


def read_existing_items(category: str) -> list[dict]:
    manifest_path = OUT / f"{category}_manifest.json"
    if not manifest_path.exists():
        return []
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    kept = []
    for item in items:
        image_path = OUT / item.get("path", "")
        if image_path.exists():
            kept.append(item)
    return kept


def collect_category(category: str, candidates: list[dict], target: int, max_side: int) -> list[dict]:
    out_dir = OUT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    collected = read_existing_items(category)
    used_source_ids = {item["source_image_id"] for item in collected}
    used_ids = {item["id"] for item in collected}
    failures = []
    if collected:
        print(f"{category}: found {len(collected)} existing images", flush=True)

    for item in candidates:
        if len(collected) >= target:
            break
        if item["source_image_id"] in used_source_ids:
            continue
        img = download_image(item, max_side=max_side)
        if img is None:
            failures.append(item)
            continue
        idx = len(collected)
        while f"{category}_{idx:03d}" in used_ids:
            idx += 1
        image_id = f"{category}_{idx:03d}"
        filename = f"{image_id}.jpg"
        img.save(out_dir / filename, "JPEG", quality=92, optimize=True)
        saved = {
            "id": image_id,
            "category": category,
            "path": f"{category}/{filename}",
            "size": {"width": img.width, "height": img.height},
            **item,
        }
        collected.append(saved)
        used_source_ids.add(item["source_image_id"])
        used_ids.add(image_id)
        if len(collected) % 25 == 0:
            print(f"{category}: collected {len(collected)}/{target}", flush=True)
        time.sleep(0.05)

    if len(collected) < target:
        raise RuntimeError(f"{category}: only collected {len(collected)} / {target}")
    (OUT / f"{category}_manifest.json").write_text(json.dumps(collected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / f"{category}_failures.json").write_text(json.dumps(failures[:500], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_contact_sheet(collected, category)
    return collected


def write_summary(items_by_category: dict[str, list[dict]]):
    lines = [
        "# Open Images V7 Source Pool Summary",
        f"Collected/updated on {date.today().isoformat()} for CLAIMFORGE benchmark expansion.",
        "This pool is for human screening and annotation, not the final benchmark split.",
        "",
    ]
    for category, items in items_by_category.items():
        min_sides = [min(item["size"]["width"], item["size"]["height"]) for item in items]
        max_sides = [max(item["size"]["width"], item["size"]["height"]) for item in items]
        people_scores = [float(item.get("people_score", 0.0)) for item in items]
        label_key = f"{category}_labels"
        labels = Counter()
        for item in items:
            labels.update((item.get(label_key) or {}).keys())

        lines.extend([
            f"## {category}",
            f"- images: {len(items)}",
            f"- total size on disk: see `{category}/`",
            f"- min side median: {round(sorted(min_sides)[len(min_sides) // 2]) if min_sides else 0} px",
            f"- max side median: {round(sorted(max_sides)[len(max_sides) // 2]) if max_sides else 0} px",
            f"- people score max: {max(people_scores, default=0.0):.3f} (filtered out >= 0.80)",
            "- top labels:",
        ])
        for label, count in labels.most_common(12):
            lines.append(f"  - {label}: {count}")
        lines.append("")

    lines.extend([
        "## Recommended next step",
        "- Review `contact_sheets/pages/*.jpg`.",
        "- Mark keep/reject for each image before slot annotation.",
        "- Prefer indoor/table/room/bathroom/kitchen surfaces where a small localized defect/object can plausibly be inserted.",
        "- Reject images dominated by signs, menus, posters, exterior-only views, heavy crowds/faces, or scenes with no plausible local edit surface.",
        "",
    ])
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names()
    label_doc = {
        "restaurant_labels": {mid: class_names.get(mid, name) for mid, name in RESTAURANT_LABELS.items()},
        "lodging_labels": {mid: class_names.get(mid, name) for mid, name in LODGING_LABELS.items()},
        "people_labels_filtered": {mid: class_names.get(mid, name) for mid, name in PEOPLE_LABELS.items()},
    }
    (OUT / "label_sets.json").write_text(json.dumps(label_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    candidates = build_candidates()
    (OUT / "candidate_counts.json").write_text(json.dumps({k: len(v) for k, v in candidates.items()}, indent=2) + "\n", encoding="utf-8")

    restaurant = collect_category("restaurant", candidates["restaurant"], target=TARGET_PER_CATEGORY, max_side=1800)
    lodging = collect_category("lodging", candidates["lodging"], target=TARGET_PER_CATEGORY, max_side=1800)
    all_items = restaurant + lodging
    (OUT / "manifest.json").write_text(json.dumps(all_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary({"restaurant": restaurant, "lodging": lodging})

    readme = f"""# Open Images V7 600-image Source Pool

This is a candidate source pool for the CLAIMFORGE benchmark expansion.

Counts:
- restaurant: {len(restaurant)} images
- lodging: {len(lodging)} images

These are not final benchmark samples yet. They are intended for human screening
and slot annotation. Source URLs, license URLs, Open Images split, labels, and
download provenance are in the manifests.

Contact sheets:
- contact_sheets/restaurant_contact_sheet.jpg
- contact_sheets/lodging_contact_sheet.jpg
- contact_sheets/pages/restaurant_*.jpg
- contact_sheets/pages/lodging_*.jpg

Primary manifests:
- manifest.json
- restaurant_manifest.json
- lodging_manifest.json
- SUMMARY.md
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("done", OUT, len(all_items), flush=True)


if __name__ == "__main__":
    main()

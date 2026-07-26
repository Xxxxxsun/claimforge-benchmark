import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from eval.opensource import build_balanced250_canonical as builder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Balanced250DeterministicSelectionTest(unittest.TestCase):
    def test_exact_eligible_counts_are_part_of_the_frozen_contract(self):
        self.assertEqual(
            builder.EXPECTED_ELIGIBLE_ROWS,
            {
                "real": 275,
                "local_mouse": 275,
                "local_cat": 251,
                "local_trash_can": 250,
                "fullframe_mouse": 275,
                "fullframe_cat": 272,
                "fullframe_trash_can": 260,
            },
        )

    def test_eligibility_fingerprint_is_order_independent_and_frozen(self):
        rows = [
            {
                "normalized_task_id": "lodging_002_slot_001",
                "raw_path": "b.jpg",
                "raw_sha256": "b" * 64,
            },
            {
                "normalized_task_id": "lodging_001_slot_001",
                "raw_path": "a.jpg",
                "raw_sha256": "a" * 64,
            },
        ]

        expected = (
            "f917fc76269e2dc6716fb10328577ed5b"
            "02fd4530a9bb60c04c124464a4f0d28"
        )
        self.assertEqual(
            builder._eligibility_set_hash(rows, "real"),
            expected,
        )
        self.assertEqual(
            builder._eligibility_set_hash(reversed(rows), "real"),
            expected,
        )
        changed = copy.deepcopy(rows)
        changed[0]["raw_path"] = "changed.jpg"
        self.assertNotEqual(
            builder._eligibility_set_hash(changed, "real"),
            expected,
        )

    def test_eligibility_fingerprint_rejects_invalid_identities(self):
        with self.assertRaisesRegex(ValueError, "invalid identities"):
            builder._eligibility_set_hash(
                [
                    {"normalized_task_id": "lodging_001_slot_001"},
                    {"normalized_task_id": "lodging_001_slot_001"},
                ],
                "real",
            )
        with self.assertRaisesRegex(ValueError, "invalid identities"):
            builder._eligibility_set_hash(
                [{"raw_path": "missing-id.jpg"}],
                "real",
            )

    def test_hash_selection_is_frozen_and_input_order_independent(self):
        rows = [
            {"task_id": "lodging_001_slot_001"},
            {"task_id": "lodging_002_slot_001"},
            {"task_id": "restaurant_003_slot_001"},
            {"task_id": "restaurant_004_slot_001"},
        ]

        ranked_a, selected_a = builder._ranked_selection(
            copy.deepcopy(rows),
            "local_cat",
            count=2,
        )
        ranked_b, selected_b = builder._ranked_selection(
            list(reversed(copy.deepcopy(rows))),
            "local_cat",
            count=2,
        )

        expected_ranked = [
            "restaurant_003_slot_001",
            "lodging_001_slot_001",
            "restaurant_004_slot_001",
            "lodging_002_slot_001",
        ]
        expected_selected = expected_ranked[:2]
        self.assertEqual(
            [row["task_id"] for row in ranked_a],
            expected_ranked,
        )
        self.assertEqual(
            [row["task_id"] for row in ranked_b],
            expected_ranked,
        )
        self.assertEqual(
            [row["task_id"] for row in selected_a],
            expected_selected,
        )
        self.assertEqual(
            [row["task_id"] for row in selected_b],
            expected_selected,
        )
        self.assertEqual(
            selected_a[0]["_selection_key"],
            "1f0327a5ca678a20a5c0209b494fdb791ecc99a9ecb8626ecc87b0027a1201b3",
        )
        self.assertEqual(
            [row["_eligibility_rank"] for row in ranked_a],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [row["_selection_rank"] for row in selected_a],
            [0, 1],
        )

    def test_real_selection_uses_first_hash_ranked_content_representative(self):
        duplicate_digest = "a" * 64
        rows = [
            {
                "task_id": "lodging_001_slot_001",
                "raw_sha256": duplicate_digest,
            },
            {
                "task_id": "lodging_002_slot_001",
                "raw_sha256": "b" * 64,
            },
            {
                "task_id": "restaurant_003_slot_001",
                "raw_sha256": "c" * 64,
            },
            {
                "task_id": "restaurant_004_slot_001",
                "raw_sha256": duplicate_digest,
            },
        ]

        ranked, selected = builder._ranked_selection(
            rows,
            "real",
            count=3,
            deduplicate_raw_sha=True,
        )

        self.assertEqual(
            [row["task_id"] for row in ranked],
            [
                "restaurant_004_slot_001",
                "lodging_002_slot_001",
                "restaurant_003_slot_001",
                "lodging_001_slot_001",
            ],
        )
        self.assertEqual(
            [row["task_id"] for row in selected],
            [
                "restaurant_004_slot_001",
                "lodging_002_slot_001",
                "restaurant_003_slot_001",
            ],
        )
        self.assertNotIn("lodging_001_slot_001", {
            row["task_id"] for row in selected
        })

    def test_rejects_duplicate_normalized_task_ids(self):
        rows = [
            {"task_id": "cat_lodging_001_slot_001"},
            {"task_id": "lodging_001_slot_001"},
        ]
        with self.assertRaisesRegex(
            ValueError,
            "duplicate normalized task IDs",
        ):
            builder._ranked_selection(rows, "local_cat", count=1)

    def test_rejects_selection_key_collisions(self):
        rows = [
            {"task_id": "cat_lodging_001_slot_001"},
            {"task_id": "cat_lodging_002_slot_001"},
        ]
        with (
            mock.patch.object(builder, "_selection_key", return_value="collision"),
            self.assertRaisesRegex(ValueError, "selection-key collision"),
        ):
            builder._ranked_selection(rows, "local_cat", count=1)

    def test_rejects_duplicate_manifest_ids(self):
        rows = [
            {"task_id": "cat_lodging_001_slot_001"},
            {"task_id": "cat_lodging_001_slot_001"},
        ]
        with self.assertRaisesRegex(
            ValueError,
            "duplicate task_id=cat_lodging_001_slot_001",
        ):
            builder._unique_by(rows, "task_id", "Cat materialized")


class Balanced250LatestRowTest(unittest.TestCase):
    def _whole_fixture(
        self,
        root: Path,
        run_rows: list[dict],
    ) -> tuple[Path, Path]:
        task_path = root / "tasks.jsonl"
        run_path = root / "run.jsonl"
        _write_jsonl(
            task_path,
            [{"task_id": "cat_restaurant_001_slot_001"}],
        )
        _write_jsonl(run_path, run_rows)
        return task_path, run_path

    def test_latest_physical_row_is_authoritative_and_indexed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_id = "cat_restaurant_001_slot_001"
            task_path, run_path = self._whole_fixture(
                root,
                [
                    {"task_id": task_id, "status": "failed", "error": "old"},
                    {"task_id": task_id, "status": "ok", "output_image": "x.png"},
                ],
            )
            with (
                mock.patch.dict(
                    builder.WHOLE_TASKS,
                    {"fullframe_cat": task_path},
                ),
                mock.patch.dict(
                    builder.WHOLE_RUNS,
                    {"fullframe_cat": run_path},
                ),
            ):
                _, latest, indexes, physical = builder._whole_sources(
                    root,
                    "fullframe_cat",
                )

            self.assertEqual(len(physical), 2)
            self.assertEqual(latest[task_id]["status"], "ok")
            self.assertEqual(indexes[task_id], 1)

    def test_latest_failure_is_not_hidden_by_an_earlier_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_id = "cat_restaurant_001_slot_001"
            task_path, run_path = self._whole_fixture(
                root,
                [
                    {"task_id": task_id, "status": "ok", "output_image": "x.png"},
                    {"task_id": task_id, "status": "failed", "error": "new"},
                ],
            )
            with (
                mock.patch.dict(
                    builder.WHOLE_TASKS,
                    {"fullframe_cat": task_path},
                ),
                mock.patch.dict(
                    builder.WHOLE_RUNS,
                    {"fullframe_cat": run_path},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "latest row is not ok",
                ),
            ):
                builder._whole_sources(root, "fullframe_cat")


class Balanced250DiffBoxAndPathSafetyTest(unittest.TestCase):
    def test_canonicalize_strips_comment_exif_and_icc_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "metadata-source.jpg"
            destination = root / "canonical.jpg"
            source = Image.new("RGB", (6, 4), (30, 60, 90))
            exif = Image.Exif()
            exif[0x010E] = "private description"
            source.save(
                source_path,
                format="JPEG",
                quality=95,
                comment=b"private comment",
                exif=exif,
                icc_profile=b"private fake ICC payload",
            )
            with Image.open(source_path) as opened:
                self.assertIn("comment", opened.info)
                self.assertIn("icc_profile", opened.info)
                self.assertTrue(opened.getexif())
                contaminated = opened.convert("RGB")

            builder._canonicalize(contaminated, destination)

            with Image.open(destination) as canonical:
                self.assertEqual(canonical.format, "JPEG")
                self.assertEqual(canonical.mode, "RGB")
                self.assertEqual(canonical.size, (6, 4))
                self.assertFalse(canonical.getexif())
                self.assertNotIn("comment", canonical.info)
                self.assertNotIn("icc_profile", canonical.info)

    def test_exact_diff_and_outside_context_pixels_are_preserved(self):
        source = Image.new("RGB", (5, 4), (10, 20, 30))
        forged = source.copy()
        forged.putpixel((1, 1), (11, 20, 30))
        forged.putpixel((4, 3), (10, 20, 31))

        mask = builder._exact_diff_mask(source, forged)

        self.assertEqual(mask.mode, "L")
        self.assertEqual(builder._mask_pixels(mask), 2)
        self.assertEqual(mask.getpixel((1, 1)), 255)
        self.assertEqual(mask.getpixel((4, 3)), 255)
        self.assertEqual(
            builder._outside_box_pixels(mask, [0, 0, 2, 2]),
            1,
        )
        self.assertEqual(
            builder._validated_box([0, 0, 2, 2], mask.size, "context"),
            [0, 0, 2, 2],
        )

    def test_diff_and_boxes_fail_closed_on_invalid_geometry(self):
        source = Image.new("RGB", (5, 4), "black")
        with self.assertRaisesRegex(ValueError, "local pair size mismatch"):
            builder._exact_diff_mask(
                source,
                Image.new("RGB", (4, 4), "black"),
            )
        with self.assertRaisesRegex(ValueError, "four-coordinate"):
            builder._validated_box([0, 0, 2], source.size, "edit")
        with self.assertRaisesRegex(ValueError, "outside"):
            builder._validated_box([0, 0, 6, 2], source.size, "edit")
        with self.assertRaisesRegex(ValueError, "outside image bounds"):
            builder._outside_box_pixels(
                Image.new("L", source.size),
                [0, 0, 6, 2],
            )

    def test_declared_size_accepts_both_frozen_encodings(self):
        self.assertEqual(
            builder._validated_declared_size(
                [864, 576],
                (864, 576),
                "Mouse size",
            ),
            (864, 576),
        )
        self.assertEqual(
            builder._validated_declared_size(
                {"width": 864, "height": 576},
                (864, 576),
                "Cat size",
            ),
            (864, 576),
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            builder._validated_declared_size(
                [865, 576],
                (864, 576),
                "bad size",
            )

    def test_repo_path_resolution_rejects_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            repo = container / "repo"
            repo.mkdir()
            inside = repo / "inside.txt"
            inside.write_text("inside", encoding="utf-8")
            outside = container / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (repo / "outside-link").symlink_to(outside)

            self.assertEqual(
                builder._resolve_repo_file(repo, "inside.txt", "fixture"),
                inside.resolve(),
            )
            for unsafe in ("../outside.txt", outside, "outside-link"):
                with self.subTest(unsafe=str(unsafe)):
                    with self.assertRaisesRegex(ValueError, "escapes repository"):
                        builder._resolve_repo_file(repo, unsafe, "fixture")

    def test_release_output_must_stay_below_repository_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            repo = container / "repo"
            repo.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes repository"):
                builder.build_release(
                    repo_root=repo,
                    output_dir=container / "outside",
                )
            with self.assertRaisesRegex(ValueError, "repository root"):
                builder.build_release(
                    repo_root=repo,
                    output_dir=repo,
                )


class Balanced250MouseMaskValidationTest(unittest.TestCase):
    def _save_mask(
        self,
        root: Path,
        *,
        mode: str = "L",
        size: tuple[int, int] = (4, 3),
        pixels: list[int] | None = None,
    ) -> Path:
        path = root / f"mask-{mode}.png"
        mask = Image.new(mode, size, 0)
        if pixels is None:
            if mode == "L":
                mask.putpixel((1, 1), 255)
        else:
            mask.putdata(pixels)
        mask.save(path, format="PNG")
        return path

    def test_accepts_native_l_binary_mask_with_matching_hash_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._save_mask(root)

            mask = builder._validate_binary_mask(
                path,
                expected_size=(4, 3),
                expected_sha256=builder.sha256_file(path),
                label="Mouse GT mask",
            )

            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (4, 3))
            self.assertEqual(builder._mask_pixels(mask), 1)

    def test_rejects_mouse_mask_hash_size_mode_and_nonbinary_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._save_mask(root)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                builder._validate_binary_mask(
                    valid,
                    expected_size=(4, 3),
                    expected_sha256="0" * 64,
                    label="Mouse GT mask",
                )
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                builder._validate_binary_mask(
                    valid,
                    expected_size=(5, 3),
                    expected_sha256=builder.sha256_file(valid),
                    label="Mouse GT mask",
                )

            rgb = self._save_mask(root, mode="RGB")
            with self.assertRaisesRegex(ValueError, "native L mode"):
                builder._validate_binary_mask(
                    rgb,
                    expected_size=(4, 3),
                    expected_sha256=builder.sha256_file(rgb),
                    label="Mouse GT mask",
                )

            nonbinary = self._save_mask(
                root,
                pixels=[0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0],
            )
            with self.assertRaisesRegex(ValueError, "not binary 0/255"):
                builder._validate_binary_mask(
                    nonbinary,
                    expected_size=(4, 3),
                    expected_sha256=builder.sha256_file(nonbinary),
                    label="Mouse GT mask",
                )


class Balanced250StageInventoryTest(unittest.TestCase):
    @staticmethod
    def _expected_names() -> tuple[set[str], set[str]]:
        return (
            {f"image-{index:04d}.jpg" for index in range(1775)},
            {f"mask-{index:04d}.png" for index in range(750)},
        )

    def test_inventory_requires_frozen_cardinalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            images, masks = self._expected_names()
            with self.assertRaisesRegex(
                ValueError,
                "expected 1775 .*canonical image names",
            ):
                builder._assert_stage_inventory(
                    staging,
                    expected_image_names=set(list(images)[:-1]),
                    expected_mask_names=masks,
                    include_manifest=False,
                )
            with self.assertRaisesRegex(
                ValueError,
                "expected 750 .*local mask names",
            ):
                builder._assert_stage_inventory(
                    staging,
                    expected_image_names=images,
                    expected_mask_names=set(list(masks)[:-1]),
                    include_manifest=False,
                )

    def test_inventory_accepts_only_the_exact_regular_file_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            image_dir = staging / "images"
            mask_dir = staging / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            images, masks = self._expected_names()
            for name in images:
                (image_dir / name).touch()
            for name in masks:
                (mask_dir / name).touch()
            for name in ("inputs.jsonl", "panel.jsonl", "source_pairs.jsonl"):
                (staging / name).touch()

            builder._assert_stage_inventory(
                staging,
                expected_image_names=images,
                expected_mask_names=masks,
                include_manifest=False,
            )

            manifest = staging / "manifest.json"
            manifest.touch()
            builder._assert_stage_inventory(
                staging,
                expected_image_names=images,
                expected_mask_names=masks,
                include_manifest=True,
            )
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                builder._assert_stage_inventory(
                    staging,
                    expected_image_names=images,
                    expected_mask_names=masks,
                    include_manifest=False,
                )
            manifest.unlink()

            extra = image_dir / "extra.jpg"
            extra.touch()
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                builder._assert_stage_inventory(
                    staging,
                    expected_image_names=images,
                    expected_mask_names=masks,
                    include_manifest=False,
                )
            extra.unlink()

            missing = image_dir / "image-0000.jpg"
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                builder._assert_stage_inventory(
                    staging,
                    expected_image_names=images,
                    expected_mask_names=masks,
                    include_manifest=False,
                )
            missing.touch()

            symlink = image_dir / "unexpected-link.jpg"
            symlink.symlink_to(staging / "inputs.jsonl")
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                builder._assert_stage_inventory(
                    staging,
                    expected_image_names=images,
                    expected_mask_names=masks,
                    include_manifest=False,
                )


if __name__ == "__main__":
    unittest.main()

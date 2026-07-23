# Trash-can source suitability audit (260 tasks)

Scope: read-only visual review of all 260 `context_crop` images with
`edit_region_in_context_xyxy` overlaid. The criterion is stricter than “the model
can draw a bin”: a complete freestanding bin must have a plausible visible
support plane, an unobstructed silhouette and base, and must not look as if it
was put on a bed, sofa, dining table, sales counter, food display, toilet, or
shower floor.

## Summary

- **91 direct replacement/exclusion recommendations**
  - 62 edit regions are primarily on a table, counter, display case, cabinet
    face, wall, radiator, or other elevated/vertical plane, without enough
    visible floor for a complete floor-standing bin.
  - 18 edit regions are on a bed, pillow, sofa, or other soft seating/sleeping
    surface.
  - 6 edit regions are on a toilet/tank, shower floor, or bathtub.
  - 4 edit regions are occupied by a person.
  - 1 source already contains a bin-like metal waste container.
- **47 additional high-risk tasks**: the target is densely occupied by chairs,
  tables, a bed edge, or other furniture. A bin can sometimes be drawn in front,
  but this may require moving outside the intended edit region, deleting
  furniture, or producing a visibly awkward aisle placement.
- **10 borderline tasks**: a plausible floor exists only as a thin band at the
  bottom or side of the edit region. These need explicit base placement and
  full-silhouette QA.
- **112 source-level clean-enough tasks** remain after all three groups above.
  If only the 91 direct exclusions and 47 high-risk tasks are replaced, 122
  remain, including the 10 borderline cases.

## Direct exclusion: counter/table/wall/display with no proper support (62)

- `trash_can_restaurant_032_slot_001`
- `trash_can_restaurant_040_slot_001`
- `trash_can_restaurant_051_slot_001`
- `trash_can_restaurant_055_slot_001`
- `trash_can_restaurant_057_slot_001`
- `trash_can_restaurant_072_slot_001`
- `trash_can_restaurant_076_slot_001`
- `trash_can_restaurant_081_slot_001`
- `trash_can_restaurant_106_slot_001`
- `trash_can_restaurant_116_slot_001`
- `trash_can_restaurant_124_slot_001`
- `trash_can_restaurant_132_slot_001`
- `trash_can_restaurant_135_slot_001`
- `trash_can_restaurant_162_slot_001`
- `trash_can_restaurant_168_slot_001`
- `trash_can_restaurant_182_slot_001`
- `trash_can_restaurant_194_slot_001`
- `trash_can_restaurant_200_slot_001`
- `trash_can_restaurant_201_slot_001`
- `trash_can_restaurant_206_slot_001`
- `trash_can_restaurant_223_slot_001`
- `trash_can_restaurant_233_slot_001`
- `trash_can_restaurant_234_slot_001`
- `trash_can_restaurant_236_slot_001`
- `trash_can_restaurant_241_slot_001`
- `trash_can_restaurant_242_slot_001`
- `trash_can_restaurant_245_slot_001`
- `trash_can_restaurant_246_slot_001`
- `trash_can_restaurant_250_slot_001`
- `trash_can_restaurant_263_slot_001`
- `trash_can_restaurant_270_slot_001`
- `trash_can_restaurant_276_slot_001`
- `trash_can_restaurant_289_slot_001`
- `trash_can_restaurant_298_slot_001`
- `trash_can_restaurant_299_slot_001`
- `trash_can_lodging_020_slot_001`
- `trash_can_lodging_028_slot_001`
- `trash_can_lodging_063_slot_001`
- `trash_can_lodging_087_slot_001`
- `trash_can_lodging_144_slot_001`
- `trash_can_lodging_169_slot_001`
- `trash_can_lodging_172_slot_001`
- `trash_can_lodging_179_slot_001`
- `trash_can_lodging_180_slot_001`
- `trash_can_lodging_190_slot_001`
- `trash_can_lodging_191_slot_001`
- `trash_can_lodging_199_slot_001`
- `trash_can_lodging_205_slot_001`
- `trash_can_lodging_219_slot_001`
- `trash_can_lodging_235_slot_001`
- `trash_can_lodging_242_slot_001`
- `trash_can_lodging_247_slot_001`
- `trash_can_lodging_249_slot_001`
- `trash_can_lodging_256_slot_001`
- `trash_can_lodging_263_slot_001`
- `trash_can_lodging_269_slot_001`
- `trash_can_lodging_270_slot_001`
- `trash_can_lodging_274_slot_001`
- `trash_can_lodging_285_slot_001`
- `trash_can_lodging_287_slot_001`
- `trash_can_lodging_293_slot_001`
- `trash_can_lodging_295_slot_001`

## Direct exclusion: bed/sofa/soft surface (18)

- `trash_can_lodging_022_slot_001`
- `trash_can_lodging_027_slot_001`
- `trash_can_lodging_049_slot_001`
- `trash_can_lodging_056_slot_001`
- `trash_can_lodging_058_slot_001`
- `trash_can_lodging_061_slot_001`
- `trash_can_lodging_070_slot_001`
- `trash_can_lodging_075_slot_001`
- `trash_can_lodging_121_slot_001`
- `trash_can_lodging_152_slot_001`
- `trash_can_lodging_183_slot_001`
- `trash_can_lodging_197_slot_001`
- `trash_can_lodging_224_slot_001`
- `trash_can_lodging_225_slot_001`
- `trash_can_lodging_261_slot_001`
- `trash_can_lodging_266_slot_001`
- `trash_can_lodging_272_slot_001`
- `trash_can_lodging_281_slot_001`

## Direct exclusion: toilet/shower/bathtub (6)

- `trash_can_lodging_001_slot_001`
- `trash_can_lodging_117_slot_001`
- `trash_can_lodging_122_slot_001`
- `trash_can_lodging_129_slot_001`
- `trash_can_lodging_154_slot_001`
- `trash_can_lodging_182_slot_001`

## Direct exclusion: person occupies the target (4)

- `trash_can_restaurant_164_slot_001`
- `trash_can_restaurant_183_slot_001`
- `trash_can_restaurant_225_slot_001`
- `trash_can_restaurant_262_slot_001`

## Direct exclusion: source already contains a bin-like object (1)

- `trash_can_restaurant_199_slot_001`

## High risk: furniture blocks or crowds the target (47)

- `trash_can_restaurant_003_slot_001`
- `trash_can_restaurant_008_slot_001`
- `trash_can_restaurant_018_slot_001`
- `trash_can_restaurant_022_slot_001`
- `trash_can_restaurant_024_slot_001`
- `trash_can_restaurant_028_slot_001`
- `trash_can_restaurant_029_slot_001`
- `trash_can_restaurant_037_slot_001`
- `trash_can_restaurant_042_slot_001`
- `trash_can_restaurant_043_slot_001`
- `trash_can_restaurant_045_slot_001`
- `trash_can_restaurant_049_slot_001`
- `trash_can_restaurant_052_slot_001`
- `trash_can_restaurant_063_slot_001`
- `trash_can_restaurant_064_slot_001`
- `trash_can_restaurant_071_slot_001`
- `trash_can_restaurant_073_slot_001`
- `trash_can_restaurant_078_slot_001`
- `trash_can_restaurant_089_slot_001`
- `trash_can_restaurant_090_slot_001`
- `trash_can_restaurant_091_slot_001`
- `trash_can_restaurant_101_slot_001`
- `trash_can_restaurant_110_slot_001`
- `trash_can_restaurant_112_slot_001`
- `trash_can_restaurant_122_slot_001`
- `trash_can_restaurant_151_slot_001`
- `trash_can_restaurant_176_slot_001`
- `trash_can_restaurant_188_slot_001`
- `trash_can_restaurant_205_slot_001`
- `trash_can_restaurant_215_slot_001`
- `trash_can_restaurant_243_slot_001`
- `trash_can_restaurant_264_slot_001`
- `trash_can_lodging_007_slot_001`
- `trash_can_lodging_013_slot_001`
- `trash_can_lodging_030_slot_001`
- `trash_can_lodging_032_slot_001`
- `trash_can_lodging_085_slot_001`
- `trash_can_lodging_088_slot_001`
- `trash_can_lodging_091_slot_001`
- `trash_can_lodging_092_slot_001`
- `trash_can_lodging_097_slot_001`
- `trash_can_lodging_108_slot_001`
- `trash_can_lodging_118_slot_001`
- `trash_can_lodging_120_slot_001`
- `trash_can_lodging_148_slot_001`
- `trash_can_lodging_208_slot_001`
- `trash_can_lodging_230_slot_001`

## Borderline: plausible floor only in a thin band (10)

- `trash_can_restaurant_111_slot_001`
- `trash_can_restaurant_115_slot_001`
- `trash_can_restaurant_134_slot_001`
- `trash_can_restaurant_137_slot_001`
- `trash_can_restaurant_149_slot_001`
- `trash_can_lodging_137_slot_001`
- `trash_can_lodging_217_slot_001`
- `trash_can_lodging_238_slot_001`
- `trash_can_lodging_273_slot_001`
- `trash_can_lodging_280_slot_001`

## Pilot evidence

The current 12-image full-object pilot confirms that prompt changes cannot fix
missing support geometry:

- `trash_can_lodging_121_slot_001`: complete bin is generated **on the bed**.
- `trash_can_lodging_272_slot_001`: complete bin is generated **on the sofa
  cushion**.
- `trash_can_restaurant_132_slot_001`: bin sits on the POS/service counter.
- `trash_can_restaurant_206_slot_001`: bin sits in front of food inside/on the
  bakery display.
- `trash_can_restaurant_299_slot_001`: bin sits on the shop counter.
- `trash_can_lodging_293_slot_001`: there is no usable floor margin, so the bin
  base touches/is cut by the bottom crop edge.
- `trash_can_lodging_013_slot_001`: no bin is generated; the chair/floor target
  is too crowded for the prompt constraints.

The pilot also shows that not every chair-rich region must be rejected:
`trash_can_restaurant_000_slot_001` and
`trash_can_restaurant_035_slot_001` produced complete, reasonably natural
floor-standing bins. `trash_can_lodging_096_slot_001` also succeeded by using
the small visible carpet area and is intentionally not in the exclusion list.

## Recommendation

For strict naturalness, replace/re-export the 91 direct exclusions before the
full run. Prefer slots whose edit region contains a visibly connected floor
patch, with the intended bin base at least 8–10% of crop height above the bottom
edge and with clear lateral margin. For the 47 furniture-crowded cases, either
replace them as well or generate only as a manually reviewed retry pool.
Prompting should not be expected to turn a bed/counter/table target into a
natural floor placement while also keeping the object inside the intended edit
region.

## Completed strict run

The 112-task strict manifest was subsequently generated and reviewed. After two
targeted repair rounds, 85/112 outputs passed the combined completeness,
placement-naturalness, and source-style criteria. The final selected directory,
27 explicit exclusions, and per-task review reasons are documented in
`docs/TRASH_CAN_COMPLETE_NATURAL_QA_2026-07-23.md` and
`annotations/trash_can_complete_natural_review_20260723.jsonl`.

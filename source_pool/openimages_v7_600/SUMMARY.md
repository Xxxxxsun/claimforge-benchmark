# Open Images V7 Source Pool Summary
Collected/updated on 2026-06-30 for CLAIMFORGE benchmark expansion.
This pool is for human screening and annotation, not the final benchmark split.

## restaurant
- images: 300
- total size on disk: see `restaurant/`
- min side median: 1200 px
- max side median: 1800 px
- people score max: 0.797 (filtered out >= 0.80)
- top labels:
  - restaurant: 178
  - table: 136
  - food: 111
  - tableware: 96
  - fast_food_restaurant: 81
  - food_court: 46
  - kitchen_dining_table: 27
  - dining_room: 11
  - chinese_restaurant: 5

## lodging
- images: 300
- total size on disk: see `lodging/`
- min side median: 1198 px
- max side median: 1800 px
- people score max: 0.000 (filtered out >= 0.80)
- top labels:
  - bedroom: 113
  - bed: 112
  - bedding: 109
  - bed_sheet: 109
  - boutique_hotel: 69
  - kitchen: 64
  - hotel: 59
  - bathroom: 47
  - living_room: 28
  - restroom: 4
  - motel: 1

## Recommended next step
- Review `contact_sheets/pages/*.jpg`.
- Mark keep/reject for each image before slot annotation.
- Prefer indoor/table/room/bathroom/kitchen surfaces where a small localized defect/object can plausibly be inserted.
- Reject images dominated by signs, menus, posters, exterior-only views, heavy crowds/faces, or scenes with no plausible local edit surface.

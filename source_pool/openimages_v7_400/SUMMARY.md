# Open Images V7 Source Pool Summary
Collected on 2026-06-29 for CLAIMFORGE benchmark expansion.
This pool is for human screening and annotation, not the final benchmark split.
## restaurant
- images: 200
- total size on disk: see `restaurant/`
- min side median: 1200 px
- max side median: 1800 px
- people score max: 0.797 (filtered out >= 0.80)
- top labels:
  - restaurant: 168
  - table: 117
  - fast_food_restaurant: 75
  - food_court: 46
  - kitchen_dining_table: 27
  - food: 21
  - tableware: 19
  - dining_room: 9
  - chinese_restaurant: 3

## lodging
- images: 200
- total size on disk: see `lodging/`
- min side median: 1198 px
- max side median: 1800 px
- people score max: 0.000 (filtered out >= 0.80)
- top labels:
  - bed: 98
  - bedroom: 97
  - bed_sheet: 96
  - bedding: 94
  - boutique_hotel: 66
  - hotel: 55
  - bathroom: 28
  - living_room: 19
  - kitchen: 11
  - restroom: 2
  - motel: 1

## Recommended next step
- Review `contact_sheets/pages/*.jpg`.
- Mark keep/reject for each image before slot annotation.
- Prefer indoor/table/room/bathroom/kitchen surfaces where a small localized defect/object can plausibly be inserted.
- Reject images dominated by signs, menus, posters, exterior-only views, heavy crowds/faces, or scenes with no plausible local edit surface.

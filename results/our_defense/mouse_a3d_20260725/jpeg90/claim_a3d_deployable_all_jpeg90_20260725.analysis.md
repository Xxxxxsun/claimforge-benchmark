# A3D evaluation

Q1 is the smallest-edit quintile and Q2-Q5 contains the remaining larger edits. The analyzer does not fit or tune any A3D parameter.

## Detection by size scope

| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 275 | 0.5662 | 0.6166 | 0.6101 | 0.1164 | 0.1745 | 0.1782 |
| q1_smallest | 55 | 0.5273 | 0.5970 | 0.5831 | 0.0545 | 0.1273 | 0.1091 |
| q2_q5_larger | 220 | 0.5795 | 0.6221 | 0.6187 | 0.1318 | 0.2045 | 0.2000 |

## Size quintiles

| Quintile | Median edit % | Full AUROC | A3D AUROC | Full pixel F1 | A3D pixel F1 | Proposal hit |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.05347 | 0.5273 | 0.5831 | 0.0923 | 0.1398 | 0.5636 |
| Q2 | 0.07780 | 0.5514 | 0.5907 | 0.1600 | 0.1295 | 0.6909 |
| Q3 | 0.11264 | 0.5712 | 0.5871 | 0.2020 | 0.1905 | 0.8182 |
| Q4 | 0.16293 | 0.5739 | 0.6486 | 0.2521 | 0.2661 | 0.8182 |
| Q5 | 0.37441 | 0.6549 | 0.6793 | 0.3117 | 0.3014 | 0.9636 |

## Paired bootstrap delta, A3D minus full

| Scope | Metric | 95% CI |
|---|---|---:|
| all | A3D local auroc | [0.0309, 0.0715] |
| all | A3D local average_precision | [0.0346, 0.0817] |
| all | A3D local tpr_at_fpr_5_percent | [0.0145, 0.1091] |
| all | A3D fused auroc | [0.0280, 0.0606] |
| all | A3D fused average_precision | [0.0317, 0.0677] |
| all | A3D fused tpr_at_fpr_5_percent | [0.0073, 0.1018] |
| all | pixel pixel_ap | [-0.0003, 0.0532] |
| all | pixel f1 | [-0.0200, 0.0234] |
| q1_smallest | A3D local auroc | [0.0162, 0.1233] |
| q1_smallest | A3D local average_precision | [0.0101, 0.1364] |
| q1_smallest | A3D local tpr_at_fpr_5_percent | [-0.0364, 0.1636] |
| q1_smallest | A3D fused auroc | [0.0102, 0.1021] |
| q1_smallest | A3D fused average_precision | [0.0031, 0.1270] |
| q1_smallest | A3D fused tpr_at_fpr_5_percent | [-0.0545, 0.1636] |
| q1_smallest | pixel pixel_ap | [0.0129, 0.1437] |
| q1_smallest | pixel f1 | [0.0024, 0.1001] |
| q2_q5_larger | A3D local auroc | [0.0225, 0.0639] |
| q2_q5_larger | A3D local average_precision | [0.0230, 0.0738] |
| q2_q5_larger | A3D local tpr_at_fpr_5_percent | [0.0000, 0.1182] |
| q2_q5_larger | A3D fused auroc | [0.0244, 0.0562] |
| q2_q5_larger | A3D fused average_precision | [0.0257, 0.0638] |
| q2_q5_larger | A3D fused tpr_at_fpr_5_percent | [-0.0045, 0.1091] |
| q2_q5_larger | pixel pixel_ap | [-0.0136, 0.0439] |
| q2_q5_larger | pixel f1 | [-0.0332, 0.0149] |

## Fixed cross-object operating point

Threshold `0.635351012` was calibrated from 80 dev-real images in a separate result set.

| Evaluation scope | FPR | TPR |
|---|---:|---:|
| all | 0.0473 | 0.1600 |
| hash_test | 0.0513 | 0.1487 |

# A3D evaluation

Q1 is the smallest-edit quintile and Q2-Q5 contains the remaining larger edits. The analyzer does not fit or tune any A3D parameter.

## Detection by size scope

| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 501 | 0.9493 | 0.9427 | 0.9567 | 0.8184 | 0.8244 | 0.8743 |
| q1_smallest | 101 | 0.9246 | 0.9499 | 0.9530 | 0.7327 | 0.8119 | 0.8416 |
| q2_q5_larger | 400 | 0.9589 | 0.9407 | 0.9587 | 0.8375 | 0.8275 | 0.8750 |

## Size quintiles

| Quintile | Median edit % | Full AUROC | A3D AUROC | Full pixel F1 | A3D pixel F1 | Proposal hit |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.36337 | 0.9246 | 0.9530 | 0.7288 | 0.8143 | 0.9901 |
| Q2 | 0.65818 | 0.9758 | 0.9842 | 0.8007 | 0.8038 | 1.0000 |
| Q3 | 0.99034 | 0.9499 | 0.9506 | 0.7934 | 0.7702 | 1.0000 |
| Q4 | 1.52681 | 0.9441 | 0.9405 | 0.7624 | 0.7012 | 1.0000 |
| Q5 | 3.10085 | 0.9702 | 0.9569 | 0.8164 | 0.6509 | 1.0000 |

## Paired bootstrap delta, A3D minus full

| Scope | Metric | 95% CI |
|---|---|---:|
| all | A3D local auroc | [-0.0179, 0.0057] |
| all | A3D local average_precision | [-0.0194, 0.0043] |
| all | A3D local tpr_at_fpr_5_percent | [-0.0399, 0.0619] |
| all | A3D fused auroc | [-0.0000, 0.0156] |
| all | A3D fused average_precision | [0.0005, 0.0130] |
| all | A3D fused tpr_at_fpr_5_percent | [0.0100, 0.0978] |
| all | pixel pixel_ap | [-0.0701, -0.0284] |
| all | pixel f1 | [-0.0502, -0.0129] |
| q1_smallest | A3D local auroc | [-0.0043, 0.0563] |
| q1_smallest | A3D local average_precision | [0.0003, 0.0428] |
| q1_smallest | A3D local tpr_at_fpr_5_percent | [-0.0099, 0.1683] |
| q1_smallest | A3D fused auroc | [0.0053, 0.0524] |
| q1_smallest | A3D fused average_precision | [0.0080, 0.0414] |
| q1_smallest | A3D fused tpr_at_fpr_5_percent | [0.0198, 0.1782] |
| q1_smallest | pixel pixel_ap | [0.0274, 0.0971] |
| q1_smallest | pixel f1 | [0.0503, 0.1198] |
| q2_q5_larger | A3D local auroc | [-0.0308, -0.0064] |
| q2_q5_larger | A3D local average_precision | [-0.0324, -0.0044] |
| q2_q5_larger | A3D local tpr_at_fpr_5_percent | [-0.0650, 0.0500] |
| q2_q5_larger | A3D fused auroc | [-0.0072, 0.0070] |
| q2_q5_larger | A3D fused average_precision | [-0.0053, 0.0069] |
| q2_q5_larger | A3D fused tpr_at_fpr_5_percent | [-0.0175, 0.0875] |
| q2_q5_larger | pixel pixel_ap | [-0.1033, -0.0551] |
| q2_q5_larger | pixel f1 | [-0.0835, -0.0403] |

## Fixed cross-object operating point

Threshold `0.635351012` was calibrated from 80 dev-real images in a separate result set.

| Evaluation scope | FPR | TPR |
|---|---:|---:|
| all | 0.0439 | 0.8683 |
| hash_test | 0.0536 | 0.8549 |

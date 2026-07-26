# A3D evaluation

Q1 is the smallest-edit quintile and Q2-Q5 contains the remaining larger edits. The analyzer does not fit or tune any A3D parameter.

## Detection by size scope

| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 250 | 0.9291 | 0.9051 | 0.9318 | 0.7280 | 0.7040 | 0.7880 |
| q1_smallest | 50 | 0.9072 | 0.9228 | 0.9376 | 0.6800 | 0.7400 | 0.7800 |
| q2_q5_larger | 200 | 0.9371 | 0.9016 | 0.9314 | 0.7250 | 0.6850 | 0.7950 |

## Size quintiles

| Quintile | Median edit % | Full AUROC | A3D AUROC | Full pixel F1 | A3D pixel F1 | Proposal hit |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.47607 | 0.9072 | 0.9376 | 0.6746 | 0.7274 | 0.9800 |
| Q2 | 0.88195 | 0.9348 | 0.9388 | 0.7231 | 0.6995 | 1.0000 |
| Q3 | 1.26413 | 0.9260 | 0.9192 | 0.7027 | 0.6451 | 1.0000 |
| Q4 | 1.84176 | 0.9564 | 0.9372 | 0.7568 | 0.6631 | 1.0000 |
| Q5 | 3.53573 | 0.9536 | 0.9368 | 0.7677 | 0.5487 | 1.0000 |

## Paired bootstrap delta, A3D minus full

| Scope | Metric | 95% CI |
|---|---|---:|
| all | A3D local auroc | [-0.0464, -0.0028] |
| all | A3D local average_precision | [-0.0470, 0.0022] |
| all | A3D local tpr_at_fpr_5_percent | [-0.1240, 0.0720] |
| all | A3D fused auroc | [-0.0102, 0.0156] |
| all | A3D fused average_precision | [-0.0088, 0.0166] |
| all | A3D fused tpr_at_fpr_5_percent | [-0.0361, 0.1280] |
| all | pixel pixel_ap | [-0.1001, -0.0354] |
| all | pixel f1 | [-0.0972, -0.0399] |
| q1_smallest | A3D local auroc | [-0.0400, 0.0724] |
| q1_smallest | A3D local average_precision | [-0.0264, 0.0558] |
| q1_smallest | A3D local tpr_at_fpr_5_percent | [-0.1000, 0.2200] |
| q1_smallest | A3D fused auroc | [-0.0096, 0.0728] |
| q1_smallest | A3D fused average_precision | [-0.0048, 0.0567] |
| q1_smallest | A3D fused tpr_at_fpr_5_percent | [-0.0600, 0.2005] |
| q1_smallest | pixel pixel_ap | [0.0219, 0.1473] |
| q1_smallest | pixel f1 | [-0.0015, 0.1116] |
| q2_q5_larger | A3D local auroc | [-0.0573, -0.0142] |
| q2_q5_larger | A3D local average_precision | [-0.0592, -0.0057] |
| q2_q5_larger | A3D local tpr_at_fpr_5_percent | [-0.1650, 0.0601] |
| q2_q5_larger | A3D fused auroc | [-0.0176, 0.0055] |
| q2_q5_larger | A3D fused average_precision | [-0.0163, 0.0112] |
| q2_q5_larger | A3D fused tpr_at_fpr_5_percent | [-0.0650, 0.1300] |
| q2_q5_larger | pixel pixel_ap | [-0.1395, -0.0679] |
| q2_q5_larger | pixel f1 | [-0.1293, -0.0651] |

## Fixed cross-object operating point

Threshold `0.635351012` was calibrated from 80 dev-real images in a separate result set.

| Evaluation scope | FPR | TPR |
|---|---:|---:|
| all | 0.0480 | 0.7880 |
| hash_test | 0.0552 | 0.7791 |

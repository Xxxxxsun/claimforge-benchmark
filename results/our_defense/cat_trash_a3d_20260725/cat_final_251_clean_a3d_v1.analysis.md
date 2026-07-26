# A3D evaluation

Q1 is the smallest-edit quintile and Q2-Q5 contains the remaining larger edits. The analyzer does not fit or tune any A3D parameter.

## Detection by size scope

| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 251 | 0.9693 | 0.9791 | 0.9808 | 0.9004 | 0.9402 | 0.9522 |
| q1_smallest | 51 | 0.9104 | 0.9558 | 0.9508 | 0.7451 | 0.8235 | 0.8431 |
| q2_q5_larger | 200 | 0.9880 | 0.9849 | 0.9898 | 0.9400 | 0.9750 | 0.9800 |

## Size quintiles

| Quintile | Median edit % | Full AUROC | A3D AUROC | Full pixel F1 | A3D pixel F1 | Proposal hit |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.28084 | 0.9104 | 0.9508 | 0.7388 | 0.8434 | 1.0000 |
| Q2 | 0.53277 | 0.9908 | 0.9944 | 0.8307 | 0.8633 | 1.0000 |
| Q3 | 0.76753 | 0.9916 | 0.9928 | 0.8675 | 0.8616 | 1.0000 |
| Q4 | 1.11163 | 0.9908 | 0.9864 | 0.8632 | 0.8379 | 1.0000 |
| Q5 | 2.25905 | 0.9928 | 0.9936 | 0.8780 | 0.7902 | 1.0000 |

## Paired bootstrap delta, A3D minus full

| Scope | Metric | 95% CI |
|---|---|---:|
| all | A3D local auroc | [-0.0020, 0.0226] |
| all | A3D local average_precision | [-0.0096, 0.0153] |
| all | A3D local tpr_at_fpr_5_percent | [0.0000, 0.0837] |
| all | A3D fused auroc | [0.0028, 0.0211] |
| all | A3D fused average_precision | [0.0023, 0.0149] |
| all | A3D fused tpr_at_fpr_5_percent | [0.0159, 0.0876] |
| all | pixel pixel_ap | [-0.0583, -0.0064] |
| all | pixel f1 | [-0.0192, 0.0278] |
| q1_smallest | A3D local auroc | [0.0011, 0.0923] |
| q1_smallest | A3D local average_precision | [0.0056, 0.0661] |
| q1_smallest | A3D local tpr_at_fpr_5_percent | [-0.0196, 0.2157] |
| q1_smallest | A3D fused auroc | [0.0054, 0.0780] |
| q1_smallest | A3D fused average_precision | [0.0087, 0.0578] |
| q1_smallest | A3D fused tpr_at_fpr_5_percent | [0.0196, 0.2157] |
| q1_smallest | pixel pixel_ap | [0.0121, 0.1031] |
| q1_smallest | pixel f1 | [0.0550, 0.1617] |
| q2_q5_larger | A3D local auroc | [-0.0132, 0.0045] |
| q2_q5_larger | A3D local average_precision | [-0.0222, 0.0049] |
| q2_q5_larger | A3D local tpr_at_fpr_5_percent | [-0.0250, 0.0650] |
| q2_q5_larger | A3D fused auroc | [-0.0033, 0.0063] |
| q2_q5_larger | A3D fused average_precision | [-0.0032, 0.0061] |
| q2_q5_larger | A3D fused tpr_at_fpr_5_percent | [0.0000, 0.0700] |
| q2_q5_larger | pixel pixel_ap | [-0.0871, -0.0269] |
| q2_q5_larger | pixel f1 | [-0.0485, 0.0036] |

## Fixed cross-object operating point

Threshold `0.635351012` was calibrated from 80 dev-real images in a separate result set.

| Evaluation scope | FPR | TPR |
|---|---:|---:|
| all | 0.0398 | 0.9482 |
| hash_test | 0.0519 | 0.9351 |

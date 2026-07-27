# A3D evaluation

Q1 is the smallest-edit quintile and Q2-Q5 contains the remaining larger edits. The analyzer does not fit or tune any A3D parameter.

## Detection by size scope

| Scope | Pairs | Full AUROC | Local AUROC | Fused AUROC | Full TPR@5% | Local TPR@5% | Fused TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 275 | 0.8180 | 0.8798 | 0.8796 | 0.4436 | 0.6727 | 0.6655 |
| q1_smallest | 55 | 0.7226 | 0.8600 | 0.8387 | 0.2182 | 0.6000 | 0.4545 |
| q2_q5_larger | 220 | 0.8537 | 0.8855 | 0.8946 | 0.5182 | 0.6955 | 0.7500 |

## Size quintiles

| Quintile | Median edit % | Full AUROC | A3D AUROC | Full pixel F1 | A3D pixel F1 | Proposal hit |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.05347 | 0.7226 | 0.8387 | 0.3468 | 0.4860 | 0.9091 |
| Q2 | 0.07780 | 0.8155 | 0.8889 | 0.4774 | 0.5502 | 0.9091 |
| Q3 | 0.11264 | 0.8575 | 0.8926 | 0.5196 | 0.5770 | 0.9818 |
| Q4 | 0.16293 | 0.8489 | 0.8648 | 0.5306 | 0.5680 | 1.0000 |
| Q5 | 0.37441 | 0.9412 | 0.9438 | 0.6260 | 0.6181 | 1.0000 |

## Paired bootstrap delta, A3D minus full

| Scope | Metric | 95% CI |
|---|---|---:|
| all | A3D local auroc | [0.0404, 0.0837] |
| all | A3D local average_precision | [0.0412, 0.0864] |
| all | A3D local tpr_at_fpr_5_percent | [0.1491, 0.3236] |
| all | A3D fused auroc | [0.0448, 0.0788] |
| all | A3D fused average_precision | [0.0462, 0.0784] |
| all | A3D fused tpr_at_fpr_5_percent | [0.1345, 0.3236] |
| all | pixel pixel_ap | [0.1003, 0.1449] |
| all | pixel f1 | [0.0392, 0.0835] |
| q1_smallest | A3D local auroc | [0.0798, 0.1884] |
| q1_smallest | A3D local average_precision | [0.0808, 0.2008] |
| q1_smallest | A3D local tpr_at_fpr_5_percent | [0.0177, 0.5818] |
| q1_smallest | A3D fused auroc | [0.0674, 0.1636] |
| q1_smallest | A3D fused average_precision | [0.0750, 0.1667] |
| q1_smallest | A3D fused tpr_at_fpr_5_percent | [0.0182, 0.4182] |
| q1_smallest | pixel pixel_ap | [0.1147, 0.2388] |
| q1_smallest | pixel f1 | [0.0736, 0.2055] |
| q2_q5_larger | A3D local auroc | [0.0095, 0.0563] |
| q2_q5_larger | A3D local average_precision | [0.0149, 0.0633] |
| q2_q5_larger | A3D local tpr_at_fpr_5_percent | [0.1045, 0.2818] |
| q2_q5_larger | A3D fused auroc | [0.0238, 0.0606] |
| q2_q5_larger | A3D fused average_precision | [0.0286, 0.0636] |
| q2_q5_larger | A3D fused tpr_at_fpr_5_percent | [0.1227, 0.3136] |
| q2_q5_larger | pixel pixel_ap | [0.0850, 0.1312] |
| q2_q5_larger | pixel f1 | [0.0199, 0.0628] |

## Fixed cross-object operating point

Threshold `0.635351012` was calibrated from 80 dev-real images in a separate result set.

| Evaluation scope | FPR | TPR |
|---|---:|---:|
| all | 0.0509 | 0.7164 |
| hash_test | 0.0564 | 0.7231 |

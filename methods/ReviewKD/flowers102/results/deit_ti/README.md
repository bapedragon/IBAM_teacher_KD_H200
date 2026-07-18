# Flowers-102 / DeiT-Ti / ReviewKD result

## Result

| Item | Value |
|---|---:|
| Method | ReviewKD |
| Teacher | ResNet56, 64.64% Top-1 |
| Student | DeiT-Ti |
| Epochs | 200 |
| Vanilla Top-1 | 50.06% |
| ReviewKD best Top-1 | **50.76%** |
| Gain over Vanilla | **+0.70pp** |
| Best epoch | 158 |
| Latest Top-1 | 50.46% |
| Elapsed time | 36m 49s |

The selected result is the best checkpoint at epoch 158.

## Method configuration

- Official repository: `dvlab-research/ReviewKD`
- Pinned commit: `cede6ea6387ae9b6127de0e561507177bf19c11e`
- Loss: `CE + ramp(epoch, 20) * 0.6 * HCL`
- Student blocks: `3, 7, 11`
- Teacher stages: `stage1, stage2, stage3`
- Base protocol: `flowers102_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_414`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

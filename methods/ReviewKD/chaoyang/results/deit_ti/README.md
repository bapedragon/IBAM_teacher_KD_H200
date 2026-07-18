# Chaoyang / DeiT-Ti / ReviewKD result

## Result

| Item | Value |
|---|---:|
| Method | ReviewKD |
| Teacher | ResNet56 latest, 81.53% Top-1 |
| Student | DeiT-Ti |
| Epochs | 100 |
| Vanilla Top-1 | 82.00% |
| ReviewKD best Top-1 | **81.53%** |
| Gain over Vanilla | **-0.47pp** |
| Best epoch | 78 |
| Latest Top-1 | 80.27% |
| Elapsed time | 13m 32s |

The selected result is the best checkpoint at epoch 78. The fixed teacher is the epoch-300 latest teacher checkpoint used by every Chaoyang KD method.

## Method configuration

- Official repository: `dvlab-research/ReviewKD`
- Pinned commit: `cede6ea6387ae9b6127de0e561507177bf19c11e`
- Loss: `CE + ramp(epoch, 20) * 0.6 * HCL`
- Student blocks: `3, 7, 11`
- Teacher stages: `stage1, stage2, stage3`
- Base protocol: `chaoyang_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_417`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

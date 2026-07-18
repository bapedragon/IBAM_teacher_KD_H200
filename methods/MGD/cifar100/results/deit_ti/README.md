# CIFAR-100 / DeiT-Ti / MGD result

## Result

| Item | Value |
|---|---:|
| Method | MGD |
| Teacher | ResNet56, 68.68% Top-1 |
| Student | DeiT-Ti |
| Epochs | 300 |
| Vanilla Top-1 | 65.08% |
| MGD best Top-1 | **73.71%** |
| Gain over Vanilla | **+8.63pp** |
| Best epoch | 250 |
| Latest Top-1 | 73.25% |
| Elapsed time | 2h 36m 43s |

The selected result is the best checkpoint at epoch 250.

## Method configuration

- Official repository: `yzd-v/MGD`
- Pinned commit: `2c9da0b28625eb948db57afc02c824452c3910fe`
- Loss: `CE + 7e-5 * masked feature reconstruction`
- Mask probability: `0.15`, channel mask
- Student block: `11`; teacher feature: `stage3`
- Base protocol: `cifar100_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_419`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

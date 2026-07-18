# CIFAR-100 / DeiT-Ti / OFA result

## Result

| Item | Value |
|---|---:|
| Method | OFA |
| Teacher | ResNet56, 68.68% Top-1 |
| Student | DeiT-Ti |
| Epochs | 300 |
| Vanilla Top-1 | 65.08% |
| OFA best Top-1 | **66.18%** |
| Gain over Vanilla | **+1.10pp** |
| Best epoch | 227 |
| Latest Top-1 | 65.92% |
| Elapsed time | 4h 00m 32s |

The selected result is the best checkpoint at epoch 227.

## Method configuration

- Official repository: `Hao840/OFAKD`
- Pinned commit: `f7bb896cac9879040800bde08a8cc2057a904c52`
- Loss: `1.0 * CE + 1.0 * final OFA + 1.0 * intermediate OFA`
- Epsilon: `1.0`; temperature: `1.0`
- Student stages: `1, 2, 3, 4`
- Base protocol: `cifar100_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_425`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

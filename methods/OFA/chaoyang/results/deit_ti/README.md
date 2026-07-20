# Chaoyang / DeiT-Ti / OFA result

## Result

| Item | Value |
|---|---:|
| Method | OFA |
| Teacher | ResNet56 latest, 81.53% Top-1 |
| Student | DeiT-Ti |
| Epochs | 100 |
| Vanilla Top-1 | 82.00% |
| OFA best Top-1 | **80.04%** |
| Gain over Vanilla | **-1.96pp** |
| Best epoch | 74 |
| Latest Top-1 | 79.43% |
| Elapsed time | 16m 05s |

The selected result is the best checkpoint at epoch 74. The same fixed
epoch-300 teacher checkpoint used by every Chaoyang generic-KD run was used.

## Method configuration

- Official repository: `Hao840/OFAKD`
- Pinned commit: `f7bb896cac9879040800bde08a8cc2057a904c52`
- Loss: `1.0 * CE + 1.0 * final OFA + 1.0 * intermediate OFA`
- Epsilon: `1.0`; temperature: `1.0`
- Student stages: `1, 2, 3, 4`
- Base protocol: `chaoyang_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_429`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

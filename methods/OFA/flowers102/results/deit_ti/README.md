# Flowers-102 / DeiT-Ti / OFA result

## Result

| Item | Value |
|---|---:|
| Method | OFA |
| Teacher | ResNet56, 64.64% Top-1 |
| Student | DeiT-Ti |
| Epochs | 200 |
| Vanilla Top-1 | 50.06% |
| OFA best Top-1 | **44.07%** |
| Gain over Vanilla | **-5.99pp** |
| Best epoch | 190 |
| Latest Top-1 | 43.96% |
| Elapsed time | 38m 19s |

The selected result is the best checkpoint at epoch 190.

## Method configuration

- Official repository: `Hao840/OFAKD`
- Pinned commit: `f7bb896cac9879040800bde08a8cc2057a904c52`
- Loss: `1.0 * CE + 1.0 * final OFA + 1.0 * intermediate OFA`
- Epsilon: `1.0`; temperature: `1.0`
- Student stages: `1, 2, 3, 4`
- Base protocol: `flowers102_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_427`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

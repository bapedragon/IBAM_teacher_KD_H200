# Flowers-102 / DeiT-Ti / MGD result

## Result

| Item | Value |
|---|---:|
| Method | MGD |
| Teacher | ResNet56, 64.64% Top-1 |
| Student | DeiT-Ti |
| Epochs | 200 |
| Vanilla Top-1 | 50.06% |
| MGD best Top-1 | **51.57%** |
| Gain over Vanilla | **+1.51pp** |
| Best epoch | 101 |
| Latest Top-1 | 50.94% |
| Elapsed time | 36m 25s |

The selected result is the best checkpoint at epoch 101.

## Method configuration

- Official repository: `yzd-v/MGD`
- Pinned commit: `2c9da0b28625eb948db57afc02c824452c3910fe`
- Loss: `CE + 7e-5 * masked feature reconstruction`
- Mask probability: `0.15`, channel mask
- Student block: `11`; teacher feature: `stage3`
- Base protocol: `flowers102_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_421`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

# Chaoyang / DeiT-Ti / MGD result

## Result

| Item | Value |
|---|---:|
| Method | MGD |
| Teacher | ResNet56 latest, 81.53% Top-1 |
| Student | DeiT-Ti |
| Epochs | 100 |
| Vanilla Top-1 | 82.00% |
| MGD best Top-1 | **81.81%** |
| Gain over Vanilla | **-0.19pp** |
| Best epoch | 61 |
| Latest Top-1 | 80.04% |
| Elapsed time | 13m 10s |

The selected result is the best checkpoint at epoch 61. The fixed teacher is the epoch-300 latest teacher checkpoint used by every Chaoyang KD method.

## Method configuration

- Official repository: `yzd-v/MGD`
- Pinned commit: `2c9da0b28625eb948db57afc02c824452c3910fe`
- Loss: `CE + 7e-5 * masked feature reconstruction`
- Mask probability: `0.15`, channel mask
- Student block: `11`; teacher feature: `stage3`
- Base protocol: `chaoyang_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_423`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

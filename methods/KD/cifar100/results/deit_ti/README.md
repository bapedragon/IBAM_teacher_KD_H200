# CIFAR-100 / DeiT-Ti / KD result

## Result

| Item | Value |
|---|---:|
| Method | Standard logit KD |
| Teacher | ResNet56, 68.68% Top-1 |
| Student | DeiT-Ti |
| Epochs | 300 |
| Vanilla Top-1 | 65.08% |
| KD best Top-1 | **67.00%** |
| Gain over Vanilla | **+1.92pp** |
| Best epoch | 191 |
| Latest Top-1 | 66.14% |
| Elapsed time | 2h 38m 40s |

The best checkpoint, not the epoch-300 latest checkpoint, is the selected
student result.

## Fixed KD configuration

- Temperature: `4.0`
- KD weight: `0.9`
- Label smoothing: `0.1`
- Seed: `42`
- Optimizer: AdamW
- Initial learning rate: `5e-4`
- Weight decay: `0.05`
- Warm-up: 20 epochs
- Schedule: cosine decay
- Batch size: `128`
- Image resolution: `224 x 224`

No per-method hyperparameter search was performed. This fixed, documented
configuration is the KD baseline configuration for subsequent datasets and
students unless the experiment policy is explicitly revised.

## Files

- `summary.json`: machine-readable configuration, metrics, timing, and teacher metadata
- `training.log`: complete H200 log from job `bapedragon_393`
- `artifact_manifest.json`: archive and checkpoint hashes

The original checkpoint ZIP is kept outside Git history so future H200 jobs do
not need to clone every completed student weight. Its canonical local archive
name is recorded in `artifact_manifest.json`.

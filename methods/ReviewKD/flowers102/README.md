# ReviewKD: Flowers-102 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `64.64%`
- Student: DeiT-Ti, scratch
- Split: official train+val for training, official test for evaluation
- Epochs: `200`
- Batch: `64`
- AdamW: LR `5e-4`, weight decay `0.05`
- LR schedule: 5-epoch warm-up and cosine decay
- ReviewKD: `CE + ramp * 0.6 * HCL`, feature ramp `20` epochs
- Image size: `224`
- Label smoothing: `0.1`
- Seed: `42`

Timing:

```bash
python methods/ReviewKD/flowers102/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/ReviewKD/flowers102/train.py --num-workers 4 --run-name reviewkd_flowers102_deit_ti_200ep --output-dir /app/output
```

## Completed result

| Best epoch | Best Top-1 | Latest Top-1 | Vanilla | Gain |
|---:|---:|---:|---:|---:|
| 158 | **50.76%** | 50.46% | 50.06% | +0.70pp |

See [results/deit_ti](results/deit_ti/) for the summary, full log, and artifact integrity manifest.

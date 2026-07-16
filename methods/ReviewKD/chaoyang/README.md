# ReviewKD: Chaoyang / DeiT-Ti

- Data: official mounted split under `/app/data/chaoyang`
- Teacher: fixed ResNet56 latest checkpoint, Top-1 `81.53%`
- Student: DeiT-Ti, scratch
- Epochs: `100`
- Batch: `64`
- AdamW: LR `5e-4`, weight decay `0.05`
- LR schedule: 5-epoch warm-up and cosine decay
- ReviewKD: `CE + ramp * 0.6 * HCL`, feature ramp `20` epochs
- Image size: `224`
- Label smoothing: `0.1`
- Seed: `42`

Timing:

```bash
python methods/ReviewKD/chaoyang/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/ReviewKD/chaoyang/train.py --num-workers 4 --run-name reviewkd_chaoyang_deit_ti_100ep --output-dir /app/output
```

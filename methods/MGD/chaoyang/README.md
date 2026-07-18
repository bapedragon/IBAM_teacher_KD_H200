# MGD: Chaoyang / DeiT-Ti

- Teacher: fixed ResNet56 latest checkpoint, epoch 300 / Top-1 `81.53%`
- Student: DeiT-Ti from scratch
- Base protocol: 100 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- MGD: official classification alpha `0.00007`, mask probability `0.15`
- Adapter: DeiT block 11 `14 x 14`, `192 -> 64`, ResNet56 stage3
- Dataset mount: `/app/data/chaoyang`
- Seed: `42`

Timing:

```bash
python methods/MGD/chaoyang/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/MGD/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name mgd_chaoyang_deit_ti_100ep --output-dir /app/output
```

Any teacher-gap adjustment is reporting-only and is not applied to training or
checkpoint selection.

## Completed result

| Best epoch | Best Top-1 | Latest Top-1 | Vanilla | Gain |
|---:|---:|---:|---:|---:|
| 61 | **81.81%** | 80.04% | 82.00% | -0.19pp |

See [results/deit_ti](results/deit_ti/) for the summary, full log, and artifact integrity manifest.

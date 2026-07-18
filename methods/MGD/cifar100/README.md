# MGD: CIFAR-100 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `68.68%`
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, warm-up 20, cosine
- MGD: official classification alpha `0.00007`, mask probability `0.15`
- Adapter: DeiT block 11 `14 x 14`, `192 -> 64`, ResNet56 stage3
- Seed: `42`

Timing:

```bash
python methods/MGD/cifar100/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/MGD/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name mgd_cifar100_deit_ti_300ep --output-dir /app/output
```

## Completed result

| Best epoch | Best Top-1 | Latest Top-1 | Vanilla | Gain |
|---:|---:|---:|---:|---:|
| 250 | **73.71%** | 73.25% | 65.08% | +8.63pp |

See [results/deit_ti](results/deit_ti/) for the summary, full log, and artifact integrity manifest.

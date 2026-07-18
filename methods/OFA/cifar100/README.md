# OFA: CIFAR-100 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `68.68%`
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, warm-up 20, cosine
- OFA: stages 1/2/3/4, epsilon `1.0`, temperature `1.0`
- Loss weights: CE `1.0`, final OFA `1.0`, intermediate OFA `1.0`
- Seed: `42`

Timing:

```bash
python methods/OFA/cifar100/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/OFA/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ofa_cifar100_deit_ti_300ep --output-dir /app/output
```

## Completed result

| Epochs | Best epoch | Best Top-1 | Latest Top-1 | Vanilla | Gain |
|---:|---:|---:|---:|---:|---:|
| 300 | 227 | **66.18%** | 65.92% | 65.08% | +1.10pp |

- Elapsed time: `4h 00m 32s`
- Best checkpoint: `/app/output/ofa_cifar100_deit_ti_300ep/student_best.pt`
- Latest checkpoint: `/app/output/ofa_cifar100_deit_ti_300ep/student_latest.pt`

See [results/deit_ti](results/deit_ti/) for the summary, full log, and artifact integrity manifest.

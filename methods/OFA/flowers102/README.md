# OFA: Flowers-102 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `64.64%`
- Student: DeiT-Ti from scratch
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- OFA: stages 1/2/3/4, epsilon `1.0`, temperature `1.0`
- Loss weights: CE `1.0`, final OFA `1.0`, intermediate OFA `1.0`
- Seed: `42`

Timing:

```bash
python methods/OFA/flowers102/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/OFA/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ofa_flowers102_deit_ti_200ep --output-dir /app/output
```

## Completed result

- Best Top-1: **44.07%** at epoch 190
- Latest Top-1: 43.96% at epoch 200
- Vanilla reference: 50.06% (gain: -5.99pp)
- Runtime: 38m 19s
- Verified files: [`results/deit_ti/`](results/deit_ti/)

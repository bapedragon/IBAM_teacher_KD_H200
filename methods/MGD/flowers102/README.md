# MGD: Flowers-102 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `64.64%`
- Student: DeiT-Ti from scratch
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- MGD: official classification alpha `0.00007`, mask probability `0.15`
- Adapter: DeiT block 11 `14 x 14`, `192 -> 64`, ResNet56 stage3
- Seed: `42`

Timing:

```bash
python methods/MGD/flowers102/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/MGD/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name mgd_flowers102_deit_ti_200ep --output-dir /app/output
```

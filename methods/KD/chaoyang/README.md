# KD on Chaoyang

## Fixed inputs

- Teacher: CIFAR-style ResNet56, frozen
- Selected checkpoint: `checkpoints/teachers/chaoyang/teacher_resnet56_chaoyang_latest.pt`
- Selected checkpoint epoch/Top-1: 300 / 81.53%
- Paper teacher reference: 77.20%
- Dataset: official Chaoyang train/test split mounted at `/app/data/chaoyang`
- DeiT-Ti Vanilla reference: 82.00%

Chaoyang intentionally uses the epoch-300 `latest` checkpoint rather than the
83.08% `best` checkpoint. The common student and KD-loss protocol is recorded
in `methods/KD/README.md`.

## Timing run

```bash
python methods/KD/chaoyang/train.py --student deit_ti --timing-run --batch-size 128 --num-workers 4
```

## Full run

```bash
python methods/KD/chaoyang/train.py --student deit_ti --student-epochs 300 --batch-size 128 --num-workers 4 --output-dir /app/output
```

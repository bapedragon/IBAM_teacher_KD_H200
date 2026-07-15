# KD on Flowers-102

## Fixed inputs

- Teacher: CIFAR-style ResNet56, frozen
- Teacher checkpoint: `checkpoints/teachers/flowers102/teacher_resnet56_flowers_best.pt`
- Teacher checkpoint epoch/Top-1: 291 / 64.64%
- Paper teacher reference: 66.33%
- Dataset: official Oxford Flowers 102 `train + val` for training and `test` for evaluation
- DeiT-Ti Vanilla reference: 50.06%

The official dataset is downloaded and verified automatically. The common
student and KD-loss protocol is recorded in `methods/KD/README.md`.

## Timing run

```bash
python methods/KD/flowers102/train.py --student deit_ti --timing-run --batch-size 128 --num-workers 4
```

## Full run

```bash
python methods/KD/flowers102/train.py --student deit_ti --student-epochs 300 --batch-size 128 --num-workers 4 --output-dir /app/output
```

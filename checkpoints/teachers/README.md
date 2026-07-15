# Fixed teacher checkpoints

These are the teacher weights selected before running any KD method. The same
checkpoint must be reused across every student and every compared KD method for
the corresponding dataset.

| Dataset | Selected file | Epoch | Top-1 | Paper teacher | Gap |
|---|---|---:|---:|---:|---:|
| CIFAR-100 | `cifar100/teacher_resnet56_cifar100_best.pt` | 297 | 68.68% | 70.43% | -1.75pp |
| Flowers-102 | `flowers102/teacher_resnet56_flowers_best.pt` | 291 | 64.64% | 66.33% | -1.69pp |
| Chaoyang | `chaoyang/teacher_resnet56_chaoyang_latest.pt` | 300 | 81.53% | 77.20% | +4.33pp |

Chaoyang intentionally uses `latest`, not `best`. Its best checkpoint reached
83.08%, while the epoch-300 latest checkpoint reached 81.53% and was selected
as the closer match to the 77.20% paper teacher reference.

Run the integrity and strict-loading check before launching an experiment:

```bash
python teacher_checkpoints.py --dataset all
```

The original datasets and ZIP archives are not included. In particular, the
Chaoyang dataset must remain mounted separately at `/app/data/chaoyang`.

# IBAM teacher training on H200

This repository contains only the reusable ResNet56 teacher-training workflow
for the IBAM KD experiments on the KAU H200 runner.

Repository:

```text
https://github.com/bapedragon/IBAM_teacher_KD_H200.git
```

## Current status

| Dataset | Teacher | Status | Current Top-1 | Paper teacher Top-1 | Gap |
|---|---|---|---:|---:|---:|
| CIFAR-100 | ResNet56 | Complete | 68.68% | 70.43% | -1.75pp |
| Flowers | ResNet56 | Complete | 64.64% | 66.33% | -1.69pp |
| Chaoyang | ResNet56 | Run complete; artifact verification pending | 83.08% | 77.20% | +5.88pp |

For downstream KD experiments, use the best checkpoint from each dataset and
keep that teacher fixed across every compared KD method.

## Repository files

```text
train_teacher_cifar100.py
train_teacher_flowers.py
train_teacher_chaoyang.py
requirements.txt
.gitignore
README.md
```

- `train_teacher_cifar100.py`: downloads/verifies CIFAR-100 and trains the
  CIFAR-100 ResNet56 teacher.
- `train_teacher_flowers.py`: downloads/verifies Oxford Flowers 102 and trains
  the Flowers ResNet56 teacher.
- `train_teacher_chaoyang.py`: validates the mounted official Chaoyang dataset
  and trains the Chaoyang ResNet56 teacher.

Legacy LG student training, LG checkpoint evaluation, the downloaded LG weight,
and GitHub-token artifact upload experiments have been removed. H200 artifacts
are collected only through `/app/output`.

## Environment

```text
Image: pytorch/pytorch:latest
Language: Python
GPU allocation: 7 (one whole H200 GPU)
```

Install dependencies when running outside the provided H200 image:

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are already included in the H200 PyTorch image.
Flowers additionally requires `scipy`; the Flowers script installs it at startup
when it is missing.

## Shared teacher protocol

The paper explicitly identifies the datasets, ResNet56 teacher, scratch
training, 224 x 224 input, PyTorch, and Top-1 evaluation. It does not fully
specify the teacher optimization recipe, so all three scripts use the same
scaffold choices:

- Optimizer: SGD
- Initial learning rate: `0.1`
- Momentum: `0.9`
- Weight decay: `5e-4`
- Warm-up: 5 epochs for a 300-epoch run
- LR schedule: cosine decay
- Batch size: `128`
- Image resolution: `224 x 224`
- Seed: `42`
- AMP: enabled on CUDA
- Checkpoint criterion: highest test Top-1
- Final artifacts: best checkpoint, latest checkpoint, and `summary.json`

The H200 runner collects only files under `/app/output`. Timing runs intentionally
omit that path so their temporary checkpoints disappear with the Pod.

## CIFAR-100 teacher

The dataset is downloaded automatically under `./data` with verified mirror
fallback.

Timing test:

```bash
python train_teacher_cifar100.py --teacher-epochs 2 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_cifar100_timing_2ep
```

Full run:

```bash
python train_teacher_cifar100.py --teacher-epochs 300 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_cifar100_300ep --output-dir /app/output
```

Collected files:

```text
/app/output/teacher_resnet56_cifar100_300ep/teacher_resnet56_best.pt
/app/output/teacher_resnet56_cifar100_300ep/teacher_resnet56_latest.pt
/app/output/teacher_resnet56_cifar100_300ep/summary.json
```

## Flowers teacher

The official Oxford Flowers 102 images, labels, and splits are downloaded and
verified automatically. Training uses official `train + val` and evaluation
uses the official `test` split.

Timing test:

```bash
python train_teacher_flowers.py --teacher-epochs 2 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_flowers_timing_2ep
```

Full run:

```bash
python train_teacher_flowers.py --teacher-epochs 300 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_flowers_300ep --output-dir /app/output
```

Collected files:

```text
/app/output/teacher_resnet56_flowers_300ep/teacher_resnet56_flowers_best.pt
/app/output/teacher_resnet56_flowers_300ep/teacher_resnet56_flowers_latest.pt
/app/output/teacher_resnet56_flowers_300ep/summary.json
```

## Chaoyang teacher

The official dataset is mounted read-only at:

```text
/app/data/chaoyang/
```

The script searches up to three nested levels for the actual dataset root and
then requires this structure:

```text
train/       4,021 images
test/        2,139 images
train.json
test.json
README.md
```

It verifies the official class counts for normal, serrated, adenocarcinoma, and
adenoma before training. The original dataset must not be committed to this
repository because its license prohibits redistribution.

Timing test:

```bash
python train_teacher_chaoyang.py --data-dir /app/data/chaoyang --teacher-epochs 2 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_chaoyang_timing_2ep
```

Full run:

```bash
python train_teacher_chaoyang.py --data-dir /app/data/chaoyang --teacher-epochs 300 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_chaoyang_300ep --output-dir /app/output
```

Collected files:

```text
/app/output/teacher_resnet56_chaoyang_300ep/teacher_resnet56_chaoyang_best.pt
/app/output/teacher_resnet56_chaoyang_300ep/teacher_resnet56_chaoyang_latest.pt
/app/output/teacher_resnet56_chaoyang_300ep/summary.json
```

## Important log lines

Each script prints the CUDA/GPU environment, resolved data and output paths,
dataset sizes, model parameter count, per-epoch loss/accuracy/time, best Top-1,
estimated 300-epoch time, and final checkpoint paths.

A successful run ends with:

```text
[FINAL_RESULT] teacher_best_top1=...
[TIMING] teacher_avg_epoch=... estimated_300_teacher=...
[FINAL_RESULT] best_checkpoint=...
[DONE] ... completed successfully; resources may be released.
```

Any dataset, runtime, or external termination problem is printed with a
`[FATAL]` marker and traceback before the process exits.

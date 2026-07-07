# IBAM LG-style CIFAR-100 H200 full training

This repository runs the LG-style baseline used for the IBAM/iBKD experiment
check on H200:

- Dataset: CIFAR-100
- Teacher: ResNet56, trained from scratch
- Student: DeiT-Ti, trained from scratch
- Method: LG-style feature distillation
- Target paper reference: **77.38% Top-1** for CIFAR-100 / DeiT-Ti / LG

This code is meant to run reliably on the KAU H200 GitHub-Issue runner. It is
not an exact official LG reproduction because the paper draft does not fully
specify every LG implementation detail such as the feature matching layer, loss
weight, exact augmentation recipe, teacher recipe, seed, and checkpoint rule.

## Current status

The smoke test has already been confirmed on H200 with:

- CUDA available
- H200 MIG detected
- CIFAR-100 download through verified mirror fallback
- ResNet56 teacher training
- DeiT-Ti student training
- LG-style feature MSE loss
- final result and checkpoint logs printed

The full run should now be submitted without `--smoke`.

## Files

- `train_lg_cifar100.py`: full training script
- `requirements.txt`: declared Python dependencies
- `.gitignore`: excludes local datasets, outputs, caches, and macOS metadata

Runtime-created paths:

- `./data`: CIFAR-100 download/extraction directory
- `./outputs/lg_cifar100_deit_tiny_full`: checkpoints and `summary.json`

## Full H200 run

The shortest full-run command is:

```bash
python train_lg_cifar100.py
```

This defaults to:

- Teacher epochs: 300
- Student epochs: 300
- Batch size: 128
- DataLoader workers: 0
- Image size: 224
- Student optimizer: AdamW
- Student learning rate: 5e-4
- Student weight decay: 0.05
- Student schedule: 20-epoch warm-up followed by cosine decay
- Feature loss weight: 1.0
- Seed: 42

For the H200 Issue form, the explicit equivalent command is:

```bash
python train_lg_cifar100.py --teacher-epochs 300 --student-epochs 300 --batch-size 128 --num-workers 0
```

`--num-workers 0` is intentional. Earlier H200 smoke runs showed that
multi-worker DataLoader execution can fail because the container has limited
shared memory (`/dev/shm`). Using 0 workers is slower on CPU-side loading, but
it is the safest setting for this runner.

## Smoke test command

Smoke mode is retained only for diagnostics:

```bash
python train_lg_cifar100.py --smoke --teacher-epochs 1 --student-epochs 1 --batch-size 64 --num-workers 0
```

Smoke accuracy is not meaningful. It only checks that the full pipeline runs.

## Training protocol

### Matched to the paper draft

- CIFAR-100
- ResNet56 teacher trained from scratch
- DeiT-Ti student
- Student training for 300 epochs
- AdamW for student optimization
- Initial student learning rate `5e-4`
- Student weight decay `0.05`
- 20-epoch warm-up followed by cosine decay
- Batch size `128`
- Image resolution `224 x 224`
- PyTorch implementation
- Top-1 accuracy reporting

### Set in this repository because the draft does not fully specify it

- Teacher training recipe: 300 epochs, SGD, LR 0.1, momentum 0.9, weight decay
  5e-4, 5-epoch warm-up, cosine decay
- LG feature matching: ResNet56 `stage3` feature to DeiT-Ti patch tokens
- Teacher feature pooling to the DeiT `14 x 14` patch grid
- 1x1 convolution projection from 64 to 192 channels
- Per-token LayerNorm before feature MSE
- Feature loss weight `1.0`
- Final loss: classification cross-entropy + feature MSE
- Label smoothing `0.1`
- RandomResizedCrop and RandomHorizontalFlip for training augmentation
- Seed `42`
- DeiT-Ti pretrained weights are not used
- Best checkpoint is selected by the highest measured CIFAR-100 test Top-1

## Dataset handling

The script downloads CIFAR-100 automatically under `./data`.

Because the official Toronto host can reset connections inside the H200 runner,
the script tries verified sources in this order:

1. A pinned Hugging Face mirror of the original `cifar-100-python.tar.gz`
2. An SJTU mirror
3. The official Toronto URL used by torchvision

Every downloaded archive must match the official CIFAR-100 MD5:

```text
eb9058c3a382ffc7106e4002c42a8d85
```

So using a mirror changes only the download host, not the dataset contents.

## Logs to check after completion

The important lines in the H200 Issue report are:

```text
[TEACHER] ...
[STUDENT] ...
[FINAL_RESULT] student_best_top1=... teacher_best_top1=... reference_lg_top1=77.38%
[FINAL_RESULT] student_gap_to_reference=... teacher_gap_to_reference=...
[DONE] Training completed successfully; resources may be released.
```

For comparison:

- Paper teacher reference on CIFAR-100: **70.43%**
- Paper LG reference for CIFAR-100 / DeiT-Ti: **77.38%**

## Recommended H200 Issue values

- **Title:** `[Request]: 박철현 LG CIFAR-100 H200 300-epoch run`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_lg_cifar100.py --teacher-epochs 300 --student-epochs 300 --batch-size 128 --num-workers 0`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `7`
- **추가 필요 모듈:** 없음. `timm==1.0.27`이 없으면 스크립트 시작 시 자동 설치함.


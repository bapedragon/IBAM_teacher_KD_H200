# Ours paper/source audit

This audit prevents an implementation choice from being reported as if it
were specified by the working paper.

## Confirmed and implemented

| Item | Evidence | Implementation |
|---|---|---|
| Total objective | Working-paper Eq. (4) | `CE + beta(e) * [lambda*L_fuse + (1-lambda)*L_align]` |
| Loss balance | Working paper: `lambda=0.5` | `--fusion-ratio 0.5` |
| Alignment | Working paper Eq. (1), supplied source | all 12 blocks, learned convex weights, `1x1` projection |
| Grid resizing | Working paper and supplied source | bilinear resize to the target stage resolution |
| Enhancement | Working paper Eq. (2), supplied source | channel attention plus deformable spatial attention |
| Deformable kernel | Working-paper implementation section | `5x5` |
| Fusion | Working paper Eq. (3), supplied source | convolutional `1x1` Q/K/V, four heads in supplied source |
| Teacher | Working paper and supplied source | frozen during student training |
| Inference | Working paper | teacher and Ours module removed; student head only |
| Source identity | Supplied file | SHA-256 `8649078970b93d750a956994611b65cdec0c24f907d35d86f29d635e8a3b8624` |

## Confirmed behavior but missing exact experiment values

| Item | What is known | What is not available | Current handling |
|---|---|---|---|
| Adaptive beta | working paper refers to ALG; ALG switches from guidance to supervised-only according to feature-distance evolution | exact statistic, threshold, window, patience, per-dataset stop | fully logged `alg_proxy`, or explicit `manual_stop` |
| Active guidance strength | public source-compatible CIFAR config uses feature weight `2.5` | working paper does not state the active numeric beta | `--beta-on 2.5`, recorded as a config choice |
| Teacher input size | supplied source uses a configurable teacher size; public CIFAR config uses 32 | no supplied Flowers/Chaoyang Ours configs | source-compatible 32 plus runtime teacher audit; full run blocked on mismatch |

## Experiment settings not fixed by the Ours section

These are reproducible repository choices and are saved in every checkpoint
and summary: dataset-specific epoch count, augmentation, normalization, label
smoothing `0.1`, seed `42`, best-checkpoint selection, and AMP. The draft's
single 300-epoch statement is intentionally not used for Flowers/Chaoyang.

## Public CIFAR config cross-check

The public `lkhl/tiny-transformers` DeiT-CIFAR feature-guidance config at
commit `d2165f74049c906b0afc9f957491960fb3c0cc8b` confirms AdamW `5e-4`, minimum
LR `5e-6`, weight decay `0.05`, 20-epoch warm-up, cosine decay, batch 128,
feature weight `2.5`, teacher input 32, and drop-path `0.1`. Its framework
defaults also use strong timm augmentation and label smoothing `0.0`.

Only the values corroborated by the supplied Ours integration and the current
comparison design are applied here: optimizer/schedule, minimum LR, teacher
input 32, and active guidance strength 2.5. Drop-path/strong augmentation and
label smoothing are not silently changed for Ours alone because the working
paper does not identify those as Ours-specific settings and Table 2 requires
shared augmentation. The repository's already-recorded common protocol is
kept (`drop_path=0`, common crop/flip, label smoothing `0.1`). If the team
decides to adopt the public strong-augmentation recipe, the vanilla and every
KD method must be rerun under the same recipe for a controlled table.

## Full-run gate

1. Run the dataset's `--timing-run` command.
2. Confirm `[TEACHER_RUNTIME_AUDIT]` is within 5 percentage points of the
   checkpoint record.
3. Confirm feature shapes, finite losses, beta configuration, and epoch time.
4. Only then run with `--accept-alg-proxy`, or use `manual_stop` when an exact
   stop epoch/config is supplied by the experiment owner.

This code can reproduce the supplied module and paper-confirmed loss structure.
It must not be described as an exact official ALG-schedule reproduction until
the missing ALG experiment config is obtained.

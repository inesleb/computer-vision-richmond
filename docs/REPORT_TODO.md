# Emoji vendor classification – to do for the report

This file lists experiments still to run and findings already made, with the reasoning
behind each one. The goal is that the report can say what each design choice bought,
not only what the final model is.


## 1. Where we are

Results so far (accuracy):

| Model                                     | Validation    | Kaggle test | Notebook                                                    |
|-------------------------------------------|---------------|-------------|-------------------------------------------------------------|
| LeNet (sigmoid, average pooling, lr 0.01) | ~20%          | –           | `lenet_implementation.ipynb` (deleted, in git history)      |
| Custom CNN from scratch, 72 px, lr 5e-4   | 75.6%         | 74.95%      | `main-cnn.ipynb`                                            |
| Same CNN, lr 1e-3 (local CPU run)         | 82.4%         | –           | (not saved)                                                 |
| EfficientNetV2-B0 + 5-fold CV, 128 px     | 90.6% ± 1.2%  | 91.4%       | `main-finetuning.ipynb`                                     |
| Run A: EfficientNetV2-B0, 224 px          | 94.9% ± 0.5%  | –           | `main_run_a.ipynb` (22 min per fold on a T4)                |
| Run A + TTA                               | 95.2% ± 0.7%  | –           | `main_run_a.ipynb`                                          |
| Run A + TTA + bias correction             | 95.35% (held-out folds) | **96.05%** | `main_run_a.ipynb`                                |
| Run B alone: ConvNeXt-Tiny, 224 px, one model (fold 1), 10 epochs, batch 32, TTA | not recorded | **96.74%** | `main.ipynb` + `submission_run_b_only.csv` (finding 2n) |
| Average of A and B (TTA, no bias correction) | not recorded | **97.00%** | `main.ipynb`, "Ensemble of runs" (finding 2n) |

> **Note:** the 128 px row was trained with the 409 duplicate files (finding 2j), so its CV
> score is slightly optimistic. Run A is the first run on the deduplicated data.

> **Note:** the two CNN runs differ in learning rate (and hardware), and each is a single run,
> so the 82.4% vs 75.6% gap is not a clean comparison. Worth re-running both with the same
> setup if the report discusses the learning rate.


## 2. Findings already made (to write up)

### a) The dataset is heterogeneous

Image sizes range from 64×64 to 240×240, some non-square (72×68, 66×72...), and PNG
modes are mixed (RGBA, palette "P", RGB, LA, L). Class counts are unbalanced: apple 1924,
google 1877, samsung 1790, facebook 1667, whatsapp 1644, messenger 580, mozilla 397.

### a') Preprocessing: making every image the same format and size

A CNN needs every image in a batch to have the same shape and number of channels, so
each image (train and test alike) goes through three steps in `load_image`, the same
in all notebooks:

1. **Composite onto white, convert to RGB.** Every mode is converted to RGBA, then
   pasted onto an opaque white background, so transparent pixels become white and every
   image ends up with exactly 3 channels. White rather than black because emojis are
   drawn to be displayed on light backgrounds, and black would merge with their dark
   outlines (alternatives are listed in experiment 13).
2. **Pad to a white square.** The image is centered on a white square whose side is its
   longest edge (72×68 → 72×72). Resizing a non-square image directly would stretch it,
   and an emoji's proportions are part of its design.
3. **Resize to `IMG_SIZE`** with LANCZOS resampling (72 px for the custom CNN, 128 px for
   the fine-tuned model).

Resizing affects the vendors differently, because each vendor comes from a different
source size (at 128 px):

| Source size | Vendors                      | Effect at 128 px                                     |
|-------------|------------------------------|------------------------------------------------------|
| 240 px      | Facebook, Messenger, Mozilla | shrunk by ~half: fine details (outlines, gradients) lost |
| 160 px      | WhatsApp                     | slightly shrunk                                      |
| 72 px       | Apple, Google, Samsung       | enlarged ×1.8: no new detail, only smoother pixels   |

Two consequences for the report:

- **The emoji always fills the canvas**, so the model cannot read the original size from
  how much of the image is used. This removes the most direct form of the metadata
  shortcut (finding 2e).
- **Resolution is a trade-off that depends on the vendor.** A higher `IMG_SIZE` keeps more
  of the 160/240 px details (Facebook vs WhatsApp, the biggest confusion of the fine-tuned
  model), but cannot help the 72 px vendors (Apple vs Google vs Samsung), which only
  better features can separate. This motivates experiment 2.

### b) LeNet did not learn at all

Training accuracy stayed at ~18% and loss at ~1.85 (random guessing among 7 classes
gives ~14% and a loss of ln(7) = 1.95): the network was predicting the most common
vendor. Causes: learning rate 0.01 too high for Adam, and stacked sigmoid activations
(vanishing gradients: the sigmoid's derivative is at most 0.25).

### c) Batch normalization made validation accuracy unstable (diagnosed with an experiment)

After each epoch, 1,000 TRAINING images were scored in evaluation mode:

| Epoch | Train (training mode) | Same train images (eval mode) | Validation |
|------:|----------------------:|------------------------------:|-----------:|
|     1 |                   27% |                           22% |        23% |
|     2 |                   37% |                           31% |        32% |
|     3 |                   45% |                           31% |        32% |
|     4 |                   52% |                           29% |        29% |
|     5 |                   57% |                           40% |        38% |

Images the model trained on scored as badly as unseen ones in evaluation mode, so the
gap was not overfitting. In evaluation mode, batch norm uses running averages of past
batch statistics, which lag behind while the weights change fast. This explains why the
CNN's validation accuracy only jumped once the learning rate had been reduced (24% at
epoch 16 → 78% at epoch 22 in the local run), and why it still swung between 72% and
76% in the Colab run. `main-finetuning.ipynb` fixes this by keeping the pretrained network's batch
norm in evaluation mode (`base(x, training=False)`).

### d) Which vendors get confused (custom CNN)

- Apple / Facebook / WhatsApp form a confusion triangle (similar glossy, shaded styles).
- In the Colab run, many images were wrongly predicted as Samsung (52 WhatsApp, 43 Apple).
- Mozilla, the rarest class, has among the lowest recall (65–71%).
- Facebook vs Messenger was NOT a major confusion, contrary to what we expected.

Compare with the confusion matrix of the new model once it has run.

### e) Image metadata almost gives away the vendor (shortcut / data leak)

Original size + PNG mode vs vendor, training set:

| Original size / mode  | Vendor                                  |
|-----------------------|-----------------------------------------|
| 160 px                | WhatsApp (almost all)                   |
| 240 px                | Facebook, Messenger or Mozilla          |
| 72 px, palette ("P")  | mostly Samsung (1574 of 1695)           |
| 72 px, RGBA           | mostly Apple or Google                  |

We decided NOT to use this in the main model: the competition asks to classify "based
solely on visual characteristics", and resizing all images to the same size removes the
direct signal (finding 2a').

But resizing does not remove it entirely. Traces of the source survive preprocessing:

- **Blur level:** a 72 px image enlarged to 128 px is visibly softer than a 240 px image
  shrunk to 128 px. A network can learn "soft edges → Apple/Google/Samsung", which is
  the original size in disguise.
- **Color palette:** palette PNGs (mode "P", mostly Samsung) contain at most 256 distinct
  colors. Resampling blends some of them, but the reduced palette remains detectable.

So "purely visual" is only approximately true for any model trained on these images,
including ours. See experiments 14 and 15 for measuring this.

**Higher resolution makes this trace stronger.** At 224 px, a 72 px source is enlarged 3.1×
(1.8× at 128 px), so its softness is even more visible, while 240 px sources stay sharp.
The model still only sees pixels, but the gain from 128 → 224 px may partly come from a
clearer "source size" signal and not only from finer vendor details. Worth one sentence
in the report, and the "errors by source group" table of `main.ipynb` (128 px vs 224 px)
shows it: if the 224 px model gains mostly on cross-group confusions (e.g. Facebook 240 px
vs WhatsApp 160 px) and little within a group (Apple vs Google, both 72 px RGBA), the
resolution helps through both channels and the two can't be separated.

### f) The 128 px model had converged: "train longer" was not a lever

`IMPROVEMENT_PLAN.md` (5.2) proposed 40 epochs because fold 5 "was still improving at its last
epoch". The training logs of `main-finetuning.ipynb` say otherwise:

- Training accuracy passed 99% after 9 fine-tuning epochs (of 25) in every fold.
- The final training loss (0.474) is at the minimum that label smoothing allows: with
  ε = 0.1 and 7 classes, the targets are 0.914 / 0.014, whose entropy is 0.446. The
  model fits the training set almost perfectly, so more epochs give it nothing to learn.
- Validation accuracy moved by less than 0.4% over the last 5 epochs in every fold. Fold 5's
  "improvement" at its last epoch is +0.3%, i.e. 6 images: noise.

So `main.ipynb` shortens fine-tuning from 25 to 20 epochs (the cosine schedule then decays
faster, so it reaches the plateau sooner), which pays for part of the 3× cost of 224 px. The
overfitting has to be addressed with information (resolution, better features) or
regularization (finding h), not with longer training.

### g) Picking the best validation epoch inflates the CV score

`main-finetuning.ipynb` kept the epoch with the best validation accuracy (`ModelCheckpoint`
+ `EarlyStopping` on `val_accuracy`). Best vs last epoch, per fold: +0.00, +0.20, +0.39,
+0.29, +0.00 points, so **+0.18% on average**. Choosing the maximum of ~25 noisy scores on the
same data used to report the score is a (small) selection bias: these late epochs differ
by noise, so the selected epoch isn't better on the test set. `main.ipynb` trains a fixed
schedule and keeps the final weights; the validation fold is then only used for scoring,
so CV is an unbiased estimate of the test accuracy. Good example for the report of why
the validation set must not be used for decisions if it is also used to report the score.

### h) `training=False` also switched off EfficientNet's stochastic depth

`main-finetuning.ipynb` called the pretrained base with `base(x, training=False)` to keep
batch normalization in evaluation mode (finding c). But `training=False` applies to every
layer of the base, including its 15 **stochastic depth** layers: during training,
EfficientNetV2-B0 drops the residual branch of a block for random images, with a rate
growing with depth from 0.02 to 0.19. It was pretrained with this regularization, and
the flag silently removed it during fine-tuning, for a model that overfits strongly
(99.9% training vs ~91% validation accuracy).

`main.ipynb` instead sets `trainable = False` on each batch normalization layer: Keras then
uses the stored statistics in training too, while dropout stays active. Checked on the
real network: with this setting, two training-mode calls on the same images differ
(dropout active), the moving statistics don't change, and with dropout disabled the
training-mode output equals the inference output (same batch normalization behavior as
before). Not measured separately (bundled in run A): its effect is part of the run A gain.

### i) How the improvements were chosen (run budget)

GPU time on Colab is the constraint (the 128 px run took 73 min on a T4; 224 px costs ~3×),
and one-fold ablations are too noisy to measure changes below ~1% (section 3). So instead
of testing each change of `IMPROVEMENT_PLAN.md` on fold 0, `main.ipynb`:

- **Bundles the changes with a strong reason and little risk into run A**: 224 px (targets
  the largest confusion, Facebook ↔ WhatsApp, see 2a'), stochastic depth restored (h),
  warmup, fixed schedule (g). Cost: the gain can't be split between them. The confusion
  pairs give an indirect split: resolution should mostly reduce Facebook ↔ WhatsApp and
  Mozilla ↔ Messenger (240/160 px sources), not Apple ↔ Google ↔ Samsung (72 px sources).
- **Measures the free changes on the out-of-fold predictions**, without extra training: TTA
  (every view's predictions are saved, accuracy with and without TTA is printed) and the
  per-vendor bias correction (fitted on 4 folds, scored on the 5th).
- **Makes the second run useful whatever its score**: ConvNeXt-Tiny (run B) has a different
  design (layer normalization, bigger), so even if it is not better alone, it should make
  different mistakes and improve the ensemble. It also tests whether better features help
  the 72 px vendors, which resolution cannot.
- **Rejects** "train longer" (f), class weights (Kaggle scores overall accuracy, 5.8), and
  colour augmentation (colours identify the vendor). MixUp/CutMix and pseudo-labelling are
  kept for a next round, depending on the results.

### j) Duplicate training files

The Drive copy of `data/train` had 409 files named `xxxx (1).png` (re-upload copies),
counted twice by `main-finetuning.ipynb` (10,288 images instead of 9,879). A copy and its
original could fall into different folds, so a model was sometimes validated on an image it
had trained on. `main.ipynb` skips them and checks that each labelled image appears once.

### k) Per-vendor bias correction

The 128 px model predicted Samsung whenever it hesitated between 72 px vendors (recall 99%,
precision 85%: 266 Apple/Google images predicted Samsung). Adding one constant per vendor
to the log-probabilities moves the decision boundaries, and the 7 constants can be fitted
to maximize out-of-fold accuracy. Because the constants are chosen on the same data, the
notebook also fits them on 4 folds and scores them on the 5th, and only keeps them if
that held-out accuracy improves. Report both numbers: a correction that only helps on the
data it was fitted on is overfitting.


### l) Run A results: resolution also helped the 72 px vendors (prediction wrong)

Out-of-fold errors, 128 px (`main-finetuning.ipynb`, ~966 errors on 10,288 images) vs
224 px run A (476 errors on 9,879 images):

| Confusion (both directions) | 128 px | 224 px (run A) | Source sizes        |
|-----------------------------|-------:|---------------:|---------------------|
| Facebook ↔ WhatsApp         |    299 |            182 | 240 vs 160 px       |
| Apple/Google ↔ Samsung      |    279 |             60 | all 72 px           |
| Apple ↔ Google              |     88 |             74 | all 72 px (RGBA)    |
| Messenger ↔ Mozilla         |     54 |             21 | all 240 px          |

Samsung is no longer a "sink": precision 0.85 → 0.97, recall still 0.99. We predicted
(2a', IMPROVEMENT_PLAN 2c) that resolution could not help the 72 px vendors, since
enlarging a 72 px image adds no information. The input argument holds, but the network
argument doesn't: a pretrained CNN looks for patterns of a fixed size in pixels, and
shrinks the image by 32 before classifying (a 4×4 grid of features at 128 px, 7×7 at
224 px). Enlarging the image makes a 1-pixel outline or shading step of the source span
several pixels, a scale its ImageNet filters were trained on. So the same information
becomes easier to use. This is a hypothesis: run A also changed stochastic depth and the
schedule (finding 2i), so the gain can't be attributed to resolution alone.

Not the palette leak: Apple images stored as palette PNGs, the same format as Samsung's
(99 images, "72 px P"), are not mistaken for Samsung (16 errors in that whole group,
accuracy 99.1%). If the model were reading "palette → Samsung", these would be its errors.

Other run A numbers worth reporting:

- **Top-2 accuracy 98.5%** vs top-1 95.2%: for most errors the right vendor is the second
  choice. This is a good case for ensembling.
- **WhatsApp is now the weakest vendor** (recall 0.889): 126 WhatsApp images predicted as
  Facebook. All 123 errors on 160 px images are "metadata-avoidable", since only WhatsApp
  uses 160 px.
- **Metadata-avoidable errors: 233 of 476 (49%).** If a metadata model removed exactly
  these, it would reach about 97.6%. This is the estimate to test with experiment M1.
- **Training was stable.** Validation accuracy plateaus from fine-tuning epoch ~12, with
  last-epoch vs best-epoch differences of about 0.1–0.2%. Each fold took 22 min on a T4
  (47 s per epoch), so 5 folds took about 2 h instead of the estimated 3.


### m) Run B: one fold of ConvNeXt, because of the GPU budget

ConvNeXt-Tiny takes about 6 min per epoch on a free Colab T4 (8× EfficientNet-B0), and only
fits with batches of 32 (64 ran out of GPU memory). 5 folds × 15 epochs would need about 9 h
of GPU, and the free Colab quota ran out during the first fold. So run B is trained on
fold 1 only (the other 4 folds for training), with 10 fine-tuning epochs instead of 15:
after 4 epochs it was already at 99.2% training and 95.7% validation accuracy (run A at
the same point: 92.4%). Cost: about 1 h 10 of GPU.

Why it is worth that cost, and how the risk is limited:

- **What "A + B" is:** the average, image by image, of the 7 vendor probabilities predicted
  by run A (its 5 fold models) and by run B. No retraining, no merging of the networks: the
  same principle as averaging the 5 fold models. It helps only if the two models make
  different mistakes. Run A's top-2 accuracy (98.5% vs 95.2% top-1) and the very different
  architectures suggest they will: typically +0.5 to +1%, i.e. 10–20% of the remaining errors.
- **Paired comparison:** run A's fold 1 model and run B are scored on exactly the same
  1,976 validation images, and so is their average. Differences on the same images are
  less noisy than two scores on different images (one fold alone: about ±0.5%).
- **Decision rule, set before the run:** if ConvNeXt does not beat run A and the average
  does not beat run A on fold 1, ConvNeXt is dropped (1 h 10 spent). Otherwise the
  average is submitted to Kaggle, whose test set (9,879 images) gives the final answer.
- Training on 100% of the images was considered and rejected: no validation score at all,
  and the comparison would only rest on Kaggle submissions.
- Lost by using one fold: no 5-model average for run B, and no bias correction for the
  ensemble (its held-out check needs at least 2 folds).

### n) Kaggle results: runs A and B

| Submission                                              | Models averaged          | Kaggle  | Test error |
|---------------------------------------------------------|--------------------------|--------:|-----------:|
| EfficientNetV2-B0, 128 px (`main-finetuning.ipynb`)     | 5 EfficientNet           | 91.40%  | 8.6%       |
| Run A: EfficientNetV2-B0, 224 px, TTA, bias correction  | 5 EfficientNet           | 96.05%  | 3.95%      |
| Run B alone: ConvNeXt-Tiny, 224 px, TTA                 | 1 ConvNeXt               | 96.74%  | 3.26%      |
| Average of A and B: TTA, no bias correction             | 5 EfficientNet + 1 ConvNeXt | **97.00%** | 3.00%   |

How each file was made:

- **A + B** is the `submission.csv` written by the run B session. With `ENSEMBLE_RUNS = None`,
  the "Ensemble of runs" section averages every run saved in `emoji_outputs`, so it combined
  run A's 5 fold models (their average, TTA) with ConvNeXt's model, with equal weights for
  A and B. There is no bias correction, because its held-out check needs at least 2 folds.
  Lesson: that default produced an ensemble when we expected a single-run submission, so the
  file to submit must be named explicitly.
- **B alone** was rebuilt afterwards from ConvNeXt's saved test predictions
  (`convnext_tiny_224/fold0_test_views.npy`), outside Colab: the average of the 5 TTA views,
  for the same 9,879 test images in the same order. TTA was kept because it helped run A; it
  changes 0.62% of the test predictions. Nothing from run A is used.
- The fold 1 validation table of the run B session (A vs B vs average on the same 1,976
  images) was not kept, so the decision rule of finding 2m was applied on Kaggle instead.

What the three scores show:

- **One ConvNeXt model beats an ensemble of 5 EfficientNet models: 96.74% vs 96.05%**, with a
  handicap: trained on 80% of the images, 10 epochs, no bias correction. A single
  EfficientNet model would score lower than run A's 5-model average (about 0.5 to 1 point,
  judging from the 128 px run), so model for model, the gap between the two networks is
  larger than 0.7 points. The better network (bigger, more recent design, layer
  normalization) brings more than averaging several copies of the weaker one.
- **The average adds only +0.26 points over ConvNeXt alone** (96.74% → 97.00%, test error
  3.26% → 3.00%, about 25 more images right). We predicted +0.5 to +1% over the best single
  run: too optimistic. Most of the +0.95 points over run A comes from ConvNeXt being better
  (+0.69), not from the averaging (+0.26). Averaging with a clearly weaker model dilutes
  the stronger one. The two still make partly different mistakes, so the average remains
  the best submission. A weight in favour of ConvNeXt might do better, but it can't be
  tuned without the fold 1 predictions, and guessing it on the leaderboard would spend
  submissions and overfit the test set.
- **Kaggle is above CV, again, by about the same amount** (+0.7 to +0.8 points for both the
  128 px model and run A). Expected: each out-of-fold prediction comes from ONE fold model,
  while the test predictions average the 5 fold models (an ensemble). So CV measures a
  single model and slightly underestimates the submitted ensemble. Worth one sentence in
  the report, as it shows the CV estimate is consistent (and conservative), not noisy.
- **Overall progress on Kaggle, image only throughout:** 74.95% (CNN from scratch) → 91.4%
  (transfer learning, 128 px) → 96.05% (224 px, stochastic depth, fixed schedule, TTA, bias
  correction) → 96.74% (ConvNeXt-Tiny) → 97.00% (average of both). The test error went
  from 25% to 3%. No metadata was given to any model.
- **Best image-only submission: the average of A and B (97.00%).** Candidate for one of the
  2 final Kaggle picks.
- **Next: the metadata model (M1).** In run A, 49% of the out-of-fold errors were
  "metadata-avoidable" (finding 2l). With the image-only ensemble at 3.0% test error, a
  metadata model can at most remove part of that 3%. Measure M1 the same way as run A
  (5-fold CV, then Kaggle), so the gain from metadata is compared like for like. Don't
  compare a CV number with a Kaggle number: Kaggle runs about 0.7 points higher (see above).


## 3. Experiments to run

How to compare fairly:

- Change ONE thing at a time and compare against the `main-finetuning.ipynb` result.
- For quick comparisons, use `FOLDS_TO_RUN = [0]` (one fold, ~20% validation). Expect
  about ±1% of noise between runs: for differences of that size, run all 5 folds and
  compare the cross-validation mean ± std.
- Record for each run: cross-validation accuracy ± std (or fold 0 accuracy), training
  time, the confusion matrix if it changes, and the Kaggle score if submitted.
- Kaggle allows 5 submissions per day, and we pick 2 final submissions.

### Priority 1 – main result

- [x] **1.** Run `main-finetuning.ipynb` with `FOLDS_TO_RUN = [0]`, compare with 75.6%. If better, run all 5
  folds and submit. This becomes the reference for all experiments below.
  *Done:* 90.6% ± 1.2% cross-validation, 91.4% on Kaggle.

### Priority 2 – choices made in `main-finetuning.ipynb` without testing them (ablations)

- [ ] **2. Image size:** `IMG_SIZE = 72` vs 128 (current) vs 160.
  *Why:* source images are 72, 160 or 240 px; shrinking them to 72 blurs the fine
  details (outlines, shading) that distinguish vendors. Bigger is slower and uses
  more memory, so report accuracy AND training time.
- [ ] **3. Horizontal flip:** add `layers.RandomFlip("horizontal")` back to the augmentation.
  *Why:* we removed it on the hypothesis that each vendor lights its emojis from a
  consistent side (e.g. highlights on the upper left), which a flip would destroy.
  If accuracy drops with the flip, the hypothesis is supported.
- [ ] **4. Augmentation fill:** `fill_mode="constant"` with white (current) vs `"reflect"` (the
  Keras default, used by the custom CNN).
  *Why:* `"reflect"` mirrors the emoji's edges into the empty area, creating shapes that
  never occur in real images.
- [ ] **5. Augmentation on vs off** (remove the three `Random*` layers).
  *Why:* quantifies how much augmentation reduces overfitting (compare the gap between
  training and validation accuracy, not just validation accuracy).
- [ ] **6. Batch norm frozen vs not:** frozen (current: `trainable = False` on each batch norm
  layer, see finding 2h) vs fully trainable batch norm.
  *Why:* directly tests finding 2c. Expect less stable validation curves without it.
- [ ] **7. Pretrained vs from scratch:** `weights="imagenet"` (current) vs `weights=None`.
  *Why:* quantifies the gain from transfer learning, with the same architecture.
- [ ] **8. Label smoothing:** 0.1 (current) vs 0.
  *Why:* label smoothing discourages overconfident predictions and reduces overfitting.
- [ ] **9. Ensemble vs single model:** submit the predictions of fold 0 alone
  (`efficientnetv2b0_224/fold0_test_views.npy` in the outputs folder, view 0) vs the 5-fold average.
  *Why:* measures what the ensemble adds on the real test set. Costs one submission.

### Priority 3 – possible further improvements

- [ ] **10. Class weights** (`class_weight` in `model.fit`) to boost Mozilla and Messenger.
  *Why:* they are the rarest classes. Check whether their recall improves without
  lowering overall accuracy (which is what Kaggle scores).
- [x] **11. Test-time augmentation:** average predictions over a few shifted/zoomed copies of
  each test image. Usually a small gain at no training cost.
  *Done (run A):* +0.25% cross-validation accuracy (better in 4 folds of 5, equal in 1),
  log loss lower in all 5 folds.
- [x] **12. Bigger pretrained networks:** EfficientNetV2B1/B2 or ConvNeXtTiny instead of
  EfficientNetV2B0. Slower, possibly more accurate.
  *Done:* ConvNeXt-Tiny (224 px, one model): 96.74% on Kaggle alone vs 96.05% for run A's
  5 EfficientNet models, 97.00% averaged with run A (finding 2n). Its confusion pairs were
  not kept, so which vendor pairs it improves is unknown.
- [ ] **13. Background:** we composite transparent images onto white. Alternatives: a black
  background, or keeping the alpha channel. Probably minor; only if time allows.

### Predictions to check when runs A and B are back

Written before the runs, so the report can say which expectations held:

- Run A beats 90.6% CV; the gain is concentrated on Facebook ↔ WhatsApp (299 errors at
  128 px) and Messenger ↔ Mozilla (54), with little change on Apple/Google ↔ Samsung (279).
  **Result: half right.** 94.9% CV (95.2% with TTA). Facebook ↔ WhatsApp 299 → 182 and
  Messenger ↔ Mozilla 54 → 21 as expected, but Apple/Google ↔ Samsung also fell, 279 → 60,
  the largest drop of all. See finding 2l.
- TTA: +0.2 to +0.5%. Bias correction: small or rejected by the held-out check.
  **Result: as expected.** TTA +0.25% (better in 4 folds of 5, equal in 1; log loss lower in
  all 5). Bias correction +0.17% on held-out folds (95.18% → 95.35%), so it was kept.
- Ensemble A + B beats the best single run by ~0.5–1%.
  **Result: too optimistic.** The best single run is ConvNeXt (96.74% on Kaggle), and the
  average adds +0.26 points (97.00%). The +0.95 points over run A come mostly from ConvNeXt
  itself (+0.69), not from averaging (finding 2n).
- Overall, still around 0.93–0.95: most remaining errors are within 72 px vendors, which
  neither resolution nor TTA can fix.
  **Result: wrong about where the errors are.** Run A alone reaches 95.2%, and the largest
  remaining confusion is Facebook ↔ WhatsApp (182 of 476 errors, 38%), not the 72 px
  vendors (Apple ↔ Google 74, Apple/Google ↔ Samsung 60).

### Side analysis for the report (NOT for the submitted model)

- [ ] **14. Metadata shortcut** (finding 2e). Train three models and compare:
  - metadata only: a small classifier on original size + PNG mode
  - image only: the main model
  - image + metadata: the main model with metadata as extra inputs

  *Why:* shows how strong the shortcut is, and how much "free" accuracy it would give.
  A good discussion point on dataset bias: a model could score well without
  learning anything about visual style.
- [ ] **15. Residual leak after preprocessing** (finding 2e). Two cheap checks:
  - Errors by source group (size + PNG mode) on the out-of-fold predictions. If the
    model almost never confuses two vendors from different source groups, it is likely
    using traces of the source.
  - A probe: train a small classifier to predict the *original size* from the
    preprocessed 128 px images. Near-100% accuracy would show the size is still
    readable after resizing.

  *Why:* tells the report how far "based solely on visual characteristics" really
  holds, and whether the leak is still in our main model.

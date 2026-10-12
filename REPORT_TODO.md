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
- [ ] **6. Batch norm frozen vs not:** `base(x, training=False)` (current) vs `base(x)`.
  *Why:* directly tests finding 2c. Expect less stable validation curves without it.
- [ ] **7. Pretrained vs from scratch:** `weights="imagenet"` (current) vs `weights=None`.
  *Why:* quantifies the gain from transfer learning, with the same architecture.
- [ ] **8. Label smoothing:** 0.1 (current) vs 0.
  *Why:* label smoothing discourages overconfident predictions and reduces overfitting.
- [ ] **9. Ensemble vs single model:** submit the predictions of fold 0 alone
  (`test_probs_fold0.npy` in the outputs folder) vs the 5-fold average.
  *Why:* measures what the ensemble adds on the real test set. Costs one submission.

### Priority 3 – possible further improvements

- [ ] **10. Class weights** (`class_weight` in `model.fit`) to boost Mozilla and Messenger.
  *Why:* they are the rarest classes. Check whether their recall improves without
  lowering overall accuracy (which is what Kaggle scores).
- [ ] **11. Test-time augmentation:** average predictions over a few shifted/zoomed copies of
  each test image. Usually a small gain at no training cost.
- [ ] **12. Bigger pretrained networks:** EfficientNetV2B1/B2 or ConvNeXtTiny instead of
  EfficientNetV2B0. Slower, possibly more accurate.
- [ ] **13. Background:** we composite transparent images onto white. Alternatives: a black
  background, or keeping the alpha channel. Probably minor; only if time allows.

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

# Task 1 — Watermark localization

**Public score: 0.6893** per-image mIoU (first submission 0.5307; all-zeros baseline 0.2553).
Best submission: `c1_ensY_ss9_s62w4.pt`. Detailed results and commands (French): [NOTES.md](NOTES.md);
literature and past-edition review: [VEILLE.md](VEILLE.md).

## The task

- 510 training images with ground-truth masks, 1,020 test images, all 512×512.
- Three invisible watermarking methods, 340 test images each: **PixelSeal** (Meta), **TrustMark**
  (Adobe), **StegaStamp**.
- After watermarking, each image was edited: outpainting, object removal, object addition or object
  transplantation.
- Output: one binary 128×128 mask per image, 1 where the image still carries the watermark.
- Metric: mIoU over the two classes {0, 1}, averaged per image. Missing a small region therefore costs
  as much as missing a large one.
- Key fact: **each test image comes with its method and its hidden message**, and the starter kit ships
  the official decoders. The decoder becomes a measuring instrument: "is this message still readable
  here?"

## How we reasoned

### 1. Measure before modelling

Decoding the known message under controlled conditions on the training set
([real_heats.py](real_heats.py) `--diag-only`) showed two regimes:

| Regime | Share | Watermarked area | Bit accuracy, whole image | Bit accuracy, ground-truth box cropped |
|---|---|---|---|---|
| F: fully watermarked, then holes restored | 50 % | ≈ 90 % | 0.97-0.98 | — |
| R: one rectangle watermarked at its own scale | 50 % | ≈ 4 % (median) | 0.52-0.55 | 0.64-0.73 |

A rectangle watermark can only be read by **cropping** it, not in place. Thresholding the whole-image
bit accuracy at 0.75 separates F from R with 12 errors out of 510. That sort alone (F → all ones,
R → all zeros) scores 0.45.

### 2. Lock the validation first

[metric.py](metric.py) reproduces the kit scorer exactly ([test_metric.py](test_metric.py)). Every
decision went through **nested 5-fold cross-validation**, stratified by method × regime
([gated_eval.py](gated_eval.py)): thresholds and models are chosen on four folds, then scored on the
fifth. A candidate replaced the best submission only if its paired gain was ≥ +0.01 with a positive
lower bound of the 95 % bootstrap interval (2,000 resamples). The rule is written down before the
result is read.

The validation turned out faithful: 0.5583 locally, 0.5572 on the public split.

### 3. Progression

| Step | What changed | Public |
|---|---|---:|
| Decoder maps + rectangle search | Per-position agreement maps, plus a search over 13,500 cropped and decoded boxes ([box_search.py](box_search.py)) | 0.5088 (nested CV) |
| Synthetic "holes" head | Mini U-Net trained on 1,440 F images generated with the official embedders, no real image ([seg/gen_f.py](seg/gen_f.py), [seg/train_f.py](seg/train_f.py)) | 0.5307 |
| Holes head on real + synthetic | Features from the decoders (agreement, logits, occlusion margins) ([segf/](segf/)) | 0.5572 |
| Pixel segmenter on StegaStamp | U-Net "pixseg" on synthetic data, routed to StegaStamp only | 0.5845 |
| Pretrained encoder | ConvNeXt-Tiny encoder (timm, ImageNet weights) | 0.6137 |
| Seed ensembles | 4 × 4,000-step models, then 6,000-step models on StegaStamp and TrustMark F, weighted seeds | 0.6377 → 0.6893 |

The pixel-segmentation code is archived as is in
[../docs/archives-worktrees-20261122/c1-pixseg/](../docs/archives-worktrees-20261122/c1-pixseg/).

### 4. What the rule rejected

Each line was measured, and each failed the rule. All are logged in
[../docs/experiences.md](../docs/experiences.md).

- PixelSeal "detection" channel: not trained in the released checkpoint (`lambda_det: 0.0`), pixel AUC ≈ 0.5.
- Re-embedding correlation map (`r̂ = embed(x′, m) − x′`): AUC 0.51 on PixelSeal, against 0.76 for agreement maps.
- Template matching per box, TruFor as an extra input channel: no gain.
- Expected-mIoU decision per image (Nowozin, 2014) instead of a threshold: −0.0001.
- 8,000 training steps (−0.004), 512 crops, ConvNeXt-Small, IoU loss, a rectangle fitted on probabilities.
- Dense localizer on TrustMark decoder features: +0.009 in routed CV, then 0.6414 on the public split
  when applied to all TrustMark images (best at the time: 0.6886).

## What the top teams did (0.79-0.81)

Read after the event, in the repositories of Murmell, Fable 4.2, LesRooters and WhatThePhoques:

- **Large synthetic sets that match the real pipeline.** Fable 4.2 found that the real watermarked
  content had been enlarged by 1.00-1.07 and then cropped after embedding. Reproducing this step
  in the generator moved one case from 0.07 to 0.83 mIoU. Our synthetic data skipped it, and our
  segmenters learned mostly from the 510 real images.
- **Decoder features as the encoder**, initialized from the official extractors (Murmell, Fable 4.2).
- **Geometric post-processing**: rectangles snapped by decoding against the known message; objects
  snapped onto Segment Anything masks.

## Sources

- Starter kit and data: `SprintML/Hackathon_Watermark_Localisation_Task_Paris` (Hugging Face).
- PixelSeal and Video Seal code: [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal)
  (`third_party/videoseal`), the PixelSeal card and the organizers' watermarking talk.
- TrustMark: [adobe/trustmark](https://github.com/adobe/trustmark). StegaStamp: Tancik et al., CVPR 2020.
- WAM, *Watermark Anything with Localized Messages* (Sander et al., 2024): framing watermark reading as segmentation.
- TruFor (Guillaro et al., CVPR 2023): forensic map tested as an input channel.
- Nowozin, *Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case*, CVPR 2014.
- ConvNeXt (Liu et al., CVPR 2022) through [timm](https://github.com/huggingface/pytorch-image-models).
- Past CISPA editions: [sprintml/HackathonParis](https://github.com/sprintml/HackathonParis),
  HackathonVienna, hackathonStockholm, and the "ship gate" of the July 2026 final
  (florianzainzinger1/cispa-hackathon-grand-finale-2026).

# SLZ — European AI Championship, Paris 2026

Team SLZ at the **CISPA × Meta hackathon of the European AI Championship** (Paris, 21-22 November 2026,
CISPA and Fraunhofer AISEC, tasks designed by SprintML): three AI-security tasks in 24 hours on the
JURECA supercomputer (A100 nodes, Slurm), with one overall ranking.

**Slides:** [slz_deck_20261122.pptx](docs/deck-20261122/projects/slz_deck_20261122/exports/slz_deck_20261122_052937.pptx)

| Task | Metric | Public score | Best solution | Method, code, validation |
|---|---|---:|---|---|
| 1. Watermark localization | per-image mIoU | **0.6893** | `c1_ensY_ss9_s62w4.pt` | [c1/](c1/), [c1/NOTES.md](c1/NOTES.md) |
| 2. Model equality | 0.1·MCC family + 0.9·MCC model | **0.20** | baseline | [c2/](c2/), [c2/PLAN.md](c2/PLAN.md) |
| 3. Unlearning | forget × retain | **0.0986** | `t3_P5.pt` | [c3/](c3/) |

Last ranking recorded: 31st of 46 teams, overall score 0.320 (Sunday 06:07, before the last Task 1 gain).
Public scores cover about 30 % of the test data; the final ranking uses the hidden 70 %.
Receipts and checksums: [docs/CLOTURE-20261122.md](docs/CLOTURE-20261122.md).

## One rule for all three tasks

Decide on data with known truth, never on the public score. Every candidate was compared with the
current best by a paired gain with a 2,000-sample bootstrap, against a rule written down before reading
the result: submit only if the gain is at least +0.01 and the lower bound of the 95 % interval is
positive. Every experiment, failures included, has a line in [docs/experiences.md](docs/experiences.md)
(hypothesis, change, GPU cost, local score, public score).

## The idea of each task

**Task 1: watermark localization.** 510 labelled training images, 1,020 test images (512×512), three
watermarking methods (PixelSeal, TrustMark, StegaStamp) and four edits (outpainting, object removal,
addition, transplantation). Each test image comes with its method and its hidden message, so the
official decoders can tell where that message still is.

- Two regimes: fully watermarked images with restored holes, or a single rectangle watermarked at its
  own scale. Decoding the known message on the whole image separates them (12 errors out of 510).
- Message-agreement maps and a search over cropped, decoded rectangles: 0.5088 in nested CV, against
  0.25 for an empty mask.
- A hole-segmentation head trained on 1,440 synthetic images made with the official embedders, then a
  pixel segmenter (pretrained ConvNeXt-Tiny) with seed ensembles and per-method routing.
- 0.5307 → 0.6893 public. The 5-fold nested CV predicted the public score to within 0.001
  (0.5583 local, 0.5572 public).

**Task 2: model equality.** 33 black-box LLM endpoints, 100 answers per endpoint for the whole weekend.
The three base models (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B, 11 endpoints each) were identified by
the likelihood of their answers under local replicas, with 528/528 pair agreement. Separating true
variants (fp8, LoRA) from disguises (system prompt, persona, temperature) failed every pre-registered
test on the reference endpoints, so the scored budget was kept and the submission stayed at baseline.

**Task 3: unlearning.** A ViT-B/16 (timm, 100 classes) must forget 3,500 diffusion-generated images
across 45 classes, without retain data, labels or a gold model. The submitted model only edits the
classification head: the 45 affected class rows are pushed down, the rest of the network is untouched.
Sweeps over the push strength and the head temperature gave the best score at ×1.75, T = 0.9.

## What the top teams did differently

Read after the event in the repositories of [Murmell](https://github.com/Wassbdr/murmell) (1st on the public leaderboard),
[Fable 4.2](https://github.com/AnMaLeNo/Fable-4.2), [LesRooters](https://github.com/Zarrock77/LesRooters)
and [WhatThePhoques](https://github.com/sinonchum/WhatThePhoques).

- **Task 2 (1.00 for at least six teams).** Disguised endpoints are the base model: they are positively
  identified (prompt-leak probes, identical openings under a watermark, exact score test at T = 1), then
  the plain-looking ones are tested against a local fp8 replica and public LoRA adapters. We had the
  replicas, but looked for one statistic that separates everything.
- **Task 3 (0.43-0.48).** A model that never saw a forget image errs toward a sibling class, so the
  per-image target is the original model's runner-up distribution, learned on the last blocks with a KL
  anchor elsewhere. Our head-only push lowered the right class but did not move the mass to that sibling.
- **Task 1 (0.79-0.81).** Dense detectors trained on large synthetic sets that reproduce the 1.00-1.07
  resize and crop applied to the real images, plus rectangle snapping by decoding and Segment Anything
  snapping for objects.

The overall score rescales each task between the worst and best team. With our other scores fixed,
Task 2 at 1.00 would have moved our overall score from about 0.32 to 0.60, more than any Task 1 gain.

## In this repository

```text
c1/        Task 1: metric, decoder maps, rectangle search, segmentation heads, probes
c2/        Task 2: query plan, prompts, offline analysis, copy of the kit scorer
c3/        Task 3: proxy evaluation, drift, soups, anchoring and distillation
common/    JURECA paths, submission script, Slurm job template, autoresearch loop
docs/      task sheet, experiment log, audits, closing report, slides
```

The Task 1 pixel-segmentation code (`pixseg`, dense pilot) is archived as is in
[docs/archives-worktrees-20261122/c1-pixseg/](docs/archives-worktrees-20261122/c1-pixseg/). Datasets,
weights and submission files are not in Git; they stayed on JURECA. Data:
`SprintML/Hackathon_Watermark_Localisation_Task_Paris`, `SprintML/model_equality`,
`SprintML/unlearning` on Hugging Face (access granted by the organizers).

Tests and lint, as in CI:

```bash
ruff check --select E9,F63,F7,F82 .
pytest -q c1/test_metric.py c2/test_analyze.py c3/test_eval.py
```

Working notes inside the task folders and `docs/` are in French.

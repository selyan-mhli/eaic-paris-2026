# Task 2 — Model equality testing

**Public score: 0.20**, a random baseline submitted early to test the format. No method passed its
pre-registered test, so the scored endpoints were never used to submit a real partition. Query plan
(French): [PLAN.md](PLAN.md).

## The task

- 33 anonymous chat endpoints (`ep-01` to `ep-33`) and 9 reference endpoints (`ref-01` to `ref-09`),
  whose answers are published (`kit/practice_answers.csv`).
- Labels: a family (same base model) and a model (exactly the same weights: same checkpoint, same
  quantization, same LoRA). A watermark, a system prompt, a persona or a decoding change does **not**
  change the model.
- Metric: 0.1 × MCC on "same family" pairs + 0.9 × MCC on "same model" pairs. The model term only
  counts pairs inside a family, so the real task is to split each family correctly.
- The scarce resource: **100 answers per endpoint for the whole weekend**, never refilled.

## How we reasoned

### 1. Learn the API on the reference endpoints

Bench runs on the references, logged in `docs/task2-bench-20261121/`:

- Only `max_tokens` and `messages` (user and assistant turns) are honoured. `temperature`, `seed`,
  `top_p`, `logprobs` and `n` are silently ignored. The `system` role is refused. Every answer is sampled.
- About 15 s per request whatever the length, with 4 parallel requests per team.

So no logprobs: everything has to come from sampled text.

### 2. Black-box statistics first

[analyze.py](analyze.py) and [features.py](features.py) compare answer distributions per pair: n-gram
kernels, character Hamming distance and the unbiased MMD² of Gao et al. On the references, with the
truth known, they reach an MCC between 0.00 and 0.52. Disguises (persona, temperature) move the answers
more than fp8 or a light LoRA, so distribution distance alone groups the wrong endpoints.

### 3. White box: identify the base models

- Scoring the collected answers by their likelihood under candidate open checkpoints identified the
  three families of the scored endpoints: **Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct and
  Llama-3.2-1B-Instruct**, 11 endpoints each, with 528/528 pair agreement.
- The references are **SmolLM2-360M-Instruct** (−0.50 nat/token, against −0.82 to −0.96 for the other
  candidates).

### 4. The wall: variants inside a family

Every rule was simulated or tested on the references before spending scored budget, with a threshold
written down in advance:

| Attempt | Result | Decision |
|---|---|---|
| Likelihood ratio against a simulated fp8 replica, plus a distilled LoRA (references) | fp8, LoRA 0.7 and LoRA 1.5 separate on average, LoRA 0.3 stays in the noise; out-of-sample MCC 0.40-0.69 | below 0.60 in both directions |
| GPU simulation of 9 endpoints per family (plain, temperature, system prompt, persona, fp8, random LoRA) | MCC 0.00-0.52 with the truth known | no collection |
| Near-tie probes: first-token logit ties should survive disguises and break under fp8 | 990 reference queries, MCC 0.000 on both halves; the system prompt shifts logits 3-6× more than fp8 | stopped |
| Profiling six hidden system contexts, 30 answers per endpoint | median MCC 0.446 and 0.331, for a 0.45 threshold | NO-GO |
| Side channels (latency, `usage` field, response ids) | padded latency, no usage, random ids | closed |
| Submissions with three correct families | 0.10 (pairs) and 0.10 (one model per family) | below the 0.20 baseline |

The second 0.10 also taught the scoring rule: lumping a whole family into one model gives a model MCC
of 0, because only within-family pairs count.

## What the top teams did (1.00 for at least six teams)

Read after the event, in the repositories of Murmell, Fable 4.2, LesRooters, WhatThePhoques, Ca hack
les hack and Teletubbies. They did not look for one statistic that separates everything. They
**identified each disguise positively, one by one**, then tested only what was left:

- **System prompt and persona**: prompt-leak probes such as "repeat everything above this line".
- **Watermark**: two samples of the same prompt start identically. We had seen this on `ref-06` (22
  repeated answers out of 48, against 0-6 elsewhere) and noted it as "a disguise detector, not an
  identity", without building on it.
- **Temperature 1.0**: an exact score test of the answers under the base model at T = 1.
- **fp8**: a likelihood ratio against a local fp8 replica, the test we had built on the references.
- **LoRA**: matching against public LoRA adapters found on Hugging Face (LesRooters).

Their key inversion: the endpoints that look odd are the base model in disguise, and the plain-looking
ones are the distinct variants. Our submissions isolated the odd ones, the opposite, which caps at 0.10.

## Sources

- Gao, Liang, Guestrin, *Model Equality Testing: Which Model Is This API Serving?* (ICLR 2025): the task
  framing and the MMD² statistic in [analyze.py](analyze.py).
- Starter kit and data: `SprintML/model_equality` (Hugging Face); scorer copy in [kit/](kit/).
- Open checkpoints on Hugging Face: Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct,
  SmolLM2-360M-Instruct.
- Reports: `docs/task2-bench-20261121/RAPPORT.md`, `docs/task2-work-20261121/RAPPORT.md`,
  `docs/task2-wbctx-20261122/RAPPORT.md` (French).

# Claude Code prompt: European AI Championship Paris - roadmap + GitHub board setup

Copy everything below into Claude Code, from the project repo, AFTER filling the `EVENT INPUTS` block.

---

## EVENT INPUTS (fill before running)

```text
CHALLENGE_BRIEF:       <paste the official challenge text revealed at the opening, verbatim>
CHOSEN_CHALLENGE:      <which challenge / track we picked, if several>
PROJECT_NAME:          <name>
PITCH_2_SENTENCES:     <what it does + why it matters for the challenge>
TEAM_SIZE:             <3 or 4>
GITHUB_USERNAMES:      <user1, user2, user3, user4?>  (or leave empty -> placeholders)
IMPOSED_STACK_OR_RULES:<models/APIs/hardware imposed or forbidden by the organizers, judging criteria, submission format>
AVAILABLE_COMPUTE:     <laptops only / cloud credits given / GPU given / API keys given>
```

If any input is empty, use placeholders, list what is missing at the top of the review document, and never invent it.

## Context

We are a team of `TEAM_SIZE` people at the **European AI Championship Paris** (regional stage of the European AI Championship led by CISPA Helmholtz Center for Information Security, co-organized by PoC Innovation, partners Meta and Epitech). Venue: Epitech, Le Kremlin-Bicetre. Theme: AI and cybersecurity. Challenges are revealed at the opening on Saturday.

Hard time constraints:

```text
Sat 15:00  hacking starts
Sun 12:00  hacking ends, demos in front of the jury
```

That is **21 hours of hacking**, overnight, with meals on site. Plan for humans who get tired: sleep slots, a hard feature freeze, and a demo that works even if someone is asleep.

You are setting up a parallel-safe roadmap and a GitHub Project board so that all team members can develop simultaneously with:

- zero blocking wherever possible;
- minimal merge conflicts;
- frozen shared contracts defined in the first 90 minutes;
- one owner per directory;
- a clean, reproducible demo that fits in the jury slot;
- a public, documented repository if the rules require it.

Do not invent the challenge, judging criteria, prize categories, model names, API keys, hardware, or jury expectations. Everything specific must come from `CHALLENGE_BRIEF` and `IMPOSED_STACK_OR_RULES`. If the brief is ambiguous, list the ambiguity and the question to ask a mentor.

## CRITICAL control rule - read first

Do **not** create, push, edit, delete, label, assign, or otherwise mutate anything on GitHub until I have reviewed and explicitly approved the plan.

### Step 1 - review only

Produce a single review document `docs/HACKATHON-REVIEW.md` containing:

1. challenge understanding: requirements, judging criteria quoted from the brief, unknowns, questions for mentors;
2. the proposed `docs/ROADMAP.md`;
3. the proposed repository structure with owners;
4. the frozen contract list and schemas;
5. the list of GitHub issues (target 25 to 40, not more), each with:
   - title;
   - short body;
   - 2 to 4 acceptance criteria;
   - owner or placeholder;
   - workstream label;
   - phase label;
   - priority;
   - dependencies;
   - files or directories it may modify;
   - mock, cached or real data;
6. the GitHub Project v2 configuration;
7. the hour-by-hour timeline including sleep slots;
8. the demo script minute by minute;
9. risks and fallbacks.

Then **STOP** and wait for my explicit approval. Keep the review readable in under 10 minutes: we have no time to read 2000 lines.

### Step 2 - only after approval

Only after I reply with explicit approval, create the repo files, labels, issues, assignments and the Project board through `gh`.

Read-only commands (`gh auth status`, repo inspection, issue listing, contributor listing) are allowed before approval. Nothing else.

## Roles

Use these placeholders until real usernames are confirmed:

- `AI_OWNER` - Person 1: models, prompts, agents or ML pipeline, evaluation, inference metrics.
- `PLATFORM_OWNER` - Person 2: backend, orchestration, tools and data integration, state, infra, startup script.
- `PRODUCT_OWNER` - Person 3: frontend, UX, demo controller, README, pitch, backup video.
- `SECURITY_OWNER` - Person 4, only if `TEAM_SIZE` is 4: threat model, attack and red-team scenarios, adversarial test set, safety and robustness checks, security section of the pitch. This role fits the CISPA cyber angle; adapt it to the challenge (for example dataset and evaluation owner if the challenge is pure ML).

If `TEAM_SIZE` is 3, fold the security and evaluation tasks into `AI_OWNER` (evaluation) and `PLATFORM_OWNER` (input validation, secrets, sandboxing).

## Design principles for a 21-hour build

- One core end-to-end flow that solves the challenge. Not three half features.
- The demo flow is defined in Phase 0 and everything serves it.
- Mock everything at the start: mock model outputs, mock API responses, mock event stream. Replace with real pieces progressively.
- Every external dependency (API, model server, dataset download, Wi-Fi) has a cached or local fallback. Venue Wi-Fi with 200 hackers is a risk.
- If the project involves an LLM agent, keep deterministic services and LLM agents clearly separated, validate every tool call against an allowlist, and never execute model output as code.
- If the project has a security angle (attack, defense, detection, privacy, robustness), the demo must show a concrete before/after: an attack or failure case, then the system catching or fixing it, with measured numbers only.
- Never fabricate metrics, benchmark gains, or user impact. Show only what we measured during the weekend.
- Do not display hidden model reasoning. Show auditable actions, tool calls, evidence and short rationales.

## Repository boundaries and ownership

Propose a structure derived from the chosen idea, following this pattern:

```text
/
├── ai/                 # AI_OWNER: models, prompts, agents, evaluation, metrics
├── backend/            # PLATFORM_OWNER: API, orchestrator, state, streaming
├── tools/              # PLATFORM_OWNER: external/internal tool adapters, caches
├── data/               # PLATFORM_OWNER (or SECURITY_OWNER for eval/attack sets)
├── security/           # SECURITY_OWNER if 4 people: threat model, attack scenarios, red-team tests
├── frontend/           # PRODUCT_OWNER
├── demo/               # PRODUCT_OWNER: demo script, reset script, backup video notes
├── docs/               # PRODUCT_OWNER, ROADMAP produced by Claude
├── contracts/          # FROZEN after Phase 0; one designated merger
├── scripts/<owner>/    # one subfolder per owner
├── .env.example        # PLATFORM_OWNER
├── docker-compose.yml  # PLATFORM_OWNER, optional
└── README.md           # PRODUCT_OWNER
```

### No-conflict rules

- one owner per directory, written in `CODEOWNERS`;
- no direct push to `main`, everything through short PRs;
- one branch per issue: `ai/<n>-<slug>`, `platform/<n>-<slug>`, `product/<n>-<slug>`, `security/<n>-<slug>`, `integration/<slug>`;
- `/contracts` frozen after Phase 0; any change = dedicated PR reviewed by the whole team;
- no global formatting or dependency upgrades outside the issue scope;
- scoped dependency files per directory rather than one shared requirements file;
- only PLATFORM_OWNER edits root config, `.env.example`, `docker-compose.yml`;
- only AI_OWNER edits system prompts and model config;
- only PRODUCT_OWNER edits design tokens and the top-level demo script;
- merges happen at planned integration checkpoints, not continuously.

### Merge order

1. scaffold + frozen contracts + mocks;
2. independent workstreams against mocks;
3. backend endpoints compatible with mocks;
4. AI pieces wired to mocked tools;
5. frontend wired to real backend events;
6. real data and model adapters;
7. security / evaluation scenarios plugged in;
8. offline and fallback path;
9. demo stabilization only.

## Frozen shared contracts

In Phase 0, create `/contracts` with one unambiguous source of truth (JSON Schema or Pydantic with generated TypeScript types). Derive the contracts from the chosen idea. At minimum:

- the main input object (what the user or system feeds in);
- the main output object (what the system produces for the demo);
- intermediate objects between AI and platform (for example `ToolRequest`, `ToolResult`, `AnalysisResult`, `Verdict`);
- a `Metric` object for measured numbers (latency, accuracy, detection rate, cost, tokens);
- a frontend event envelope:

```json
{
  "event_id": "...",
  "run_id": "...",
  "event_type": "...",
  "timestamp": "...",
  "sequence": 1,
  "payload": {}
}
```

- a complete mock stream `/contracts/mocks/demo_event_stream.jsonl` that replays the whole demo, so the frontend can be built before anything else works.

Every datum shown in the UI carries a source tag: `live`, `cached`, `seeded_demo`, `user_input`, `model_inference`.

## Phases and timeline (Saturday 15:00 to Sunday 12:00)

Adapt durations if the brief imposes intermediate checkpoints.

### Phase 0 - Unblock and freeze (Sat 15:00 to 16:30)

- read the brief, pick the idea, write the 2-sentence pitch and the demo flow;
- confirm usernames, create repo, scaffold, `CODEOWNERS`, branch policy;
- `.env.example` without secrets;
- hello-world for every critical dependency (model call, API call, frontend build);
- freeze contracts and mock stream;
- create the board.

Nothing else starts before contracts are frozen.

### Phase 1 - Parallel build (Sat 16:30 to Sun 01:00)

Everyone builds against mocks in their own directories. Checkpoint at Sat 20:00 (after dinner): 15-minute sync, each person shows their piece running on mocks.

### Phase 2 - Integration (Sun 01:00 to 06:00)

Wire pieces in merge order. First complete end-to-end run must happen before 04:00. Sleep in shifts: never more than one person asleep during integration, and never the PLATFORM_OWNER during the first end-to-end run.

### Phase 3 - Stabilization and submission (Sun 06:00 to 10:30)

- deterministic reset and rerun of the demo;
- fallbacks tested (cached data, precomputed outputs clearly labeled);
- measured metrics displayed;
- README, architecture diagram, submission material required by the brief;
- backup demo video recorded before 10:00.

### Buffer - Feature freeze (Sun 10:30 to 12:00)

No new features. Only bug fixes, copy, rehearsal (at least 2 full runs of the pitch with the timer), final submission check.

## Seed tasks

Generate concrete issues from these seeds, adapted to the idea. Each issue finishable in 1 to 3 hours, one owner, exact directories, 2 to 4 testable acceptance criteria.

### Shared / Phase 0

- Parse the brief: requirements, judging criteria, submission format, questions for mentors.
- Confirm usernames and role mapping.
- Repo scaffold, CODEOWNERS, branch and PR policy.
- Freeze contracts and generate types.
- Full mock event stream for the demo.
- `.env.example` and secrets handling.
- Hello-world for each critical dependency.
- Board creation.

### AI_OWNER

- Model access and health check (local or API, as allowed by the rules).
- Shared inference client with timeout, retry, structured-output validation.
- Core prompt(s) / agent(s) or ML pipeline against mocked tools.
- Evaluation set (at least 20 cases including edge and failure cases) and runner.
- Measured metrics collection.
- Optional only after green end-to-end: tuning, few-shot, fine-tuning.

### PLATFORM_OWNER

- Backend scaffold and health endpoint.
- Deterministic orchestrator / state machine for the demo flow.
- Event streaming (SSE or WebSocket) compatible with the mock stream.
- Tool adapters with allowlist, validation, cache and fallback.
- Seeded demo data, clearly labeled.
- Run reset/start endpoints.
- Offline / cached mode switch.
- One-command startup script.

### PRODUCT_OWNER

- Frontend scaffold playing the mock stream.
- Main demo view: input, live processing trace, result, metrics panel.
- Source badges and error/fallback states.
- Wire to the real backend stream.
- Architecture diagram, README with setup order.
- Pitch script and backup video.

### SECURITY_OWNER (if 4 people)

- Threat model of the system and of the problem space, one page.
- Attack or failure scenarios that the demo will show (before/after).
- Adversarial / red-team test set and runner.
- Input validation, prompt-injection and secret-leak checks on our own system.
- Security and limitations section for the pitch and README.

### Shared integration / demo

- First end-to-end run with one input.
- End-to-end run of the full demo scenario.
- Fallback drill: kill the network, kill the model server, check the demo survives.
- Reset and rerun 3 times in a row without errors.
- Measure and display real metrics.
- Pitch rehearsal with timer and speaker handoffs.
- Final submission checklist against the brief.

## GitHub Project v2 configuration

Fields:

- **Status**: Backlog, Ready, In Progress, In Review, Blocked, Done
- **Priority**: P0 demo blocking, P1 core, P2 nice to have, P3 optional
- **Phase**: Phase 0 Unblock, Phase 1 Build, Phase 2 Integration, Phase 3 Demo, Buffer
- **Workstream**: AI, Platform, Product, Security (if 4), Shared

Labels:

```text
workstream:ai
workstream:platform
workstream:product
workstream:security
workstream:shared
phase:0-unblock
phase:1-build
phase:2-integration
phase:3-demo
phase:buffer
priority:p0
priority:p1
priority:p2
priority:p3
contract-change
demo-blocker
fallback
security
eval
```

Views:

- Board by Status;
- Board by Assignee;
- Table by Phase;
- P0 demo blockers;
- Integration queue;
- Blocked / unassigned.

## Demo

Write the demo minute by minute for the jury slot (ask a mentor for the exact duration; assume 3 minutes demo + Q&A if unknown):

1. the problem, in one concrete scene;
2. live run of the core flow;
3. the moment that proves it works (attack caught, failure fixed, measurable result);
4. measured numbers;
5. what is real vs mocked/seeded, said honestly;
6. one closing sentence.

## Fallback hierarchy

- Level 0: full live demo.
- Level 1: external API or Wi-Fi down -> cached responses, real model calls if local.
- Level 2: model server unstable -> restart, smaller model, then precomputed outputs clearly labeled as fallback.
- Level 3: frontend broken -> backend trace in terminal + slides.
- Level 4: everything down -> backup video recorded Sunday morning.

## Definition of done

The demo is ready only if:

- [ ] the core flow runs end to end on the scenario;
- [ ] it addresses the challenge requirements from the brief, checked one by one;
- [ ] contracts were respected (no last-minute schema hacks);
- [ ] source tags are visible;
- [ ] measured metrics are shown, none invented;
- [ ] reset and rerun works;
- [ ] offline / cached fallback works;
- [ ] backup video exists;
- [ ] README explains setup and limitations;
- [ ] repo is public if required, no secrets committed;
- [ ] pitch rehearsed twice with timer.

## Explicit non-goals

- training a model from scratch;
- more than one core scenario;
- more than 5 agents if agents are used;
- pixel-perfect design before end-to-end works;
- production security certification;
- any feature started after Sun 10:30.

## `gh` CLI rules

Before any write after approval:

1. `gh auth status` and report the active account;
2. verify the target repo and its visibility;
3. verify assignee usernames;
4. verify permissions for issues and Projects;
5. report any mismatch and stop.

Then use `gh label create`, `gh issue create --assignee --label --body`, `gh project create`, `gh project field-create`, `gh project item-add`. Repo files go through branches and PRs, never a direct push to `main`.

## Final instruction

Produce `docs/HACKATHON-REVIEW.md` first, short and actionable, so that every team member can start in their own directory within 90 minutes of the challenge reveal. Then STOP and wait for explicit approval.

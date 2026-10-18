# Chat Safety Classifier Service
Authors: Jaiden Lee
Last Updated: 10/18/2026
Status: Draft

## Context And Problem
Real-time chat is central to many online games and social platforms, and a large
share of their users are minors. Chat volume on popular platforms far exceeds what
human moderators can review, so first-line moderation must be automated and must
run on every message before it is delivered.

Some harms are comparatively easy to catch automatically: profanity and overt
toxicity are largely lexical, and existing classifiers handle them well. This
project targets a harder and higher-stakes category: **the sharing of personal
information and attempts to move conversations to unmoderated channels** (e.g.,
exchanging phone numbers, addresses, school names, or messaging-app handles).
Off-platform contact removes users from the protections a platform provides,
which makes it a key child-safety risk.

This category is difficult for three reasons:

- **Context dependence.** Meaning often spans multiple messages. A request like
  "what's your discord?" followed by a bare username is only harmful as a pair;
  each message alone looks benign. Decisions must be made at the conversation
  level, not per message.
- **Adversarial obfuscation.** Users deliberately evade filters with spacing,
  character substitution, spelled-out numbers, or splitting information across
  messages, which defeats keyword and regex approaches.
- **Strict serving constraints.** Every message must be scored before delivery,
  within a latency budget small enough that chat still feels instant, at
  platform-scale throughput, and at a cost per message that is viable for a
  free product.

Large language models can judge conversational context well, but their per-message
latency and cost are too high to run on every chat message at scale. The goal of
this project is to build and characterize a **small, conversation-aware classifier
service** that closes most of that gap within an explicit latency, throughput,
quality, and cost budget.

## Goals and Non-Goals
## 2. Goals and Non-Goals

### Goals

**Product goals**

- **G1: Conversation-level detection of personal-information sharing.** Given a new
  message and the recent conversation it belongs to, decide whether the conversation
  now contains an exchange of personal or off-platform contact information. In scope:
  - Messaging and social media handles (usernames on other apps)
  - Phone numbers and email addresses
  - Physical location details (home address, school name)
  - Explicit requests to move the conversation to another platform
- **G2: Robustness to common evasion.** Detection should hold up against spacing,
  character substitution, spelled-out numbers, and information split across
  multiple messages, the cases where keyword and regex filters fail.
- **G3: Meet the serving budget.** Score every message before delivery within the
  latency, throughput, and cost targets defined in Section 3.

**Engineering and learning goals**

- **G4: Characterize the system, not just build it.** Measure how latency, throughput,
  and cost change with hardware (CPU vs. GPU), batch size, model compression, and
  context-handling strategy, and compare each measurement to the predictions in
  Section 7.
- **G5: Reproducibility.** Evaluation harness, load generator, configs, and raw
  results live in the repo so every number in this doc can be regenerated.
- **G6: Privacy-preserving by design.** The service handles personal information by
  definition, so it must not persist raw message text in logs or metrics. Only
  scores, labels, lengths, and timings are recorded.

### Non-Goals

### Goals

**Product goals**

-  **G1: Context-aware detection on every new message.** When a message is sent,
  score that message, using recent conversation context, for whether it shares or
  solicits personal or off-platform contact information. Output is a single binary
  label (plus a score) for the new message. Works in group and one-to-one chats:
  context is speaker-tagged with relative roles ([SELF], [U1], [U2], …) and
  selected by a rule-based context builder under a token budget (target 256, hard
  cap 512). One model call per message, independent of participant count.
- **G2: Robustness to common evasion.** Detection should hold up against spacing,
  character substitution, spelled-out numbers, and information split across
  multiple messages, the cases where keyword and regex filters fail.
- **G3: Meet the serving budget.** Score every message before delivery within the
  latency, throughput, and cost targets defined in Section 3.

**Engineering and learning goals**

- **G4: Characterize the system, not just build it.** Measure how latency, throughput,
  and cost change with hardware (CPU vs. GPU), batch size, model compression, and
  context-handling strategy, and compare each measurement to the predictions in
  Section 7.
- **G5: Reproducibility.** Evaluation harness, load generator, configs, and raw
  results live in the repo so every number in this doc can be regenerated.
- **G6: Privacy-preserving by design.** The service handles personal information by
  definition, so it must not persist raw message text in logs or metrics. Only
  scores, labels, lengths, and timings are recorded.

### Non-Goals

- **Other harm categories.** Harassment, hate speech, sexual content, scams, spam,
  and self-harm detection are out of scope. Generic toxicity is handled well by
  existing classifiers and is not the focus here.
- **Intent or grooming detection.** This project detects *information exchange*,
  not the intent behind it. Detecting predatory behavior requires long-horizon
  behavioral signals, expert-labeled data, and human review, all beyond a personal
  project.
- **Real user data.** No real chat logs, and in particular no data from minors.
  Training and evaluation use public datasets and synthetic conversations only.
- **Enforcement and UX.** The service returns scores and labels. What a platform
  does with them (block, mask, hold for review, warn), appeals, and user-facing
  behavior are out of scope.
- **LLMs in the request path.** Large models may be used offline (e.g., generating
  or labeling synthetic data) but never to score live messages.
- **Non-English text, images, voice, and profile fields** (display names, bios).
- **Production-grade infrastructure.** Multi-region deployment, autoscaling, and
  high availability are out of scope. The service runs on a single node; scaling
  beyond that is estimated in Section 4, not built.

## Requirements/Constraints
All latency targets are measured **server-side**: from request arrival at the service
to response sent, excluding network transit to and from the chat backend.

### Definitions

- **Latency** is measured server-side: request arrival to response sent, excluding
  network transit to/from the chat backend. Measured under a realistic mix of
  context lengths unless stated otherwise.
- **Recall** = fraction of contact-sharing messages that are blocked.
- **False-block rate (FPR)** = fraction of clean messages that are blocked.
  Unlike precision, recall and FPR don't depend on how common positives are,
  so they are used for requirements.
- **Implied precision** at an assumed real-world positive rate π:
  precision = recall·π / (recall·π + FPR·(1 − π)). Reported for context, not a gate.
- **Operating threshold** is chosen on the validation set as the threshold meeting
  Q1, then fixed for all test and serving measurements.

### Serving

| ID | Requirement | Target | Rationale | Verified in |
|---|---|---|---|---|
| R1 | Latency, typical | p50 ≤ 15 ms | Chat feels instant at roughly 100 ms end to end; moderation is one hop of several. Estimated cost on a T4 (~1–2 ms compute, a few ms launch overhead, ≤ 5 ms batching wait) leaves headroom. | P1, P3 |
| R2 | Latency, tail | p99 ≤ 50 ms at the R4 load | Tail latency is what users notice in busy chats. Includes batching wait. | P3 |
| R3 | Worst-case input | p99 ≤ 100 ms for 512-token contexts | Long contexts are rare, but under fail-closed (R7) a timeout becomes a blocked message. | P3 |
| R4 | Throughput, per node | ≥ 1,000 msg/s on one T4 while meeting R2 | Sets the node size for capacity planning. The number that matters is max throughput *within* the SLO, not absolute max. Feasibility checked in §4. | P3 |
| R5 | Fleet scale (assumption) | 30,000 msg/s peak | Assumes ~2M concurrent chatting users sending ~1 message/min. Used only for capacity and cost estimates (≈ R5 ÷ R4 nodes); not built. | §4 |
| R6 | Cost | ≤ $1 per million messages at peak utilization | Runs on every message of a free product, so must be cheap at scale. | §4, P3 |
| R7 | Failure behavior | Fail-closed: on error or if scoring exceeds 100 ms, the message is held (not delivered) and the sender sees a pending/retry state | A missed safety check is worse than a delayed message. Consequence: this service is on chat's critical path, so its reliability and tail latency directly bound chat's. | P1, P3 |

### Quality
All quality targets are evaluated at the single operating threshold.

| ID | Requirement | Target | Rationale | Verified in |
|---|---|---|---|---|
| Q1 | Overall detection | Recall ≥ 0.90 at FPR ≤ 0.1% (≤ 1 false block per 1,000 clean messages) | Misses are safety failures, so recall comes first; the FPR cap bounds how often innocent users are blocked. | P1, P2 |
| Q2 | Hard cases | Recall ≥ 0.75 on obfuscated and cross-message positives | These are the cases that justify a model over keyword filters. | P1, P2 |
| Q3 | Hard negatives | FPR ≤ 0.5% on look-alike clean messages (game scores, in-game friend requests, talking *about* other apps) | Look-alike negatives are what drive false blocks in practice. | P1, P2 |
| Q4 | Beats baseline | Recall ≥ 0.15 above a regex + PII-tagger baseline at the same FPR | If the model doesn't clearly beat rules, it isn't worth its cost. | P1 |
| Q5 | Compression guardrail | Optimized models lose ≤ 1 point of recall vs. the full model at the same FPR | Speedups can't come out of safety. | P2 |

### Operational

| ID | Requirement | Target | Rationale |
|---|---|---|---|
| O1 | Privacy | No raw message text in logs, traces, or metrics | The service handles personal information by definition (G6). |
| O2 | Memory | Model + runtime fit on one T4 (16 GB) and on an 8 GB CPU node | Keeps the CPU option and cheap GPUs viable. |

## Estimates
## Design Decisions
## Evaluation Plan
### 6.1 Datasets

| Set | Purpose | Composition |
|---|---|---|
| Train | Fit the model | Synthetic multi-party conversations plus public PII examples. **Class-balanced** (oversampled positives) so the model sees enough positive examples to learn from. |
| Validation | Choose the operating threshold and compare model candidates | Enriched positive rate (~10–20%), with hard positives and hard negatives represented. |
| Test | Final, one-time measurement of every Q requirement | Same construction as validation, generated separately. Never used for tuning. |
| Hard positives | Q2 | Positives tagged at generation time: **obfuscated** (spacing, substitution, spelled-out numbers, look-alike characters), **cross-message** (request and answer in different messages, possibly different speakers), **implicit** (no telltale keyword). |
| Hard negatives | Q3 | Clean messages that resemble positives: game scores and other long numbers, in-game friend requests, mentioning other apps as a topic, referring to another player's in-game username. |

**Leakage control.** Splits are made by conversation *and* by generation template or
seed, so near-duplicate synthetic conversations can't appear in both train and test.
Without this, synthetic data inflates every metric.

**Why the test set is enriched, not realistic.** At a realistic positive rate
(≤ 1%), measuring recall precisely would need hundreds of thousands of examples.
Instead, the test set contains plenty of both classes, and recall and FPR, which
don't depend on the positive rate, are measured directly. Precision at the real
rate is then *computed* (§6.3).

**Sizing.** Targets: ≥ 2,000 positives and ≥ 50,000 clean messages in test.
- Recall at 0.90 on 2,000 positives has a 95% interval of about ±1.3 points.
- FPR at 0.1% on 50,000 negatives means about 50 false blocks, an interval of
  roughly 0.07–0.13%.

That's tight enough to tell whether Q1 is met.

### 6.2 Metric choices

| Metric | Role here | Why |
|---|---|---|
| **Recall** (= 1 − FNR) | **Primary target** (Q1, Q2) | A missed contact exchange is a safety failure. This is the cost we most want to minimize. |
| **False-block rate (FPR)** | **Constraint on recall** (Q1, Q3) | Fraction of innocent messages that get held. Maps directly to user harm under fail-closed (R7), and doesn't depend on the positive rate (§6.3). |
| **Recall at FPR ≤ 0.1%** | **The requirement and operating point** | Describes the exact threshold that would be deployed. |
| Precision | **Reported, computed** at assumed rates π = 0.1% and 1% | The most informative number for rare positives ("what fraction of blocks are wrong?"), but only meaningful at the real positive rate, which is unknown. Computed from recall and FPR rather than measured on synthetic data. |
| Recall at precision | Not used as the gate | Equivalent to recall at FPR only if π is pinned. Using it would make pass/fail depend on a guessed number. Preferred when labeled production traffic exists, or when each flag costs reviewer time. |
| PR AUC | **Model selection** (comparing candidates on the same validation set) | Threshold-free summary that behaves well with rare positives. Says how good the whole curve is, not how good the deployed point is. |
| ROC AUC | Reported only | Threshold-free, but driven by FPR over a huge pool of negatives, so it can look excellent while precision at the real rate is poor. Not used for decisions. |
| F1 | Not used | Weights false positives and false negatives equally; here a miss is worse than a false block. It also doesn't correspond to a deployable threshold. (F2, which weights recall higher, would be the closer choice if a single number were needed.) |
| Accuracy | Not used | With rare positives, predicting "clean" for everything scores ≥ 99%. |
| Calibration (ECE) | Not required now | Matters if scores are consumed downstream (e.g., risk-based routing). Here only the thresholded decision is used. Revisit if a "hold for review" tier is added. |

### 6.3 Why FPR instead of precision

Precision = correct blocks ÷ all blocks. Correct blocks scale with the number of
*harmful* messages; wrong blocks scale with the number of *clean* messages. So the
same model's precision changes with how rare harmful messages are:

| Real positive rate π | Recall | FPR | Precision of the same model |
|---|---|---|---|
| 1% | 0.90 | 0.1% | ≈ 0.90 |
| 0.1% | 0.90 | 0.1% | ≈ 0.47 |

Recall only looks at harmful messages and FPR only looks at clean ones, so neither
moves with π. Since π is unknown and the synthetic test set doesn't reflect it, the
requirement uses recall and FPR. Precision is still reported at assumed rates using

    precision(π) = recall·π / (recall·π + FPR·(1 − π))

so a small FPR can't hide the fact that, at very low π, a large share of blocks
may be wrong. That is the base-rate effect.

### 6.4 Threshold protocol

1. Train candidates; compare them by PR AUC on validation.
2. For the chosen model, set the threshold on validation: the threshold with the
   **highest recall subject to FPR ≤ 0.1%**.
3. Freeze the threshold. Measure Q1–Q3 **once** on test, with 95% confidence
   intervals (bootstrap).
4. No changes to model or threshold after viewing test results. If requirements
   are missed, that is the reported result.

Optimized variants (quantized, distilled, compiled) are evaluated at their own
re-selected validation threshold and must meet Q5 relative to the full model.

### 6.5 Baseline (Q4)

A rules baseline of regular expressions for phone numbers, emails, and handle
patterns, plus an off-the-shelf PII entity detector (e.g., Microsoft Presidio),
evaluated on the same test set at the same FPR. Expected to do well on plain-text
positives and near zero on hard positives.

### 6.6 Slices

Every quality metric is also broken down by:
- Hard-positive category (obfuscated / cross-message / implicit)
- Context length (bucketed by tokens)
- Number of speakers in the context (1:1 vs. group)
- Channel type (direct message vs. group)

A model that meets Q1 overall but fails one slice badly should be visible here.

### 6.7 Performance measurement

- **What:** server-side latency (p50, p95, p99), max throughput while meeting R2,
  GPU/CPU utilization, and cost per million messages.
- **Load generator:** open-loop with Poisson arrivals, so requests arrive
  independently of responses. A closed loop, where the next request waits for the
  previous one, hides queueing and understates p99.
- **Inputs:** context lengths sampled from the test set's distribution; separate
  runs at the 512-token worst case for R3.
- **Placement:** load generator runs on the same host or region as the service, so
  network latency isn't measured as model latency.
- **Hygiene:** warmup before measuring; runs long enough for stable p99
  (≥ 10,000 requests per load level); exact hardware, software versions, and config
  recorded with every result.
- **Quality re-check:** any performance optimization is re-evaluated against Q5
  before its numbers count.

### 6.8 Limitations

- **Synthetic ≠ real.** Real users are more varied and more adversarial than
  generated data. Results are an upper bound on real-world quality.
- **The positive rate is assumed.** Implied precision is a scenario, not a
  measurement.
- **Adversaries adapt.** Evasion tactics shift over time, so a fixed test set
  measures a snapshot. P4's drift monitoring is the partial mitigation.

## Predictions
## Observability
## Milestones
## Risks/Open Questions
## Results (TBD)
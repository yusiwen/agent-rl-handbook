# Study Plan (16 Weeks)

Seven to eight hours a week: about two hours of reading and derivation, five to six hours of hands-on work. **If a deliverable does not pass, stop there** — do not move on.

| Week | Content | Milestone check |
|---|---|---|
| 1 | Part 0 + Chapters 1–2 | Your task written as an MDP; compute split decided |
| 2–3 | Chapters 3–4 | GAE derived by hand; selection table with negative arguments |
| 4–5 | Chapter 5 + Project 1 | A new verifier marking answers end to end, with its failure boundaries written down |
| 6–7 | Chapter 6 + Project 2 | A GRPO training log + a credit-assignment design document |
| 8 | Chapters 7–8 | A pass-rate report for your task set + a minimal reproducible environment |
| 9–10 | Chapter 9 + Project 3 | A memory ledger + DPO/GRPO configs that reproduce on the Spark |
| 11–12 | Chapter 10 + Projects 4 and 5 | Two end-to-end run logs (0.6B scale) |
| 13 | Chapter 11 | An evaluation report template + a regression script |
| 14–15 | Chapter 12 + Project 6 | A skill service running continuously on the Spark |
| 16 | Chapter 13 + extras | The pinboard notebook takes shape |

## Stage checkpoints

Ask these once per stage, and write the answers into that chapter's "backfill" section:

- [ ] **Stage 1 (Chapters 1–3)**: can I write any agent task as an MDP with a reward draft?
- [ ] **Stage 2 (Chapters 4–6)**: can I say where my reward comes from, how fine-grained it is, and how it could be gamed?
- [ ] **Stage 3 (Chapters 7–9)**: can I estimate an experiment's GPU-hours and memory, and say where the bottleneck is?
- [ ] **Stage 4 (Chapters 10–11)**: can a third party reproduce — or disprove — my result?
- [ ] **Stage 5 (Chapters 12–13)**: can I place a new paper and decide whether to follow it in five minutes?

## Rhythm rules

1. **Deliverables first.** Reading is not learning; shipping the deliverable ends a chapter.
2. **Do not skip Stage 3.** Chapters 7 and 8 look like "engineering detail" and get skipped — they are exactly where "I changed the algorithm and nothing happened" comes from.
3. **Do not postpone Chapter 11.** Start building the evaluation script as soon as Project 3 is done; before that, every comparison you make is untrustworthy.
4. **Backfill the four questions every chapter.** It is the book's completeness check: if one question stays blank for a long time, the studying has drifted.

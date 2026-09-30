# Writing and Formatting Rules

This handbook is maintained over a long period, so consistency matters more than style. Two rules are non-negotiable: **everything is written in English**, and **everything is written for a beginner**.

## Chapter skeleton (fixed)

```text
# Chapter N · Title
> Compute tier / status / project / special notes (blockquote)
## Plain English first
## What you'll be able to do
## Lessons            (- [ ] checklist, tick as you go)
## Project            (if one hangs off this chapter)
## Deliverable
## How you know you passed
## Reading
## Backfill           (which of the four questions did this answer?)
```

## Plain-language rules

The audience is a beginner with basic Python and some fine-tuning experience. That means:

1. **Explain before you use.** Every acronym gets its plain meaning on first use in a chapter: "the reward model (a second model trained to score answers)".
2. **One idea per sentence.** If a sentence contains "however, because, which means", split it.
3. **No naked formulas.** Show the formula only after saying in words what it does, and add one sentence explaining each symbol.
4. **Concrete first, abstract second.** A toy example (a two-step task, a single tool call) before the general case.
5. **Say what it costs.** Every method gets a "what it costs" half-sentence — this is what beginners are missing most.
6. **Keep English technical terms in English.** Do not translate `rollout`, `checkpoint`, `loss mask` — translated variants make searching and matching against papers impossible.
7. **Analogies are welcome, but only one per concept**, and they must not contradict the mathematics.

## Terminology

| Use | Do not use |
|---|---|
| Tool-Call RL | Toolcall RL / ToolCall RL |
| RLHF, RLAIF, RLVR (spell out on first use) | invented abbreviations that match no paper |
| Reagent-U (define on first use: U = Unified Feedback Integration) | Reagent U / REAGENT-U |
| PPO, GRPO, DPO | variant names of your own invention |
| rollout, worker, checkpoint, verifier | home-made translations of them |

## Citations and honesty

- Papers: give the title plus the arXiv ID. **If the ID has not been checked by hand, mark it "ID to verify".**
- Claims come in exactly three kinds, and must not be blended:
  - **Textbook fact** — settled, textbook-level (for example the Bellman equation derivation)
  - **Paper result** — cite it, and mark "reported, not reproduced"
  - **My judgement** — label it as a judgement, and give the counterexample or boundary
- Measured numbers (throughput, memory, wall-clock) are invalid unless the hardware and configuration are stated with them.

## Maths

- Inline: `$...$`. Display: `$$...$$` (MathJax is enabled in `book.toml`).
- Consistent symbols: policy $\pi_\theta$, return $G_t$, value $V^\pi(s)$, advantage $A^\pi(s,a)$, discount $\gamma$, KL coefficient $\beta$.

## Naming and links

- Files and directories: English lowercase slugs with numeric prefixes (for example `09-training-runtime/index.md`).
- Cross-references: relative markdown links, never absolute paths.
- Checklists use `- [ ]`, flipped to `- [x]` when done — this is the book's progress-tracking mechanism.

## Keeping it honest over time

- Add new material to an existing chapter rather than opening a new one; if a new chapter really is needed, update `SUMMARY.md` and the map on the home page in the same change.
- When a fact changes (a framework gains sm_121 support, for example), **edit the sentence and keep one clause saying when it changed** — never delete silently.

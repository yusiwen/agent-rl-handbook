# Agent RL Explained: A Beginner's Handbook

This is not a course transcript. It is a **study handbook written for myself and for anyone starting from zero** — the same ideas as a professional Agent RL course, but re-ordered so a beginner can actually follow them, with every piece of jargon explained in plain words before it is used.

The whole book circles four questions. They are also the acceptance test: if you can answer them about *your own* project, you learned something.

> 1. **When should I reach for RL at all?** (Supervised fine-tuning, DPO, GRPO, PPO, skills and evaluation each fit a different stage.)
> 2. **How do I design the reward?** (Verifiable rewards, process rewards, LLM-as-judge, mixed rewards — and how to stop the model from cheating them.)
> 3. **How do I debug a long multi-turn failure?** (Was it planning, the tool call, the environment feedback, the summary, or the credit assignment?)
> 4. **Where does my training system quietly lie to me?** (Rollouts, masks, PRM judges, async training, weight sync, stale samples.)

## What "Agent RL" means, in one paragraph

**Reinforcement learning (RL)** is learning by trial and error from feedback: try something, see how it went, do more of what worked. An **agent** is a model that can take several steps — search, call a tool, run code, look at the result — instead of answering in one shot. **Agent RL** is therefore: let the model attempt a multi-step task, then use the outcome (and sometimes feedback on each step) to improve the model's weights. That is the whole idea. Everything else in this book is about doing it *reliably*.

## Who this book is for

You are in the right place if:

- You can write basic Python, and you have met a neural network before (at least the idea that "training changes numbers in a big table").
- You have used an LLM, maybe fine-tuned one with LoRA, and keep bumping into words like PPO, GRPO, DPO, reward model, rollout.
- You want to understand *why* each method exists, not memorise formulas.

If you have never touched Python or PyTorch, do that first — this book will not teach you to program.

## How to read it

1. **Straight through.** The six parts are a ladder; each part assumes the one before. Chapter 2's MDP exercise comes back in Chapter 6, where you rewrite it.
2. **As a checklist.** Every chapter ends with a deliverable and a "how you know you passed" list. Reading is not finishing — shipping the deliverable is.
3. **As an index.** Chapter 13 maps 260+ papers. The [project index](appendix/projects.md) tells you which chapter a given piece of technology lives in.

## Book map

| Part | Chapter | Lessons | Compute | Deliverable |
|---|---|---|---|---|
| **0 Before you start** | [Part 0](start/index.md) | 3 | ✅ | A goal-alignment sheet |
| **1 A shared language** | [1 · The Big Picture](01-landscape/index.md) | 4 | ✅ | A draft decision on whether to use RL |
| | [2 · From MDP to Post-Training](02-mdp/index.md) | 4 | ✅ | Your task written as an MDP |
| | [3 · Value Functions → Policy Gradients](03-policy-gradient/index.md) | 6 | ✅ | Hand-derived GAE + a sparsity note |
| **2 Signals** | [4 · Alignment Algorithms and PPO](04-alignment-algorithms/index.md) | 10 | ✅ | An algorithm selection table |
| | [5 · Reward Engineering and Verifiers](05-reward-engineering/index.md) | 9 | ✅ | A working verifier |
| | [6 · Credit Assignment](06-credit-assignment/index.md) | 6 | ✅ | A credit-assignment design doc |
| **3 Industrial** | [7 · Data and Task Engineering](07-data-and-tasks/index.md) | 6 | ⚠️ | A task-set health report |
| | [8 · Environment Engineering](08-environment-engineering/index.md) | 6 | ✅ | A concurrent, resettable environment |
| | [9 · The Training Runtime](09-training-runtime/index.md) | 10 | ⚠️ | A memory ledger + a working config |
| **4 End-to-end** | [10 · End-to-End Projects](10-end-to-end-projects/index.md) | 9 | ⚠️ | Two reproducible run logs |
| | [11 · Evaluation and Shipping](11-evaluation/index.md) | 8 | ✅ | An evaluation report + a regression script |
| **5 Frontier** | [12 · Skills and Self-Evolution](12-skills/index.md) | 11 | ✅ | A long-running skill service |
| | [13 · Papers and Judgement](13-papers/index.md) | 8 | ✅ | A "where does this paper sit" notebook |
| | [Extra modules](electives/index.md) | 4 blocks | ✅ | — |

The map above lists **100 lessons** in total: about 20–22 hours of reading, plus 60–100 hours of hands-on work. See the [16-week plan](appendix/schedule.md).

## How hard is the compute?

Every chapter carries a compute tier. The reference machine is a **DGX Spark** (GB10, 128 GB unified memory, aarch64 + sm_121):

| Tier | Meaning |
|---|---|
| ✅ | Runs comfortably on the Spark |
| ⚠️ | Runs, but you must shrink things (small models, QLoRA, a different attention backend) |
| ❌ | Needs multiple GPUs or a rented machine |

The plan: the Spark is the always-on workbench; heavy experiments are rented by the hour. Details in [compute tiers](appendix/compute-tiers.md).

## How every chapter is built

```text
First, the gist      — the chapter in four sentences, no jargon
What you'll be able to do
Lessons              — a checkable list; tick as you go
Project              — if a project hangs off this chapter
Deliverable          — the thing you must produce
How you know you passed
Reading
Backfill             — which of the four questions did this answer?
```

## Progress

- [ ] Part 0 · Before you start
- [ ] Chapter 1 · The Big Picture and How to Choose an Algorithm
- [ ] Chapter 2 · From MDP to LLM Post-Training
- [ ] Chapter 3 · From Value Functions to Policy Gradients
- [ ] Chapter 4 · The Alignment Algorithm Family (and PPO in Full)
- [ ] Chapter 5 · Reward Engineering and Verifiers
- [ ] Chapter 6 · Credit Assignment: From Token to Turn
- [ ] Chapter 7 · Data and Task Engineering
- [ ] Chapter 8 · Environment Engineering and Rollout Infrastructure
- [ ] Chapter 9 · The Industrial Training Runtime
- [ ] Chapter 10 · End-to-End Project Cluster
- [ ] Chapter 11 · Evaluation and Shipping Validation
- [ ] Chapter 12 · Skills: Capability Libraries and Self-Evolution
- [ ] Chapter 13 · Paper Deep-Dives and a Judgement Framework
- [ ] Extra modules

## Reading and building locally

```bash
mdbook serve --open     # live preview at http://localhost:3000
mdbook build            # writes the site to book/
nix build               # fully reproducible build; result/ is the whole site
```

The toolchain is pinned by the Nix flake (mdBook 0.5.4), so every machine gets the same environment. See `.envrc`.

## Honesty rules for this book

- This is a personal study handbook. It is not affiliated with any course or vendor; a course outline was used only to choose an order.
- Every claim is tagged: a **textbook fact**, a **paper's reported result** (marked "reported, not reproduced"), or **my own judgement**.
- Numbers (throughput, memory, runtime) are meaningless without the hardware and configuration that produced them, so both are always written down.
- Anything I have not verified myself is marked "to verify" — never stated as settled truth.
- Every term is defined in the [glossary](appendix/glossary.md); formatting conventions live in [writing and formatting rules](appendix/conventions.md).

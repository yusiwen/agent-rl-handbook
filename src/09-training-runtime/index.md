# Chapter 9 · The Industrial Training Runtime

> **Compute tier**: ⚠️ (single-GPU small models are fine; multi-node and Megatron routes need rented hardware) | **Status**: stub
> **Project**: [Project 3 · The VeRL training runtime](../appendix/projects.md)
> **Rule of this book**: this chapter is about **system implementation only**; the meaning of the algorithms is in [Chapter 4](../04-alignment-algorithms/index.md).

## Plain English first

Training a real model needs several models alive at once (the one being trained, a frozen copy for reference, a scorer, sometimes a critic), spread across several GPUs, all while generating fresh samples. A "runtime" is the software that keeps that circus organised. This chapter reads one such runtime (VeRL) end to end: how data flows, what the workers do, where the loss is computed — and what to do when it crashes at 3 a.m.

## What you'll be able to do

- Describe what actually happens after you type one training command.
- Explain how the same runtime runs PPO, GRPO, DPO and a dozen variants without being rewritten.
- Account for where the memory goes, and know which knob to turn when you run out.
- Recover a crashed run from a checkpoint without breaking the curves.

## Lessons

- [ ] 1. Why a runtime is needed at all: four models × distribution × online sampling
- [ ] 2. VeRL's global picture and its six-layer architecture
- [ ] 3. DataProto and the single controller: the two ideas that make distribution look like one process
- [ ] 4. The training path, step by step: from one command to one complete run
- [ ] 5. How algorithms are composed: PPO, GRPO, DAPO and the rest in one codebase
- [ ] 6. Reward loops and agent loops: from a single-turn trainer to a multi-turn tool-calling runtime
- [ ] 7. **Operations**: checkpoints and resume, failure recovery, parallelism (TP/PP/DP/EP), and the **memory ledger**
- [ ] 8. **Frameworks compared**: VeRL, ROLL, AReaL, SkyRL, Agent Lightning, OpenRLHF, TRL
- [ ] 9. Project 3a: a GRPO run on Qwen3 (Spark-sized: small model, `sdpa` attention backend, single GPU)
- [ ] 10. Project 3b: a DPO run for preference alignment (lighter — do this one first)

## The memory ledger (the core table of this chapter — fill in measured values)

| Component | Parameters | Optimiser state | Gradients | Activations | Notes |
|---|---|---|---|---|---|
| Policy (actor) | | | | | |
| Reference | | | | | |
| Critic (PPO only) | | | | | |
| Reward model | | | | | |
| Rollout KV cache | | | | | |

## Deliverable

- A memory/cost ledger filled in with numbers measured on your own machine
- A training configuration that reproduces on the Spark (including attention backend and compile-concurrency limits)
- A troubleshooting sheet for loss curves: what entropy collapse, KL explosion and a flat reward each mean

## How you know you passed

- You can list the five things that happen behind one command, and roughly what share of the time each takes.
- After an interrupted run, training resumes from the checkpoint and the curve continues smoothly.
- You can set breakpoints at the actor loss, the advantage computation and the KL term, and explain how the parameters change.

## Reading

- VeRL documentation: <https://verl.readthedocs.io/>
- To fill in: the HybridFlow paper; the home repositories of AReaL, SkyRL, Agent Lightning and ROLL.

## Backfill

- [ ] Question 4: Where does my training system quietly lie to me? (This chapter is the main battlefield.)

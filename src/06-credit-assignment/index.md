# Chapter 6 · Credit Assignment: From Token to Turn

> **Compute tier**: ✅ (0.5–3B models are comfortable) | **Status**: stub
> **Project**: [Project 2 · Process feedback and long-trajectory credit assignment](../appendix/projects.md)

## Plain English first

When a ten-step agent task fails, it is almost never true that *every* step was wrong. **Credit assignment** is the job of working out which steps deserve the blame (or the praise) — and then turning that judgement into something a training loop can use. The practical trick is to score smaller pieces: a step, a tool call, one turn of the conversation, instead of the whole run. This chapter gives you a map of those options, then walks through one implementation (Reagent) in detail.

> Terminology: the "U" in **Reagent-U** stands for *Unified Feedback Integration* — merging written critiques and numeric scores into one RL loop.

## What you'll be able to do

- Explain why "give every step a score" is *not* the same thing as "credit assignment is solved".
- Place any credit-assignment method on the token / step / turn / trajectory map and say what it costs.
- Describe how written feedback and numeric reward can be combined, and what breaks when they conflict.

## Lessons

- [ ] 1. Why credit assignment is hard: long trajectories, sparse feedback, discrete tool calls, noisy negative samples
- [ ] 2. **The solution map**: token-level, step-level, turn-level and trajectory-level signals
- [ ] 3. Reagent-U: merging written critiques with scalar rewards, and the two-stage sampling it uses
- [ ] 4. What the paper claims, and what its experiments actually show
- [ ] 5. Project 2a: Reagent's architecture, plus the Docker environment
- [ ] 6. Project 2b: agent SFT, rule/model reward fusion, a GRPO run and an inference-service loop

## The solution map (the core table of this chapter — to be filled in)

| Signal level | Example methods | What it costs | Best suited to |
|---|---|---|---|
| Token | GAE ([Chapter 3](../03-policy-gradient/index.md)) | needs a critic | single-turn long reasoning |
| Step | PRM, GiGPO | needs labels or extra sampling | tasks with clear step boundaries |
| Turn | TRACE, Agent Lightning's LightningRL | needs turn boundaries to be detected | multi-turn tool calls |
| Trajectory | outcome reward + voting/comparison | weakest supervision | when only the final result is visible |

## Deliverable

- The MDP draft from [Chapter 2](../02-mdp/index.md), upgraded into a design document that includes process feedback and a credit-assignment scheme.
- A working GRPO training log, including how rule rewards and model rewards were weighted.

## How you know you passed

- You can explain why scoring every step does not automatically solve credit assignment.
- You can describe your policy for resolving rule-vs-model reward conflicts, and why you chose it.
- The training log shows feedback actually reaching the next round of sampling — not just scores being recorded.

## Reading

- To fill in: the Reagent paper.
- GiGPO (groups within groups; injects step-level signals): <https://arxiv.org/abs/2505.10978>
- A survey and taxonomy of credit assignment (47 methods, 2024–2026): <https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL>
- TRACE (turn-level reward assignment): arXiv:2607.13988 (ID to verify)

## Backfill

- [ ] Question 3: How do I debug a long multi-turn failure? (This chapter is the main battlefield.)

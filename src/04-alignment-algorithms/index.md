# Chapter 4 · The Alignment Algorithm Family (and PPO in Full)

> **Compute tier**: ✅ | **Status**: stub
> **Rule of this book**: the *meaning* of every algorithm is explained here, once. [Chapter 9](../09-training-runtime/index.md) only covers how a runtime switches between and extends them — it does not repeat the objectives.

## First, the gist

By the end of this chapter you will know five names — RLHF, RLAIF, DPO, RLVR, GRPO — and, more importantly, *which problem each one is for*. The short version: PPO is the general-purpose but expensive option; GRPO is a cheaper variant that works when you can score answers automatically; DPO skips the whole "train a judge, then sample from the model" loop and learns straight from preference pairs. Choosing well is a matter of four things: where feedback comes from, how fine-grained the reward is, whether you need fresh samples, and what your cluster costs.

## What you'll be able to do

- Say what each acronym stands for, and what it is *for*, in one sentence each.
- Write down what PPO, GRPO and DPO actually optimise, and how much sampling each needs.
- Explain in plain words why "DPO needs no reward model" is true — and what you pay for that.
- Give a first answer to "which one fits my task", with reasons for *not* choosing the others.

## Lessons

- [ ] 1. Why alignment is not solved by making the model bigger: HHH, the limits of SFT, and where RLHF begins
- [ ] 2. **The full PPO objective**: clipping, advantage estimates, the KL penalty — and why KL is the safety valve
- [ ] 3. Four models in one training run: policy, reward model, critic and reference
- [ ] 4. GRPO: sample a *group* of answers, compare them with each other, drop the critic
- [ ] 5. DPO, fully unpacked: from Bradley-Terry to the closed form — it is **offline preference optimisation** with an implicit reward
- [ ] 6. The DPO family (KTO, ORPO, SimPO and friends): what the "theoretical patches" are patching
- [ ] 7. RLAIF and scalable oversight: Constitutional AI, LLM-as-judge, self-rewarding
- [ ] 8. RLVR: reinforcement learning with verifiable rewards — the paradigm and its boundaries
- [ ] 9. Five alignment traps: reward hacking, distribution shift, over-optimisation and friends
- [ ] 10. Choosing: feedback source × reward granularity × online sampling × system cost

## Deliverable

A one-page selection table that **must include the negative argument**:

| Task feature | Pick | Why not the other two |
|---|---|---|
| A provably right answer exists (maths, code) | | |
| Only preferences exist (writing, style) | | |
| Multi-turn tool calling | | |
| Plenty of offline data, no GPU | | |

## How you know you passed

- You can write the objectives of PPO, GRPO and DPO, and point out how each depends on fresh sampling.
- You can explain what the KL penalty prevents in theory and in practice.
- You can state DPO's "no reward model needed" claim *and* its hidden price in one sentence.

## Reading

- To fill in: InstructGPT, DPO, DeepSeekMath (GRPO), Constitutional AI, DeepSeek-R1.

## Backfill

- [ ] Question 1: When should I reach for RL at all? (The algorithm side settles here.)
- [ ] Question 2: How do I design the reward? (From a reward model, from AI feedback, or from a program.)

# Chapter 13 · Paper Deep-Dives and a Judgement Framework

> **Compute tier**: ✅ | **Status**: stub

## First, the gist

New RL papers appear every week, and reading them one by one is a losing game. The fix is a *map*: a fixed set of slots ("this is a PPO-improvement paper", "this is an asynchronous-training paper") so a new paper can be filed in thirty seconds instead of understood from scratch. This chapter builds that map, then walks through the six areas where the paper traffic is heaviest.

## What you'll be able to do

- Place an unseen paper into the map quickly, and name its nearest neighbours.
- Explain the recurring problems of GRPO-style training — normalisation bias, entropy collapse, vanishing gradients — and how the literature attacks each.
- Decide whether a paper is worth reproducing, using a minimal experiment instead of enthusiasm.

## Lessons

- [ ] 1. How to use the map: four axes (algorithm family × engineering theme × application domain × agent capability)
- [ ] 2. Improving and industrialising PPO: from "alchemy" to engineering
- [ ] 3. The DPO wars: 18 theoretical patches and 15 engineering deployments
- [ ] 4. Reading GRPO through 41 papers: normalisation bias, entropy collapse, vanishing gradients, and how each was attacked
- [ ] 5. Reward modelling and robustness: training a judge that does not get fooled
- [ ] 6. Training systems: asynchronous vs synchronous — which is actually better?
- [ ] 7. The three hard problems of multi-turn agent RL: credit assignment, sparse rewards, training collapse
- [ ] 8. **A reproduction checklist**: the smallest experiment that tells you whether a paper is worth following

## Deliverable

A "pinboard notebook" you maintain yourself — one line per paper: where it sits, follow or not, and why.

```text
[GRPO family] Title (arXiv:xxxx) — patches the normalisation bias of group baselines
Verdict: follow. Why: directly relevant to my verifiable task, cheap to implement (loss change only).
```

## How you know you passed

- Given a new paper, you can say within five minutes which slot it belongs in and which existing paper it is cousins with.
- Every paper you decided to "follow" maps to a concrete minimal reproduction you could run.
- A healthy share of your verdicts are "do not follow" — following everything is the same as judging nothing.

## Reading

- To fill in: the concrete taxonomy of the paper map (8 top-level categories, 40+ sub-topics).

## Backfill

- [ ] All four questions: this chapter is where the four questions harden into long-term judgement.

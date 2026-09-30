# Chapter 3 · From Value Functions to Policy Gradients

> **Compute tier**: ✅ | **Status**: stub
> **Note**: the full PPO objective is **not** here — it lives in [Chapter 4](../04-alignment-algorithms/index.md). This chapter stops at GAE and answers one question: *where does the gradient come from?*

## Plain English first

To improve a model you need a direction to move in. There are two ways to find it: estimate *how good each situation is* (values), or directly nudge whatever the model did more/less depending on how it turned out (policy gradients). This chapter builds both, then combines them — and explains why, for language models, feedback arrives so late that estimating "how good this was" is genuinely hard.

## What you'll be able to do

- Explain the difference between a value, a Q-value and an advantage without looking anything up.
- Derive the policy-gradient formula in a few lines, and say why a "baseline" reduces noise.
- Explain why the LLM setting is naturally sparse: not because feedback is rare, but because it arrives at the *end* of a long sequence.

## Lessons

- [ ] 1. Policies, returns and value functions: $V$, $Q$ and the advantage $A$
- [ ] 2. The Bellman equations: why a return can be written recursively
- [ ] 3. Three ways to solve an RL problem: dynamic programming → Monte Carlo → temporal difference (TD)
- [ ] 4. Value methods vs policy methods vs actor-critic — and where the off-policy line (Q-learning, DQN) sits
- [ ] 5. Policy-gradient theorem → REINFORCE → variance reduction → **GAE (we stop here)**
- [ ] 6. A bridge: from online RL to offline preference optimisation (setting up DPO in Chapter 4)

## Deliverable

- A hand derivation of the policy gradient and GAE, including *why* a baseline is needed at each step.
- A short note: why an LLM's "action space" is vocabulary size × sequence length, and what that implies.

## How you know you passed

- You can write the Bellman expectation equation, the REINFORCE gradient and the GAE $\lambda$-return from memory.
- You can explain "LLM tasks are naturally sparse" in your own words — the reason is where the feedback lands, not how much of it there is.
- You can define on-policy vs off-policy and place PPO, DPO and GRPO on the correct side.

## Reading

- To fill in: Sutton & Barto (Chapters 3 and 13), the original GAE paper.

## Backfill

- [ ] Question 4: Where does my training system quietly lie to me? (Gradient-estimate variance and bias are born on this layer.)

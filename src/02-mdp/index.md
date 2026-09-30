# Chapter 2 · From MDP to LLM Post-Training

> **Compute tier**: ✅ | **Status**: stub

## Plain English first

An MDP is a tidy way of writing down a trial-and-error problem: *what I see*, *what I can do*, *what I get for doing it*, *what happens next*, and *how much I care about the future*. Writing an LLM task in this form forces you to answer questions you would otherwise leave fuzzy — how big is one "action", where does the feedback come from, and how long is one episode. Those answers decide which training method you can even use.

## What you'll be able to do

- Write any agent task as a five-part MDP, including the awkward part (partial observability).
- Explain in plain words why supervised fine-tuning (SFT) is not enough, and what exactly it misses.
- Say what one "action" is in your task, and why that choice matters later.

## Lessons

- [ ] 1. The engine of RL: from the interaction loop to expected return
- [ ] 2. Why SFT stops being enough: five classes of problems it cannot fix
- [ ] 3. The MDP five-tuple, explained slowly — and why POMDP shows up
- [ ] 4. Trajectory, episode and state in agent settings: **many turns = one long episode, state = context + environment**

## Deliverable

Your agent task written as an MDP, plus a first draft of the reward:

```text
State s        =
Action a       =
Reward r       =
Transition P   =
Discount γ     =
```

## How you know you passed

- The **action granularity** is explicit. One token? One tool call? One whole turn? That choice determines which credit-assignment scheme you will need in Chapter 6.
- You can say whether your reward is sparse or dense, and whether it judges the final answer or the process.
- You can point at where the partial observability comes from.

## Reading

- To fill in: a standard MDP/POMDP reference, plus work that formalises LLM generation as RL.

## Backfill

- [ ] Question 1: When should I reach for RL at all? (The answer starts with "what SFT cannot do".)

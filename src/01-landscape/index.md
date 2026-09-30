# Chapter 1 · The Big Picture and How to Choose an Algorithm

> **Compute tier**: ✅ | **Status**: stub

## First, the gist

There are only a handful of ways to make a model behave better, and they are constantly confused with each other. This chapter draws the map before any maths: what each family of methods is for, where the feedback comes from, and what it costs. One idea matters more than the rest — **RL amplifies what a model can already do; it does not inject new knowledge**. If the model cannot solve a task even once in a while, no amount of training will teach it.

## What you'll be able to do

- Name the four families of alignment methods, and say for each what problem it solves and what it costs.
- Explain, for *your* task, what "RL is an amplifier, not an injector" implies.
- Give a concrete example of a task where you should **not** use RL.

## Lessons

- [ ] 1. Eight years of alignment in one pass: the four families
- [ ] 2. Where does the feedback come from? Agents, search, vision, emotion
- [ ] 3. Amplifier vs injector: does RL have a ceiling? (Payoff: what belongs to sampling at inference time, and what genuinely needs training)
- [ ] 4. A first intuition for compute cost: rollouts dominate, multi-turn multiplies, and sometimes RL is the wrong purchase

## Deliverable

A one-page draft: "my task → should I use RL at all → if yes, which family".

## How you know you passed

- For each of the four families, you can say in one sentence what it fixes and what it costs.
- You can translate "RL is an amplifier" into a concrete statement about your own task.
- You can name a task where RL would be wasted money, and explain why.

## Reading

- To fill in: one representative paper per family, plus an open-source implementation.

## Backfill

- [ ] Question 1: When should I reach for RL at all?

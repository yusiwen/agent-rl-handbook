# Chapter 7 · Data and Task Engineering

> **Compute tier**: ⚠️ (generating and filtering needs inference; small scale is fine on the Spark) | **Status**: stub
> **Why this chapter exists**: the source course has no standalone data chapter, yet bad data is the most common reason "I swapped the algorithm and nothing improved".

## Plain English first

Training can only learn from the examples you give it. If every task is either trivially easy or impossibly hard, the model gets no useful signal — a score of 100% or 0% tells it nothing about what to change. So before touching algorithms, run a health check on your task set: how hard is each item, how varied are they, and are any of them secretly copies of your test set?

## What you'll be able to do

- Explain, with numbers from your own task set, why an over-easy or over-hard set produces no learning.
- Build the cold-start supervised data that makes later RL trainable in the first place.
- Screen a task set for difficulty distribution and test-set contamination.

## Lessons

- [ ] 1. Task distribution sets the ceiling: three real cases where changing the algorithm changed nothing
- [ ] 2. Cold-start SFT data: filtering, format, and the **label protocol** that decides whether RL can train at all
- [ ] 3. Rejection sampling and self-play data
- [ ] 4. Difficulty and curriculum: **pass-rate filtering** (too easy and too hard both give zero gradient)
- [ ] 5. Scaling the task set: programmatic task generation and environment synthesis
- [ ] 6. Decontamination and keeping the evaluation set separate

## Deliverable

A health report for "my task set":

| Metric | Result |
|---|---|
| Number of items, and how they break down by task family | |
| Pass-rate histogram from a baseline model | |
| Share of items inside the trainable band (pass rate 0.1–0.9) | |
| Contamination check against the evaluation set | |
| Verdict: should this set go into RL at all? | |

## How you know you passed

- The pass-rate histogram visibly clusters at "always right" / "always wrong", and you propose a fix based on it.
- You can explain why both a very high and a very low pass rate yield no gradient.
- The contamination check is a script you can re-run, not a visual inspection.

## Reading

- To fill in: work on programmatic task generation (for example SkyRL's Endless Terminals idea).

## Backfill

- [ ] Question 1: When should I reach for RL at all? (If the data fails the check, the answer is "not yet".)

# Chapter 11 · Evaluation and Shipping Validation

> **Compute tier**: ✅ (the Spark is ideal for running evaluations around the clock) | **Status**: stub
> **Why this chapter exists, and why it matters most**: without it, every training run you do degrades into "it looks better".

## First, the gist

A rising reward curve does not prove the model got better — it may only prove the model learned to please the scorer. Evaluation is how you avoid fooling yourself: run the same fixed test many times, report a score *with an error bar*, use judges whose biases you have measured, and keep a regression suite so an improvement stays an improvement. This is the chapter that turns a hobby into engineering.

## What you'll be able to do

- Report a result as "improved by X points ± Y over N seeds" instead of a single flattering number.
- Define success for a multi-turn task in a way that cannot be gamed.
- Detect the common failure modes of an LLM judge.
- Keep a regression suite that catches silent breakage.

## Lessons

- [ ] 1. "Did it get better?" — why the reward curve is not an answer
- [ ] 2. Defining success for multi-turn tasks: final-outcome vs process-based judging
- [ ] 3. pass@k vs pass@1: what RL actually improves
- [ ] 4. Variance and significance: multiple seeds, confidence intervals, how to design an ablation
- [ ] 5. Judge reliability: systematic bias in LLM-as-judge, calibration, and how much human spot-checking to keep
- [ ] 6. The offline-to-online gap: regression tests, A/B tests, gradual rollout and rollback
- [ ] 7. Contamination and "overfitting the test set"
- [ ] 8. Building a reusable evaluation harness

## Deliverable

- An evaluation report template that includes multiple seeds and confidence intervals
- A regression script you can run after any change to model, reward or data
- A table mapping each of your metrics to the real-world risk it covers

## How you know you passed

- Every training comparison you report comes with a spread, not a single number.
- You can name at least two judge biases and show an example of each (for instance a preference for longer or prettier answers).
- The regression script runs on a schedule or in CI, and a failure points at the specific examples that changed.

## Reading

- To fill in: work on multi-seed evaluation, variance and significance, and judge calibration.

## Backfill

- [ ] All four questions: evaluation is the final referee.

# Chapter 5 · Reward Engineering and Verifiers

> **Compute tier**: ✅ | **Status**: stub
> **Project**: [Project 1 · Verifiable-reward infrastructure](../appendix/projects.md)

## Plain English first

A reward is just a score you give the model's attempt. Reward engineering is deciding *who gives that score, how often, and how precisely*. Give it only at the very end and learning is slow; give it at every step and you must build the scorer yourself; let a model do the scoring and it can be fooled. This chapter walks that trade-off, then builds a real automatic grader (a "verifier") that marks maths, code or logic answers by running a program instead of asking a human.

## What you'll be able to do

- Place any reward scheme on the dense↔sparse spectrum and say what it costs.
- Explain the difference between outcome, process and generative reward models — ORM, PRM, GRM.
- Build a verifier for a new task: task definition → model calls → automatic marking → replayable dataset.
- List the ways your verifier will be cheated, *before* the model finds them.

## Lessons

- [ ] 1. Reward granularity: from a single final score to dense step-by-step supervision
- [ ] 2. Outcome vs process: ORM and PRM, and the interface they give to credit assignment
- [ ] 3. Generative reward models (GRM): from "here is a number" to "here is why"
- [ ] 4. Unsupervised and intrinsic rewards, including implicit-process routes such as PRIME
- [ ] 5. Mixed rewards: what to do when a rule-based score and a model-based score disagree
- [ ] 6. **Training and refreshing the reward model itself**: preference pairs, generalisation, and how often to retrain it
- [ ] 7. Reward hacking and over-optimisation: causes, detection and defences (KL limits, model ensembles, adversarial examples, production monitoring)
- [ ] 8. Inside the Verifiers framework: from a reward function to reward infrastructure
- [ ] 9. Project 1 hands-on: task definition, model calls, automatic marking, replayable datasets, evaluation loop

## Project 1 · Verifiable-reward infrastructure

Turning "was the answer right?" into an automatic, scalable training signal:

- Break down the verifier service, the rule orchestration, automatic marking and the replayable dataset
- Walk the full RLVR loop: generate data → sample → mark → recover the reward
- Build a first verifier for a new task and predict its failure modes

Environment: a small local model served by vLLM (the Spark can do this) plus the Verifiers library.

## Deliverable

- A verifier for a new task that marks answers automatically and reproducibly
- A replayable marking dataset containing both passing and failing examples
- A **failure-boundary list** for that verifier: when will it give the wrong signal?

## How you know you passed

- Re-run on a fresh batch from the same distribution and the marks are stable and reproducible.
- You can write down at least three ways an agent could cheat this verifier.
- The RLVR loop runs end to end and prints the recovered rewards.

## Reading

- Verifiers (Prime Intellect): <https://github.com/PrimeIntellect-ai/verifiers>
- To fill in: the original PRM, GRM and PRIME papers.

## Backfill

- [ ] Question 2: How do I design the reward? (This chapter is the main battlefield.)

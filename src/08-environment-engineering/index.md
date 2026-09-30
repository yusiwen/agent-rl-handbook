# Chapter 8 · Environment Engineering and Rollout Infrastructure

> **Compute tier**: ✅ (the Spark is a good place to practise this) | **Status**: stub
> **Why this chapter exists**: in agent RL, the bottleneck is usually the *environment*, not the algorithm.

## First, the gist

The "environment" is whatever the agent acts on: a sandbox, a terminal, a database, a website. Training means running the agent against it thousands of times, so the environment must be fast to start, easy to reset, safe to run in parallel, and honest in what it reports back. When training looks broken, the cause is often an unstable environment pretending to be an algorithm problem.

## What you'll be able to do

- Define a clean environment interface: what state the agent sees, and what "reset" means.
- Run many environments in parallel without them contaminating each other.
- Reproduce a run from a task seed, or explain precisely why you cannot.
- Estimate how much GPU time one training step's rollouts will cost.

## Lessons

- [ ] 1. The environment as an interface: abstraction, reset semantics, observation format
- [ ] 2. Sandbox pools, concurrency and quotas (Docker isolation, timeouts, circuit breakers)
- [ ] 3. Reproducibility: controlling randomness, rolling state back, task seeds
- [ ] 4. Feedback stability: noisy environments, unreliable negative samples, partial success
- [ ] 5. Lessons from terminal/tool environments: long horizons, discrete interactions, **a tiny fraction of tokens actually changes the environment**, many failure modes
- [ ] 6. Rollout throughput and cost accounting: how many GPU-hours does one batch of samples burn?

## Deliverable

A minimal environment that is concurrent, resettable and reproducible:

- A Dockerfile with pinned dependencies
- A reset script plus a documented task-seed convention
- A throughput benchmark: time and GPU-hours per rollout batch

## How you know you passed

- The same task seed gives the same trajectory twice — or you can name exactly where the non-determinism comes from.
- N environments run side by side without contaminating each other, and a broken one is killed and recycled.
- You can state the rollout cost of "one training step" and whether the bottleneck is the environment or the model.

## Reading

- To fill in: engineering write-ups on terminal-environment agentic RL (the "bitter lesson" genre).

## Backfill

- [ ] Question 3: How do I debug a long multi-turn failure? (Rule out "the environment itself is flaky" first.)
- [ ] Question 4: Where does my training system quietly lie to me? (Environment noise is the most upstream lie of all.)

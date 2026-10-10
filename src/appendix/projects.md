# Index of the Six Projects

Ordered along one line: **where the reward comes from → how credit is assigned → the system → end to end → asynchronous online → no weight updates at all**.

**What this page is, and what it is not.** It is a specification: what each project is, what you produce, and how you know it worked. No reference implementation ships with this book today; if one is published later it will be linked here, project by project, with the chapter and a tag. Everything you write goes into that project's own note file, against the template below.

## The six at a glance

| # | Project | What you practise | Chapter | Compute |
|---|---|---|---|---|
| 1 | **Verifiers** — verifiable rewards for maths / code / logic tasks | Turning "is the answer right?" into an automatic, scalable signal; the verifier service, rule orchestration, marking, replayable datasets | [Chapter 5](../05-reward-engineering/index.md) | [✅](../appendix/compute-tiers.md) |
| 2 | **Reagent** — process feedback and long-trajectory credit assignment | Deploying an RRM, running agent SFT, fusing rule and model rewards, closing the loop with GRPO training plus an inference service | [Chapter 6](../06-credit-assignment/index.md) | [✅](../appendix/compute-tiers.md) |
| 3 | **VeRL** — an industrial PPO / GRPO / DPO post-training runtime | DataProto, worker groups, rollout loops, reward loops, agent loops; breakpoints at actor loss, advantage and KL | [Chapter 9](../09-training-runtime/index.md) | [⚠️](../appendix/compute-tiers.md) |
| 4 | **DeepAnalyze** — a data-science agent trained end to end | An 8B model doing analysis, running code, producing charts and reports; cold-start label protocol → GRPO | [Chapter 10](../10-end-to-end-projects/index.md) | [⚠️](../appendix/compute-tiers.md) |
| 5 | **OpenClaw-RL** — an online RL loop for multi-turn tool callers | Three stages (SFT / RL / PRM+RL); the four-ring async architecture, next-state signals, loss masks, weight sync, sample freshness | [Chapter 10](../10-end-to-end-projects/index.md) | [⚠️](../appendix/compute-tiers.md) |
| 6 | **Memento-Skills** — deployed-agent skill self-evolution | The read-execute-reflect-write loop; skill discovery, routing, verification, write-back and version governance; getting better without touching the weights | [Chapter 12](../12-skills/index.md) | [✅✅](../appendix/compute-tiers.md) |

## One template for every project

Write each project's notes with this skeleton. If a section is empty, leave it empty — do not skip it. Every project is written up against these **nine faces** — problem, environment, data, architecture, configuration, source, runs, failure modes, evaluation:

```text
1. Problem      : which engineering problem does this project solve? (one sentence)
2. Environment  : dependencies, Docker, image source, platform notes (aarch64 / sm_121)
3. Data         : dataset source, label protocol, size and difficulty distribution
4. Architecture : data-flow diagram + what each key module is responsible for
5. Config       : key hyper-parameters, measured memory use, wall-clock time
6. Source       : where you set breakpoints, and what you saw
7. Runs         : exact commands, logs, curves
8. Failure modes: what broke, and which layer it traced back to
9. Evaluation   : how you decided it worked
```

## What you actually do, project by project

The chapters hold the deliverable and the pass criteria. These lines are the actions that produce them — two to four per project, in the order you would do them.

### Project 1 · Verifiers

- Stand the verifier service up and run one complete loop: generate candidate answers, mark them, and return the reward to a sampler.
- Write the first verifier for a task of your own, then write down where it lies — what it accepts that should have failed, and what it rejects that should have passed.
- Make the dataset replayable: same task, same marking, twice, the same reward. A verifier you cannot re-run is not infrastructure.

### Project 2 · Reagent

- Deploy the RRM (reasoning reward model) and read its three outputs on a real trajectory: the reasoning trace, the focused critique, the score.
- Fuse the rule reward with the model reward at explicit weights, and record what happens where the two disagree — that boundary is the interesting part, not the training curve.
- Run agent SFT, then the GRPO loop plus the inference service, and check that the feedback from one round visibly changes the next round's samples.

### Project 3 · VeRL

- Follow one full path: from the launch command through DataProto, the worker groups, the rollout loop and the reward loop.
- Set breakpoints at the actor loss, the advantage computation and the KL term, and explain in your own words what the parameters do between them.
- Fill in the chapter's memory ledger with measured numbers, and get one configuration that reproduces on your own machine. Do the DPO run first — it is the lighter of the two.

### Project 4 · DeepAnalyze

- Bring the stack up — vLLM behind an API, the file service, the code sandbox, the front end — then trace one request end to end and label what each step produces.
- Write down the cold-start label protocol (thinking, tool call, output format) and say why each field has to exist for the later RL to be trainable at all.
- Run the GRPO script and follow a single reward from a sandbox result back to the policy parameters.

### Project 5 · OpenClaw-RL

- Map the four asynchronous rings — serving, rollout collection, reward judging, policy training — and find the queue between each pair. This is where the word "async" stops being abstract.
- Read the reward path in the source: the process reward model scores the pair (response, next state) and returns a scalar vote, three votes by default, and a hint is extracted when the next state contains a real correction. Note what a hint is for — it is not a reward, it becomes a teacher signal.
- Learn the three paradigms before touching a launcher, because the naming is unforgiving: **binary RL** (the scalar reward above), **OPD** — on-policy distillation, where the hint is appended to the prompt and a teacher model supplies a per-token signal on the response — and the **combination** that weights both losses, `w_RL * L_GRPO + w_OPD * L_OPD`, one apiece by default. *(the expansions and the defaults are from the project's own README, read 2026-10-09; the source course outline calls this "the OPD combination reward" without expanding it)*
- Locate the loss mask, the session header and the weight-sync call, and say what each one breaks in the gradient when it is wrong.
- **Know the scale before you start.** The project's own launcher assumes an eight-GPU node: four for the actor, two for rollout, one for the process reward model, one for the teacher. On one machine you are re-running the tool-call path at the 0.6B scale with reduced parallelism — the mechanism is the deliverable, not the throughput.

### Project 6 · Memento-Skills

- Install, configure and verify the service, then put it in front of a real task and let a skill fail.
- Follow the loop on that failure — read, execute, reflect, write, verify — and keep the diff that was written back, because the diff is the evidence.
- Author one skill of your own: the description, the script, and the failure boundary that tells the router when not to use it.
- Measure retrieval rather than describing it: recall at k or hit rate, against a plain semantic-search baseline.

## What the six projects have in common

They cover the **three sources of reward** (rules, models, environments) and the **two ways an agent gets better** (updating weights, or growing a skill library). Seen together, they form one engineering map of where agent capability actually comes from.

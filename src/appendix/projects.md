# Index of the Six Projects

Ordered along one line: **where the reward comes from → how credit is assigned → the system → end to end → asynchronous online → no weight updates at all**. Each project is broken down across eight faces: environment, data, source code, configuration, service, training script, inference, evaluation.

| # | Project | What you practise | Chapter | Compute |
|---|---|---|---|---|
| 1 | **Verifiers** — verifiable rewards for maths / code / logic tasks | Turning "is the answer right?" into an automatic, scalable signal; the verifier service, rule orchestration, marking, replayable datasets | [Chapter 5](../05-reward-engineering/index.md) | ✅ |
| 2 | **Reagent** — process feedback and long-trajectory credit assignment | Deploying an RRM, running agent SFT, fusing rule and model rewards, closing the loop with GRPO training plus an inference service | [Chapter 6](../06-credit-assignment/index.md) | ✅ |
| 3 | **VeRL** — an industrial PPO / GRPO / DPO post-training runtime | DataProto, worker groups, rollout loops, reward loops, agent loops; breakpoints at actor loss, advantage and KL | [Chapter 9](../09-training-runtime/index.md) | ⚠️ |
| 4 | **DeepAnalyze** — a data-science agent trained end to end | An 8B model doing analysis, running code, producing charts and reports; cold-start label protocol → GRPO | [Chapter 10](../10-end-to-end-projects/index.md) | ⚠️ |
| 5 | **OpenClaw-RL** — an online RL loop for multi-turn tool callers | Three stages (SFT / RL / PRM+RL); the four-ring async architecture, next-state signals, loss masks, weight sync, sample freshness | [Chapter 10](../10-end-to-end-projects/index.md) | ⚠️ |
| 6 | **Memento-Skills** — deployed-agent skill self-evolution | The read-execute-reflect-write loop; skill discovery, routing, verification, write-back and version governance; getting better without touching the weights | [Chapter 12](../12-skills/index.md) | ✅✅ |

## One template for every project

Write each project's notes with this skeleton. If a section is empty, leave it empty — do not skip it:

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

## What the six projects have in common

They cover the **three sources of reward** (rules, models, environments) and the **two ways an agent gets better** (updating weights, or growing a skill library). Seen together, they form one engineering map of where agent capability actually comes from.

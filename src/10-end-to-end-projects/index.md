# Chapter 10 · End-to-End Project Cluster

> **Compute tier**: ⚠️ (a 0.6B model runs end to end; 8B needs QLoRA and small batches) | **Status**: partial - Project 4 written and measured, Project 5 outlined
> **Projects**: [Project 4 · DeepAnalyze](../appendix/projects.md), [Project 5 · OpenClaw-RL](../appendix/projects.md)
> **Why these are merged**: the source course splits them into two chapters with the same rhythm. Here they form one cluster, following a single storyline: **single-turn → multi-turn, synchronous → asynchronous**.

## First, the gist

This is where the theory becomes a working system. Project 4 takes a small (8B) model and trains it to do data analysis end to end: it writes code, runs it in a sandbox, looks at the output, and produces a report. Project 5 goes one level harder: an agent that calls tools over many turns, trained *while it is being used*, with the sampling, judging and training all running at once.

## What you'll be able to do

- Trace how a model, an inference server, a sandbox and a front end fit together into one agent product.
- Explain how a reward signal travels from a sandbox result back into the model's weights.
- Describe what makes asynchronous training hard (stale samples, weight sync, masking) and when it is worth it.

## Lessons

**Project 4 · DeepAnalyze (end-to-end agent RL)**

- [x] 1. The whole picture: architecture, capabilities and data flow
- [x] 2. Deploying the data-science agent: the executor, the image, and what it is allowed to be
- [x] 3. The marker protocol and the reasoning loop, with the two ways a correct program still loses
- [x] 4. SFT + RL + code sandbox: the cold start, and what the sandbox contributes to the reward

**Project 5 · OpenClaw-RL (online RL for multi-turn tool callers)**

- [ ] 5. How multi-turn agents should actually be trained: environment, data and the reward loop
- [ ] 6. OpenClaw-RL dissected: next-state signals, the four-ring asynchronous architecture, three core algorithms
- [ ] 7. Reading the core source: from sample to PPO loss to combined update
- [ ] 8. The tool-call RL pipeline: how rollout, actor and PRM join into one line (loss masks, session headers, weight sync, sample freshness)
- [ ] 9. A three-stage run on Qwen3-0.6B: ReTool-SFT → ReTool-RL → PRM process reward

### Lesson 1 · The whole picture: architecture, capabilities and data flow

Project 4 is small enough to hold in one head, and its shape is the shape of every agent product: a question, a table, a program the model writes, a sandbox that runs it, an observation, and an answer. What makes it a *project* rather than a demo is that each arrow is a component with a contract, and each contract is where a number can go wrong.

```
question + table.csv
      |
      v
  policy (served)  --<code>-->  sandbox (container, no network)  --stdout-->  policy (served)
      |                                                                              |
      +--------------------------<report>--------------------------------------------+
                                     |
                                     v
                             verifier: does the reported number equal the task's?
```

Four things travel along those arrows, and only one of them is text:

- **The prompt** carries the question, the column names and five rows. Not the table: a model that can see the whole CSV can answer without running anything, which would measure arithmetic instead of tool use.
- **The program** is the model's only real action. It is written to `prog.py`, next to the table, and executed with no arguments and no network.
- **The observation** is stdout and stderr plus an exit code and a timeout flag. It goes back into the conversation as text, which is the only way a language model can read a program's result.
- **The verdict** is not produced by the model at all. It is a comparison between the number in `<report>` and a number stored in the fixture, and that separation is the whole reason this project is trainable: the reward comes from an executor, not from a judge.

Measured end to end on the Spark on 2026-11-07, six tasks with Qwen3-1.7B served from a vLLM instance and a container as the executor: **5 of 6 solved, 7 code runs, 0 sandbox errors, 17.3 seconds of wall time** (`runs/p4-20261107-204926`, `code_sha git:951a165`). The same policy against a worse executor, twenty minutes earlier, produced **1 of 6 solved and 6 sandbox errors** (`runs/p4-20261107-190738`). Both numbers are real. Which one you publish depends on a single line in a Dockerfile, and Lesson 2 is about that line.

### Lesson 2 · Deploying the data-science agent: the executor, the image, and what it is allowed to be

There is no web UI in this project and no quantisation step, because the two lessons worth having here are about the executor. The policy is 1.7B parameters served by vLLM at temperature zero; the interesting decisions are all in how the program runs.

**The executor is a container with five limits.** `--network=none` (a program that cannot reach the network cannot download its way around the task), `--memory=512m --memory-swap=512m` (no swap, so the limit is a limit), `--cpus=1.0`, `--pids-limit=128` (a fork bomb is a crash, not a denial of service) and a wall-clock kill. The kill path matters: `docker run` in a subprocess with a timeout leaves the container alive unless it is named and explicitly killed, and an orphaned container holding a core is how one bad program becomes a slow machine.

**The image is part of the metric, and the first one was wrong.** The obvious choice is `python:3.12-slim`. Six tasks, six programs, and every one of them opened with the same line:

```
ModuleNotFoundError: No module named 'pandas'
```

All six runs exited 1, all six episodes were scored as "did not report the expected number", and the summary said **1 of 6 solved with 6 sandbox errors**. The one success was the model guessing a value after its program failed - which is exactly the failure mode a leaderboard would have recorded as a partial success.

The fix is four lines long (`p4-deepanalyze/sandbox/Dockerfile`): `python:3.12-slim` plus pandas and numpy. The same policy, the same six tasks, the same prompt, 15.7 seconds later: **4 of 6 solved, 0 sandbox errors**. Nothing about the model changed. The environment was scoring it.

That is the chapter's first transferable rule, and it is stronger than "install your dependencies":

> **A sandbox that cannot run the obvious program is not a neutral environment; it is a reward source that says "wrong" to every attempt.**

Which is why the record has a column for it. `sandbox_errors` is counted separately from `solved`, `timeouts` separately again, and a run whose sandbox errors equal its code runs is a run whose score should not be quoted at all. The summary in `runs/p4-20261107-190738/summary.json` says `solved: 1, sandbox_errors: 6`; a reader who looks only at the first number draws the opposite conclusion from the one the data supports.

**What "deployment" means at this size.** The service is vLLM (`--enable-lora`, 32 GB container cap) and the executor is Docker; the harness is a stdlib Python process that speaks HTTP to the former and `docker run` to the latter. On this machine that is a `docker run` and a `python3 run_p4.py`; the platform notes at the end of the chapter record what each piece needed on `sm_121` hardware, including the parts that do not work.

### Lesson 3 · The marker protocol and the reasoning loop, with the two ways a correct program still loses

A multi-turn agent needs a format, or the harness cannot tell a thought from an action. This project uses three markers, and the parser is strict on purpose:

| marker | meaning | what the harness does |
| --- | --- | --- |
| `<analyze> ... </analyze>` | optional reasoning | ignored; kept in the transcript |
| `<code> ... </code>` | run this program | writes `prog.py`, runs it, returns `<observation>` |
| `<report> ... </report>` | final answer | ends the episode, and is scored |

Exactly one action per reply. Zero actions ("I think the answer is 3"), two actions, an unclosed marker and an empty block are each refused with a *named* reason, and the episode stops. A parser that guesses - take the first block, strip the prose, coerce the answer - turns a format failure into a wrong answer, and those need different fixes: one is a data problem, the other is an infrastructure problem.

**Compliance was not the problem here.** The 1.7B model, asked with the system prompt in Lesson 1 and with Qwen's thinking mode off, emitted a well-formed action on **6 of 6 tasks, in both runs** - a hundred percent, at temperature zero. That is worth stating plainly because it is the opposite of what the previous chapters found for a 1.7B model on the p2 tasks (43 of 43 failures of a stricter format): markers that come in pairs and structures the model has seen in pretraining are easy; a bespoke action vocabulary with a grammar is not. Before you build a cold-start dataset, measure the compliance rate of the format you chose, because "small models cannot follow formats" is not a fact, it is a per-format measurement.

**Two failures, both informative, both in the run that "worked".** Four of six solved; the two that did not are worth more than the four.

*The program was right and printed nothing.* For "which month had the largest order", the model wrote a correct pandas program - read the CSV, find the max amount, take that row's month - and ended with the bare expression `max_month`. In a notebook that prints the value. In a script it evaluates and discards it. stdout was empty, the exit code was 0, and the model, given an empty observation, reported **March**: a guess, not the value its own program had computed. The sandbox saw no error. The record saw a wrong answer.

> A program can be right and silent, and a silent program is a lost turn. If the harness only checks exit codes, "produced no output" is invisible.

*The program answered a table where a number was asked for.* For the inventory shortfall the model computed `stock - reorder_level` (the sign the question did not ask for) across every item, and `print(result)` on a pandas Series produced four lines. The report took the first number it could see - `-80` - and the verifier, which reads the last number in the report, scored it against the expected **80**. Two errors compounded: a sign, and an aggregate that was never aggregated.

Both failures point at the same missing turn: **a silent or malformed observation should be a first-class outcome, so the model gets a chance to repair it.** That hypothesis was cheap to test, and it took three runs to settle - the table is the honest record of the harness changing, not the model:

| run | what changed in the harness | solved | lost turns | forced repairs | code runs |
| --- | --- | --- | --- | --- | --- |
| `p4-20261107-190831` | nothing: notes were not even printed | 4/6 | 0 (not counted) | 0 | 6 |
| `p4-20261107-204715` | the observation says "printed nothing" / "printed 4 lines", and the episode counts it | **4/6** | 2 | 0 | 6 |
| `p4-20261107-204827` | plus: the first report after a lost turn is refused once | **5/6** | 2 | 2 | 9 |
| `p4-20261107-204926` | plus: the prompt says every program is a new process, six turns | **5/6** | 1 | 1 | 7 |

Three things were learned, and only the first was the hypothesis.

**Telling the model is not the same as making it act.** With both warnings firing on exactly the two broken episodes, a 1.7B model read the note and reported anyway - 4 of 6, unchanged. A note in the observation is information; a refused report is a constraint. The constraint moved the month task from wrong to right.

**"Separate process" has to be said out loud.** The repair turn for the inventory task began with a continuation of the previous program - `... < df["reorder_level"]]` - and died with `NameError: name 'df' is not defined`, because every program runs in a fresh container with an empty working set. The model assumed a notebook. Saying it in the prompt removed the class of error, and the run of `-204926` shows the month task solved in two turns, with no repair at all.

**What survives is a real error.** The inventory task still fails, and now for a good reason: the program subtracts `stock - reorder_level` where the question asks how far *below* the reorder level the item is, and reports `-80` for `80`. No harness change fixes a sign error; that is the kind of failure a training signal is supposed to see, and it is a better failure to be left with than a missing library or an empty observation.

The budget is now part of the finding too. With four turns, the repaired task consumed the episode's whole allowance and the other one died of `no report within 4 turns` - the run of `-204827` has one sandbox error and nine code runs for that reason. Six turns and a clearer prompt recovered the efficiency: seven code runs, one forced repair, and the same score. A harness that pays for a repair has to budget for it.

### Lesson 4 · SFT + RL + code sandbox: the cold start, and what the sandbox contributes to the reward

The end-to-end story has three stages and this project has evidence for one of them.

**Stage 1 - the cold start (not run here).** The demonstrations have to come from somewhere, and the honest source is the oracle: for each of the six tasks a program that provably computes the answer (the same programs the gate runs in `checks/check_p4.py`). Supervising on *(question, oracle program, observation, report)* teaches the format and the habit of printing; the model already has the arithmetic. What to supervise matters: the code and the report, not the prose - a model trained on its own `<analyze>` text learns to write plausible reasoning, which is not the capability being tested.

**Stage 2 - the reward (built here).** The sandbox is the reward source, and the reward has exactly three outcomes, which the record keeps apart:

| outcome | signal | where it lives |
| --- | --- | --- |
| the report matches the task | +1 | `solved` |
| the report does not match | 0 | `solved: false` with `reason` |
| the program crashed or timed out | not a verdict on the answer | `sandbox_errors`, `timeouts` |

That third row is the one that matters for training. A crashed program is the *model's* fault - it wrote code that does not run - but it is not evidence about the answer, and folding it into "wrong" teaches from a signal the task never gave. In the run of Lesson 2, six of six programs crashed for a reason the model could not have known (a missing library); training on that run would have taught the policy to avoid pandas.

**Stage 3 - the RL update (not run here).** With the three verdicts in the record, the update is Chapter 6's machinery applied to a new reward: a group of attempts per task, an advantage from the group's spread, and a three-tier credit where a turn that produced a runnable program is positive even if the final answer is wrong. Two facts bound what this chapter can claim about it. The first is that the reward only has a gradient when the group's outcomes differ, and on six easy tasks a 1.7B model that solves four of them at temperature zero will produce groups with no spread at all - the same "0 of 60 rollouts" ceiling Chapter 11 measured, one project over. The second is that the training runtime is the one Chapter 9 could not start on this hardware, so the honest status of stage 3 is: designed, not run, with the missing piece named (`sm_121` rollout support) rather than described as a future improvement.

What can be claimed, and is enough for a first end-to-end system: **a policy, a server, a sandbox and a verifier are wired into one loop, the loop's numbers come from the executor rather than from a judge, and the record separates the three ways an attempt can end.** The measured difference between 1 of 6 and 4 of 6 belongs to the container, not to the model, and the chapter says so in the same sentence as the number.

## Platform notes (Spark)

| Step | Risk | What actually happened on the Spark (2026-11-07) |
|---|---|---|
| Starting vLLM | upstream images do not support sm_121 | `vllm/vllm-openai:v0.31.0` (arm64, cu13) runs and serves Qwen3-1.7B with LoRA; `/health` 200 after ~160 s |
| An older vLLM, for a pinned runtime | the 0.11.2 arm64 image ships torch 2.9.0+cu129, which supports compute capability 8.0-12.0 only | it refuses the GPU outright: `Found GPU0 NVIDIA GB10 which is of cuda capability 12.1` - see Chapter 9 |
| Official Docker images | mostly x86_64 | the vLLM and PyTorch images used here are arm64; PyPI's `vllm` has no aarch64 wheel at all |
| The executor image | `python:3.12-slim` has no pandas, and six of six programs asked for it | build `p4-deepanalyze/sandbox/Dockerfile` (pandas 2.2.3, numpy 2.1.3); 1 of 6 solved becomes 4 of 6 |
| Attention backend | FlashAttention has no sm_121 support | SDPA is the default and was never the bottleneck at this size |
| Building from source | high parallelism can exhaust the shared memory | not needed for either image; cap `MAX_JOBS` if you try |

The executor row is the one this chapter is about. The others are the same lesson at a larger
scale: an image is a decision about what the machine can do, and a run record that does not name
the image cannot tell you which decision produced its numbers.

## Deliverable

- Two reproducible end-to-end run logs, each with a list of failure modes encountered
- One paragraph of judgement on "synchronous vs asynchronous", with reasons and limits

## How you know you passed

- You can draw the full path from data to a deployed service for the 8B model, labelling what each step produces.
- You can explain why the loss mask and the session header change the gradients.
- At least two of the three 0.6B stages (SFT → RL → PRM) run to completion, with curves saved.

## Reading

- DeepAnalyze: <https://github.com/ruc-datalab/DeepAnalyze>
- OpenClaw-RL: <https://github.com/Gen-Verse/OpenClaw-RL>
- ReTool (cold-start SFT + tool-augmented RL): arXiv:2504.11536

## Backfill

- [ ] Question 3: How do I debug a long multi-turn failure?
- [ ] Question 4: Where does my training system quietly lie to me? (Stale samples and weight sync in async setups.)

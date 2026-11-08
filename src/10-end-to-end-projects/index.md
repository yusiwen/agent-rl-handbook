# Chapter 10 · End-to-End Project Cluster

> **Compute tier**: ⚠️ (a 0.6B model runs end to end; 8B needs QLoRA and small batches) | **Status**: written - both projects measured
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

- [x] 5. How multi-turn agents should actually be trained: environment, data and the reward loop
- [x] 6. OpenClaw-RL dissected: next-state signals, the four-ring asynchronous architecture, three core algorithms
- [x] 7. Reading the core source: from sample to PPO loss to combined update
- [x] 8. The tool-call RL pipeline: how rollout, actor and PRM join into one line (loss masks, session headers, weight sync, sample freshness)
- [x] 9. A three-stage run: ReTool-style SFT, outcome RL, and a process reward that taught the wrong lesson

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

**Stage 1 - the cold start (measured).** The demonstrations have to come from somewhere, and the honest source is the oracle: for each task a program that provably computes the answer, run in the sandbox so the observation in the demonstration is the stdout it really produces. Twelve supervised turns - `<code>` for each of six tasks, `<report>` for the value each program printed - with the prompt masked, LoRA rank 16 on the four projections, and nothing else.

Training on the six tasks it will be tested on would measure recall, so the evaluation uses a **held-out** set: the same six schemas over different rows (`tasks-heldout.jsonl`, regenerated by the same script from different CSVs), never seen in training. Three runs, same base model, same loop, same executor:

| policy | training | held-out solved | marker compliance | code runs |
| --- | --- | --- | --- | --- |
| base, no fine-tune | - | 5/6 | 100% | 7 |
| v1 | 12 epochs, lr 2e-4, final loss **0.0001** | 5/6 | **83%** | 5 |
| v2 | 4 epochs, lr 1e-4, final loss **0.076** | **6/6** | 100% | 6 |

Three findings, and the first two are about the score being the wrong instrument.

**Twelve demonstrations change behaviour, not just wording.** The base model fails both sets the same way - `stock - reorder_level` with the sign reversed, `-50` where `50` was asked for. After the cold start the held-out inventory task is solved, and its rows were never trained on: what transferred is a rule ("how far *below* the reorder level"), not a remembered number. That is the smallest honest version of "the cold start is what makes RL possible" - the loop now has a policy that sometimes succeeds, which is the precondition for a non-zero advantage.

**Training to loss 0.0001 costs the format.** The v1 model, asked the temperature question it had seen, reproduced the oracle program *verbatim* - and emitted it **without the `<code>` wrapper**, one long prose turn: `import csv ... print(round(...))`. The protocol refused it (`no marker`), correctly. Over-fitting taught the content of the demonstration and untaught its envelope; on the training set compliance fell to 67%, below the base model it started from.

**The trade is a knob, not a law.** Stopping at loss 0.076 keeps both: 6 of 6 on the held-out set, 100% compliance, six code runs, 8.8 seconds. The difference between the two fine-tunes is four epochs and a learning rate, and the only way to see it is to report compliance next to the score - a table with `solved` alone would have called v1 and v2 equal (5/6 and 6/6 differ by one task, but v1's failure was a format regression the score did not name).

What this is not: six held-out rows of the same six schemas is a schema-transfer test, not a benchmark, and 6 of 6 on six items is one task away from 5 of 6. The claim the chapter makes is the narrow one that survived three runs: a cold start on oracle demonstrations, sized in the tens of turns, moves a 1.7B policy from "fails on the sign" to "solves the held-out rows", and it will cost the format if it is trained to the end.

**Stage 2 - the reward (built here).** The sandbox is the reward source, and the reward has exactly three outcomes, which the record keeps apart:

| outcome | signal | where it lives |
| --- | --- | --- |
| the report matches the task | +1 | `solved` |
| the report does not match | 0 | `solved: false` with `reason` |
| the program crashed or timed out | not a verdict on the answer | `sandbox_errors`, `timeouts` |

That third row is the one that matters for training. A crashed program is the *model's* fault - it wrote code that does not run - but it is not evidence about the answer, and folding it into "wrong" teaches from a signal the task never gave. In the run of Lesson 2, six of six programs crashed for a reason the model could not have known (a missing library); training on that run would have taught the policy to avoid pandas.

**Stage 3 - the RL update (built, run, and it made the policy worse).** The runtime Chapter 9 could not start was a *framework* problem, and a single-machine loop does not need the framework: sample from the policy in process, score with the executor, take a policy-gradient step. That fits in one script, and it is split into three phases because three different machines are involved - `sample` needs the GPU, `reward` needs Docker (the training container has no socket), `update` needs the GPU again. Every program is re-executed in the real container before it becomes a reward, and the verdict that trains the policy is that one, not the subprocess verdict the rollouts were conditioned on.

The reward is deliberately blunt: **1.0** if the report matches, **0.2** if a program ran and the answer was wrong, **0.0** otherwise. The advantage is the group mean subtracted, with no division by the standard deviation - dividing would make `[0.2, 0.2, 0.2, 0.0]` look like a stronger lesson than `[1.0, 0.2, 0.0, 0.0]`, which is backwards, and `max(std, epsilon)` hides it rather than fixing it.

Then the measurement, which is the whole point. Twenty-four rollouts - six tasks, four samples each, temperature 0.8, from the cold-started policy:

| group | solved | spread | contributes |
| --- | --- | --- | --- |
| `sales-north` | 4/4 | 0.0 | nothing |
| `sales-top-product` | 4/4 | 0.0 | nothing |
| `orders-biggest-month` | 4/4 | 0.0 | nothing |
| `inventory-shortfall` | 4/4 | 0.0 | nothing |
| `students-pass-rate` | 3/4 | 1.0 | 4 samples |
| `temps-mean` | 2/4 | 1.0 | 4 samples |

**Four of six groups taught nothing, because the cold start had already solved them**, and the update ran on the eight samples that were left. Three step sizes, same rollouts, evaluated on the held-out set:

| update | held-out | code runs | sandbox errors |
| --- | --- | --- | --- |
| cold start, no RL | **6/6** | 6 | 0 |
| lr 1e-5, 8 steps | **6/6** | 6 | 0 |
| lr 5e-5, 8 steps | **4/6** | 16 | **12** |
| lr 5e-5, positive advantage only, 5 steps | 5/6 | 11 | 6 |

At 1e-5 the update is undetectable - the same six answers, the same six code runs. At 5e-5 it costs two tasks, and the transcripts say what it did: six consecutive programs failing with `AttributeError: 'DictReader' object has no attribute 'fields'`, then `'items'`, then `'keys'`, then `'next'`. The policy did not merely stop solving those tasks; it started inventing an API. That is what whole-reply REINFORCE does with a negative advantage: the sample that lost still contained a mostly correct program, and the gradient pushes *every* token of it away, `csv.DictReader` included. Keeping only the winners limits the damage to one task instead of two, so the sign is part of the mechanism - but not all of it: five steps at 5e-5 still cost a task, which says the step is too large for eight samples regardless of sign.

The honest conclusion is the one Chapter 11 already argued from a different direction: **eight informative samples cannot support a conclusion, and they cannot support an update either.** What this stage needs is not a bigger learning rate sweep. It needs headroom - a task set the policy does not already solve, so that all six groups have spread - and two orders of magnitude more rollouts per group, which is what the framework in Chapter 9 exists to provide.

What can be claimed, and is enough for a first end-to-end system: **a policy, a server, a sandbox and a verifier are wired into one loop, all four stages of the pipeline ran, and every number in this chapter came from executing code rather than from a judge.** The differences the chapter reports are attributable: 1 of 6 to 4 of 6 from the container, 4 of 6 to 5 of 6 from the harness, 5 of 6 to 6 of 6 from twelve supervised turns, and 6 of 6 back to 4 of 6 from one badly sized policy-gradient step.

That last one is the honest end of the project rather than a failure to hide. A pipeline that can be measured well enough to show its own update hurting is more useful than one that reports an improvement it cannot attribute - and the reason it hurt is legible in a single line of the record: four of six groups had no spread, because the cold start had already solved them. The next experiment this project needs is a harder task set, not a better trainer.

### Lesson 5 · How multi-turn agents should actually be trained: the world, the traces, the reward

Project 4's agent ran one program and answered. Project 5's agent calls tools, reads what they return, and decides again - and that difference changes what the training machinery has to be.

**The world owes you six things.** Reset must be free (otherwise you are measuring environment setup), every step must return an observation the model can read, the episode must end for a reason the record can name, every task must be solvable so a zero reward means the policy failed rather than the task being impossible, the tools must be the real ones the product uses, and the oracle must prove the last two by solving every task itself. This project's world is six tasks over five documents with four tools (`search`, `open`, `calc`, `submit`), a reset measured in microseconds, and an oracle that opens the documents the task names, computes when the question needs arithmetic, and submits.

**Two of those six are usually skipped, and both cost you the measurement.** A task set that is not provably solvable turns a broken fixture into a policy regression, and a world whose reset is slow makes every staleness number about your filesystem instead of your loop.

**The task set needs headroom, and the demonstrations must not fill it.** The cold start is trained on six of the nine tasks - the ones a trace can plausibly teach - and the loop trains on all nine. The three held-back tasks need two or three documents and a calculation. The gate asserts that split (`test_the_demonstrations_cover_six_tasks_and_leave_three`), because the headroom is the only thing that can make an update non-zero, and it is exactly what a helpful colleague deletes first when the training data looks thin.

**The demonstration is a trajectory, not a transcript of one.** Twenty supervised turns, each the state before a call and the call itself, with the observation the world actually returned. Same discipline as Project 4: build them by running the oracle, and let `--check` fail when the world changes underneath them. What they teach is the format (one JSON object per reply, or the protocol refuses it) and the habit (submit the value the question asked for). What they cannot teach is search strategy - six traces do not contain a policy for "which document do I need", which is exactly the part stage 2 is supposed to learn and, on this world, could not.

**The reward has to name the failure, not just count it.** Four outcomes are recorded separately, because they are four different problems: a **protocol error** (the reply was not a tool call), a **tool error** (the call was malformed or referred to a document that does not exist), a **wrong answer**, and a **turn budget exhausted**. Lesson 9 shows what happens when they are collapsed: a policy that learned to call tools forever and never submit looks identical to one that is thinking hard.

### Lesson 6 · OpenClaw-RL dissected: next-state signals and the four rings

The reference design this project is named after is an *asynchronous* agent trainer, and its shape is worth understanding even where this repository implements only part of it. Four rings:

| ring | what it does | where it blocks |
| --- | --- | --- |
| serving | holds the weights and generates | on the model |
| rollout | plays episodes against the tools | on the model and the tools |
| judge | scores outcomes (and, in the third algorithm, partial traces) | on the judge |
| trainer | consumes samples and produces weights | on the GPU |

In a synchronous loop these four run one after another: generate a batch, score it, train, publish, repeat. Every ring waits for the one before it, so the GPU is idle while the model generates and the model is idle while the GPU trains. The asynchronous architecture lets rollout and training overlap, and pays for it in one currency: **the samples a trainer sees were produced by a policy that is no longer current.**

That is why the reference design's central data structure is not the trajectory but the **transition** - the state, the action, and the next state. A trajectory-level trainer has to hold a whole episode before it can compute anything; a transition-level one can train on the part of the episode that has already happened, which is what makes overlap possible and what makes the staleness bookkeeping mandatory.

The three algorithms in the outline are three answers to "what is a reward here":

- **outcome RL** (ReTool-style): the episode ends, the answer is checked, the reward is 1 or 0. Cheap, blunt, and only as good as the variance in the task set;
- **a process reward**: a model (or a rule) scores partial traces, so a step that moved toward the answer is rewarded before the answer exists. This is the algorithm that can, in principle, learn from failure - and Lesson 9 shows the version that learns the wrong thing;
- **asynchronous PPO**: the same outcome reward, with the ratio computed against the behaviour policy that actually produced each sample, and a staleness bound so the ratio does not describe a policy from three updates ago.

What this repository implements, measured, is: the rollout ring, the trainer ring, the judge as a rule, weight sync over HTTP, and staleness recorded per sample. What it does not implement: a learned process reward model, a separate judge service, and more than one machine. The chapter says so rather than borrowing the reference design's numbers.

### Lesson 7 · Reading the core source: from sample to masked loss to combined update

The most useful thing to learn from any agent-RL implementation is *where to look*, because the same four decisions are made in every one of them, and each is a place where a bug hides silently.

**1. Where the mask is applied.** In this repository, `_turn_losses` builds `labels = [-100] * len(prompt_ids) + reply_ids`. Everything before the assistant's own tokens is -100, so the loss covers the reply and nothing else. A trainer that supervises the prompt learns to reproduce the prompt: on this project that means a model that writes the system prompt back instead of calling a tool, and the loss curve looks *better* while it happens.

**2. Where the baseline comes from.** `train_step` groups the batch by `task_id`, computes the mean reward per group, and subtracts it. That is the whole of GRPO's idea, and the failure mode is group size: a group of one sample has mean equal to itself, so its advantage is zero and the sample is silently discarded. The loop records `singleton_groups` per round for exactly this reason - asynchronous workers produce them constantly (two to four per round here), and a run that "trains" on nothing looks no different from one that trains.

**3. Where the version is stamped.** Every sample carries the adapter name that generated it (`p5-v3`), and `credit.partition` converts that into an age against the current one. Without the stamp, staleness is unknowable and the run is uninterpretable; with it, `--stale-bound` can drop the too-old samples *and count them*.

**4. Where the weights are published.** `load_adapter` posts the new adapter to the server; workers pick up the new name on their next episode. This is the cheapest possible weight sync - an adapter is tens of megabytes, and the base weights never move - and it is also where this project's three silent failures lived (Lesson 8).

The reading exercise, in one line: find the sample record, find the mask, find the baseline, find the version, find the publish. If any of the five is missing from the *record* (not from the code), the run cannot be debugged later, and the person debugging it will be you.

### Lesson 8 · The pipeline measured: throughput, freshness, and three bugs that looked like health

Same nine tasks, same starting adapter, four rounds, 72 samples per run, one machine, the served model on the same GPU as the trainer:

| run | workers | samples / min | sample staleness (mean / max) | samples with an advantage | dropped |
| --- | --- | --- | --- | --- | --- |
| synchronous | 1 | 36.1 | 0.04 / 1 | 0 | 0 |
| asynchronous | 1 | 35.9 | 0.06 / 1 | 2 | 0 |
| asynchronous | 3 | **108.2** | 0.18 / 1 | 3 | 0 |

<figure class="book-figure">{{#include ../figures/10-p5-bimodal-and-throughput.svg}}</figure>

*Figure 10.1 - the two measurements Project 5 is built on. On the left, each task's eighteen sampled episodes: seven solved every time, two never, and none in between - which is why the outcome loops found three samples worth training on out of seventy-two. On the right, the same four rounds under three configurations: the two one-worker runs measure almost the same throughput, so the third bar is three workers rather than the reward of overlap.*

**Asynchrony on its own bought nothing.** At one worker the two modes are within 0.2 samples per minute of each other, because the bottleneck is the served model, not the trainer: the trainer's step takes a couple of seconds, the generation it waits for takes thirty. What the third row shows is not the value of overlap but the value of *permission* - once sampling does not block training, you can run three workers, and the throughput triples.

**The staleness bound never bit, and that is the honest reading.** Maximum age stayed at 1 in every run: a worker completes an episode in about a second and a half, a round takes about thirty, so at most one version boundary is crossed. With slower tools - a container, a browser, a real API - the same configuration would produce ages of three or four, and the bound would start discarding. The number to publish is not "staleness of at most 1 is fine"; it is "this world is too fast to make staleness expensive, and here is the measurement that shows it".

**Three bugs, all of which produced a healthy-looking run:**

- *The starting adapter was not served.* Workers got 404s, died in their threads, and the loop waited for samples that never came. The summary said `rounds: 0` - which reads like "the run did nothing", not like "the run could not start". Fixed with a preflight that lists the served models and refuses to start.
- *The served path was hardcoded* (`/models/p5-v<N>/adapter`) while the save path followed the run's prefix. A run with its own prefix published its weights to one directory and loaded another experiment's from a different one - and the eval still returned a number, because a LoRA adapter of the same rank loads without complaint.
- *`load_inplace: true` left the registry name unregistered.* The server accepted the load, replaced the weights, and kept serving the old name; `/v1/models` stopped growing, and every version after the first pointed at nothing. Fixed by unloading and loading, which is one extra round trip and no ambiguity.

The general lesson is the one this book keeps arriving at from different directions: **a quiet loop looks exactly like a working one.** Throughput, loss and wall clock were all normal in every one of these failures. What caught them was a record that names the served model, the saved path and the version - and a check that fails when they disagree.

### Lesson 9 · A three-stage run: SFT, outcome RL, and a process reward that taught the wrong lesson

**Stage 1 - supervised fine-tuning.** Twenty demonstrations, 30 steps, LoRA rank 16 on 0.372% of the parameters, loss 2.74 → 0.0008 in 47.8 seconds, peak 22.8 GB. Greedy evaluation on all nine tasks: **7 of 9 solved**, one protocol error, 31 steps, 13.3 seconds.

**Stage 2 - outcome reinforcement.** Three loops (synchronous, asynchronous with one worker, asynchronous with three) on 72 samples each, and the reward found almost nothing to work with. The reason is in the per-task counts, and it is the cleanest statement of a rule this book has now made three times in three costumes:

| task | solved | why |
| --- | --- | --- |
| seven tasks | 8/8 | the cold start generalises: same skill, different numbers |
| `total-revenue` | **0/8** | one wrong program, reproduced every time |
| `units-gap` | **0/8** | one protocol error, reproduced every time |

**A task solved every time and a task failed every time contribute exactly the same amount to an update: nothing.** Variance is the resource. A deterministic failure is not a signal; it is a constant, and constants have no gradient. The loops recorded 0, 2 and 3 samples with a non-zero advantage out of 72, and the evaluation after each was unchanged at 7 of 9 - which is what a nearly-zero update should produce.

**Stage 3 - a process reward.** Same loop, same samples, with step-level credit: +1 for the answer, +0.1 per valid tool call (up to three), -0.3 per tool error. The number of samples carrying an advantage jumped from 3 to **21** - a real effect, and the mechanism the reference design is after: a failure can still teach something if its *steps* are scored.

The policy then failed the evaluation in a new way: **3 of 9 solved, 28 tool errors, 5 episodes out of turns, 44 steps** where the baseline used 31. It had learned to call tools. Each valid call paid 0.1 whether or not it moved toward the answer, the answer paid 1.0 and was rarely reached, so the gradient pointed at tool calls - and the episodes spent their budget crawling through documents and never submitting.

That is not a bug in the loop; it is a bug in the reward, and it is the most useful result in this project:

> **A process reward that pays for valid steps without a hard requirement on the outcome is a licence to farm steps.** The step term has to be a shaping term, not an objective.

Two honest limits before leaving it. First, this process reward is a *rule*, not a learned model - the reference design's PRM is scored by a reward model trained on partial traces, and a learned one would not pay for a useless call. Second, nine tasks and four rounds is a demonstration of the machinery, not a measurement of a method: the numbers above say what this pipeline did, and nothing about what a PRM would do at scale.

What the three stages did establish, and what this project owes the reader: a multi-turn agent can be trained end to end on one machine, with generation out of process, weight sync over HTTP, staleness recorded per sample, and every number in the record recomputable from the samples. The interesting failures were in the environment, the reward and the adapter registry - never in the idea.

## Platform notes (Spark)

| Step | Risk | What actually happened on the Spark (2026-11-07) |
|---|---|---|
| Starting vLLM | upstream images do not support sm_121 | `vllm/vllm-openai:v0.31.0` (arm64, cu13) runs and serves Qwen3-1.7B with LoRA; `/health` 200 after ~160 s |
| An older vLLM, for a pinned runtime | the 0.11.2 arm64 image ships torch 2.9.0+cu129, which supports compute capability 8.0-12.0 only | it refuses the GPU outright: `Found GPU0 NVIDIA GB10 which is of cuda capability 12.1` - see Chapter 9 |
| Official Docker images | mostly x86_64 | the vLLM and PyTorch images used here are arm64; PyPI's `vllm` has no aarch64 wheel at all |
| The executor image | `python:3.12-slim` has no pandas, and six of six programs asked for it | build `p4-deepanalyze/sandbox/Dockerfile` (pandas 2.2.3, numpy 2.1.3); 1 of 6 solved becomes 4 of 6 |
| Attention backend | FlashAttention has no sm_121 support | SDPA is the default and was never the bottleneck at this size |
| Building from source | high parallelism can exhaust the shared memory | not needed for either image; cap `MAX_JOBS` if you try |

| Serving adapters for a loop | every version needs a name the server knows, and `load_inplace` keeps the old one | unload then load; verify with `GET /v1/models` before starting the loop (`p5-openclaw-rl/env/loop.py` preflights this) |
| Weight sync | a full merged restart costs minutes | publishing an adapter costs one round trip (~16 ms measured in Chapter 9) and the base weights never move |

The executor row is the one that changed a score. The others are the same lesson at a larger
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

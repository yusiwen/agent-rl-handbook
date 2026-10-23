# Chapter 8 · Environment Engineering and Rollout Infrastructure

> **Compute tier**: ✅ (a laptop runs the simulator in this chapter; one GPU host runs the container measurements) | **Status**: draft — all six lessons have prose, and the pool, quota, isolation, reset and cost numbers in them were measured on 2026-10-20 on one machine (a DGX Spark: aarch64, Docker, a cached `python:3.13-alpine` image), with the commands written into the lessons; what remains open is named at the end of Lesson 6, and it is the reason this is a draft rather than a claim.
> **Why this chapter exists**: in agent RL, the bottleneck is usually the *environment*, not the algorithm.

## First, the gist

The "environment" is whatever the agent acts on: a sandbox, a terminal, a database, a website. Training means running the agent against it thousands of times, so the environment has to be four things at once - fast to start, easy to reset, safe to run in parallel, and honest about what happened. When training looks broken, the cause is often one of those four properties failing while the algorithm is blamed for it.

The reason this chapter sits between data (Chapter 7) and the training runtime (Chapter 9) is arithmetic. One rollout is a model call plus a handful of environment steps. On the machine this chapter was measured on, a step in a pure-Python simulator costs **3.2 microseconds** and a call to a 1.7B model costs **141 milliseconds** - a factor of about 44,000. That is the good case, and in it the environment is free and the model is the bottleneck. Put a Docker container inside the step and the same measurement becomes **60 to 234 milliseconds**, which is the same order as the model call. At that point the environment is no longer overhead; it is half of your training budget, and its failure modes are half of your debugging.

That single ratio is why this chapter exists, and it is where the measurements start.

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

### Lesson 1 · The environment as an interface: abstraction, reset semantics, observation format

An environment is five things written down, and a project that skips one of them pays for it later:

1. **State** - what exists in the world, in a form you can inspect and compare. Not "the agent's belief about the world": the world.
2. **Action space** - the set of things the agent may do *right now*. In a terminal environment this is "any command", which is why the interesting question is not the action space but the *observation*: what the agent is allowed to see before acting.
3. **Transition** - what happens when an action is applied, including the parts that fail. A transition that silently succeeds when it should have failed is worse than one that crashes.
4. **Observation** - what the agent sees after the transition, and *only* that. This is where most designs leak: the observation quietly includes the answer, the hidden test output, or the whole filesystem.
5. **Done, and a signal** - how the episode ends, and what the environment can say about the result. It is fine to say "no signal here, the checker owns it"; it is not fine to leave it undefined and let the trainer guess.

Our project 2 environment is a small, deliberate instance of this, and the interface fits on one screen: `start(task_id)` returns an attempt; `attempt.menu()` is the finite set of actions allowed this turn; `attempt.step(action)` applies one and returns the observation plus a done flag; `attempt.solved()` is the goal check; `attempt.progress()` is how far along the attempt is, and `attempt.signature()` is a digest of the state used for logging and for the update's bookkeeping.

**Reset is four different operations, and they cost different things.** This is the part most people collapse into one word:

| reset kind | what it clears | what it does *not* clear | measured cost |
| --- | --- | --- | --- |
| new task instance | the world, the turn counter, the history | the process, imported modules, caches | **1.6 us** (pure Python) |
| new process | the interpreter, module state, environment variables | the filesystem, the container image | tens of milliseconds |
| new container | the filesystem, processes, network namespace | the host kernel, cached images | **234 ms** (warm image) |
| new host | everything | the registry | minutes |

Choosing a reset kind is a real design decision with a price attached, and Lesson 2 measures the last two. The rule that keeps it honest: **the reset must be the same operation for every attempt, including the first one.** A first attempt that starts from a constructor and later attempts that start from a reset path are two different environments, and the difference will show up as a mysterious first-round effect.

**The observation format has one hard constraint that is easy to miss.** The text the environment returns is *not* produced by the policy, so it must never be supervised: mask it out of the loss, or the model learns to generate its own observations. Chapter 6, Lesson 6 and Chapter 2, Lesson 3 both make this point from the model side; here it is an environment requirement, because the environment is what decides how much of the context is its own text.

The last interface rule is the one this project learned by breaking it. In the first version, the `files` family shared one action menu across tasks, and the delete task's `rm tmp.log` was simply not in it. The loop ran, the reward stayed at zero, and the run looked exactly like a model that could not learn. It was an environment that could not express the solution. The fix was to generate the menu per task, and the gate now asserts that every task is solvable from its own menu. **If the environment cannot express the solution, the failure will present itself as a learning failure** - and you will spend a day tuning a learning rate before you check the menu.

### Lesson 2 · Sandbox pools, concurrency and quotas

Once an environment is a container, three questions become measurable and none of them can be answered by reading documentation: what does a container cost, how well do they run side by side, and what happens when one misbehaves.

Measured on one DGX Spark, Docker, image `python:3.13-alpine` (55 MB, already cached), 2026-10-20. Every command below is the one that produced the number.

**Start cost.** `docker run --rm python:3.13-alpine python -c pass`, five times, after one warm-up run: **239, 226, 237, 224, 234 ms**. Median **234 ms**. That is a cold *container* on a warm image; the first ever run of an image additionally pays for unpacking, and a remote registry adds the pull.

**Concurrency.** N containers in parallel, each running `sleep 0.5`:

| N | wall time | speedup vs serial (N x 740 ms) |
| --- | --- | --- |
| 1 | 740 ms | 1.0x |
| 2 | 743 ms | 2.0x |
| 4 | 881 ms | 3.4x |
| 8 | 1016 ms | 5.8x |
| 16 | 1424 ms | 8.3x |

Sixteen containers finish in 1.42 seconds where sixteen in a row would take 11.8. The marginal cost of one more container on this machine is about **46 ms** - that is the number to put in a capacity plan, not the 234 ms start cost, because the starts overlap.

**Quotas, and what a violation looks like.**

| quota | command | result |
| --- | --- | --- |
| memory 64 MB, allocating 256 MB | `docker run --rm --memory=64m ... python -c "bytearray(256*1024*1024)"` | exit **137** (killed), not a wrong answer |
| 2 CPUs vs 0.5 CPU, same loop | `--cpus=2` / `--cpus=0.5` | **421 ms** vs **630 ms**; subtract the 240 ms start and it is 181 vs 390 ms, i.e. **2.15x** - the quota behaves as advertised |
| wall clock, 3 s, container sleeping 60 s | `timeout 3 docker run --rm ...` | return code **124**, and **zero** containers left behind |

Two details in that table are worth more than the numbers. First, the memory cap produces **exit 137**, a code you can branch on; a sandbox that instead returns an empty output and a zero reward turns a quota violation into a training signal. Second, the CPU measurement only makes sense after you subtract the start cost - 421 against 630 looks like a 1.5x effect and is really the 2x the flag promises. This is the general shape of environment measurement: **the thing you meant to measure is usually smaller than the thing you measured it with.**

**Isolation, checked rather than assumed.** With `--network=none`, a connection attempt to `1.1.1.1:53` returns in **0.04 s** with a failure - it is blocked, not slow. A file written to `/tmp` in one container is absent in the next. Filesystem and network isolation are the two properties a pool depends on; both are one flag away and neither is on by default for the filesystem case (a shared bind mount would erase the second result, which is precisely why the mount list belongs in the sandbox definition, not in the caller).

**Reuse versus fresh.** Ten attempts as ten fresh containers: **2788 ms** (279 ms each). Ten attempts as ten `docker exec` calls into one container: **597 ms** (60 ms each). Reuse is **4.7x cheaper**, and it is the wrong default unless reset is real: a reused container carries the previous attempt's files, environment and processes. The honest pattern is a small pool of long-lived sandboxes with an explicit reset between attempts, and a recycle policy that throws a sandbox away after k uses or any suspicious failure.

**The circuit breaker.** Whatever the pool manager is called, it needs four things and they are all consequences of the table above:

1. **A per-attempt timeout** in the driver, not inside the container. Return code 124 and a *named* failure beat a job that hangs until the queue notices.
2. **A failure taxonomy** - timeout, OOM (137), crash (non-zero), unparseable output - because each one has a different remedy, and "the environment failed" as a single bucket tells you nothing.
3. **Quarantine with backoff.** A sandbox that fails twice is not retried immediately; it is drained, destroyed and replaced. The pool's health is the number of *consecutive* failures per sandbox.
4. **Backpressure.** When the queue outruns the pool, the correct behaviour is to slow the producer, not to start 200 containers and let the kernel's OOM killer pick winners.

**What a missing cap costs.** *(measured, 2026-10-22)* A training loop in this project held one
autograd graph per training pair and kept about 70 GB alive until a single backward. The host has
unified memory, so the trainer, the inference server and the ssh session were all drawing from the
same pool. `earlyoom` was running with a preference list naming `python3` and `vllm`, and it did fire -
at two percent of memory left, thirteen seconds after the kernel had already begun killing. The kernel
chose session processes first, because systemd gives them `oom_score_adj` 100 to 200 while a trainer
runs at 0: the login session died, and the model server was killed alongside it. Capping the container
at 48 GB (`--memory=48g --memory-swap=48g`, equal values meaning no swap) turns the same mistake into
one dead process group. Verified rather than assumed: a container capped at 1 GB that allocates 2 GB
exits 137, and the host's free memory does not move.

None of this is specific to Docker. It is the same list you would write for a pool of VMs, a pool of remote browser sessions, or a fleet of API-backed environments. Docker just makes the costs small enough to measure in an afternoon.

### Lesson 3 · Reproducibility: controlling randomness, rolling state back, task seeds

A run is reproducible when someone else can obtain the same *setup* and the same *distribution of outcomes* - not necessarily the same digits. That distinction is what makes this lesson tractable: chasing bit-identical GPU kernels is a different, much harder project than knowing where your variance comes from.

There are five sources of randomness in an agent rollout, and each has its own remedy:

| source | remedy | what it costs |
| --- | --- | --- |
| task sampling | a seed for the sampler, recorded in the header | nothing |
| environment randomness | a seed inside the environment, derived from the task seed | nothing, if the environment is written to accept one |
| model sampling | temperature, top-p, and **a seed if the server supports one** | a seed you may not have |
| image and dependency versions | digest, not tag: `@sha256:...` | nothing, and it is the difference between "same environment" and "same environment, probably" |
| host clock, network, scheduling | eliminate from the reward path; record the rest | design time |

This project has a measured example of the fourth row's absence and the third row's cost. The same step-A run - one task, five samples, two rounds, same code, same configuration, `temperature 1.0` - was run twice and produced **4/5 then 5/5** the first time and **5/5 then 3/5** the second. Nothing about the loop changed; the sampler has no seed, so the group is drawn fresh each round. The lesson is not "seed everything" but the narrower, more useful one: **name the non-determinism, in the same place you report the number.** A record that says `temperature 1.0, no seed` is honest; one that says nothing invites a reader to assume the two runs are comparable sample by sample.

There is a second, uglier failure mode in this category, and this repository hit it: **identities made of timestamps collide.** Run directories were named with a second-resolution UTC stamp; two runs that started in the same second wrote into the same directory, the second header overwrote the first, and a whole configuration's data was gone - not corrupted, gone, which is the quietest kind of data loss. The fix was a `_free_run_dir` that appends `-2`, `-3` and so on, plus a test. The general rule: **an identifier must be unique by construction, not by luck.** Timestamps are not identifiers; they are labels that happen to look like one most of the time.

The positive half of reproducibility is the header. Every run in this project writes `env_version`, `config_hash`, `code_sha`, the task set and a sampling budget before the first round: a number without a header is a number with no owner. That is not bureaucracy - it is the difference between "our solve rate went up" and "our solve rate went up, at this revision, on this task set, with these flags", which is the only form of the claim someone else can check.

### Lesson 4 · Feedback stability: noisy environments, unreliable negative samples, partial success

There are three kinds of noise in a rollout, and they need to be separated before any of them is blamed on the algorithm:

1. **Environment noise** - the same action on the same state does something different. Usually a race, a network call, or a shared resource.
2. **Grader noise** - the environment is deterministic and the score still moves, because the thing grading it is not.
3. **Policy noise** - the same state, the same policy, a different sample. This is the one that is supposed to be there.

The reason to separate them first is that the remedies are unrelated, and the third one is not a problem at all. The rule that makes the separation cheap: **a reward that changes while the state does not is a bug.** Log the state digest beside every reward; if two attempts reached the same digest and got different scores, you have found either 1 or 2 and can skip the hyperparameters entirely.

Project 2 had a clean instance of 2, and the cleanliness is the point. The environment is deterministic: the same action on the same state produces the same next state, bit for bit, and the rule critic is a pure function of the trajectory. So when an 8B judge was added as a second critic and disagreed with the rule **106 times across 35 judged episodes**, the disagreement could only be grader noise - and it could be inspected rather than averaged. It turned out to be systematic: the judge scored "inspect the state before writing" as 0.0, on turn after turn, while the rule scored the same turn 0.5. A noisy grader is annoying; a *systematically* wrong grader is a lesson. It was removed from the reward and kept as a diagnostic, which is the correct response to a signal that disagrees with ground truth more than half the time: not a smaller weight, but a different job.

**Partial success is the other half of stability.** A binary reward at the end of a multi-turn episode throws away everything the episode knew: which turns made progress, whether the failure was early or late. Project 2's first attempt at using that information was a per-turn activity score - 0.8 for "something changed" - and the policy learned to do harmless things until the turn cap stopped it, reproducing Chapter 6's Echo Trap without anyone writing it down. The second attempt averaged the turn scores, and padding beat acting. The third used the *maximum progress reached*, and in the ledger family the distance-to-goal metric paid for the distractor (`add 5` landed nearer to 7 than the correct `add 12` did). Three failed designs before one that worked, all three caught by looking at what the policy actually did rather than at the reward curve.

The stability check that catches this class in one number is the **same-sample comparison**: score the same sample ids in the next round and see whether the scores move. In this project's records, `runs/p2-loop-rule-only-turn-6` re-scores **129 of 129** sample ids across three rounds; if those numbers had drifted while the states did not, the reward would have been the suspect, not the policy.

### Lesson 5 · Lessons from terminal/tool environments

Terminal and tool environments look like a special case and are better understood as the *general* case with the cost of a step turned up. Five properties matter, and each has a design consequence.

**Long horizons, cheap actions.** An episode can be sixty turns and still touch the filesystem three times. The ratio of tokens to state changes is enormous - in this project's episodes, two or three actions among a hundred-odd tokens - and the consequence is the one Chapter 6 opens with: the thing that decided the outcome is a tiny fraction of what was generated. Mask the observations, credit the actions, and do not expect a trajectory-level score to find the three tokens that mattered.

**Discrete interactions.** A tool call is a jump: arguments in, a process runs, an observation appears. There is no gradient through the jump, and the observation was not written by the policy. Two mechanical rules follow: the reward must be a function of the *result* of the jump (the next state), and the loss mask must exclude the observation (Chapter 2, Lesson 3; Chapter 6, Lesson 6).

**A small set of common failures, and a long tail.** In a container sandbox the top five are worth memorising because each has a distinct signature: timeout (124), out of memory (137), crash (non-zero exit plus a stack trace), missing tool (127, "command not found"), and partial write (exit 0, a file with half the bytes). The long tail - wrong working directory, locale, a network that is blocked when the task needed it - is why an environment needs to record the *command, the exit code and the output* on every step rather than a summary.

**The environment can lie in ways that look like learning.** Three examples, all from this project, all found by replaying a single failing episode by hand rather than by tuning anything:

- a shared menu that omitted the correct action, so the task was impossible and the loop "failed to learn";
- a progress metric that paid for the distractor, so the policy "learned the wrong thing";
- an activity reward that paid for doing something, so the policy "learned to be busy".

Each of the three had a plausible algorithmic explanation - bad credit assignment, wrong learning rate, exploration collapse - and each was an interface bug. **Before you touch a hyperparameter, replay one failing episode by hand.** It takes ten minutes and it has a better hit rate than a sweep.

**Determinism is a feature you build, not a property you inherit.** A terminal environment with a filesystem, a clock and a network is non-deterministic by default; making it deterministic means fixing the clock (or removing it from the reward path), pinning the image, isolating the network, and giving the sandbox a fresh filesystem per attempt. Every one of those costs something in realism; every one of them buys something in debuggability. The trade is yours to make, but it should be a decision, not an accident.

### Lesson 6 · Rollout throughput and cost accounting

This is the lesson the chapter exists for, and the figure is the whole argument:

<figure class="book-figure">{{#include ../figures/08-rollout-cost.svg}}</figure>

*Figure 8.1 — what one unit of work costs, log scale, measured on one machine on 2026-10-20. The bars are the four environments this chapter discusses: a pure-Python simulator, a command inside a reused container, a fresh container, and a model call.*

The numbers behind the bars, with their sources:

| unit of work | measured | how |
| --- | --- | --- |
| simulator reset | **1.6 us** | 20,000 `start()` calls, `time.perf_counter` |
| simulator step | **3.2 us** | 20,000 episodes of `start()` + one `step()` |
| command in a reused container | **60 ms** | 10 `docker exec` calls |
| fresh container start | **234 ms** | 5 warm-image `docker run`, median |
| 1.7B model call, served over HTTP | **141 ms** | 1,336 calls over 189 s, 387 episodes |
| in-process 1.7B generation | about **180 ms** per turn | 27 s per round of 60 rollouts, about 150 turns |
| one training round (60 rollouts, 6 tasks x 10) | **27 s** rollout-only, **61 s** with three update epochs | `runs/p2-rl-step-2` |

**The formula is boring, which is why it is worth writing down.** One training step costs

```
time = R x (T x t_model + S x t_step) + t_update
```

where R is rollouts per step, T is turns per rollout, S is environment steps per turn, and t_update is the trainer's own time. Every term is measured, and the ratio between the two inside the bracket decides your architecture. With a simulator, `S x t_step` at 3.2 us is noise; with a container per step, `S x t_step` at 60 to 234 ms is the same order as `t_model` at 141 ms, and the environment is half the bill.

**What that means in practice.** For this project's step C - sixty rollouts per round, six tasks, ten samples - the rollout phase took 27 seconds and a step with three update epochs took 61. That is about **8,000 rollouts per hour** on one model instance, or 0.45 s per rollout. A ten-thousand-rollout batch is about 75 minutes of wall clock, and the arithmetic that matters is not the total but the split: at 0.45 s per rollout with two to three model turns, roughly 90% of it is the model and roughly 10% is everything else. Change the environment to one container per step and that 10% becomes the majority.

**GPU-hours are the second conversion, and it is where people fool themselves.** A GPU-hour is wall time x devices, so a single-GB10 run of 75 minutes is 1.25 GPU-hours, and the same batch on eight GPUs with perfect parallelism is 1.25 GPU-hours divided by eight *only if* the model server is not the bottleneck. In this project it is: the served model held 1.7B in about 7 GB of VRAM and answered 1,336 calls in 189 seconds, which is 7 calls per second. A pool of 16 sandboxes can produce work faster than that model can score it, so the correct scale-up is a second model instance, not a bigger pool.

**And the honest limit.** Everything above was measured on one machine, with one image, one model size and one simulator. The *shape* of the result - simulator steps are free, container steps are not, and the model dominates until a tool call appears in the loop - is the transferable part; the constants are not. A cluster with a warm image cache and a local registry will have a smaller start cost than 234 ms; a sandbox that boots a kernel per attempt will have a much larger one. Measure your own three rows before you quote anyone else's.

## Deliverable

A minimal environment that is concurrent, resettable and reproducible. In this repository that exists as three concrete things:

- **A Dockerfile with pinned dependencies.** `p2-reagent/env/Dockerfile` pins the base image by digest (`nvcr.io/nvidia/pytorch:26.09-py3@sha256:...`) and adds exactly two packages; `build.sh` prints the resulting image id and size, so "which image did this run on" has an answer.
- **A reset script plus a documented task-seed convention.** The simulator's reset is `env.start(task_id)` (1.6 us) and the seed lives in the task generator (`checks/make_tasks.py`, fixed seed, 43 tasks committed as a fixture). The convention, stated once: the task set is a file, not a random draw at run time; a run that needs a different set generates a different file and records its digest.
- **A throughput benchmark: time and GPU-hours per rollout batch.** This chapter's Lesson 6 table, reproducible with the commands in Lessons 2 and 6, and anchored to `runs/p2-rl-step-2` for the training-round figure.

## How you know you passed

- **The same task seed gives the same trajectory twice, or you can name exactly where the non-determinism comes from.** This project names it: the task and environment are deterministic; the sampler is not seeded, and the two step-A runs (4/5, 5/5 and 5/5, 3/5) are the evidence.
- **N environments run side by side without contaminating each other, and a broken one is killed and recycled.** Sixteen containers in 1.42 seconds, a network-blocked container failing in 0.04 s, a filesystem write invisible to the next container, a 3-second timeout returning 124 with zero containers left behind.
- **You can state the rollout cost of one training step and whether the bottleneck is the environment or the model.** 27 seconds for 60 rollouts, 61 with three update epochs, 0.45 s per rollout, model-bound by a factor of about ten in the simulator and no longer model-bound once a container is in the step.

## Reading

Three entries, each checked against its arXiv abstract page on 2026-10-20; links are to the paper pages, and no number in this chapter is quoted from them.

- **Terminal-Bench** - *Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces*, arXiv:2601.11868 - verified. What a terminal environment looks like as a benchmark: task definitions, a container per task, and a verifier per task. Read it for the interface, not the leaderboard.
- **SWE-smith** - *Scaling Data for Software Engineering Agents*, arXiv:2504.21798 - verified. Synthetic bug injection as a way to *manufacture* environments instead of collecting them; the lesson for this chapter is that environment cost is a data-generation cost too.
- **R2E-Gym** - *Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents*, arXiv:2504.07164 - verified. Procedural environments plus hybrid verifiers: the pairing this chapter argues for, where the environment's determinism is what lets the verifier be trusted.
- **Docker's resource-constraint documentation** - <https://docs.docker.com/config/containers/resource_constraints/> - not a paper, but the reference for the flags measured in Lesson 2, and the place to check what `--cpus` actually promises.

## Backfill

- [x] Question 3: How do I debug a long multi-turn failure? - rule out the environment first, and this chapter is the checklist. Replay one failing episode by hand; compare state digests at the point the reward changed; check the menu, the timeout code, and the exit code before touching a hyperparameter. Project 2's three interface bugs were each found this way, after a plausible algorithmic explanation had already been written down.
- [x] Question 4: Where does my training system quietly lie to me? - the most upstream lie is the environment reporting success it did not achieve: exit 0 with a half-written file, a menu that omits the action, a reward that changes without a state change. The second is the record: an identity made of a timestamp, an image tagged `latest`, a number with no header. Both are cheap to check and expensive to discover late.

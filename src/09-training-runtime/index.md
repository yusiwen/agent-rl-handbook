# Chapter 9 · The Industrial Training Runtime

> **Compute tier**: ⚠️ (single-GPU small models are fine; multi-node and Megatron routes need rented hardware) | **Status**: written — the loop closes and the number moves, both measured on one DGX Spark: two steps to show the gradient path, eight steps at n=6 on 24 prompts to show a held-out gain (0.667 to 0.859). One machine, one seed, eight steps: a transmission check, not a convergence result, and the run ends at the 118 GB memory cap.
> **Project**: [Project 3 · The VeRL training runtime](../appendix/projects.md)
> **Rule of this book**: this chapter is about **system implementation only**; the meaning of the algorithms is in [Chapter 4](../04-alignment-algorithms/index.md).

## First, the gist

Training a real model needs several models alive at once (the one being trained, a frozen copy for reference, a scorer, sometimes a critic), spread across several GPUs, all while generating fresh samples. A "runtime" is the software that keeps that circus organised. This chapter reads one such runtime (VeRL) end to end: how data flows, what the workers do, where the loss is computed — and what to do when it crashes at 3 a.m.

It also documents a failure, and then what fixed it. The plan was to install VeRL and run a GRPO loop on the Spark. The first attempt established something narrower: the package installs, its supervised-training path runs and can be measured, and its **reinforcement-learning path cannot start as configured**, because the async rollout backends in this version are vLLM, SGLang and TensorRT-LLM and none of them was installed. That is not bad luck; it is the difference between a plan verified by a dependency resolver and a plan verified by an import, and it is Lesson 9 - which also records the two things that turned the failure into a run: the image the package actually declares, and a dependency nobody declared.

## What you'll be able to do

- Describe what actually happens after you type one training command, in the order it happens.
- Explain how one runtime runs PPO, GRPO and their variants without being rewritten.
- Account for where the memory goes on a single GPU, with numbers from your own machine.
- Recognise the four ways a runtime fails at the boundary between "installed" and "runs".
- Decide whether a given runtime can start on your hardware **before** you spend a day downloading.

## Lessons

- [ ] 1. Why a runtime is needed at all: four models × distribution × online sampling
- [ ] 2. VeRL's global picture and its six-layer architecture
- [ ] 3. DataProto and the single controller: the two ideas that make distribution look like one process
- [ ] 4. The training path, step by step: from one command to one complete run
- [ ] 5. How algorithms are composed: PPO, GRPO, DAPO and the rest in one codebase
- [ ] 6. Reward loops and agent loops: from a single-turn trainer to a multi-turn tool-calling runtime
- [ ] 7. **Operations**: checkpoints and resume, failure recovery, parallelism (TP/PP/DP/EP), and the **memory ledger**
- [ ] 8. **Frameworks compared**: VeRL, ROLL, AReaL, SkyRL, Agent Lightning, OpenRLHF, TRL
- [ ] 9. Project 3a: a GRPO run on Qwen3 (Spark-sized: small model, `sdpa` attention backend, single GPU)
- [ ] 10. Project 3b: a preference-alignment run, and why it is the lighter first target

### Lesson 1 · Why a runtime is needed at all: four models × distribution × online sampling

A single-GPU fine-tune is a data pipeline and a loop: tokenise, forward, backward, step. Nothing about it needs a coordinator. Learning from feedback breaks that shape in four ways, and each one drags in machinery.

**More than one model.** A PPO-style loop holds at least the policy being trained, a frozen reference for the KL term, a reward model or a judge, and - for PPO rather than GRPO - a critic. Those live in different states (trainable, frozen, inference-only) and compete for the same memory. On one device they must be time-shared; across devices they must be placed.

**Two execution modes that cannot share a pass.** Sampling is a generation loop with a KV cache and no gradients; training is a forward-backward over the same weights. A runtime has to move weights between those two representations. In the served loop of Project 3 that move is an adapter load and takes 12-13 ms when the adapter is small; a full-weight sync by checkpoint-and-restart is 156 seconds, which is why every serious runtime has a weight-sync path that does not go through disk.

**The data is generated, not read.** A supervised step consumes rows that already exist. A step of reinforcement learning consumes rows the current policy produced, so a bug in rollouts changes the distribution being trained on. The only defence is recording which policy produced which sample, and that is a runtime concern rather than a model concern.

**Failure is partial.** One worker dies, the sampler stalls, a rollout exceeds its timeout. The useful runtime says *which* of the sixty rollouts in a group failed and whether the group is still usable, instead of losing the round.

The consequence: the interesting engineering is in the plumbing. How the trainer hands weights to the sampler, how samples come back as tensors, how groups are assembled, how a checkpoint is written without stopping the world. This chapter's measurements are about that plumbing, not about the loss.

### Lesson 2 · VeRL's global picture and its six-layer architecture

Read from the installed package (verl 0.9.1, `site-packages/verl`), the structure is six layers, and each maps to a question you will ask when something breaks.

1. **Entry points** - `verl/trainer/`: `main_ppo.py`, `sft_trainer.py`, `sft_trainer_ray.py`, `main_eval.py`, `main_generation_server.py`, plus a `distillation/` package. The layer is thin: parse a Hydra config, build a Ray or torchrun job, call `fit()`.
2. **Configs** - `verl/trainer/config/`: one YAML per shape (`ppo_trainer.yaml`, generated variants for megatron / torchtitan / veomni, `legacy_data.yaml`, and `rollout/`, `actor/`, `critic/`, `reward/` fragments). Hydra composes them, so a single override on the command line lands somewhere in this tree, and knowing which file is the difference between a five-minute fix and an hour.
3. **Trainer and controller** - the loop: build batches, dispatch to workers, collect results, log, checkpoint.
4. **Workers** - `verl/workers/`: role implementations (actor, critic, reward, rollout) over an *engine*. On a fresh install the engine registry lists FSDP, FSDP2, Megatron, torchtitan, veomni, NeMo-automodel and MindSpeed, and only the FSDP family is importable; the rest announce themselves as unavailable, one warning each.
5. **Rollout backends** - `verl/workers/rollout/`: `vllm_rollout`, `sglang_rollout`, `trtllm_rollout`, plus `hf_rollout.py` and `naive/`. The directory suggests five options; the registry says three.
6. **Utilities** - `verl/utils/`: dataset classes (RL and multi-turn SFT), tracking, checkpoint engines, profiling, distributed helpers. Most "why does it not start" answers live here: the data layer's chat-template assertion, the logger's API key, the missing `uvicorn` on the server path.

What the map buys you is triage. A missing module during import is layer 5 or 6. A config error naming a dot path is layer 2. A worker that starts and then runs out of memory is layer 4 with a number in layer 2 (`engine.model_dtype`, `optim`, `ulysses_sequence_parallel_size`). "The sampler never returned" is layer 5, usually because the backend is not installed - which is exactly the failure this chapter's own plan hit.

### Lesson 3 · DataProto and the single controller: the two ideas that make distribution look like one process

Two abstractions carry the runtime, and both are about hiding distribution without hiding failure.

**The batch object.** VeRL passes a batch rather than a list of tensors: a dictionary of named tensors (input ids, attention masks, position ids) plus a side channel of non-tensor metadata (the original prompts, the task ids, the per-rollout tags). The tensor half can be split, gathered and moved between workers. The metadata half cannot, which is why it must be carried explicitly instead of recomputed - and why a runtime that loses it produces rollouts nobody can attribute to a task. This book's own served loop has the same distinction in miniature: a `Rollout` record whose `segments` are the text half and whose `mask` is the tensor half. When the two disagree, the training is on something other than what the sampler saw.

**The single controller.** VeRL's trainer reads like a single-process program: a loop over batches, a call that returns a batch with rewards, a call that performs an update. Under it, each call dispatches to remote workers over Ray, with the batch serialised and scattered. The design choice is that the *control flow* stays in one place and only the *work* is distributed, which is what makes a stack trace readable: the failing line names the step, not a queue.

The trade-off is real. A single controller is a single point of coordination: it cannot overlap the next batch's sampling with this batch's update unless it explicitly pipelines them, and every round trip through the controller costs a serialisation. Runtimes that go further - fully asynchronous actor/learner loops, or a shared-memory weight server - buy overlap and lose the ability to reason about a round as a unit. For a first run the single controller is the right choice, and VeRL's async generation server is where it stops being one.

### Lesson 4 · The training path, step by step: from one command to one complete run

This is what the measured run did, in order, with the times from the Spark on 2026-11-07. The job was deliberately trivial - Qwen3-1.7B, four rows, two steps, one GPU - because the point is the path, not the result.

1. **torchrun starts one process.** `python -m torch.distributed.run --standalone --nproc_per_node=1` sets `RANK`, `LOCAL_RANK` and `WORLD_SIZE`. Without them the trainer fails immediately with `environment variable RANK expected, but not set`; the entry point assumes it is launched, not called.
2. **Hydra composes the config.** Defaults from `trainer/config/`, overridden by the command line. The composed config is about 6,000 characters and is printed on failure - which is how you discover that `engine.model_dtype` defaults to `fp32` for a target everyone describes as a bf16 fine-tune.
3. **The process group forms and the dataset is built.** The multi-turn SFT dataset tokenises each turn with the chat template and asserts that per-turn concatenation equals whole-sequence templating. On this model they differ, so it stops with a message that says exactly that; the documented escape hatch is `data.ignore_input_ids_mismatch=true`. Read the assertion before disabling it: the mismatch means the loss mask was computed on one tokenisation and the model sees another.
4. **The model is loaded and wrapped.** Measured here: *before* the FSDP wrap, 0.00 GB allocated and 0.00 GB reserved, with the device already at 78.94 of 121.63 GB in use because a 32 GB-capped inference server is resident and the pool is unified. *After* the wrap: **6.41 GB allocated, 7.78 GB reserved**, device at 87.80 GB.
5. **The steps run.** Two steps, batch 2, micro-batch 1, 120 tokens per step. Step 1: loss **5.972**, grad-norm **244.06**, peak **32.11 GB allocated / 34.21 GB reserved**. Step 2: loss **4.276**, grad-norm 141.34, peak reserved 34.40 GB. A validation pass follows at loss 2.20.
6. **The job exits.** Wall time **1 minute 52 seconds**, of which the first training step was 4.55 s and the validation pass about 100 s. Checkpointing was off (`save_freq=-1`), so nothing was written; had it been on, the FSDP state dict would have gone to `checkpoints/<project>/<experiment>`.

Read the ratio, not the absolute numbers: **6.41 GB of weights, 32.11 GB at peak**. Four fifths of the footprint is not weights, and that gap is the memory ledger.

<figure class="book-figure">{{#include ../figures/09-memory-ledger.svg}}</figure>

*Figure 9.1 - where the memory went for a 1.7B full fine-tune on one 121.63 GB unified pool: 6.41 GB of sharded weights, 32.11 GB peak during a step, 34.40 GB reserved after the second step, and 58.3 GB of host memory in Ray and the dataloader workers. Measured 2026-11-07, verl 0.9.1, FSDP, `engine.model_dtype=fp32`, batch 2, sequence length 256.*

### Lesson 5 · How algorithms are composed: PPO, GRPO, DAPO and the rest in one codebase

One codebase runs PPO, GRPO, RLOO, REINFORCE++ and the rest because the differences are three small functions, not three programs. Reading the config tree in 0.9.1:

- **The estimator is a config value.** `algorithm.adv_estimator` selects how advantages are computed from rewards: `gae` for PPO's critic-based advantage, `grpo` for the group-relative one, `rloo`, `reinforce_plus_plus`, and so on. The trainer does not branch on algorithm names; it asks for an advantage function and applies it to a batch.
- **The critic's presence is derived.** GAE needs a value head and GRPO does not, so the runtime builds a critic worker when the estimator needs one - which is why `critic.*` overrides are quietly ignored under GRPO, a detail that costs an hour the first time you tune a critic that is not running.
- **The KL term is a config block**, not a line in the loss: `kl_penalty` chooses `kl`, `abs`, `mse` or `low_var_kl`, and `use_kl_in_reward` decides whether it enters the reward or only the loss. Chapter 4 explains what each does; the runtime's job is to make the choice visible in the record.
- **DPO is not in 0.9.1.** The trainer directory has no DPO entry point. If you need it, pin a version that ships it or write the loss yourself, which Chapter 4 does in thirty lines and which is why this chapter does not claim to have run it.

The generalisation: when someone says a framework "supports twenty algorithms", ask which of these four is a config value - the advantage estimator, the presence of a critic, the KL placement, the sampling temperature - and which is code. In a well-built runtime the first three are config; in a hostile one they are forks.

### Lesson 6 · Reward loops and agent loops: from a single-turn trainer to a multi-turn tool-calling runtime

The single-turn trainer assumes a batch of prompts, one generation each, one scalar reward per generation. An agent runtime assumes a trajectory, a tool call in the middle, a reward that may arrive only at the end, and a state that must be reset between attempts. Moving from one to the other is where the interesting work sits, and this repository has a runtime on each side of it.

**The served loop (Project 3).** Our runtime treats the environment as the source of truth: each round samples over HTTP from a vLLM server, scores with the environment, computes a gradient in a separate process, writes an adapter, and hot-loads it as the next round's policy. Measured: 60 rollouts in about 50 seconds for six tasks at ten samples; adapter registration 12-13 ms against 156 seconds for checkpoint-and-restart. The cost is that sampling and training cannot overlap - the loop is sequential by construction, and it says so.

**The runtime's version.** VeRL's PPO path expects the rollout to come from a server with a synchronisation protocol: weights are pushed to the backend, the backend samples, the trainer collects a batch, updates, and pushes again. In 0.9.1 that protocol is implemented for vLLM, SGLang and TensorRT-LLM only (`_ROLLOUT_REGISTRY` in `verl/workers/rollout/base.py` has exactly three entries, all `mode="async"`); the `hf` and `naive` rollout files in the same directory are not reachable through it.

That is the boundary this chapter wants you to see: **the agent part is a protocol between two processes**, and whether it works depends on which backends the runtime can speak to. A plan that says "use the HF rollout to avoid installing vLLM" is a good plan for a synchronous trainer and a dead end for an asynchronous one.

### Lesson 7 · Operations: checkpoints and resume, failure recovery, parallelism (TP/PP/DP/EP), and the memory ledger

**Parallelism, and what each axis costs.**

- **DP (data parallel)** splits the batch; every rank holds a full copy of the weights. FSDP is DP with the copy sharded between steps, and it is what ran here: 1.7B on one rank, 6.41 GB allocated.
- **TP (tensor parallel)** splits the matrices inside a layer. It is what makes 70B models fit and the axis that punishes slow interconnect hardest, because a TP layer is a collective per layer.
- **PP (pipeline parallel)** splits layers into stages: throughput for fit, and the axis that makes a 3 a.m. crash hardest to read, because the failure appears in a stage rather than a line.
- **EP (expert parallel)** partitions mixture-of-experts layers, where the all-to-all becomes the bottleneck.

On a single-GPU Spark none of these is exercised beyond DP=1. The honest statement: this chapter measured the FSDP path and cannot claim numbers for the rest.

**The memory ledger, filled in.** Qwen3-1.7B, 1,726,997,504 parameters, FSDP, defaults of this version:

| component | measured or derived | value |
| --- | --- | --- |
| sharded weights, after wrap | measured | **6.41 GB allocated**, 7.78 GB reserved |
| peak during a training step | measured | **32.11 GB allocated**, 34.21 GB reserved |
| peak after step 2 | measured | 34.40 GB reserved |
| optimiser state (AdamW, fp32 moments and master copy) | derived: 8 bytes per parameter for m and v, 4 for the master | about 20.7 GB for 1.73B |
| activations (2 sequences × 256 tokens, gradient checkpointing on) | measured, inside the 32.11 GB | a few hundred MB |
| host memory (Ray and 8 dataloader workers) | measured | 58.3 GB |
| device pool | measured | 78.94 GB used before the wrap, 87.80 GB after |

The ratio between rows one and two is the lesson: **weights 6.41 GB, peak 32.11 GB.** The gap is optimiser state and activations, which is why "it is only a 1.7B model" is not an argument about memory. Two knobs move the gap: `engine.model_dtype` (fp32 master weights are the default here; bf16 shrinks the copy but not the Adam moments) and the batch and sequence shape (activations scale with both).

**Checkpoints and resume.** A checkpoint must contain enough to continue: weights, optimiser state, scheduler position, step counter, and the data iterator's position. VeRL's `checkpoint.save_contents` defaults to `["model", "optimizer", "extra"]`, which is that contract - and "extra" is the interesting word, because a resumed run that restarts its data iterator at zero will silently repeat the first batches. Test resume at an epoch boundary and check that the second epoch does not open at the loss of step one.

**Failure recovery, measured on our own runtime.** The most instructive failure in this project was not a training bug: a container without a memory cap held 70 GB of autograd graphs on the unified pool, and the kernel OOM killer took the inference server **and** the ssh session, because session processes carry a higher `oom_score_adj`. The fix belongs in every runtime's defaults: cap every container (48 GB for the trainer, 32 GB for the server here), and treat "the server died" as a memory event until proven otherwise.

### Lesson 8 · Frameworks compared: what to measure, not what to believe

This chapter ran one runtime and read one runtime, so it can offer criteria rather than a league table. Six questions decide whether a framework fits your loop, and each is answerable in an afternoon:

1. **What does it speak to for rollout?** Count the registry entries and check that one is installed on your machine. This is the question that killed this chapter's plan.
2. **Which engines are importable?** `megatron`, `torchtitan`, `veomni`, `nemo_automodel` and `mindspeed` were all absent from the fresh install, each with a one-line warning. "Supports Megatron" means "if you install Megatron", which is a second project.
3. **How does weight sync work, and how long does it take?** 12 ms for an adapter, 156 s for a restart, something in between for a sharded full-weight copy. Measure it on your own model; it decides how many rounds per hour you get.
4. **Does it support the multi-turn shape you need?** Tool calls inside a trajectory, per-turn loss masks, and a reward that arrives at the end are three separate capabilities.
5. **What does a record look like?** If the only output is a log line, you cannot publish from it. The minimum is code revision, config hash, dataset digest, per-sample ids and per-round aggregates.
6. **What is the failure behaviour?** A runtime that loses a round to one bad rollout, or hangs on a dead sampler, costs more than a slower one that isolates failures.

Against those criteria on this machine: VeRL 0.9.1 installs cleanly, its FSDP supervised path runs and is measurable, its RL path needs a rollout backend that is not present, and Megatron-class scaling is out of reach. The alternatives named in the outline - ROLL, AReaL, SkyRL, Agent Lightning, OpenRLHF, TRL - were not installed or measured here, and this chapter does not pretend otherwise. If you need that comparison, run the six questions and write your own table; a borrowed one is about someone else's hardware.

### Lesson 9 · Project 3a: a GRPO run on Qwen3, and the four failures on the way in

The plan: install VeRL in its own environment, keep the environment's torch, avoid vLLM by using the HF rollout backend, run a two-prompt GRPO step on Qwen3-1.7B, and record the ledger. Three of those four happened.

**What was measured.** A 21 GB virtual environment (`/work/verl/env`, `--system-site-packages` so the NGC torch is visible), installed in under three minutes. Versions: verl **0.9.1**, transformers **5.10.4**, ray **2.60.0**, numpy **2.1.0**, torch **2.14.0a0+b2c75dd062.nv26.9**, vLLM **absent**. Note the drift against the resolution recorded in Project 3's README before the install: ray 2.59 versus 2.60.0, numpy 2.5 versus 2.1.0. A resolver's answer is a prediction; the installed tree is the fact.

**The four failures, in the order they arrived.** Each is a category, not an anecdote:

1. **The entry point assumes a launcher.** `python -m verl.trainer.sft_trainer` fails with `environment variable RANK expected`; the same module under `torch.distributed.run --standalone` starts. Read how a project expects to be launched before debugging its config.
2. **A dependency is imported but not declared.** The package imports `uvicorn` from `verl/workers/rollout/utils.py`, which is on the import path of the checkpoint engine, which is on the import path of the SFT trainer. `uvicorn` is not in the dependency list; installing it (and `fastapi`) is the fix. This is the commonest "installed but will not run" failure in the wild.
3. **A default logger wants a network and a key.** `wandb` is the default tracker; with no API key the job dies before the first step. `WANDB_MODE=disabled` plus `trainer.logger=["console"]` are the offline settings.
4. **A data-layer assertion that is also a warning.** The multi-turn dataset asserts that per-turn tokenisation equals whole-sequence tokenisation; here it does not, and the run stops unless `data.ignore_input_ids_mismatch=true`. Disabling it is a decision about the loss mask, and the warning it downgrades says so.

**And the fifth, which is the one that matters.** `_ROLLOUT_REGISTRY` contains exactly `("vllm", "async")`, `("sglang", "async")`, `("trtllm", "async")`. There is no synchronous entry and no `hf`, so the plan's central avoidance - do not install vLLM - is not available in this version.

**The second attempt, and where it stopped.** PyPI ships no aarch64 wheel for `vllm` (every release has two files, and neither names an arm build), so the route had to be the official image. `vllm/vllm-openai:v0.11.2` is published for arm64 and is the generation that matches verl 0.9.1's stack: torch 2.9.0+cu129, vllm 0.11.2, transformers 4.57.1, ray 2.51.1. Inside it, `verl 0.9.1` installs over the image's own torch, and **both imports succeed** - which is further than the first attempt got, and required two more undeclared dependencies: `transferqueue` (for the async data path) and the already-known `uvicorn`.

Then the first line of the training log answers the question:

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is
(8.0) - (12.0)
```

and the package's own arch list ends at `sm_120`.

**This paragraph used to claim a no-go, and it was wrong on all three counts.** The correction is worth reading as a case of the book's own rule, because each of the three sentences was an inference recorded as a measurement:

- **The image was not the "generation that matches verl 0.9.1."** verl 0.9.1's `[vllm]` extra declares `vllm==0.24.0` (`torch==2.11.0`). `v0.11.2` is three release lines below what the package asks for, and it was chosen from the vLLM 0.11 era because that era looked like the right vintage.
- **"The arch list ends at `sm_120`" explains nothing.** Every stack that works on this machine ends there: the training image's `torch 2.14.0a0+cu134`, the serving container's `2.13.0+cu130`, and the corrected image's `2.11.0+cu130`. CUDA runs `sm_120` cubins on `sm_121` hardware within the same major version, which is why a device capability of 12.1 is not a reason to refuse a build that stops at 12.0.
- **"No pair exists" was never tested.** It followed from the two sentences above.

What actually refused was **torch 2.9's device guard**, which raises when a device's capability exceeds the build's maximum - a property of that torch, not of the GPU and not of vLLM. The pair the package declares works: `vllm/vllm-openai:v0.24.0` (arm64, CUDA 13.0.2 base, `torch 2.11.0+cu130`) plus `verl 0.9.1` runs `main_ppo` to two completed steps on Qwen3-1.7B, eight prompts with two rollouts each, GRPO with no critic:

| | step 1 | step 2 |
| --- | --- | --- |
| rewards (mean / max / min) | -0.125 / 1.0 / -1.0 | -0.375 / 1.0 / -1.0 |
| advantage max | 0.707 | 0.707 |
| actor loss | -0.0006 | 0.0120 |
| gen / update_actor / update_weights | 6.37 / 12.20 / 2.34 s | 6.37 / 8.45 / 2.57 s |
| step | 28.89 s | 19.81 s |
| throughput | 141.6 tok/s | 215.5 tok/s |

The rewards differ between the two steps because the first step's update changed the policy: a loop, not one forward pass. Four things sit between the image and a run, and each is a version fact rather than a hardware one: `transferqueue` (imported, undeclared), `HF_DATASETS_CACHE` (the HF mount is read-only), `attn_implementation=sdpa` (the FSDP path defaults to FlashAttention 2, which has no aarch64 wheel), and a `data_source` the reward loop recognises (in 0.9.1 it keys the reward on `data_source`, not on `custom_reward_function`). The run also needs the inference server stopped: step two holds 93.1 GB of CPU-side memory on a 121 GB machine, and with an 18 GB server up the dataloader is killed at the next step.

The ledger entry for this lesson changes accordingly, and so does the honest summary of the project: a GRPO loop at 1.7B scale runs on this hardware, and the paragraph below records what eight steps of it did to a held-out set - which is still not a convergence claim, because eight steps on 24 prompts is not training a model, it is testing whether the machinery carries a gradient.

**What remains for the reader.** A measured supervised run through the same runtime (Lesson 4), a filled ledger (Lesson 7), and - now measured - a GRPO run that actually moves the policy.

**The learning check, added the same day.** Two steps show the loop closes; they say nothing about learning. Answering that badly is easy: train on a task set the model already solves, which is exactly what the first three generated arithmetic sets were - 369/384, 388/448 and 332/448 solved, with only 24 of the 112 prompts in the last two sets landing strictly between always-solved and never-solved. Those 24 became the training set; a fresh draw of 24 from the same four families became the held-out set, measured **before** training at 128/192 = 0.667 (Wilson [0.597, 0.730]).

Eight GRPO steps at n=6, lr 1e-6 and a 384-token cap took the training reward from 0.00 to 0.875 and the held-out number to **165/192 = 0.859** (Wilson [0.803, 0.902]): **+19.3 points**, intervals disjoint. Three held-out items went from 0 of 8 to 8 of 8; one went the wrong way, 8 of 8 to 5 of 8. The largest advantage in any group fell from 2.04 to 0.91 over the eight steps as the policy saturated the training prompts, which is the shrinking-gradient signature this chapter describes, now measured on this box rather than asserted from a paper.

The honest boundaries: the held-out items are new *instances* of four known families, not new task types; eight epochs over 24 prompts means each prompt is seen about five times, which is why the held-out column carries the argument; and the container is OOM-killed after step 8 at a 118 GB cap, so the run ends where the memory budget ends. A task set with real headroom, longer runs, and several seeds is the next increment, not a footnote. The alternative this repository does have is its own served loop, measured end to end in Chapter 6: smaller, fewer features, no multi-node story, and it answers the same system questions at 1.7B scale.

### Lesson 10 · Project 3b: a preference-alignment run, and why the record is not the one the outline promised

The outline put preference alignment second because it is lighter: no sampler, no reward model, one forward pass over chosen and rejected pairs. That reasoning survives - a DPO step is closer to supervised fine-tuning than to RL - but the specific instruction ("run VeRL's DPO trainer") does not, because **verl 0.9.1 does not ship one**. The trainer directory contains PPO, SFT, evaluation, a generation server and distillation; `find -iname '*dpo*'` returns nothing.

Three honest options, in increasing order of effort:

1. **Pin a version that has it.** Check the tag before installing; the cost is re-validating the rest of this chapter against that version.
2. **Implement the loss on top of the supervised path.** DPO's gradient needs four log-probabilities per pair - policy and reference, chosen and rejected. The reference can be the same weights with the adapter disabled, and the loss is a `logsigmoid` of a difference. Chapter 4 writes it out; a runtime that can compute log-probabilities for a batch can run it.
3. **Use a library that targets preference optimisation directly** and accept a second environment.

What this chapter can say with evidence is narrower and still useful: the supervised path runs on this machine, its ledger is measured, and the difference between supervised fine-tuning and DPO in the ledger is one extra frozen copy of the model - 6.41 GB here - because the reference needs neither gradients nor optimiser state. That is the whole reason preference alignment is the lighter target, stated as a number instead of an adjective.

## The memory ledger (measured on the Spark, 2026-11-07)

| Component | Parameters | Optimiser state | Gradients | Activations | Notes |
|---|---|---|---|---|---|
| Policy (actor), FSDP, one rank | 1.73B | AdamW fp32, about 20.7 GB derived | sharded with the weights | inside the 32.11 GB peak | 6.41 GB of weights alone |
| Reference | 1.73B | none | none | forward only | not measured here; the ledger says one weight copy, about 6.41 GB in bf16 |
| Critic (PPO only) | 1.73B | as the actor | as the actor | as the actor | GRPO needs none; PPO was not run here |
| Reward model | 1.7B-class | none | none | forward only | absent by design in Project 3: the reward is a rule |
| Rollout KV cache | - | - | - | - | not measured in this runtime; the served loop caps the server at 32 GB and reserves about 13 GB of KV at `--gpu-memory-utilization 0.10` |
| Host memory | - | - | - | - | 58.3 GB for Ray and eight dataloader workers |

## A troubleshooting sheet for loss curves

| symptom | what it usually is | what to check first |
| --- | --- | --- |
| loss falls, reward flat | the loss is measuring the wrong tokens - prompt instead of reply, or a mask that includes the observation | the mask in the batch; `trained_tokens` against `tokens` in the record |
| loss flat, reward flat | every group is constant, so every advantage is zero | the fraction of groups with non-zero spread in the round record |
| reward rises, held-out falls | the policy learned the scorer, or memorised the trained tasks | a held-out set the policy never saw, reported per task |
| entropy collapses | the KL coefficient or the length normalisation over-rewards certainty | per-turn entropy in the record; the `kl_penalty` setting |
| KL explodes after a resume | the scheduler or the reference was not restored | checkpoint contents; the step counter in the record |
| OOM at a step boundary rather than inside one | two copies of the model alive across a weight sync | the sync path and the container cap |

## Deliverable

- A memory/cost ledger filled in with numbers measured on your own machine - the table above, with measured rows marked measured and derived rows marked derived.
- A training configuration that reproduces on the Spark: `verl 0.9.1` in a virtual environment over the NGC torch, `torch.distributed.run --standalone --nproc_per_node=1`, FSDP, `WANDB_MODE=disabled`, `trainer.logger=["console"]`, and - for the multi-turn dataset - `data.ignore_input_ids_mismatch=true` with the reason written down.
- A troubleshooting sheet for loss curves: the table above, plus the four launch and import failures in Lesson 9, which happen before any curve exists.

## How you know you passed

- **You can list the five things that happen behind one command, and roughly what share of the time each takes.** Here: launcher, config composition, process group and dataset, model load and FSDP wrap (6.41 GB), steps (4.55 s for the first), validation (about 100 s of the 112 s wall time).
- **You can state what your runtime cannot do, with evidence.** This chapter can: no GRPO on this machine without a vLLM or SGLang backend, because the registry in 0.9.1 has three async entries and none is present.
- **A resumed run continues the curve.** Not demonstrated here - checkpointing was off in the measured run - and the chapter says so rather than implying it.
- **You can set breakpoints at the actor loss, the advantage and the KL term and explain the parameter change.** Lesson 5 is the short version; Chapter 4 is the long one.

## Reading

- **VeRL documentation** - <https://verl.readthedocs.io/> - the source for the configuration surface and the launch commands; read it before the code, because 0.9.1's config is composed rather than written.
- **HybridFlow: A Flexible and Efficient RLHF Framework** - *Sheng et al.*, arXiv:2409.19256 - verified. The design document behind VeRL's split between a single controller and distributed workers.
- **Proximal Policy Optimization Algorithms** - *Schulman et al.*, arXiv:1707.06347 - verified. The objective the runtime computes, and why a critic is optional in the estimator sense.
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** - *Shao et al.*, arXiv:2402.03300 - verified. GRPO's group-relative advantage, the estimator this chapter could not run here.
- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** - *Rafailov et al.*, arXiv:2305.18290 - verified. The loss that makes preference alignment lighter than PPO, and the reason Lesson 10 spends 6.41 GB on a frozen reference.
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** - *Rajbhandari et al.*, arXiv:1910.02054 - verified. Where the memory ledger comes from: parameter, gradient and optimiser-state sharding.
- **PyTorch FSDP documentation** - <https://pytorch.org/docs/stable/fsdp.html> - the wrap policy and the `use_orig_params` / `model_dtype` knobs behind the 6.41 GB.

## Backfill

- [ ] Question 1: How do I know a run is real? - the runtime's record, not its log line: config hash, code revision, dataset digest, per-sample table. The measured run printed a 6,000-character config on failure and one `step:1` line on success; only the second is a number you can publish, and only if it is written down.
- [ ] Question 2: Which tasks should be in the test set? - the runtime does not care, which is the point: it will overfit whatever the dataset contains.
- [ ] Question 3: How do I debug a long multi-turn failure? - isolate the layers: environment, rollout protocol, batch assembly, loss mask. The four failures in Lesson 9 are all layers 5 and 6, and none of them produced a curve.
- [x] Question 4: Where does my training system quietly lie to me? - this chapter is the main battlefield. Three lies, all measured: a resolver whose prediction differed from the install; a config whose `model_dtype` is fp32 while everyone says bf16; and a "planned" rollout backend that the installed tag does not contain. Each was found by a command that takes seconds, and each would have cost a day if it had been found after the model download.

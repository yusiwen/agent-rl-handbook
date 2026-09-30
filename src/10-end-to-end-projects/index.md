# Chapter 10 · End-to-End Project Cluster

> **Compute tier**: ⚠️ (a 0.6B model runs end to end; 8B needs QLoRA and small batches) | **Status**: stub
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

- [ ] 1. The whole picture: architecture, capabilities and data flow
- [ ] 2. Deploying the data-science agent: environment, quantisation, starting vLLM
- [ ] 3. A one-command Docker deployment of the web UI, plus the agent's reasoning loop (ReAct-style turns and its marker system)
- [ ] 4. SFT + RL + code sandbox: from the cold-start label protocol to a full GRPO run

**Project 5 · OpenClaw-RL (online RL for multi-turn tool callers)**

- [ ] 5. How multi-turn agents should actually be trained: environment, data and the reward loop
- [ ] 6. OpenClaw-RL dissected: next-state signals, the four-ring asynchronous architecture, three core algorithms
- [ ] 7. Reading the core source: from sample to PPO loss to combined update
- [ ] 8. The tool-call RL pipeline: how rollout, actor and PRM join into one line (loss masks, session headers, weight sync, sample freshness)
- [ ] 9. A three-stage run on Qwen3-0.6B: ReTool-SFT → ReTool-RL → PRM process reward

## Platform notes (Spark)

| Step | Risk | Workaround |
|---|---|---|
| Starting vLLM | upstream images do not support sm_121 | use a GB10-patched image, or fall back to Ollama / SGLang |
| Official Docker images | mostly x86_64 | rebuild for aarch64 from the Dockerfile, or run only the inference side |
| Attention backend | FlashAttention has no sm_121 support | switch to FlashInfer or PyTorch SDPA |
| Building from source | high parallelism can exhaust the shared memory | cap `MAX_JOBS` |

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

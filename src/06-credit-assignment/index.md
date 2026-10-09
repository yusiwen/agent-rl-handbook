# Chapter 6 · Credit Assignment: From Token to Turn

> **Compute tier**: ✅ (0.5–3B models are comfortable) | **Status**: draft — all six lessons have prose; the project's runs have not been executed yet
> **Project**: [Project 2 · Process feedback and long-trajectory credit assignment](../appendix/projects.md)

## First, the gist

When a ten-step agent task fails, it is almost never true that *every* step was wrong. **Credit assignment** is the job of working out which steps deserve the blame (or the praise) — and then turning that judgement into something a training loop can use. The practical trick is to score smaller pieces: a step, a tool call, one turn of the conversation, instead of the whole run. This chapter gives you a map of those options, then walks through one implementation (Reagent) in detail: agent supervised fine-tuning (SFT), a merged reward, and the GRPO (group relative policy optimisation) loop it trains with.

> **Terminology**: the "U" in **Reagent-U** stands for *Unified Feedback Integration* — merging written critiques and numeric scores into one RL loop, as described in the Reagent paper *(Fan et al., 2026 — in Reading)*.

## What you'll be able to do

- Explain why "give every step a score" is *not* the same thing as "credit assignment is solved".
- Place any credit-assignment method on the token / step / turn / trajectory map and say what it costs.
- Describe how written feedback and numeric reward can be combined, and what breaks when they conflict.
- Say which part of a long failure a given signal can even *see*, before blaming the algorithm for not fixing it.

## Lessons

- [ ] 1. Why credit assignment is hard: long trajectories, sparse feedback, discrete tool calls, noisy negative samples
- [ ] 2. **The solution map**: token-level, step-level, turn-level and trajectory-level signals
- [ ] 3. Reagent-U: merging written critiques with scalar rewards, and the two-stage sampling it uses
- [ ] 4. What the claims are, and what the published experiments actually show
- [ ] 5. Project 2a: Reagent's architecture, plus the Docker environment
- [ ] 6. Project 2b: agent SFT, rule/model reward fusion, a GRPO run and an inference-service loop

### Lesson 1 · Why credit assignment is hard

**The problem in one line.** A training update needs a weight per decision; a terminal reward gives you one number for all of them. Credit assignment is the act of *inventing* the missing weights — and every invention is an assumption you can be wrong about. *(textbook fact for the formal statement; the consequences are my judgement)*

**Four difficulties, and each one is a different thing.**

**1. Long trajectories.** The information a terminal reward carries per decision shrinks with length. *(arithmetic on stated assumptions, not a measurement)* One score over a 2,000-token single-turn answer is one signal per 2,000 decisions. The same one score over a 20,000-token ten-turn agent trace is one signal per 20,000. Nothing about the algorithm changed; the *density* of what you can learn from fell by an order of magnitude. This is why the same recipe that works on single-turn maths stalls on agent traces ([Chapter 5](../05-reward-engineering/index.md), Lesson 1).

**2. Sparse feedback.** Many tasks return a binary outcome. With a 30% pass rate, most of your samples say "no", and the useful part — *which* of the ten steps broke it — is missing from every one of them. Worse, the failure that matters is often a single bad decision early on, with nine reasonable steps after it. A trajectory-level negative gradient punishes all ten equally. *(textbook fact)*

**3. Discrete tool calls, and boundaries that are not the model's.** A tool call is a large, discrete jump: the model emits arguments, something else runs, and a new observation appears. The tokens in the *observation* were not produced by the policy — training on them is a silent corruption ([Chapter 2](../02-mdp/index.md), Lesson 3's POMDP (partially observable Markov decision process) framing; the mechanism is the loss mask, [Chapter 8](../08-environment-engineering/index.md)). So "per token" is not a well-defined unit once tools are in the loop: the natural unit is the turn, and a turn boundary is something your system has to *detect and record*, not something the environment hands you. *(my judgement)*

**4. Noisy negative samples.** Not every step in a failed trajectory is bad, and not every step in a successful one is good. A run can succeed through a lucky guess after a broken plan; a run can fail after nine correct steps and one environment error. Both are label noise at the trajectory level, and no reward-model improvement fixes it, because the missing information is *structural*. *(my judgement)*

**Why "score every step" is not the answer by itself.** *(my judgement; the underlying measurement problem is [Chapter 5](../05-reward-engineering/index.md), Lesson 2)* Suppose you do the obvious thing and get a score for every step. Three problems remain:

1. **The step score is a proxy too.** A PRM (process reward model) trained on constructed labels inherits the verifier's blind spots; one trained on human labels inherits the annotators' disagreements. You now have a dense proxy instead of a sparse one — not a grounded signal.
2. **"Step" is your modelling choice, not a fact about the task.** Is a code edit one step or five? Is a search call one step or two (the query, then reading the result)? Change the segmentation and every score means something different.
3. **Scores do not say what to do differently.** A per-step 0.3 tells the optimiser to make that step less likely. It does not tell the model the *alternative*, and in a multi-turn task the alternative usually involves an action three steps earlier.

**Where this chapter is going.** The map (Lesson 2) tells you what each signal level costs and needs. The project (Lessons 5 and 6) builds one full loop — a reasoning reward model that emits a critique *and* a score, and a training scheme that merges them. And the honest conclusion up front: there is no known method that solves credit assignment in general. What exists is a set of instruments, each with a stated price, plus the discipline of measuring whether the instrument is telling you anything.

### Lesson 2 · The solution map

**The core table of this chapter.** Read it as "what would I have to *build* to get a signal at this level", not as a menu of algorithms — most real systems use two or three levels at once. *(my judgement for the prices and breakages; the named methods are the papers in Reading)*

| Signal level | Example methods | What you must provide | Where it breaks | Best suited to |
|---|---|---|---|---|
| **Token** | GAE (generalised advantage estimation) and the actor-critic machinery ([Chapter 3](../03-policy-gradient/index.md), Lesson 5) | A critic: a second trained model that scores half-written states | The critic is asked to value incomplete work far from any reward; its errors become the policy's weights (Chapter 3, Lesson 1) | Single-turn long reasoning, where there is no turn structure to exploit |
| **Step** | PRM ([Chapter 5](../05-reward-engineering/index.md), Lesson 2); GiGPO's step grouping | Step labels — human, or constructed by resampling from that step | Steps are a modelling choice; the label source's bias becomes the reward's bias | Tasks with clean step boundaries: proof steps, code edits, tool calls with checkable effects |
| **Turn** | TRACE; turn-level credit in collaborative tasks; Agent Lightning's attribution layer | Turn boundaries recorded in the trace, and a scoring rule per turn | Boundaries must be detected; a turn score can reward a plausible-sounding turn | Multi-turn tool use, where the model acts, observes, then acts again |
| **Trajectory** | Outcome reward, plus voting or comparison across samples | A checker for the final state (or a judge) | One number for everything; the steps that actually decided the outcome are invisible | When only the final result is visible, and as the fallback layer of everything above |

**What the rows really differ in.** Three things, and it is worth naming them because they are what you will actually negotiate:

1. **What you must own.** Per-token needs a critic you train. Per-step needs labels or extra sampling. Per-turn needs your *plumbing* to record boundaries. Per-trajectory needs only the final checker — which is why it is always the fallback.
2. **What the signal can see.** A token-level value sees the state. A step score sees a step in isolation. A turn score sees an action plus its environment response — strictly more information than the step level, because the *response* is part of it.
3. **How much it costs per episode.** Extra model calls scale with steps; extra sampling scales with steps × group size; the trajectory level is free.

**The free one hiding in the table.** In a tool-calling environment, the observation itself is feedback: the command printed an error, the test failed, the page did not load. That is a per-turn signal that costs nothing to obtain and needs no model — the *next-state signal* in the [glossary](../appendix/glossary.md). Turning it into a reward is a design decision (Chapter 8 owns the environment side); remembering that it exists is what keeps people from training a critic to predict something the environment already tells them for free. *(my judgement)*

**Composition, not selection.** *(my judgement)* The shapes that work in practice:

- **Trajectory outcome + turn-level shaping.** The outcome is the truth; per-turn signals guide. This is the most common combination, and it is exactly what the Reagent paper calls unifying feedback.
- **Trajectory outcome + step-level grouping.** No labels and no critic: compare steps *within* a trajectory and trajectories *within* a group, which is the GiGPO idea.
- **Environment next-state + trajectory outcome.** The cheap turn signal plus the ground truth at the end.
- **Token-level GAE alone.** Still the right answer for single-turn reasoning, and a poor fit for multi-turn agents.

**Choosing for a task, in four questions.** *(my judgement)*

1. **Can the environment tell me something per turn for free?** If yes, use it before you train anything.
2. **Are the step boundaries objective?** If a human would disagree about where the steps are, a step-level reward will encode that disagreement.
3. **Can I afford labels or resampling per step?** If not, go up a level.
4. **What does the failure look like?** If failures are "the whole plan was wrong", trajectory-level is honest. If failures are "step 7 of 10 was wrong", you need better than trajectory-level — and [Chapter 11](../11-evaluation/index.md)'s traces are how you find that out.

### Lesson 3 · Reagent-U

**What the project is, from its own paper.** *(textbook fact from the paper's abstract; the numbers are reported, not reproduced — Reagent, in Reading)* The Reagent project's centrepiece is **Agent-RRM (Agent Reasoning Reward Model)**, a reward model for agent trajectories that emits three things instead of one:

1. an **explicit reasoning trace** — the model's own working,
2. a **focused critique** — refinement guidance that highlights the reasoning flaws it found,
3. an **overall score** — a process-quality number for the turn or trajectory.

The paper's motivation is this chapter's problem in one sentence: outcome-only supervision "fails to differentiate high-quality intermediate reasoning from entirely incorrect attempts". *(quoted from the abstract, verified against the arXiv page on 2026-10-09)*

**Three ways to use such a model, and the difference between them.** The same paper compares three integration strategies, and they are the clearest available taxonomy of "what do I do with written feedback": *(paper result, reported and not reproduced)*

| Strategy | What the text is used for | What it becomes in the training loop |
|---|---|---|
| **Reagent-C** — text-augmented refinement | The critique is fed back into the **prompt**, so the model refines its answer in-context | Not a gradient signal at all: it is in-context learning, and it does not change the weights |
| **Reagent-R** — reward-augmented guidance | The score is used as the reward, and the text is discarded | A dense reward from a model judge — [Chapter 5](../05-reward-engineering/index.md), Lesson 3's GRM (generative reward model), with its parser and its biases |
| **Reagent-U** — unified feedback integration | Critique and score are harmonised into one signal, and sampling is critique-augmented | Both, merged — the "U" of the chapter's terminology note |

**Why "U" is the interesting one, and where its difficulty lives.** *(my judgement)* Merging two signals of different *kinds* is not a matter of a weighted average. Text has no units. So the merge has to happen at one of three points, and each choice has a distinct failure mode:

1. **Text → number (a judge converts the critique into a score).** You now have two scalars — and the conflict problem of [Chapter 5](../05-reward-engineering/index.md), Lesson 5, *inside* a single reward, with the added twist that both scalars come from the same model family and can share a bias.
2. **Text → context (the critique stays text and is shown to the policy during the next rollout).** Powerful and gradient-free: it can change behaviour within a run without a single weight update. The trap is attribution: when the numbers improve, you cannot say whether the *weights* learned anything or the model simply read the critique.
3. **Text → filter (the critique decides which trajectories are trained on).** Cheap and interpretable. It throws away the trajectories it rejects, which is exactly where the learning signal about failures would have been — and it biases the training set toward whatever the critique model happens to like.

**The two-stage sampling the project describes.** *(as described by the project outline; the paper's abstract describes the same idea as critique-augmented sampling and multi-source reward harmonisation — my reading, not a quotation)* The sensible allocation is obvious once stated: score everything with the cheap rule reward first, then spend the expensive critique model only where it can change the outcome — the uncertain middle, where the policy sometimes succeeds. That is an *uncertainty-based budget for judgement*, and it is what makes a dense multi-signal reward affordable at all. The failure to watch for is the gate becoming a learned surface: if the policy can influence *which* trajectories get critiqued, it can learn to be judged only when it looks good.

**What "U" costs, stated plainly.** *(my judgement)* One more model to train, serve and version; a mapping from text to reward that has to be written down and monitored; the disagreement problem from Chapter 5, now internal to your reward; and a sampling policy that decides when the expensive judge runs. In exchange you get the only thing the sparse setting lacks: a signal that can say *why*, and therefore a signal you can debug.

### Lesson 4 · What the claims are, and what the published experiments actually show

**The claim being evaluated.** Read strictly, the Reagent paper claims: a structured reward model (trace + critique + score) trained and integrated as Reagent-U improves agent performance on tool-use and search benchmarks, over outcome-only training. Its reported headline numbers are **43.7% on GAIA and 46.2% on WebWalkerQA across 12 benchmarks** *(paper result, reported and not reproduced — read from the abstract on 2026-10-09; the code and models are released, which is what makes reproduction possible at all)*.

**What would count as evidence for the mechanism.** *(my judgement)* A benchmark number is evidence that a *training recipe* works, not evidence for *why*. The mechanism claim is about the credit signal, so the interesting question is an ablation: if you keep the extra compute (the critique model's forward passes, the second sampling stage) but replace the structured feedback with a plain scalar from the same model, how much of the gain survives? If most of it survives, you have bought a better judge or more compute, not better credit assignment. That is not a criticism of the paper specifically — it is the standard of proof this whole sub-field is held to, and the reason to read the ablations before the headline.

**Independent findings worth weighing, from neighbouring settings.** *(paper results, reported and not reproduced — all in Reading)*

- **Multi-turn training has a multi-turn-specific failure mode.** A study of self-evolution in multi-turn agents reports the "Echo Trap": the policy's reasoning collapses into repetitive templates, entropy falls, and the diversity that multi-turn RL depends on dies. *(RAGEN)* If your run's outputs start looking the same regardless of the task, that is this, and it is not a reward bug.
- **Step-level signals can be manufactured without a PRM.** Grouping steps *within* a trajectory and grouping trajectories *within a prompt* produces step-level advantages with no labels and no critic. *(GiGPO)* This is the cheapest route to the Step row of the map.
- **Turn-level credit works when the turn boundary is explicit.** In a collaborative two-player setting, per-turn credit against a reference improved outcomes over trajectory-level only. *(SWEET-RL)* Note the precondition: the turns were defined by the problem.
- **For tool use, reward design dominated.** A study of tool-learning RL found the reward's shape — format plus correctness terms — mattered more than the algorithm choice. *(ToolRL)* Read alongside Chapter 5, Lesson 5: this is the mixing problem, not the credit-assignment problem.
- **Attribution can be middleware.** A framework attributes trajectory-level rewards to individual LLM calls inside an unmodified agent framework, which means the credit-assignment layer does not have to live inside the trainer. *(Agent Lightning)* That is a plumbing result, and a practical one.
- **Offline step-level objectives exist.** A step-level offline method for multi-step reasoning shows the dense route is not limited to on-policy RL. *(OREO)*

**What none of this shows.** *(my judgement)* There is no result here that says credit assignment is solved, and no method that removes the need to check whether the signal is real. What the literature does show, consistently, is that *more specific* feedback beats one number at the end — and that the price of specificity is always paid somewhere: labels, sampling, a second model, or plumbing.

**What you can actually conclude for your own run.** Three transferable claims: (1) trajectory-only supervision is a floor, not a target, whenever failures are localised; (2) the cheapest dense signal is usually the environment's own next state; (3) if you buy a model-produced signal, budget for the measurement that tells you it is still valid — the agreement measurement from [Chapter 5](../05-reward-engineering/index.md), Lesson 6.

### Lesson 5 · Project 2a: Reagent's architecture, plus the Docker environment

**Project overview.** Deploy the reasoning reward model, run agent supervised fine-tuning (SFT), fuse rule and model rewards, and close the loop with a GRPO (group relative policy optimisation) run plus an inference service — the full description is in [the project index](../appendix/projects.md).

**The architecture in four blocks.** *(my judgement on the shape; the components are the project's)*

1. **The agent loop** — the policy acts, calls tools, observes, and produces a trajectory. Each turn's boundary is *recorded*, because Lesson 2's Turn row depends on it and nothing else will reconstruct it later.
2. **The RRM** — the reward/critique service from Lesson 3, producing trace, critique and score per turn or per trajectory, behind an interface so the training loop does not care which signal it is consuming.
3. **The fusion point** — one function, in one file: rule reward and model reward in, one scalar out, with the weights and the conflict policy written next to it ([Chapter 5](../05-reward-engineering/index.md), Lesson 5's rule).
4. **The trainer** — GRPO over grouped rollouts, with rule and model reward terms logged *separately* (a merged number cannot be debugged).

<figure class="book-figure">{{#include ../figures/06-reagent-loop.svg}}</figure>

*Figure 6.1 — the loop the project closes: rollouts from the current policy, a cheap rule score, the expensive critique on the uncertain part, and one merged reward the GRPO update consumes.*

**Why a container, and what to pin.** *(my judgement)* Multi-turn agents need an environment that is deterministic and *resettable between attempts* — the same task must start from the same state every time, or your reward differences are noise from leaked state. That means a container per task family, pinned by image digest, with the tool set, the filesystem baseline and the dependency versions all recorded. The rule from [Chapter 5](../05-reward-engineering/index.md), Lesson 8 applies one level up: the environment's *version* belongs in every rollout row, because a rollout marked by verifier v2 and trained against v3 is a silent experiment.

**The failure modes specific to multi-turn infrastructure.** *(my judgement)*

| Symptom | The layer it is actually in |
|---|---|
| "The model gets worse at long tasks but fine at short ones" | Turn boundaries or truncation — the trace is being cut, or turns are merged |
| "Reward is high, behaviour is bad" | The fusion weights, or a critique model that rewards plausible text (Lesson 3) |
| "Training is unstable, outputs repeat" | The multi-turn collapse, not a learning rate (Lesson 4, RAGEN) |
| "The same task scores differently across runs" | The environment is not reset, or a tool is non-deterministic |
| "Nothing improves after step N" | Every group has one score — no advantage to learn from ([Chapter 4](../04-alignment-algorithms/index.md), Lesson 4) |
| "The model learned to answer without using tools" | The loss mask, or a reward that can be satisfied without the tools (Chapter 5, Lesson 7) |

**Where to set breakpoints.** *(my judgement, echoing the project template)* The three places that answer most questions: the **advantage tensor** (does any group have a spread?), the **fusion function** (what did each term contribute, per sample?), and the **turn boundaries** (does the reconstructed trace match the log?). Inspecting those three first is what keeps a multi-turn debugging session from turning into a week of hyperparameter guessing.

### Lesson 6 · Project 2b: agent SFT, rule/model reward fusion, a GRPO run and an inference-service loop

**Four stages, in order, with the exit condition of each.** *(my judgement; the stages are the project's)*

| Stage | What you do | Exit condition |
|---|---|---|
| **1. Agent SFT** | Supervised fine-tuning on demonstrations of the tool-use format, so the policy can act at all | The policy completes tasks end to end without breaking the format — a cold start, not a good policy |
| **2. Reward fusion** | Wire the rule reward and the RRM's score into one function; write the weights down | Both terms are logged separately, and you can point at the disagreements between them |
| **3. GRPO run** | Group rollouts per task, compute group-relative advantages, update | The advantage has spread in the majority of groups, and entropy is not collapsing |
| **4. Inference-service loop** | Serve the trained policy, generate the *next* round's rollouts from it, and repeat | The log shows the feedback from round N changing what round N+1 samples — not merely that scores were recorded |

**The discipline that makes the loop honest.** *(my judgement)* Round N+1's rollouts must come from the policy that round N produced — otherwise you are training on another model's behaviour and calling it self-improvement. That means: a version per checkpoint, a version per prompt template, and a rollout row that records both. This is also the point where [Chapter 9](../09-training-runtime/index.md)'s machinery (weight sync, sample freshness) becomes relevant: doing this asynchronously is a throughput decision with a correctness cost.

**The SFT stage is where the loss mask earns its keep.** Only the policy's own tokens contribute to the gradient. In a multi-turn trace, that is most of the text: tool outputs, environment messages and any prompt scaffolding must be masked out. A mask that is wrong by one turn teaches the policy to generate tool output — a failure that looks like hallucination and is actually a data bug. *(textbook fact for the mask; the diagnosis is my judgement)*

**What the pass criterion really asks for.** The chapter's test is that "the training log shows feedback actually reaching the next round of sampling — not just scores being recorded". *(my judgement on how to demonstrate it)* The cheap way to satisfy it deliberately: log, per round, the mean reward of each term, the reward vector the update actually consumed for a handful of sample ids, and the same ids' rewards in the previous round. If the terms move and the samples change, the loop is closed. If the numbers move but the sample ids never change, you have built a dashboard.

**What it costs.** *(my judgement)* Two models to serve (policy and RRM), a rollout service, a container per task family, a fusion rule to maintain, and a training run whose wall-clock is dominated by generation. This is the most expensive project in the book's first half, and it is the one that pays for every later chapter: [Chapter 9](../09-training-runtime/index.md) is this loop industrialised, and [Chapter 10](../10-end-to-end-projects/index.md) is it pointed at a real task.

## Project

**Project 2 · Process feedback and long-trajectory credit assignment** — deploy an RRM, run agent SFT, fuse rule and model rewards, and close the loop with GRPO plus an inference service. Full description and the six-project map: [the project index](../appendix/projects.md).

| Stage | What it produces | What goes wrong here |
|---|---|---|
| Environment | A pinned container per task family, with a real reset | State leaks between attempts; a non-deterministic tool |
| Agent SFT | A policy that can act, with the loss mask verified | Tool output leaking into the gradient; format collapse |
| Reward fusion | One reward function, both terms logged | Weights nobody wrote down; one term silently dominating |
| GRPO run | Grouped rollouts and an advantage with spread | Zero-variance groups; entropy collapse; the Echo Trap |
| Inference loop | Round N+1's rollouts from round N's policy | Versions unrecorded; "improvement" measured against a moving policy |

**Environment.** Docker (or equivalent) for the task containers, a small policy and the RRM served locally, and the Verifiers-style rollout plumbing from [Chapter 5](../05-reward-engineering/index.md), Lesson 8. Write the environment, data, config and run notes into the project's own note file using the nine-face template in [the project index](../appendix/projects.md).

**Sequencing.** Get the *loop* right before the *signal*: a run that trains on a rule reward only, with the fusion point in place but returning just the rule term, proves the plumbing. Add the RRM afterwards, when a wrong result can only be the signal's fault.

## Deliverable

- The MDP (Markov decision process) draft from [Chapter 2](../02-mdp/index.md), upgraded into a design document that includes process feedback and a credit-assignment scheme — for every decision type in your task, the signal level you chose (Lesson 2) and the reason you did not choose the level below.
- A working GRPO training log, including how rule rewards and model rewards were weighted: the fusion rule in code, both terms logged per round, and the sample ids that show the feedback reaching the next round of sampling.

## How you know you passed

- You can explain why scoring every step does not automatically solve credit assignment, and name which of Lesson 1's four difficulties your task is actually suffering from.
- You can describe your policy for resolving rule-vs-model reward conflicts and why you chose it — with one example of a conflict you found and what you did about it.
- The training log shows feedback reaching the next round of sampling: per-round means for each reward term, and the same sample ids re-scored in the following round.
- Your design document names which tokens are masked out of the loss, and one failure that a wrong mask would produce.

## Reading

Reused entries were verified where they say; new entries were checked against the arXiv API on 2026-10-09 (IDs marked "verified"), and the numbers quoted from the Reagent paper were read from its abstract page on the same date. Numbers quoted from papers are **reported, not reproduced**.

The project this chapter builds:

- **The Reagent paper, and the source of this chapter's terminology** — Fan et al., *Exploring Reasoning Reward Model for Agents*, arXiv:2601.22154 — verified. Agent-RRM's three outputs (trace, critique, score), the three integration strategies (Reagent-C / Reagent-R / Reagent-U), and the reported 43.7% GAIA and 46.2% WebWalkerQA. Code: <https://github.com/kxfan2002/Reagent>. The stub's placeholder for the Reagent paper is now a verified entry, with the ID checked against the API.
- **The survey and taxonomy this chapter's map is drawn from** — <https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL> — a maintained list of credit-assignment methods (no arXiv ID; a reading list, not a result).

Credit assignment at each level of the map:

- **Turn-level reward assignment** — Tao et al., *TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents*, arXiv:2607.13988 — verified. The stub carried this as "ID to verify"; the ID is correct.
- **Step-level advantages without a PRM** — Feng et al., *Group-in-Group Policy Optimization for LLM Agent Training* (GiGPO), arXiv:2505.10978 — verified. The Step row of the map, and Lesson 4's second finding.
- **Turn-level credit with an explicit boundary** — Zhou et al., *SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks*, arXiv:2503.15478 — verified.
- **Attribution as middleware** — Luo et al., *Agent Lightning: Train ANY AI Agents with Reinforcement Learning*, arXiv:2508.03680 — verified. The Turn row's plumbing route.
- **Offline, step-level objectives** — Wang et al., *Offline Reinforcement Learning for LLM Multi-Step Reasoning* (OREO), arXiv:2412.16145 — verified.
- **Multi-turn training collapse** — Wang et al., *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*, arXiv:2504.20073 — verified. The Echo Trap in Lesson 4 and Lesson 5's table.
- **Reward design for tool use** — Qian et al., *ToolRL: Reward is All Tool Learning Needs*, arXiv:2504.13958 — verified. The finding that dominates the algorithm choice.
- **Fusing preferences with correctness signals** — Peng et al., *Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems*, arXiv:2502.19328 — verified. The fusion problem of Lesson 3, from the reward-model side.
- **Hierarchical, per-turn formalisation of a multi-turn task** — Zhou et al., *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL*, arXiv:2402.19446 — verified in Chapter 2's Reading, 2026-09-30. The Turn row's formal ancestor.
- **Where the token-level default comes from** — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438 — verified in Chapter 3's Reading, 2026-09-30.
- **Long-context RL at scale, as a boundary** — Kimi Team, *Kimi k1.5: Scaling Reinforcement Learning with LLMs*, arXiv:2501.12599 — verified. What a long-horizon run looks like when the compute budget is not yours.

## Backfill

- [x] Question 3: How do I debug a long multi-turn failure? — this chapter is the main battlefield, and it contributes the diagnosis order. First ask what the signal can even *see* (Lesson 2's map): if failures are localised to one step and your reward is one number for the trajectory, the bug is the signal's resolution, not the algorithm. Then check the four difficulties in Lesson 1 against your symptom — sparsity, discrete boundaries, structural label noise, length. Lesson 5's symptom table maps a symptom to the layer it is actually in, and the three breakpoints (advantage spread, fusion terms, turn boundaries) answer most "why did it not learn" questions before any hyperparameter is touched. [Chapter 11](../11-evaluation/index.md) supplies the traces.
- [x] Question 2: How do I design the reward? — the second half of the answer, from the signal side: which level of feedback to buy, what each level costs to own, and how to merge a written critique with a number without lying to yourself about what improved (Lessons 1 to 4).

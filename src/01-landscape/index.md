# Chapter 1 · The Big Picture and How to Choose an Algorithm

> **Compute tier**: ✅ (this chapter trains nothing; it is reading and deciding)
> **Status**: draft — all four lessons have prose, no numbers measured on hardware yet
> **Rule of this book**: this chapter only **chooses**. What each algorithm *means* is explained once, in [Chapter 4](../04-alignment-algorithms/index.md). What a runtime does with them is [Chapter 9](../09-training-runtime/index.md). Neither is repeated here.

## First, the gist

There are only a handful of ways to make a model behave better, and they are constantly confused with each other. This chapter draws the map before any maths: what each family of methods is for, where the feedback comes from, and what it costs. One idea matters more than the rest — **RL amplifies what a model can already do; it does not inject new knowledge**. If the model cannot solve a task even once in a while, no amount of training will teach it.

## What you'll be able to do

- Name the four families of alignment methods, and say for each what problem it solves and what it costs.
- Explain, for *your* task, what "RL is an amplifier, not an injector" implies.
- Give a concrete example of a task where you should **not** use RL.

## Lessons

- [ ] 1. Eight years of alignment in one pass: the four families
- [ ] 2. Where does the feedback come from? Agents, search, vision, emotion
- [ ] 3. Amplifier vs injector: does RL have a ceiling? (Payoff: what belongs to sampling at inference time, and what genuinely needs training)
- [ ] 4. A first intuition for compute cost: rollouts dominate, multi-turn multiplies, and sometimes RL is the wrong purchase

### Lesson 1 · Eight years of alignment in one pass: the four families

**Three stages, not one.** A model that answers you well has been through three stages. *(textbook fact)*

| Stage | What the model is trained to do | What you end up with |
|---|---|---|
| 1. Pretraining | Predict the next token, on a very large pile of text | A model that can *continue* text, not one that *answers* you |
| 2. Supervised fine-tuning (SFT) | Reproduce a demonstration: given this instruction, say this | An assistant that imitates the demonstrations you wrote |
| 3. Preference stage | Choose between its **own** attempts | An assistant that is nudged toward what you rate as better |

This book is about stage 3. The whole field calls it "alignment", "post-training" or "preference tuning"; those words overlap, and this chapter keeps them apart.

**Why stage 3 exists at all.** SFT can only copy. It has three blind spots, and every method in this chapter was invented to patch one of them:

1. **SFT cannot express "better".** You can write down a good answer, but you cannot write down *this answer is better than that one* — and for most tasks the second statement is much easier to make than the first.
2. **A wrong step poisons the rest.** SFT trains on complete sequences. If the model says something silly in step three of ten, steps four to ten are trained as mistakes caused by a mistake.
3. **There is no ceiling to climb.** SFT has no reason to prefer a careful answer over a lucky one. Asking for a better answer is not something a demonstration can teach.

**One question sorts every method: who scores the attempt?** If you remember one table from this chapter, make it this one.

| Family | Who provides the score | Famous names | What it costs |
|---|---|---|---|
| 1. Preference-based RL | Humans rank two answers; a **reward model** (a second model trained to imitate those rankings) learns to give the score | RLHF, PPO | Four models alive at once (policy, frozen reference, reward model, critic), heavy sampling, fiddly to stabilise |
| 2. AI feedback / scalable oversight | A **model** judges, guided by written principles | RLAIF, Constitutional AI | Cheap labels, but the judge's biases *become* your reward; the judge must be audited against humans |
| 3. Direct preference optimisation | The **preference pairs themselves** are the loss; no reward model is trained and nothing is sampled during training | DPO | Offline: it cannot explore or discover anything the pairs do not already contain; needs fresh pairs; weaker where correctness is checkable |
| 4. Verifiable-reward RL | A **program** scores: check the final answer, run the unit tests, validate the schema | RLVR, GRPO | Only usable where a checker exists; invites reward hacking (the model pleases the checker instead of doing the task) |

Two details are worth noticing on first reading, because beginners meet them as separate topics later:

- **GRPO removes the critic.** Instead of learning how good a situation is (a value network, which is the fourth model in the table), it scores several attempts at the same question and compares them *with each other*. Cheaper in memory, noisier. The full derivation is in [Chapter 4](../04-alignment-algorithms/index.md).
- **SFT is not a fifth family.** It is the floor all four stand on. Every method here assumes the model already follows instructions.

<figure class="book-figure">{{#include ../figures/01-family-choice.svg}}</figure>

*Figure 1.1 — choosing a family. Each question you answer "no" moves you one step down a more expensive ladder: a program that answers in milliseconds, then a model that answers in seconds, then GPU hours of sampling.*

**Eight years, four dates.** The sequence explains why the four families look so different. *(dates and attributions are from the papers listed under Reading)*

| When | What appeared | Why it mattered |
|---|---|---|
| 2017 | PPO (proximal policy optimisation) | Made policy-gradient training stable enough to use on real problems |
| 2022 | InstructGPT: RLHF at scale; Constitutional AI: a model as the judge | Preference tuning beat plain SFT; the judge could be an AI |
| 2023 | DPO | Preference tuning without a reward model and without sampling |
| 2024 | GRPO in DeepSeekMath; "RLVR" named in Tulu 3 | Verifiable rewards + no critic made RL reproducible on maths and code |
| 2025 | DeepSeek-R1; ReTool; agentic and asynchronous RL | Reasoning traces from RL alone; RL for **tool calls** over many turns |
| 2026 (now) | Where you are: agent RL, asynchronous runtimes, skill libraries | Parts 3–5 of this book |

**The order to learn them in.** *(my judgement)*

Learn **RLVR first**, then DPO, then PPO. The reasoning is about feedback latency, not prestige. A checker answers in milliseconds and never gets bored, so the shortest possible loop "try → score → adjust" is available to you on day one. DPO needs only a static pair file, so it teaches you what preferences do without any sampling infrastructure. PPO is the general machinery — you should reach it once you have a task where no checker exists, because that is the only situation that justifies its cost.

### Lesson 2 · Where does the feedback come from? Agents, search, vision, emotion

**The pivot question.** Every method in Lesson 1 differs in one practical way: *who scores the attempt, how fast, how cheaply, and how honestly?* Pick the scorer first and the method mostly picks itself.

**Six places a score can come from.** *(the category list is a textbook taxonomy; the verdicts about reliability are my judgement)*

| Source | Latency and cost | How it fails | Can the model game it? | Agent example |
|---|---|---|---|---|
| A **program** (checker) | Milliseconds, free after you write it | Your checker is wrong, or checks less than you meant | Hard, but possible (it finds the loophole you left) | Unit tests pass; the produced JSON validates |
| A **labelled dataset** | Slow and expensive to build, free to reuse | Labels go stale; the set leaks into training | Yes, if the model can see the set | The final answer matches a stored gold answer |
| A **judge model** | Fast, costs inference | Inherits the judge's taste, including its biases | Yes — this is the well-documented failure | A model rates the tone of a written report |
| The **environment** | Fast, but you must build it | Flaky environments look like algorithm failures | Not really — the world does not negotiate | The file really appears; the API call really returns 200 |
| **Domain measurement** | Depends on the domain | Measures a proxy, not the thing you want | Yes, via the proxy | Retrieval hit-rate for search; box overlap for vision; human agreement for emotion |
| **The model itself** | Free | Drifts toward "easy to score", not "correct" | By construction | Self-critique, or agreement between two of its own answers |

**Agents: the environment is the scorer.** For an agent, success is external and unambiguous — the test suite runs, the report file exists, the booking is confirmed. That is the cleanest signal in the table. You pay for it in engineering: the sandbox must start fast, reset cleanly and survive thousands of parallel attempts. The score also arrives only at the **end**, which creates the credit-assignment problem — which of the ten steps deserves the blame? That is [Chapter 6](../06-credit-assignment/index.md), and [Chapter 8](../08-environment-engineering/index.md) is the environment itself.

**Search, vision, emotion: the same question in three costumes.**

- **Search.** The cheapest honest signal is *did the retrieved document contain the answer*, checked against a stored label. It is noisy, and the labels rot as the world changes. *(my judgement)*
- **Vision.** Prefer an exact, geometric check (did the box overlap the true box; does the OCR text match character for character) over a judge's opinion — but beware: an exact-match rule will punish a transcription that is *more* correct than the label. Write the rule so the boundary is where you want it.
- **Emotion and other subjective properties.** There is no checker, by definition. You need a small human-labelled set even when the in-loop judge is a model, for one reason: without it you cannot tell whether the judge agrees with people. Measuring that agreement is [Chapter 11](../11-evaluation/index.md).

**Self-generated signals: use last, and with the lights on.** A model scoring its own output needs no humans and no checker, which makes it attractive. The failure mode is structural, not accidental: the training pushes the model toward answers it finds easy to defend, and "easy to defend" drifts away from "true". *(paper result, reported and not reproduced here)*

**The rule of thumb.** *(my judgement, and the boundary is: anything subjective sits below the line on purpose)*

> Prefer a **program**; then a **labelled dataset**; then an **audited judge**; then **self-generated** signal. Move down the list only when what you want cannot be checked, and never move down for convenience.

**Granularity is the second half of the decision.** A score can arrive once per run (outcome) or once per step (process). Outcome rewards are cheap and sparse; process rewards are informative and expensive, and they need their own model. Both are covered in [Chapter 5](../05-reward-engineering/index.md), and the way they interact with long agent runs is [Chapter 6](../06-credit-assignment/index.md).

### Lesson 3 · Amplifier vs injector: does RL have a ceiling?

**The claim.** Training with RL does not add knowledge. It *re-weights* what the model already does, moving probability toward the attempts that scored well. Data — not the optimiser — is what injects new capability. The practical consequence: **RL can only work with behaviours your model already produces, at least occasionally.**

**Make it checkable in one afternoon: pass@k.** Take your task and your *un-trained* model. Let it attempt the same question $k$ times and count how many attempts succeed. That number — pass@k — is the raw material RL will work with. *(textbook fact; pass@k is standard practice in code benchmarks)*

| What you measure | What it means | What to do |
|---|---|---|
| pass@1 = 0%, pass@8 = 0% | The behaviour is not in the model at all | Do **not** start RL. Add data, add a demonstration set, use a bigger model, or give the model a tool |
| pass@1 = 2%, pass@8 = 25% | The model can do it, but unreliably | RL is exactly the right purchase: raise the good path's probability |
| pass@1 = 60%, pass@8 = 95% | The model mostly does it already | Training buys little. Sample several times at inference and pick, or fine-tune a little to stabilise |
| pass@1 = 60%, pass@8 = 62% | Almost no diversity | Sampling cannot help either. The task may be too easy, or the sampler is too deterministic |

**Why zero success means zero learning.** In policy-gradient methods, an action's probability is raised in proportion to how much better it did than expected (its advantage). If every sampled attempt fails, every advantage is equal — and equal advantages cancel out. The gradient points nowhere, and training spins without progress. This is a mechanism, not a rule of thumb. *(textbook fact)*

**Sampling is often the cheaper half of the ceiling.** Before training, spend one day on inference-time effort: sample $n$ answers and pick the best, or let the model think longer. Reported result: on reasoning tasks, spending compute at test time can beat spending it on a bigger model, and the *way* you spend it matters more than the amount. *(paper result, reported and not reproduced; see Snell et al. under Reading)*

**Where the strong version of the claim breaks.** DeepSeek-R1 reports that longer reasoning chains and self-correction behaviour *emerged* during RL on verifiable maths problems — behaviour that was hard to find in the base model. *(paper result, reported and not reproduced)* The honest summary, and the one this book uses:

- **Reliable:** RL sharpens, selects and stabilises behaviour the model can already produce.
- **Unreliable and expensive:** RL sometimes appears to create new strategies. It happens on tasks with a graded, checkable signal and a lot of room to explore, and it is not something you can schedule.

**The ceiling has a second edge.** Amplification can cut both ways. Reported result: RLVR-style training can collapse output diversity, with pass@1 improving while pass@k gets *worse* — the model learns one good path and forgets the others. *(paper result, reported and not reproduced; see the R1-Zero-like training critique under Reading)* Practical consequence: measure both pass@1 and pass@k after training, or you will not see it happen.

**Two examples that make the boundary concrete.** *(my judgement)*

- The model already writes valid tool-call syntax but calls the tool at the wrong moment. RL fixes this: it amplifies *when* to call.
- The model has never seen your internal API in pretraining. RL will not teach it the API. Put the schema in the prompt, or fine-tune on demonstrations of the API being used.

**The point of this lesson.** Answer "is the behaviour present at all?" before you answer "which algorithm?". It is a twenty-minute experiment that decides whether the next two weeks are worth spending.

### Lesson 4 · A first intuition for compute cost: rollouts dominate

**Where the money goes.** In RL, most of the compute is *generation*: the model writes attempts so that they can be scored. The gradient step that follows is comparatively small. This is the opposite of SFT, where every token is used exactly once for training. *(textbook fact about the training loop)*

**The multiplier, in words, then in symbols.** Cost per iteration is driven by four numbers multiplied together: how many prompts you use, how many attempts you make per prompt, how many turns an attempt takes, and how long each turn is.

$$
\text{generated tokens per iteration} \approx \text{prompts} \times \text{samples per prompt} \times \text{turns} \times \text{tokens per turn}
$$

Every factor is easy to raise by accident. Eight samples is standard. An agent task with tool calls is easily four to ten turns. Multiply them and a modest-looking run has already grown two orders of magnitude past a single completion.

**A worked example with the arithmetic shown.** *(this is arithmetic on stated assumptions, not a measurement)*

| Input | Value |
|---|---|
| Prompts per iteration | 1,000 |
| Samples per prompt | 8 |
| Turns per sample (agent, with tool calls) | 4 |
| Tokens per turn (prompt + answer) | 500 |
| **Generated tokens per iteration** | **1,000 × 8 × 4 × 500 = 16,000,000** |

Sixteen million generated tokens at, say, one epoch's worth of iterations, before you count the scoring passes, the reference model and the retries. That is why "just run RL for a bit" is not a plan. Turn the tokens into hours only with **your own measured throughput on your own hardware** — the method is in [Chapter 9](../09-training-runtime/index.md), and the hardware ladder this book is calibrated against is in [Compute tiers](../appendix/compute-tiers.md). A throughput number quoted from someone else's machine is not evidence about yours.

**The second, quieter cost: waiting.** In a synchronous loop, the trainer sits idle while the model generates. Asynchronous designs overlap generation with training, at the price of training on slightly stale samples — which is a real trade, not a free win. *(reported practice, not reproduced here; the architecture is [Chapter 9](../09-training-runtime/index.md))* For a single-machine learner this shows up as the difference between a run that finishes overnight and one that finishes next week.

**Three purchases that are often cheaper than a training run.** *(my judgement)*

1. **Better feedback.** A cleaner checker or two hundred better task items usually beats another training epoch.
2. **Test-time sampling.** Best-of-$n$, self-consistency, or a longer reasoning budget — no training infrastructure at all.
3. **A narrower task.** A smaller model on a tighter problem often trains in an afternoon on hardware you already own.

**The comparison that matters for a beginner.**

| Method | What you pay for | Rough scaling relative to SFT |
|---|---|---|
| SFT | One pass over your demonstrations | 1× |
| DPO | One pass over static preference pairs, two models in memory | ~1–2× |
| RLVR with GRPO | Generation of many samples per prompt, no critic | tens of × |
| Full RLHF with PPO | Generation, plus reward model and critic training and serving | tens to hundreds of × |

**The decision you can now make.** The deliverable at the end of this chapter is one page: *my task → should I use RL at all → if yes, which family*. You now have every ingredient for it: the four families and their costs (Lesson 1), the scorer you can actually build (Lesson 2), the pass@k experiment that says whether training can work at all (Lesson 3), and the size of the bill (Lesson 4).

## Deliverable

A one-page draft: "my task → should I use RL at all → if yes, which family". It must name the feedback source you can build, and the pass@k number you measured.

## How you know you passed

- For each of the four families, you can say in one sentence what it fixes and what it costs.
- You can translate "RL is an amplifier" into a concrete statement about your own task.
- You can name a task where RL would be wasted money, and explain why.

## Reading

Every title and arXiv ID below was checked against the arXiv API on 2026-09-30 (IDs marked "verified"). Numbers quoted from these papers in this chapter are **reported, not reproduced**.

- **Preference RL at scale** — *Training language models to follow instructions with human feedback* (Ouyang et al., InstructGPT), arXiv:2203.02155 — verified.
- **The optimiser underneath** — *Proximal Policy Optimization Algorithms* (Schulman et al., 2017), arXiv:1707.06347 — verified.
- **A model as the judge** — *Constitutional AI: Harmlessness from AI Feedback* (Bai et al.), arXiv:2212.08073 — verified.
- **Preference tuning without a reward model** — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al.), arXiv:2305.18290 — verified.
- **Where GRPO comes from** — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al.), arXiv:2402.03300 — verified.
- **Verifiable rewards as a named practice** — *Tulu 3: Pushing Frontiers in Open Language Model Post-Training* (Lambert et al.), arXiv:2411.15124 — verified.
- **Reasoning from RL, and the emergent-behaviour claim** — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948 — verified.
- **RL for tool use** — *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs* (Feng et al.), arXiv:2504.11536 — verified.
- **The training runtime used later in this book** — *HybridFlow: A Flexible and Efficient RLHF Framework* (Sheng et al., VeRL), arXiv:2409.19256 — verified.
- **Test-time compute versus bigger models** — *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* (Snell et al.), arXiv:2408.03314 — verified.
- **The counter-argument on diversity and entropy** — *Understanding R1-Zero-Like Training: A Critical Perspective* (Liu et al.), arXiv:2503.20783 — verified.

## Backfill

- [x] Question 1: When should I reach for RL at all? — answered by Lesson 3 (pass@k decides) and Lesson 4 (the bill decides).

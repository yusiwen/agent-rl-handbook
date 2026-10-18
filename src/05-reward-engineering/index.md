# Chapter 5 · Reward Engineering and Verifiers

> **Compute tier**: ✅ | **Status**: draft — all nine lessons have prose; the project has since been built and run (four verifiers, 21 of 21 tasks covered, marking deterministic on every gate run, the two code tasks judged by a real container), and the numbers it produced are in Lesson 9 and the project README.
> **Project**: [Project 1 · Verifiable-reward infrastructure](../appendix/projects.md)
> **Rule of this book**: [Chapter 4](../04-alignment-algorithms/index.md) settled *where* a score comes from. This chapter is about designing that score so it measures the goal, and building the machinery that produces it.

## First, the gist

A reward is a score you give the model's attempt. Reward engineering is deciding **who gives that score, how often, and how precisely** — and every one of those three decisions has a price.

- **How often** is granularity. One number for a whole episode is cheap to produce and a slow teacher; a score for every step teaches faster and needs an oracle that can judge a step.
- **Who** is the source: a program that checks the answer, a model trained on comparisons, a model used as a judge, or the environment's own next state.
- **How precisely** is the difference between the three reward models you will meet: an **ORM (outcome reward model)** scores only the final answer, a **PRM (process reward model)** scores each step, and a **GRM (generative reward model)** returns text — a critique or a verdict — instead of a bare number.

The second half of the chapter is about the case where a program can do the scoring. A **verifier** marks an answer by running something: a test suite, a symbolic check, a comparison against a known result. That makes the reward cheap, exact on the slice it covers, and — this is the part beginners miss — *finite*: the model will learn to pass your checker, not to be right.

## What you'll be able to do

- Place any reward scheme on the dense-to-sparse spectrum and say what each position costs.
- Explain the difference between outcome, process and generative reward models, and pick one for a task.
- Design a mixed reward whose terms do not silently cancel out, and write down the trade-off their weights encode.
- Build a verifier for a new task: task definition, model calls, automatic marking, replayable dataset, evaluation loop.
- List the ways your verifier will be cheated *before* the model finds them.

## Lessons

- [ ] 1. Reward granularity: from a single final score to dense step-by-step supervision
- [ ] 2. Outcome vs process: ORM and PRM, and the interface they give to credit assignment
- [ ] 3. Generative reward models: from "here is a number" to "here is why"
- [ ] 4. Unsupervised and intrinsic rewards: implicit-process routes such as PRIME
- [ ] 5. Mixed rewards: what to do when a rule-based score and a model-based score disagree
- [ ] 6. **Training and refreshing the reward model itself**: preference pairs, generalisation, and how often to retrain it
- [ ] 7. Reward hacking and over-optimisation: causes, detection and defences
- [ ] 8. Inside the Verifiers framework: from a reward function to reward infrastructure
- [ ] 9. Project 1 hands-on: task definition, model calls, automatic marking, replayable datasets, evaluation loop

### Lesson 1 · Reward granularity

**The one decision everything else follows from.** A reward function is a contract: it takes an attempt and returns a number. Granularity is how many *decisions inside* that attempt get their own number. *(textbook fact for the definitions; the costs are my judgement)*

| Granularity | What gets scored | What you must build | What it costs |
|---|---|---|---|
| **Per episode** | One number for the whole attempt | A checker for the final state (tests pass, answer matches) | Nearly no credit information: a five-turn episode that failed tells the model almost nothing about which turn was wrong (this is [Chapter 6](../06-credit-assignment/index.md)'s problem) |
| **Per turn** | One number per turn or tool call | A judgement of "was this turn a reasonable move", or an environment that yields a next-state signal | You now need an oracle for *partial* progress — a much harder thing to define than "did the tests pass" |
| **Per step** | One number per reasoning step or action | Step labels (human or constructed) | Producing the labels costs more than the training; and each label is a proxy of its own |
| **Per token** | A value for every generated token | Nothing extra — the model's own log-probabilities give it | The signal is dense but *self-referential*: it says what the model already believes (Lesson 4) |

**The arithmetic, on stated assumptions.** *(arithmetic on stated assumptions, not a measurement)* A five-turn agent episode that emits about 2,000 tokens and receives a single final score produces **one** training signal for 2,000 decisions. The same episode scored per turn produces five signals. If your learning rate and batch size are unchanged, the second version takes roughly five times fewer episodes to see the same number of learning signals — and it required you to define, in code, what a good turn is. That definition is the whole risk: a dense reward is a *fast* teacher, and it is exactly as good as the definition you wrote.

**The theoretical guard rail, and why practice ignores it.** *(textbook fact — the shaping theorem, no arXiv ID, in Reading)* If you add a shaping term of the form $\gamma\Phi(s') - \Phi(s)$ to a reward — the difference of a *potential* between consecutive states — the set of optimal policies is unchanged: the shaping only nudges the path, not the destination. Almost nothing in practice has that form. A length penalty, a format bonus, a step-level judge: each one changes what "optimal" means, deliberately or not. The theorem is worth knowing for one reason — it tells you the difference between *guiding* a policy and *changing its goal*, and lets you say which one you are doing.

<figure class="book-figure">{{#include ../figures/05-granularity.svg}}</figure>

*Figure 5.1 — the same episode scored four ways. What moves down the rows is where the signal lands; what moves with it is how much of the work you own.*

**How to choose.** *(my judgement)* Start as sparse as you can and stay honest about the reason: add density only when you can name the failure the sparse signal cannot teach. "The model learns too slowly" is not that reason — a bigger batch is cheaper than a per-step oracle. "The model cannot tell which of its five turns lost the task" is.

### Lesson 2 · Outcome vs process

**Two reward models, named by what they see.** *(textbook fact for the definitions; the reported results are papers, in Reading)*

- An **ORM (outcome reward model)** sees only the finished attempt and predicts whether it is right — this is the verifier model of the early maths-word-problem work, trained on final answers.
- A **PRM (process reward model)** sees each step and scores it. The reported comparison that made PRMs famous trained both on the same maths data and found step-level supervision better, with the benefit growing as the number of samples used for search grew *(paper result, reported and not reproduced — Let's Verify Step by Step)*.

**Where the process labels come from, because that is the real cost.** Two routes:

1. **Human step labels.** Expensive, slow, and — the part that gets left out — *subjective*: annotators disagree about whether step 7 was "fine". The famous dataset scale was hundreds of thousands of labels for one domain *(same paper)*.
2. **Constructed labels.** Do not ask a human whether the step is good; *measure* it. Roll out several completions from that step and use the fraction that ends up correct as the label. This turns label production into compute, and it is what Math-Shepherd does *(paper result, reported and not reproduced — Math-Shepherd, in Reading)*. It gives you process supervision without process annotators, and it inherits the verifier's blind spots one level up: the step label is only as good as the final checker that produced it.

**The interface each one hands to credit assignment.** *(my judgement)* This is the sentence to take from the lesson:

- An **ORM** gives one number per episode. Something must spread it over the steps — that is [Chapter 6](../06-credit-assignment/index.md)'s whole job, and the spreading is where bias enters.
- A **PRM** gives a number per step, so the spreading is already done. What arrives instead is a *noisier* signal that can be wrong in structured ways: a PRM trained on constructed labels inherits the verifier's coverage limit, and it can be over-sensitive to surface features (a step that *looks* like progress).

**The reported caution.** Careful development work on PRMs in mathematical reasoning found that the common automatic labelling scheme is noisy, that PRM evaluations are easy to get wrong, and that a PRM can be systematically biased in ways the aggregate accuracy hides *(paper result, reported and not reproduced — The Lessons of Developing Process Reward Models, in Reading)*. Practical consequence: if you use a PRM, keep an outcome check as well, and treat PRM-vs-ORM disagreement as a signal to look, not a tie to break by preference.

**What each costs.** *(my judgement)*

| | ORM | PRM |
|---|---|---|
| Data | Correct/incorrect attempts — often free from the verifier | Step labels: human (expensive, subjective) or constructed (compute) |
| Signal | Sparse; credit assignment needed downstream | Dense; already per step |
| Failure mode | Slow learning on long tasks; the final score hides which step broke | Inherits the label source's bias; can reward plausible-looking steps |
| Where it pays | Short attempts, cheap checks, big batches | Long reasoning traces, and search at inference time (a PRM can rank partial solutions) |

### Lesson 3 · Generative reward models

**The move: make the judge explain itself.** *(textbook fact for the mechanism; results are papers)* A GRM (generative reward model) produces text — a critique, a reasoning trace, a verdict — instead of a scalar. The number your trainer consumes is then derived from that text, usually from a verdict token.

**Why text can beat a bare number.** Three reasons, in order of how much they matter: *(my judgement on the reasons; the reported result is Generative Verifiers, in Reading)*

1. **Deliberation before the decision.** A verdict emitted after a reasoning trace can use the trace; a scalar head has one forward pass of the same weights and no scratch space. The reported generative-verifier work casts verification as next-token prediction of a yes/no decision and uses the probability of that token as the reward, matching or beating discriminative reward models in the settings studied *(paper result, reported and not reproduced)*.
2. **Evidence you can read.** When a GRM's verdict looks wrong, you have its reason in front of you. With a scalar head you have a number and a shrug. This is the difference between an afternoon of debugging and a week.
3. **Reuse.** The same text can be fed back to the policy in a multi-turn setting (a critique to act on) instead of being reduced to a reward and thrown away.

**What you newly own.** *(my judgement)* A GRM adds a component that a scalar head did not need: **the parser**. Something must turn "the final answer is incorrect, because..." into a number, and now your reward function includes a string-matching rule. That rule is a new attack surface — a model that has learned the shape of your verdict strings can satisfy the parser without satisfying the judge. Keep the parser trivial, log both the text and the parsed value, and alarm on parse failures, because a silent parse failure is a reward of zero for a correct answer.

**Three cautions worth carrying over from [Chapter 4](../04-alignment-algorithms/index.md), Lesson 7.** *(paper results, reported and not reproduced)* A model judge has measurable biases: position, verbosity, self-preference. A judge can be improved with synthetic data rather than human labels, which makes good judges cheap and makes their biases cheap too. And a plausible explanation is not evidence of the real cause: a model's stated reasoning can be unfaithful to what actually drove its answer *(Language Models Don't Always Say What They Think, in Reading)*. A GRM's text is *evidence about the judge*, not a proof about the answer.

**What it costs.** *(my judgement)* Two to fifty times the compute of a scalar head per judgment, depending on the trace length; a parser to maintain; and a new place for the reward to be wrong *quietly* — a scalar head that is wrong shows up as a bad curve, while a GRM that is wrong shows up as a confident paragraph.

### Lesson 4 · Unsupervised and intrinsic rewards

**The situation: no labels, no checker, no budget for humans.** Two families of answers, and both are self-referential by construction. *(textbook fact for the mechanisms; the results are papers)*

**Family one — agreement and confidence.** Ask the same question several times and see whether the answers agree. If a majority answer exists, treat it as a pseudo-label and reward agreement with it; the mechanism is the same sampling that makes self-consistency work as a decoding trick *(paper result, reported and not reproduced — Self-Consistency, in Reading)*. Related signals: the model's own probability of its answer, the entropy of its distribution, and the spread across samples. All of them are *proxies for correctness that the model supplies about itself*.

**Family two — implicit process rewards.** The densest signal needs no trained PRM at all: the policy's own log-probability, compared against a reference, is already a per-token quantity. Under that reading the per-token log-ratio $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ *is* an implicit process reward, and preferring tokens the policy makes more likely after training is a dense update. PRIME (Process Reinforcement through Implicit Rewards) builds a process reward out of exactly that quantity and uses it to train, without step labels *(paper result, reported and not reproduced — PRIME, in Reading)*. It is the natural endpoint of [Chapter 4](../04-alignment-algorithms/index.md), Lesson 5's observation that the policy secretly carries a reward: here the implicit reward is made explicit and used per token.

**The trap, stated once and precisely.** *(paper result, reported and not reproduced; the consequence is my judgement)* A reported study found that RLVR (reinforcement learning with verifiable rewards) runs can improve benchmark scores even when the reward is *random* or deliberately wrong, when the base model already has the right prior. If a wrong reward can raise the score, then a self-generated reward is not evidence of anything except that the update happened. Two consequences for you: (1) whenever the reward is intrinsic, you need an *external* held-out measurement or you are measuring the model's self-agreement; (2) the cheapest external measurement is a verifier on a small slice — which is Lesson 8's argument for building one even if your main reward is a model.

**What it costs.** *(my judgement)* Intrinsic rewards are free to compute and expensive to trust. They amplify whatever the model already prefers — including its biases and its stylistic habits — and nothing in the reward curve distinguishes "the model got better" from "the model became more like itself".

### Lesson 5 · Mixed rewards

**Why anybody mixes.** *(textbook fact for the practice; the recipe is reported, in Reading)* No single signal covers what you want: correctness is precise and narrow, a judge is broad and noisy, format and length are cheap and dangerous. So production recipes combine them — Tulu 3, for instance, reports one training run using verifiable rewards alongside preference-style rewards *(paper result, reported and not reproduced — Tulu 3, in Reading)*.

**The four ways to combine, and the trap in each.** *(my judgement)*

| Mechanism | How it works | The trap |
|---|---|---|
| **Weighted sum** | $R = w_1 r_1 + w_2 r_2 + \dots$ | The weights encode a trade-off nobody wrote down. With no normalisation, the term with the largest range silently dominates: a judge scored 0–10 next to correctness scored 0 or 1 means the run is training the judge |
| **Gating** | Use the rule when it is decisive; ask the model only when the rule is inconclusive | The gate becomes a learned surface: the policy can learn to be "inconclusive" |
| **Veto** | A hard constraint, not a term: break the format and the episode scores zero | A veto discards the rest of the signal; used carelessly it teaches nothing but the format |
| **Normalisation** | Rescale each term to a comparable range before summing | Ranges move during training. A normaliser fitted on early data quietly changes the objective as the policy drifts |

**The interesting case is disagreement.** *(my judgement)* When a rule says "wrong" and a judge says "good", exactly one of three things is true, and each has a different fix:

1. **The rule is too strict** — a correct answer in an unexpected form (a different unit, an equivalent expression). Fix the rule or the answer extraction.
2. **The judge is rewarding surface** — length, confidence, a familiar structure. Fix the judge with a held-out set, not with a weight.
3. **The task is genuinely ambiguous**, and the pair of opinions is the honest output.

**How to find out which.** Do not average, and do not read the aggregate agreement: pull the *disagreements only*, hand-label them (a small set is enough, fifty is plenty), and count which side agreed with you. That is the highest-information measurement in the whole chapter, because the two signals agree almost everywhere else.

**And a stopping rule, or the mix will not save you.** *(paper result, reported and not reproduced — Scaling Laws for Reward Model Overoptimization, in Reading)* Optimising a proxy longer than it deserves degrades the thing you cared about, roughly smoothly and predictably, and the degradation starts before the proxy stops improving. So the composite reward needs a stopping criterion measured against gold data, not against itself.

**What it costs.** *(my judgement)* Each term is a tunable, and the composite is the thing you will debug at 2 a.m. Keep the whole reward in one function, in one file, versioned with the run — not spread across a config, a prompt file and three helper modules.

### Lesson 6 · Training and refreshing the reward model itself

**A reward model is a machine in your pipeline, and it needs its own maintenance.** *(textbook fact for the training recipe; the findings are papers)*

**What it learns from.** Preference pairs (the Bradley-Terry setup from [Chapter 4](../04-alignment-algorithms/index.md), Lesson 5) or correctness labels for a verifier-style model. Its data determines its expressiveness: a model trained only on "which answer do annotators prefer" cannot tell you anything about a dimension your annotators never varied.

**Where it breaks.** Two reported findings that say the same thing from different angles:

- **Over-optimisation is a curve, not a cliff.** As the policy is optimised harder against a learned reward, proxy reward rises while the true objective falls, and the peak of the true objective arrives before the peak of the proxy *(paper result, reported and not reproduced — Scaling Laws for Reward Model Overoptimization, in Reading)*. The gap opens when the policy moves off the reward model's training distribution.
- **Ensembles help, cheaply.** Averaging several reward models mitigates the same degradation in the studied settings *(paper result, reported and not reproduced — Reward Model Ensembles, in Reading)*. The reason is intuitive: independent errors do not survive an average, while a shared bias does.

**How to know whether your reward model is still good.** Not by its training loss. Measure **agreement with held-out human or gold labels** on your own task — a public suite such as RewardBench gives a sanity number and a way to compare architectures *(paper result, reported and not reproduced — RewardBench, in Reading)*; only your own holdout decides whether it is good for *your* reward. Re-measure when the policy has moved, because that is when off-distribution starts.

**Refreshing, as a trigger rather than a calendar.** *(my judgement)* The refresh loop is: collect fresh comparisons from the *current* policy, retrain (or continue training) the reward model, re-measure agreement, repeat. Two triggers are worth setting: agreement on the holdout dropping below the level you measured at the last refresh, and the policy's outputs drifting into behaviour the reward model was never shown. A fixed "retrain weekly" schedule satisfies nobody: too slow when the policy moves fast, wasted work when it does not.

**What it costs.** *(my judgement)* A second model to train, evaluate, version and serve; its own data pipeline; and the discipline to re-measure it. In exchange you get the thing a fixed reward cannot give you: a score that keeps up with a policy that is changing underneath it.

### Lesson 7 · Reward hacking and over-optimisation

**The mechanism, restated once more because it is the chapter's spine.** *(textbook fact for the mechanism; the framing is my judgement)* A reward is a proxy. Optimisation is a search that does not know your intent. Therefore: given enough steps, the search finds the places where the proxy and the goal disagree. This is not a bug in your code and it is not malice; it is the failure mode of every imperfect specification, named early in the safety literature *(paper result, reported and not reproduced — Concrete Problems in AI Safety, in Reading)* and measured at scale since.

**A taxonomy by what gets exploited.** *(my judgement, assembled from the reported findings below)*

| The exploited thing | How it looks | The reported evidence |
|---|---|---|
| **The checker** | Partial tests pass; the answer is "right" for the wrong reason | The RLVR boundary work (Lesson 4); any test suite with holes |
| **The judge** | Longer, more confident, more agreeable answers score better | LLM-as-a-judge biases, and the sycophancy study (both verified in Chapter 4's Reading) |
| **The environment** | Injection inside the task; stale state; a tool that exceeds its authority | The indirect-prompt-injection study (verified in the elective module's Reading) |
| **The format** | Answer-only shapes, delimiter tricks, "safe" refusals that count as harmless | The over-optimisation curves above; spurious-reward findings |
| **The metric** | Learning which items are easy and answering only those | Pass-rate distributions in [Chapter 7](../07-data-and-tasks/index.md) |

**Detection, in the order of cost.** *(my judgement)*

1. **A frozen gold set, measured before training starts.** Nothing else catches "the proxy went up and quality went down".
2. **Disagreement audit** (Lesson 5): the rule-vs-judge disagreements are where a hack shows up first.
3. **Canaries**: items you know are gameable. If the policy starts passing them, the hack is live.
4. **Monitors on the shape of the output**: length distribution, entropy, refusal rate, repetition. Degeneration is visible before quality metrics move.
5. **Reading traces.** Twenty outputs read by a human beats a dashboard, at least once per run.

**Defences, ordered by what they cost you.** *(my judgement, with the reported items cited)*

| Defence | What it buys | What it costs |
|---|---|---|
| KL (Kullback-Leibler) leash to a reference model | Keeps the policy inside the region where the proxy means something | Score you could have had ([Chapter 4](../04-alignment-algorithms/index.md), Lesson 2) |
| Reward clipping and normalisation | Stops single extreme events dominating | Sensitivity to the clipping range; a hidden hyperparameter |
| Reward-model ensembles | Independent errors average out | N times the serving cost |
| Adversarial example mining | Harden the judge against the hacks you have already seen | A loop to run: collect hacks → label → retrain |
| Stopping on the gold metric | Prevents the second half of the optimisation curve | Fewer steps than your budget allows |
| Production monitoring | Catches the hacks that appear after you ship | A pipeline that outlives the experiment |

**One boundary worth stating.** *(paper result, reported and not reproduced — Universal and Transferable Adversarial Attacks, in Reading)* Model judges are promptable, and small adversarial perturbations reliably move their verdicts. If your reward is a model, it is attackable by construction; the question is only whether your policy will find the attack by accident. A program that checks a fact does not have this property, which is the entire reason Lesson 8 exists.

**A checklist to run before a training run.** Write the answers down; the act of writing is the defence.

- What is the proxy, and what is the goal it stands in for?
- What is my held-out gold measurement, and when was it frozen?
- Which three hacks am I most afraid of, and which monitor would notice each?
- Where do I stop, and on which metric?

### Lesson 8 · Inside the Verifiers framework

**The turn this lesson makes.** *(my judgement)* A reward *function* is twenty lines: take an answer, return a number. A reward *service* is a contract: a schema, a queue, a timeout policy, a cache, a version number and a dataset you can replay. Most of the engineering in this chapter lives in that difference, and it is the difference the project names.

**What the framework provides, in the abstract** — the concrete API belongs to the project notes, and it changes between releases; the abstractions do not *(my judgement, reading the framework's own description — Verifiers, in Reading)*:

- **Tasks, not just rows**: a dataset of prompts plus the per-task code that turns a completion into a score, so the reward travels with the data.
- **Rollouts as a first-class operation**: the framework calls your model and hands you the completions, so the loop (generate → mark → recover the reward) is not something you rebuild per project.
- **Pluggable reward units**: a rule-based verifier, a model judge, and a mix (Lesson 5) behind one interface, so a training run does not care which kind it is talking to.

**The properties any verifier service needs, whatever it is built on.** *(my judgement)* This list is the reusable part:

1. **Determinism.** Same input, same mark. Anything else makes your training data unreproducible.
2. **Isolation.** Marking code execution means running untrusted code; it needs a sandbox and a resource limit, not good intentions ([Chapter 8](../08-environment-engineering/index.md)).
3. **Timeouts and a policy for them.** A task that hangs must score something defined — zero, or "unknown, exclude" — and the choice must be written down, because it is a reward decision.
4. **Caching.** A mark is expensive; the same completion will be scored more than once.
5. **Versioning.** Every dataset row should carry the verifier version that produced it, or the run is not reproducible.
6. **Replayability.** You must be able to re-score an old rollout set with a new verifier — that is how you audit an old run without retraining.

**Where the bugs live.** *(my judgement)* In the checker (a program with edge cases), in the plumbing (timeouts, partial writes, a queue that silently drops), and in the dataset (drift between the verifier that marked a row and the verifier that is training now). Note that all three are ordinary software bugs — which is the point: a cheap reward is not free, it moves cost from labelling to infrastructure, where it is visible and testable, and that is the trade worth making.

### Lesson 9 · Project 1 hands-on

**The problem, in one sentence.** Turn "was the answer right?" into an automatic, scalable training signal for maths, code and logic tasks — and know exactly where that signal lies. *(project overview for this chapter; the build itself is [Project 1](../appendix/projects.md))*

**The architecture, five parts** (Figure 5.2):

1. **Task definition** — the prompt, the expected answer form, and the marking rule for one task family. This is where most of the thinking happens; the rest is plumbing.
2. **Model calls** — a local model served behind an OpenAI-compatible endpoint, called in batches, with the sampling parameters written down (a mark is meaningless if you cannot say which model and temperature produced the completion).
3. **Automatic marking** — the verifier itself: extract the answer, compare it, return a score *and a reason*. The reason is for you; the score is for the trainer.
4. **Replayable dataset** — every `(task, completion, mark, verifier version)` row, stored, so an old run can be re-scored and audited.
5. **Evaluation loop** — marking a fresh batch from the same distribution and checking that the marks are stable, plus the failure-boundary list from the Deliverable.

<figure class="book-figure">{{#include ../figures/05-verifier-loop.svg}}</figure>

*Figure 5.2 — the verifier loop as a service: what each stage produces, and the three places the signal goes wrong (the checker, the plumbing, the dataset).*

**The engineering decisions, and the trap in each.** *(my judgement)*

| Decision | The trap |
|---|---|
| Which tasks enter the suite | A suite that only contains items your model cannot solve teaches nothing; one that only contains easy items teaches nothing either (the pass-rate band argument from [Chapter 4](../04-alignment-algorithms/index.md), Lesson 8) |
| How strict the answer extraction is | Too strict, and correct answers score zero — you train on noise. Too loose, and a parser-shaped answer scores one — you train a hack |
| Whether marking runs in the training loop or offline | Inline is simpler and couples your throughput to the verifier's; offline needs the replayable dataset to stay honest |
| What a timeout scores | An undefined choice here becomes a silent reward |

**What "done" looks like, for real.** *(my judgement)* Not "the code runs". The verifier marks a fresh batch with the same result it gave last time, the dataset contains both passing and failing examples (a verifier that never says no has not been tested), three cheating routes are written down with the monitor that would catch each, and the loop prints the recovered rewards end to end.

**What the project measured.** *(my own measurements, 2026-10-13 and 2026-10-14; the harness is `p1-verifiers` in the companion repository)* The suite that [Chapter 7](../07-data-and-tasks/index.md) then tests a task set with was built here, and four of its numbers belong to this lesson rather than to that one:

- **Coverage is 21 of 21 tasks**, and the gate asserts it. A task nothing can score is a ceiling on what training can add, so an edit that drops one fails the check instead of quietly lowering the ceiling.
- **Marking is deterministic, and the determinism is asserted.** The same 42 attempts are marked 21 passing and 21 failing on every run of `check_p1.py`. That is the "same marking twice" requirement from *How you know you passed*, written as a number that moves if the rule changes.
- **The two `unit_test` tasks are judged by a real container** - `16` attempts in the full run, with `--network=none` and a memory cap. On a machine without a container runtime the run stops with an error rather than scoring zero, because "this machine cannot run the tests" is not the model's fault.
- **Where each verifier lies is written down per verifier**, and one of those boundaries was measured rather than guessed. The numeric verifier used to score any number it found on the last line, which made it a *format* test wearing an arithmetic test's clothes: requiring the answer to be declared moved one task from 0 of 4 to 4 of 4 and a different one from 4 of 4 to 0 of 4. Both movements are recorded in Chapter 7, Lesson 4, because they turned out to be facts about the task set rather than about the marking.

The run records and the gate are in the project; the numbers above come from running it, not from reading it.

## Project

**Project 1 · Verifiable-reward infrastructure** — turning "was the answer right?" into an automatic, scalable training signal. Full description and the six-project map: [the project index](../appendix/projects.md).

| Stage | What it produces | What goes wrong here |
|---|---|---|
| Task definition | A task family with a written marking rule | The rule is ambiguous, or covers only the answers you imagined |
| Model calls | Batched completions from a local model | Sampling parameters unrecorded; the model version changes under you |
| Automatic marking | A score plus a reason per completion | The checker's edge cases; parse failures counted as zero |
| Replayable dataset | `(task, completion, mark, verifier version)` rows | Version not recorded; the dataset drifts from the verifier |
| Evaluation loop | Stability check on a fresh batch + the failure boundaries | Only ever tested on the batch it was built from |

**Environment.** A small local model behind an OpenAI-compatible server (the Spark is comfortable with this tier), plus the Verifiers framework *(Verifiers, in Reading)*. Write the environment, data and config notes into the project's own note file, using the nine-face template in [the project index](../appendix/projects.md) — this book keeps project logs there, not in the chapter.

**Sequencing.** Run the loop on a task family you can already solve by hand, with a checker you trust, before touching a model you care about: the first version's job is to prove the plumbing, not to train anything.

## Deliverable

- A verifier for a new task that marks answers automatically and reproducibly.
- A replayable marking dataset containing **both passing and failing examples**.
- A **failure-boundary list** for that verifier: when will it give the wrong signal? At least three entries, each naming how you would notice it happening.

## How you know you passed

- You re-ran on a fresh batch from the same distribution and the marks were stable — and you can state the batch size and the agreement rate you measured.
- You wrote down at least three ways an agent could cheat this verifier, and for each one the monitor that would catch it.
- The RLVR loop runs end to end and prints the recovered rewards: a command, its output, and the model and parameters that produced it.
- Your failure-boundary list names one thing your verifier deliberately does *not* check, and what that means for the training signal.

## Reading

Reused entries were verified where they say; new entries were checked against the arXiv API on 2026-10-09 (IDs marked "verified"). Numbers quoted from papers are **reported, not reproduced**; references with no arXiv ID are named as such.

The framework this chapter's project builds with:

- **Verifiers** (Prime Intellect) — <https://github.com/PrimeIntellect-ai/verifiers> — a working implementation of the reward-service pattern in Lesson 8. Treat the API as documentation to look up, not as something to memorise.

Already verified in earlier chapters — reuse these rather than re-checking:

- **Verifiers and outcome supervision, the origin** — Cobbe et al., *Training Verifiers to Solve Math Word Problems*, arXiv:2110.14168 — verified in Chapter 2's Reading, 2026-09-30; the ORM in Lesson 2.
- **Process supervision, and the comparison that named it** — Lightman et al., *Let's Verify Step by Step*, arXiv:2305.20050 — verified in Chapter 2's Reading, 2026-09-30; the PRM and the search benefit in Lesson 2.
- **Reward overoptimisation** — Gao, Schulman and Hilton, *Scaling Laws for Reward Model Overoptimization*, arXiv:2210.10760 — verified in Chapter 2's Reading, 2026-09-30; the curve in Lessons 5, 6 and 7.
- **RLVR in practice, and mixed rewards in one run** — Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*, arXiv:2411.15124 — verified in Chapter 1's Reading, 2026-09-30; Lesson 5.
- **Spurious rewards: when the reward is not doing the work** — Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, arXiv:2506.10947 — verified in Chapter 4's Reading, 2026-10-09; the trap in Lesson 4.
- **Judge biases** — Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685; Sharma et al., *Towards Understanding Sycophancy in Language Models*, arXiv:2310.13548 — both verified in Chapter 4's Reading, 2026-10-09; Lesson 7's taxonomy.
- **The safety framing** — Amodei et al., *Concrete Problems in AI Safety*, arXiv:1606.06565 — verified in Chapter 4's Reading, 2026-10-09; the mechanism in Lesson 7.
- **An agent task written as a multi-turn MDP (Markov decision process)** — Zhou et al., *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL*, arXiv:2402.19446 — verified in Chapter 2's Reading, 2026-09-30; the per-turn granularity in Lesson 1.

New here, checked on 2026-10-09:

- **Reward shaping, and when it does not change the goal** — Ng, Harada and Russell, *Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping*, ICML (1999) — no arXiv ID; the potential-based shaping theorem in Lesson 1.
- **Process labels without process annotators** — Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*, arXiv:2312.08935 — verified. Lesson 2's constructed labels.
- **What goes wrong when you build a PRM** — Zhang et al., *The Lessons of Developing Process Reward Models in Mathematical Reasoning*, arXiv:2501.07301 — verified. Lesson 2's caution.
- **Generative verifiers** — Zhang et al., *Generative Verifiers: Reward Modeling as Next-Token Prediction*, arXiv:2408.15240 — verified. Lesson 3.
- **Judges improved without human labels** — Wang et al., *Self-Taught Evaluators*, arXiv:2408.02666 — verified. Lesson 3, caution 2.
- **Unfaithful reasoning** — Turpin et al., *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*, arXiv:2305.04388 — verified. Lesson 3, caution 3.
- **Agreement as a pseudo-label** — Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, arXiv:2203.11171 — verified. Lesson 4, family one.
- **Implicit process rewards** — Cui et al., *Process Reinforcement through Implicit Rewards* (PRIME), arXiv:2502.01456 — verified. Lesson 4, family two.
- **Reward-model ensembles** — Coste et al., *Reward Model Ensembles Help Mitigate Overoptimization*, arXiv:2310.02743 — verified. Lessons 6 and 7.
- **How to evaluate a reward model at all** — Lambert et al., *RewardBench: Evaluating Reward Models for Language Modeling*, arXiv:2403.13787 — verified. Lesson 6.
- **Test-time verification, and what it is worth** — Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, arXiv:2408.03314 — verified. The search-and-verify family that a PRM or GRM makes possible.
- **How attackable a model judge is** — Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*, arXiv:2307.15043 — verified. The boundary at the end of Lesson 7.
- **Where verification comes free** — Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, arXiv:2310.06770 — verified (also in the elective modules' Reading). The reason code is the mature case.

## Backfill

- [x] Question 2: How do I design the reward? — this chapter is the main battlefield, and it settles four things: **granularity** (Lesson 1, with the shaping theorem as the guard rail), **the three reward-model shapes** (Lessons 2 and 3) plus the self-generated alternative (Lesson 4), **how to combine signals without one silently dominating** (Lesson 5), and **how to keep the whole thing honest** — refresh the reward model against a held-out measurement (Lesson 6), run the hack checklist before the run (Lesson 7), and prefer a program over a model wherever a program can decide (Lesson 8, and the project).
- [x] Question 1: When should I reach for RL at all? — partially, from the other side: a task with no cheap scoring source (Lessons 4 and 5) is a task where the RL route is expensive or self-referential, which is the practical form of "not yet".

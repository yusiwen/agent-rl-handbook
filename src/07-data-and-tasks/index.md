# Chapter 7 · Data and Task Engineering

> **Compute tier**: ⚠️ (generating and filtering needs inference; small scale is fine on the Spark) | **Status**: draft — all six lessons have prose, and every measured number in them carries the configuration that produced it; the contamination check against a held-out evaluation set has not been run yet.
> **Why this chapter exists**: the source course has no standalone data chapter, yet bad data is the most common reason "I swapped the algorithm and nothing improved".

## First, the gist

Training can only learn from the examples you give it. If every task is either trivially easy or impossibly hard, the model gets no useful signal — a score of 100% or 0% tells it nothing about what to change. So before touching algorithms, run a health check on your task set: how hard is each item, how varied are they, and are any of them secretly copies of your test set?

The check has four parts, and each one is a data decision that is usually mistaken for a code decision:

- **A difficulty distribution.** An item solved 8 times out of 8, like one solved 0 times out of 8, is dead weight: neither can produce a gradient. The items that teach are the ones in between — the **trainable band**.
- **A scorer, because a difficulty is a measurement.** A pass rate is produced by a verifier, so an item nothing can score is a ceiling on what training can add.
- **A cold start.** RL cannot improve behaviour that never happens. A small supervised set — the cold-start SFT (supervised fine-tuning) set — teaches the *shape* of a scorable attempt before any RL run starts.
- **A separation.** Whether your items, or near copies of them, also sit in the set you report your score on.

The second half of this chapter is a lab notebook: 22 candidate tasks measured on one machine with one configuration, at two sampling temperatures. Two readings changed my mind while I was writing this, and both corrections are kept in place — one about a prompt's closing sentence, and one about how many temperatures a difficulty claim needs before it is a claim at all.

## What you'll be able to do

- Explain, with numbers from your own task set, why an over-easy or over-hard set produces no learning.
- Build the cold-start supervised data that makes later RL trainable in the first place.
- Screen a task set for difficulty distribution and test-set contamination.
- Filter the model's own attempts into training data without turning the set into a copy of what the model already does.
- Decide when to generate tasks programmatically instead of writing them by hand, and say what the generation pipeline's filtering stage actually costs.

## Lessons

- [ ] 1. Task distribution sets the ceiling: three real cases where changing the algorithm changed nothing
- [ ] 2. Cold-start SFT data: filtering, format, and the **label protocol** that decides whether RL can train at all
- [ ] 3. Rejection sampling and self-play data
- [ ] 4. Difficulty and curriculum: **pass-rate filtering** (too easy and too hard both give zero gradient)
- [ ] 5. Scaling the task set: programmatic task generation and environment synthesis
- [ ] 6. Decontamination and keeping the evaluation set separate

### Lesson 1 · Task distribution sets the ceiling

**A task set is a curriculum, whether you designed one or not.** *(textbook fact for the gradient mechanism; the framing is my judgement)* A policy-gradient update needs samples of the *same* task to disagree about the outcome. If every sample of an item earns reward 1, the group's average is 1, every advantage is 0, and the update it produces is 0. An item that never passes does the same thing from below. This is the group-relative machinery of [Chapter 4](../04-alignment-algorithms/index.md), Lesson 4, read from the data side: **the only items that teach are the ones the policy sometimes solves.** No algorithm change repairs a set where that set is empty.

**Why the numbers below are worth reading at all.** Every measurement in this chapter comes from one machine and one configuration, and a number without its configuration is a story. *(my own measurements; the harness is Project 1 in the companion repository `agent-rl-handbook-projects`, whose live runner is `p1-verifiers/checks/run_p1_live.py`)*

| What | Setting |
|---|---|
| Hardware | DGX Spark (NVIDIA GB10, 128 GB unified memory, aarch64 + sm_121) |
| Model | Qwen/Qwen3-1.7B |
| Server | vLLM v0.27.1; chat-template thinking disabled; `max_model_len` 8192 |
| Sampling | temperature 0.7 and 1.2; 8 samples per task in the probe, 4 in the stability run |
| Token counts | read from the server's own `/tokenize` endpoint, never from a host-side tokenizer |
| Scoring | the four verifiers of Lesson 4, through Project 1's task and verifier interface |
| Provenance | the two harness defects in Lesson 5 were found live in this setup |

**Three cases where the algorithm was blamed and the data was at fault.** *(my own measurements; each one is reproduced below with its numbers)*

**Case 1: the five candidates that a single temperature approved.** The first pass over the candidate set ran at temperature 0.7 only. Five candidates looked like they sat in the band, and all five were adopted into the task set on that evidence. The second temperature put two of them at 8 of 8: solved every single time, so free, so no signal. Both were taken back out. The model, the server, the sampler and the verifier were identical between the two readings; one sampling temperature differed.

**Case 2: the sentence that was worth three times the pass rate.** Seven word problems, the same answers, the same sampler, three prompts that differ by one closing sentence. The first arm passed 11 of 42 attempts; the third passed 37 of 42. Nothing a hyperparameter sweep would touch changed. The prompt did. The three prompts and the reading are in Lesson 2, because they are also the clearest evidence for what a cold-start set is for.

**Case 3: the task that improved between runs.** One task's pass rate moved 0 of 8, then 2 of 4, then 3 of 4, then 3 of 4, across four readings taken at different times. The natural reading is "this task got easier" or "something is drifting". The measured reading is thinner: inside a single run, that task's samples flipped reward between the two rounds of Lesson 4's stability check, so part of the movement is the sampler's own coin flips. A rate computed from 4 to 8 samples is an interval, not a number.

**What the three cases have in common.** *(my judgement)* In each one, the number that moved was produced by something other than the thing being tuned: a second temperature, one sentence of the prompt, and the sampler. Each check was cheap, and each had to run *before* the conclusion was drawn. **The data is part of the experiment**, and it is the part that is easiest to change without noticing.

**The ceiling, stated once, from a paper that measured it.** *(paper result, reported and not reproduced — Endless Terminals, in Reading; verified against the arXiv abstract page on 2026-10-14)* That paper opens with the sentence "Environments are the bottleneck for self-improving agents", and its evidence is the shape a real pipeline takes: 3,255 generated terminal tasks, a completion test per task, a solvability filter, and then a *vanilla* PPO (proximal policy optimisation) run with a binary per-episode reward. The algorithm stayed boring, the environment scaled, and the models moved — on the paper's held-out dev set from 4.0% to 18.2% (Llama-3.2-3B), 10.7% to 53.3% (Qwen2.5-7B) and 42.6% to 59.0% (Qwen3-8B-openthinker-sft). Read it as a ceiling rather than a recipe: your set bounds what any algorithm can learn from it, and it is sometimes cheaper to fix the set than the algorithm.

**The order of operations, and the price of the check.** Score every item with a baseline policy, classify it as wall, band or free, read the misses to find out whether a "wall" is arithmetic or format (both in Lesson 4), fix the composition, and only then train. The check is small and countable: 22 items × 8 samples × 2 temperatures = 352 generations of a 1.7B model *(arithmetic on stated assumptions, not a measurement)*, against training runs that generate orders of magnitude more.

### Lesson 2 · Cold-start SFT data

**What a cold start is, and why RL cannot skip it.** *(textbook fact for the mechanism; the numbers are my measurements)* As Lesson 1 said, an update needs reward spread inside a group of samples for one task. A policy that never emits an answer the verifier can read scores 0 on every sample of every task, so every group is flat and every advantage is 0. Training then runs, logs a loss, and changes nothing. The **cold start** is the small supervised stage that ends that state of affairs: it teaches the shape of a scorable attempt — where the reasoning goes, where the number goes, and what a declared answer looks like. The pattern is established practice, not a trick of mine: ReTool reports a cold-start supervised stage followed by an outcome reward as the way to teach a model *when* to run code *(paper result, reported and not reproduced — ReTool, in Reading)*.

**The label protocol is part of the reward, not the prompt's cosmetics.** *(my judgement, from the measurement below)* Project 1's numeric verifier has a rule that reads like a formatting requirement and behaves like a reward definition: **the answer must be declared** — the final line carries exactly one number, or the value is never scored. So "the model knows the answer" and "the verifier can see the answer" are two different claims, and only the second one produces a reward.

**The measurement: one sentence, three pass rates.** Seven word problems, the same answers, the same sampler (temperature 0.7, 6 samples per problem, so 42 attempts per arm), and three task prompts that differ in the closing sentence alone: *(my own measurement)*

| The prompt's closing demand | Passed | Declared an answer |
|---|---|---|
| "End with the number alone on the last line." | 11 of 42 | 11 of 42 |
| "End with the answer on its own last line: one number, nothing else on that line - no words, no units, no LaTeX." | 13 of 42 | 42 of 42 |
| "End your reply with exactly one final line, in the form ANSWER: [number]." | 37 of 42 | 37 of 42 |

**Four readings.** *(my judgement on the readings; the numbers are the measurement above)*

1. **A demand moves the declaration rate first, and the pass rate only later.** Arms 1 and 2 declare in 11 of 42 and 42 of 42 attempts, while the pass rate barely moves (11 to 13). Asking the model to declare an answer is cheap; making the declaration scoreable is a different act.
2. **The pass rate follows when the demand is mechanical — a template with a slot.** Arm 3 names the exact shape of the final line, and both numbers rise together: 42 of 42 declared, 37 of 42 correct.
3. **A prohibition does not work.** Arm 2 forbids words, units and LaTeX on the final line. The model complied by stripping its working and answered wrongly more often than the plain prose of arm 1 had, so the pass rate did not follow the declaration rate. A negative instruction changes what the model emits without telling it what to emit instead; the fix is a template, not a veto.
4. **A stronger demand buys readability by spending signal.** At 37 of 42 the pass rate is 88%, and six of the seven problems are flat at 6 of 6. Arm 3 is the best *protocol* of the three and the worst *training set*: what is left to learn from lives in a single item.

**The correction I owe from an earlier reading.** *(my judgement, corrected against the measurement)* The first reading of this same data was "the model computes correctly but will not declare the answer", and it was too generous. Forced to declare exactly one number, the model was wrong in most of those attempts. The extra prose had been hiding a wrong answer, not withholding a right one. Two consequences: a demand that makes the model commit turns a measurement of prose into a measurement of arithmetic, and a pass rate computed from attempts that were never scored is not a pass rate at all.

**What actually goes into the cold-start set.** Three rules: *(my judgement)*

1. **Keep only attempts the verifier scored.** An unscorable attempt is a bug report about the label protocol, not a training item.
2. **Keep the format, including the declaration, exactly as the verifier reads it** — including the string the verifier parses for. A supervised example that declares the answer differently from the way the reward reads it teaches the wrong protocol.
3. **Do not keep only the successes.** That is the filter bias of Lesson 3, and a cold-start set is where it starts.

**On volume, and on what this stage cannot teach.** The reported alignment work behind LIMA found 1,000 curated examples beating bulk instruction data, because surface form is what imitation copies most cheaply *(paper result, reported and not reproduced — LIMA, in Reading)*. The boundary has a clean statement too: supervised fine-tuning memorises, and generalisation is what the RL stage adds *(paper result, reported and not reproduced — SFT Memorizes, RL Generalizes, in Reading)*. A cold-start set expected to teach the *skill* is being asked for the wrong thing.

**What it costs.** *(my judgement)* The cold-start stage buys trainability and spends diversity. A set drawn from one template teaches the template, and the failure looks like a model that writes `ANSWER: ` for every question, including the ones it should have reasoned through. The guard rail is to keep the working visible in the supervised examples and put the demand only in the last line — arm 2 above is what happens when the demand is read as an instruction to delete the reasoning. Hold out a format variant, and check that the model still reasons when the final line changes.

### Lesson 3 · Rejection sampling and self-play data

**Rejection sampling, in one paragraph.** *(textbook fact for the mechanism; the prices are my judgement)* Sample several attempts per task from the current policy, score them with the verifier, keep the ones that pass, and train on the kept set. That is the whole loop, and it is how most synthetic supervised data is made. It is the same sampling that self-consistency uses for a different purpose — taking the majority answer as a pseudo-label ([Chapter 5](../05-reward-engineering/index.md), Lesson 4) — and it is the cheapest way to turn a scorer into data.

**The three decisions, and what each does to the set.** *(my judgement)*

| The decision | The cheap answer | What it does to the set |
|---|---|---|
| Which attempts to keep | the ones the verifier passes | the set becomes a picture of what the policy already does well |
| How many attempts per task | as many as the budget allows | easy items supply most of the data, because they pass more often |
| What the failures are for | throw them away | the reason the policy fails — the only genuinely new information in the sample — is deleted |

**A filter is a label, not a deletion.** *(my judgement)* Keep the failures, with their verification detail: which constraint failed, which number was declared, whether anything was declared at all. A failed attempt plus the verifier's stated reason is the most informative row you can hold, and it costs nothing extra — the attempt was already generated. It is also the raw material for [Chapter 6](../06-credit-assignment/index.md)'s critique signals, and for the "why each miss missed" reading in Lesson 4.

**The style trap.** *(my judgement)* Training on the policy's own successes narrows the distribution toward its habits: at one turn the phrasing variety collapses, and over many turns it becomes the Echo Trap of [Chapter 6](../06-credit-assignment/index.md), where reasoning degenerates into repeated templates. Watch the length distribution and the number of distinct opening phrases — the reward curve looks healthy throughout.

**Self-play data is a loop, not a corpus.** *(my judgement; the loop itself is Chapter 6's project)* The task set stays fixed and the attempts are regenerated from the current policy each round, so the data tracks the policy instead of drifting behind it. Its precondition is a scorer that marks a *fresh* attempt automatically. Without one there is no filter, and what you have is a diary rather than self-play. The discipline from [Chapter 6](../06-credit-assignment/index.md), Lesson 6 applies unchanged: round N+1's data must come from round N's policy, or you are training on another model's behaviour and calling it self-improvement.

**Before you filter anything, measure coverage.** *(my judgement)* An item nothing can score contributes nothing: it can never produce a reward, so it can never produce a gradient. Project 1's verifier suite scores 21 of 21 fixture tasks, and that coverage number is the precondition for every other number in this chapter — Lesson 4 lists the four verifiers and the rule each one encodes.

**Keep controls.** Three of the 22 items in the probe were not candidates at all but controls: items an earlier run had shown to move. *(my own measurement)* Their purpose is diagnostic. If a control reads as a wall at both temperatures, the harness changed rather than the tasks, and every other reading in the same batch is suspect. A control with a known reading is the cheapest instrument in this chapter.

**What it costs.** *(my judgement)* Filtering costs one verifier pass per attempt and shrinks the yield, and the yield is not uniform: the items you most want data for are the ones that pass least. So the budget line to watch is **samples per usable item**, not samples — and the kept set is smaller and more homogeneous than the one you sampled from, which is the style trap arriving through the budget.

### Lesson 4 · Difficulty and curriculum

**The band.** *(textbook fact for the gradient argument; the boundaries are a convention, and the arithmetic is on stated assumptions)* An item whose pass rate is very low or very high teaches nothing, for the reason in Lesson 1. The working convention is to keep the items whose pass rate sits between **0.1 and 0.9** — the **trainable band** — and the convention has a hidden dependency worth naming: at 8 samples per item the resolution is 0.125, so "0.1 to 0.9" means "at least 1 of 8 and at most 7 of 8"; at 4 samples the grid is 0.25, so the band becomes 1, 2 or 3 of 4. Your sample count sets the band's edge, which is a second reason to state the sample count beside every rate.

**The measured set.** 22 candidate tasks: 14 arithmetic questions, 5 logic puzzles whose unique solution was verified by enumerating all assignments, and 3 controls that earlier runs had shown to move. Eight samples per task at temperature 0.7 and again at 1.2, on the configuration in Lesson 1. *(my own measurement)*

| Where the item sits | How many of the 22 | The items |
|---|---|---|
| **In the band at both temperatures** | 6 | a three-colour logic puzzle; a unit-price word problem; an average; a greatest-common-divisor question; 7 factorial; a rectangle perimeter |
| **In the band at one temperature only** | 6 | sum of a series; sum of squares; sum of evens; a workers-rate problem; a tap-and-drain problem; a four-seat puzzle |
| **A wall at both (0 of 8)** | 6 | every letter-count question; a five-seat puzzle; the series sum as phrased |
| **Free at both (8 of 8)** | 4 | a discount chain; three of the five puzzles |

**Only twelve of the 22 ever show a pass rate inside the band, at either temperature.** Everything else is a wall or free at every reading, and both are dead weight. This is the histogram the chapter's deliverable asks for, drawn:

<figure class="book-figure">{{#include ../figures/07-pass-rate-band.svg}}</figure>

*Figure 7.1 — where my 22 candidate tasks sit, at 8 attempts each and two temperatures. What the figure makes visible is the thin middle: the two dead ends hold most of the set, and the two temperature series disagree about the small part that is left.*

**The band is a property of a policy and a sampler, not of a task.** *(paper result, reported and not reproduced — Endless Terminals, in Reading)* The same generated environment set gives a 3B model a 4.0% starting point and an 8B supervised model 42.6%: items that are a wall for one policy are the band for another. So the health report is always "my model, my sampler, my set, on this date", and one legitimate fix for a wall is a better base model rather than a better task.

**Two temperatures, and a reading of why each miss missed.** *(my judgement, from the corrected adoptions in Lesson 1)* One temperature is not evidence about a task. Two are necessary and still not sufficient: classify the misses before believing the band. Three kinds appear in the probe, and only the third says anything about difficulty:

| The miss | What it looks like | What it is a symptom of |
|---|---|---|
| Format miss | the attempt declared nothing, or declared text the verifier cannot parse | the label protocol from Lesson 2 — a prompt or cold-start problem |
| Length miss | the attempt ran out at `max_model_len` 8192 mid-reasoning, visible in the `/tokenize` count | a budget problem, not a capability one |
| Arithmetic miss | a single number declared, and wrong | the policy cannot do this yet — the only real difficulty evidence |

A wall made of format misses and a wall made of arithmetic misses need opposite fixes, and the pass rate alone cannot tell them apart.

**Stability: what repeated, and what did not.** *(my own measurement)* 16 tasks, 4 samples each, two rounds run with the *same sample ids*, at both temperatures. Six of the 64 samples changed their reward between the two rounds at 0.7, and 8 of 64 at 1.2 — about 90% reproducible sample by sample. The useful number is the other one: the number of usable groups was identical in both rounds, 2 of 16 at 0.7 and 4 of 16 at 1.2. **The band is the stable quantity; the individual sample is not.** So decide on the group count, treat a rate from 4 to 8 samples as an interval, and remember that one run's ladder is evidence about the band, not about a task's rate — which is exactly the movement in Lesson 1's case 3.

**The scorers, because a pass rate is only as good as the thing that produced it.** *(my judgement; the suite is Project 1's)*

| Verifier | What it scores | The design decision inside it |
|---|---|---|
| `exact_match` | a string answer against the expected string, after normalisation | cheap and brittle: a correct answer with different spacing is a zero unless the normalisation is written down |
| `numeric` | a number within a stated tolerance | the answer must be declared — the final line carries exactly one number, or the value is never scored. It refuses to guess at a number buried in prose |
| `unit_test` | a program the attempt wrote, against the task's own tests | the attempt runs in a container with no network, a memory cap and an external timeout; a missing container runtime raises instead of scoring zero |
| `logic_puzzle` | an assignment against the task's constraints | the constraints *are* the answer, and a near miss is a hard zero whose detail names the constraint that failed |

The suite scores 21 of 21 fixture tasks. Two of those rules are worth copying deliberately: `numeric` refuses to fish for an answer in prose, which is what makes the label protocol of Lesson 2 a real requirement; and `unit_test` refuses to turn an infrastructure failure into a zero, because a missing container runtime scored as 0 teaches the model that it cannot write code when the truth is that nothing ran. `unit_test` also needs the reset discipline of [Chapter 6](../06-credit-assignment/index.md): a task must start from the same state on every attempt, or the pass rate is measuring leaked state.

**Fixing a set, in the order that pays.** *(my judgement)*

1. **Repair the walls that are format walls** — prompt and cold-start problems, and the cheapest items to recover.
2. **Rephrase the walls you believe in.** The same series sum appears twice in the probe: one phrasing sits in the band at a single temperature, and the phrasing that shipped is a wall at both. Same mathematics, same verifier, one sentence different.
3. **Use the format demand as a difficulty knob, deliberately.** Lesson 2's table moves the pass rate from 11 of 42 to 37 of 42 with no change to the tasks at all, and the same lever points the other way if a task has become too easy.
4. **Then add items in the middle**, and read the new histogram at two temperatures.
5. **Order the set last.** At this scale the composition of a task set matters more than the order it is presented in, and a curriculum built over a set with no middle is an ordering of nothing.

**What it costs.** Every reading is items × samples × temperatures: the probe above is 352 generations of a 1.7B model, and half of it — one temperature — is not evidence, which is the corrected adoption of Lesson 1, paid for. The reading also expires, because the histogram is a property of the policy and the sampler: it carries the model, the server version and the date beside the numbers, and it is re-run when the policy has moved far enough to matter.

### Lesson 5 · Scaling the task set

**The claim, from a paper that measured it.** *(paper result, reported and not reproduced — Endless Terminals, in Reading; abstract verified 2026-10-14)* Endless Terminals states the premise in one line — environments are the bottleneck for self-improving agents — and then builds the pipeline: generate diverse task descriptions, build and validate a containerised environment for each, generate a completion test, filter for solvability. That produces 3,255 terminal tasks without human annotation, and training on them with vanilla PPO (proximal policy optimisation) and a binary episode-level reward produced the gains quoted in Lesson 1. The algorithm was the boring part and the data pipeline was the contribution.

**The same shape, twice more, in other domains.** *(paper results, reported and not reproduced)* EnvScaler generates tool-interactive environments programmatically — skeleton first, then task scenarios plus a rule-based validator per environment — and reports 191 environments and about 7,000 scenarios used for SFT and RL. RandomWorld generates interactive tools and compositional tool-use data, and reports that downstream performance scales with the amount of generated data. Three domains, one shared structure, and in two of them the generated *validator* is as much of the contribution as the generated task.

**The pipeline is a generator plus a filter, and the filter is the engineering.** *(my judgement, from the three papers and from my own toy version)* Each paper spends its detail on validation, not on generation: build the environment, prove it starts, generate the check, filter for solvability. My 22-item set is the same pipeline at toy scale, and every step that made it healthy was a filter rather than a generator — enumerating a puzzle's assignments to prove it has exactly one solution, running two temperatures, asserting that the answer was declared, keeping three controls with known readings.

**Prove the task, do not assume it.** *(my own measurement; the general rule is the book's negative-test habit)* The probe's logic puzzles were checked by enumerating every assignment of the constraint system. Two candidate puzzles were rejected while they were being built, because the enumeration found 3 and 4 solutions. **A task with more than one answer is a defect, not a hard task**: the verifier marks a correct answer wrong, and the reward then teaches the model away from a right answer — the worst label noise available, and invisible in the reward curve, because a task like that simply looks hard. The rule for generated material: the generator must emit the *check* along with the item, and the check must be run against the item before the item joins the set. Prove a task the way you would negative-test a check: inject the defect into a copy, and assert that the injection was caught.

**Two defects, as engineering history.** *(my own defects, both found live in this harness; the rules attached are my judgement)*

1. **The container wrote as root.** Docker ran the attempt as root, so a generated attempt's `import main` created a `__pycache__` directory owned by root inside the bind mount. The host user could then no longer delete its own working directory, and the cleanup step raised `PermissionError` and killed the run instead of reporting a score. The fix was two rules: run the container as the caller (`--user uid:gid`), and treat cleanup as housekeeping — **a failure to tidy up must never be able to fail a scoring run.** The general lesson is that a verifier's infrastructure failures must be loud and separate from its verdicts; this one disguised itself as a crash.
2. **Two runs shared a second, and a configuration's rows disappeared.** Run ids were second-resolution timestamps. Two runs started inside the same second, and the second run's header and round record overwrote the first's, so an entire configuration's results were gone. The writer now refuses to reuse an existing directory that it did not create. The rule: **an id that is a timestamp is not unique**, and a harness that loses rows is more dangerous than a model that scores badly, because it deletes the evidence you would use to find out why.

**What it costs.** *(my judgement; the counts are the papers')* Generation is one model call per item. Validation is a container build, a reset, a generated completion test, a solvability filter, and enough samples to place the item in a band — and *that* is the bill. Published synthetic sets advertise the count, which is the output of the filter rather than its input. Budget the sampling you need to place the items, not the generation, and keep controls with known readings in every generated batch, so a change in the tasks can be told apart from a change in the harness.

### Lesson 6 · Decontamination and keeping the evaluation set separate

**What contamination is.** *(textbook fact for the definition)* A task set is **contaminated** when an item in it, or something close enough to be answered by recall, also appears in the set you report your score on. The benchmark number then measures whether the model has seen the item, not whether it can solve the task. Decontamination is the measurement and removal of that overlap, and the reason it belongs in a chapter about task engineering is that it is a property of your data, not of your model.

**Why a contaminated gain is so hard to see.** *(paper result, reported and not reproduced — Spurious Rewards, in Reading)* Reported work on RLVR (reinforcement learning with verifiable rewards) found runs that improved benchmark scores even when the reward was deliberately wrong, on models whose prior already favoured the right answers. A benchmark gain is therefore not evidence about your reward — and contamination makes that worse, because if part of a gain is recall of items the model has met, the very measurement you use to test the reward has stopped testing it.

**Four shapes of leakage.** *(my judgement)*

| Shape | What it looks like | Where it comes from |
|---|---|---|
| Exact reuse | the same item, or the same item after whitespace normalisation | curating training data from a source that also fed the evaluation set |
| Near-duplicate | the same problem with different constants | generating variants of an evaluation item, or of a template it came from — the template is the leak, not the numbers |
| Template and format leakage | the evaluation set's answer format appears in your cold-start data | Lesson 2's protocol: at 42 of 42 declared against 11 of 42, format alone is worth a large gain, and none of it is capability |
| Feedback leakage | an item you measured, then fed into a training set in a later round | your own run loop, which is the one place you fully control |

**The check has to be a script, not an inspection.** *(my judgement; it is also this chapter's pass criterion)* The shape that works: normalise training and evaluation items the same way (lower case, collapse whitespace, strip punctuation, keep digits), exact-match the normalised forms, then run a character n-gram or token-overlap check with a written threshold for near-duplicates, and a separate template check for items that share a skeleton but not constants. Report the flagged pairs with their scores rather than a count, because "0 overlaps" from a checker nobody has tested is a claim, not a measurement. Negative-test the checker the way the generated tasks are tested: inject a copy of an evaluation item into the training file, run the check, and assert the copy was caught. A contamination checker that has never been shown to fire has not been verified — it has been run.

**The separation is a discipline, not a filter result.** *(my judgement)* Keep the evaluation set physically out of the path that writes training files, give the two sets different id spaces, and hold one rule firmly: an item that fails the training health check is **retired, not moved to the evaluation set**. Moving it is how a set that was too hard to train on becomes a benchmark the model scores badly on while looking honest.

**Honest status.** *(my own state of the work)* This is the one row of the chapter's health report I have not filled in. What exists is the design above, and the reason the other four rows are not a substitute: a clean histogram says nothing about whether the items were also in the model's training corpus, and a low pass rate on a contaminated item looks exactly like a hard item. The check is the next thing I run, and the status line at the top of this chapter says so.

**What it costs.** *(my judgement)* The check itself is cheap — a normalisation pass and a similarity scan over a few thousand short items. What is expensive is the decision it forces: abandoning or rewriting items you have already measured, and accepting that an edit to a contaminated item produces a *new* item that inherits its parent's template, so it needs its own two-temperature reading before it can be trusted.

## Deliverable

A health report for "my task set":

| Metric | Result |
|---|---|
| Number of items, and how they break down by task family | |
| Pass-rate histogram from a baseline model | |
| Share of items inside the trainable band (pass rate 0.1–0.9) | |
| Contamination check against the evaluation set | |
| Verdict: should this set go into RL at all? | |

**How to fill it in.** Row 1 counts the items by task family — a family is a group of items that share a task *shape* and a verifier, so arithmetic word problems, code tasks and logic puzzles are three families with three different scorers. Row 2 is Figure 7.1 for your set. Row 3 is a single number read off row 2, and it needs the sample count beside it, because the band's edge is set by how many samples you took (Lesson 4). Row 4 is the output of the scripted check in Lesson 6. Row 5 is a sentence you write *before* the training run, since after it you will have a reason to write a different one.

**What the report looks like for the set this chapter measured.** *(my own measurements, configuration in Lesson 1)* 22 items: 14 arithmetic, 5 logic puzzles with enumerated unique solutions, 3 known-moving controls; Figure 7.1 is the histogram. Inside the band: 6 of 22 at both temperatures, and 12 of 22 at at least one — so the share asked for in row 3 is 6 of 22 at both temperatures, which is the only version of the number that means anything for training. The contamination row is not filled in (Lesson 6). The verdict, written from the four rows above it: **not yet** — the composition is the problem, not the algorithm. Ten of the 22 items can contribute no gradient at either temperature, several of the walls are format walls rather than arithmetic ones, and the set needs items added in the middle rather than a longer training run.

**The report is a script, not a document.** These numbers come from running the harness: Project 1's live runner in the companion repository (`agent-rl-handbook-projects`, `p1-verifiers/checks/run_p1_live.py`) generates the samples, scores them with the four verifiers, and writes one row per attempt. Re-run it after every change to the set, and compare group counts rather than rates (Lesson 4's stability reading).

## How you know you passed

- The pass-rate histogram visibly clusters at "always right" / "always wrong", and you propose a fix based on it.
- You can explain why both a very high and a very low pass rate yield no gradient.
- The contamination check is a script you can re-run, not a visual inspection.
- Every number in the report carries its model, sampling configuration and date, and the band figure is reported for two sampling temperatures rather than one.
- The verdict was written down before the training run, and it names what the run would learn if the set went in unchanged.

## Reading

Reused entries were verified where they say; the three new entries were checked against the arXiv API on **2026-10-14** (IDs marked "verified"), and every paper here is filed on [the paper map](../appendix/paper-map.md). Numbers quoted from papers are **reported, not reproduced**.

Programmatic task generation and environment synthesis (Lesson 5):

- **Endless Terminals: Scaling RL Environments for Terminal Agents** — Gandhi et al., arXiv:2601.16443 — verified. The four-stage pipeline (task descriptions, containerised environments built and validated, completion tests, solvability filtering), 3,255 generated terminal tasks, vanilla PPO with a binary per-episode reward, and the reported dev-set gains of 4.0% to 18.2%, 10.7% to 53.3% and 42.6% to 59.0% for the three models. This is the paper behind the placeholder this chapter's Reading used to carry, now a checked citation. Code: <https://github.com/kanishkg/endless-terminals>.
- **EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis** — Song et al., arXiv:2601.05808 — verified. Environment skeletons, then task scenarios generated together with rule-based trajectory validators; 191 environments and about 7,000 scenarios, used for SFT and RL. The generated *validator* is the part to copy.
- **Procedural Environment Generation for Tool-Use Agents** (RandomWorld) — Sullivan et al., arXiv:2506.11045 — verified. Interactive tools and compositional tool-use data generated procedurally, with downstream performance scaling in the amount of generated training data.

What the cold-start stage is for, and its boundary (Lesson 2):

- **Cold-start supervised data, then an outcome reward** — Feng et al., *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*, arXiv:2504.11536 — verified on [the paper map](../appendix/paper-map.md), 2026-10-09.
- **Curation beats volume for surface form** — Zhou et al., *LIMA: Less Is More for Alignment*, arXiv:2305.11206 — verified on the paper map, 2026-10-09.
- **What supervised tuning memorises, and what RL adds** — Chu et al., *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*, arXiv:2501.17161 — verified on the paper map, 2026-10-09.

Why a benchmark gain can mean nothing (Lesson 6):

- **A wrong reward can still raise a benchmark score** — Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, arXiv:2506.10947 — verified in Chapter 5's Reading, 2026-10-09.

## Backfill

- [x] Question 1: When should I reach for RL at all? (If the data fails the check, the answer is "not yet".) — the answer is a short checklist, and this chapter is where its first items live: (1) there is a scorer you can run on every attempt, and its coverage is measured — an item nothing can score is a ceiling (Lesson 4's suite, 21 of 21 fixtures); (2) the items sit in the band under more than one sampling setting, and you have read the misses well enough to know whether each wall is arithmetic or format (Lesson 4); (3) the policy can already produce a scorable attempt, which is what the cold-start set buys (Lesson 2); (4) the items are not also your evaluation set (Lesson 6). When any of the four fails, the honest answer is "not yet", and that answer is cheap: the placement probe in this chapter is 352 generations, against training runs that generate orders of magnitude more. What the four checks buy is the one thing a training run cannot tell you afterwards — what the run was able to learn from in the first place.

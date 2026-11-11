# Chapter 11 · Evaluation and Shipping Validation

> **Compute tier**: ✅ (an evaluation harness is cheap; the Spark can run it around the clock) | **Status**: written — the harness, the score tables and the interval rule come from six project runs in the companion repository; every number is regenerated from a run record that names its code revision, and every score is reported beside its compliance column. Nothing in the chapter is a published benchmark result.
> **Why this chapter exists, and why it matters most**: without it, every training run you do degrades into "it looks better".

## First, the gist

A rising reward curve does not prove the model got better - it may only prove the model learned to please the scorer. Evaluation is how you avoid fooling yourself: define success in code rather than in prose, run the same fixed tasks many times, report a score *with an interval*, use judges whose biases you have measured, and keep a regression suite so that an improvement stays an improvement. This is the chapter that turns a hobby into engineering.

The chapter is written from the referee's side of one repository: every number below was produced by a run whose record names its code revision, and the numbers are regenerated from those records by a script rather than copied into a table by hand. That last detail is not housekeeping. Two of the numbers in this chapter were wrong for two weeks because the table was written by hand, and one of them was wrong for a reason so ordinary that it deserves to be Lesson 2.

## What you'll be able to do

- Report a result as "improved by X points, 95% interval [a, b], N rollouts" instead of a single flattering number.
- Define success for a multi-turn task in a way that cannot be gamed, and say what your definition deliberately excludes.
- Tell a policy from a sampler, and pass@1 from pass@k.
- Detect the three common failures of an LLM judge, and keep a judge as a diagnostic instead of a reward.
- Keep a regression suite that catches silent breakage, and know which of your tasks has been contaminated by your own development.

## Lessons

- [ ] 1. "Did it get better?" - why the reward curve is not an answer
- [ ] 2. Defining success for multi-turn tasks: final-outcome vs process-based judging
- [ ] 3. pass@k vs pass@1: what RL actually improves
- [ ] 4. Variance and significance: multiple seeds, confidence intervals, how to design an ablation
- [ ] 5. Judge reliability: systematic bias in LLM-as-judge, calibration, and how much human spot-checking to keep
- [ ] 6. The offline-to-online gap: regression tests, A/B tests, gradual rollout and rollback
- [ ] 7. Contamination and "overfitting the test set"
- [ ] 8. Building a reusable evaluation harness

### Lesson 1 · "Did it get better?" - why the reward curve is not an answer

A reward curve is a plot of the reward. It is not a plot of the policy, and the two come apart in both directions.

**Flat while the policy moves.** In Project 2's first real training loop, eight rounds on five calibrated tasks produced 19, 17, 20, 18, 16, 19, 19 and 20 solved out of 25. The spread is eight points and the trend is nothing. Two of those eight rounds had no gradient at all: every sample in every group succeeded, so every advantage was zero, and the update was exactly zero. A reader looking at that curve would say "training is not working"; the correct reading is "the measurement has no power, and two of the eight steps did nothing on purpose".

**Moving while nothing is learned.** The same loop, with a budget that fits the question - six tasks, ten samples, three passes per round - went from 42 of 60 to 60 of 60 in one round and then contributed nothing for five more rounds, because a group at one hundred percent has a zero advantage. The curve flattens because the loop has *finished*, not because it is stuck. Nothing in the curve distinguishes "done" from "cannot start", and both look like a horizontal line.

**Same rule, two shapes.** Three-tier credit at `lr 1e-4` with three passes produced 0, 5, **14**, 12, 10, 3 solved out of 30: it reaches, then decays. The same rule at `lr 3e-5` with one pass produced 0, 0, 11, 17, **20, 20**. The second run is not a better algorithm; it is the same algorithm with a step size small enough not to break, in one round, what it learned in the previous one. If you only plot the produced number per round, you cannot tell those two stories apart - and the stories call for opposite fixes.

What to plot instead, and it is three things, not one:

1. **Solved per round, per task.** A total hides a swap. `10/10, 0/10, 10/10` and `7/10, 6/10, 7/10` can share a total and mean different things: the first is two learned plans and one wall, the second is three partial plans.
2. **The fraction of groups with non-zero spread.** That is the fraction of the batch that can produce a gradient at all. A round with zero spread is a round in which the loss will move for numerical reasons and teach nothing; the repository's round records carry `spread_fraction` for exactly this.
3. **The reward terms separately.** `reward_terms` is a dictionary, not a scalar, and a run that reports only the sum cannot say which term paid. The three-tier rule exists because the sum was hiding the difference between "grew the plan" and "did something legal".

### Lesson 2 · Defining success for multi-turn tasks: final-outcome vs process-based judging

The most expensive bug in this project's evaluation was a one-line ambiguity, and I wrote it.

Every rollout record carries a field called `goal_met`. In the environment it means: the *world* is in the state the task asked for. For a rename task, the file has the new name. For a ledger task, the balance is right and the plan was followed in order. There is a second, stricter notion in the same module, `solved()`: the episode finished, **and** the reported answer equals the expected one, **and** the world is right. The docstring gives the reason with an example worth memorising:

> Checking only the answer would score `rm draft.txt` followed by `done 1` as a success on a rename task, which is exactly the kind of quiet false positive this repository exists to avoid.

When the first version of this chapter's tables was assembled, the column was computed from `goal_met`. It overstated one rung by fourteen episodes out of sixty - the band task that solves nothing looked like fourteen successes, because the policy really did delete the right file, then reported 5 where the task asked for 4 - and it moved a second number from 3 to 4. Both numbers were published, one of them in a README, for about two weeks. Nobody was lying; the field was named after a goal, and the goal was not the task.

Three ways to lose this argument, in increasing order of cost:

- **State right, answer wrong.** Deletion succeeded, count wrong. `goal_met` true, `solved` false. The user of the environment is misled unless they read the checker.
- **Answer right, state wrong.** The ledger task whose five moves are executed in the wrong order, totals correctly (`8`), and reports `8`. The number looks perfect; the behaviour is wrong. This is the failure that four rounds of process-reward design were built to charge, and it is invisible to any metric that reads only the answer.
- **Neither, but "finished".** A truncated episode that reports something. Any metric that counts a report as a result will call it an attempt. `truncated` is in the record for this reason; it has to be part of the predicate, not a note beside it.

The rule that follows is blunt: **write the definition of success once, in code, as a conjunction, and derive every table from it.** In this repository that is `Attempt.solved()`, and since this chapter's audit it is also the predicate in `make_eval_claims.py`, the script that generates the numbers in Lesson 4 from the records. The table is no longer a second description of the truth; it is a view of it, and the view is regenerated.

What it cost: about two hundred lines - a generator, a checker, and a selftest, all described in Lesson 8 - and one afternoon, most of which was spent discovering that the field everyone had been reading was not the field anyone meant.

### Lesson 3 · pass@k vs pass@1: what RL actually improves

`pass@k` answers "would a user who retries k times eventually succeed"; `pass@1` answers "would a user who tries once succeed". RL is often defended with the first and shipped as the second.

The clearest pair of measurements in this project separates them. Six calibrated tasks, ten samples each, at temperature 1.7, with a fine-tuned policy that almost never completes the required form: **3 of 60** rollouts solve anything (95% interval 1.7% to 13.7%). One update - three passes over those sixty rollouts, 1e-4 - and the next round solves **60 of 60**; the following rounds hold at 60 and 58. Then the adapter that entered the last round is sampled **greedily**, temperature 0: **60 of 60**, all of them `goal_met`, no sampler luck involved.

The naive summary is "pass@60 went from 1 to 1, and pass@1 went from 0 to 1". The useful summary is that the update changed the *distribution*, and the greedy measurement is the only one that can say so. A policy whose success depends on sampling sixty times is a policy that has not learned the task; the retry budget is doing the work.

The failure mode runs the other way too, and it is easier to miss. On three five-step ledger plans, the fine-tuned policy solves **0 of 30** greedily and **0 of 30** at temperature 1.7. Zero at both temperatures is not "hard"; it is a policy that never succeeds, which is a statement about variance (Lesson 1) as much as about difficulty. Before that measurement, the same tasks with five samples instead of ten had made one of them look impossible - `files-r09` was 0 of 5, then 0 of 10 at the larger budget, and ten of ten after one update. The task was never a wall; the group was too small to contain a success.

Three reporting rules fall out:

- **Report the greedy number or report nothing.** A pass@k without a temperature-0 companion cannot distinguish a policy from a sampler.
- **Report k.** "Solved" without "out of how many samples, at which temperature" is not a measurement.
- **Do not compare pass@k of one policy to pass@1 of another.** Most accidental improvement claims in agent RL are made this way.

### Lesson 4 · Variance and significance: multiple seeds, confidence intervals, how to design an ablation

Sixty rollouts is not a large sample, and the honest way to say what it supports is an interval. Every proportion below is a Wilson 95% interval computed from the counts in the claims manifest; the counts are the ones the run records contain.

<figure class="book-figure">{{#include ../figures/11-proportion-intervals.svg}}</figure>

*Figure 11.1 - the ten proportions this chapter rests on, with Wilson 95% intervals computed from their counts. The two flat rows at the top are not "zero": at sixty rollouts, a true rate of five percent produces no successes about one time in twenty-two, so the band's floor is an upper bound rather than a measurement. The two rows for demonstrations overlap, which is why their totals are reported as a direction and not as an effect.*

| measurement | solved | rate | 95% interval |
| --- | --- | --- | --- |
| band, temperature 1.0, `lr=0` | 0/60 | 0% | 0 - 6.0% |
| band, temperature 1.3, `lr=0` | 0/60 | 0% | 0 - 6.0% |
| band, temperature 1.7, `lr=0` | 3/60 | 5.0% | 1.7 - 13.7% |
| band, after one update | 60/60 | 100% | 94.0 - 100% |
| deep five-step plans, files at T=0 | 30/30 | 100% | 88.6 - 100% |
| deep five-step plans, files at T=1.7 | 16/30 | 53.3% | 36.1 - 69.8% |
| deep five-step plans, ledger at both temperatures | 0/60 | 0% | 0 - 6.0% |
| three-tier credit, final adapter, greedy | 20/30 | 66.7% | 48.8 - 80.8% |
| demonstrations, eight epochs | 50/60 | 83.3% | 72.0 - 90.7% |
| demonstrations, two epochs | 40/60 | 66.7% | 54.1 - 77.3% |
| held-out tasks, before the update | 23/40 | 57.5% | 42.2 - 71.5% |
| held-out tasks, after the update | 30/40 | 75.0% | 59.8 - 85.8% |

Read three rows together and the discipline becomes concrete. 0/60 does not mean zero; it means "below six percent", and at five percent a true effect of that size would produce zero successes in sixty rollouts about one time in twenty-two. That is why the band needed a second rung (a harder task set) rather than more rounds: the first rung's floor was inside its own interval. The final row pair, 50/60 against 40/60 for eight and two epochs of demonstrations, **overlaps**: the ten-episode difference is inside the noise, and the claim "more epochs is better" is not supported by those totals. What the experiment does support is a difference in *composition* - the eight-epoch policy solves all thirty of its trained plans and twenty of thirty unseen ones, the two-epoch policy solves all thirty trained and ten unseen - and even that rests on ten samples per task, which is why it is stated as a direction rather than a magnitude.

Designing an ablation follows from the same arithmetic:

- **Change one thing.** Same tasks, same temperature, same sample count, same code revision except the difference under test. The runs cited here differ in a single flag or step size, and the records say which.
- **Include a null arm.** `lr=0` runs the whole loop - sampling, scoring, recording - and applies nothing. They measure the environment's and the sampler's own variance, which is the floor any claimed effect has to clear. Two such runs at temperatures 1.0 and 1.3 returned 0/60 and 0/60; the temperature that produced 3/60 was found by varying one number.
- **Pre-declare the threshold.** Before the judge experiment, the decision rule was written down: a conflict rate above fifty percent means the judge does not enter the reward. The measured rate was above seventy-five percent, and the honest response was not to re-tune the weights until it looked better.
- **Report the per-task breakdown.** Aggregates average away the phenomenon you are trying to see. `files-r09` at 0/5 and `files-r05` at 4/5 average to 4/10, which describes neither.

And a caution the interval table cannot express: rollouts within a group share a task, a prompt and a policy. They are not independent draws from one urn. Treating sixty rollouts as sixty independent samples makes every interval above too narrow, which is precisely the wrong error - it makes weak results look published.

### Lesson 5 · Judge reliability: systematic bias in LLM-as-judge, calibration, and how much human spot-checking to keep

The project tried to replace its rule-based reward with an 8B model asked to judge each trajectory. The result is the chapter's cleanest example of a measurement that is worth more than the thing it measured.

Three findings, in order:

1. **The small judge cannot hold the format.** A 1.7B model judging six-turn traces stopped after about four turns; the parser refused to invent the rest. A judge that cannot see the whole trajectory is not a weak judge, it is an uncalibrated one - and the failure is systematic, not noisy.
2. **The 8B judge holds the format and disagrees with the rule, systematically.** On 35 judged episodes it produced valid verdicts 35 of 35 times, and disagreed with the rule on **106 turns** - about three quarters of the judged turns, at least 0.25 apart on the score, which was the study's threshold for "disagreement". The disagreements clustered on exactly the turns where the policy reads before acting: the rule calls `ls` and `cat` neutral ("this turn taught nothing"), the judge calls them failures ("this turn did not advance the task").
3. **Fusing it would have taught the wrong thing.** A dense reward that charges reading teaches a policy not to look before it acts, which is the opposite of the behaviour the environment was built to reward. The judge stayed in the repository as a diagnostic and was removed from the reward.

Before any of that could be measured, the task set had to be usable: the rule's scores were calibrated so that a judge had something to disagree with. On the frozen 43 tasks, the fraction falling inside the uncertain band - the only place where a second opinion can change anything - went from 14-19% with the first calibration to 36-45% with a wider band, and to 100% once the task generator was allowed to produce tasks at the right difficulty. The judge experiment would have been meaningless at a 15% band: agreeing on the easy tasks and disagreeing on nothing is what a judge looks like when it is not being tested.

What to keep, then:

- **Human spot checks on the disagreements, not on the agreements.** A random sample of trajectories the rule likes tells you the rule is self-consistent. The informative sample is the one where the two judges disagree: read ten of those and ask which one is right. That is the only calibration that transfers.
- **A conflict rate as a first-class metric.** Report it next to the score. A judge whose conflict rate is 75% is not a reward with noise; it is a different definition of the task.
- **A judge never silently enters the reward.** If it does, the training loop optimises the judge's biases, and the failure looks like learning. The reward-model over-optimisation literature is the general form of this; the specific version here took one afternoon to measure and would have taken weeks to diagnose from a flat-then-declining curve.

### Lesson 6 · The offline-to-online gap: regression tests, A/B tests, gradual rollout and rollback

Training changes one distribution; the product depends on several. The gap between them is where "it got better" goes to die, and the cheapest instrument for finding it is a set of tasks you already solve.

The project's held-out check is four tasks held out from training and measured before and after a single update: 23 of 40 before, 30 of 40 after. That headline is a gain of seven. The table underneath is more useful:

| task | before | after |
| --- | --- | --- |
| `files-r01` (rename) | 10/10 | 10/10 |
| `files-r03` (rename) | 10/10 | 10/10 |
| `files-d05` (delete) | 3/10 | **0/10** |
| `ledger-03` (sequence) | 0/10 | **10/10** |

One round taught the ledger procedure - an unseen task of an unseen family went from nothing to perfect - and cost a sibling operation: the delete task fell from three successes to none. The mechanism is not mysterious. The two families share vocabulary (`final.txt`, `notes.md`) and differ in what the work is; a policy that has just learned "follow the ordered plan and report" applies it to the delete task too, where it does not apply. Aggregate gain, individual regression. A harness that reports only the aggregate would have shipped this and called it progress.

That is what a regression suite is for: a named list of behaviours that must not get worse, checked by name, reported by name. Not "the score went up" but "these two tasks went from 10/10 to 0/10, here are the transcripts". In a product this is the A/B test and the canary; in a training loop it is the held-out set and the same reporting discipline.

The demonstration experiment in this chapter's sibling (Lesson 7's table) shows the other half of the gap. A supervised policy trained on 18 demonstrations solves 30 of 30 trained plans and 20 of 30 unseen ones; a policy trained by the three-tier reward solves 20 of 30 trained and 30 of 30 unseen. Same total, opposite composition. Offline, both look like "fifty of sixty". Online - on tasks neither was tuned for - they are different products.

### Lesson 7 · Contamination and "overfitting the test set"

Contamination is usually discussed as a property of pretraining corpora: the benchmark leaked into the model. In agent work the more common version is smaller and entirely within your control: **the test set leaked into your development loop**.

Three examples from this repository, ordered by how easy they are to avoid:

- **A held-out set that was never trained on, and was committed before it was used.** Three five-step ledger plans were generated, verified solvable by the oracle in six turns, and written to `fixtures/holdout.jsonl` in a commit that precedes any training run that could have seen them. The claim "never seen" is therefore a claim with a date and a hash, not a memory. That is the standard the rest of this chapter holds other numbers to.
- **A test set that was read while building the system.** The frozen 43 tasks are *not* unseen: they were used throughout development, chosen for difficulty by hand, and their failures drove the environment's design. The honest label is "not trained on", not "unseen", and every table that uses them says so. The distinction matters because "not trained on" bounds overfitting to the *policy*, while "unseen" would bound overfitting to the *developer*, and only the first is true.
- **A demonstration set that becomes a test set by accident.** The supervised baseline is trained on three plans and evaluated on those same three plus three new ones. The trained three score 30 of 30 - a memorisation number, and the reason the table reports the two columns separately. Reporting only the total (50 of 60) would have described memorisation and transfer as one quantity, and the two point in opposite directions here.

Two practices follow. First, **state the exposure of every set**: trained on, read during development, or never seen - and if it is "never seen", say when it was created and commit it before use. Second, **treat a test set as disposable**. Once a set has been read enough to shape decisions, it has become a development set; the next honest number needs a new one. In this project that means every chapter that reports a held-out result also names the fixture that produced it, so the reader can tell which sets have been spent.

There is a third, subtler contamination that has nothing to do with data: **evaluation-code drift**. The templates, the parser, the turn budget and the scoring rule are all code, and code changes. A held-out number from a version whose parser accepted a different action format is not comparable to one from today. That is why the manifest carries a `code_sha` per claim rather than a global "repository version" - the two numbers can be compared only if they were produced by comparable rules.

### Lesson 8 · Building a reusable evaluation harness

The harness this chapter describes is four files and about three hundred lines. It is worth building once, because everything above depends on it.

**1. An evaluation path that cannot train.** `rl_served.py --eval-only` samples the adapter the *server* holds, applies no update, and writes a record whose notes say so. It deliberately does not load local weights: a number produced by a local copy describes the copy, and the copy and the server drift. The first version of these records claimed `epochs=3; lr=0.0001` because those were the argparse defaults on a path that trains nothing; a reader re-deriving from the record would have concluded the policy had been updated three times. Eval records now say `eval-only: no epochs, no lr, no update`.

**2. A record whose rows are as traceable as its header.** A run header names the code revision, the dataset and the configuration hash. The rollouts inside it carried `git:none` for two months, because a default argument was convenient and nothing checked. A row is the unit a reader re-derives from, so a row that cannot be traced to its code is a number with no owner - the exact failure the repository's own rules forbid. Both call sites now pass the revision they wrote into the header.

**3. A manifest generated from the records, and a checker for it.** `make_eval_claims.py` recomputes every number Chapter 11 quotes - nineteen claims, 980 rollouts - and writes `fixtures/eval-claims.json`. `eval_claims.py --check` verifies the arithmetic on any machine: per-task counts sum to the claim, every task has the same sample count, a per-round series ends where the cited round says. `--check --runs runs` goes further where the records exist and recomputes each claim, including the code revision, refusing any difference. On a clone the second half is skipped, because `runs/` is gitignored; the first half is not, so the gate is meaningful everywhere.

**4. A selftest that encodes the mistake.** The checker's selftest builds a rollout whose world is right and whose report is wrong, and fails if it is ever counted as solved; it also mutates a valid manifest three ways - inflated total, wrong series, invented revision - and requires each mutation to be caught. This is the part that makes the harness more than a formality: the first version of the generator had the `goal_met` bug, and a checker that shares a bug with its generator is a checker that certifies it. The bug is now a test.

What the harness bought, measured: two wrong numbers found in an afternoon (24 versus 16, 4 versus 3), a record class that lied about its own training, and a class of drift - the manifest and the records disagreeing after a re-run - that now fails the gate instead of being noticed by a human reading a diff. What it costs: the manifest must be regenerated after every new run, which is one command and a review of the diff, and that diff is exactly the review the chapter is asking you to do anyway.

The report template the harness produces is five fields, and anything that cannot fill all five is not a result:

| field | example |
| --- | --- |
| claim | three-tier credit lifts the two learned plans to 10/10 |
| run and revision | `p3-rl-served-37`, `git:7711269` |
| metric and budget | solved per task, 10 samples, greedy |
| interval | 20/30, 66.7% [48.8, 80.8] |
| threshold and decision | pre-declared: keep if the greedy score clears 15/30 |

## Deliverable

- An evaluation report template that includes multiple seeds and confidence intervals - the five-field table above, and the manifest that fills it.
- A regression script you can run after any change to model, reward or data - `eval_claims.py --check --runs runs`, plus the held-out tasks reported per task rather than in aggregate.
- A table mapping each of your metrics to the real-world risk it covers. In this project: solved/greedy covers "does the policy work when the user tries once"; solved/sampled covers "does retrying help, and how much"; the per-task regression table covers "did anything we already had break"; the conflict rate covers "are we optimising a judge's bias"; the code revision covers "is this number comparable to the last one".

## How you know you passed

- **Every training comparison you report comes with a spread, not a single number.** The chapter's own table reports 60-rollout proportions with Wilson intervals, and one of its conclusions is that a ten-episode difference is inside the noise - a conclusion that only exists because the intervals are there.
- **You can name at least two judge biases and show an example of each.** Here: the small judge truncates the trajectory at four turns; the large judge systematically calls read-before-act turns failures, disagreeing with the rule on 106 of the judged turns in the direction that would teach a policy not to look.
- **The regression script runs in CI, and a failure points at the specific examples that changed.** `files-d05` falling from 3/10 to 0/10 while the total rose from 23/40 to 30/40 is the example that justifies the sentence.
- **You can name a number you published that was wrong, and the check that now catches it.** 24/30 was 16/30. The check is `eval_claims.py`'s selftest, and it fails if `goal_met` is ever counted as a solve again.

## Reading

Seven entries, each checked against its arXiv abstract page on 2026-11-07; links are to the paper pages, and no number in this chapter is quoted from them.

- **Evaluating Large Language Models Trained on Code** - *Chen et al.*, arXiv:2107.03374 - verified. The origin of pass@k, and the reason the estimator is unbiased for a fixed k rather than "run k times and see": read section 2.3 for the estimator, then note how often it is quoted without its sampling assumptions.
- **Deep Reinforcement Learning at the Edge of the Statistical Precipice** - *Agarwal et al.*, arXiv:2108.13264 - verified. Why single-run comparisons in RL are usually not evidence, and what to report instead: stratified bootstrap intervals, performance profiles, and the argument that a handful of seeds cannot support a ranking. This chapter's intervals are the small, distribution-free version of the same point.
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** - *Zheng et al.*, arXiv:2306.05685 - verified. The catalogue of judge biases - position, verbosity, self-enhancement - and the agreement rates that make a judge useful as a filter. Read it before trusting any judge, and note that the paper's own conclusion is "a good judge, with known biases", not "a replacement".
- **Large Language Models are not Fair Evaluators** - *Wang et al.*, arXiv:2305.17926 - verified. A concrete demonstration that the order of candidates changes the verdict, and a calibration-style correction. The chapter's Lesson 5 is the agent-trajectory version of the same failure.
- **NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark** - *Sainz et al.*, arXiv:2310.18018 - verified. Contamination as a measurement problem rather than a scandal: what can be detected, what cannot, and why "we did not train on it" is a claim that needs a date.
- **Scaling Laws for Reward Model Overoptimization** - *Gao, Schulman, Hilton*, arXiv:2210.10760 - verified. The curve that Lesson 5 warns about: optimising a proxy reward improves the proxy while the true objective turns down. The judge in this chapter is a proxy that never entered the reward, which is the only reason the failure did not happen.
- **Categorizing Variants of Goodhart's Law** - *Manheim, Garrabrant*, arXiv:1803.04585 - verified. Four ways a proxy diverges from the goal, and which of them a benchmark harness can detect. Useful vocabulary for the moment a metric starts improving for the wrong reason.

## Backfill

- [ ] Question 1: How do I know a run is real? - the record is the answer: a header naming the code revision, the dataset and the configuration, and rows that can be recomputed. A number without a header is a rumour; a header whose rows disagree with it is worse.
- [ ] Question 2: Which tasks should be in the test set? - the ones you have not read. Where that is impossible, say what exposure each set has had, and keep a small set that was created and committed before the work it measures.
- [ ] Question 3: How do I debug a long multi-turn failure? - per task, then per turn. An aggregate tells you that something failed; the turn-level record tells you which definition it failed: world, report, or order.
- [ ] Question 4: Where does my training system quietly lie to me? - in the name of a field. `goal_met` and `solved` differ by a conjunction, and the conjunction is the task.

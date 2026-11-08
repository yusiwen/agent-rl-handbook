# Chapter 13 · Paper Deep-Dives and a Judgement Framework

> **Compute tier**: ✅ (reading, and the cheapest possible reproduction - one baseline, one comparison) | **Status**: written
> **Structure**: eight lessons. Lessons 1 and 2 are the instrument; 3 to 7 point it at the areas where the paper traffic is heaviest; 8 is the checklist that turns a verdict into a week of work.

## First, the gist

New RL papers arrive faster than anyone can read them, and reading them one by one is a losing game because most of them are *patches* - a fix to one clause of one objective, measured on one benchmark. The fix is not speed-reading. It is a fixed set of questions, asked in the same order every time, so that a paper can be filed in half an hour and reproduced in a week, or rejected without guilt. This chapter builds that instrument, then uses it in the five areas where the traffic is heaviest, and ends with the experiment that tells you whether a paper is worth following: **your baseline, your data, with and without the intervention, and an interval.**

## What you'll be able to do

- Place an unseen paper into the map quickly, and name the paper it is patching.
- Ask the seven questions that separate a measurable claim from a plausible one.
- Read a GRPO-line paper and predict its failure modes on your own task set.
- Decide whether to reproduce a paper, with a one-week plan and kill criteria written before the work starts.

## Lessons

- [x] 1. How to use the map: four axes, eight shelves, and the empty slots
- [x] 2. The judgement framework: seven questions, asked in the same order
- [x] 3. Improving and industrialising PPO: what the GRPO line actually changed
- [x] 4. The DPO shelf: a reading protocol for a family you may not need
- [x] 5. Reward modelling and robustness: a judge is an instrument, and instruments drift
- [x] 6. Training systems: asynchronous versus synchronous, and which bottleneck the ratio assumes
- [x] 7. The three hard problems of multi-turn agent RL: credit, sparsity, collapse
- [x] 8. **A reproduction checklist**: the smallest experiment that tells you whether a paper is worth following

### Lesson 1 · How to use the map: four axes, eight shelves, and the empty slots

The book keeps a map in the appendix: eight shelves, forty-two slots, seventy-six papers. It is not a reading list. It is a filing system, and four axes are the handles:

| axis | the question it answers | example |
|---|---|---|
| algorithm family | what is optimised, and with what estimator | PPO, GRPO, DPO, REINFORCE |
| engineering theme | which part of the machine the paper moves | loss term, sampler, judge, weight sync, kernel |
| application domain | what the reward is attached to | maths, code, tools, environments, office work |
| agent capability | what the agent should be able to do afterwards | call tools, use memory, plan, recover from failure |

Filing a paper takes three sentences: **what it freezes, what it moves, and what it measures.** A paper that moves the ratio from tokens to sequences (GSPO against GRPO) is a different filing from one that moves the sampler (DAPO's dynamic sampling), even though both call themselves a GRPO improvement. If you cannot fill the three sentences, you have not read the paper - you have read its abstract.

Two habits come out of the axes:

- **Find the parent.** Every paper patches something, and the parent tells you which failure is being attacked. A paper with no parent is either a new shelf or a paper that has not explained its motivation.
- **Read the empty slots.** The map names the slots with nothing in them, and those are the honest statement of what the field has not built - usually not "nobody thought of this" but "this is expensive to measure". The empty slots are the highest-value reading list, because the first paper to fill one is not competing with forty others.

The map is also the antidote to novelty fatigue. When the thirty-first GRPO patch arrives, the useful response is not to read it more carefully than the first thirty; it is to ask which slot it lands in, whether that slot was already occupied, and what it measures that its neighbour did not.

### Lesson 2 · The judgement framework: seven questions, asked in the same order

This is the chapter's instrument. Seven questions, in this order, because each one can end the review early.

**1. What is frozen, and what moves?** Weights, prompt, router, judge, environment, sampler. A paper that claims self-improvement while moving only the prompt is a prompt-engineering paper with a better title. A paper that moves the judge and the policy at once has two treatments and no way to separate them.

**2. What does the verifier check, and what can fool it?** The verifier is the measurement instrument, so a paper's claims are only as good as its checker. Ask for the failure taxonomy: a wrong answer, a crashed sandbox, a timeout and a truncated reply are four different events, and a paper that counts them as one number has not measured anything you can reproduce. This book's own version is the Project 4 table where a container missing one library scored 1 of 6 - the environment produced that number, not the model, and only a separate error count made it visible.

**3. What is n, and is there an interval?** A rate without a denominator and an interval is an anecdote with a decimal point. Chapter 11's arithmetic applies to other people's papers exactly as it applies here: at a hundred instances, 34 against 29 is a direction, not an effect.

**4. Is the baseline at parity on compute and data?** The commonest way a paper wins is by training longer, sampling more, or starting from a better base model. A GRPO paper that beats a PPO baseline trained on a tenth of the rollouts has measured its budget.

**5. What is the compute, in a unit you can compare?** GPU-hours, tokens generated, rollouts scored, wall clock. A number like "2.3x faster" is meaningless without the bottleneck stated, which is Lesson 6's whole subject.

**6. Which ablation isolates the claim?** A paper with four changes and one ablation table is a paper with one measured change and three decorations. Look for the row that removes *only* the thing the abstract claims.

**7. What would falsify this, and did they run that?** Every honest paper has a result it would rather not have published. The presence of one is evidence of method; its absence is evidence of selection.

Two rules run across all seven, and both come from this book's projects:

> **The two-condition rule.** Any claim about an intervention is measured with and without, on the same instances, with the same checker. Not "after" against "before"; the same instances, twice.

> **The ceiling rule.** Before blaming an intervention for a null result, hand the system the *correct* input and measure again. Project 6's outcome table had a condition that injected the true procedure with no retrieval error, and it scored 25 of 100 against the bare floor's 34: without that row, a null result is indistinguishable from a broken router.

### Lesson 3 · Improving and industrialising PPO: what the GRPO line actually changed

Chapter 4 covered the machinery. This lesson is about what to read for, and what to expect when you run it yourself.

The line goes like this. PPO optimises a clipped ratio with a value function to reduce variance, and it is expensive because it carries four models - policy, reference, reward, value. **GRPO** (arXiv:2402.03300, DeepSeekMath, 2024) removes the value network and uses the group mean as the baseline: sample several answers to the same question, score them, subtract the group's mean. **DAPO** (arXiv:2503.14476, 2025) adds four fixes to that recipe, of which two matter for an agent builder: clipping the ratio asymmetrically so a rare good token is not clipped away, and **dynamic sampling**, which drops questions where every sample scored the same because they contribute nothing. **GSPO** (arXiv:2507.18071, 2025) moves the ratio itself from token level to sequence level. **RLOO** (arXiv:2402.14740, 2024) makes the same argument in the other direction: use a leave-one-out baseline and skip the critic entirely.

Four questions to ask of any paper in this line, in this order:

1. **Where does the baseline come from, and what happens to a group with no variance?** Dynamic sampling is DAPO's answer; the arithmetic behind it is Chapter 6's, and the measurement is cheap - count the groups whose samples all scored the same.
2. **Is the loss token-level or sequence-level?** This changes what a long answer is worth relative to a short one, and papers are frequently unclear about which one they implemented.
3. **What happens to a truncated trajectory?** Mask it, penalise it, or drop it: three different gradients from the same episode.
4. **What is the reference model, and is it really frozen?** The KL term is a safety valve, but a moving reference turns the objective into something else.

This project's own numbers are why question 1 comes first. In Project 4, six task families produced twenty-four rollouts, and **four of the six groups were four for four** - zero variance, zero gradient. In Project 5, seven of nine tasks were solved in all eight sampled attempts and the two failures were deterministic, so the outcome loop found a non-zero advantage in three samples out of seventy-two. Neither result is a bug in the trainer; both are the flat-group arithmetic that dynamic sampling exists to handle. A reader who has measured that once reads the DAPO section of any paper differently: the interesting claim is not that clipping helps, it is what fraction of their groups were informative, which is usually not reported.

### Lesson 4 · The DPO shelf: a reading protocol for a family you may not need

**Direct Preference Optimization** (arXiv:2305.18290, 2023) is the paper that made preference learning without a reward model a default. The derivation replaces the RL objective with a classification loss over pairs, and the practical consequence is large: no rollouts, no sampler, no scorer, and a training step that looks like supervised learning.

This chapter gives the reading protocol rather than a reproduction, and the honest reason is the framework's first question: **we had a verifier.** Every project in this book that trained anything had an environment or a checker that could score an attempt. DPO answers a different situation - you have preferences and no scorer - so reproducing it here would have measured a technique against a task it was not designed for.

What to ask when you do read a DPO paper:

- **What does the reference model freeze?** The objective is defined relative to a frozen reference; a paper that also trains or swaps the reference has changed the family.
- **What is the preference data, and who produced it?** Model-generated pairs carry a systematic bias toward the generator's style, and the loss will happily learn it.
- **Is the win-rate length-controlled?** A policy that learns "longer is better" wins on a naive judge and loses on a careful one; that is Chapter 11's measurement problem wearing a different hat.
- **Does the improvement survive a different judge?** If the gain exists only under the judge used to build the pairs, it is a circularity, not a capability.

The shelf also holds the family's repairs, and they are worth reading as a lesson in how a field patches an objective: each fixes a specific failure - implicit reward leakage, reference drift, length bias, the gap between preference and correctness - and each costs a paragraph of derivation. The judgement rule is the same as everywhere else here: find the failure being patched, then ask whether your task has that failure.

### Lesson 5 · Reward modelling and robustness: a judge is an instrument, and instruments drift

A reward model is the most consequential component in a post-training pipeline, because it is the thing that decides what "better" means for every subsequent gradient step. The literature's failure catalogue is long - verbosity bias, sycophancy, reward hacking, judge-policy collusion - and the reading discipline is to treat the judge as an instrument with a calibration record rather than as an oracle.

This book has one measured case. In Project 2, an 8B judge and a rule-based judge scored the same 35 judged episodes. The model judge returned a valid verdict **35 times out of 35** - format was never its problem - and disagreed with the rule on **106 turns**, about three quarters of the judged turns, by at least 0.25 on the score. The disagreements clustered on exactly the turns where the policy reads before acting: the rule calls `ls` and `cat` neutral ("this turn taught nothing"), the judge calls them failures ("this turn did not advance the task"). Three lessons followed, and they generalise to any paper on this shelf:

- **Format validity is not agreement.** A judge that always answers looks healthy on the first metric you measure and can be wrong on most hard turns, in a direction that is perfectly systematic.
- **Disagreement is a first-class number.** A pipeline that reports only the final score has thrown away its own uncertainty; the count of turns where two judges differ is the cheapest error bar available, and its *direction* - here, "reading is a failure" - is the diagnosis.
- **The judge is only as good as its failure taxonomy.** A judge that scores a crashed sandbox, a timeout and a wrong answer the same way has taught the policy that three different things are equally bad, which is how Project 4's policy learned that avoiding the risky API was the safe move.

When a paper proposes a new reward model or a new rubric, the framework's questions 2 and 6 are the ones to press: what is the verifier checking, and which ablation shows the gain is the judge and not the extra data? A judge change and a data change are usually confounded in these papers, and the confound is rarely stated.

### Lesson 6 · Training systems: asynchronous versus synchronous, and which bottleneck the ratio assumes

A systems paper in this field makes one of two claims: *same result, less time* or *same time, better result*. Both are ratios, and a ratio hides its denominator. The question to ask is therefore not "how much faster" but **which resource was the bottleneck while the faster number was measured.**

Project 5 measured it, on one machine, with generation out of process and weight sync over HTTP. Four rounds, seventy-two samples per run, identical tasks:

| configuration | samples per minute | sample staleness (mean / max) |
|---|---|---|
| synchronous, one worker | 36.1 | 0.04 / 1 |
| asynchronous, one worker | 35.9 | 0.06 / 1 |
| asynchronous, three workers | **108.2** | 0.18 / 1 |

**Asynchrony on its own bought nothing.** At one worker the two designs are within 0.2 samples per minute of each other, because the bottleneck was the served model, not the trainer: a training step took a couple of seconds and the generation it waited for took thirty. The threefold gain came from running three workers, which asynchrony *permits* rather than *provides*. Any paper that reports an asynchronous speedup without naming the idle resource is telling you about its cluster, not about its algorithm.

The second half of the lesson is the price. Staleness - the gap between the policy that produced a sample and the policy that trains on it - is what you pay for overlapping, and it has to be measured and bounded, not assumed. Our maximum stayed at one because this world was fast; with a container, a browser or a real API in the loop, the same configuration would produce ages of three or four and the staleness bound would start discarding samples. The paper to trust is the one that reports the staleness distribution *and* the fraction of samples it dropped, because those two numbers are the honest cost of the ratio it is advertising.

### Lesson 7 · The three hard problems of multi-turn agent RL: credit, sparsity, collapse

Three problems recur across the agent-RL literature, and this book has a measurement for each.

**Credit assignment** - which turn deserves the reward. Trajectory-level advantages are the default because they are cheap; TRACE (arXiv:2607.13988, 2026) is one of the papers that assigns credit at turn level via an estimate rather than a uniform broadcast. What to check: does the method require a second model, and does its gain survive when the episode is short? A method that only helps long horizons should say so.

**Sparse reward** - the environment says "wrong" and nothing else. Project 4 is the clearest case: six families, four with identical scores across four samples, and an update built from the two informative groups. Project 5's version was starker - seven of nine tasks solved every time, two failed every time, and a deterministic failure contributes exactly as much as a success: nothing. The reading rule is to ask for the *variance profile* of the task set, not its mean difficulty. A paper reporting 40 per cent success on a set where every task is solved by 40 per cent of samples has a very different training problem from one reporting 40 per cent where half the tasks are at 100 per cent.

**Training collapse** - the policy gets worse while the loss curve looks fine. Project 4 met it directly: at learning rate 5e-5 the policy fell from 6 of 6 to 4 of 6 with twelve sandbox errors, and the failure signature was a repeated `AttributeError` as the model tried four different ways to avoid the API it had been punished for using. Negative advantages taught "do not do the risky thing", which is a valid lesson and not the one the reward intended. Project 5 met the mirror image: a process reward that paid per valid tool call taught a policy to spend forty-four steps across nine tasks making tool calls and to submit in only three of them.

Collapse is where the judgement framework earns its keep, because the symptom is invisible in a single aggregate: the number goes down, but *which* tasks and *which* behaviours went down is the diagnosis. Report per family, per task, per failure class - which is what Chapters 10 and 12 do, and what most papers do not.

### Lesson 8 · A reproduction checklist: the smallest experiment that tells you whether a paper is worth following

The checklist is four days of work, deliberately smaller than a reproduction. The goal is not to confirm the paper; it is to find out whether the paper's claim could be true *on your task*, which is a different and cheaper question.

**Day 1: can you run anything at all?** One sample through the code path, one scored instance, one weight update. If the artifact does not run, that is the verdict - "unrunnable" is information, and most reproduction failures are environment failures, not algorithm failures. This book's version took two containers and a missing `pandas` to learn.

**Day 2: reproduce the floor, not the headline.** Run the *baseline* the paper compares against, on your data, and see whether you land near the number they report for it. If your baseline is far from theirs, the headline is uninformative whatever it says - you are not in the same regime.

**Day 3: the two-condition comparison, n of at least 100.** With and without the intervention, same instances, same checker, interval reported. Write the interval down before you look at the difference, so the verdict is not a negotiation with yourself.

**Day 4: the ablation that isolates the claim.** Remove only the thing the abstract names. If the gain survives its removal, the paper has measured something else.

<figure class="book-figure">{{#include ../figures/13-reproduction-gates.svg}}</figure>

*Figure 13.1 - the four days, as four gates. Each one is cheaper than the next day it protects, and each one has a verdict attached, so the answer to "is this paper worth following" arrives with the reason it arrived. A paper that dies at gate one has cost a day, not a fortnight.*

Kill criteria, decided on day 0:

- **The floor does not reproduce.** Stop. The headline tells you about their cluster, not your task.
- **The interval overlaps.** You have a direction, not an effect. File it as "promising, unmeasured" and move on; do not ship it into a product on an overlapping interval.
- **The ablation does not isolate the claim.** Reduce the paper to the component you verified, and cite it that way.
- **You cannot state the compute.** You cannot compare it to anything, including your own pipeline.

What survives all four days goes on the pinboard, in one line, with a verdict that says which question decided it:

```text
[GRPO family] DAPO (arXiv:2503.14476) - dynamic sampling drops zero-variance groups
Verdict: adopt the sampler, park the rest. Why: our task sets are 4 of 6 and 7 of 9 saturated, so the
zero-variance filter is the only change that can act on our failure; clipping cannot help a group with no spread.
Reproduction: day 2 floor reproduced within 2 points; day 4 ablation isolates the sampler.
```

And the discipline that makes the notebook worth keeping: **a healthy share of verdicts are "do not follow".** Following everything is the same as judging nothing, and most papers in a fast-moving field are patching a problem someone else's task set has.

## Deliverable

A "pinboard notebook" you maintain yourself - one line per paper: where it sits, follow or not, and which of the seven questions decided the verdict.

```text
[GRPO family] Title (arXiv:xxxx) - patches the normalisation bias of group baselines
Verdict: follow. Why: directly relevant to my verifiable task, cheap to implement (loss change only).
Question 3 decides: at my n of 100, their reported gain is outside the interval.
```

## How you know you passed

- Given a new paper, you can say within five minutes which slot it belongs in and which existing paper it is patching.
- Every paper you decided to "follow" maps to a concrete experiment with a baseline, a comparison and a kill criterion.
- A healthy share of your verdicts are "do not follow", and each names the question that killed it.
- You have caught at least one claim that dissolved under question 3 - a rate with no interval, or an interval that overlapped.

## Reading

- The map itself: [the paper map](../appendix/paper-map.md) - seventy-six papers on eight shelves, with the empty slots named.
- Lessons 3 and 6 lean on this book's own records: Project 4's per-family table (Chapter 10), Project 5's loop measurements (Chapter 10), and Project 6's two-condition outcome table (Chapter 12).

## Backfill

- [ ] All four questions: this chapter is where the four questions harden into long-term judgement.

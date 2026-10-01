# Part 0 · Before You Start

> **Compute tier**: ✅ | **Status**: draft — the three lessons are written; the map gains a pointer or two as the later chapters settle

## First, the gist

Before learning anything, write down two things: *what problem am I trying to solve*, and *what hardware do I have*. Almost every disappointment in this field comes from skipping those two answers — people pick a method first, then discover it needs eight GPUs, or that it solves a problem they do not have.

## What you'll be able to do

- Say in one sentence which of the four questions you are actually stuck on.
- Decide up front what runs on your own machine and what needs rented hardware.
- Route yourself to the right chapter when something in your run goes wrong.

## Lessons

- [ ] 1. Why Agent RL is the dividing line in large-model work (copying good answers, or scoring attempts)
- [ ] 2. The map, part 1: RL basics and reward engineering
- [ ] 3. The map, part 2: the industrial battlefield and the judgement framework

### Lesson 1 · Why Agent RL is the dividing line in large-model work

There are only two ways to make a language model better at something, and everything in this book is built on the difference between them.

**Teach it by copying good answers.** You collect demonstrations — a prompt, and the answer you wish the model had given — and you train the model to produce answers like those. This is supervised fine-tuning (SFT), and it is how almost every model you have used reached the competence it has. *(textbook fact)*

**Let it try, score the attempt, and train it to do more of what scored well.** That is reinforcement learning (RL) in one sentence, and RL itself is defined on the [home page](../index.md). *(textbook fact)*

| | Supervised fine-tuning (SFT) | Reinforcement learning (RL) |
|---|---|---|
| What you need | Demonstrations: a prompt and the answer you wanted | A task, something to attempt it against, and a way to score the attempt |
| What the model learns | To imitate the demonstrations | To make higher-scoring attempts more likely |
| The ceiling | Whatever the demonstrations contain | Whatever the score can recognise |
| The cost | One pass over the data; cheap by comparison | Many attempts per update; this is where the money goes |
| The characteristic failure | Imitation of the surface form, with no recovery behaviour | Reward hacking: optimising the score instead of the goal |

*(my judgement on the failure column; the mechanism of both is textbook.)*

The ceiling row is the important one. Copying can only teach behaviour that someone already demonstrated, and the behaviour an agent needs most is the one nobody writes down: the fourth tool call failed, here is the clever retry. You cannot demonstrate every recovery from every failure — but you can *score* whether the task ended up solved. That is the whole reason this field exists.

**Where the dividing line falls for an agent.** An agent's task is not one answer, it is a sequence of decisions: search, read, write a file, run it, look at the error, try again. Two things change when you move from "answer this question" to "finish this task":

1. **You need an environment** — something the model can act on that answers back. A shell, a browser, a test runner, a simulated customer. Without one, there is no second step, and without the feedback from the first step there is no learning signal at all.
2. **The score arrives late.** Five turns of work produce one reward at the end, which leaves the question of which of the five hundred decisions deserved the credit — the subject of [Chapter 6](../06-credit-assignment/index.md).

**Why agent tasks are the tractable half of the field.** Scoring is cheap when the goal is a fact rather than an opinion. "Does the code pass the tests" is a fact; "is this summary good" is an opinion. A fact can be checked thousands of times an hour by a machine, and RL needs exactly that — thousands of scored attempts. This single property, not any algorithmic trick, is why progress has concentrated where a verifier can be written. *(my judgement, and here is its boundary: plenty of valuable work has no such checker, and [Chapter 5](../05-reward-engineering/index.md) is about what to do when you cannot write one.)*

**What it costs, before you commit.** RL spends its money on attempts, not on training. A small run — 1,000 tasks, 8 attempts each, 4 turns per attempt, 500 tokens per turn — has to generate about 16 million tokens per iteration just to produce the data. [Chapter 1](../01-landscape/index.md) does that arithmetic properly, and it is the number to look at before choosing a method.

**What changes for you.** Your job stops being "curate a dataset" and becomes "design a task, an environment, and a reward — and then defend the honesty of that feedback". *(my judgement)* Two prerequisites, honestly stated:

- You can write basic Python and you have met PyTorch. You have fine-tuned a model at least once, even with LoRA (low-rank adaptation). If not, do that first; this book does not teach either.
- You have patience for infrastructure. Most of the work in Part 3 is plumbing — queues, sandboxes, restarts, version numbers — not mathematics. Researchers who dislike that part should read this book and then rent someone to build it.

And one thing the dividing line is **not**: RL is not always the answer. A better prompt, a better tool, more demonstrations, or simply a smaller task often wins for a fraction of the cost. Decide that first, with [Chapter 1's decision test](../01-landscape/index.md) and [Figure 1.1](../01-landscape/index.md).

### Lesson 2 · The map, part 1: RL basics and reward engineering

The book is six parts. The first two give you the language and the signals; they are the part you read in order.

**Part 1 — A shared language.** Three chapters that make the vocabulary work:

- [Chapter 1 · The Big Picture](../01-landscape/index.md) — the four families of method, sorted by *who scores the attempt*, and how to tell which one your problem needs.
- [Chapter 2 · From MDP to LLM Post-Training](../02-mdp/index.md) — the vocabulary of trial and error, written down precisely, plus the five things SFT cannot fix and what it is actually for.
- [Chapter 3 · From Value Functions to Policy Gradients](../03-policy-gradient/index.md) — where the training signal comes from, ending with the generalised advantage estimate you will meet in every implementation.

**Part 2 — Signals.** Three chapters that decide whether your training signal tells the truth:

- [Chapter 4 · The Alignment Algorithm Family](../04-alignment-algorithms/index.md) — the whole family in outline and PPO (proximal policy optimisation) in full, once, properly.
- [Chapter 5 · Reward Engineering and Verifiers](../05-reward-engineering/index.md) — building a reward that is cheap to compute, hard to game, and actually measures the goal.
- [Chapter 6 · Credit Assignment](../06-credit-assignment/index.md) — from a reward at the end of the episode back to the individual decision, and the designs that survive contact with real tasks.

**Why this order, and not the order a course uses.** Four choices were made deliberately while re-arranging the syllabus: *(my judgement, and the reason for each)*

1. **Vocabulary before algorithms.** Read PPO first and it is a bag of ratios with a clipping function. Read Chapters 2 and 3 first and every symbol in it — advantage, critic, KL (Kullback-Leibler) penalty — already has a job.
2. **One algorithm in full, not six in outline.** Chapter 4 derives PPO completely and treats GRPO (group relative policy optimisation) and DPO (direct preference optimisation) as variations you can then read from their papers. Six partial derivations teach less than one complete one.
3. **Reward before runtime.** A perfect trainer with a broken reward produces a worse model, faster. Chapters 5 and 6 come before the industrial part on purpose.
4. **The industrial part last.** You cannot debug a distributed training run before you can read a single one, so Part 3 assumes Part 2.

**The cheap way in, if you are impatient.** Read Chapter 1, do Chapter 2's exercise on your own task, read Chapter 3, then stop reading and build the smallest thing that has a checker: one small programming task, a test suite that either passes or fails, and a reward that reports that fact. Everything after that is making it reliable. *(a plan, not a measurement — nobody has timed this path for you.)*

And the shortest useful answer to "which method should I use": read [Chapter 1](../01-landscape/index.md) and [Chapter 4](../04-alignment-algorithms/index.md) and stop. The rest of the book is for the people who have to build it, run it, and explain why it got worse. *(my judgement)*

### Lesson 3 · The map, part 2: the industrial battlefield and the judgement framework

The second half of the book is about the difference between a method that works in a paper and a system that works on Monday morning.

**Part 3 — Where projects actually die.** Three chapters, in the order the failures appear:

- [Chapter 7 · Data and Task Engineering](../07-data-and-tasks/index.md) — the task set *is* the specification. A task set that does not measure what you care about yields a model that is good at the wrong thing, and no amount of training fixes it. The deliverable is a task-set health report.
- [Chapter 8 · Environment Engineering](../08-environment-engineering/index.md) — environments that reset cleanly, run many attempts in parallel, and fail in ways you can see. If you cannot run a hundred attempts at once, the arithmetic from Lesson 1 never closes.
- [Chapter 9 · The Industrial Training Runtime](../09-training-runtime/index.md) — the six moving parts of a real RL loop (actor, critic, reference model, reward, rollout workers, weight synchronisation) and the ways it lies to you: a mask that is off by one, samples that went stale while queued, weights that never arrived at the worker. This is the ⚠️ chapter: it runs on the reference machine, shrunken.

**Part 4 — End to end.** [Chapter 10](../10-end-to-end-projects/index.md) has you run two complete projects and keep the logs, because the log is the evidence and the score is not. [Chapter 11](../11-evaluation/index.md) builds the evaluation that is not part of training, plus the regression script you run before every change — the thing that tells you a fix did not break something else.

**Part 5 — The frontier, and judgement.** [Chapter 12](../12-skills/index.md) covers capability libraries and self-evolving agents: handing the model a set of skills it can grow, and running that as a service rather than a script. [Chapter 13](../13-papers/index.md) maps the paper landscape and ends with the framework this whole book is really trying to give you: for any new claim, in any paper or release note, ask

1. **Which of the four questions does it answer?** Most announcements answer one and are silent on the others.
2. **What exactly did it measure?** A benchmark score, a throughput number, a human preference vote — each is evidence for a different claim, and none of them generalises for free.
3. **What would it cost me to reproduce?** Hardware, data, and hours, before you believe the result applies to your task.

That framework is the actual deliverable of this book. The methods will churn — the field has rewritten its own default algorithm several times in a few years — and the four questions have not. *(my judgement)*

The four questions, and the chapters that answer them:

<figure class="book-figure">{{#include ../figures/00-questions.svg}}</figure>

*Figure 0.1 — the four questions this book circles, and where each is answered. It is a routing table, not a reading order: come back to it on the day something is broken and you cannot tell what.*

## Deliverable

A one-page alignment sheet:

| Question | My answer |
|---|---|
| Which of the four questions am I stuck on? | |
| What agent task am I actually working on? | |
| What compute do I have (local / rented / cluster)? | |
| How many hours per week can I spend? | |
| What counts as "done" for me? | |

## How you know you passed

- You can name the *one* problem you are learning this for. "Because it is hot" does not count.
- You have written down which chapters you will run for real and which you will only read.
- Faced with a symptom — a reward that gets gamed, a run that crawls, a fix that breaks something else — you can name the chapter that owns it, using the routing table above.

## Folded away (admin, kept for reference only)

- The companion course materials: video, slides, community. This handbook is the main text; the materials are a second opinion, and a course outline was used only to choose an order.
- The course's own app, payments, community chat and resource-centre links, which do not belong in a personal handbook.
- Two items that used to live here have moved into the lessons: the prerequisites (basic Python, PyTorch, one fine-tuning run) into Lesson 1, and the four design trade-offs behind this reading order into Lesson 2.

## Reading

Nothing technical yet — this part is yours to fill in. Four reference pages in the appendix are worth skimming before you start:

- The [glossary](../appendix/glossary.md) defines every term the book uses, in two columns: the formal meaning, and the same thing in plain words.
- The [compute tiers](../appendix/compute-tiers.md) is the hardware ladder behind every chapter's ✅ / ⚠️ / ❌ label.
- The [project index](../appendix/projects.md) maps each hands-on project to the chapter that builds it.
- The [16-week plan](../appendix/schedule.md) turns the parts and chapters into a schedule, with checkpoints.

## Backfill

This opening part answers no technical question. Its only job is to pick which of the four questions is *yours* — and to leave you a routing table for the day you cannot tell what broke.

# Chapter 12 · Skills: Capability Libraries and Self-Evolution

> **Compute tier**: ✅ (no weight updates involved - a 1.7B model and an index) | **Status**: written
> **Project**: [Project 6 · Memento-Skills](../appendix/projects.md)
> **Structure**: the main track (8 lessons) is engineering you can ship. The optional lessons (3) are unsettled research, clearly marked as such.

## First, the gist

So far, getting better meant changing the weights. There is another way: keep the model frozen and let it accumulate **skills** - reusable, inspectable procedures kept outside the model - then teach it to fetch the right one at the right moment. That is cheaper, reversible, and auditable. This chapter builds one, measures it, and reports the result that did not cooperate: on a small task set, handing the model a correct procedure did not improve its score, and on two families it ruined work the model was already doing right.

## What you'll be able to do

- Say what counts as a skill, and where the line runs between a skill, a tool, a prompt and a weight update.
- Build retrieval over a skill library and measure it against baselines rather than impressions.
- Refuse a skill that only sounds right, using an admission gate instead of a review meeting.
- Decide, per capability, between training something into the weights and keeping it outside - and say what evidence would change your mind.

## Lessons

- [x] 1. What an agent skill actually is: five shapes and five boundaries
- [x] 2. Seven ways a skill library fails, and the fix that answers each
- [x] 3. Skill retrieval: routing, decoys, and what happens at a hundred skills
- [x] 4. Skills are code: versioning, the admission gate, and auditability
- [x] 5. Project 6a: a task set that recurs on purpose
- [x] 6. Project 6b: the loop measured - four conditions, one hundred instances
- [x] 7. Project 6c: writing a skill that helps, and the one that unlearned a working habit
- [x] 8. The decision framework: data, latency, reversibility, measurability - **train it into the weights, or keep it outside?**

### Lesson 1 · What an agent skill actually is: five shapes and five boundaries

A skill is **a procedure with a trigger and a provenance**: what to do, when to reach for it, and what happened last time it was used. Project 6 stores exactly that, because the storage format is the argument:

```python
@dataclass(frozen=True)
class SkillDoc:
    name: str        # stable identifier, versioned outside the text
    family: str      # the shape of problem it answers
    trigger: str     # the phrase a router matches against
    procedure: str   # what to do, in the imperative
```

Five shapes do the work, and they are not interchangeable:

| shape | carries | fails when |
|---|---|---|
| procedure note | the order of steps and the gotcha | it describes the task instead of the trap |
| script or snippet | a working implementation | the environment differs by one library version |
| checklist | what to verify before reporting | it is used as a description rather than a gate |
| worked example | the format, by imitation | the example is memorised and the rule is not |
| routing rule | which of the above applies | two rules claim the same task |

Five boundaries, each of which people cross by accident:

- **Skill versus tool.** A tool is a capability the harness exposes (`search`, `calc`); a skill is what to do with it. "Use `calc` for arithmetic" is neither: it is a prompt line the model should not need.
- **Skill versus prompt.** A prompt is written for one task and dies with it. A skill recurs, so it has to survive the task that produced it - which means it needs a trigger and a name.
- **Skill versus system prompt.** System-prompt text is always on and always paid for. A library is paid for once per retrieval instead of once per token.
- **Skill versus weights.** A weight update changes behaviour everywhere and cannot be rolled back per-task. A skill changes behaviour where it is retrieved and is a diff away from being removed. Chapter 12's last lesson is the decision procedure.
- **Skill versus documentation.** Documentation is for a human who chose to read it; a skill is retrieved by a machine that did not.

The test that separates them in one question: **if the model receives this text on every call, is the cost worth it?** If yes it is a prompt or a system message. If no, and it is worth carrying a name, a trigger and an evidence trail, it is a skill.

### Lesson 2 · Seven ways a skill library fails, and the fix that answers each

A library is a system with seven failure modes, and each one has a cheaper fix than the one people reach for first.

| failure | what it looks like | the fix this project uses |
|---|---|---|
| retrieval collision | two skills share vocabulary; the wrong one wins | document frequency weighting, and a symptom in the query |
| the plausible decoy | a procedure that is right about the topic and wrong about the rule | a behavioural admission gate, not a review |
| the stale skill | the world changed, the skill did not | provenance with the instances it was validated on |
| over-general skill | one skill claims five families and helps none | one skill per family, enforced by the library's key |
| context bloat | ten skills injected "just in case" | one retrieval per attempt, top-1 only |
| version drift | nobody knows which skill produced which result | the version is part of the name and the record |
| non-monotone regressions | adding a skill breaks a task that already worked | per-family measurement, not an aggregate |

The first two are the ones this chapter spends its budget on, because they are the ones that look solved. Retrieval collision is invisible when your library has five skills and fatal at fifty. The decoy is invisible when a human reviews the text, because a plausible procedure reads exactly like a true one - the difference is in what it makes the model *do*.

The last row is the one that cost this project its headline. An aggregate success rate is a sum over families, and a sum cannot tell you that two families went from ten out of ten to three out of ten while one went from zero to four. Every table in this chapter is therefore reported per family, and the aggregate is reported with its interval beside it.

### Lesson 3 · Skill retrieval: routing, decoys, and what happens at a hundred skills

The project generates ten families of recurring task - sum a value per group, collapse duplicates keeping the last record, normalise three date formats, total with tax and half-up rounding, rank with a tie-break, count every cell of a pair table, carry the last non-empty value down, convert units with a rounding rule, split an integer and place the remainder, find the first record that breaks a schema. Each comes back twenty times with different numbers, which is the property a skill library exists to exploit.

Each family also ships a **decoy**: a procedure that is on-topic and wrong in exactly one place. For `sum-by-group` the true skill says *sort the groups by name*; the decoy says *keep first-appearance order*. A reader cannot tell which is right without running them; that is the point.

Four routers, all standard library, measured on 60 queries (six instances per family), with the library padded to ten, fifty and a hundred documents. The padding is the experiment: at fifty and a hundred, ten of the extra documents are the decoys and the rest are unrelated skills.

| library | documents | keyword | vector | hybrid | idf |
|---|---|---|---|---|---|
| ten | 10 | 0.900 | 0.700 | 1.000 | 1.000 |
| fifty | 50 | 0.600 | 0.367 | 0.400 | **0.800** |
| hundred | 100 | 0.600 | 0.367 | 0.400 | **0.800** |

<figure class="book-figure">{{#include ../figures/12-retrieval-ladder.svg}}</figure>

*Figure 12.1 - recall at 1 for four routers at three library sizes, from the same 60 queries. The step down happens when the ten on-topic decoys enter, not when the library grows. Two routers start at 1.00; the one that collapses to 0.40 is the one weighting the task's own words, and the one that holds at 0.80 is the one weighting rarity.*

Three conclusions, each of which contradicts an intuition:

**The size of the library is not the problem.** Fifty and a hundred documents score identically. What costs recall is *similarity*: ten adjacent skills do more damage than eighty unrelated ones. A team that budgets for "we will outgrow a naive router" is budgeting for the wrong variable.

**A router that matches the task's own words is matching the wrong thing.** The decoys repeat the task vocabulary; the true skill names a gotcha. Plain keyword overlap falls from 0.900 to 0.600 the moment the decoys appear, and the character-trigram vector falls further (0.700 to 0.367) because it is even better at matching topic and even worse at matching intent.

**Rarity is the cheapest signal that separates a procedure from a restatement of the task.** The `idf` router weights tokens by how rare they are across the library, and adds a share of the **failure symptom** the previous attempt reported. It holds at 0.800. It was also written *after* the first three collapsed, which is the honest order of events and the reason its number is in the table rather than in a footnote.

The instructive failure is `hybrid`. It had the failure symptom available and still scored 0.400: a long procedure dilutes a Jaccard score while a short decoy concentrates it. **Having the right input is not the same as weighting it correctly**, and a router that looks sophisticated in review can be beaten by one line of document frequency.

### Lesson 4 · Skills are code: versioning, the admission gate, and auditability

The gate is the only part of a library that decides what is true, so it has to be behavioural. A candidate is admitted only if the **same policy**, on **held-out instances of its family**, answers more of them correctly with the procedure injected than without it:

```
admit(candidate)  iff  solved(instances | candidate.procedure) - solved(instances | nothing) >= 1
```

Three properties fall out of that definition and each one is load-bearing:

- **It cannot be satisfied by good writing.** A decoy that reads like a senior engineer's advice changes the model's behaviour for the worse, so its gain is negative and it is rejected. In the project's test suite ten true procedures are admitted and ten decoys are rejected, with a scripted policy standing in for the model so the gate's logic is tested without a GPU.
- **It cannot be satisfied by memorisation.** The comparison runs on held-out instances of the family, not on the instances the failure came from. A skill that only repairs the three examples it was written from is a cache entry dressed as knowledge.
- **It has a cost model.** One injection, one family, one procedure. Ten skills injected "just in case" is a prompt, and prompts are what this chapter is trying to avoid.

Versioning is the second half of the same idea: a skill is a document whose **name is stable and whose revision is append-only**, and every run record names the revision it used. The library keeps one skill per family and a second candidate replaces the incumbent only if it wins on its own held-out set - a tie keeps the incumbent, because churn is itself a failure mode. `Verdict` records the counts, so "why is this skill live" is a question with an answer in a file rather than in a memory.

This is what "skills are code" should mean in practice: a diff, a test, a version, and a rollback. Not a folder of markdown a human curates by vibes.

### Lesson 5 · Project 6a: a task set that recurs on purpose

No public dataset is built to make the same *shape* of problem come back with different numbers, and that recurrence is the entire premise of a skill library. So the project generates one: ten families, twenty seeded instances each, two hundred instances, every one of them reproducible from its family name and seed.

The design rules, in the order they matter:

1. **The oracle defines correct, not the skill.** Each family ships a reference implementation, and the checker compares exact text after collapsing whitespace. A task whose answer needs a human to judge cannot measure a skill's effect; this chapter's numbers are exact-match numbers.
2. **Every family carries a gotcha that fits in one sentence.** Sort by the label, not by first appearance. The remainder belongs to the last part. Round half up, not half even. Emit the empty cells. If the trap needs a paragraph, the "skill" is a chapter, and a chapter in the context window is a prompt, not a retrieved procedure.
3. **Every family also ships a decoy.** The decoy is the test of the gate, and the reason the retrieval measurement is honest: a library padded with unrelated skills is easy, and a library padded with plausible wrong ones is the real condition.
4. **The families are boring on purpose.** They are data-handling tasks with no tools, no filesystem and no ambiguity about the output format, so a failure is attributable to the model's arithmetic or its convention, never to a flaky sandbox. Project 5 already has the sandbox; this project is trying to isolate one variable.

The cost is that the task set is small and synthetic, and the chapter says so in every table. A skill library is a claim about *recurrence*, and ten families is enough to measure retrieval and enough to catch a broken injection, but it is not evidence about the population of real agent tasks.

### Lesson 6 · Project 6b: the loop measured - four conditions, one hundred instances

One policy (`qwen3-1.7b-base`), temperature zero, identical instances, identical checker, four conditions:

| condition | solved | truncated | solved / answered |
|---|---|---|---|
| bare: the question alone | **34/100** | 40 | 34/60 |
| oracle-skill: the true procedure, no retrieval error | 25/100 | 33 | 25/67 |
| keyword retrieval | 27/100 | 28 | 27/72 |
| idf retrieval | 29/100 | 28 | 29/72 |

Two facts about the harness belong in the result rather than in a footnote, because both would have silently produced a prettier table:

- **A reasoning model eats the budget before it answers.** The first run used a 96-token reply limit and every condition scored 0 of 20: the model wrote `<think>` until it ran out. At 2048 tokens, 40 of the 100 bare replies still end inside the reasoning block. Those rows are counted as **errors**, never as wrong answers, and the answered-only column sits beside the total instead of replacing it.
- **The ceiling condition exists to be a ceiling.** `oracle-skill` hands the model the true procedure for its own family, with no router in the way, so it is the best case a library could ever reach. It scored **nine points below the floor**.

The honest reading of the totals is a direction, not an effect. At a hundred instances the Wilson intervals are `bare` 25.5-43.7, `oracle-skill` 17.5-34.3, `keyword` 19.3-36.4 and `idf` 21.0-38.5; every pair overlaps, so "the library made it worse" is not a claim this sample size supports. Conditioning on the replies that were actually produced widens the gap - 56.7 per cent answered correctly in `bare` against 40.3 per cent in `idf` - and still does not settle it.

**Where these numbers come from.** Two committed records, not a table typed from a notebook: `p6-memento-skills/fixtures/retrieval.json` (60 queries, three library sizes, four routers) and `p6-memento-skills/fixtures/outcome.json` (400 rows: four conditions times one hundred instances, each with the answer, the expected answer and any harness error). Both are recomputed by `p6-memento-skills/checks/p6_claims.py`, which the repository's gate runs on every change - including the two claims that are easy to lose in a proofread: that the ceiling condition scored below the floor, and that the four totals' Wilson intervals all overlap.

What the run does settle is what it *rules out*: routing was not the bottleneck. The `idf` router put the right family on top for **100 rows out of 100**, so the difference between the conditions is the injection itself, not a wrong skill arriving.

### Lesson 7 · Project 6c: writing a skill that helps, and the one that unlearned a working habit

The per-family table is where the aggregate stops being noise and starts being an explanation:

| family | bare | oracle-skill | keyword | idf |
|---|---|---|---|---|
| sum-by-group | 3 | 3 | 3 | 3 |
| dedupe-keep-last | **10** | 7 | 7 | 5 |
| normalise-dates | 0 | 0 | 0 | 0 |
| round-currency | 3 | 2 | 1 | 3 |
| top-k-by | 5 | 5 | 5 | 6 |
| pivot-counts | **0** | 4 | 5 | 4 |
| fill-forward | 0 | 0 | 0 | 0 |
| unit-convert | 1 | 0 | 0 | 3 |
| split-remainder | **10** | 2 | 4 | 3 |
| schema-check | 2 | 2 | 2 | 2 |

<figure class="book-figure">{{#include ../figures/12-outcome-per-family.svg}}</figure>

*Figure 12.2 - the idf router measured against the question alone, family by family. The aggregate is a sum of this picture: one family gained four, two lost seven and five, four did not move. A total of 29 against 34 is exactly what these bars add up to, and it is the least informative number on the page.*

**One family was fixed by being told the rule.** `pivot-counts` went from 0 to 4: the failure was that the model omitted pairs that never occurred, and the procedure's one sentence says to enumerate the grid and default the missing cells to zero. That is a skill doing exactly what a skill is for - naming a convention the model could not guess.

**Two families were broken by being told a rule they already followed.** `split-remainder` went from 10 to 3 and `dedupe-keep-last` from 10 to 5. Both are integer procedures the model performed correctly when left alone; with the procedure in context it started rounding the share, or sorting the ids, or losing the last-record rule. The behaviour was there and the instruction overwrote it.

This is the third time this book has met the same shape - Project 4's warnings changed nothing until the harness refused the report, and Project 5's procedure convinced a policy to call tools forever - and the general form is worth stating plainly:

> **Telling a policy something is not free.** It is a change to the context, and a context change can remove a working habit as easily as it can supply a missing one.

The authoring consequence is the useful part. Before writing a skill, classify the failure:

| failure class | what the skill should be | what it should not be |
|---|---|---|
| the model does not know the convention | one sentence naming the convention and the default | a worked example it can imitate without the rule |
| the model knows and gets it wrong | a check to run before reporting, or nothing | an explanation of the convention |
| the format is wrong | the format line, verbatim | a procedure that buries it |
| the model cannot do the arithmetic | nothing - this is a weights problem | advice about being careful |

`split-remainder` and `dedupe-keep-last` are rows two and four wearing row one's clothes: the model already knew, and the skill had nothing to add. A library that measured per family and rejected a skill on the families that did not need it would have kept both of those ten-out-of-ten columns and still fixed `pivot-counts`.

What would make the library pay? Larger sample sizes would shrink the intervals but not the direction; a larger policy would follow a procedure more reliably; and a task set whose failures are convention-shaped rather than arithmetic-shaped would give the skill something to fix. All three are testable, and none of them is assumed here.

### Lesson 8 · The decision framework: data, latency, reversibility, measurability - **train it into the weights, or keep it outside?**

Four criteria, each answerable before the work starts:

| criterion | favours weights | favours a library |
|---|---|---|
| **data** | you have a verifier and can generate thousands of attempts | you have a handful of traces and a human who can name the rule |
| **latency** | the capability must be there every call, without retrieval | the capability is needed in one family of tasks |
| **reversibility** | a wrong update can be re-trained away | a wrong skill must be removable per task, today |
| **measurability** | you can measure the outcome, not just the format | you can measure with and without the injection, on held-out instances |

Two project results make the trade concrete rather than rhetorical. Training (Projects 4 and 5) cost a GPU, a served policy, a weight-sync path and a loop; it changed behaviour where the reward had variance and nowhere else. A library costs a router, a gate and an afternoon; it changed behaviour where the *procedure* named something the model did not know, and un-changed it where the model already did.

The decision procedure, in four questions:

1. **Can I state the capability as a rule?** If not, it is a policy question and belongs in weights.
2. **Can I measure one instance of it cheaply?** If not, neither option is available yet - the library is the cheaper way to find out whether the task is learnable at all.
3. **Does the model fail because it does not know the convention, or because it cannot execute?** Convention goes outside; execution goes into the weights.
4. **What is the rollback?** If the answer is "another training run", the cost of being wrong is high enough that the library is the better first experiment.

And one honest limit this chapter measured: with a 1.7B policy and these ten families, **the library could not be shown to help even when retrieval was perfect.** That is not an argument against skills. It is an argument against shipping a library without the `oracle-skill` row: without it, a negative result is indistinguishable from a broken router, and a positive one is indistinguishable from a lucky batch.

## Optional: the research zone

*(Lessons 9 to 11 are research results, not settled engineering — the same warning this chapter's metadata carries. Read them as a roadmap, not as instructions.)*

- [x] 9. Memento-Skills: skills as markdown memory, and the reflective loop
- [x] 10. EvoSkill / Trace2Skill / SkillClaw: three ways to put guardrails on a self-evolving agent
- [x] 11. A combined exercise: a daily AI-news assistant (fetch, evaluate, write a digest)

### Lesson 9 · Memento-Skills: skills as markdown memory, and the reflective loop

Memento-Skills (arXiv:2603.18743, March 2026) is the clearest statement of the design this chapter builds a small version of: an agent that improves itself **without touching the weights**, by keeping its skills as structured markdown files that act as persistent memory, and revisiting them through what the paper calls Read-Write Reflective Learning. The skills encode both the behaviour and the context, and the system starts from elementary capabilities - search, terminal - and grows the library by designing task-specific agents.

Three things are worth taking from it, and one is worth arguing with:

- **The storage format is the interface.** Markdown, not a binary index: a human can read the library, diff it, and delete a bad entry. Everything in Lesson 4 about versioning and auditability is a consequence of that choice.
- **The loop is closed by reflection, not by reward.** There is no reward signal; what plays the role of the gradient is the agent reading its own trace and deciding the skill was wrong.
- **Skills are memory, so retrieval is a memory problem.** The paper's framing - stateful prompts - is why this chapter treats the router as part of the method rather than as plumbing.
- **What the paper does not give you is a stopping condition.** A library that grows by reflection has no gate of its own; that is what Project 6's admission test is for, and it is the piece this chapter would add to the design.

### Lesson 10 · EvoSkill / Trace2Skill / SkillClaw: three ways to put guardrails on a self-evolving agent

Three 2026 papers, each attacking a different part of "the agent writes its own skills, and that goes wrong":

**EvoSkill** (arXiv:2603.02766, March 2026) discovers skills by **iterative failure analysis**: it reads execution failures, proposes candidate skills, and refines them against the tasks that failed. The interesting claim for a practitioner is that the *artifact being evolved* is the skill - a domain workflow plus code - rather than a prompt, which is what makes the result transferable across models.

**Trace2Skill** (arXiv:2603.25158, May 2026 revision) goes the other way through the same problem: it **consolidates many execution trajectories in parallel into one skill directory** by inductive reasoning, and it can either deepen a human-written skill or turn a weak model-written draft into something usable. It is the paper to read if your failure mode is "we have a thousand traces and no procedure".

**SkillClaw** (arXiv:2604.08377, April 2026) is about the multi-user case: skills stay static after deployment, so the same workflows and failures are rediscovered by every user. It treats interactions across users and across time as complementary signals for **collective** skill evolution. The governance question it raises is the one this chapter's gate answers for a single team: whose evidence admits a skill, and how is a bad one removed?

Read together, they map onto the three rings of Lesson 2: EvoSkill is the candidate generator, Trace2Skill is the distiller, SkillClaw is the fleet. None of them is a substitute for the admission test, because all three measure their own proposals with the same policy that proposed them.

*Citation note: all four arXiv identifiers in this chapter were resolved against the arXiv API before publication, and the one-line descriptions above are taken from the abstracts. Details this chapter did not verify - a specific elite-pool mechanism in EvoSkill, or Memento-Skills' exact loop ordering - are deliberately not asserted.*

### Lesson 11 · A combined exercise: a daily AI-news assistant (fetch, evaluate, write a digest)

A design brief, not a result. The exercise is to build a skill library for a task that recurs every day, and to measure it the way this chapter measures its ten families:

1. **Define the recurrence.** Fetch twenty sources, keep the five that matter, write a digest. The "family" is the digest task; the instances are days.
2. **Write the check before the skill.** What makes a digest good is not exact-match, so the exercise forces you to build a small rubric - coverage of named events, no duplicate stories, every claim traceable to a fetched URL - and to score it mechanically. A skill library without a checker becomes an opinion poll.
3. **Ship three skills, not thirty.** One for source triage, one for deduplication, one for the digest's format. Then measure with and without each, per day, and keep the ones whose gain survives the interval.
4. **Include the oracle condition.** Hand the assistant the *correct* procedure for a day you have already digested, and see whether it improves. If it does not, the library is not the problem you think you have - this is exactly the check that turned Project 6's headline upside down.

The deliverable is a report with four numbers per skill (with, without, instances, interval) and a sentence of the form "we kept this skill because ..." that cites them.

## Deliverable

- A versioned skill library with an admission gate that rejects a plausible decoy (Project 6: ten admitted, ten rejected)
- A retrieval-quality report at three library sizes, against two baselines plus the router you would actually ship
- An outcome table with four conditions - bare, oracle-skill, and two routers - reported per family, not only in aggregate
- A written record of one "internalise vs externalise" decision using the four criteria

## How you know you passed

- Every skill creation or edit is auditable as a diff with a version, and the run record names the revision it used.
- You can quote a retrieval number with its baselines and the library size it was measured at, rather than an impression.
- You have an `oracle-skill` row in your outcome table, and you can say what it ruled out.
- For one specific skill, you can explain why you chose *not* to train it into the weights - and what result would change your mind.

## Reading

- Memento-Skills (read-execute-reflect-write loop, base model frozen): arXiv:2603.18743
- EvoSkill (separates create from edit; uses an elite pool to bound growth): arXiv:2603.02766
- Trace2Skill (distils trajectory-local lessons into transferable skills): arXiv:2603.25158
- SkillClaw (collective skill evolution across users): arXiv:2604.08377

## Backfill

- [ ] Question 1: When should I reach for RL at all? (Here the answer is sometimes "I should not".)

# Extra Modules (Optional)

> **Compute tier**: ✅ | **Status**: draft — four module outlines; each becomes a worked note when the plan reaches week 16
> **Rule of this book**: these modules sit outside the numbered chapters, so they carry the same skeleton rather than a looser one. A reader deciding whether to spend a week here needs the same deliverable, the same pass criteria and the same source list as anywhere else — and, just as often, needs permission to answer "not for me".

## First, the gist

Four gaps that sit outside the main track: one or two lessons each, read on demand, none of them blocking the main line. In plain words:

- **Environment-side safety** — the same story as reward hacking, told from the other side: the model exploited the *world* rather than the judge. Poisoned training environments, sandbox escapes, tool calls that exceed their authority.
- **Multi-agent RL** — several models with different roles, one shared outcome, and the question of who owns a failure.
- **Domain agents** — software engineering and computer use: the two domains where checking an answer is unusually cheap, and what that does to reward design.
- **Context management as a policy** — memory, compression, summarisation and retrieval are decisions too, and they can be trained rather than hand-coded.

If you are working through the main line, the honest default is to skip all four on the first pass. Come back when one of them names the thing that is blocking you, and note that a "no" is a real answer — the [16-week plan](../appendix/schedule.md) puts the modules in week 16 for exactly that reason.

## What you'll be able to do

- Say which of the four bears on the task you are actually working on, and write down the "not yet" for the other three.
- Name the verification source each module brings — what it can check, and the first thing it cannot.
- Decide, for one module, whether it changes your next step or is a detour.
- Explain why two of the four are *domain* concerns (software engineering, computer use) and two are *system* concerns (safety, context).

## Lessons

- [ ] 1. Environment-side safety: when the model exploits the world, not the judge
- [ ] 2. Multi-agent RL: cooperative rewards, and who owns the failure
- [ ] 3. Domain agents: software engineering and computer use
- [ ] 4. Context management as a policy

### Lesson 1 · Environment-side safety

**The reframing.** Reward hacking is usually told as "the model fooled the judge". This module is the other half of the story: "the model exploited the *world*". The tools it exploits are not the reward model but the sandbox, the file system, the tool API and the training environment itself. *(my judgement on the framing; the mechanisms below are the module's outline)*

- What kind of policy do you train when prompt injection has poisoned the training environment? Indirect injection is the reported mechanism: text the agent reads on its way to a task carries instructions, and a policy trained to obey instructions has no way to tell "the user's task" from "text that mentions a task" *(paper result, reported and not reproduced — Indirect Prompt Injection, in Reading)*.
- Sandbox escapes, and tool calls that exceed their authority: the failure is not the model's imagination but the *permissions* it was handed.
- Reward functions hijacked by environment bugs — reward hacking, systems edition. A test harness that always exits zero is a reward model with a bug in it.
- Defences, in the order they pay for themselves: environment auditing, least privilege, injection detection, and isolating the data your reward depends on.

**What it costs.** Auditing a training environment is a pipeline you own and maintain, and least privilege shrinks the set of tasks you can train on at all: the sandbox that stops the agent from writing outside its directory also stops it from demonstrating the file-management skill you wanted. *(my judgement)*

### Lesson 2 · Multi-agent RL

**Why the shared reward is the hard part.** With one agent, the reward is about one policy. With several roles — planner and executor, writer and reviewer, seller and buyer — the outcome is shared, and a shared reward cannot say which role earned it. That is credit assignment ([Chapter 6](../06-credit-assignment/index.md)) with an extra axis: not only *which step*, but *whose step*. *(textbook fact)*

- Designing cooperative rewards when agents have different roles: per-role rewards risk rewarding a role for a failure the group survived.
- Credit assignment across roles: who owns the failure? A group that succeeds despite one bad role teaches that role nothing.
- When multi-agent beats a single agent, and when it honestly does not. The reported failure taxonomy of multi-agent systems is a useful cold shower: most failures are specification and coordination problems rather than model capability *(paper result, reported and not reproduced — Why Do Multi-Agent LLM Systems Fail?, in Reading)*.

**What it costs.** The rollout bill multiplies by the number of interacting roles, and the credit-assignment design is the deliverable — the training run is the easy half. *(my judgement)*

### Lesson 3 · Domain agents: software engineering and computer use

**Why these two domains, and not "agents" in general.** They are where verification is cheap: a test suite decides whether a patch is correct, and a rendered screen state decides whether a click landed. Cheap verification is what makes the RLVR (reinforcement learning with verifiable rewards) route ([Chapter 4](../04-alignment-algorithms/index.md), Lesson 8) affordable, and it is why these two domains matured first. *(my judgement, on top of the reported benchmarks below)*

- Software engineering: the reported benchmark family is SWE-bench and its agent interface *(paper results, reported and not reproduced — SWE-bench and SWE-agent, in Reading)*. The reward is the test outcome; the lesson is how much of the difficulty lives in the *interface* rather than the model.
- Computer use: WebArena and AgentBench-style environments make the reward the goal state rather than the pixels *(paper results, reported and not reproduced — both in Reading)*.
- How reward design and environment construction differ from general tasks: the verifier is an artefact you build and version, not a model you train.

**What it costs.** The environment is the project. A test suite that passes on the solution you happen to have in mind is a specification bug, and the agent will find it (the coverage- is-the-ceiling argument from [Chapter 4](../04-alignment-algorithms/index.md), Lesson 8).

### Lesson 4 · Context management as a policy

**The claim to test.** Memory, compression, summarisation and retrieval are usually built as plumbing around the model. This module treats them as decisions the model makes, and therefore as things it can be trained to make better. *(my judgement)*

- Long contexts are not free: what a model actually uses of a long context is not what a reader assumes, and information in the middle is the most likely to be missed *(paper result, reported and not reproduced — Lost in the Middle, in Reading)*.
- Treating memory as an operating system rather than a buffer: paging, explicit memory tools, and the model deciding what to keep *(paper result, reported and not reproduced — MemGPT, in Reading)*.
- How this divides labour with the external-skill approach in [Chapter 12](../12-skills/index.md): trained-in context habits are fast and generalise badly; an external library is inspectable and costs a tool call.
- What long contexts do to rollout cost: every extra token is paid at every step of every episode, so context policy is a line in the compute budget ([Chapter 8](../08-environment-engineering/index.md)).

**What it costs.** Training a memory policy needs a task where memory actually matters, and building that task is harder than the training. The cheaper first step is usually to fix the plumbing and measure. *(my judgement)*

## Deliverable

One page per module, answering a single question: **does this affect the task I am working on, and should it be my next step?** A module you reject gets a reason, not silence.

| Module | Does it touch my task? | The verification source it brings | Next step, or why not |
|---|---|---|---|
| Environment-side safety | | | |
| Multi-agent RL | | | |
| Domain agents (software engineering, computer use) | | | |
| Context management as a policy | | | |

## How you know you passed

- Each of the four rows has an answer, and the "why not" column names a real cost rather than "later".
- For the module you chose, you can point at the verification source you would build first — a sandbox policy, a per-role reward, a test suite, or a measurement of what the model actually uses from its context.
- You can say which of this book's four questions your chosen module serves, and which it leaves alone.

## Reading

Titles and arXiv IDs were checked against the arXiv API on 2026-10-09 (IDs marked "verified"). Entries marked as reused were verified where they say, and are not re-checked here. Numbers quoted from papers are **reported, not reproduced**.

- **The safety framing this module starts from** — Amodei et al., *Concrete Problems in AI Safety*, arXiv:1606.06565 — verified in Chapter 4's Reading, 2026-10-09; the reward-hacking and safe-exploration problem statements.
- **The injection mechanism** — Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*, arXiv:2302.12173 — verified. Lesson 1.
- **Multi-agent systems, and why they fail** — Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, arXiv:2503.13657 — verified. Lesson 2's taxonomy.
- **Software engineering as a benchmark** — Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, arXiv:2310.06770, and Yang et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*, arXiv:2405.15793 — both verified. Lesson 3.
- **Computer use as a benchmark** — Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents*, arXiv:2307.13854, and Liu et al., *AgentBench: Evaluating LLMs as Agents*, arXiv:2308.03688 — both verified. Lesson 3.
- **What a long context is actually used for** — Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv:2307.03172 — verified. Lesson 4.
- **Memory as an operating system** — Packer et al., *MemGPT: Towards LLMs as Operating Systems*, arXiv:2310.08560 — verified. Lesson 4.
- **Where the trained-in/external split is argued** — [Chapter 12](../12-skills/index.md)'s Reading, once that chapter is written.

## Backfill

- [ ] Question 2: How do I design the reward? (Safety module)
- [ ] Question 3: How do I debug a long multi-turn failure? (Multi-agent, context management)

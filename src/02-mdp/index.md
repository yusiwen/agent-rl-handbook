# Chapter 2 · From MDP to LLM Post-Training

> **Compute tier**: ✅ (this chapter trains nothing; it is notation and definition)
> **Status**: draft — all four lessons have prose, no numbers measured on hardware yet
> **Rule of this book**: this chapter builds the **shared language**. The value functions and the Bellman equation that use it are [Chapter 3](../03-policy-gradient/index.md). The algorithms that optimise it are [Chapter 4](../04-alignment-algorithms/index.md).

## First, the gist

An MDP is a tidy way of writing down a trial-and-error problem: *what I see*, *what I can do*, *what I get for doing it*, *what happens next*, and *how much I care about the future*. Writing an LLM task in this form forces you to answer questions you would otherwise leave fuzzy — how big is one "action", where does the feedback come from, and how long is one episode. Those answers decide which training method you can even use.

## What you'll be able to do

- Write any agent task as a five-part MDP, including the awkward part (partial observability).
- Explain in plain words why supervised fine-tuning (SFT) is not enough, and what exactly it misses.
- Say what one "action" is in your task, and why that choice matters later.

## Lessons

- [ ] 1. The engine of RL: from the interaction loop to expected return
- [ ] 2. Why SFT stops being enough: five classes of problems it cannot fix
- [ ] 3. The MDP five-tuple, explained slowly — and why partial observability (POMDP) shows up
- [ ] 4. Trajectory, episode and state in agent settings: **many turns = one long episode, state = context + environment**

### Lesson 1 · The engine of RL: from the interaction loop to expected return

**The loop.** Reinforcement learning is the study of one picture, repeated. *(textbook fact)*

```text
        ┌──────────────────────────────────────────────┐
        │                                              │
   observe state s ──► choose action a ──► receive reward r
        ▲                                    and the next state s'
        └──────────────────────────────────────────────┘
```

Two words carry the whole picture. The **agent** is the thing that chooses; the **environment** is everything else, including the scoring. The agent never sees the future, and it never gets to try every option.

**What is actually being maximised.** Not the reward you happened to collect, but the reward you collect *on average, over many attempts*. One number is defined to make that precise — the **return**, the sum of all rewards from this moment on, with later rewards faded by a discount factor:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

- $G_t$ — the return from step $t$ onwards.
- $r_{t+k}$ — the reward received $k$ steps after now.
- $\gamma$ (gamma), between 0 and 1 — how much a reward now is worth compared to the same reward later.

**Why the discount exists.** Three reasons, and beginners usually hear only the first. *(textbook fact)*

1. It expresses impatience. A reward today is worth more than the same reward tomorrow, because the future is uncertain.
2. It makes the sum finite when the episode never ends, so the arithmetic does not diverge.
3. It is a knob for *how far ahead* the agent plans. $\gamma = 0$ gives a greedy agent that only sees the next reward; $\gamma$ close to 1 gives a patient one that optimises the far future.

**The objective.** The agent's behaviour is a **policy**: a rule that turns a situation into a probability for each action, written $\pi_\theta(a \mid s)$. For a language model (LLM), that is exactly the softmax over the next token, and $\theta$ is the model's weights — which is why "training an LLM with RL" means "adjusting the weights so that better-scoring text becomes more likely". Training maximises the expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ G \right]
$$

The expectation symbol $\mathbb{E}$ is doing real work here: you cannot control which attempt the model produces on a given day, so you control the *distribution* of attempts instead. Every method in this book is a way of estimating or improving that expectation without seeing all possible attempts.

**The tension the loop creates: explore or exploit.** If you always take the action that currently looks best, you never find out whether a different action was better. If you always try something new, you never use what you learned. Every RL algorithm is a compromise between the two, and every compromise costs samples. *(textbook fact)*

**What this lesson is for.** Everything in the rest of this book — advantage, KL (Kullback-Leibler) penalties, group-relative baselines — is machinery for estimating $J(\theta)$ reliably from a small number of expensive attempts. The notation above is the vocabulary those chapters assume.

### Lesson 2 · Why SFT stops being enough: five classes of problems it cannot fix

Chapter 1 said the preference stage exists because supervised fine-tuning (SFT) can only copy. Here are the five specific failures, with the cost of each. *(the mechanisms are textbook; the papers cited are marked as reported results)*

**1. SFT cannot express "this is better than that".** A demonstration says *this answer is acceptable*. It cannot say *this answer is better than that one* — and for hard tasks the ranking is far easier to produce than the perfect answer. Cost: your labelling budget buys you examples instead of comparisons, and comparisons carry more information per minute of human attention. Reported: a small set of excellent demonstrations (roughly a thousand) can carry an SFT model surprisingly far, which shows the limit is the *quality and variety* of the demonstrations, not only their number. *(paper result, reported and not reproduced — LIMA, under Reading)*

**2. SFT cannot explore.** It learns from the paths you show it. If a better path exists that nobody demonstrated, SFT has no mechanism for finding it — no sampling, no scoring, no pressure toward it. Cost: any behaviour that is not in your data is unreachable, at any training budget.

**3. Errors compound, and the model never practises recovering.** SFT is trained on clean demonstrations, one token at a time, with the correct prefix always given. At inference time the model sees *its own* output as the prefix. The two distributions differ, and a wrong step in the middle of an answer is a mistake the model has never been trained to climb out of. Cost: long outputs and long agent runs degrade with length. This is why "make the model check its own work" is hard to teach by demonstration alone.

**4. Imitation copies style, not capability.** A model fine-tuned to imitate a stronger model's outputs can look much better while being no more capable on held-out problems: surface form is easy to imitate, the reasoning behind it is not. Reported with direct measurements on hard benchmarks. *(paper result, reported and not reproduced — The False Promise of Imitating Proprietary LLMs, under Reading)* And in a controlled comparison, SFT was found to memorise the training task while RL generalised better to out-of-distribution variants of it. *(paper result, reported and not reproduced — SFT Memorizes, RL Generalizes, under Reading)*

**5. SFT has no way to use a checker.** For verifiable tasks you can *compute* whether an answer is right — the whole point of Chapter 1, Lesson 2. SFT cannot consume that signal directly; your only lever is to label more data by hand. Cost: your human labelling budget becomes a hard ceiling, and it never improves by itself.

**What SFT is still irreplaceable for.** *(my judgement, and the boundary is: RL needs a non-zero starting point)*

- **Format and behaviour** — answering as an assistant rather than continuing text; following a house style; the syntax of your tools.
- **Teaching the model about things it never saw in pretraining** — your internal API, your database schema, your file layout. This is the "injector" half of Chapter 1: data injects, RL amplifies.
- **Getting pass@k above zero.** RL multiplies a probability that already exists. If the behaviour is absent, SFT (or a tool, or a bigger model) is what puts it there.

**The practical pipeline.** SFT first, until the model sometimes succeeds. Then the preference stage, to make success reliable. The order is not a fashion: RL on a model that never succeeds has nothing to amplify, and SFT on top of RL is how "distillation of your own best trajectories" works when you need to freeze the gains. Both halves appear again in [Chapter 4](../04-alignment-algorithms/index.md).

### Lesson 3 · The MDP five-tuple, explained slowly — and why POMDP shows up

**The five-tuple.** A task is an MDP (Markov decision process) if you can write it as $(S, A, P, R, \gamma)$. *(textbook fact)* The letters are unhelpful on first contact, so here they are twice: in plain words, and in an agent task.

| Element | Plain words | In an agent task | What beginners get wrong |
|---|---|---|---|
| $S$ — states | Everything relevant about the situation right now | The prompt plus the whole conversation history, plus what the environment currently is | Forgetting that the *model's own previous output* is part of the state, not part of the action |
| $A$ — actions | What the agent may choose to do | One token, or one tool call, or one whole turn — **your choice** | Never deciding this explicitly, then wondering why credit assignment is hard |
| $P$ — transition | What happens next as a result of the action | Appending a token to the context; a tool's real effect on a real system | Assuming the environment is "the model"; it is everything outside the policy |
| $R$ — reward | The score for this step | Usually zero everywhere, one number at the end | Believing a dense reward is always better — it is not, it costs a scorer per step |
| $\gamma$ — discount | How much the future counts | Usually $\gamma = 1$ (episodes are finite), less when a run can drag on | Setting $\gamma < 1$ on a long episode and quietly losing the terminal reward |

**The Markov property: the one-line definition worth memorising.** The state is "enough" if knowing it makes the past irrelevant — two different histories that lead to the same state must have the same future. *(textbook fact)* Cost of getting it wrong: if your state omits something the reward depends on (a file whose content matters, a tool that was called earlier), the task is not an MDP, and the usual algorithms silently train against a moving target.

**Why the token-level view is a real MDP.** For a language model, generating one token changes the context by appending exactly that token. Given the token, the "transition" is deterministic: the next state is the old context plus the sampled token. So the task can legitimately be written as a *token-level MDP*, with one token per action. This matters later, because most of the machinery you will meet (discounting per token, the generalised advantage estimate) assumes exactly this shape. *(the token-level formalisation is standard practice; see the references under Reading)*

**Two levels of action granularity, and what each costs.** *(my judgement)*

| Granularity | The action is | Pros | Costs |
|---|---|---|---|
| Token level | One token | Fits the standard machinery; per-token probabilities are already available | A 500-token answer is 500 decisions; the reward arrives once at the end of all of them |
| Turn level | One message or one tool call | Matches how humans talk about agent progress; far fewer decisions to credit | Throws away the per-token probabilities the model gives you for free; hides *why* a turn was bad |

Most production systems use a hybrid: the objective is defined over tokens (because that is what the optimiser can touch), and the *diagnosis* — "which turn was the bad one" — happens at turn level. That split is the subject of [Chapter 6](../06-credit-assignment/index.md).

**POMDP: the honest version of the picture.** In practice the agent does not see the true state. It sees an observation: the tool output you chose to show, the part of the file you printed, the screenshot you took. The formal name is a partially observable MDP, and the practical consequence is blunt: **the agent's context is a belief about the world, not the world.** *(textbook fact — Sutton and Barto; POMDP, Kaelbling, Littman and Cassandra, under Reading)* Three consequences worth more than the notation:

1. **The observation is a design decision, not a given.** Reading out the last 50 error lines instead of the whole log is a choice with training consequences: it decides what the model can possibly learn to react to.
2. **Memory is a summary of the past, and it can be wrong.** Compaction, summaries and scratchpads are all approximations of a state the agent cannot see.
3. **Environment feedback quality is part of the reward design.** An environment that returns "error" instead of the actual error message has made the task harder than it needed to be. [Chapter 8](../08-environment-engineering/index.md) is where that gets engineered.

### Lesson 4 · Trajectory, episode and state in agent settings

**Four words, defined once.**

- **Trajectory** — one full sequence of interaction: state, action, reward, next state, repeated. It is the unit of data RL learns from. *(textbook fact)*
- **Episode** — one trajectory, from start to a **terminal state**, after which the environment resets.
- **Horizon** — the length of an episode, in steps. Finite for tasks, potentially endless for continuing problems.
- **Loss mask** — which tokens you actually train on. This one is specific to language models, and it is where beginners get burned.

**In an agent, one episode is one whole task, not one turn.** This is the single most useful sentence in this lesson. A five-turn conversation with two tool calls is **one** episode with many steps, and one terminal reward at the end.

**Why that hurts, in numbers.** *(arithmetic on stated assumptions, not a measurement)*

| Quantity | Value |
|---|---|
| Turns per episode | 5 |
| Model-generated tokens per turn | 400 |
| Tool output tokens per turn (not your model's doing) | 200 |
| Tokens the model produced in the episode | 2,000 |
| Reward signals received | **1** |

Two thousand decisions, one number of feedback. That ratio is the credit-assignment problem, and it is why Part 3 of this book exists rather than skipping straight to the algorithms.

<figure class="book-figure">{{#include ../figures/02-loss-mask.svg}}</figure>

*Figure 2.1 — one episode of five turns. The model's own tokens are the only ones trained on; the tool's output is masked out of the loss; and a single reward arrives at the end of the whole task, not at the end of each turn.*

**The loss mask, concretely.** A trajectory contains tokens from three sources: the system prompt, the environment (tool outputs, files, user messages) and the model itself. Only the last group is your policy, so only the last group may contribute to the loss. Failing to mask tool output teaches the model to *imitate the tool* — and it will happily do that. *(my judgement on the failure mode; the mechanism is standard practice)* This is why a training framework's data structure carries a mask alongside the tokens, and why [Chapter 9](../09-training-runtime/index.md) spends time on exactly that structure.

**Deciding where the episode ends is a design decision with a bill.** *(my judgement)*

| Terminal rule | What the agent learns if you choose it |
|---|---|
| Success is detected | The thing you actually wanted |
| Step cap reached with no success | Keep trying is not enough; but a cap set too low teaches it to rush |
| Timeout or tool crash | Luck matters; failures that were not its fault get punished equally |
| Never (no terminal state) | Loops and stalls are never punished — the most common self-inflicted failure in agent RL |

**State in an agent setting is two things, and they drift apart.** The model's state is the context it can see: system prompt, task, history, tool results. The environment's state lives outside the model entirely: files on disk, database rows, a half-finished order, a browser's actual page. The model's belief can be *stale* (it read the file before its own edit) or *wrong* (a tool reported success that did not happen). Every serious agent project eventually builds something to keep the two in sync. Cost if you do not: the reward tells the model it failed for reasons it could not observe, and no amount of training fixes an unobservable cause.

**A worked example: filling in the tuple for a real task.**

> **Task**: an agent that fixes failing tests in a Python repository.
>
> | Element | Value for this task |
> |---|---|
> | $S$ state | The conversation so far, plus the repository's real contents (partially observed) |
> | $A$ action | One tool call (read a file, edit a file, run the test suite) — turn-level |
> | $P$ transition | The tools' real effects on the repository, plus your own appending of results |
> | $R$ reward | $+1$ if the suite passes at the end; $0$ otherwise; possible small penalty per step |
> | $\gamma$ | $1$ — episodes are finite and the reward is at the end |
> | Terminal | Suite passes, or the step cap is hit |
> | Partial observability | The model never sees the whole repository, only what it has read |
> | Loss mask | Tool output and test logs are masked out; only the model's own tokens are trained |

**Checklist before you move on.** *(this is the acceptance test for the deliverable)*

- [ ] The action granularity is written down (token, tool call, or turn).
- [ ] The reward is labelled sparse or dense, and outcome or process.
- [ ] The terminal condition is stated, including the failure cases.
- [ ] You can name where the partial observability comes from and what you show the model.
- [ ] You know which tokens will be masked out of the loss.

## Deliverable

Your agent task written as an MDP, plus a first draft of the reward:

```text
State s        =
Action a       =
Reward r       =
Transition P   =
Discount γ     =
```

Add three lines the template does not show: the terminal condition, the source of partial observability, and which tokens are masked out of the loss.

## How you know you passed

- The **action granularity** is explicit. One token? One tool call? One whole turn? That choice determines which credit-assignment scheme you will need in Chapter 6.
- You can say whether your reward is sparse or dense, and whether it judges the final answer or the process.
- You can point at where the partial observability comes from.

## Reading

Titles and arXiv IDs were checked against the arXiv API on 2026-09-30 (IDs marked "verified"). Book and blog references carry no arXiv ID, and are named as such. Numbers quoted from papers are **reported, not reproduced**.

- **The standard reference** — Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd edition, free official PDF) — no arXiv ID; the source for the loop, return, discounting, Markov property and exploration used above.
- **Partial observability** — Kaelbling, Littman and Cassandra, *Planning and acting in partially observable stochastic domains*, Artificial Intelligence 101 (1998) — no arXiv ID; the origin of the POMDP treatment in Lesson 3.
- **Generalised advantage estimation** — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438 — verified. The discounting and advantage machinery this book uses in Chapter 3.
- **What a small, excellent demonstration set can do** — Zhou et al., *LIMA: Less Is More for Alignment*, arXiv:2305.11206 — verified. Evidence for the ceiling of pure SFT (Lesson 2, point 1).
- **Style is not capability** — Gudibande et al., *The False Promise of Imitating Proprietary LLMs*, arXiv:2305.15717 — verified. Evidence for Lesson 2, point 4.
- **SFT memorises, RL generalises** — Chu et al., *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*, arXiv:2501.17161 — verified. The controlled comparison cited in Lesson 2, point 4.
- **Writing an agent task as a multi-turn MDP** — Zhou et al., *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL*, arXiv:2402.19446 — verified. The hierarchical, per-turn formalisation of the "one episode = one task" idea in Lesson 4.
- **Verifiers and process supervision** — Cobbe et al., *Training Verifiers to Solve Math Word Problems*, arXiv:2110.14168; Lightman et al., *Let's Verify Step by Step*, arXiv:2305.20050 — both verified. Background for outcome versus process reward, developed in Chapter 5.
- **Reward overoptimisation** — Gao, Schulman and Hilton, *Scaling Laws for Reward Model Overoptimization*, arXiv:2210.10760 — verified. What happens when the score stops measuring the thing you wanted; developed in Chapter 5.
- **Chain-of-thought** — Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, arXiv:2201.11903 — verified. The behaviour agent RL most often tries to amplify.

## Backfill

- [x] Question 1: When should I reach for RL at all? — this chapter supplies the first half: what SFT cannot do (Lesson 2), and what "one action" means in your task (Lessons 3–4).

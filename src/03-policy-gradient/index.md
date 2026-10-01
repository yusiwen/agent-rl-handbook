# Chapter 3 · From Value Functions to Policy Gradients

> **Compute tier**: ✅ (this chapter trains nothing; it is derivation on paper)
> **Status**: draft — all six lessons have prose, no numbers measured on hardware yet
> **Note**: the full PPO (proximal policy optimisation) objective is **not** here — it lives in [Chapter 4](../04-alignment-algorithms/index.md). This chapter stops at GAE (generalised advantage estimation) and answers one question: *where does the gradient come from?*
> **Rule of this book**: the meaning of every algorithm is explained once, in [Chapter 4](../04-alignment-algorithms/index.md). This chapter is the machinery underneath them.

## First, the gist

To improve a model you need a direction to move in. There are two ways to find it: estimate *how good each situation is* (values), or directly nudge whatever the model did more/less depending on how it turned out (policy gradients). This chapter builds both, then combines them — and explains why, for language models, feedback arrives so late that estimating "how good this was" is genuinely hard.

## What you'll be able to do

- Explain the difference between a value, a Q-value and an advantage without looking anything up.
- Derive the policy-gradient formula in a few lines, and say why a "baseline" reduces noise.
- Explain why the LLM setting is naturally sparse: not because feedback is rare, but because it arrives at the *end* of a long sequence.

## Lessons

- [ ] 1. Policies, returns and value functions: $V$, $Q$ and the advantage $A$
- [ ] 2. The Bellman equations: why a return can be written recursively
- [ ] 3. Three ways to solve an RL problem: dynamic programming → Monte Carlo → temporal difference (TD)
- [ ] 4. Value methods vs policy methods vs actor-critic — and where the off-policy line (Q-learning, DQN) sits
- [ ] 5. Policy-gradient theorem → REINFORCE → variance reduction → **GAE (we stop here)**
- [ ] 6. A bridge: from online RL to offline preference optimisation (Chapter 4 takes it from here)

### Lesson 1 · Policies, returns and value functions: value, Q-value and advantage

**One policy, three questions.** The policy $\pi_\theta(a \mid s)$ is the thing being trained. The three functions below are three different questions you can ask about it. *(textbook fact)*

| Symbol | Ask it this | Formula | Plain words |
|---|---|---|---|
| $V^\pi(s)$ | *Value* | $\mathbb{E}_\pi[G_t \mid s_t = s]$ | How good is this **situation** if I keep playing like this? |
| $Q^\pi(s,a)$ | *Action value* | $\mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$ | How good is this **move** from this situation, if I then play like this? |
| $A^\pi(s,a)$ | *Advantage* | $Q^\pi(s,a) - V^\pi(s)$ | How much **better than usual** is this move here? |

**The advantage is the one that matters.** Read the three definitions again and notice what subtraction does. The value $V(s)$ contains the difficulty of the situation: if a question is hard, every move from it scores badly. The advantage removes that common part and leaves only *this move compared to my usual behaviour here*. That is precisely the quantity you want to increase or decrease. When later chapters say "the advantage is the weight on the gradient", this is the quantity they mean.

**A worked illustration.** *(arithmetic on stated numbers, not a measurement)*

> In a two-option situation, the policy currently gets $Q(s, a_1) = 0.2$ and $Q(s, a_2) = 0.4$ on average, with $V(s) = 0.35$. Then $A(s,a_1) = -0.15$ and $A(s,a_2) = +0.05$: $a_1$ is worse than this policy's own average, $a_2$ marginally better.
> The **values themselves** would have told you "this situation is bad". Only the advantages tell you *which way to move*.

**Two ways to obtain them, and what each costs.** *(textbook fact, plus my judgement on the LLM consequences)*

| Route | How you get $V$ or $Q$ | What it costs |
|---|---|---|
| Model-based | You know the environment's transition probabilities and compute the expectation exactly | Requires a known model — rarely available; in a real sandbox with real tools, never |
| Learned | Train a second network (the **critic**) to predict it from experience | Memory for a second model, a second training loop, and a new source of noise: when the critic is wrong, the policy is steered wrong |

**Why this is genuinely hard for a language model.** A value for a *situation* in a game is a useful summary, because many games return to similar situations. A conversation, by contrast, is almost never revisited: every context is unique, and the value must be estimated at every token position of a half-written answer. The critic is therefore being asked to judge incomplete work, very far from any reward. *(my judgement, and the boundary: this is the main reason the field went looking for critic-free methods such as GRPO (group relative policy optimisation) in [Chapter 4](../04-alignment-algorithms/index.md))*

### Lesson 2 · The Bellman equations: why a return can be written recursively

**The identity, in plain words.** The return you will get from here is *the reward you just received*, plus *the return you will get from the next state*, faded by the discount. Written for the value function: *(textbook fact)*

$$
V^\pi(s) = \mathbb{E}_\pi\left[ r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

**Why this is legitimate, in three lines.** The step that looks like sleight of hand is just the definition of the return, split at the first term:

$$
G_t = r_{t+1} + \gamma\left( r_{t+2} + \gamma r_{t+3} + \cdots \right) = r_{t+1} + \gamma G_{t+1}
$$

The bracket is the definition of $G_{t+1}$, unchanged. Taking the expectation of both sides gives the Bellman equation. Two assumptions do the remaining work: the expectation is linear (so you may split it), and the **Markov property** holds (so the tail's expected value depends only on the next state, not on the whole history). Remove the Markov property and the identity silently becomes false — which is why [Chapter 2](../02-mdp/index.md) insisted the state be "enough".

**The optimality version.** For the best possible policy: *(textbook fact)*

$$
V^*(s) = \max_a \mathbb{E}\left[ r_{t+1} + \gamma V^*(s_{t+1}) \right]
$$

The difference is one word: **max**. Instead of averaging over the current policy's moves, you pick the best one. This single change is what makes the equation non-linear, and it is the reason a family of algorithms (value iteration, Q-learning) exists rather than one formula.

**What you pay for it.** *(my judgement, with the LLM consequences made explicit)*

- **You need the environment's transitions** to take that expectation. Without the model, the equation is a description of the answer, not a method for finding it.
- **The max over actions is expensive** when the action space is a vocabulary of tens of thousands of tokens: taking the best token at every one of thousands of positions is itself a search, and the optimal action at position 500 may only pay off at position 900.
- The payoff for paying this price is that the recursion makes the value function *learnable*: one equation replaces an infinite sum over all futures. Value iteration converges to the unique fixed point (a contraction, in the textbook treatment).

### Lesson 3 · Three ways to solve an RL problem: dynamic programming → Monte Carlo → temporal difference (TD)

**Three answers to the same question: where do I get numbers for the value?** *(textbook fact)*

| Method | How it estimates $V(s)$ | Bias | Variance | Needs the episode to finish? | What it costs |
|---|---|---|---|---|---|
| Dynamic programming | Sweep the Bellman equation over the whole state space | None (exact) | None | No | Requires the model **and** a state space small enough to enumerate — hopeless for LLMs directly |
| Monte Carlo | Sample whole episodes and average the returns observed from $s$ | None (unbiased) | **High** — the whole run's luck is in the estimate | **Yes** | Waiting for terminal rewards; enormous spread between samples |
| Temporal difference (TD) | Bootstrap: use the reward you got plus your *current estimate* of the next state's value | Bias from an imperfect estimate | Much lower | No, update as you go | Bias does not vanish just because you sample more: it is inherited from the critic |

**The one-sentence version of the trade.** Monte Carlo is honest and noisy; TD is quiet and can be systematically wrong. Everything between them is a dial.

**The dial is $\lambda$.** Mix a little of both: $\lambda = 0$ is pure one-step TD, $\lambda = 1$ is pure Monte Carlo. Values around $0.95$ are the usual compromise, and Lesson 5 shows exactly what this dial buys.

**Why this matters for LLMs, concretely.** *(my judgement)*

- A Monte Carlo estimate needs the *end* of the episode. For a five-turn agent task that is minutes of wall-clock; for a long training run you are paying for idle GPUs while you wait to compute a gradient.
- TD needs a critic that can judge a *partial* answer. That is a hard prediction problem, and a bad critic poisons training in a way that looks like an algorithm bug.
- This tension — "wait for the honest signal" versus "trust an unreliable simulator of the future" — is the reason this book spends [Chapter 6](../06-credit-assignment/index.md) on credit assignment rather than treating it as an implementation detail.

### Lesson 4 · Value methods vs policy methods vs actor-critic

**Three families, sorted by what they store.** *(textbook fact)*

| Family | What is learned | How an action is chosen | The catch |
|---|---|---|---|
| **Value-based** (Q-learning, DQN) | $Q(s,a)$ | Take the action with the highest value | Needs the max over actions, and chooses deterministically — awkward when the best behaviour is genuinely random |
| **Policy-based** (REINFORCE) | $\pi_\theta(a \mid s)$ directly | Sample from the policy | Higher variance; a single bad update can ruin the policy, and there is no notion of "how far is safe" |
| **Actor-critic** (PPO, GRPO) | Both: a policy (actor) plus a value estimate (critic) | Sample from the actor; use the critic for the baseline | Two things to train, and the critic's errors leak into the actor's updates |

**The action space decides the family.** Value-based methods need "the best action", which means a max. For a language model the action space is the whole vocabulary at every position: the max is affordable at one position, but the value of a token depends on tokens chosen far later, so the greedy choice is short-sighted. Policy-based and actor-critic methods sidestep this by never taking a max: they only need $\nabla \log \pi_\theta$, which the model's own softmax gives you for free. This is why LLM post-training is dominated by actor-critic and direct-preference methods, not by Q-learning. *(my judgement)*

**On-policy versus off-policy: the dividing line that decides your electricity bill.** *(textbook fact)*

- **On-policy**: the data must come from the policy being trained. Improve the policy, and your data is stale — so you must generate fresh samples after each update. PPO and GRPO are on-policy, which is exactly why their rollout cost dominates ([Chapter 1](../01-landscape/index.md), Lesson 4).
- **Off-policy**: the data may come from somewhere else — an older version of the policy, a different policy, a human, a fixed dataset. Q-learning and DQN use a replay buffer for this reason, and it is what makes them sample-efficient.

**Where DPO (direct preference optimisation) sits.** DPO is trained on a *fixed set of preference pairs* collected from some other policy. That is off-policy by construction, and it is the source of both its cheapness and its ceiling: no sampling infrastructure, but also no exploration. *(my judgement, with the derivation deferred to [Chapter 4](../04-alignment-algorithms/index.md))*

### Lesson 5 · Policy-gradient theorem → REINFORCE → variance reduction → GAE

**The theorem, and the one sentence that makes it click.** *(textbook fact; the original policy-gradient theorem is Sutton et al., under Reading)*

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a \mid s)\, Q^\pi(s,a) \right]
$$

Read that as an instruction, not as mathematics:

> **You never differentiate the reward.** You differentiate the *log-probability of what you did*, and multiply by how good it turned out to be.

That is the whole idea. The reward acts as a weight on the gradient of a log-probability, so "make this more likely" and "make this less likely" are the only two operations the optimiser has.

**Where the log comes from (the log-derivative trick).** Start from $J(\theta) = \sum_s d(s)\sum_a \pi_\theta(a \mid s) Q^\pi(s,a)$ and differentiate. The product rule hits $\pi_\theta$, and the identity $\nabla \pi_\theta = \pi_\theta \nabla \log \pi_\theta$ turns that derivative back into an expectation, which you can estimate by sampling. That substitution is the entire reason policy gradients are practical: it moves the derivative off an unknown function and onto something you can compute.

**REINFORCE: use the return as the weight.** The name is historical — it comes from Williams's 1992 paper (see Reading); what it means here is simple: replace the unknown $Q^\pi(s,a)$ with the sampled return $G_t$:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t
$$

It is unbiased, it needs no critic, and the variance is brutal — a single sample of $G_t$ carries the luck of the entire episode. *(textbook fact)*

**Variance reduction, three steps.** Each step is cheap and each one removes a specific source of noise. *(textbook fact)*

1. **Subtract a baseline.** Replace $G_t$ with $G_t - b(s)$. Any $b$ that does not depend on the action leaves the expectation unchanged, because $\sum_a \nabla \pi_\theta(a \mid s) = \nabla 1 = 0$. Using $V(s)$ as the baseline turns the weight into the advantage $A = Q - V$, which is what Lesson 1 was building toward.
2. **Respect causality.** A reward received before step $t$ cannot have been caused by the action at step $t$. Replace the full return with the reward-to-go $\sum_{t' \ge t} r_{t'}$. It removes the noise contributed by the past, for free.
3. **Discount and blend.** Discount the reward-to-go with $\gamma$, then blend the estimate across horizons with $\lambda$ — which is GAE.

**GAE in four lines.** Define the **TD error** first: the surprise at step $t$, meaning "what I got plus what I now think the next state is worth" minus "what I expected here".

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Then the advantage estimate is a geometrically decayed sum of those surprises:

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^{l}\, \delta_{t+l}
$$

- $\gamma$ — how much the future counts.
- $\lambda$ — how far into the future you trust the critic's estimates rather than the actual rewards.
- $V$ — the critic, the second network from Lesson 1.

**What the dial does.** *(textbook fact for the extremes; my judgement for the practice)*

| $\lambda$ | The estimate becomes | Bias | Variance | When it is the right choice |
|---|---|---|---|---|
| $0$ | One-step TD: trust the critic almost entirely | High — the critic's error is inherited | Low | The critic is already accurate; you want cheap updates |
| $1$ | Monte Carlo: trust only real rewards | None | High | The critic is untrustworthy and episodes are short |
| $\approx 0.95$ | The usual compromise | Some | Manageable | The default for a reason — it is what the implementations ship |

**Where the honest accounting happens.** The advantage you compute is only as good as $V$. If the critic is wrong, GAE propagates that error into every token's weight, and the training curve looks like an algorithm failure. Two reported findings keep this in perspective:

- **Implementation details, not the algorithm, dominate results.** In a careful comparison of PPO and TRPO, code-level choices (advantage normalisation, value clipping, reward scaling) explained most of the performance difference between published results. *(paper result, reported and not reproduced — Implementation Matters, under Reading)*
- **Once you use adaptive optimisers and deep networks, the update you actually apply can have little to do with the policy gradient the theory describes.** *(paper result, reported and not reproduced — A Closer Look at Deep Policy Gradients, under Reading)*

The practical reading, and the seed for the next chapter's backfill question: **this layer is where bias and variance enter your training run, and where a system can quietly mislead you.** A loss curve that goes down proves that the numbers the optimiser saw went down — not that the model got better.

### Lesson 6 · A bridge: from online RL to offline preference optimisation

**The loop's bill, restated.** Everything above assumes you can sample from the policy and score it. The loop is: sample → score → update → resample, and the resampling is the expensive part. Every alternative in [Chapter 4](../04-alignment-algorithms/index.md) is an attempt to buy the same improvement without repeating that loop.

**The offline idea, in one paragraph.** Suppose you are given pairs of answers, one preferred over the other, produced by an earlier policy. Can you improve the current policy directly from those pairs — no reward model, no sampling? *(paper result, reported and not reproduced; the derivation is Chapter 4's job)* The answer turns out to be yes: for the policy that maximises the preference signal while staying as close as possible to the reference model, there is a closed-form relation between the policy and the reward it implies. Substitute that relation back into the objective and the reward model disappears, leaving a loss over the pairs themselves. That is DPO in one sentence.

**The two purchases, side by side.** *(my judgement)*

| | Online RL (PPO, GRPO) | Offline preference (DPO) |
|---|---|---|
| Data | Generated fresh by the current policy | A fixed file of pairs |
| Can it discover behaviour nobody demonstrated? | Yes — that is the point of sampling | No — bounded by what the pairs contain |
| Cost | Rollouts dominate; the loop never stops | One pass over a static dataset |
| Main risk | Instability, and the bill | Convincing yourself the data is better than it is |

**Why every method keeps a reference model.** The reference model (a frozen copy) appears in all of these methods as a leash: a penalty proportional to how far the trained policy has drifted from it. Its purpose is not nostalgia. Reward signals are imperfect, and a policy that optimises an imperfect signal hard enough will find the flaw — so the leash trades a little measured reward for staying in the region where the signal still means what you thought. *(the mechanism is textbook; the failure mode it prevents is developed in [Chapter 5](../05-reward-engineering/index.md))*

**Which to reach for as a beginner.** *(my judgement)* Start offline: preference pairs are cheap, the training loop is one pass, and failures are easier to diagnose. Move to online RL when you can name the behaviour you want more of and it already occurs sometimes — that is the pass@k test from [Chapter 1](../01-landscape/index.md), Lesson 3, and it is also the point where the sampling bill becomes worth paying.

## Deliverable

- A hand derivation of the policy gradient and GAE, including *why* a baseline is needed at each step.
- A short note: why an LLM's "action space" is vocabulary size × sequence length, and what that implies (hint: the max is affordable, the greedy choice is short-sighted, and only $\nabla \log \pi_\theta$ is free).

## How you know you passed

- Your derivation runs from the Bellman equation to the GAE $\lambda$-return, and marks at each variance-reduction step which source of noise that step removes.
- For your own task, you can say whether the advantage estimate is limited mainly by the critic's error or by sampling noise — and what you would change first.
- Your sparsity note names a number: roughly how many tokens separate the first decision from the first reward in your task.

## Reading

Titles and arXiv IDs were checked against the arXiv API on 2026-09-30 (IDs marked "verified"). References with no arXiv ID are named as such. Numbers quoted from papers are **reported, not reproduced**.

- **The standard treatment, chapters 3 and 13** — Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition (free official PDF) — no arXiv ID. Bellman equations, dynamic programming, Monte Carlo, TD and the policy-gradient theorem as used above.
- **The origin of dynamic programming** — Bellman, *Dynamic Programming*, Princeton University Press (1957) — no arXiv ID; the recursion in Lesson 2.
- **The original policy-gradient result** — Sutton, McAllester, Singh and Mansour, *Policy Gradient Methods for Reinforcement Learning with Function Approximation*, NeurIPS (2000) — no arXiv ID; the theorem in Lesson 5.
- **REINFORCE** — Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, Machine Learning 8 (1992) — no arXiv ID; the estimator in Lesson 5.
- **Generalised advantage estimation** — Schulman, Moritz, Levine, Jordan and Abbeel, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438 — verified.
- **Trust regions, the idea PPO simplifies** — Schulman, Levine, Abbeel, Jordan and Moritz, *Trust Region Policy Optimization*, arXiv:1502.05477 — verified.
- **PPO itself** — Schulman, Wolski, Dhariwal, Radford and Klimov, *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 — verified. Used in Chapter 4.
- **Parallel sampling as the standard architecture** — Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, arXiv:1602.01783 — verified. Why rollouts, not gradients, dominate the wall-clock.
- **The implementation-details result** — Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph and Madry, *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*, arXiv:2005.12729 — verified.
- **The other side of the same argument** — Ilyas et al., *A Closer Look at Deep Policy Gradients*, arXiv:1811.02553 — verified.
- **A large-scale ablation of what actually matters** — Andrychowicz et al., *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study*, arXiv:2006.05990 — verified. Useful before you spend a week tuning something this study already measured.

## Backfill

- [x] Question 4: Where does my training system quietly lie to me? — this chapter locates the origin: the advantage is estimated (bias from the critic, Lesson 5), the gradient is estimated (variance from sampling, Lesson 5), and past that point the optimiser's direction stops matching the theory (the two reported results at the end of Lesson 5).

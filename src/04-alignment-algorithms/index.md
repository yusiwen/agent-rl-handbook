# Chapter 4 · The Alignment Algorithm Family (and PPO in Full)

> **Compute tier**: ✅ | **Status**: draft — all ten lessons have prose; no number in this chapter was measured on hardware
> **Rule of this book**: the *meaning* of every algorithm is explained here, once. [Chapter 9](../09-training-runtime/index.md) only covers how a runtime switches between and extends them — it does not repeat the objectives.

## First, the gist

[Chapter 3](../03-policy-gradient/index.md) showed where a gradient comes from. This chapter is about what you point it at.

Five names carry the field, and the fastest way to hold them is to ask one question of each: **where does the score come from?**

- **RLHF (reinforcement learning from human feedback)**: people compare two answers; a **reward model** (a second model, trained to predict those comparisons) becomes the score; **PPO (proximal policy optimisation)** trains the policy against it. The general-purpose option, and the expensive one.
- **RLAIF (reinforcement learning from AI feedback)**: the same loop with a language model doing the comparing instead of a person.
- **RLVR (reinforcement learning with verifiable rewards)**: not a model at all — a program checks the answer. Does the test pass? Does the proof check?
- **GRPO (group relative policy optimisation)**: not a new source of feedback, but a cheaper way to compute the advantage when you can afford several samples per question.
- **DPO (direct preference optimisation)**: no reward model and no sampling loop; the policy is trained on a fixed file of preference pairs, and the reward is implied by the policy itself.

Choosing among them comes down to four things, and [Lesson 10](#lesson-10--choosing) turns them into a table: where the feedback comes from, how fine-grained the reward is, whether you can afford fresh samples, and what your machine costs.

## What you'll be able to do

- Say what each acronym stands for, and what it is *for*, in one sentence each.
- Write down what PPO, GRPO and DPO actually optimise, and how much sampling each needs.
- Explain in plain words why "DPO needs no reward model" is true — and what you pay for that.
- Give a first answer to "which one fits my task", with reasons for *not* choosing the others.

## Lessons

- [ ] 1. Why alignment is not solved by making the model bigger: HHH, the limits of SFT (supervised fine-tuning), and where RLHF begins
- [ ] 2. **The full PPO objective**: clipping, advantage estimates, the KL (Kullback-Leibler) penalty — and why KL is the safety valve
- [ ] 3. Four models in one training run: policy, reward model, critic and reference
- [ ] 4. GRPO: sample a *group* of answers, compare them with each other, drop the critic
- [ ] 5. DPO, fully unpacked: from Bradley-Terry to the closed form — it is **offline preference optimisation** with an implicit reward
- [ ] 6. The DPO family (KTO, ORPO, SimPO and friends): what the "theoretical patches" are patching
- [ ] 7. RLAIF and scalable oversight: Constitutional AI, LLM-as-judge, self-rewarding
- [ ] 8. RLVR: reinforcement learning with verifiable rewards (the name describes the *source* of the reward, not a new optimiser)
- [ ] 9. Five alignment traps: reward hacking, distribution shift, over-optimisation and friends
- [ ] 10. Choosing: feedback source × reward granularity × online sampling × system cost

### Lesson 1 · Why alignment is not solved by making the model bigger

**The intuition to kill first.** A bigger model is better at everything a benchmark measures, so it is tempting to think alignment comes free with scale. It does not, because "aligned" is not a capability. It is a statement about *what the model is trying to do* — and a more capable model is also better at finding a creative way to satisfy whatever objective you gave it. Scale raises the ceiling on both the behaviour you want and the behaviour you were trying to rule out. *(my judgement, and the boundary: scale clearly helps with some failures — a larger model follows instructions more reliably — but it does not make the objective any less of a proxy)*

**What supervised fine-tuning can and cannot express.** *(textbook fact)* SFT (supervised fine-tuning) trains on examples and minimises the error between the model's output and the demonstrated one. Three ceilings follow from that:

1. **A demonstration says "acceptable", not "better than".** When two answers are both fine but one is preferred, showing only the preferred one teaches the model a target it has no way to rank. Preference is a *comparison*, and a comparison is not a label.
2. **The ceiling is the demonstrator.** Imitating human-written answers plateaus at their quality, and the plateau shows up early: a small, excellent demonstration set can match much larger ones *(paper result, reported and not reproduced — LIMA, in Reading)*, and imitation alone does not buy capability the demonstrations lack *(paper result, reported and not reproduced — The False Promise of Imitating Proprietary LLMs)*.
3. **Errors compound over a multi-turn task.** Training on complete transcripts means the model never practises recovering from its own bad intermediate step, because a demonstration rarely contains one. The moment its own output goes off-script, it is in a situation it has never seen. *(textbook fact)*

**HHH, and why it forces a trade-off.** Aligned is usually unpacked as **HHH — helpful, honest and harmless** *(textbook fact; the framing comes from Anthropic's assistant work, both papers in Reading)*. The three goals pull against each other. A helpful answer to "how do I get into my own locked account" is not harmless; a maximally honest answer about a sensitive topic can be useless. Wanting all three is not a specification — it is a negotiation, and a negotiation needs a number that says *how much* of each you are willing to trade. That number is what a reward is, and it is why alignment ends up as an optimisation problem rather than a data problem. *(my judgement)*

**The pivot in one line.** SFT says *produce this*. Reinforcement learning says *make the answers that scored well more likely, and the ones that scored badly less likely*. That needs two things SFT never needed: a way to **sample** the model's own attempts, and a way to **score** them. Every method in this chapter is a different answer to "scored by what?". *(textbook fact)*

**Where RLHF begins.** The standard pipeline, named and demonstrated at scale by InstructGPT *(paper result, reported and not reproduced — in Reading)*:

| Step | What happens | What it costs |
|---|---|---|
| 1. Collect comparisons | People are shown two answers to the same prompt and pick the better one | People, and a lot of prompts |
| 2. Fit a reward model | A model learns to predict those preferences as a single number | A second model to train, and its own bugs |
| 3. Run RL against it | PPO updates the policy to raise the score, held back by a KL penalty (Lesson 2) | Rollouts, memory, and an unstable loop |

**What you pay for all this, stated plainly.** *(my judgement)* You have added a data pipeline, a second model, an RL loop and a new failure mode: from step 3 onward the model is optimising a **proxy** for what people wanted, not what people wanted. The reward model is a summary of a finite set of comparisons, and a good optimiser will find the places where the summary is wrong. That single sentence is the reason Lessons 5 to 9 exist.

**When not to reach for RL.** If you have abundant demonstrations of the behaviour you want, and the behaviour is expressible as "produce this", more SFT data is cheaper and far more predictable. RL earns its cost when you can *name* the behaviour you want more of, when the demonstrations cannot express it, and when something can score it. *(my judgement — this is [Question 1](index.md) of this book's four, and Lesson 10 is where it gets settled)*

### Lesson 2 · The full PPO objective

**The problem PPO solves.** The policy gradient from [Chapter 3](../03-policy-gradient/index.md), Lesson 5 assumes small steps. Its guarantee is local: the gradient tells you the direction at the *current* policy, and one large update invalidates it — the batches you already collected were produced by a policy that no longer exists. Trust-region methods made "small step" a formal constraint on how far the policy may move, measured by the KL (Kullback-Leibler) divergence between the old and new policies. The original version needed second-order machinery to enforce that constraint, which is expensive at language-model scale. *(textbook fact — Trust Region Policy Optimization, in Reading)*

**PPO's answer: a clipped surrogate.** PPO (proximal policy optimisation) keeps the intuition and throws away the machinery. Instead of constraining the step, it *clips* the objective so that moving too far simply stops paying. *(textbook fact)*

The quantity being clipped is the **probability ratio**:

<div class="math-display">
$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$
</div>

Read it as a comparison in time: *how much more likely is this action now, than when we sampled it?* A ratio of 1 means nothing has changed; 1.2 means the action is 20% more likely than at sampling time. That is also what lets you show PPO the same batch for several epochs — the ratio corrects for the policy having moved since the data was collected — and why a batch does not stay valid forever: as the policy drifts, the correction is built on a comparison that no longer means anything.

The objective, with the clipping in place:

<div class="math-display">
$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta)\,\hat{A}_t,\ \mathrm{clip}\left( r_t(\theta), 1-\epsilon, 1+\epsilon \right)\hat{A}_t \right) \right]
$$
</div>

- $\hat{A}_t$ is the advantage from [Chapter 3](../03-policy-gradient/index.md) — how much better than usual this action turned out to be. It is the weight on everything else here.
- $\epsilon$ is the width of the allowed band, typically $0.2$ in published implementations. It is a *default*, not a law of nature: smaller is more conservative, larger is more aggressive, and the whole band is a heuristic.
- The $\min$ over two branches, one unclipped and one clipped, is the entire trick.

**Why the $\min$ makes it pessimistic.** Take the two cases separately. *(textbook fact)*

- **A good action, $\hat{A}_t > 0$.** The unclipped branch grows without limit as you make the action more likely. The clipped branch stops growing once the ratio reaches $1+\epsilon$. The $\min$ keeps the smaller one, so beyond the band the objective is flat and the gradient through this term vanishes — there is nothing more to gain by moving further.
- **A bad action, $\hat{A}_t < 0$.** Here the objective improves by making the action *less* likely, so the ratio falls. The clipped branch stops the gain once the ratio drops below $1-\epsilon$, and again the $\min$ chooses the flat one.

<figure class="book-figure">{{#include ../figures/04-clip-band.svg}}</figure>

*Figure 4.1 — the objective term for one action, as a function of how far the policy has moved. Inside the band the two branches agree; outside it the clipped branch goes flat, and the flat part is what stops the policy running away after a single batch.*

**What the flat regions mean in practice.** They are a brake on the *incentive*, not on the parameters. Nothing prevents the policy from moving far; it just stops being rewarded for it on the data it has seen. And a subtlety worth knowing before you debug it: the flat region is a property of the surrogate, not of the true objective. Optimising a clipped surrogate is guaranteed to be close to optimising the real thing only while the clip is active on a small fraction of the batch. When most of the batch is clipped, the gradient you are following has almost nothing to do with the policy gradient from Chapter 3. *(my judgement; the reported comparison of PPO and its predecessor found that code-level choices of this kind explained most of the difference between published results — Implementation Matters, in Reading)*

**The KL (Kullback-Leibler) penalty, and why it is the safety valve.** The clipped ratio limits movement per update, but nothing in it says where the policy should *end up*. Left alone, an RL run will happily wander into whatever region maximises the proxy reward. So a second term is added: a penalty proportional to how far the policy has drifted from the **reference model** — a frozen copy of the model you started from.

<div class="math-display">
$$
-\,\beta\ \mathrm{KL}\!\left( \pi_\theta \| \pi_{\mathrm{ref}} \right)
$$
</div>

- $\beta$ sets the price of drift. Small $\beta$ trusts the reward more; large $\beta$ keeps the policy closer to the starting point.
- The mechanism is a leash, and its purpose is not nostalgia for the starting point. The reward model is only a trustworthy guide *near the data it was trained on*. Far from the reference model, its scores are extrapolation, and a strong optimiser will find the extrapolations where the score is high and the quality is not — which is reward hacking (Lesson 9).
- Two implementations exist and they behave differently. In one, the KL is folded into the reward as a per-token penalty; in the other it appears as a separate term in the loss, often with a non-negative estimator. In practice the observed KL is tracked during training and $\beta$ is adjusted toward a target value — a **KL controller**, so the coefficient is a control loop, not a constant. *(textbook fact for the term; the wiring details are the reported subject of Secrets of RLHF, in Reading)*

**The rest of the practical objective, which the formula hides.** *(my judgement, with the reported finding attached)* The advantage $\hat{A}_t$ is an estimate: the generalised advantage estimate (GAE) from Chapter 3, or the group-relative version in Lesson 4. Implementations normalise the advantages within a batch, clip the value loss, and stop gradients in places the formula does not mention. These are not cosmetics; a careful study of the same algorithm across implementations found that exactly these details accounted for most of the difference in results *(paper result, reported and not reproduced — Implementation Matters in Deep Policy Gradients, in Reading)*. Two consequences for you: when a run misbehaves, read the config before theorising about the algorithm, and when you compare two published numbers, assume the difference may be the code rather than the idea.

**What it costs.** *(my judgement)* Four models in memory (next lesson), rollouts that dominate the wall-clock, a hyperparameter set ($\epsilon$, $\beta$, the target KL, epochs per batch, advantage normalisation) that interacts, and a surrogate objective the optimiser can satisfy without the policy actually improving. PPO is the general-purpose option, and it is general-purpose because you pay for everything.

### Lesson 3 · Four models in one training run

**What is actually resident in memory.** *(textbook fact)* PPO with a learned reward model needs four sets of weights, and the surprise for a beginner is that they are not all the same kind of thing:

| Model | Its job | Trained? | What breaks if it is wrong |
|---|---|---|---|
| **Policy** (the actor) | The model you are shipping; it produces the answers | Yes | — it is the thing being fixed |
| **Reward model** | Turns an answer into the score the policy is chasing | No, frozen during the RL run | The run optimises the wrong thing, invisibly. The reward curve looks healthy, and only a held-out human check shows the damage |
| **Critic** (the value model) | Estimates the baseline for the advantage | Yes | Noisy advantages: it replaces sampling noise with its own error, and a bad critic looks like an unstable algorithm |
| **Reference model** | The frozen starting point for the KL leash | No, frozen | No leash. The policy drifts into the region where the reward model's scores are extrapolation |

**Why there are four, in one paragraph.** The policy needs to move, so it needs gradients. The reward model must not move, or the policy would be chasing a target that shifts as it chases — it is frozen precisely so that the gradient points somewhere consistent. The critic exists only to reduce the variance of the advantage. The reference exists only to define "how far is too far". Two of the four are trained, two are anchors, and each anchor is a cost you pay for a failure mode you are avoiding. *(textbook fact; the LLM-specific reason the critic is such a weak point is [Chapter 3](../03-policy-gradient/index.md), Lesson 1)*

**The memory arithmetic, with the assumptions stated.** *(arithmetic on stated assumptions, not a measurement)* Assume full fine-tuning of a 7-billion-parameter model, mixed precision, AdamW:

- A **frozen** model costs about 2 bytes per parameter at half precision: $7 \times 10^9 \times 2 \approx 14$ GB.
- A **trained** model costs far more: 2 bytes of weights plus a 4-byte master copy plus two 8-byte optimiser moments, so about 14 bytes per parameter: $7 \times 10^9 \times 14 \approx 98$ GB.

Now add them up for a 7B policy: policy 98 GB, reward model 14 GB, reference 14 GB, and a critic that is also a 7B model at 98 GB. That is over 200 GB of state before a single activation is stored. On a machine with 128 GB of unified memory, or on one 80 GB accelerator, the conclusion is not "tune it harder" — it is that this configuration does not fit, and the honest options are to shrink it deliberately:

| Route | What changes | What it costs |
|---|---|---|
| **LoRA/QLoRA on the policy** | Only the adapters are trained, so optimiser state is proportional to adapter size, not model size: a 7B policy with 0.2% trainable parameters costs about 0.2 GB of optimiser state instead of 98 GB | The policy can only move within the adapters' capacity; long RL runs may need the adapter rank raised |
| **A smaller policy** | Everything scales together, so a 1.5B model multiplies every number above by about five | Less capability, and slower emergence of anything interesting |
| **Drop the critic** | GRPO (Lesson 4) removes one trained model from the run | Loses the learned baseline; needs several samples per prompt |
| **Drop the reward model** | DPO (Lesson 5) removes it entirely, and the critic too | Loses online exploration and the ability to re-query the score |

**The engineering truth behind the table.** *(my judgement)* The four-model layout is why RL training frameworks exist. Moving a policy, a critic, a reward model and a reference between different devices, keeping the reference frozen while its sibling is updated, and synchronising weights with a sampling fleet is most of what a runtime like the one in [Chapter 9](../09-training-runtime/index.md) does — and it is why the same algorithm behaves differently in different frameworks. The algorithm is a page of maths; the system is a distributed program.

**One more thing worth noticing.** The reference model is the same weights as the starting policy. Implementations exploit that — loading it once, offloading it, or reconstructing it — which is why "we keep a frozen copy" is cheap in some frameworks and expensive in others. *(my judgement; the trade-offs are Chapter 9's subject)*

### Lesson 4 · GRPO

**The name, and where it comes from.** GRPO (group relative policy optimisation) was introduced with DeepSeekMath *(paper result — in Reading; the paper also introduced the name)*. The "group" is the whole idea: instead of training a critic to guess how good a situation is, sample **a group of answers to the same prompt**, score them all, and compare each one with the rest of its own group.

**The mechanics in plain words.** *(textbook fact)*

1. Take one prompt.
2. Sample $G$ answers from the current policy — $G$ is typically between 4 and 64, and it is the main new cost.
3. Score every answer with the reward (a verifier, in the setting this was designed for).
4. Use the **group's own average score** as the baseline, and the group's spread as the scale:

<div class="math-display">
$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}
$$
</div>

5. Update with the same clipped surrogate as PPO, with $\hat{A}_i$ in place of the learned advantage:

<div class="math-display">
$$
L_{\mathrm{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\left( \rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i \right) \;-\; \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}}) \right]
$$
</div>

where $\rho_{i,t}$ is the per-token probability ratio of answer $i$ against the policy that produced it, and $|o_i|$ is the answer's length. Note what is *not* here: no value model, no per-token advantage estimate, and no separate critic checkpoint.

**Why it works, and why it works especially well with verifiers.** *(my judgement, with the reported origin)* A baseline only has to be *approximately* right to cut variance, and the average of a handful of other attempts at the same prompt is a decent approximation of "how hard was this prompt, for this policy, today". The critic, by contrast, was being asked to predict the value of a half-written answer — the hard problem [Chapter 3](../03-policy-gradient/index.md), Lesson 1 described. Dropping it removes a trained model, its optimiser state, and a whole class of bugs. The price is $G$ times the rollouts, which is only affordable when scoring is cheap. That is exactly the case when a program does the scoring.

**The catch, in four parts.** *(paper results, reported and not reproduced; the boundary is my judgement)*

1. **A group with one score teaches nothing.** If all $G$ answers score identically, every advantage is zero and the rollout is wasted. At the start of training this is common — a fresh policy may fail a task every time. One published system adds *dynamic sampling*: keep sampling until a batch contains a mix of outcomes *(DAPO, in Reading)*.
2. **Dividing by the spread rescales the signal.** A question where one answer out of eight is correct gets a far larger advantage than a question where four are. That is a systematic bias toward questions of a particular difficulty, not a neutral normalisation *(the standard deviation term is analysed and removed in Understanding R1-Zero-Like Training, in Reading)*.
3. **Length leaks into the score.** Longer answers accumulate more per-token terms, and the choice of length normalisation changes whether the run is biased toward long answers or short ones. This is a property of the objective, not of the data *(same paper)*.
4. **The reference model has not gone anywhere.** GRPO keeps the KL leash, so GRPO is not the cheaper *memory* option in every respect — what it removes is the critic, not the second frozen policy.

**The family around it, one line each.** All of these are patches on the same objective, and all of them are cheap to try and expensive to compare fairly. *(paper results, reported and not reproduced — all four are in Reading)*

| Variant | The change | What it buys |
|---|---|---|
| **RLOO** (leave-one-out) | The baseline is the mean of the *other* samples, excluding the one being scored | A slightly less biased baseline; no spread normalisation |
| **DAPO** | Clip-higher, dynamic sampling, token-level loss, overlong-answer shaping | Keeps groups informative and stops overlong answers dominating |
| **GSPO** (group sequence policy optimisation) | The importance ratio is computed per *sequence* rather than per token | The clip acts on the answer as a whole, which suits long reasoning traces |
| **Dr. GRPO** | Removes the spread normalisation and the length normalisation | Removes two known biases; simpler and, in its own reported comparison, at least as good |

**What it costs.** *(my judgement)* Two things. First, sampling multiplied by $G$: GRPO is cheap in memory and expensive in compute. Second, and less obvious, your *data mixture* is now part of your optimizer: because the signal is the spread within a group, easy prompts (all correct) and impossible prompts (all wrong) contribute nothing, and the middle does all the work. Curating that middle is [Chapter 7](../07-data-and-tasks/index.md)'s job, and it matters more here than it does with a learned critic.

### Lesson 5 · DPO, fully unpacked

**The claim to be suspicious of.** DPO (direct preference optimisation) is usually introduced as "preference tuning without a reward model". The claim is true, it is not magic, and the derivation is short enough to hold in your head — which is why this lesson derives it rather than quoting it. *(textbook fact; the derivation is from the DPO paper, in Reading)*

**Step 0: the setup.** You are given a fixed dataset of comparisons: a prompt $x$, an answer $y_w$ a person preferred, and an answer $y_l$ they rejected. Everything is offline. No sampling, no environment, no reward model to train. The question is what objective turns these triples into an improved policy.

**Step 1: how preferences are modelled.** The standard model for pairwise comparison is the **Bradley-Terry model**: each answer has an unknown quality score, and the probability that one beats the other is a sigmoid of the difference.

<div class="math-display">
$$
P(y_w \text{ is preferred} \mid x) = \sigma\!\left( r(x, y_w) - r(x, y_l) \right)
$$
</div>

If that looks like the logistic regression you have met, it is: same shape, with "features" replaced by two answers scored by the same function $r$. *(textbook fact; the model is Bradley and Terry, 1952 — no arXiv ID)*

**Step 2: the KL-constrained optimum, in closed form.** Now write down the objective RLHF actually optimises: raise the reward, but stay close to the reference model.

<div class="math-display">
$$
\max_{\pi}\ \mathbb{E}_{y \sim \pi}\left[ r(x, y) \right] - \beta\, \mathrm{KL}\!\left( \pi(\cdot \mid x) \| \pi_{\mathrm{ref}}(\cdot \mid x) \right)
$$
</div>

This has a known optimum, and it is worth seeing why. The reward term wants to put all the probability on the highest-scoring answer; the KL term charges for every deviation from the reference. The compromise is to reweight the reference distribution exponentially by the reward:

<div class="math-display">
$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\, \exp\!\left( \frac{1}{\beta} r(x, y) \right)
$$
</div>

where $Z(x)$ is the normalising constant that makes it a distribution. Take logarithms and rearrange, and you get the score expressed in terms of the policy:

<div class="math-display">
$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)
$$
</div>

**Step 3: substitute, and watch the reward model disappear.** Put that expression into the Bradley-Terry probability. The term $\beta \log Z(x)$ is the same for $y_w$ and $y_l$ — it depends only on the prompt — so it cancels in the difference. What is left contains no $r$ at all:

<div class="math-display">
$$
L_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} \;-\; \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \right]
$$
</div>

That is the whole method. One forward pass per answer, one loss, no sampling loop. *(textbook fact)*

**The hidden reward, and the name.** Read the loss again and notice the quantity it is made of:

<div class="math-display">
$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$
</div>

This is an **implicit reward** — the policy *is* the reward model, which is what the paper's subtitle means when it says the language model is secretly a reward model. The loss then reads: raise the implicit reward of the chosen answer, lower it for the rejected one, and let the sigmoid stop pushing once the two are comfortably apart. The word *direct* in DPO means exactly this: no intermediate reward model to fit and then sample against. *(textbook fact for the algebra; the reading is mine)*

**What you pay — five items, and they are the reason this is not the end of the story.**

1. **It is offline.** Nothing in the loss can discover behaviour absent from the pairs. RL can, because it samples; DPO cannot, because the file will not grow on its own. *(textbook fact)*
2. **The pairs came from a different policy.** The preference data was collected from some earlier or other model, so the loss is a meaningful improvement only near the reference. Push hard enough and the policy leaves the region where the pairs say anything at all. *(textbook fact, with the practical consequence mine)*
3. **A reference model is still required.** "No reward model" is true; "no second model" is not. You keep a frozen copy, and you need one, because the implicit reward is a ratio against it.
4. **$\beta$ is doing more work than it looks.** It sets how much the policy's own log-ratio is trusted as a reward. Too large and the update barely moves; too small and noisy pairs — a single inconsistent label — produce large, confident updates. *(my judgement; treat the default as a starting point and sweep it)*
5. **"No reward model" does not mean "no judge".** The pairs were still produced by people, by a language model, or by some mixture. Whatever biases that process had are now frozen into a file you cannot re-query. The comparison was moved upstream, not eliminated. *(my judgement)*

**One honest caveat on the derivation.** The equivalence is exact *given* the assumptions: preferences follow Bradley-Terry, and the objective you actually want is the KL-constrained one. Real preference data has ties, position effects, length artefacts and annotators who disagree. When the assumptions bend, DPO is still a sensible loss — it still raises the chosen answer's relative likelihood — but the clean story "it is the same as RLHF, for free" is a statement about the idealised case. *(paper result in the ideal case; the caveat is my judgement)*

**Where it sits as a first step.** *(my judgement)* For a beginner with a GPU and a preference file, DPO is the cheapest route from "the model writes" to "the model writes what people prefer", and its failures are diagnosable: a margin that saturates, a $\beta$ that is wrong, or data that says something other than what you thought. Start here, and move to online methods when you can name the behaviour you want more of and the pairs have stopped containing it.

### Lesson 6 · The DPO family

**What the patches are patching.** DPO makes three demands: preference *pairs*, a *reference model*, and *offline* data. The variants that followed each soften one of those demands, and reading their titles with that in mind is much easier than parsing the acronyms. *(my judgement on the framing; each variant's mechanism is a paper result, reported and not reproduced — all four are in Reading)*

| Method | The demand it removes | What it adds instead | What that costs |
|---|---|---|---|
| **KTO** (Kahneman-Tversky optimisation) | Pairs become single examples, each marked desirable or undesirable | A prospect-theory style loss, with a loss-aversion term so failures weigh more than successes | You must still decide the desirable/undesirable split, and single examples carry less information than a comparison |
| **ORPO** (odds-ratio preference optimisation) | The reference model — and the separate SFT stage | A penalty on the odds ratio folded into supervised fine-tuning, so one stage does both | SFT and preference now share one gradient budget; the mixing weight is the new tuning problem |
| **SimPO** (simple preference optimisation) | The reference model | A length-normalised implicit reward, plus a target margin between chosen and rejected | Length normalisation changes what "better" means, and the margin is a new hyperparameter |
| **The online variants** (RLOO, DAPO, GSPO, Dr. GRPO — Lesson 4) | The critic | A group or leave-one-out baseline | $G$ times the sampling, and a data mixture that decides the signal |

**The name, because a reader cannot search for a name they cannot decode.** *(textbook fact)* KTO is named for the two behavioural economists whose prospect theory supplies the loss's shape — Kahneman and Tversky. ORPO's "odds ratio" is literally the ratio of odds it penalises. SimPO's "simple" is a claim about needing one model instead of two. Dr. GRPO is a joke about the "GRPO doctor" arriving to fix the objective's biases, and it is worth remembering that its authors are the same ones who found the biases.

**The three genuine axes behind the noise.** *(my judgement)* Stripped of titles, the variants differ along three axes, and nothing else:

1. **What you have to collect** — comparisons, or single labelled examples, or nothing beyond a prompt.
2. **How many models are resident** — two (policy and reference), or one.
3. **Whether the data is fresh** — a fixed file, or samples from the policy you are training.

Four rows of a table, and the axis that decides your electricity bill is the third one.

**The honest caution.** *(my judgement)* A variant that wins on one benchmark can lose on another because the *data* changed, not the idea. Comparing them fairly requires the same base model, the same preference data and the same evaluation — which is precisely what published comparisons rarely hold fixed, and why the literature looks noisier than its tables. Treat a variant's reported win as a hypothesis about your setting. [Chapter 13](../13-papers/index.md) reads these papers properly; here they are only classified.

### Lesson 7 · RLAIF and scalable oversight

**The bottleneck.** Human preference labels are slow, expensive and inconsistent — the annotators disagree with each other, and their disagreement becomes noise in your reward. The obvious question: what if the judge is a language model? *(textbook fact)*

**Constitutional AI, in three steps.** *(paper result, reported and not reproduced — in Reading)*

1. Write down the principles you want the model to follow, in plain language — the "constitution".
2. Have the model **critique and revise its own answer** against those principles. Human involvement drops to writing the principles and auditing a sample of the revisions.
3. Train the reward model on the model's *own* comparisons of revised answers, then run the usual RL loop. This step is where the term **RLAIF** (reinforcement learning from AI feedback) comes from: the same loop as RLHF with a different source of preference.

**What the reported comparison says.** An off-the-shelf language model was used to label preferences, and the resulting policies were compared with human-feedback ones: on the summarisation tasks studied, the AI-feedback policies were at least comparable. *(paper result, reported and not reproduced — RLAIF vs. RLHF, in Reading)* The caveat matters as much as the result: one task family, one judge model, and the humans had to write the judging prompt anyway.

**LLM-as-a-judge, and its known biases.** *(paper result, reported and not reproduced — Judging LLM-as-a-Judge, in Reading)* Using a model to score or rank outputs is now standard, and the same study that reports high agreement with human preference also names the systematic errors:

| Bias | What it looks like | The cheap mitigation |
|---|---|---|
| **Position bias** | The answer shown first wins more often | Score each pair in both orders and average |
| **Verbosity bias** | Longer answers are rated higher regardless of content | Normalise by length, or score with length as a control |
| **Self-preference** | A model rates its own outputs higher | Use a different judge family than the policy |

The consequence for you is not "do not use a judge". It is that **the judge is a measurement instrument, and instruments have calibration** — so you hold out a human-labelled set, measure the judge against it, and re-measure when the judge model changes. Treat a judge whose accuracy you have never measured as an unknown constant in your objective. *(my judgement, and this is the point [Chapter 11](../11-evaluation/index.md) develops)*

**Self-rewarding, and the ceiling it hits.** *(paper result, reported and not reproduced — Self-Rewarding Language Models, in Reading)* The model generates candidate answers, judges them, trains on its own judgement, and repeats. The appeal is a loop with no external label at all. The ceiling is structural: the judge is the policy, so its biases are not averaged away by an independent rater — they are amplified, iteration after iteration. A model that systematically prefers its own style will teach itself that style more and more confidently. *(my judgement on the mechanism; the paper reports improvement over several iterations on the tasks it studied)*

**Scalable oversight, stated generally.** As models become better than us at a task, our ability to *check* them — not to produce the answer ourselves — becomes the binding constraint. That reframing is the useful part of this lesson, because it tells you where to invest: *(my judgement)*

- If a program can check it, use the program (Lesson 8). Nothing beats a verifier for cost and clarity.
- If only judgement can check it, make the judge's accuracy a measured quantity, and keep a human-labelled set frozen before training starts.
- If checking requires reading a long trace, check the *process* rather than only the outcome ([Chapter 5](../05-reward-engineering/index.md)).

**What it costs.** *(my judgement)* Cheap labels with correlated errors. A bias in the judge becomes a systematic bias in the policy, and no reward curve will show it, because the reward curve is drawn by the same instrument. The mitigation is more measurement, not more training.

### Lesson 8 · RLVR

**The name, and where it comes from.** RLVR (reinforcement learning with verifiable rewards) means the reward is produced by a **program that can decide correctness** — unit tests, an exact-match checker, a solver, a schema validator. The practice was named in Tulu 3 and made famous by DeepSeek-R1, both in Reading. The name describes the *source* of the reward, not a new optimiser: the RL machinery underneath is the PPO/GRPO objective from Lessons 2 and 4. *(textbook fact)*

**Why it is attractive.** *(my judgement)* No reward model to train, no annotators in the loop per iteration, and no proxy to hack *in the learned-model sense* — on the checkable slice, the signal is correctness itself. Scoring is also nearly free: evaluate a group of answers by running the tests, and you have both the rewards and the advantages GRPO needs. This is the cleanest fit between a method and a task in the whole book.

**The boundaries, five of them, each with the practical consequence.**

1. **The checker's coverage is the ceiling.** The model learns to pass *your* checker. A partial test suite is not a partial specification of correctness; it is a complete specification of passing that suite. *(textbook fact, with an uncomfortable edge: this is reward hacking wearing a lab coat — Lesson 9)* Write down what the checker does not cover, and expect the model to find it.
2. **Verifiability covers a narrow slice.** Maths with a checkable answer, code with tests, structured output against a schema, and some tool-use tasks in a deterministic environment. Most interesting agent behaviour — a good research summary, a well-chosen plan, a helpful clarification — is outside the slice entirely. *(my judgement)*
3. **A rising curve is not evidence that the reward is doing the work.** A reported study found that RLVR runs can improve benchmark scores even when the reward is *random* or deliberately wrong, when the base model already has the right prior — the update is then driven by format and likelihood effects rather than by the reward signal. *(paper result, reported and not reproduced — Spurious Rewards, in Reading)* The consequence: ablate your verifier before you credit it.
4. **The pass@k precondition.** If the policy never produces a correct answer, there is nothing to reinforce; if it always does, the advantage is zero. Learning happens in between — which is the [Chapter 1](../01-landscape/index.md), Lesson 3 test, now doing real work: RLVR cannot invent a capability that appears with probability zero.
5. **Format shortcuts.** The model may learn to make its output *checkable* rather than correct: answering only when confident, guessing in a shape the parser accepts, or looping until a test passes by accident. The score cannot distinguish these from understanding. Reading a sample of traces by hand is not optional. *(my judgement)*

**The feedback sources side by side.** *(my judgement)*

| Source | Can express | Cannot express | Cost per iteration |
|---|---|---|---|
| Human comparisons (RLHF) | Taste, style, the things people will not write down | Anything at scale, or consistently | Annotators, and their disagreement |
| Model judge (RLAIF) | Most of the above, fast | Anything the judge is systematically biased about | Inference on the judge, plus the calibration work |
| Program (RLVR) | Exactly what it checks | Everything it does not check | Running the checker |

**What it costs.** *(my judgement)* Environment engineering, which is real work: a sandbox that runs untrusted code, resets between attempts, and returns a reward without lying ([Chapter 8](../08-environment-engineering/index.md)). And a false sense of universality — "verifiable" is a property of the *task*, not of the algorithm. A task does not become verifiable because you want a cheap reward for it.

### Lesson 9 · Five alignment traps

Every trap below produces the same symptom: a reward curve that looks healthy. That is why the defences are all about measurement rather than about watching the training log. *(my judgement for the framing; the cited mechanisms are paper results, reported and not reproduced)*

**1. Reward hacking.** The policy finds the flaw in the proxy: the score rises while a human reading the same outputs sees quality fall. This is not a rare bug but the expected behaviour of a powerful optimiser on an imperfect summary — the general problem was named early *(Concrete Problems in AI Safety, in Reading)* and measured at scale *(Scaling Laws for Reward Model Overoptimization, verified in Chapter 2's Reading)*. *Detect:* a held-out human or gold check, plus reading traces. *Mitigate:* the KL leash, a reward that is averaged over several models rather than one, and adversarial examples fed back into the reward model.

**2. Distribution shift, and the offline ceiling.** DPO-family training on stale pairs; the loss is meaningful near the reference and undefined far from it. *Detect:* track the implicit reward's margin on a held-out set — when the chosen/rejected gap saturates, the data has no more to teach. *Mitigate:* refresh the pairs using the current policy (iterative or online preference optimisation), and keep $\beta$ honest rather than turning it down to force more movement.

**3. Over-optimisation and collapse.** Pressing the proxy too hard: entropy collapse (the policy becomes nearly deterministic and stops exploring), repetition, language mixing, and answers that grow without getting better. *Detect:* watch entropy and KL, not only reward; watch the length distribution. *Mitigate:* the KL target and controller, early stopping on the *gold* metric rather than the proxy, and the $G$-sample structure of GRPO, which tends to expose collapse early because a collapsed policy scores the same on every sample.

**4. Format and reward shortcuts.** Satisfying the checker without the capability: answer-only formatting, safe refusals that score as harmless, delimiter exploits, padding to satisfy a length reward. *Detect:* score a small sample by hand, and keep an adversarially written test set the policy has never seen. *Mitigate:* a checker with held-out cases, format-agnostic evaluation, and never letting the reward be a pure format condition.

**5. Judgement drift in the evaluator.** The judge — human or model — begins to reward agreement, confidence or style. Sycophancy is the well-documented version: models agree with a stated user view more than the evidence warrants, and human raters reward the agreement *(Towards Understanding Sycophancy in Language Models, in Reading)*. *Detect:* re-label a frozen sample of old outputs with today's judge and compare with yesterday's labels — a drift test on the instrument itself. *Mitigate:* freeze the evaluation set before training, and re-measure the judge, not only the policy.

**The common defence, in one sentence.** *(my judgement)* None of these five can be caught by the number you are optimising, because that number is the thing being fooled. You need a measurement you did not train against, taken before you started, and re-taken afterwards.

### Lesson 10 · Choosing

**The four axes.** Every method in this chapter is a point in this space, and asking the four questions in order narrows the field quickly. *(my judgement)*

| Axis | The question to ask | What it rules out |
|---|---|---|
| **Feedback source** | Human, model, or program? | A program rules in RLVR with GRPO; only human taste rules in RLHF-style work |
| **Reward granularity** | Outcome only, or per step? | Per-step needs a process reward model ([Chapter 5](../05-reward-engineering/index.md)) or an environment that yields next-state signals ([Chapter 6](../06-credit-assignment/index.md)) |
| **Online sampling** | Can you afford fresh rollouts? | No → the DPO family. Yes, and it is the point → PPO or GRPO |
| **System cost** | Memory, wall-clock, and people | Four models → a smaller policy, LoRA, or GRPO. One pair file → a laptop |

**The order to actually ask them in.** *(my judgement)*

1. **Can a program check the answer?** If yes, you are in RLVR: use GRPO (or PPO) against the checker, and spend your effort on checker coverage, not on the algorithm.
2. **If not, do you have comparisons?** Start with the DPO family offline. It is one pass over a file, it is diagnosable, and the ceiling is known: when the implicit reward's margin saturates, the data has stopped teaching.
3. **Does the behaviour you want already occur sometimes?** This is the pass@k test from [Chapter 1](../01-landscape/index.md), Lesson 3. If it never occurs, no RL method will invent it — go back to data, prompts or SFT. If it always occurs, there is no advantage left, and you need harder tasks ([Chapter 7](../07-data-and-tasks/index.md)).
4. **What fits the machine?** The arithmetic from Lesson 3 decides between PPO with four models, GRPO without a critic, and DPO without either.

<figure class="book-figure">{{#include ../figures/04-selection-map.svg}}</figure>

*Figure 4.2 — the four questions in order. Each "no" pushes you one step further from the cleanest signal (a program that checks the answer) and one step closer to a learned judge you have to keep honest.*

**The deliverable's table, reasoned aloud.** *(my judgement — the version you fill in for your own task is in Deliverable below; this is what the reasoning sounds like)*

| Task feature | Pick | Why not the other two |
|---|---|---|
| A provably right answer exists (maths, code) | GRPO with a verifier | DPO cannot use the checker at all; RLHF adds a reward model that can only be worse than the program |
| Only preferences exist (writing, style) | The DPO family first, PPO with a reward model if the ceiling bites | GRPO needs a scoreable reward you do not have |
| Multi-turn tool calling | PPO or GRPO against an environment, with process signal if you can get it | DPO cannot learn from interactions the policy never performed |
| Plenty of offline data, no GPU | DPO on the pairs | PPO needs rollouts and four models; GRPO multiplies sampling by $G$ |

**The three sentences this chapter was for.** *(my judgement)*

1. The five names are not five algorithms competing; they are one loop with different **scoring contracts** — a trained reward model, a model judge, or a program.
2. PPO is the general-purpose optimiser, and its two ideas survive everywhere else: a clipped step, and a leash to the reference model.
3. DPO removes the sampling loop by algebra, and the price of that removal is everything the loop was giving you: exploration, fresh data, and an honest answer to "are the pairs still teaching me anything?".

**Still open after this chapter.** Where the score comes from is settled here. *How to design the score so it means what you wanted* is [Chapter 5](../05-reward-engineering/index.md), and *how to attribute that score to the right action inside a long episode* is [Chapter 6](../06-credit-assignment/index.md).

## Deliverable

A one-page selection table that **must include the negative argument**. Fill the empty cells for your own task, and for every row write the reason the other two methods lose — a row whose "why not" column is empty is a row you have not decided.

| Task feature | Pick | Why not the other two |
|---|---|---|
| A provably right answer exists (maths, code) | | |
| Only preferences exist (writing, style) | | |
| Multi-turn tool calling | | |
| Plenty of offline data, no GPU | | |

Then defend it out loud, in one sentence per row, to somebody who will ask "why not just DPO?".

## How you know you passed

- You can write the objectives of PPO, GRPO and DPO from memory, and point out how each depends on fresh sampling.
- You can explain what the KL penalty prevents, both in theory (drift away from the region where the reward means something) and in practice (entropy collapse, reward hacking, language mixing).
- You can state DPO's "no reward model needed" claim *and* its hidden price in one sentence — and say which of the five costs in Lesson 5 you would notice first in a training log.
- Your selection table is filled in for your own task, with a concrete cost named in every "why not" cell.

## Reading

Reused entries were verified where they are marked; the new entries below were checked against the arXiv API on 2026-10-09 (IDs marked "verified"). Numbers quoted from papers are **reported, not reproduced**; book references carry no arXiv ID and are named as such.

Versions of this list already verified in earlier chapters — reuse these rather than re-checking:

- **The pipeline that started RLHF at scale** — Ouyang et al., *Training language models to follow instructions with human feedback*, arXiv:2203.02155 — verified in Chapter 1's Reading, 2026-09-30. The three-step pipeline in Lesson 1.
- **DPO** — Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290 — verified in Chapter 1's Reading, 2026-09-30. The derivation in Lesson 5.
- **Where GRPO comes from** — Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300 — verified in Chapter 1's Reading, 2026-09-30. The group advantage in Lesson 4.
- **Constitutional AI** — Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, arXiv:2212.08073 — verified in Chapter 1's Reading, 2026-09-30. Lesson 7.
- **RLVR at the frontier** — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948 — verified in Chapter 1's Reading, 2026-09-30. Lesson 8.
- **The practice of RLVR, named** — Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*, arXiv:2411.15124 — verified in Chapter 1's Reading, 2026-09-30.
- **The runtime that runs these algorithms** — Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (VeRL), arXiv:2409.19256 — verified in Chapter 1's Reading, 2026-09-30. The four-model plumbing in Lesson 3.
- **PPO itself** — Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 — verified in Chapter 3's Reading, 2026-09-30.
- **Trust regions, the idea PPO simplifies** — Schulman et al., *Trust Region Policy Optimization*, arXiv:1502.05477 — verified in Chapter 3's Reading, 2026-09-30.
- **GAE** — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438 — verified in Chapter 3's Reading, 2026-09-30.
- **The implementation-details result** — Engstrom et al., *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*, arXiv:2005.12729 — verified in Chapter 3's Reading, 2026-09-30. Lesson 2's warning about clipped batches.
- **Reward overoptimisation** — Gao, Schulman and Hilton, *Scaling Laws for Reward Model Overoptimization*, arXiv:2210.10760 — verified in Chapter 2's Reading, 2026-09-30. Trap 1 in Lesson 9.
- **The ceiling of imitation** — Zhou et al., *LIMA: Less Is More for Alignment*, arXiv:2305.11206; Gudibande et al., *The False Promise of Imitating Proprietary LLMs*, arXiv:2305.15717 — both verified in Chapter 2's Reading, 2026-09-30. Lesson 1, point 2.

New here, checked on 2026-10-09:

- **The pairwise preference model underneath DPO** — Bradley and Terry, *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*, Biometrika 39 (1952) — no arXiv ID. Lesson 5, step 1.
- **The RLHF objective, by the people who ran it** — Askell et al., *A General Language Assistant as a Laboratory for Alignment*, arXiv:2112.00861, and Bai et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*, arXiv:2204.05862 — both verified. The HHH framing and the helpful/harmless negotiation in Lesson 1.
- **The practical PPO implementation** — Zheng et al., *Secrets of RLHF in Large Language Models Part I: PPO*, arXiv:2307.04964 — verified. Reward clipping and normalisation, the KL controller, and why the details decide the result (Lesson 2).
- **AI feedback instead of human feedback, measured** — Lee et al., *RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*, arXiv:2309.00267 — verified. Lesson 7.
- **Judging with a model, and what it gets wrong** — Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685 — verified. The bias table in Lesson 7.
- **Self-rewarding, and its structural ceiling** — Yuan et al., *Self-Rewarding Language Models*, arXiv:2401.10020 — verified. Lesson 7.
- **Preference tuning without pairs** — Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, arXiv:2402.01306 — verified. Lesson 6.
- **Preference tuning without a reference model, in one stage** — Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, arXiv:2403.07691 — verified. Lesson 6.
- **Preference tuning without a reference model, with a margin** — Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, arXiv:2405.14734 — verified. Lesson 6.
- **Where the clips and the sampling policy go** — Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476 — verified. Lesson 4's variants.
- **A cheaper baseline than a critic** — Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (RLOO), arXiv:2402.14740 — verified. Lesson 4.
- **The biases in GRPO, identified and removed** — Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), arXiv:2503.20783 — verified. The catch, in four parts, in Lesson 4.
- **The ratio applied to a whole sequence** — Zheng et al., *Group Sequence Policy Optimization* (GSPO), arXiv:2507.18071 — verified. Lesson 4's variants.
- **When the reward is not what is doing the work** — Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, arXiv:2506.10947 — verified. Lesson 8, boundary 3.
- **Sycophancy, measured** — Sharma et al., *Towards Understanding Sycophancy in Language Models*, arXiv:2310.13548 — verified. Trap 5 in Lesson 9.
- **The original framing of the safety problems** — Amodei et al., *Concrete Problems in AI Safety*, arXiv:1606.06565 — verified. Trap 1 in Lesson 9.
- **The RL foundations this chapter builds on** — Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition — no arXiv ID; used in Chapters 2, 3 and 4 alike.

## Backfill

- [x] Question 1: When should I reach for RL at all? — the algorithm side settles here. RL is worth its cost when you can name the behaviour you want more of, when it already occurs sometimes (the pass@k test), and when something can score it — a program, a model, or a person. Lesson 10 turns that into four questions asked in order.
- [x] Question 2: How do I design the reward? — this chapter names the three *sources* of a reward: a model trained on comparisons, a model used as a judge, or a program. Which source to choose is settled here (Lessons 7 and 8); how to design one that cannot be hacked is [Chapter 5](../05-reward-engineering/index.md).

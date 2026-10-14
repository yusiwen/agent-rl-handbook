# Glossary

Maintained in the order terms appear in the book. Definitions follow this book's usage; the "In plain words" column is my own phrasing and is not necessarily how a paper says it.

## Foundations

| Term | What it formally means | In plain words / notes |
|---|---|---|
| RL (reinforcement learning) | Learning a policy from interaction by maximising expected reward | Trial and error with a scoreboard |
| MDP | Markov decision process: the five-tuple $(S, A, P, R, \gamma)$ | A tidy way to write down a trial-and-error problem |
| POMDP | A Markov decision process where the agent cannot see the full state | What multi-turn agents really are: you never see the whole world |
| Trajectory | The sequence of states, actions and rewards from one interaction | For an agent, one multi-turn session; the unit of training data |
| Return, $G_t$ | The discounted sum of future rewards from step $t$ | A score, but counting the future |
| Value function, $V^\pi(s)$ | Expected return from state $s$ under policy $\pi$ | "How good is this situation on average" |
| Advantage, $A^\pi(s,a)$ | How much better an action is than the average action | The quantity PPO and GRPO actually use; GRPO compares against the group's average |
| GAE | Generalised advantage estimation | A dial between noisy-but-unbiased and stable-but-biased; derivation ends in [Chapter 3](../03-policy-gradient/index.md) |
| On-policy / off-policy | Learning from data the current policy just produced vs. from older data | Decides how fresh your samples must be |
| Critic | A learned estimate of the value of a state, used as the baseline for the advantage | A second trained model, and a second source of error; GRPO removes it |
| Reference model | A frozen copy of the model training started from, used to measure drift | The leash in the KL penalty; not the same thing as a "teacher" |
| Clip ratio (probability ratio) | The new policy's probability of an action divided by the sampling policy's | How much the policy has moved since the data was collected; the thing PPO clips |
| PPO | Proximal policy optimisation: a clipped surrogate objective plus a KL penalty | The general-purpose optimiser of this field, and the expensive one; the objective lives in [Chapter 4](../04-alignment-algorithms/index.md) |
| Pass@k | The probability that at least one of $k$ samples is correct | The precondition for learning: RL amplifies what already happens sometimes |

## Alignment algorithms

| Term | What it formally means | In plain words / notes |
|---|---|---|
| SFT | Supervised fine-tuning on example answers | Teaches format and basic skill; cannot teach preference or long-horizon judgement |
| RLHF | Human-feedback RL: SFT → reward model → PPO with a KL penalty | Four models in play: policy, reward model, critic, reference |
| HHH | Helpful, honest and harmless: the usual decomposition of "aligned" | Three goals that pull against each other, so they need a reward to trade between them |
| Reward model (RM) | A model trained to predict which answer a human would prefer | An automatic judge, and therefore also an automatically hackable judge |
| KL penalty | A term that keeps the trained policy close to a reference policy | Stops the model drifting into nonsense while chasing the score |
| KL controller | A control loop that adjusts the KL coefficient toward a target divergence | The coefficient is tuned during training, not fixed once |
| RLAIF | RL from AI feedback instead of human labels | Constitutional AI is the best-known implementation |
| DPO | Direct preference optimisation: skip the reward model and the sampling loop | **Offline preference optimisation** with an implicit reward |
| Bradley-Terry model | A pairwise comparison model: the probability one item beats another is a sigmoid of their score difference | The standard way to turn "A is better than B" into a likelihood |
| Implicit reward | In DPO, the quantity $\beta \log (\pi_\theta / \pi_{\text{ref}})$ | The policy is secretly the reward model — that is what "direct" means |
| RLVR | RL with verifiable rewards | Maths and code: a program checks the answer, no human needed |
| GRPO | Group relative policy optimisation: sample a group, compare within it, drop the critic | Cheaper than PPO; suits verifiable tasks |
| RLOO | REINFORCE leave-one-out: the baseline is the mean of the other samples | Keeps the group idea, drops the spread normalisation |
| DAPO | An open RL system whose recipe includes clip-higher, dynamic sampling and token-level loss | The patches that keep a group informative and overlong answers in check |
| GSPO | Group sequence policy optimisation: the importance ratio is per sequence, not per token | The clip acts on the whole answer, which suits long reasoning traces |
| Dr. GRPO | The GRPO recipe with the spread and length normalisations removed | A paper that names two biases in GRPO and deletes them; "doctor" is a joke about fixing the objective |
| KTO | Model alignment with a prospect-theory loss over single labelled examples | Preference tuning without pairs; failures can be made to weigh more than successes |
| ORPO | Odds-ratio preference optimisation: the preference penalty is folded into SFT | One stage instead of two, and no reference model |
| SimPO | Simple preference optimisation: a length-normalised, reference-free implicit reward | No reference model, plus a margin between chosen and rejected |
| LLM-as-a-judge | Using a language model to score or rank outputs | A measurement instrument with known biases: position, verbosity, self-preference |
| Reward hacking | The model optimises the score instead of the goal | Defences: KL limits, model ensembles, adversarial examples, production monitoring |
| Over-optimisation | Reward keeps rising while real quality falls | The classic curve once a judge has been hacked |
| Entropy collapse | The policy becomes nearly deterministic, so exploration stops | Shows up as falling entropy and identical samples — not as a bad reward curve |
| Sycophancy | Agreeing with the user beyond what the evidence supports, because agreement was rewarded | A trap in the evaluator, not in the model's code |

## Rewards and credit assignment

| Term | What it formally means | In plain words / notes |
|---|---|---|
| ORM | Outcome reward model: scores only the final answer | Strongest signal, coarsest resolution |
| PRM | Process reward model: scores each step | Needs labels or extra sampling; you trade resolution for cost |
| GRM | Generative reward model: produces feedback text rather than a number | More information, and human-readable |
| Credit assignment | Deciding which actions are responsible for the final outcome | A map of the options is in [Chapter 6](../06-credit-assignment/index.md) |
| Next-state signal | Using the environment's next state as feedback | The key mechanism in multi-turn tool settings |
| Pass rate | The share of attempts that succeed on a task | Below 0.1 or above 0.9, there is almost no learning signal left |
| Reward granularity | How many decisions inside one attempt get their own score | The choice behind every other decision in [Chapter 5](../05-reward-engineering/index.md): dense teaches faster and costs more |
| Reward shaping | Adding a guidance term to the reward to make learning easier | Provably harmless only when the term is the difference of a potential between states (the shaping theorem) |
| Process supervision | Scoring each step rather than only the final answer | Buys credit assignment; the labels are the expensive part |
| Pseudo-label (agreement) | Taking the majority answer across several samples as if it were a label | Free, and only as good as the model's own consistency |
| Implicit process reward | A per-token dense signal read off the policy's own log-ratio against a reference | No labels and no trained PRM; PRIME is the named route, and it amplifies the model's own preferences |
| Reward-model refresh | Re-collecting comparisons from the current policy and retraining the reward model | A trigger (agreement dropping), not a calendar |
| Reward-model agreement | The rate at which a reward model matches held-out human or gold labels | The measurement that decides whether it is still good; loss is not a substitute |
| Hack canary | A task item known to be gameable, watched as an early warning | Cheaper and more sensitive than waiting for the gold metric to move |
| Failure boundary | The written list of where a verifier gives the wrong signal | The deliverable of Project 1; a verifier without one is an untested verifier |
| Verifier service | A reward function turned into a contract: schema, queue, timeout policy, cache, version | Where a cheap reward's real cost lives ([Chapter 5](../05-reward-engineering/index.md), Lesson 8) |
| Replayable dataset | Every `(task, completion, mark, verifier version)` row, stored | Lets an old run be re-scored without retraining; without the version it proves nothing |
| Best-of-n verification | Sampling several answers and ranking them with a verifier or reward model instead of training | Test-time compute buying accuracy; it needs a scorer that is better than the model |
| Step boundary | Where one step in a trajectory ends and the next begins | Your modelling choice, not a fact about the task: change the segmentation and every score means something else ([Chapter 6](../06-credit-assignment/index.md), Lesson 1) |
| Turn boundary | The recorded line between the policy's action and the environment's response | Turn-level credit needs it recorded in the trace; nothing reconstructs it later |
| RRM (reasoning reward model) | A reward model that emits a reasoning trace, a focused critique and an overall score | Agent-RRM is the named implementation the project builds on (Reagent, [Chapter 6](../06-credit-assignment/index.md)) |
| Reagent-C / Reagent-R / Reagent-U | Three ways to use a reasoning reward model: critique in the prompt, score as the reward, or both merged | In-context refinement, a dense model reward, or unified feedback integration — the last is the project's target |
| Reward fusion | Merging a rule reward and a model reward into one number | The weights encode a trade-off; write them down and log both terms separately ([Chapter 5](../05-reward-engineering/index.md), Lesson 5) |
| Critique-augmented sampling | Spending the expensive judge only on the samples whose outcome is still uncertain | Makes a dense multi-signal reward affordable; watch that the gate does not become a surface the policy can play |
| Advantage spread | The within-group variation an update needs in order to learn | No spread, no signal: an all-correct or all-wrong group teaches nothing ([Chapter 4](../04-alignment-algorithms/index.md), Lesson 4) |
| Echo Trap | The multi-turn collapse where reasoning degenerates into repetitive templates and entropy dies | Reported for multi-turn agent training; it looks like a bug and is a property of the setting |

## Data and tasks

| Term | What it formally means | In plain words / notes |
|---|---|---|
| Task family | A group of items that share a task shape and a scorer | Word problems, code tasks and logic puzzles are three families with three verifiers ([Chapter 7](../07-data-and-tasks/index.md)) |
| Trainable band | The pass-rate region where a policy sometimes succeeds and sometimes fails | 0.1 to 0.9 by convention, and the edge is set by your sample count; both ends give a gradient of zero |
| Pass-rate filtering | Keeping the items whose pass rate falls inside the trainable band | The measured counterpart of Chapter 4's zero-variance group ([Chapter 7](../07-data-and-tasks/index.md), Lesson 4) |
| Cold start | A small supervised stage that teaches the shape of a scorable attempt before RL starts | RL learns nothing from a policy that never produces a scoreable answer, because every group is flat |
| Label protocol | What the verifier accepts as a declared answer, written down next to the reward | Part of the reward, not the prompt: in Project 1's numeric verifier the answer must be declared, and the final line carries exactly one number |
| Rejection sampling | Sampling several attempts per task and keeping the ones the verifier passes | Cheap supervised data, and biased toward the tasks and styles the policy already handles |
| Self-play data | The attempts are regenerated from the current policy each round, on a fixed task set | A loop rather than a corpus; it needs an automatic scorer, or it is a diary ([Chapter 6](../06-credit-assignment/index.md), Lesson 6) |
| Decontamination | Measuring and removing the overlap between a training set and the evaluation set | A script you can re-run, never a visual inspection: a checker that has never been shown to fire has not been verified |

## Systems and engineering

| Term | What it formally means | In plain words / notes |
|---|---|---|
| rollout | Sampling trajectories from the current policy | The most expensive step in RL, often costlier than the training update itself |
| DataProto | The data-exchange protocol used by VeRL | Half of why distributed training can look like one process |
| loss mask | A flag marking which tokens contribute to the gradient | Decides whether multi-turn samples can be trained correctly at all |
| Sample freshness | How far an asynchronously collected sample is from the current policy | Older means more biased; the trade-off against throughput |
| Weight sync | Pushing updated weights to the sampling side in an async setup | How often you do it decides how stable training is |
| Verifier | A program that marks an answer correct or incorrect | A rulebook, not a judge with taste |
| Internalise vs externalise | Train a capability into the weights, or keep it in an external library | Decided across four criteria: data, latency, scenario, ecosystem |

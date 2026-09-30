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

## Alignment algorithms

| Term | What it formally means | In plain words / notes |
|---|---|---|
| SFT | Supervised fine-tuning on example answers | Teaches format and basic skill; cannot teach preference or long-horizon judgement |
| RLHF | Human-feedback RL: SFT → reward model → PPO with a KL penalty | Four models in play: policy, reward model, critic, reference |
| Reward model (RM) | A model trained to predict which answer a human would prefer | An automatic judge, and therefore also an automatically hackable judge |
| KL penalty | A term that keeps the trained policy close to a reference policy | Stops the model drifting into nonsense while chasing the score |
| RLAIF | RL from AI feedback instead of human labels | Constitutional AI is the best-known implementation |
| DPO | Direct preference optimisation: skip the reward model and the sampling loop | **Offline preference optimisation** with an implicit reward |
| RLVR | RL with verifiable rewards | Maths and code: a program checks the answer, no human needed |
| GRPO | Group relative policy optimisation: sample a group, compare within it, drop the critic | Cheaper than PPO; suits verifiable tasks |
| Reward hacking | The model optimises the score instead of the goal | Defences: KL limits, model ensembles, adversarial examples, production monitoring |
| Over-optimisation | Reward keeps rising while real quality falls | The classic curve once a judge has been hacked |

## Rewards and credit assignment

| Term | What it formally means | In plain words / notes |
|---|---|---|
| ORM | Outcome reward model: scores only the final answer | Strongest signal, coarsest resolution |
| PRM | Process reward model: scores each step | Needs labels or extra sampling; you trade resolution for cost |
| GRM | Generative reward model: produces feedback text rather than a number | More information, and human-readable |
| Credit assignment | Deciding which actions are responsible for the final outcome | A map of the options is in [Chapter 6](../06-credit-assignment/index.md) |
| Next-state signal | Using the environment's next state as feedback | The key mechanism in multi-turn tool settings |
| Pass rate | The share of attempts that succeed on a task | Below 0.1 or above 0.9, there is almost no learning signal left |

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

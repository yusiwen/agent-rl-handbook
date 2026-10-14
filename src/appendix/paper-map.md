# The Paper Map

> **Status**: growing. **76 papers on eight shelves, in 42 slots** - plus 7 slots still empty (49 slots in total). Every ID on this page was checked against the arXiv API: the 73 rows filed on **2026-10-09**, and the three added on **2026-10-14**.
> **What this page is for**: filing, not reading. Its job is to answer "have I seen this shape of paper before, and is it worth my evening?" in about thirty seconds.

## First, the gist

Reading RL papers one at a time does not work, and the reason is not the volume - it is that there is nowhere to put them. Every week brings a new PPO variant, a new preference objective, a new reward shape, and without a fixed set of places to put them, each one has to be understood from scratch before it can be judged.

A map fixes that by making **filing** the first act and **reading** the second. You look at a new paper, decide which slot it belongs in, and only then decide whether the slot is one you care about. Most papers get filed and never read, and that is the point: you have a reason, and the reason took thirty seconds.

An empty slot is not a hole in this page. It is a to-read list with a name on it - it tells you exactly what kind of paper you are missing, which a list of titles never does.

## The four axes, and how to file a paper

Every paper on this page can be described along four axes. Ask them in order:

1. **Algorithm family** - which of the four families does this change: reward modelling, preference optimisation, policy-gradient methods, or the system that runs them? (Chapter 1 sorts the families by who scores the attempt.)
2. **Engineering theme** - which engineering problem does it move: reward design, credit assignment, data and tasks, environment, throughput, evaluation?
3. **Application domain** - maths, code, tool use, web, terminal, dialogue? A result in one domain is a hypothesis in another.
4. **Agent capability** - does it change the weights, the environment, the reward, or the skill library? Only the first of those is training.

Then the paper goes on one shelf. If it seems to belong on two, it goes where **its evidence** is strongest and gets a cross-reference in the sentence - the same rule the chapters follow.

## The eight shelves, at a glance

| Shelf | What lands here | Slots | Papers |
|---|---|---|---|
| **1** | Classical RL results that every later chapter quietly assumes. | 5 | 8 |
| **2** | Why supervised fine-tuning is not enough, and what the first RL stage looked like. | 5 | 8 |
| **3** | Replacing human labelers with models, and the new failure modes that buys. | 4 | 6 |
| **4** | One objective, no reward model, no rollouts - and five years of patching it. | 4 | 5 |
| **5** | The shelf that produced the current default: score what a program can check. | 5 | 10 |
| **6** | Building the scorer, and the long list of ways it gets fooled. | 6 | 13 |
| **7** | Where the reward has to be attached to a sequence of decisions, not one answer. | 7 | 14 |
| **8** | The frontier shelf: everything that is neither a reward question nor an algorithm question. | 6 | 12 |
| **Total** | | **42** | **76** |

## Shelf 1: Foundations and the policy-gradient line

*Classical RL results that every later chapter quietly assumes.*

**Trust regions and the clipped objective**

- Proximal Policy Optimization Algorithms (PPO), `1707.06347` - the clipped objective, and how far a policy is allowed to move per update *Used in Ch 1, Ch 3, Ch 4.*
- Trust Region Policy Optimization (TRPO), `1502.05477` - the constrained-step idea PPO approximates in order to be cheap *Used in Ch 3, Ch 4.*

**Advantage estimation and its variance**

- High-Dimensional Continuous Control Using Generalized Advantage Estimation, `1506.02438` - the advantage estimator every implementation calls; the lambda knob in it trades bias for variance *Used in Ch 2, Ch 3, Ch 4, Ch 6.*
- A Closer Look at Deep Policy Gradients, `1811.02553` - why a policy-gradient estimate is high-variance by construction *Used in Ch 3.*

**What actually moves the curve (implementation sensitivity)**

- Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO, `2005.12729` - code-level choices, not the objective, explain much of PPO's reported edge *Used in Ch 3, Ch 4.*
- What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study, `2006.05990` - a large audit of which knobs matter and which are folklore *Used in Ch 3.*

**Asynchrony in classical RL**

- Asynchronous Methods for Deep Reinforcement Learning, `1602.01783` - A3C: parallel actors, gradients that are already slightly stale - the ancestor of Chapter 9's async trainers *Used in Ch 3.*

**The safety framing that still sets the agenda**

- Concrete Problems in AI Safety, `1606.06565` - reward hacking, side effects and scalable oversight, named years before LLMs *Used in Ch 4, Ch 5, Extras.*

## Shelf 2: Instruction following, the limits of SFT, and RLHF

*Why supervised fine-tuning is not enough, and what the first RL stage looked like.*

**The three-stage recipe and where it came from**

- Training language models to follow instructions with human feedback, `2203.02155` - the SFT, then reward model, then PPO recipe as shipped *Used in Ch 1, Ch 4.*
- Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback, `2204.05862` - helpfulness and harmlessness treated as two objectives that pull against each other *Used in Ch 4.*
- A General Language Assistant as a Laboratory for Alignment, `2112.00861` - preference modelling and its measurements, at small scale *Used in Ch 4.*

**The limits of imitation**

- LIMA: Less Is More for Alignment, `2305.11206` - 1,000 curated examples beat bulk instruction data: surface form is cheap to copy *Used in Ch 2, Ch 4, Ch 7.*
- The False Promise of Imitating Proprietary LLMs, `2305.15717` - imitation copies style, not capability - and the gap shows up on hard tasks *Used in Ch 2, Ch 4.*

**What SFT cannot teach**

- SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training, `2501.17161` - the cleanest statement of the difference, with the experiments behind it *Used in Ch 2, Ch 7.*

**Open post-training recipes**

- Tulu 3: Pushing Frontiers in Open Language Model Post-Training, `2411.15124` - a full open recipe with its data and its decision log *Used in Ch 1, Ch 4, Ch 5.*

**The RL stage up close**

- Secrets of RLHF in Large Language Models Part I: PPO, `2307.04964` - what actually breaks inside the PPO stage, and the fixes that were tried *Used in Ch 4.*

## Shelf 3: Feedback that comes from models: judges, constitutions, self-reward

*Replacing human labelers with models, and the new failure modes that buys.*

**Principle-driven feedback**

- Constitutional AI: Harmlessness from AI Feedback, `2212.08073` - critique and revise against written principles, then RL from the model's own feedback *Used in Ch 1, Ch 4.*

**AI feedback at scale**

- RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback, `2309.00267` - AI labels match human labels on the tasks tested, at a fraction of the cost *Used in Ch 4.*

**Self-rewarding and self-taught evaluation**

- Self-Rewarding Language Models, `2401.10020` - the model writes its own preference pairs, then trains on them *Used in Ch 4.*
- Self-Taught Evaluators, `2408.02666` - a judge trained with no human preference labels at all *Used in Ch 5.*

**Judging the judges**

- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, `2306.05685` - position, verbosity and self-preference bias, measured rather than suspected *Used in Ch 4, Ch 5.*
- RewardBench: Evaluating Reward Models for Language Modeling, `2403.13787` - a benchmark for the scorers themselves *Used in Ch 5.*

## Shelf 4: Direct preference optimisation and its repairs

*One objective, no reward model, no rollouts - and five years of patching it.*

**The objective and its implicit reward**

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model, `2305.18290` - preference pairs straight into the policy; no reward model, no sampling *Used in Ch 1, Ch 4.*

**Reference-free and monolithic variants**

- SimPO: Simple Preference Optimization with a Reference-Free Reward, `2405.14734` - drop the reference model and keep the margin *Used in Ch 4.*
- ORPO: Monolithic Preference Optimization without Reference Model, `2403.07691` - fold supervised fine-tuning and preference tuning into one stage *Used in Ch 4.*

**Non-pairwise and prospect-theoretic variants**

- KTO: Model Alignment as Prospect Theoretic Optimization, `2402.01306` - learn from a single thumbs-up or thumbs-down instead of a pair *Used in Ch 4.*

**What the objective is really doing (theory repairs)**

- Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs, `2402.14740` - a plain REINFORCE-style objective competes with DPO - a reason to question the machinery *Used in Ch 4.*

## Shelf 5: Verifiable rewards, group methods, and tool-use rewards

*The shelf that produced the current default: score what a program can check.*

**The group-relative idea**

- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, `2402.03300` - GRPO: replace the critic with the average of a sampled group *Used in Ch 1, Ch 4.*

**Reasoning RL at scale**

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, `2501.12948` - RL on verifiable rewards alone produces reasoning behaviour *Used in Ch 1, Ch 4.*
- Kimi k1.5: Scaling Reinforcement Learning with LLMs, `2501.12599` - the same conclusion from another lab, with long-context detail *Used in Ch 6.*

**Normalisation bias, entropy collapse, length bias**

- DAPO: An Open-Source LLM Reinforcement Learning System at Scale, `2503.14476` - four named fixes for group-method instability *Used in Ch 4.*
- Understanding R1-Zero-Like Training: A Critical Perspective, `2503.20783` - a critical audit of what the recipe really depends on *Used in Ch 1, Ch 4.*
- Group Sequence Policy Optimization, `2507.18071` - move the ratio and the clip from the token level to the sequence level *Used in Ch 4.*

**Verifiable rewards can mislead**

- Spurious Rewards: Rethinking Training Signals in RLVR, `2506.10947` - on some models a wrong reward still improves the benchmark - so the gain was not the reward *Used in Ch 4, Ch 5, Ch 7.*

**Group methods for agents, and rewards for tool use**

- Group-in-Group Policy Optimization for LLM Agent Training, `2505.10978` - credit at two granularities: turn and step *Used in Ch 6.*
- ToolRL: Reward is All Tool Learning Needs, `2504.13958` - reward design for tool calls, not only for final answers *Used in Ch 6.*
- ReTool: Reinforcement Learning for Strategic Tool Use in LLMs, `2504.11536` - cold-start SFT plus an outcome reward teaches when to run code *Used in Ch 1, Ch 7, Ch 10.*

## Shelf 6: Reward modelling and robustness

*Building the scorer, and the long list of ways it gets fooled.*

**Verifiers and outcome reward models**

- Training Verifiers to Solve Math Word Problems, `2110.14168` - the verifier-as-a-second-model idea, and the grade-school maths set that came with it *Used in Ch 2, Ch 5.*

**Process supervision**

- Let's Verify Step by Step, `2305.20050` - process beats outcome on hard maths - and here is what it costs *Used in Ch 2, Ch 5.*
- Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations, `2312.08935` - step-level labels without a human writing them *Used in Ch 5.*
- The Lessons of Developing Process Reward Models in Mathematical Reasoning, `2501.07301` - what actually failed in a real process-reward build *Used in Ch 5.*

**Implicit and generative rewards**

- Process Reinforcement through Implicit Rewards, `2502.01456` - process reward from an implicit reward, with no step labels *Used in Ch 5.*
- Generative Verifiers: Reward Modeling as Next-Token Prediction, `2408.15240` - score by generating a verdict instead of emitting a number *Used in Ch 5.*

**Reward models for agents**

- Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems, `2502.19328` - combine human preference with checks a program can run *Used in Ch 6.*
- Exploring Reasoning Reward Model for Agents, `2601.22154` - per-step trace, critique and score: the reward model behind Chapter 6's project *Used in Ch 6.*

**Overoptimisation**

- Scaling Laws for Reward Model Overoptimization, `2210.10760` - more optimisation eventually makes the model worse, and there is a curve for it *Used in Ch 2, Ch 4, Ch 5.*

**Ensembling, attack and flattery: the robustness of the judge**

- Reward Model Ensembles Help Mitigate Overoptimization, `2310.02743` - average several judges to flatten that curve *Used in Ch 5.*
- Universal and Transferable Adversarial Attacks on Aligned Language Models, `2307.15043` - the scorer can be attacked through its input *Used in Ch 5.*
- Towards Understanding Sycophancy in Language Models, `2310.13548` - models agree with the user - and a preference-trained scorer inherits the taste *Used in Ch 4, Ch 5.*
- Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting, `2305.04388` - the stated reasoning is not the real one, so a rationale is not evidence *Used in Ch 5.*

## Shelf 7: Agents: environments, multi-turn training, credit

*Where the reward has to be attached to a sequence of decisions, not one answer.*

**Agent benchmarks and environments**

- AgentBench: Evaluating LLMs as Agents, `2308.03688` - many environments, one harness - the shape every agent eval copies *Used in Extras.*
- WebArena: A Realistic Web Environment for Building Autonomous Agents, `2307.13854` - a web environment you can actually reset *Used in Extras.*
- SWE-bench: Can Language Models Resolve Real-World GitHub Issues?, `2310.06770` - a verifiable task built from real repositories; the ancestor of code-agent RL *Used in Ch 5, Extras.*
- SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering, `2405.15793` - the interface matters as much as the model *Used in Extras.*

**Multi-turn training**

- ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL, `2402.19446` - hierarchical multi-turn RL: a value estimate per turn *Used in Ch 2, Ch 5, Ch 6.*
- RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning, `2504.20073` - self-evolution in multi-turn RL, and how it collapses *Used in Ch 6.*
- SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks, `2503.15478` - per-turn credit on collaborative tasks *Used in Ch 6.*

**Offline and hierarchical credit**

- Offline Reinforcement Learning for LLM Multi-Step Reasoning, `2412.16145` - credit assignment without fresh rollouts *Used in Ch 6.*

**Turn-level credit**

- TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents, `2607.13988` - turn-level reward assignment by estimating credit *Used in Ch 6.*

**Failure analysis**

- Why Do Multi-Agent LLM Systems Fail?, `2503.13657` - a taxonomy of agent failures - useful even when you build a single agent *Used in Extras.*

**When the observation is an attack surface**

- Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection, `2302.12173` - tool output and retrieved text are untrusted input *Used in Extras.*

**Programmatic task and environment synthesis**

- Endless Terminals: Scaling RL Environments for Terminal Agents, `2601.16443` - a four-stage generator (task descriptions, validated containers, completion tests, solvability filter) that produced 3,255 terminal tasks, trained on with vanilla PPO and a binary reward *Used in Ch 7.*
- EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis, `2601.05808` - environment skeletons, then task scenarios generated together with rule-based validators: the task and its check arrive together *Used in Ch 7.*
- Procedural Environment Generation for Tool-Use Agents (RandomWorld), `2506.11045` - interactive tools and compositional tool-use data generated procedurally; downstream performance scales with the amount of generated data *Used in Ch 7.*

## Shelf 8: Systems, inference-time effort, memory and skills

*The frontier shelf: everything that is neither a reward question nor an algorithm question.*

**Runtime architecture**

- HybridFlow: A Flexible and Efficient RLHF Framework, `2409.19256` - the paper behind VeRL: separate the control flow from the computation *Used in Ch 1, Ch 4.*

**Training agents with existing trainers**

- Agent Lightning: Train ANY AI Agents with Reinforcement Learning, `2508.03680` - turn an agent you already wrote into something trainable *Used in Ch 6.*

**Asynchronous online RL**

- OpenClaw-RL: Train Any Agent Simply by Talking, `2603.10165` - four asynchronous rings, and three paradigms: binary reward, on-policy distillation, or both *New on the map: not yet cited in a chapter.*
- RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System, `2602.02488` - build environment, policy and reward together in one dynamic loop *New on the map: not yet cited in a chapter.*

**Self-distillation as a teacher signal**

- Self-Distillation Enables Continual Learning, `2601.19897` - self-distillation as a way to keep learning without forgetting *New on the map: not yet cited in a chapter.*
- Reinforcement Learning via Self-Distillation, `2601.20802` - the same idea turned into an RL objective *New on the map: not yet cited in a chapter.*

**Inference-time effort as the alternative to training**

- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters, `2408.03314` - thinking longer and sampling more can beat a bigger model - the cheap half of the ceiling *Used in Ch 1, Ch 5.*
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, `2201.11903` - the behaviour that RL later trains in *Used in Ch 2.*
- Self-Consistency Improves Chain of Thought Reasoning in Language Models, `2203.11171` - sample many, keep the majority answer *Used in Ch 5.*

**Memory and capability outside the weights**

- MemGPT: Towards LLMs as Operating Systems, `2310.08560` - the model as an operating system, with memory managed outside it *Used in Extras.*
- Memento-Skills: Let Agents Design Agents, `2603.18743` - skills discovered, verified and written back while the weights stay frozen *Used in Ch 12.*
- Lost in the Middle: How Language Models Use Long Contexts, `2307.03172` - long context is not uniform attention - where you put the skill matters *Used in Extras.*

## Slots that are still empty

The names in this table are the papers this book has not read yet. When one of them is filled, the row moves onto its shelf and disappears from here - so this section shrinking is how the map reports progress.

| Shelf | The slot waiting to be filled |
|---|---|
| 2 | Instruction data at scale: mixture, deduplication, decontamination |
| 3 | Judge calibration: making a judge agree with a small human sample |
| 5 | Reward shaping for long-horizon agent tasks |
| 6 | Reward hacking in agent settings (as opposed to single-turn) |
| 8 | Staleness and weight synchronisation, with the throughput accounting |
| 8 | Evaluation variance: multiple seeds and confidence intervals |
| 8 | Regression suites for RL pipelines |

## How this map grows

Three rules keep it useful:

1. **One line per paper, written when it is filed.** A paper is filed by adding its line to a shelf - title, ID, one clause on what it settles. If the clause will not fit in one sentence, the paper has not been understood well enough to file yet.
2. **Every ID carries the date it was checked.** An ID recalled from memory can point at an unrelated paper, and a wrong citation is worse than a missing one. All 76 rows here were checked against the arXiv API - the 73 filed on 2026-10-09, and the three added on 2026-10-14; a row added later carries its own date. See [the citation rules](conventions.md).
3. **A new paper is filed in the same commit that mentions it.** A paper discussed in a chapter but missing from this page is a paper that will be filed twice, or not at all.

**The one number to keep honest.** The source course's own map counts 260+ papers. This page holds 76, every one of them checked, and it is meant to be filled in over the years. A smaller map you can trust beats a larger one you cannot check - and the empty slots above say exactly where the next ones go.

**Titles, labels and links.** The titles here are the papers' own, shortened where a subtitle adds nothing; the ID is the pointer, and pasting it into arXiv lands on the paper. A paper that also appears in a chapter's Reading list carries the same ID there, and the "used in" line says which chapters lean on it.

## Change log

- **2026-10-14** - Chapter 7 cites three papers on programmatic task and environment synthesis, so they are filed here and the shelf-7 slot named "Programmatic task and environment synthesis" is filled: 76 papers in 42 slots, 7 empty. The three IDs were checked against the arXiv API on the same date. Four rows already on the map gained Chapter 7 in their "used in" line (LIMA, SFT Memorizes RL Generalizes, Spurious Rewards, ReTool).
- **2026-10-09** - page created: 73 papers filed on eight shelves, 41 slots filled, 8 named as empty. All 69 IDs already cited in the chapters were re-checked against the arXiv API, and 4 were added and checked for the first time (OpenClaw-RL, RLAnything, and the two self-distillation papers it builds on).

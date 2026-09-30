# Extra Modules (Optional)

> **Compute tier**: ✅ | **Status**: stub

Four gaps that sit outside the main track. One or two lessons each, read on demand; none of them blocks the main line.

## 1. Environment-side safety

Reward hacking is usually told as "the model fooled the judge". Here the story is "the model exploited the *world*":

- What kind of policy do you train when prompt injection has poisoned the training environment?
- Sandbox escapes and tool calls that exceed their authority
- Reward functions hijacked by environment bugs — reward hacking, systems edition
- Defences: environment auditing, least privilege, injection detection, isolating training data

## 2. Multi-agent RL

- Designing cooperative rewards when agents have different roles
- Credit assignment across roles: who owns the failure?
- When multi-agent beats a single agent, and when it does not

## 3. Domain agents: software engineering and computer use

- Why SWE is the most mature domain (verification comes free: do the tests pass?)
- Where a GUI agent's verifiability comes from
- How reward design and environment construction differ from general tasks

## 4. Context management as a policy

- Memory, compression, summarisation and retrieval are themselves decisions worth training
- How this divides labour with the external-skill approach in [Chapter 12](../12-skills/index.md)
- What long contexts do to rollout cost

## Deliverable

One page per module, answering a single question: **does this affect the task I am working on, and should it be my next step?**

## Backfill

- [ ] Question 2: How do I design the reward? (Safety module)
- [ ] Question 3: How do I debug a long multi-turn failure? (Multi-agent, context management)

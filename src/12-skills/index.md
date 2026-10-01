# Chapter 12 · Skills: Capability Libraries and Self-Evolution

> **Compute tier**: ✅✅ (no weight updates involved — this is the Spark's home turf) | **Status**: stub
> **Project**: [Project 6 · Memento-Skills](../appendix/projects.md)
> **Structure**: the main track (8 lessons) is engineering you can ship. The optional lessons (3) are unsettled research, clearly marked as such.

## First, the gist

So far, "getting better" meant changing the model's weights. There is another way: keep the model frozen and let it accumulate **skills** — reusable, auditable pieces of procedure and code — in a library outside the model, then teach it to fetch the right one at the right moment. That is cheaper, reversible, and inspectable. The hard parts are retrieval (finding the right skill) and governance (knowing which version is live). This chapter covers both, then asks the decision question: which capabilities belong in the weights, and which should stay outside?

## What you'll be able to do

- Say what counts as a skill, and where the boundary is between a skill, a tool and a prompt.
- Build retrieval over a skill library and measure whether it works (not just "it feels better").
- Version and audit skills so a bad edit can be rolled back.
- Decide, per capability, between training it into the weights and keeping it external.

## Lessons

- [ ] 1. What an agent skill actually is: five shapes and five boundaries
- [ ] 2. A problem/solution map: seven pain points against eight families of fixes
- [ ] 3. Skill retrieval: routing, skill graphs, and search at scale
- [ ] 4. Skills are code, the LLM is the processor: versioning and auditability
- [ ] 5. Project 6a: installing, configuring and verifying Memento-Skills
- [ ] 6. Project 6b: the core loop on a real task — read → execute → reflect → write
- [ ] 7. Project 6c: authoring your own skill — better descriptions, scripting, stable execution
- [ ] 8. The decision framework: data, latency, scenario, ecosystem — **train it into the weights, or keep it outside?**

## Optional: the research zone

*(Lessons 9 to 11 are research results, not settled engineering — the same warning this chapter's metadata carries. Read them as a roadmap, not as instructions.)*

- [ ] 9. SkillRL: hub-and-spoke co-evolution and its four paths
- [ ] 10. EvoSkill / Trace2Skill / SkillClaw: three ways to put guardrails on a self-evolving agent
- [ ] 11. A combined exercise: a daily AI-news assistant (fetch, evaluate, write a digest)

## Deliverable

- A long-running skill-evolution service, with a versioned skill library
- A retrieval-quality report (a behaviour-aligned router versus a semantic-search baseline)
- A written record of one "internalise vs externalise" decision using the four criteria

## How you know you passed

- The service runs continuously, and every skill creation or edit is auditable as a diff with a version.
- You can quote a retrieval number (recall@k or hit rate) rather than an impression.
- For one specific skill, you can explain why you chose *not* to train it into the weights.

## Reading

- Memento-Skills (read-execute-reflect-write loop, base model frozen): arXiv:2603.18743
- EvoSkill (separates create from edit; uses an elite pool to bound growth): arXiv ID to verify
- Trace2Skill (distils trajectory-local lessons into transferable skills): arXiv ID to verify
- SkillClaw (collective skill evolution across users): arXiv ID to verify

## Backfill

- [ ] Question 1: When should I reach for RL at all? (Here the answer is sometimes "I should not".)

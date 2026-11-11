# Agent RL Explained: A Beginner's Handbook

A plain-English study handbook for **Agent RL** — using reinforcement learning to train models that take several steps (search, call tools, run code) instead of answering in one shot.

The book is not a course transcript and is not affiliated with any vendor. It takes the topic list of a professional Agent RL course and re-orders it so a beginner can follow: every acronym is explained at first use, plain language always comes before a formula, and every method gets a "what it costs" sentence.

**Status: complete draft, early access.** All fifteen units have their prose — thirteen chapters, the electives page and the appendices — and nothing is a skeleton. Chapters 1–4 argue from published work and state no hardware measurement; Chapters 5–12 carry measured numbers, each with the run that produced it. What the book does not yet claim is named per unit in [ROADMAP.md](ROADMAP.md).

## The four questions this book answers

1. **When should I reach for RL at all?** (Supervised fine-tuning, DPO, GRPO, PPO, skills and evaluation each fit a different stage.)
2. **How do I design the reward?** (Verifiable rewards, process rewards, LLM-as-judge, mixed rewards — and how to stop the model cheating them.)
3. **How do I debug a long multi-turn failure?** (Was it planning, the tool call, the environment feedback, the summary, or the credit assignment?)
4. **Where does my training system quietly lie to me?** (Rollouts, masks, PRM judges, async training, weight sync, stale samples.)

Every chapter ends by asking which of these four it answered. They are also the book's acceptance test.

## Who it is for

You can write basic Python and you have met a neural network before (at least the idea that training changes numbers in a big table). You have used an LLM, maybe fine-tuned one with LoRA, and you keep meeting words like PPO, GRPO, DPO, reward model, rollout. If you have never touched Python or PyTorch, do that first — this book will not teach you to program.

## Reading it

```bash
direnv allow          # once; the pinned toolchain then loads automatically
mdbook serve --open   # live preview at http://localhost:3000
mdbook build          # writes the static site to book/
nix build             # fully reproducible build; result/ points at the site
```

The toolchain (mdBook 0.5.4, git, ripgrep, fd, jq) is pinned by `flake.nix`, so the environment is identical on every machine. Nix and direnv are the only prerequisites; without them, any mdBook >= 0.5 will also do.

## Structure

| Part | Chapters | Theme |
|---|---|---|
| Part 0 · Before you start | 3 lessons | What problem are you solving, and with what hardware |
| 1 · A shared language | 1–3 | The big picture, MDPs, value functions and policy gradients |
| 2 · Signals | 4–6 | Alignment algorithms, reward engineering, credit assignment |
| 3 · Making it industrial | 7–9 | Data and tasks, environments and rollouts, the training runtime |
| 4 · End-to-end | 10–11 | Two full projects, and how to prove anything improved |
| 5 · Frontier and judgement | 12–13 | Skills without retraining, and a map of 260+ papers |

Six hands-on projects hang off the chapters: Verifiers, Reagent, VeRL, DeepAnalyze, OpenClaw-RL and Memento-Skills. See `src/appendix/projects.md`.

Every chapter is labelled with a **compute tier** measured against a DGX Spark (GB10, 128 GB unified memory, aarch64 + sm_121): ✅ comfortable, ⚠️ shrink it, ❌ needs rented hardware. The intended split is: small models and services run locally, heavy multi-GPU experiments are rented by the hour.

## Repository layout

```text
book.toml      mdBook configuration (English, MathJax, search, navy theme)
flake.nix      pinned toolchain + reproducible site build
.envrc         direnv entry point
src/           the book: 13 chapters, an extras module, and 5 appendices
src/index.md   home page (start reading here)
AGENTS.md      rules for coding agents working in this repository
```

## Publishing

Every push to `main` builds the site and publishes it to GitHub Pages (`.github/workflows/pages.yml`), running `scripts/verify.sh` first — the same gate you run locally — so a page that breaks the rules never reaches the site.

The site is served at **https://yusiwen.cn/agent-rl-handbook/** .

## The companion repository

Every measured number in Chapters 5–12 is recomputed from a public companion repository: **<https://github.com/yusiwen/agent-rl-handbook-projects>**, pinned in this build at commit `19152ea`. It contains the six projects (P1–P6), the fixtures the numbers derive from, and the gate that re-derives each claim:

```bash
git clone https://github.com/yusiwen/agent-rl-handbook-projects
cd agent-rl-handbook-projects
uv run --frozen bash scripts/verify.sh
```

The book is not a course transcript and the repository is not a submission box: the code is there so a reader can check a number rather than trust it.

## Working on it

- Currently **everything is in English**, and the build enforces it: `scripts/verify.sh` rejects any character that is not English — that is, anything outside ASCII and Latin-1 except the typographic marks the book itself uses (em dash, arrows, the ✅/⚠️/❌ tier marks). Accented Latin letters (`é`, `ü`, `ñ`) are allowed, so no contributor gets tripped up by a name. If you would like to help with a translation, open an issue first: the toolchain has to change before translated text can be merged.
- **Written for a beginner.** The full rule set lives in `src/appendix/conventions.md`; contributor and agent rules live in `AGENTS.md`.
- Claims are labelled honestly: a textbook fact, a paper's reported result ("reported, not reproduced"), or a judgement — and unverified arXiv IDs are marked "ID to verify".
- Before finishing a change, run the repository's gate: `nix develop --command bash scripts/verify.sh` (build with no warnings, English-only, no remote assets, figure colours from the theme).

## License

Copyright (c) 2026 Siwen Yu. **The text, the figures and the code listings in this book are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** — the full legal code is [LICENSE](LICENSE). You may read, quote, adapt and redistribute them with attribution, for non-commercial purposes; an adaptation has to carry the same license.

Two boundaries are worth stating plainly, because the book ships next to a repository:

- **The companion code repository is MIT, not this license.** `github.com/yusiwen/agent-rl-handbook-projects` is software: MIT, usable commercially. A listing as it appears *in a chapter* is covered by the CC license above; the same code *in the repository* is MIT.
- **Third-party material keeps its own terms.** Papers, quotations and reproduced figures belong to their authors. The book cites them; it does not relicense them.

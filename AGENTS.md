# AGENTS.md — agent-rl handbook

Instructions for coding agents working in this repository. Keep this file short and pointer-based: rules that live in `src/appendix/conventions.md` must not be duplicated here.

## What this repository is

A personal study handbook for Agent RL, written as an mdBook site. It is **not** a course transcript and is not affiliated with any course or vendor: a course outline was used only to decide the order of topics. Nothing from any course's paid material may be copied in.

The book is organised around four questions, which double as its acceptance test:

1. When should I reach for RL at all?
2. How do I design the reward?
3. How do I debug a long multi-turn failure?
4. Where does my training system quietly lie to me?

## Non-negotiables

1. **English everywhere.** Every file in this repository is English: book pages, `book.toml`, `flake.nix`, `.envrc`, `.gitignore`, commit messages, code comments. **No Chinese characters may exist anywhere**, including the full-width punctuation in the CJK compatibility range (U+FF00–U+FFEF, for example the full-width vertical bar U+FF5C) — that character once slipped in as a table separator and broke this rule. Run the check below before finishing any change.
2. **Written for a beginner.** The reader can write basic Python and has seen a neural network, but knows no RL. Explain every acronym at first use, put plain language before any formula, keep one idea per sentence, and give every method a "what it costs" clause. Full rule set: `src/appendix/conventions.md`.
3. **Honesty labels.** Every claim is a textbook fact, a paper result (marked "reported, not reproduced"), or a judgement of mine. arXiv IDs that were not checked by hand are marked "ID to verify". Measured numbers are meaningless without the hardware and configuration that produced them — always state both.
4. **Do not silently rewrite facts.** When something changes (a framework gains `sm_121` support, for example), edit the sentence and keep a clause saying when it changed.

## Layout

```text
README.md                 reader-facing intro: what this is, how to build it, license
LICENSE                   MIT (the house default across this user's repositories)
book.toml                 mdBook config: language=en, search, fold, navy theme, site-url for Pages
mathjax/                  vendored MathJax 3 (SVG output) + this book's config — no CDN, works offline
src/figures/              hand-authored SVG figures, included inline so they follow the theme
css/figures.css           figure wrapper + caption styling (additional-css)
scripts/verify.sh         the gate: build with no warnings, English-only, no remote assets
.github/workflows/pages.yml   build + publish to GitHub Pages on every push to main
flake.nix / flake.lock    pinned toolchain (mdbook 0.5.4) + reproducible site build
.envrc                    direnv: NIX_CONFIG experimental-features + `use flake`
src/SUMMARY.md            the table of contents — keep in step with the home-page map
src/index.md              home page: the four questions, book map, tiers, progress list
src/start/                Part 0 (the opening part)
src/01-landscape/ … src/13-papers/   one chapter per directory, each src/NN-slug/index.md
src/electives/            optional extra modules
src/appendix/             project index, compute tiers, 16-week plan, writing rules, glossary
book/  result/  .direnv/  generated or local — git-ignored, never commit
```

## Locked book design

These decisions were made deliberately; do not "fix" them back:

- **13 chapters.** Chapters 7 (data/task engineering), 8 (environment engineering) and 11 (evaluation) are additions that many courses lack — they answer "I changed the algorithm and nothing improved" and "did it actually get better?".
- **PPO appears once.** The objective is explained in Chapter 4 only. Chapter 3 stops at GAE; Chapter 9 covers runtime implementation, never the objective again.
- **One project rhythm.** Every chapter with a project uses: problem/paper → architecture → engineering. A "project overview" lesson appears only once per chapter.
- **Chapter 12 is split.** Main track (shippable skill-library engineering) plus an optional research zone (SkillRL, EvoSkill, Trace2Skill, SkillClaw) that must stay labelled as unsettled.
- **Six projects, fixed mapping.** 1 Verifiers→Ch5, 2 Reagent→Ch6, 3 VeRL→Ch9, 4 DeepAnalyze→Ch10, 5 OpenClaw-RL→Ch10, 6 Memento-Skills→Ch12. See `src/appendix/projects.md`.
- **Compute tiers** reflect a DGX Spark (GB10, 128 GB unified, aarch64 + sm_121): ✅ comfortable, ⚠️ shrink it, ❌ needs rented hardware.

## Chapter skeleton (fixed)

```text
# Chapter N · Title
> **Compute tier**: … | **Status**: …
## First, the gist
## What you'll be able to do
## Lessons            (- [ ] checklist)
## Project            (only if one hangs off this chapter)
## Deliverable
## How you know you passed
## Reading
## Backfill           (which of the four questions did this answer?)
```

## Toolchain and commands

The toolchain is pinned by the Nix flake; `mdbook` is **not** assumed to exist system-wide.

```bash
direnv allow            # once, then the shell loads automatically in this directory
mdbook serve --open     # live preview at http://localhost:3000
mdbook build            # writes the site to book/
nix build               # reproducible build of the whole site; result/ points at it
nix flake update        # only when intentionally bumping nixpkgs (rewrites flake.lock)
```

mdBook is pinned to >= 0.5 on purpose: 0.5 renamed the `book.toml` key `curly-quotes` to `smart-punctuation`, and this book uses the 0.5 spelling. Pinning an older nixpkgs breaks the config.

## Verification before finishing any change

```bash
# everything the repository insists on, in one command - this is what CI runs
nix develop --command bash scripts/verify.sh

# what changed, for review
git status --short && git diff --stat
```

`scripts/verify.sh` runs `mdbook build` and fails on any `WARN` line, then checks: no CJK anywhere in the sources, no remote asset in the built site, and no hardcoded colour in `src/figures/` (only the accent classes and `var(--x, #fallback)` defaults are allowed). It needs mdbook and ripgrep, i.e. the flake dev shell. A build that succeeds says nothing about rendering: for figures and maths, open a page and look.

The build must print no `WARN` line. mdBook's two HTML warnings (`unclosed HTML tag`, `Saw EOF in state Comment`) both mean a figure got wrapped wrong or contains a blank line.

Report the exact commands you ran and their results. If a check could not run, say so instead of implying success.

**A successful build says nothing about rendering.** The failure mode is invisible to mdBook: if the maths renderer does not load, the page still builds and simply displays raw `$...$`. For any change that touches a formula or `book.toml`, open a page that contains maths and confirm it typesets — the renderer is local, so this works with the network off.

## Commits

- Conventional Commits, English, imperative subject: `docs(book): …`, `build(nix): …`, `chore: …`.
- One logical change per commit; stage first and show `git diff --stat` when the user wants to review.
- There is no remote configured yet; do not push.
- `book/`, `result/` and `.direnv/` are ignored — if a generated file shows up as untracked, the ignore rules are wrong, not the file.

## Pitfalls already paid for

- **Full-width punctuation is easy to introduce by accident** when writing tables and headers; the CJK check above catches it.
- **mdBook 0.5 config keys differ from 0.4** (see the pinning note above).
- **`nix build` writes `result`** into the repository root — it is a symlink into `/nix/store`, ignored by git, and safe to delete.
- **In a git repository, flakes only see files in the git index**: new chapter files must be `git add`-ed before `nix develop` / `nix build` can see them.
- **A new chapter means three edits**: the file, `src/SUMMARY.md`, and the book map table on `src/index.md`.
- **A one-page part must not be nested under a part heading.** mdBook then prints the name twice (sidebar and breadcrumbs showed `Part 0 · Before you start » Part 0 · Before You Start`). List it as a plain link before the first `# Part …` heading, the way Part 0 is listed.
- **Never depend on a CDN for the maths.** mdBook's `mathjax-support` injects MathJax 2.7 from cdnjs and lets it fetch its extensions lazily; when those requests are blocked or offline, the page silently shows raw LaTeX with no error anywhere in the build log. The renderer is therefore vendored in `mathjax/` (single SVG bundle, no font or extension downloads) and loaded via `additional-js`, with `mathjax-support = false`.
- **No maths in headings or link text.** The sidebar is rendered in the browser from `toc-*.js` after load, so heading maths shows as dollar notation there even when the body renders; `config.js` re-typesets the sidebar on load as a safety net, not as a licence to write formulas in titles.
- **A figure is not a markdown element.** `<svg>` is not in CommonMark's block-tag list, and a blank line inside the file ends the HTML block early: an included SVG must be blank-line-free and wrapped in `<figure class="book-figure">`, or mdBook warns (`unclosed HTML tag`, `Saw EOF in state Comment`). Colour must come from the theme variables (`--fg`, `--quote-bg`, `--quote-border`, `--sidebar-fg`, `--table-alternate-bg`) and the file must be **included, not referenced as an image** — CSS variables do not cross into `<img>`. Full rules: `src/appendix/conventions.md`.
- **Never add a `CNAME` file to this repository.** The custom domain `yusiwen.cn` belongs to the user site (`yusiwen.github.io`), and GitHub automatically serves every project site of that account under it, so this book is published at `https://yusiwen.cn/agent-rl-handbook/` with no domain configuration here. A `CNAME` file (or `cname` in `book.toml`) would claim the apex domain for this project and fight the user site. If the book ever moves to its own subdomain, that is the moment to set `cname` in `book.toml` and add the DNS record — not before.
- **`site-url` in `book.toml` must equal the repository path.** Everything else in the site uses relative links, so a stale value is invisible until a reader hits a missing page: the 404 page's home link then points at a path that does not exist (it was `/agent-rl/` while the repository is `agent-rl-handbook`). After renaming a repository, change this line in the same breath.

## Keeping this file current

Update `AGENTS.md` when the language rule, the chapter skeleton, the toolchain, or a locked design decision changes. Move anything longer than a few lines into `src/appendix/conventions.md` and leave a pointer here.

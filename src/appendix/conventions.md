# Writing and Formatting Rules

This handbook is maintained over a long period, so consistency matters more than style. Two rules are non-negotiable: **everything is written in English**, and **everything is written for a beginner**.

## Chapter skeleton (fixed)

```text
# Chapter N · Title
> Compute tier / status / project / special notes (blockquote)
## First, the gist
## What you'll be able to do
## Lessons            (- [ ] checklist, tick as you go)
## Project            (if one hangs off this chapter)
## Deliverable
## How you know you passed
## Reading
## Backfill           (which of the four questions did this answer?)
```

**The section names are fixed; extra sections are allowed after them.** A chapter that needs a split — Chapter 12's shippable main track against its unsettled research zone, for example — labels the boundary inside `## Lessons`, or gives the second half a section of its own *after* it. Renaming a standard section (Chapter 12's was once `## Main track`) makes one chapter navigate differently from the other twelve and drops that page out of the skeleton check.

The `## Lessons` checklist item and its `### Lesson N` heading must open with the same clause. The item may add a parenthetical promise — a payoff, or what the lesson sets up — but the heading stays short. A reader who ticks item 4 should land on a section that is recognisably item 4.

**Titles and roadmaps are labels, not uses.** A chapter title, or a line that deliberately stays in plain words ("a trial-and-error problem" rather than "an MDP"), may name an acronym before the section that defines it. Every technical use after that definition point must have the expansion behind it.

**Two terms are book-level and are expanded once for the whole book**: `RL` (reinforcement learning), on the home page, and `LLM` (large language model), at its first technical use in Chapter 2. Chapters do not repeat those two expansions; every other acronym is expanded at its first use in each chapter, and re-expanded in a chapter that leans on it for the first time.

## Plain-language rules

The audience is a beginner with basic Python and some fine-tuning experience. That means:

1. **Explain before you use.** Every acronym gets its plain meaning on first use in a chapter: "the reward model (a second model trained to score answers)".
2. **One idea per sentence.** If a sentence contains "however, because, which means", split it.
3. **No naked formulas.** Show the formula only after saying in words what it does, and add one sentence explaining each symbol.
4. **Concrete first, abstract second.** A toy example (a two-step task, a single tool call) before the general case.
5. **Say what it costs.** Every method gets a "what it costs" half-sentence — this is what beginners are missing most.
6. **Keep English technical terms in English.** Do not translate `rollout`, `checkpoint`, `loss mask` — translated variants make searching and matching against papers impossible.
7. **Analogies are welcome, but only one per concept**, and they must not contradict the mathematics.
8. **Say where a name comes from**, in the lesson where it first appears, whenever the name is not self-explanatory: REINFORCE (Williams, 1992), the "group" in GRPO, the "direct" in DPO, GAE. One clause is enough. A name a reader cannot decode is a name they cannot search for — the Part 1 review found REINFORCE used five times with no explanation of what the word means.
9. **Counts are claims.** "Roughly 100 lessons" is a claim about a number nobody counted. Count it and write the number ("100 lessons in total"). The same applies to "six parts", "five lessons", "260+ papers": if it is written down, it was counted.

## Terminology

| Use | Do not use |
|---|---|
| Tool-Call RL | Toolcall RL / ToolCall RL |
| RLHF, RLAIF, RLVR (spell out on first use) | invented abbreviations that match no paper |
| Reagent-U (define on first use: U = Unified Feedback Integration) | Reagent U / REAGENT-U |
| PPO, GRPO, DPO | variant names of your own invention |
| rollout, worker, checkpoint, verifier | home-made translations of them |
| loss mask | "loss filter", "training mask" |
| turn, episode, trajectory | "round" for turn; "session" for episode (a session is not an episode) |
| advantage, critic, reference model | translated variants; "teacher model" for reference model |
| pass@k | invented spellings such as "pass rate at k" |
| reward hacking | "reward cheating", "gaming the metric" |
| KL (Kullback-Leibler) | expanding it inconsistently, or leaving it unexplained on first use |

## Citations and honesty

- Papers: give the title plus the arXiv ID. **If the ID has not been checked by hand, mark it "ID to verify".**
- **A verified ID is verified once.** When a paper is cited in more than one chapter, reuse the entry and say where it was verified (for example "verified in Chapter 1's Reading, 2026-09-30") instead of checking it again. Re-check only when the citation itself changes.
- Claims come in exactly three kinds, and must not be blended:
  - **Textbook fact** — settled, textbook-level (for example the Bellman equation derivation)
  - **Paper result** — cite it, and mark "reported, not reproduced"
  - **My judgement** — label it as a judgement, and give the counterexample or boundary
- Measured numbers (throughput, memory, wall-clock) are invalid unless the hardware and configuration are stated with them.

## Maths

- Inline: `$...$`. Display: `$$...$$`.
- MathJax is **vendored in `mathjax/`** and loaded through `additional-js`; `mathjax-support` in `book.toml` stays **off**. mdBook's built-in support injects MathJax 2.7 from cdnjs, which fetches its extensions lazily — if those requests are blocked, the page silently shows raw LaTeX. The vendored SVG bundle is one file with no subresources, so the site renders offline.
- **No maths in headings, sidebar entries or link text.** The sidebar is built in the browser after load, so a formula there reads as raw dollar notation. Write the words in the title (`value, Q-value and advantage`) and put the symbol in the sentence.
- Consistent symbols: policy $\pi_\theta$, return $G_t$, value $V^\pi(s)$, advantage $A^\pi(s,a)$, discount $\gamma$, KL coefficient $\beta$.
- Keep the delimiters plain: `$...$` outside code spans. A literal dollar sign is written `\$`.

## Figures

Figures are **hand-authored SVG files in `src/figures/`**, named `NN-slug.svg` after the chapter that owns them, pulled into a page with mdBook's include directive and wrapped in a `<figure>`:

```markdown
<figure class="book-figure">{{#include ../figures/01-family-choice.svg}}</figure>

*Figure 1.1 — choosing a family. Each question you answer "no" moves you one step down a more expensive ladder.*
```

Four rules, all of which cost a broken build when ignored:

1. **Colour comes from the theme variables**, never from literals: `--fg` for text, `--sidebar-fg` for muted text, `--quote-bg` for cards, `--quote-border` for card outlines and connectors, `--table-alternate-bg` for the secondary card shade. One file then serves every theme, including the reader's own theme switch. Use `--quote-border`, **not** `--table-border-color`: in the navy theme the table border is *darker* than the card fill, and the outline vanishes.
2. **Include the file, never reference it as an image.** CSS custom properties do not cross into `<img>` or `<object>`, so a figure referenced as an image loses the theme entirely. Inline it.
3. **No blank lines inside the file.** `<svg>` is not one of CommonMark's block tags, so the wrapper `<figure>` is what makes the fragment a single HTML block; a blank line inside ends that block early and mdBook warns about unbalanced HTML (or reports `Saw EOF in state Comment`). Also keep the viewBox only as wide as the drawing needs — the page scales a figure to the content column (~750 px), so a wide viewBox shrinks the type.
4. **Labels are plain ASCII text, and no LaTeX inside a figure.** MathJax does not enter SVG; write `V(s)`, `A(s,a)` or words instead. The only literal colours allowed are the semantic accent bars (`.fig-a-cheap`, `.fig-a-mid`, `.fig-a-dear`), mid-tones chosen to read on both themes.

Captions are ordinary italic markdown on the line after the figure, numbered `Figure <chapter>.<n>`. `css/figures.css` styles the wrapper and the caption.

## Naming and links

- Files and directories: English lowercase slugs with numeric prefixes (for example `09-training-runtime/index.md`).
- Cross-references: relative markdown links, never absolute paths.
- Checklists use `- [ ]`, flipped to `- [x]` when done — this is the book's progress-tracking mechanism.

## Keeping it honest over time

- Add new material to an existing chapter rather than opening a new one; if a new chapter really is needed, update `SUMMARY.md` and the map on the home page in the same change.
- When a fact changes (a framework gains sm_121 support, for example), **edit the sentence and keep one clause saying when it changed** — never delete silently.
- A term a chapter introduces joins the table above (or the [glossary](glossary.md)) **in the same commit**, with the variants not to use. The Part 1 review found ten terms in use — `loss mask`, `pass@k`, `critic`, `reference model`, `advantage`, `turn`, `episode`, `trajectory`, `KL`, `reward hacking` — and none of them registered anywhere, so a reader had nowhere to look them up. The review probe now reports a page that uses one of them without registering it.

### Making the reviews stick

Every finding from a review has to end up in one of three places, or it comes back:

1. **A rule on this page**, if the fix is a matter of judgement.
2. **A check that runs** — `scripts/verify.sh` in CI, or the review probe for the structural rules.
3. **A pitfall in the writing workflow**, if the cause was a mistake in the process rather than in the prose.

Three habits follow from that, and each of them has already paid for itself:

- **A rule that only exists as text does not survive.** "Expand every acronym at first use" was written into this page with Chapter 1, and the Part 1 review still had to add twelve expansions across the three chapters, because nothing ran it. The check does not have to be perfect; it has to exist and be run before the chapter is called done.
- **When a check and the text disagree, decide which one is wrong first.** Twice the checker was the buggy party: a lesson-count script that read the home-page map row by row, and an acronym scan that counted a chapter title quoted inside a link. "Fixing" the prose to satisfy a broken check would have made the book worse. A check that fires is a hypothesis, not a verdict.
- **Edit labels in pairs, and sweep rather than spot-fix.** Item ↔ lesson heading, status ↔ what the page actually contains, section name ↔ the fixed skeleton, page ↔ home-page map row, part label ↔ every other copy of it. Every structural defect found so far was one half of a pair updated without the other: a chapter's Lesson 3 heading drifted away from its checklist item when the item gained a parenthetical, and Chapter 12 dropped out of the skeleton check by renaming a standard section. A label that lives in one place usually lives in five (`SUMMARY.md`, the page H1, the home-page map, the progress list, the hygiene docs): grep the old string across the tracked tree and require zero hits before calling a rename done.

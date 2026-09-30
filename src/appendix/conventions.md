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

## Plain-language rules

The audience is a beginner with basic Python and some fine-tuning experience. That means:

1. **Explain before you use.** Every acronym gets its plain meaning on first use in a chapter: "the reward model (a second model trained to score answers)".
2. **One idea per sentence.** If a sentence contains "however, because, which means", split it.
3. **No naked formulas.** Show the formula only after saying in words what it does, and add one sentence explaining each symbol.
4. **Concrete first, abstract second.** A toy example (a two-step task, a single tool call) before the general case.
5. **Say what it costs.** Every method gets a "what it costs" half-sentence — this is what beginners are missing most.
6. **Keep English technical terms in English.** Do not translate `rollout`, `checkpoint`, `loss mask` — translated variants make searching and matching against papers impossible.
7. **Analogies are welcome, but only one per concept**, and they must not contradict the mathematics.

## Terminology

| Use | Do not use |
|---|---|
| Tool-Call RL | Toolcall RL / ToolCall RL |
| RLHF, RLAIF, RLVR (spell out on first use) | invented abbreviations that match no paper |
| Reagent-U (define on first use: U = Unified Feedback Integration) | Reagent U / REAGENT-U |
| PPO, GRPO, DPO | variant names of your own invention |
| rollout, worker, checkpoint, verifier | home-made translations of them |

## Citations and honesty

- Papers: give the title plus the arXiv ID. **If the ID has not been checked by hand, mark it "ID to verify".**
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

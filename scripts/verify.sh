#!/usr/bin/env bash
# The repository's verification gate.
#
# Run this before every "done": it builds the book and enforces the rules that make this
# repository what it is. .github/workflows/pages.yml runs exactly this script before it
# publishes, so a green run here is a green deployment there.
#
# Needs mdbook and ripgrep - i.e. the flake dev shell:
#     nix develop --command ./scripts/verify.sh
set -euo pipefail

cd "$(dirname "$0")/.."

log=$(mktemp -t agent-rl-build.XXXXXX.log)

echo "== 1. build (any WARN is a real defect) =="
mdbook build 2>&1 | tee "$log"
if grep -q ' WARN ' "$log"; then
  echo "FAIL: mdbook printed a warning (an unbalanced figure, a broken one-page part, a bad config key)" >&2
  exit 1
fi

echo
echo "== 2. English only: nothing outside ASCII and Latin-1, apart from the allowlist =="
# The book is written in English, so a character outside ASCII fails - unless it is an accented
# Latin letter or one of the typographic marks the book actually uses. The original check only
# covered the CJK ranges, which let other scripts through silently.
#   allowed: Latin-1 Supplement minus its two invisible codes (A1-AC and AE-FF: letters such as
#            e-acute, u-umlaut, n-tilde, and that block's punctuation), plus em dash, en dash,
#            ellipsis, arrows, box drawing, the status marks, approximation, and the Greek
#            letters this book uses as notation (gamma, lambda, delta, pi, theta, beta, tau -
#            a figure cannot typeset LaTeX, so its labels need the symbols directly).
#   not allowed: NBSP (A0) and soft hyphen (AD) - invisible, and nearly always pasted by mistake -
#            and every script outside Latin apart from those notation letters: Cyrillic, Arabic,
#            CJK and friends.
allowlist='—–…→↔►▲─│┌┐└┘✅⚠️❌≈γλδπθβτ'
if rg -n --hidden "[^\x00-\x7F\x{A1}-\x{AC}\x{AE}-\x{FF}${allowlist}]" \
     -g '!book/**' -g '!result/**' -g '!.direnv/**' -g '!.git/**' -g '!mathjax/**' . ; then
  echo "FAIL: character found outside ASCII and Latin-1 (and not on the allowlist above)" >&2
  echo "      In a table or heading this is usually full-width punctuation pasted by accident." >&2
  exit 1
fi
echo "clean"

echo
echo "== 3. the site must not fetch anything remote =="
# rg exits 1 when it matches nothing, which set -e would take for a failure.
count() { { rg -o "$1" book/ || true; } | wc -l | tr -d ' '; }
remote_js=$(count '<script[^>]*src="https?://[^"]*"')
remote_css=$(count '<link[^>]*href="https?://[^"]*"')
echo "remote scripts: $remote_js, remote stylesheets: $remote_css"
if [ "$remote_js" -ne 0 ] || [ "$remote_css" -ne 0 ]; then
  echo "FAIL: the built site references a remote asset - it must work offline" >&2
  exit 1
fi

echo
echo "== 4. figures: colour comes from the theme variables =="
if rg -n '#[0-9a-fA-F]{3,6}' src/figures/ | rg -v 'fig-a-|var\(--[a-z-]+, #' ; then
  echo "FAIL: a figure hardcodes a colour instead of using a theme variable" >&2
  exit 1
fi
echo "clean"

echo
echo "== 5. display maths: wrapped, and never touched by Markdown =="
# Markdown parses the source inside a $$ block as ordinary text, and an underscore pair such as
# \hat{r}_\theta ... \pi_\theta is then read as emphasis. The tag lands in the middle of the LaTeX and
# MathJax silently leaves that one equation unrendered - the page still looks right at a glance, and
# nothing in the build log says so. Hence a check rather than a habit, in two halves: the cause, and
# the symptom.
#   (a) every display block is wrapped in <div class="math-display">. Markdown passes an HTML block
#       through untouched, so inside the wrapper the cause cannot occur - which is the point of the
#       wrapper. This half catches a new block written without it.
#   (b) no tag sits immediately before a LaTeX command in the built page. This half catches anything
#       the wrapper does not cover, and would have caught the four equations lost in Chapter 4.
blocks=$(( $( { rg -N -o '^\$\$$' -g '*.md' src/ || true; } | wc -l | tr -d ' ') / 2 ))
# Counted on lines that are *only* the wrapper: the conventions page names this class in prose, and a
# check that counts a mention of a thing as the thing is a check that fails for the wrong reason.
wrapped=$( { rg -N -o '^<div class="math-display">$' -g '*.md' src/ || true; } | wc -l | tr -d ' ')
if [ "$blocks" -ne "$wrapped" ]; then
  echo "FAIL: $blocks display block(s) in src/ but $wrapped wrapped - wrap every \$\$ block in <div class=\"math-display\">" >&2
  exit 1
fi
if rg -n -g '*.html' -e '<(em|strong)>\\[a-zA-Z]{2,}' book/ ; then
  echo "FAIL: Markdown inserted a tag inside a maths block - MathJax leaves that equation unrendered" >&2
  exit 1
fi
echo "clean ($blocks display block(s), all intact)"

echo
echo "all checks passed"

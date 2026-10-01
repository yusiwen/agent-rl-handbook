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
echo "all checks passed"

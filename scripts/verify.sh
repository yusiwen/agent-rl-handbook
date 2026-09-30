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
echo "== 2. English only: no CJK anywhere in the sources =="
# Han + CJK punctuation + full-width forms. A full-width vertical bar (U+FF5C) once slipped in
# as a table separator; this check is what catches that class of accident.
if rg -n --hidden "[\p{Han}\x{3000}-\x{303F}\x{FF00}-\x{FFEF}]" \
     -g '!book/**' -g '!result/**' -g '!.direnv/**' -g '!.git/**' . ; then
  echo "FAIL: CJK characters found - this repository is English-only" >&2
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

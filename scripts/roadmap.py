"""Regenerate ROADMAP.md from what the repositories actually contain.

    python3 scripts/roadmap.py            # rewrite ROADMAP.md
    python3 scripts/roadmap.py --check    # fail if it is out of date

Why generated rather than written: a hand-kept checklist drifts, and it drifts in the direction of
optimism. This repository has already paid for that once - a "pending" item in a chat summary had
been fixed a day earlier, and nothing in the repository could have told anyone. So the facts here
are read, not remembered:

* the book's chapters, their declared status, their length and their lesson count, from `src/`;
* the companion projects' state, from whether they have code and how many tests and run records
  they have accumulated;
* the open decisions, which are the one part a script cannot derive, preserved verbatim between
  two markers and edited by hand.

The lesson checkboxes in the chapters are deliberately *not* used as a progress signal: they are a
plan of what a chapter will contain, and no chapter ticks them, so reading them would report every
written chapter as unstarted.

Exit status is non-zero under `--check` when the file on disk differs from what the sources imply,
so `verify.sh` can keep it current.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
PROJECTS_DEFAULT = BOOK.parent / "agent-rl-projects"
OUT = BOOK / "ROADMAP.md"

OPEN_BEGIN = "<!-- manual:open-decisions -->"
OPEN_END = "<!-- /manual:open-decisions -->"

DEFAULT_OPEN = """\
- A chapter is only `written` when every lesson has prose; `skeleton` means the outline is there and
  the prose is not. The statuses below are read from each chapter's own `Status:` line.
"""

_CHAPTER = re.compile(r"^#\s+(.+)$", re.MULTILINE)
# The status is not at the start of its line: chapters open with a compute tier and then a status,
# separated by bars, so the pattern has to find the label wherever the line puts it.
_STATUS = re.compile(r"\*\*Status\*\*\s*:\s*([A-Za-z-]+)")
_LESSON = re.compile(r"^-\s+\[[ x]\]\s+\d+\.", re.MULTILINE)
_WORDS = re.compile(r"\S+")


def _first(pattern: re.Pattern[str], text: str, default: str = "") -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else default


def chapters() -> list[dict]:
    rows = []
    for path in sorted(BOOK.glob("src/*/index.md")):
        text = path.read_text(encoding="utf-8")
        words = len(_WORDS.findall(text))
        status = _first(_STATUS, text, "unknown")
        lessons = len(_LESSON.findall(text))
        # A chapter with prose and a skeleton chapter differ by an order of magnitude in length; the
        # threshold is here so the table says what a reader would see, not what the status word says.
        if status == "stub" or words < 900:
            state = "skeleton"
        elif words < 2500:
            state = "partial"
        else:
            state = "written"
        rows.append(
            {
                "unit": path.parent.name,
                "title": _first(_CHAPTER, text, path.parent.name),
                "status": status,
                "state": state,
                "words": words,
                "lessons": lessons,
                "has_reading": "## Reading" in text,
                "has_figure": "![" in text or ".svg" in text,
            }
        )
    return rows


def projects(root: Path) -> list[dict]:
    rows = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p") and p.name[1].isdigit()):
        readme = directory / "README.md"
        code = sorted((directory / "src").rglob("*.py")) if (directory / "src").exists() else []
        tests = sorted((directory / "tests").rglob("test_*.py")) if (directory / "tests").exists() else []
        checks = sorted((directory / "checks").glob("*.py")) if (directory / "checks").exists() else []
        rows.append(
            {
                "name": directory.name,
                "state": "built" if code else "README only",
                "modules": len(code),
                "test_functions": sum(
                    len(re.findall(r"^\s+def test_", p.read_text(encoding="utf-8"), re.MULTILINE))
                    for p in tests
                ),
                "checks": len(checks),
            }
        )
    return rows


def run_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for header in (root / "runs").glob("*/header.json"):
        try:
            payload = json.loads(header.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        project = str(payload.get("project", "unknown"))
        counts[project] = counts.get(project, 0) + 1
    return counts


def manual_block(existing: str | None) -> str:
    if existing and OPEN_BEGIN in existing and OPEN_END in existing:
        inner = existing.split(OPEN_BEGIN, 1)[1].split(OPEN_END, 1)[0]
        return inner.strip("\n")
    return DEFAULT_OPEN.strip("\n")


def render(existing: str | None, projects_root: Path) -> str:
    chapters_rows = chapters()
    # Guarded, not just guarded-for-display: CI checks out the book alone, and an eager call here
    # crashed the gate on a directory that does not exist there.
    project_rows = projects(projects_root) if projects_root.exists() else []
    counts = run_counts(projects_root) if projects_root.exists() else {}

    written = sum(1 for row in chapters_rows if row["state"] == "written")
    skeleton = sum(1 for row in chapters_rows if row["state"] == "skeleton")
    total_words = sum(row["words"] for row in chapters_rows)

    lines = [
        "# Roadmap",
        "",
        f"> Generated by `scripts/roadmap.py` on {date.today().isoformat()} from this repository and",
        f"> `{projects_root.name}/`. Do not edit the tables by hand: edit the sources, or the open",
        "> decisions block, which is preserved verbatim between its markers.",
        "",
        "## The book",
        "",
        f"**{written} of {len(chapters_rows)} units have their prose written, {skeleton} are skeletons, "
        f"{total_words:,} words in total.** `written` means a long chapter with its lessons; "
        "`partial` means prose without the full lesson set; `skeleton` means the outline only.",
        "",
        "| Unit | Chapter | Status | State | Words | Lessons | Reading | Figure |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in chapters_rows:
        lines.append(
            f"| `{row['unit']}` | {row['title']} | `{row['status']}` | **{row['state']}** | "
            f"{row['words']:,} | {row['lessons']} | {'yes' if row['has_reading'] else 'no'} | "
            f"{'yes' if row['has_figure'] else 'no'} |"
        )

    if not projects_root.exists():
        # CI checks out the book alone. Regenerating from a missing directory would report every
        # project as unbuilt, so the table already in the file is kept verbatim instead - the local
        # run is the one that can see both repositories.
        if existing and "## The companion projects" in existing:
            kept = existing.split("## The companion projects", 1)[1].split("## Open decisions", 1)[0]
            lines += ["", "## The companion projects" + kept.rstrip(), ""]
            lines += ["## Open decisions", "", OPEN_BEGIN, manual_block(existing), OPEN_END, ""]
            return "\n".join(lines)
        return "\n".join(lines + ["", "## Open decisions", "", OPEN_BEGIN, manual_block(existing), OPEN_END, ""])

    lines += [
        "",
        "## The companion projects",
        "",
        "| Project | State | Modules | Test functions | Check scripts | Run records |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in project_rows:
        lines.append(
            f"| `{row['name']}` | **{row['state']}** | {row['modules']} | {row['test_functions']} | "
            f"{row['checks']} | {counts.get(row['name'], 0)} |"
        )

    lines += [
        "",
        f"Run records total {sum(counts.values())}; each names the code revision that produced it.",
        "",
        "## Open decisions",
        "",
        OPEN_BEGIN,
        manual_block(existing),
        OPEN_END,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regenerate ROADMAP.md from the repositories")
    parser.add_argument("--check", action="store_true", help="fail instead of writing")
    parser.add_argument("--projects", default=str(PROJECTS_DEFAULT))
    args = parser.parse_args(argv)

    projects_root = Path(args.projects).resolve()
    existing = OUT.read_text(encoding="utf-8") if OUT.exists() else None
    rendered = render(existing, projects_root)
    if args.check:
        if existing != rendered:
            print("FAIL ROADMAP.md is out of date; run: python3 scripts/roadmap.py")
            return 2
        print("ok ROADMAP.md matches the repositories")
        return 0
    OUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT.relative_to(BOOK)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

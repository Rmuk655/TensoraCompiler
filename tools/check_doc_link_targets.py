#!/usr/bin/env python3
"""Drift guard: every numbered cross-reference link must point at the
section its own visible text names, not merely at *some* existing
anchor. A markdown link like "[RM 3.6](...#37-matrix-multiplication)"
resolves cleanly for any plain link checker (the file and anchor both
exist), but it is silently wrong: the number in the link text (3.6)
does not match the number of the heading the link actually lands on
(3.7). This script catches that class of mistake, which
tools/check_test_matrix.sh's cousins (markdown-link-check, lychee,
etc.) cannot: they only prove a link resolves, never that it resolves
to the *right* place.

Scope: every ``.md`` file under docs/ plus the top-level README. Only
text that is already a markdown link ``[label](target)`` is checked;
a bare mention of "RM 3.6" in prose with no link is left alone (it was
never asked to point anywhere).

Recognized label conventions (the ones this project's docs actually
use):
  RM 3.6                 -> Language Reference Manual section 3.6
  Section 4 / Section 6a  -> a numbered section in the *same* file
  ADR ARCH-007 / ARCH-007 -> a Decision Record heading in DECISIONS.md
  INTERFACES SS4a         -> a numbered section in INTERFACES.md
  whitepaper SS5 / whitepaper Section 5 -> a numbered section in the
                             whitepaper

Registered as the CTest INT-DOCLINKS-SYNC (see cmake/Tests.cmake).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_ROOTS = [REPO_ROOT / "docs", REPO_ROOT / "README.md"]

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```")


def slug(text: str) -> str:
    """Reproduce GitHub's heading-anchor algorithm closely enough for
    this repo's headings: lowercase, drop anything that is not a word
    character/space/hyphen, then collapse whitespace runs to one
    hyphen each."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    return text


def iter_markdown_files():
    for root in DOC_ROOTS:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.md"))


def extract_headings(path: Path) -> dict[str, str]:
    """Return {anchor_slug: raw_heading_text} for one file, skipping
    fenced code blocks (a '#'-led line inside a ``` fence is a comment
    or shell prompt, not a heading) and de-duplicating repeated
    headings the same way GitHub does (second occurrence gets a
    '-1' suffix, and so on)."""
    headings: dict[str, str] = {}
    seen: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(errors="ignore").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        raw = m.group(2).strip()
        base = slug(raw)
        n = seen.get(base, 0)
        seen[base] = n + 1
        anchor = base if n == 0 else f"{base}-{n}"
        headings[anchor] = raw
    return headings


def heading_number(raw: str) -> str | None:
    """Pull the leading section number (e.g. "3.6", "4a", "6.2") off a
    heading like "3.6 Name and Scope Rules" or "4a. Analysis...". None
    if the heading has no leading number (e.g. "Abstract")."""
    m = re.match(r"^(\d+[a-z]?(?:\.\d+)?)\.?\s", raw)
    return m.group(1) if m else None


def decision_id(raw: str) -> str | None:
    m = re.match(r"^(ARCH-\d+)\b", raw)
    return m.group(1) if m else None


# Each rule: (label_pattern, expected-id extractor, target-id extractor,
# which file the target lives in). `same_file` means "whatever file the
# link itself is written in" (a local #anchor with no path).
RULES = [
    # "ADR ARCH-007" or a bare "ARCH-007" in the label
    (re.compile(r"\bARCH-(\d+)\b"), lambda m: f"ARCH-{m.group(1)}", decision_id, "docs/architecture/DECISIONS.md"),
    # "RM 3.6" / "RM Section 3.6"
    (re.compile(r"\bRM (?:Section )?(\d+\.\d+)\b"), lambda m: m.group(1), heading_number,
     "docs/language/03_LANGUAGE_REFERENCE_MANUAL.md"),
    # "INTERFACES SS4a" (section mark written as a literal '§')
    (re.compile(r"INTERFACES \u00a7(\d+[a-z]?)"), lambda m: m.group(1), heading_number,
     "docs/architecture/INTERFACES.md"),
    # "whitepaper SS5" / "whitepaper Section 5" / "whitepaper 8.3"
    (re.compile(r"whitepaper (?:\u00a7|Section )?(\d+(?:\.\d+)?)"), lambda m: m.group(1), heading_number,
     "docs/language/00_WHITEPAPER.md"),
    # "Section 4" / "Section 6a" with no file qualifier: same file only
    (re.compile(r"^Section (\d+[a-z]?)$"), lambda m: m.group(1), heading_number, None),
]


def resolve_target_file(current: Path, link_path: str) -> Path | None:
    if not link_path:
        return current
    if link_path.startswith(("http://", "https://")):
        return None
    return (current.parent / link_path).resolve()


def check_file(path: Path, cache: dict[Path, dict[str, str]]) -> list[str]:
    problems = []
    text = path.read_text(errors="ignore")
    # Strip fenced code blocks before scanning for links: a link-looking
    # string inside a ``` block is example text, not a real reference.
    lines = text.splitlines()
    in_fence = False
    kept = []
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            kept.append("")
            continue
        kept.append("" if in_fence else line)
    scannable = "\n".join(kept)

    for label, target in LINK_RE.findall(scannable):
        link_path, _, anchor = target.partition("#")
        target_file = resolve_target_file(path, link_path)
        if target_file is None:
            continue  # external URL: not this script's job
        if not anchor:
            continue  # whole-file link, no section claim to verify

        for label_re, expect_fn, actual_fn, required_file in RULES:
            m = label_re.search(label)
            if not m:
                continue
            if required_file is not None and target_file != REPO_ROOT / required_file:
                continue
            if required_file is None and target_file != path:
                continue
            if target_file not in cache:
                if not target_file.exists():
                    continue  # markdown-link-check's job, not this script's
                cache[target_file] = extract_headings(target_file)
            heading_text = cache[target_file].get(anchor)
            if heading_text is None:
                continue  # broken anchor: markdown-link-check's job
            expected = expect_fn(m)
            actual = actual_fn(heading_text)
            if actual is not None and actual != expected:
                problems.append(
                    f"{path.relative_to(REPO_ROOT)}: link labeled "
                    f"'{label}' points at "
                    f"{target_file.relative_to(REPO_ROOT)}#{anchor} "
                    f"(heading '{heading_text}', id '{actual}'), "
                    f"but the label says '{expected}'"
                )
            break  # first matching rule wins; don't double-report
    return problems


def main() -> int:
    cache: dict[Path, dict[str, str]] = {}
    all_problems: list[str] = []
    n_files = 0
    for path in iter_markdown_files():
        n_files += 1
        all_problems.extend(check_file(path, cache))

    if all_problems:
        print(f"Found {len(all_problems)} mismatched doc cross-reference(s):\n")
        for p in all_problems:
            print(f"  - {p}")
        print(
            "\nEach of these links resolves (the file/anchor both exist), "
            "but lands on a different numbered section than its own label "
            "claims. Fix the anchor, or fix the label, so they agree."
        )
        return 1

    print(f"OK: numbered cross-references in {n_files} Markdown file(s) "
          "point at the sections their labels claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

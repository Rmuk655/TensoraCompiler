#!/usr/bin/env python3
"""Regenerate docs/development/10_APPENDIX_SOURCE_LISTING.md.

The appendix is a mechanical concatenation of the real source tree
(see its own Section 10.10), not a hand-maintained document. It is
built fresh on demand and is not committed -- run this after any
change to a file it lists, or any time you need the listing:

    ./tools/regen_appendix.py

The committed input is docs/development/appendix_template.md: the same
document with every code fence left empty. It holds the parts that
cannot be derived from the source tree -- the prose, the Section
10.3-10.8 grouping, and the "### `path`" headers saying which file
belongs in which section. Everything else is filled in from disk here.

What it does:
  - Starts from the template, so the output never depends on a
    previous run's output, and the appendix can always be rebuilt
    from a fresh clone.
  - Fills every "### `path`" file dump (Sections 10.3-10.9) from the
    file's current content on disk, tagged with the language matching
    its extension.
  - Generates the Section 10.2 repository manifest (`find grammar
    include src tools examples -type f | sort`) and the Section 10.9
    test manifest (`find tests -type f | sort`).
  - Updates Section 10.10's base commit hash and generation date.
  - Reports, but does not silently fix, two situations that need a
    human decision: a manifest entry with no "### `path`" dump
    anywhere (a newly added file -- decide which of Sections 10.3-10.8
    it belongs in), and a "### `path`" dump whose file no longer
    exists on disk (a removed or renamed file -- decide whether to
    delete the section or update the path).

Fix both of those in the template, not in the generated appendix:
edits to the appendix are discarded on the next run. The grouping into
Sections 10.3-10.8 is a curated-by-topic breakdown (one section per
compiler phase), not something a manifest diff can infer on its own,
which is why it lives in a committed file.
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APPENDIX = REPO / "docs/development/10_APPENDIX_SOURCE_LISTING.md"
TEMPLATE = REPO / "docs/development/appendix_template.md"

LANG_BY_EXT = {
    ".h": "cpp",
    ".hpp": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".g4": "antlr",
    ".cmake": "cmake",
    ".txt": "text",
    ".sh": "bash",
    ".py": "python",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".md": "markdown",
    ".tns": "text",
}


def lang_for(path: str) -> str:
    if path.rsplit("/", 1)[-1] == "CMakeLists.txt":
        return "cmake"
    for ext, lang in LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return "text"


def find_sorted(*dirs: str) -> list[str]:
    out = subprocess.run(
        ["find", *dirs, "-type", "f", "-not", "-path", "*/__pycache__/*"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    return sorted(out.splitlines())


_CLOSE_FENCE_RE = re.compile(r"^`{3,}[ \t]*$", re.MULTILINE)


def fence_marker_for(content: str) -> str:
    """A backtick run one longer than any run already in `content`, so the
    fence itself can never be confused with a literal ``` inside content
    that itself talks about markdown fences (this script's own source,
    dumped into the appendix, is exactly that case)."""
    runs = re.findall(r"`+", content)
    longest = max((len(r) for r in runs), default=0)
    return "`" * max(3, longest + 1)


def replace_fenced_block(text: str, start: int, new_lines: list[str]) -> tuple[str, int]:
    fence_start = text.index("```", start)
    line_end = text.index("\n", fence_start)
    body_start = line_end + 1
    close_match = _CLOSE_FENCE_RE.search(text, body_start)
    fence_end = close_match.end()
    new_body = "\n".join(new_lines)
    marker = fence_marker_for(new_body)
    replaced_block = f"{marker}text\n{new_body}\n{marker}"
    replaced = text[:fence_start] + replaced_block + text[fence_end:]
    return replaced, fence_start + len(replaced_block)


def refresh_file_dumps(text: str, end_idx: int) -> tuple[str, int, list[str]]:
    body, rest = text[:end_idx], text[end_idx:]
    header_re = re.compile(r"^### `([^`]+)`.*$", re.MULTILINE)
    matches = list(header_re.finditer(body))
    missing_on_disk = []
    for m in reversed(matches):
        rel_path = m.group(1)
        full_path = REPO / rel_path
        if not full_path.is_file():
            missing_on_disk.append(rel_path)
            continue
        content = full_path.read_text().rstrip("\n")
        lang = lang_for(rel_path)
        marker = fence_marker_for(content)
        header_end = m.end()
        fence_start = body.index("```", header_end)
        line_end = body.index("\n", fence_start)
        close_match = _CLOSE_FENCE_RE.search(body, line_end + 1)
        body = body[:fence_start] + f"{marker}{lang}\n{content}\n{marker}" + body[close_match.end() :]
    return body + rest, len(matches), sorted(missing_on_disk)


def find_heading(text: str, heading: str) -> int:
    """Locate `heading` as a whole line, not a substring match. A plain
    `text.index(heading)` would also match this same heading text
    quoted inside this script's own docstring/source once this script
    dumps itself into the appendix (Section 10.8) -- anchoring to a
    full line rules that false match out, since the quoted occurrences
    are substrings of a longer source-code line, never a whole line by
    themselves."""
    m = re.search(r"^" + re.escape(heading) + r"$", text, re.MULTILINE)
    if m is None:
        raise ValueError(f"heading not found as a whole line: {heading!r}")
    return m.start()


def main() -> int:
    if not TEMPLATE.is_file():
        print(f"error: missing structure template {TEMPLATE.relative_to(REPO)}", file=sys.stderr)
        return 1
    text = TEMPLATE.read_text()
    # The template opens with an HTML comment saying what it is. That is
    # for whoever edits the template, not for the appendix.
    text = re.sub(r"\A<!--.*?-->\n", "", text, count=1, flags=re.DOTALL)
    end_idx = find_heading(text, "## 10.10 Mechanical Generation")
    text, n_dumps, missing_on_disk = refresh_file_dumps(text, end_idx)

    repo_manifest = [f"tensora/{p}" for p in find_sorted("grammar", "include", "src", "tools", "examples")]
    test_manifest = [f"tensora/{p}" for p in find_sorted("tests")]

    manifest_start = find_heading(text, "## 10.2 Repository Manifest")
    text, _ = replace_fenced_block(text, manifest_start, repo_manifest)

    test_manifest_start = find_heading(text, "## 10.9 Test Listing")
    text, _ = replace_fenced_block(text, test_manifest_start, test_manifest)

    end_idx = find_heading(text, "## 10.10 Mechanical Generation")
    dumped_paths = {m.group(1) for m in re.finditer(r"^### `([^`]+)`.*$", text[:end_idx], re.MULTILINE)}
    manifest_paths = {p.removeprefix("tensora/") for p in repo_manifest}
    undumped = sorted(manifest_paths - dumped_paths)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    text = re.sub(r"Base commit at generation time:\n`[0-9a-f]+`", f"Base commit at generation time:\n`{commit}`", text)
    text = re.sub(r"Generation\ndate: \d{4}-\d{2}-\d{2}", f"Generation\ndate: {date.today().isoformat()}", text)
    text = re.sub(r"Generation date: \d{4}-\d{2}-\d{2}", f"Generation date: {date.today().isoformat()}", text)

    APPENDIX.write_text(text)

    print(f"Refreshed {n_dumps} file dumps.")
    print(f"Repository manifest: {len(repo_manifest)} files. Test manifest: {len(test_manifest)} files.")
    if missing_on_disk:
        print(f"\nNEEDS ATTENTION -- {len(missing_on_disk)} dumped path(s) no longer exist on disk:")
        for p in missing_on_disk:
            print(f"  {p}")
    if undumped:
        print(f"\nNEEDS ATTENTION -- {len(undumped)} manifest path(s) have no ### dump in Sections 10.3-10.8 yet:")
        for p in undumped:
            print(f"  {p}")
    if missing_on_disk or undumped:
        print("\nPlace/remove these by hand in the right Section 10.3-10.8, then re-run this script.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

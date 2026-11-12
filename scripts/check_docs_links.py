"""Check relative Markdown links in this repository.

The repository has no mkdocs or markdown-link-check configuration, so this
script is the documentation-link gate. It walks every tracked Markdown file,
skips fenced code blocks and inline code, and resolves each relative link
against the file that contains it.

Generated trees are not link targets: ``output/`` is produced by the pipeline
and is absent from a fresh checkout, so links into it are reported as
``generated`` rather than missing. Links that leave the repository root are
external to this checkout and are skipped.

Exit codes:
    0: every relative link resolves inside the checkout
    1: at least one relative link points at a missing path
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".omo",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "output",
    }
)

#: Generated trees that exist only after a producer run. Their own producer and
#: checker own their contents; the link gate does not require them to be present.
GENERATED_PREFIXES = ("output/", "docs/_generated/")

INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def markdown_files(root: Path) -> list[Path]:
    """Return every Markdown file under *root*, excluding generated trees."""

    paths: list[Path] = []
    for path in root.rglob("*.md"):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        paths.append(path)
    return sorted(paths)


def _strip_non_links(text: str) -> str:
    """Blank out code fences, inline code, and HTML comments, keeping newlines."""

    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            kept.append("\n" if line.endswith("\n") else "")
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            kept.append("\n" if line.endswith("\n") else "")
            continue
        kept.append(line)
    text = "".join(kept)
    text = HTML_COMMENT.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    return INLINE_CODE.sub(lambda match: " " * len(match.group(0)), text)


def _target_path(raw: str) -> str | None:
    """Normalize one link target, or return None when it is not a local path."""

    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if not target or target.startswith("#") or SCHEME.match(target):
        return None
    target = target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
    if not target:
        return None
    return unquote(target)


def missing_links(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return ``(file, line, target, reason)`` for every unresolved link."""

    problems: list[tuple[Path, int, str, str]] = []
    for path in markdown_files(root):
        text = _strip_non_links(path.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = _target_path(match.group(1))
            if target is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            if target.startswith("/"):
                resolved = (root / target.lstrip("/")).resolve()
            else:
                resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue  # sibling checkout or other out-of-tree reference
            relative = resolved.relative_to(root.resolve()).as_posix()
            if any(
                relative == prefix.rstrip("/") or relative.startswith(prefix)
                for prefix in GENERATED_PREFIXES
            ):
                continue
            if not resolved.exists():
                problems.append((path, line, target, "missing"))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to scan (default: this checkout)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    problems = missing_links(root)
    scanned = len(markdown_files(root))
    if problems:
        for path, line, target, reason in problems:
            print(f"{path.relative_to(root)}:{line}: {reason}: {target}", file=sys.stderr)
        print(f"{len(problems)} broken relative link(s) across {scanned} files", file=sys.stderr)
        return 1
    print(f"relative links resolve across {scanned} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

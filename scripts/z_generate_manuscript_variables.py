"""Generate manuscript variables and resolved markdown for template rendering."""

from pathlib import Path
import os
import re
import sys

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.manuscript_variables import (
    compute_variables,
    write_variables,
)
from active_fractal_rabbit.render_freshness import write_resolved_render_fingerprint

ROOT = Path(__file__).resolve().parents[1]

TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
EXCLUDED = {"AGENTS.md", "README.md", "SYNTAX.md"}


def _template_root() -> Path | None:
    candidates = [
        os.environ.get("TEMPLATE_REPO_ROOT"),
        str(ROOT.parents[2] / "template"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "infrastructure").is_dir():
            return Path(candidate)
    return None


def _verify_tokens(variables: dict[str, str]) -> int:
    used: set[str] = set()
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in EXCLUDED:
            continue
        used.update(TOKEN_RE.findall(path.read_text(encoding="utf-8")))
    missing = sorted(used - set(variables))
    if missing:
        raise SystemExit(
            f"manuscript uses token(s) the generator does not emit: {missing}"
        )
    return len(used)


def main() -> int:
    variables = compute_variables(ROOT)
    used = _verify_tokens(variables)
    output = write_variables(ROOT)
    template_root = _template_root()
    if template_root is None:
        print(output)
        print(f"[ok] {used} manuscript token(s) generated; template injection skipped")
        return 0
    if str(template_root) not in sys.path:
        sys.path.insert(0, str(template_root))
    from infrastructure.rendering.manuscript_injection import (
        write_resolved_manuscript_tree,
    )

    resolved = write_resolved_manuscript_tree(ROOT, variables)
    fingerprint = write_resolved_render_fingerprint(ROOT, resolved)
    print(output)
    print(
        f"[ok] {used} manuscript token(s) generated; injected -> {resolved}; "
        f"render fingerprint={fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

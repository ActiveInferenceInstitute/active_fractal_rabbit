#!/usr/bin/env bash
# Single gate entry point for Active FractalRabbit.
#
# Every arm below is the exact command CI runs; CI calls this script instead of
# repeating the command strings. Arms are independent, so a fast local loop can
# run one arm without paying for the others.
#
#   ./run.sh lint       ruff static checks
#   ./run.sh test       fast suite (excludes the requires_fractalrabbit and slow markers)
#   ./run.sh test-full  documented gate: full suite with the 90% coverage floor
#   ./run.sh docs       relative-link check plus generated-timings contract
#   ./run.sh gnn        pinned GNN conformance gate (strict validation + tests)
#   ./run.sh all        lint, test, test-full, docs, gnn in that order
#   ./run.sh bench      re-measure every arm and rewrite docs/_generated/gate-timings.md
#   ./run.sh --list     print the measurable arm names
#
# There is deliberately no `types` arm: this repository configures no type
# checker (no mypy/pyright configuration or dependency), so adding one would be
# a new gate rather than an arm of the existing one. See docs/RUN_GUIDE.md.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

UV=(uv run --frozen --extra dev)
ARMS=(lint test test-full docs gnn)

usage() {
    cat <<'EOF'
Single gate entry point for Active FractalRabbit.

  ./run.sh lint       ruff static checks
  ./run.sh test       fast suite (excludes the requires_fractalrabbit and slow markers)
  ./run.sh test-full  documented gate: full suite with the 90% coverage floor
  ./run.sh docs       relative-link check plus generated-timings contract
  ./run.sh gnn        pinned GNN conformance gate (strict validation + tests)
  ./run.sh all        lint, test, test-full, docs, gnn in that order
  ./run.sh bench      re-measure every arm and rewrite docs/_generated/gate-timings.md
  ./run.sh --list     print the measurable arm names

There is deliberately no `types` arm: this repository configures no type
checker (no mypy/pyright configuration or dependency). See docs/RUN_GUIDE.md.
EOF
}

run_arm() {
    local arm="$1"
    shift
    case "$arm" in
        lint) "${UV[@]}" ruff check . ;;
        test) "${UV[@]}" pytest tests/ -m "not requires_fractalrabbit and not slow" --no-cov -q ;;
        test-full) "${UV[@]}" pytest tests/ --cov=src --cov-fail-under=90 ;;
        docs)
            "${UV[@]}" python scripts/check_docs_links.py
            # bench writes the report it would otherwise check; skip the
            # self-check only for that measurement pass.
            if [[ -z "${GATE_REPORT_BENCH:-}" ]]; then
                "${UV[@]}" python scripts/gate_report.py --check
            fi
            ;;
        gnn) "${UV[@]}" python scripts/check_gnn_conformance.py "$@" ;;
        bench)
            # The reporter invokes the other arms, which use `uv run`; running
            # it under `uv run` too would nest uv and deadlock on its cache
            # lock. Use the already-synced project interpreter instead.
            if [[ ! -x .venv/bin/python ]]; then
                echo "bench requires a synced .venv; run uv sync --frozen --extra dev first" >&2
                return 2
            fi
            .venv/bin/python scripts/gate_report.py "$@"
            ;;
        *)
            echo "unknown arm: $arm" >&2
            usage >&2
            return 2
            ;;
    esac
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    --help|-h)
        usage
        ;;
    --list)
        printf '%s\n' "${ARMS[@]}"
        ;;
    all)
        status=0
        for arm in "${ARMS[@]}"; do
            printf '\n== %s ==\n' "$arm"
            run_arm "$arm" || status=1
        done
        exit "$status"
        ;;
    *)
        run_arm "$@"
        ;;
esac

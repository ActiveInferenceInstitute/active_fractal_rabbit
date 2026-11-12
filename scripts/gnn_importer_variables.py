"""Print the GNN-derived manuscript variables for this repository.

Run this script with the pinned GNN interpreter. It calls the shared reader's
``importer_variables`` so the manuscript producer never re-reads the pin JSON.
The ``gnn_*`` key names are the contract's, so prose does not change.

Usage::

    python scripts/gnn_importer_variables.py --repo . \
        --job gnn-conformance --job-name "gnn conformance (pinned)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn.importers import importer_variables, read_pin

PIN_PATH = Path(".github") / "gnn-pin.json"
JOB = "gnn-conformance"
JOB_NAME = "gnn conformance (pinned)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root")
    parser.add_argument("--job", default=JOB, help="conformance job key")
    parser.add_argument(
        "--job-name", default=JOB_NAME, help="conformance job display name"
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()
    pin = read_pin(repo_root / PIN_PATH)
    variables = importer_variables(
        pin, repo_root=repo_root, job=args.job, job_name=args.job_name
    )
    # The schema names the map, not the manuscript variable file.
    variables.pop("schema", None)
    print(json.dumps(variables, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

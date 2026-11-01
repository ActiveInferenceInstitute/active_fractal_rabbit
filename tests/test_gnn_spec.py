"""The committed GNN specs are the checked source of truth for model structure.

These tests parse ``models/*.gnn.md`` through ``active_fractal_rabbit.gnn_spec``
and hold the GNN-derived declarations against the in-code
``single_factor_spec``/``factorial_spec`` definitions field by field. Editing one
side without the other fails loudly.

When the pinned upstream GNN checkout is available (sibling
``Generalized_Notation_Notation``, or ``$GNN_CHECKOUT``), the suite additionally
re-parses each spec with the upstream
``gnn.parsers.markdown_parser.MarkdownGNNParser``, runs the upstream
``gnn.cli validate --strict`` gate, and re-verifies that the recorded pin still
matches the checkout. Those checks skip with an explicit reason when the
checkout or its virtualenv is absent.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from active_fractal_rabbit.gnn_spec import (
    GNNSpecError,
    load_spec,
    load_specs,
    models_dir,
    parse_spec_file,
    spec_names,
    spec_path,
)
from active_fractal_rabbit.model_spec import (
    GenerativeModelSpec,
    factorial_spec,
    single_factor_spec,
)

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / ".github" / "gnn-pin.json"
DEFAULT_SIZES = dict(
    num_spatial_states=16,
    num_time_bins=4,
    num_speed_categories=4,
    num_burst_categories=2,
)


def _gnn_checkout() -> Path | None:
    """Locate the pinned upstream GNN checkout, if present locally."""

    override = os.environ.get("GNN_CHECKOUT")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_dir() else None
    candidate = ROOT.parent / "Generalized_Notation_Notation"
    return candidate if candidate.is_dir() else None


def _gnn_python(checkout: Path) -> Path:
    """Return the upstream virtualenv interpreter, or skip with a reason."""

    python = checkout / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip(f"upstream GNN virtualenv not found at {python}")
    probe = subprocess.run(
        [str(python), "-c", "import gnn"],
        cwd=str(checkout),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if probe.returncode != 0:
        pytest.skip(
            "upstream GNN package not importable: " + probe.stderr.strip()[:200]
        )
    return python


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


#: Parser sources the upstream virtualenv imports from the checkout's ``src/``
#: tree (editable install); they must match the pinned revision.
_PINNED_PARSER_FILES = (
    "src/gnn/parsers/markdown_parser.py",
    "src/gnn/parsers/common.py",
)

#: Runs under the upstream virtualenv: parse each spec with the real upstream
#: parser and dump a JSON structural summary for comparison in this suite.
_UPSTREAM_PARSE_BRIDGE = '''
import json
import sys

from gnn.parsers import markdown_parser
from gnn.parsers.markdown_parser import MarkdownGNNParser

summary_path = sys.argv[1]
parser = MarkdownGNNParser()
models = {}
for path in sys.argv[2:]:
    result = parser.parse_file(path)
    model = result.model
    models[path] = {
        "success": bool(result.success),
        "errors": [str(error) for error in (result.errors or [])],
        "structure_errors": [str(error) for error in model.validate_structure()],
        "model_name": model.model_name,
        "version": model.version,
        "variables": [
            {
                "name": variable.name,
                "dimensions": [int(value) for value in variable.dimensions],
            }
            for variable in model.variables
        ],
        "parameters": {parameter.name: parameter.value for parameter in model.parameters},
        "edges": [
            [source, target]
            for connection in model.connections
            for source in connection.source_variables
            for target in connection.target_variables
        ],
    }
payload = {"parser_module": markdown_parser.__file__, "models": models}
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
'''


def _upstream_structure(parsed: dict) -> dict[str, object]:
    """Derive the declared model structure from an upstream parser summary."""

    dimensions = {entry["name"]: entry["dimensions"] for entry in parsed["variables"]}
    factor_names = tuple(
        entry["name"][2:]
        for entry in parsed["variables"]
        if entry["name"].startswith("s_")
    )
    observation_names = tuple(
        entry["name"][2:]
        for entry in parsed["variables"]
        if entry["name"].startswith("o_")
    )
    control_names = tuple(
        entry["name"][2:]
        for entry in parsed["variables"]
        if entry["name"].startswith("u_")
    )
    factor_index = {name: index for index, name in enumerate(factor_names)}
    a_factors: dict[str, list[str]] = {name: [] for name in observation_names}
    b_factors: dict[str, list[str]] = {name: [] for name in factor_names}
    for source, target in parsed["edges"]:
        if not source.startswith("s_"):
            continue
        factor = source[2:]
        if factor not in factor_index:
            continue
        if target.startswith("A_"):
            modality = target[2:]
            if modality in a_factors and factor not in a_factors[modality]:
                a_factors[modality].append(factor)
        elif target.startswith("B_"):
            target_factor = target[2:]
            if target_factor in b_factors and factor not in b_factors[target_factor]:
                b_factors[target_factor].append(factor)
    return {
        "name": parsed["parameters"].get("model_id"),
        "factor_names": factor_names,
        "factor_sizes": tuple(dimensions[f"s_{name}"][0] for name in factor_names),
        "observation_names": observation_names,
        "observation_sizes": tuple(
            dimensions[f"o_{name}"][0] for name in observation_names
        ),
        "control_names": control_names,
        "control_sizes": tuple(dimensions[f"u_{name}"][0] for name in factor_names),
        "a_dependencies": tuple(
            tuple(factor_index[factor] for factor in a_factors[name])
            for name in observation_names
        ),
        "b_dependencies": tuple(
            tuple(factor_index[factor] for factor in b_factors[name])
            for name in factor_names
        ),
        "conditional_independence": parsed["parameters"].get(
            "conditional_independence"
        ),
    }


def _break_spec(name: str, tmp_path: Path, transform) -> Path:
    text = spec_path(name).read_text(encoding="utf-8")
    broken = tmp_path / f"broken_{name}.gnn.md"
    broken.write_text(transform(text), encoding="utf-8")
    return broken


def test_gnn_spec_files_are_committed() -> None:
    assert spec_names() == ("single_factor_reference", "factorial_latent_mode")
    for name in spec_names():
        path = spec_path(name)
        assert path.is_file(), path
        assert path.parent == models_dir()


def test_single_factor_spec_literal_expectations() -> None:
    spec = load_spec("single_factor_reference")
    assert spec.name == "single_factor_reference"
    assert spec.factor_names == ("cell",)
    assert spec.factor_sizes == (16,)
    assert spec.observation_names == (
        "location",
        "time_of_day",
        "speed",
        "report_burst",
    )
    assert spec.observation_sizes == (16, 4, 4, 2)
    assert spec.a_dependencies == ((0,), (0,), (0,), (0,))
    assert spec.b_dependencies == ((0,),)
    assert spec.control_sizes == (2,)
    assert spec.conditional_independence is True


def test_factorial_spec_literal_expectations() -> None:
    spec = load_spec("factorial_latent_mode")
    assert spec.name == "factorial_latent_mode"
    assert spec.factor_names == ("cell", "movement_mode", "cognitive_proxy_mode")
    assert spec.factor_sizes == (16, 3, 3)
    assert spec.observation_names == (
        "location",
        "time_of_day",
        "speed",
        "report_burst",
        "cell_delta",
        "return_flag",
    )
    assert spec.observation_sizes == (16, 4, 4, 2, 2, 2)
    assert spec.a_dependencies == ((0,), (0, 2), (1, 0), (2,), (1,), (1,))
    assert spec.b_dependencies == ((0, 1), (1, 2), (2,))
    assert spec.control_sizes == (2, 2, 2)
    assert spec.conditional_independence is False


def test_gnn_specs_match_in_code_declarations_field_by_field() -> None:
    """Drift guard: the GNN source of truth and the code must agree exactly."""

    specs = load_specs()
    assert set(specs) == {"single_factor_reference", "factorial_latent_mode"}
    expected = {
        "single_factor_reference": single_factor_spec(**DEFAULT_SIZES),
        "factorial_latent_mode": factorial_spec(**DEFAULT_SIZES),
    }
    for name, in_code in expected.items():
        assert isinstance(specs[name], GenerativeModelSpec)
        for field in dataclasses.fields(GenerativeModelSpec):
            actual = getattr(specs[name], field.name)
            wanted = getattr(in_code, field.name)
            assert actual == wanted, (
                f"{name}.{field.name} drifted: GNN={actual!r} code={wanted!r}"
            )


def test_missing_required_section_raises_typed_error(tmp_path: Path) -> None:
    broken = _break_spec(
        "single_factor_reference",
        tmp_path,
        lambda text: text.replace("## Connections", "## ConnectionsRenamed", 1),
    )
    with pytest.raises(GNNSpecError):
        parse_spec_file(broken)


def test_missing_likelihood_declaration_raises_typed_error(tmp_path: Path) -> None:
    broken = _break_spec(
        "single_factor_reference",
        tmp_path,
        lambda text: "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith("A_report_burst[")
        ),
    )
    with pytest.raises(GNNSpecError):
        parse_spec_file(broken)


def test_missing_factor_declaration_raises_typed_error(tmp_path: Path) -> None:
    broken = _break_spec(
        "single_factor_reference",
        tmp_path,
        lambda text: "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith("s_cell[")
        ),
    )
    with pytest.raises(GNNSpecError):
        parse_spec_file(broken)


def test_unparseable_connection_raises_typed_error(tmp_path: Path) -> None:
    broken = _break_spec(
        "single_factor_reference",
        tmp_path,
        lambda text: text.replace(
            "## Equations", "this is not a connection\n\n## Equations", 1
        ),
    )
    with pytest.raises(GNNSpecError):
        parse_spec_file(broken)


def test_unknown_spec_name_raises_typed_error() -> None:
    with pytest.raises(GNNSpecError):
        load_spec("no_such_model")


def test_pin_file_records_upstream_revision() -> None:
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert pin["upstream_repository"].endswith(
        "ActiveInferenceInstitute/Generalized_Notation_Notation.git"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
    assert pin["tag"] == f"v{pin['version']}"
    assert pin["version"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", pin["pinned_on"])


def test_pinned_revision_and_parser_source_when_checkout_available() -> None:
    """The pin is the tag target, and the parser in use is the pinned source.

    The upstream virtualenv installs ``gnn`` editable from the checkout's
    ``src/`` tree, so the working-tree parser files are the parser the
    conformance tests import.
    """

    checkout = _gnn_checkout()
    if checkout is None:
        pytest.skip(
            "upstream GNN checkout not found at sibling "
            "'Generalized_Notation_Notation' and $GNN_CHECKOUT is unset"
        )
    if not (checkout / ".git").exists():
        pytest.skip(f"upstream checkout at {checkout} is not a git repository")
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    tag = _git(checkout, "rev-parse", f"{pin['tag']}^{{commit}}")
    if tag.returncode != 0:
        pytest.skip(
            f"upstream checkout does not carry tag {pin['tag']!r}; the pin "
            "cannot be re-verified against it"
        )
    assert tag.stdout.strip() == pin["revision"], (
        f"pin file records {pin['revision']} but tag {pin['tag']} dereferences "
        f"to {tag.stdout.strip()}"
    )
    present = _git(checkout, "cat-file", "-e", f"{pin['revision']}^{{commit}}")
    assert present.returncode == 0, (
        f"pinned revision {pin['revision']} is not present in {checkout}"
    )
    for relative in _PINNED_PARSER_FILES:
        pinned_blob = _git(checkout, "show", f"{pin['revision']}:{relative}")
        if pinned_blob.returncode != 0:
            pytest.skip(f"cannot read {relative} at the pinned revision")
        working = (checkout / relative).read_text(encoding="utf-8")
        assert working == pinned_blob.stdout, (
            f"{relative} in {checkout} differs from the pinned revision "
            f"{pin['revision']}; the parser in use is not the pinned parser "
            "(re-pin the checkout or restore the file)"
        )


def test_models_readme_restates_the_pin() -> None:
    readme = (models_dir() / "README.md").read_text(encoding="utf-8")
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert pin["upstream_repository"] in readme
    assert pin["revision"] in readme
    assert pin["version"] in readme


def test_specs_pass_upstream_gnn_validator_when_available() -> None:
    checkout = _gnn_checkout()
    if checkout is None:
        pytest.skip(
            "upstream GNN checkout not found at sibling "
            "'Generalized_Notation_Notation' and $GNN_CHECKOUT is unset"
        )
    python = _gnn_python(checkout)
    for name in spec_names():
        result = subprocess.run(
            [
                str(python),
                "-m",
                "gnn.cli",
                "validate",
                str(spec_path(name)),
                "--strict",
            ],
            cwd=str(checkout),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"upstream GNN validator rejected {name}:\n{result.stdout}\n{result.stderr}"
        )


def test_upstream_parser_structure_matches_in_repo_declarations(
    tmp_path: Path,
) -> None:
    """The real upstream parser reads exactly the in-repo declared structure.

    Factor/observation/control cardinalities, the ``A``/``B`` dependency edges,
    and the conditional-independence flag are compared against both the in-code
    ``single_factor_spec``/``factorial_spec`` declarations and the in-repo
    ``gnn_spec`` loader, so a drifted GNN file or a drifted declaration fails.
    """

    checkout = _gnn_checkout()
    if checkout is None:
        pytest.skip(
            "upstream GNN checkout not found at sibling "
            "'Generalized_Notation_Notation' and $GNN_CHECKOUT is unset"
        )
    python = _gnn_python(checkout)
    bridge = tmp_path / "upstream_parse_bridge.py"
    bridge.write_text(_UPSTREAM_PARSE_BRIDGE, encoding="utf-8")
    summary_path = tmp_path / "upstream_parse_summary.json"
    result = subprocess.run(
        [
            str(python),
            str(bridge),
            str(summary_path),
            *[str(spec_path(name)) for name in spec_names()],
        ],
        cwd=str(checkout),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "single_factor_reference": single_factor_spec(**DEFAULT_SIZES),
        "factorial_latent_mode": factorial_spec(**DEFAULT_SIZES),
    }
    for name in spec_names():
        parsed = summary["models"][str(spec_path(name))]
        assert parsed["success"] is True, parsed["errors"]
        assert parsed["errors"] == [], parsed["errors"]
        assert parsed["structure_errors"] == [], parsed["structure_errors"]
        upstream = _upstream_structure(parsed)
        assert upstream["control_names"] == upstream["factor_names"]
        for source_label, declared in (
            ("in-code", expected[name]),
            ("gnn_spec", load_spec(name)),
        ):
            for field, value in upstream.items():
                if field == "control_names":
                    continue
                assert getattr(declared, field) == value, (
                    f"{name}.{field} differs between the GNN file (upstream "
                    f"parser) and the {source_label} declaration: "
                    f"GNN={value!r} {source_label}={getattr(declared, field)!r}"
                )

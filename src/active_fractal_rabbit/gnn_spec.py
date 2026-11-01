"""GNN-sourced generative-model specifications.

The committed GNN specs under ``models/`` are the single source of truth for
the *structure* of this project's active-inference generative models: the
hidden factors and their cardinalities, the observation modalities and their
cardinalities, the ``A``/``B`` factor-dependency structure, the control
cardinalities, and whether the conditional-independence approximation applies.
This module parses those declarative sections into the same
``GenerativeModelSpec`` dataclass that ``model_spec.py`` uses, so the in-code
declarations and the GNN files can be checked against each other
(``tests/test_gnn_spec.py``).

Tensors are deliberately *not* declared here. The ``A``/``B``/``D`` values are
learned from declared training rows, so the specs are matrix-free by design and
carry ``default=uniform`` normalization-contract hints instead of numbers.

Module scope is stdlib-only; ``model_spec.GenerativeModelSpec`` (and its numpy
import) is resolved lazily inside the parse entry points.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .model_spec import GenerativeModelSpec

__all__ = [
    "GNNSpecError",
    "GNN_SPEC_FILES",
    "REQUIRED_SECTIONS",
    "models_dir",
    "spec_path",
    "spec_names",
    "parse_spec_file",
    "load_spec",
    "load_specs",
]


class GNNSpecError(ValueError):
    """A GNN spec file is missing a required declaration or is malformed."""


#: Canonical GNN spec file per declared model name.
GNN_SPEC_FILES: dict[str, str] = {
    "single_factor_reference": "single_factor_reference.gnn.md",
    "factorial_latent_mode": "factorial_latent_mode.gnn.md",
}

#: Sections every GNN spec file must carry (per the upstream GNN v1.1 spec).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
)

_VAR_RE = re.compile(r"^(?P<name>[A-Za-z_\u03c0][A-Za-z0-9_\u03c0]*)\[(?P<dims>[^\]]*)\]")
_CONNECTION_RE = re.compile(
    r"^(?P<src>[A-Za-z_\u03c0][A-Za-z0-9_\u03c0]*)"
    r"(?P<op>[>\-])"
    r"(?P<dst>[A-Za-z_\u03c0][A-Za-z0-9_\u03c0]*)"
    r"(?::(?P<label>[A-Za-z0-9_]+))?$"
)
_LABELED_ANNOTATION_KEYS: dict[str, str] = {
    "Spec estimation contract": "estimation_contract",
    "Spec tuning contract": "tuning_contract",
    "Spec notes": "notes",
}
_LABELED_ANNOTATION_RE = re.compile(
    r"^(?P<label>Spec estimation contract|Spec tuning contract|Spec notes):\s*(?P<value>.*)$"
)
_TRUE_TOKENS = {"true", "yes", "1"}
_FALSE_TOKENS = {"false", "no", "0"}


def models_dir() -> Path:
    """Return the committed ``models/`` directory for this repository."""

    return Path(__file__).resolve().parents[2] / "models"


def spec_names() -> tuple[str, ...]:
    """Return the declared model names in stable order."""

    return tuple(GNN_SPEC_FILES)


def spec_path(name: str) -> Path:
    """Return the GNN spec path for a declared model name."""

    try:
        filename = GNN_SPEC_FILES[name]
    except KeyError as exc:
        raise GNNSpecError(
            f"unknown GNN spec name {name!r}; known names: {spec_names()}"
        ) from exc
    return models_dir() / filename


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(raw)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def _parse_declarations(block: str) -> list[tuple[str, tuple[str, ...]]]:
    """Parse ``StateSpaceBlock`` variable declarations as (name, dimension tokens)."""

    declarations: list[tuple[str, tuple[str, ...]]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        match = _VAR_RE.match(line)
        if match is None:
            continue
        dimensions: list[str] = []
        for part in match.group("dims").split(","):
            part = part.strip()
            if not part:
                continue
            # ``type=`` and ``default=`` are metadata, not dimensions.
            if "=" in part:
                continue
            dimensions.append(part)
        declarations.append((match.group("name"), tuple(dimensions)))
    return declarations


def _parse_model_parameters(block: str) -> dict[str, str]:
    parameters: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        parameters[key.strip()] = value.strip()
    return parameters


def _parse_annotations(block: str) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for raw in block.splitlines():
        match = _LABELED_ANNOTATION_RE.match(raw.strip())
        if match is None:
            continue
        key = _LABELED_ANNOTATION_KEYS[match.group("label")]
        value = match.group("value").strip()
        if value:
            annotations[key] = value
    return annotations


def _parse_edges(block: str, *, source: str) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _CONNECTION_RE.match(line)
        if match is None:
            raise GNNSpecError(f"{source}: unparseable connection {line!r}")
        edges.append((match.group("src"), match.group("dst")))
    return edges


def _positive_int(token: str, *, context: str) -> int:
    try:
        value = int(token)
    except (TypeError, ValueError) as exc:
        raise GNNSpecError(
            f"{context}: expected an integer dimension, got {token!r}"
        ) from exc
    if value < 1:
        raise GNNSpecError(f"{context}: dimension must be positive, got {value}")
    return value


def _bool_flag(token: str | None, *, context: str) -> bool:
    if token is None:
        raise GNNSpecError(f"{context}: missing `conditional_independence` parameter")
    lowered = token.strip().lower()
    if lowered in _TRUE_TOKENS:
        return True
    if lowered in _FALSE_TOKENS:
        return False
    raise GNNSpecError(
        f"{context}: expected a boolean `conditional_independence` value, got {token!r}"
    )


def parse_spec_file(path: str | Path) -> "GenerativeModelSpec":
    """Parse one GNN spec file into a ``GenerativeModelSpec``.

    Raises ``GNNSpecError`` when a required section, factor/modality/control
    declaration, dependency edge, or normalization-contract annotation is
    missing or internally inconsistent.
    """

    # Lazy import keeps this module importable without numpy at module scope.
    from .model_spec import GenerativeModelSpec

    spec_file = Path(path)
    if not spec_file.is_file():
        raise GNNSpecError(f"GNN spec file not found: {spec_file}")
    source = spec_file.name
    sections = _split_sections(spec_file.read_text(encoding="utf-8"))
    for required in REQUIRED_SECTIONS:
        if required not in sections or not sections[required].strip():
            raise GNNSpecError(f"{source}: missing required section '## {required}'")

    parameters = _parse_model_parameters(sections.get("ModelParameters", ""))
    annotations = _parse_annotations(sections.get("ModelAnnotation", ""))
    declarations = _parse_declarations(sections["StateSpaceBlock"])

    factor_names: list[str] = []
    factor_sizes: list[int] = []
    observation_names: list[str] = []
    observation_sizes: list[int] = []
    a_declared: dict[str, tuple[str, ...]] = {}
    b_declared: dict[str, tuple[str, ...]] = {}
    d_declared: dict[str, tuple[str, ...]] = {}
    control_sizes: dict[str, int] = {}

    for name, dimensions in declarations:
        if not dimensions:
            raise GNNSpecError(f"{source}:{name}: declaration has no dimensions")
        if name.startswith("s_"):
            factor_names.append(name[2:])
            factor_sizes.append(
                _positive_int(dimensions[0], context=f"{source}:{name}")
            )
        elif name.startswith("o_"):
            observation_names.append(name[2:])
            observation_sizes.append(
                _positive_int(dimensions[0], context=f"{source}:{name}")
            )
        elif name.startswith("A_"):
            a_declared[name[2:]] = dimensions
        elif name.startswith("B_"):
            b_declared[name[2:]] = dimensions
        elif name.startswith("D_"):
            d_declared[name[2:]] = dimensions
        elif name.startswith("u_"):
            control_sizes[name[2:]] = _positive_int(
                dimensions[0], context=f"{source}:{name}"
            )

    if not factor_names:
        raise GNNSpecError(f"{source}: no hidden-factor declarations (`s_<name>`)")
    if not observation_names:
        raise GNNSpecError(f"{source}: no observation declarations (`o_<name>`)")

    factor_index = {name: index for index, name in enumerate(factor_names)}
    factor_var = {f"s_{name}": name for name in factor_names}
    a_deps: dict[str, list[str]] = {name: [] for name in observation_names}
    b_deps: dict[str, list[str]] = {name: [] for name in factor_names}
    for src, dst in _parse_edges(sections["Connections"], source=source):
        factor = factor_var.get(src)
        if factor is None:
            continue
        if dst.startswith("A_"):
            modality = dst[2:]
            if modality in a_deps and factor not in a_deps[modality]:
                a_deps[modality].append(factor)
        elif dst.startswith("B_"):
            target = dst[2:]
            if target in b_deps and factor not in b_deps[target]:
                b_deps[target].append(factor)

    for factor in factor_names:
        if factor not in control_sizes:
            raise GNNSpecError(
                f"{source}: hidden factor {factor!r} has no control declaration "
                f"(`u_{factor}`)"
            )
        if factor not in d_declared:
            raise GNNSpecError(
                f"{source}: hidden factor {factor!r} has no prior declaration "
                f"(`D_{factor}`)"
            )
        expected_d = (str(factor_sizes[factor_index[factor]]),)
        if d_declared[factor] != expected_d:
            raise GNNSpecError(
                f"{source}: prior D_{factor} declares {d_declared[factor]} but "
                f"factor size implies {expected_d}"
            )
        if not b_deps[factor]:
            raise GNNSpecError(
                f"{source}: hidden factor {factor!r} has no transition dependency "
                f"(`B_{factor}`)"
            )
        declared_b = b_declared.get(factor)
        expected_b = (
            str(factor_sizes[factor_index[factor]]),
            str(factor_sizes[factor_index[factor]]),
            str(control_sizes[factor]),
        )
        if declared_b != expected_b:
            raise GNNSpecError(
                f"{source}: transition B_{factor} declares {declared_b} but "
                f"factor/control sizes imply {expected_b}"
            )

    a_dependencies = tuple(
        tuple(factor_index[factor] for factor in a_deps[modality])
        for modality in observation_names
    )
    b_dependencies = tuple(
        tuple(factor_index[factor] for factor in b_deps[factor])
        for factor in factor_names
    )

    for index, modality in enumerate(observation_names):
        declared = a_declared.get(modality)
        if declared is None:
            raise GNNSpecError(
                f"{source}: observation modality {modality!r} has no likelihood "
                f"declaration (`A_{modality}`)"
            )
        if not a_deps[modality]:
            raise GNNSpecError(
                f"{source}: observation modality {modality!r} has no likelihood "
                f"dependency edge (`s_<factor>>A_{modality}`)"
            )
        expected = (str(observation_sizes[index]),) + tuple(
            str(factor_sizes[factor]) for factor in a_dependencies[index]
        )
        if declared != expected:
            raise GNNSpecError(
                f"{source}: likelihood A_{modality} declares {declared} but "
                f"modality/factor sizes imply {expected}"
            )

    name = parameters.get("model_id")
    if not name:
        raise GNNSpecError(f"{source}: missing `model_id` in ModelParameters")
    for annotation_key in ("estimation_contract", "tuning_contract", "notes"):
        if annotation_key not in annotations:
            raise GNNSpecError(
                f"{source}: missing labeled annotation for {annotation_key!r}"
            )

    return GenerativeModelSpec(
        name=name,
        factor_names=tuple(factor_names),
        factor_sizes=tuple(factor_sizes),
        observation_names=tuple(observation_names),
        observation_sizes=tuple(observation_sizes),
        a_dependencies=a_dependencies,
        b_dependencies=b_dependencies,
        control_sizes=tuple(control_sizes[factor] for factor in factor_names),
        conditional_independence=_bool_flag(
            parameters.get("conditional_independence"), context=source
        ),
        estimation_contract=annotations["estimation_contract"],
        tuning_contract=annotations["tuning_contract"],
        notes=annotations["notes"],
    )


def load_spec(name: str) -> "GenerativeModelSpec":
    """Load the parsed spec for one declared model name."""

    return parse_spec_file(spec_path(name))


def load_specs() -> dict[str, "GenerativeModelSpec"]:
    """Load every committed GNN spec, keyed by declared model name."""

    return {name: parse_spec_file(spec_path(name)) for name in GNN_SPEC_FILES}

from __future__ import annotations

from pathlib import Path

import pytest

from active_fractal_rabbit import fractalrabbit
from active_fractal_rabbit.fractalrabbit import resolve_java


def test_resolve_java_env_override_wins(tmp_path, monkeypatch) -> None:
    fake_home = tmp_path / "jdk"
    fake_java = fake_home / "bin" / "java"
    fake_java.parent.mkdir(parents=True)
    fake_java.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("AFR_JAVA_HOME", str(fake_home))
    monkeypatch.setattr(
        fractalrabbit,
        "_java_major_version",
        lambda exe: 21 if str(exe) == str(fake_java) else 0,
    )

    resolved = resolve_java()

    assert resolved is not None
    java, home = resolved
    assert java == fake_java
    assert home == fake_home.resolve()


def test_resolve_java_returns_none_when_no_suitable_jdk(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AFR_JAVA_HOME", raising=False)
    monkeypatch.setattr(fractalrabbit.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        fractalrabbit, "HOMEBREW_JDK_CANDIDATES", (tmp_path / "nope" / "java",)
    )
    monkeypatch.setattr(fractalrabbit, "_java_home_lookup", lambda major: None)

    assert resolve_java() is None


def test_resolve_java_skips_too_old_path_java(monkeypatch, tmp_path) -> None:
    old_java = tmp_path / "old" / "java"
    old_java.parent.mkdir(parents=True)
    old_java.write_text("#!/bin/sh\n", encoding="utf-8")
    new_java = tmp_path / "new" / "bin" / "java"
    new_java.parent.mkdir(parents=True)
    new_java.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("AFR_JAVA_HOME", raising=False)
    monkeypatch.setattr(fractalrabbit.shutil, "which", lambda name: str(old_java))
    monkeypatch.setattr(fractalrabbit, "HOMEBREW_JDK_CANDIDATES", (new_java,))
    monkeypatch.setattr(fractalrabbit, "_java_home_lookup", lambda major: None)
    versions = {str(old_java): 17, str(new_java): 21}
    monkeypatch.setattr(
        fractalrabbit, "_java_major_version", lambda exe: versions.get(str(exe), 0)
    )

    resolved = resolve_java()

    assert resolved is not None
    assert resolved[0] == new_java


@pytest.mark.requires_fractalrabbit
def test_resolve_java_finds_a_real_jdk21_on_this_host() -> None:
    """Live probe: after `brew install openjdk@21` the resolver must succeed.

    The probe is environment-bound (an external-lane dependency on the host),
    so where no real JDK 21 is resolvable — e.g. the CI runner — it skips with
    the missing dependency named instead of failing. (removal condition: a
    JDK 21 is provisioned on the test host; owner: external lane)
    """

    resolved = resolve_java()
    if resolved is None:
        pytest.skip(
            "no real JDK 21 resolvable on this host (AFR_JAVA_HOME, PATH, "
            "Homebrew candidates and java_home all empty); install JDK 21 "
            "(external lane) to run this live probe"
        )
    java, home = resolved
    assert Path(java).exists()
    assert (Path(home) / "bin" / "java").exists()

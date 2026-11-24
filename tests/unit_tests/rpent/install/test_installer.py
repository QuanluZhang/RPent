# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for installer lifecycle, destination guards, doctor, and CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rpent.cli import install as install_cli
from rpent.install import installer
from rpent.install.errors import InstallError


def _write_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "rpent"\ndependencies = ["numpy"]\n'
        '[project.optional-dependencies]\nplanner = ["httpx"]\n',
        encoding="utf-8",
    )
    (tmp_path / "components/infra/a").mkdir(parents=True)
    (tmp_path / "components/infra/a/component.toml").write_text(
        'schema = 2\nname = "infra-a"\nprobe = "json"\ndependencies = []\n',
        encoding="utf-8",
    )
    (tmp_path / "components/infra/b").mkdir(parents=True)
    (tmp_path / "components/infra/b/component.toml").write_text(
        'schema = 2\nname = "infra-b"\nprobe = "json"\ndependencies = ["scipy"]\n',
        encoding="utf-8",
    )
    return tmp_path


def _fake_venv(path: Path) -> Path:
    (path / "bin").mkdir(parents=True, exist_ok=True)
    (path / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (path / "bin/python").write_text("#!/bin/sh\n", encoding="utf-8")
    return path


def test_install_rejects_non_empty_non_venv_directory(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    target = tmp_path / "user-data"
    target.mkdir()
    important = target / "notes.txt"
    important.write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    for force in (False, True):
        with pytest.raises(InstallError, match="not an empty directory or virtualenv"):
            installer.install(["infra-a"], root=repo, venv=target, force=force)
    assert important.read_text(encoding="utf-8") == "keep me"
    assert not (target / "pyvenv.cfg").exists()


def test_install_preserves_overlays_on_rerun_and_forwards_index_and_torch_backend(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = tmp_path / ".venv-a"
    commands: list[tuple[list[str], dict | None]] = []

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        commands.append((list(cmd), env))
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)

    fp = installer.install(
        ["infra-a"],
        root=repo,
        venv=venv,
        index_url="https://mirror.example/simple",
        torch_backend="cu126",
        verbose=False,
    )
    assert fp["targets"] == ["infra-a"]
    pip_cmd, pip_env = next(c for c in commands if c[0][:3] == ["uv", "pip", "install"])
    assert "--index-url" in pip_cmd and "https://mirror.example/simple" in pip_cmd
    assert pip_env is not None and pip_env["UV_TORCH_BACKEND"] == "cu126"

    # Re-running the same closure updates in place without nuking overlaid files.
    overlay_marker = venv / "custom-overlay.txt"
    overlay_marker.write_text("kept", encoding="utf-8")
    installer.install(["infra-a"], root=repo, venv=venv, verbose=False)
    assert overlay_marker.read_text(encoding="utf-8") == "kept"

    # Different closure is rejected without --force.
    with pytest.raises(InstallError, match="different environment"):
        installer.install(["infra-b"], root=repo, venv=venv, verbose=False)


def test_doctor_warns_on_overlaid_packages_and_fails_on_unknown_python(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    fp = {
        "targets": ["infra-a"],
        "python": "3.11.9",
        "platform": installer.runtime_platform(),
        "components": {
            "infra-a": {
                "manifest_sha256": installer.discover(repo)
                .components["infra-a"]
                .manifest_sha256
            }
        },
        "requirements_sha256": installer.requirements_sha256(
            installer.resolve_targets(["infra-a"], installer.discover(repo), root=repo),
            root=repo,
            core_dependencies=("numpy",),
            planner_dependencies=("httpx",),
        ),
        "freeze_sha256": installer.freeze_sha256("numpy==1.26.4\n"),
        "installed_at": "2026-01-01T00:00:00Z",
    }
    installer.write_fingerprint(venv, fp)
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")

    def fake_diag(cmd):
        if cmd[:3] == ["uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(
                cmd, 0, "numpy==1.26.4\nmodelscope==1.0\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(installer, "_run_diag", fake_diag)
    checks = installer.check_installation(venv, root=repo)
    assert not any(c.status == installer.FAIL for c in checks)
    assert any(c.status == installer.WARN and "overlaid" in c.message for c in checks)

    # Unknown Python version fails closed.
    fp["python"] = "unknown"
    installer.write_fingerprint(venv, fp)
    checks = installer.check_installation(venv, root=repo)
    assert any(
        c.status == installer.FAIL and "Python version" in c.message for c in checks
    )


def test_current_venv_prefers_virtual_env_and_skips_bootstrap_cache(
    tmp_path, monkeypatch
):
    active = _fake_venv(tmp_path / "active")
    cache = _fake_venv(tmp_path / "uv-cache")
    monkeypatch.setenv("VIRTUAL_ENV", str(cache))
    monkeypatch.setenv("RPENT_INSTALL_BOOTSTRAP", "1")
    monkeypatch.setenv("RPENT_CALLER_VENV", str(active))
    assert installer.current_venv() == active

    monkeypatch.setenv("RPENT_CALLER_VENV", "")
    assert installer.current_venv() is None


def test_cli_list_and_check(tmp_path, capsys):
    repo = _write_repo(tmp_path / "repo")
    assert install_cli.main(["--root", str(repo), "list"]) == 0
    assert "infra-a" in capsys.readouterr().out
    assert install_cli.main(["--root", str(repo), "check", "--all"]) == 0
    assert "infra-a: ok" in capsys.readouterr().out

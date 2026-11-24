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

"""Unit tests for ``tests.tooling.layout``."""

from pathlib import Path

from tests.tooling import layout


def _write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_real_checkout_passes_layout():
    assert layout.main(["--root", str(layout.REPO_ROOT)]) == 0


def test_same_named_sim_cannot_import_robot_layer(tmp_path):
    _write(tmp_path, "robots/__init__.py")
    _write(tmp_path, "robots/alpha/__init__.py", "__all__ = []\n")
    _write(
        tmp_path,
        "robots/alpha/component.toml",
        'schema = 2\nname = "robot-alpha"\nprobe = "robots.alpha"\n',
    )
    _write(tmp_path, "components/__init__.py")
    _write(tmp_path, "components/sim/__init__.py")
    _write(tmp_path, "components/sim/alpha/__init__.py")
    _write(
        tmp_path,
        "components/sim/alpha/component.toml",
        'schema = 2\nname = "sim-alpha"\nprobe = "components.sim.alpha"\n',
    )
    _write(
        tmp_path,
        "components/sim/alpha/env_server.py",
        "from robots.alpha import robot_spec\n",
    )

    findings = layout.check_imports(tmp_path)
    assert any(
        "sim 'alpha' must not import robot 'alpha'" in f.message for f in findings
    )


def test_check_imports_ignores_venv_and_blocks_core_to_component(tmp_path):
    _write(tmp_path, ".venv-demo/lib/bad.py", "import robots.alpha\n")
    _write(tmp_path, "rpent/bad.py", "import components.models.openpi\n")

    findings = layout.check_imports(tmp_path)
    assert len(findings) == 1
    assert "core must not import components.models" in findings[0].message


def test_lazy_entries_reject_type_checking_else_and_dynamic_model_import(tmp_path):
    _write(
        tmp_path,
        "robots/alpha/__init__.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import os\n"
        "else:\n"
        "    import torch\n",
    )
    _write(
        tmp_path,
        "components/models/demo/__init__.py",
        "import importlib\nimportlib.import_module('torch')\n",
    )

    findings = layout.check_lazy_entries(tmp_path)
    assert any("robot __init__.py must stay lazy" in f.message for f in findings)
    assert any("third-party 'torch'" in f.message for f in findings)

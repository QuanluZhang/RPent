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

"""Import an optional dependency or raise an actionable component install error."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from rpent.install.installer import current_venv, read_fingerprint


class MissingComponentError(ImportError):
    """An optional module is missing from the current environment."""


def _install_hint(
    component: str, venv: Path | None, fingerprint: dict[str, Any] | None
) -> str:
    if fingerprint is None:
        return (
            f"install component {component!r} from an RPent checkout with "
            f"`./rpent-install.sh install {component}`"
        )
    venv_flag = f" --venv {venv}" if venv is not None else ""
    targets = list(fingerprint.get("targets") or [])
    if targets and component in fingerprint.get("components", {}):
        return (
            f"component {component!r} belongs to this environment; rerun "
            f"`rpent-install install {' '.join(targets)}{venv_flag}` to repair it"
        )
    installed = ", ".join(targets) or "no recorded components"
    combined = " ".join([*targets, component])
    return (
        f"this environment was installed for {installed}; install {component!r} "
        f"in its own environment with `rpent-install install {component}`, or "
        f"rebuild this environment with `rpent-install install {combined}"
        f"{venv_flag} --force`"
    )


def require(component: str, module: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingComponentError` naming ``component``.

    Attempts ``importlib.import_module(module)`` directly so any environment
    that already supplies ``module`` (such as ``infra-unit_cpu`` or a venv with
    overlaid packages) works regardless of which component name was recorded in
    ``rpent-install.json``. Only when ``module`` or one of its parent packages
    is actually missing does this consult the fingerprint to build the hint.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is None or not (
            missing == module or module.startswith(f"{missing}.")
        ):
            raise
        venv = current_venv()
        fingerprint = read_fingerprint(venv) if venv is not None else None
        raise MissingComponentError(
            f"component {component!r} requires {module!r}, which is not "
            f"importable in this environment; "
            f"{_install_hint(component, venv, fingerprint)}"
        ) from exc

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

"""Unit tests for component resolution, Python selection, and requirements rendering."""

from rpent.install.resolve import (
    choose_python,
    manylinux_baseline,
    python_errors,
    render_requirements,
    requirements_sha256,
    resolve,
)


def test_resolve_closure_and_render_requirements(make_component, make_catalog):
    sim = make_component("sim-s", dependencies=["mujoco==3.3.0"])
    robot = make_component(
        "robot-r", planner=True, requires=["sim-s"], dependencies=["scipy"]
    )
    catalog = make_catalog([sim, robot])

    res = resolve(
        ["robot-r", "sim-s"],
        catalog,
        core_dependencies=("numpy",),
        planner_dependencies=("httpx",),
    )
    assert res.ok
    assert set(res.components) == {"robot-r", "sim-s"}
    rendered = render_requirements(
        res, core_dependencies=("numpy",), planner_dependencies=("httpx",)
    )
    assert "# Rerun: rpent-install install robot-r sim-s" in rendered
    assert "numpy" in rendered and "httpx" in rendered and "mujoco==3.3.0" in rendered


def test_resolve_detects_conflicts_and_index_disagreement(make_component, make_catalog):
    a = make_component(
        "sim-a",
        env_conflicts=["sim-b"],
        dependencies=["torch==2.6.0"],
        index_url="https://idx-a",
        extra_index_urls=["https://extra-1"],
    )
    b = make_component(
        "sim-b",
        dependencies=["torch==2.8.0"],
        index_url="https://idx-b",
        extra_index_urls=["https://extra-2"],
    )
    res = resolve(["sim-a", "sim-b"], make_catalog([a, b]))
    assert not res.ok
    joined = "\n".join(res.errors)
    assert "env_conflicts" in joined
    assert "conflicting pins for 'torch'" in joined
    assert "selects index" in joined
    assert res.extra_index_urls == ("https://extra-1", "https://extra-2")


def test_choose_python_and_python_errors_handle_patch_bounds_and_unknown(
    make_component,
):
    comp = make_component("robot-a", python=">=3.11.4,<3.12")
    assert choose_python([comp]) == "3.11"
    assert python_errors("3.11", [comp]) == []
    assert python_errors("3.11.5", [comp]) == []
    assert python_errors("3.11.2", [comp])
    assert python_errors("unknown", [comp])
    assert python_errors("", [comp])


def test_requirements_sha256_ignores_override_comments(
    make_component, make_catalog, tmp_path
):
    override = tmp_path / "override.txt"
    override.write_text("# comment 1\nnumpy<2\n", encoding="utf-8")
    comp = make_component("sim-a", overrides=[override])
    res = resolve(["sim-a"], make_catalog([comp]))
    h1 = requirements_sha256(res, root=tmp_path)

    override.write_text("# different comment\nnumpy<2\n", encoding="utf-8")
    h2 = requirements_sha256(res, root=tmp_path)
    assert h1 == h2

    override.write_text("numpy<1.26\n", encoding="utf-8")
    h3 = requirements_sha256(res, root=tmp_path)
    assert h3 != h1


def test_manylinux_baseline_selects_newest(make_component):
    a = make_component("sim-a", manylinux="2_31")
    b = make_component("sim-b", manylinux="2_35")
    assert manylinux_baseline([a, b]) == "2_35"

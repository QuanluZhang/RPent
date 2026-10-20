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

"""LeKiwi toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.lekiwi import tools as lekiwi_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools
from rpent.tools.mapping import MappingMixin
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager
    from rpent.tools.human_in_the_loop import HumanInTheLoopInput


class LeKiwiToolkit(MappingMixin, Toolkit):
    """Common RPent tools plus safe LeKiwi planner primitives."""

    _tools_module = lekiwi_tools
    _primitives_cls = lekiwi_tools.LeKiwiPrimitives

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
        operator_input: HumanInTheLoopInput | None = None,
    ) -> None:
        runtime_kwargs = dict(runtime_kwargs)
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._lingbotmap_client = runtime_kwargs.pop("lingbotmap_client", None)
        self._gdino_sam2_client = runtime_kwargs.pop("gdino_sam2_client", None)
        self._operator_input = operator_input
        self._primitives = self._primitives_cls(
            check_cancelled=self.raise_if_cancelled,
            **runtime_kwargs,
        )
        self._register_tools()
        self._state.reset()
        record = self._tools_module.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
        )
        self._publish_step(record)

    def _register_tools(self) -> None:
        self.add_tools(iter_tools(self._primitives))
        for definition in iter_tools(self._tools_module):
            self.add_tool(
                definition.with_handler(partial(definition, state=self._state))
            )
        if self._lingbotmap_client is not None:
            self.register_mapping_tools()

    def _map_frame(self) -> tuple[Any, list[float] | None, float]:
        """Return the current front RGB frame, base pose, and timestamp.

        The LeKiwi env reports measured body velocity only, so the base pose
        stays ``None`` and the mapping service estimates pose from frames.
        """
        import time as _time

        observation = self._primitives.env.get_observation()
        frame = observation.get("main_images")
        if frame is None:
            raise RuntimeError("LeKiwi observation has no main_images frame")
        return frame, None, _time.time()

    def solved(self) -> bool:
        """The velocity-controlled base reports no success signal of its own."""
        return False

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        record = self._tools_module.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = self._tools_module.view_env_state(record.step_idx, state=self._state)
        output.data["agent_elapsed_s"] = elapsed_s
        return output

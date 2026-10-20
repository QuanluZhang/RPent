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

"""Offline test for the LeKiwi RPC facade/client round-trip over HTTP."""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import numpy as np

from robots.lekiwi.env_client import LeKiwiEnvClient
from robots.lekiwi.env_server import LeKiwiEnvFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


class FakeBackend:
    def get_env_meta(self):
        return {"ok": True, "action_dim": 3}

    def reset(self):
        return {"ok": True, "states": np.zeros(3, dtype=np.float32)}

    def get_robot_state(self):
        return {"velocity": [0.0, 0.0, 0.0]}

    def get_observation(self):
        return {"states": np.zeros(3, dtype=np.float32)}

    def get_camera_meta(self):
        return {"cameras": {}}

    def drive(self, velocity, *, duration_s):
        return {
            "ok": True,
            "requested_velocity": np.asarray(velocity),
            "duration_s": duration_s,
        }

    def stop(self):
        return {"ok": True}


def test_lekiwi_client_and_facade_round_trip_numpy_over_http():
    facade = LeKiwiEnvFacade(FakeBackend())
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        client = LeKiwiEnvClient(HttpRpcClient(f"http://{host}:{port}"))
        result = client.drive([0.1, 0.0, 5.0], duration_s=0.5)
        stopped = client.stop()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["ok"] is True
    np.testing.assert_allclose(result["requested_velocity"], [0.1, 0.0, 5.0])
    assert result["duration_s"] == 0.5
    assert stopped["ok"] is True


def test_lekiwi_facade_registers_readonly_observation_routes():
    facade = LeKiwiEnvFacade(FakeBackend())

    readonly = facade._readonly_methods
    assert "env.get_env_meta" in readonly
    assert "env.get_observation" in readonly
    assert "env.get_camera_meta" in readonly
    assert "env.drive" not in readonly
    assert "env.stop" not in readonly


def test_local_worker_info_owns_hardware_in_process(
    fake_rlinf_realworld_modules, monkeypatch
):
    # The server runs without Ray: the WorkerInfo it hands the env must keep
    # every part local (node_rank=None everywhere), where RobotPart.connect
    # opens devices in-process instead of hosting scheduler workers.
    from robots.lekiwi.env_server import _local_worker_info
    from robots.lekiwi.runtime_config import load_runtime_config

    class FakeDataclass:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    captured: dict[str, Any] = {}

    def fake_lekiwi_config(**kwargs: Any) -> Any:
        captured["lekiwi_config"] = kwargs
        return FakeDataclass(**kwargs)

    def fake_robot_info(**kwargs: Any) -> Any:
        captured["robot_info"] = kwargs
        return FakeDataclass(**kwargs)

    def fake_worker_info(**kwargs: Any) -> Any:
        captured["worker_info"] = kwargs
        return FakeDataclass(**kwargs)

    class FakeAcceleratorType:
        NO_ACCEL = "NO_ACCEL"

    def fake_worker_address(_name: str, _rank: int) -> Any:
        return object()

    discovery = types.ModuleType("rlinf.robotics.discovery")
    discovery.RobotInfo = fake_robot_info
    monkeypatch.setitem(sys.modules, "rlinf.robotics.discovery", discovery)
    scheduler = types.ModuleType("rlinf.scheduler")
    scheduler.AcceleratorType = FakeAcceleratorType
    scheduler.WorkerAddress = fake_worker_address
    scheduler.WorkerInfo = fake_worker_info
    monkeypatch.setitem(sys.modules, "rlinf.scheduler", scheduler)

    example = (
        __import__("pathlib").Path(__file__).parents[4]
        / "robots/lekiwi/config/example.yaml"
    )
    runtime = load_runtime_config(example)

    # Swap the recording stand-in in only after config loading, which
    # validates keys against the fixture's dataclass fields.
    robots_lekiwi = sys.modules["rlinf.robotics.robots.lekiwi"]
    monkeypatch.setattr(robots_lekiwi, "LeKiwiConfig", fake_lekiwi_config)

    _local_worker_info(runtime.rlinf)

    # Local ownership: no node rank on the hardware config, none on the worker.
    assert captured["lekiwi_config"]["node_rank"] is None
    assert captured["worker_info"]["cluster_node_rank"] is None
    # The robot itself rides the worker's hardware slot the env reads.
    assert len(captured["worker_info"]["hardware_infos"]) == 1
    assert captured["robot_info"]["type"] == "LeKiwi"


def test_backend_drive_commands_unbatched_velocity(
    fake_rlinf_realworld_modules, monkeypatch
):
    # Regression: the backend clipped a commanded (3,) velocity against the
    # vector env's batched (1, 3) action space, which broadcast the action to
    # (1, 3) and the base rejected it ("Expected velocity shape (3,)").
    # drive() must send each env a plain (3,) action.
    import gymnasium as gym
    import numpy as np

    stepped: list[tuple[tuple[int, ...], ...]] = []

    class FakeInnerEnv:
        def __init__(self) -> None:
            self.action_space = gym.spaces.Box(
                -np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)
            )
            self.config = type("C", (), {"step_frequency": 50.0})
            self.unwrapped = self

        def get_measured_velocity(self):
            return np.zeros(3, dtype=np.float32)

        def step(self, action):
            stepped.append(tuple(np.asarray(action).shape))
            return ({"states": np.zeros((1, 3), dtype=np.float32)}, 0.0, False, False, {})

    inner = FakeInnerEnv()

    class FakeVector:
        # Mirrors NoAutoResetSyncVectorEnv: a batched (1, 3) action space that
        # slices the batch off before stepping each env.
        def __init__(self) -> None:
            self.envs = [inner]
            self._batched = gym.spaces.Box(
                -np.ones((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)
            )

        @property
        def action_space(self):
            return self._batched

        @property
        def single_action_space(self):
            return inner.action_space

        def call(self, _attr: str, _name: str) -> list[Any]:
            return [inner.config]

        def step(self, actions):
            return (inner.step(actions[0])[0], 0.0, False, False, {})

    class FakeRealWorldEnv:
        def __init__(self, _cfg: Any, **_rest: Any) -> None:
            self._vector = FakeVector()

        @property
        def env(self):
            return self._vector

        @property
        def action_space(self):
            return self._vector.action_space

        def step(self, actions):
            return self._vector.step(actions)

    real_env_module = types.ModuleType("rlinf.envs.real.env")
    real_env_module.RealWorldEnv = FakeRealWorldEnv
    monkeypatch.setitem(sys.modules, "rlinf.envs.real.env", real_env_module)

    import robots.lekiwi.env_server as env_server

    monkeypatch.setattr(env_server, "_local_worker_info", lambda _cfg: None)
    cfg = type("Cfg", (), {"env": type("Env", (), {"eval": None})()})()

    backend = env_server.LeKiwiEnvBackend(
        cfg=cfg, controller_config={"drive_max_duration_s": 2.0}
    )
    result = backend.drive([0.5, 0.0, 0.0], duration_s=0.1)

    assert result["ok"] is True
    assert stepped, "drive() must step the env at least once"
    assert all(shape == (3,) for shape in stepped), stepped
    np.testing.assert_allclose(backend.velocity_limits, [1.0, 1.0, 1.0])

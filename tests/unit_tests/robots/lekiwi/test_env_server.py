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

import threading

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

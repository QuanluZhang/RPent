LeKiwi
======

LeKiwi is a three-wheeled omnidirectional mobile base: three Feetech STS3215
wheel servos on a serial bus and one USB front camera. RPent drives it through
the RLinf LeKiwi real-world environment: the agent commands one bounded body
velocity ``[x, y, theta]`` (m/s, m/s, deg/s in the base frame) and reads the
measured wheel velocity plus the synchronized front-camera frame after every
command.

The base moves only while a ``drive`` call is executing — the environment stops
the wheels at the end of every control step, so an interrupted or crashed run
cannot leave the base rolling. There is no success signal of its own; the
planner decides success from the task evidence.

The environment server runs on the robot's onboard computer (the Pi), and the
RPent client attaches from another machine over HTTP. The LeKiwi dashboard spec
marks the environment as external: dashboard runs require ``--env-endpoint``
and never spawn their own environment process. Plain terminal runs may either
attach with ``--env-endpoint`` or let RPent spawn a local environment server.

Dependencies and installation
-----------------------------

Use Linux and Python 3.11 on the robot's computer. Install RPent with the
``lekiwi`` extra, then RLinf from a LeKiwi-supporting revision — the released
``rlinf`` extras do not yet include the LeKiwi environment layer. Until that
lands in a release, use the ``dev/lekiwi`` branch:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
   git -C /path/to/RLinf checkout dev/lekiwi
   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]' -e /path/to/RLinf
   uv pip check

The agent machine needs only RPent itself (``uv pip install -e '.[lekiwi]'``)
plus network reachability to the robot's computer.

Robot and task configuration
----------------------------

Copy ``robots/lekiwi/config/example.yaml`` outside versioned source and fill
the wheel bus port, the front camera's UVC serial, and the drive limits. Find
the camera serial with ``ls /dev/v4l2/by-id/`` or
``v4l2-ctl --list-devices``. Exactly one camera device must set ``main: true``;
its name becomes ``main_image_key``. The ``drive`` section's keys are RLinf
environment fields and are validated against the RLinf ``LeKiwiEnvConfig``:
an unknown key fails at startup instead of being silently ignored.

Tasks are registered in ``robots/lekiwi/tasks.py``. Task 0
(``drive_smoke_test``) exercises one conservative forward drive, one rotation,
and inspection after each; task 1 (``room_survey``) drives bounded segments
and reports what the camera sees. Pass the task with ``--task-id``.

Start the environment server
----------------------------

On the robot's computer, start Ray with the node rank exported — Ray captures
the environment when it starts — then run the env server bound to the
interface the agent machine can reach:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --transport http --host 0.0.0.0 --port 8120

The server launches the RLinf ``RealWorldEnv`` worker in a Ray group, connects
the wheel bus and the camera, and serves the ``env.*`` RPC methods. Use
``--print-config`` to print the resolved RLinf configuration and exit without
touching hardware. Stop the server with SIGINT; the worker stops the wheels on
shutdown.

Attach from the agent machine
-----------------------------

From the agent machine (for example a laptop on the same network), attach to
the running server and let the planner drive:

.. code-block:: bash

   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/robot.yaml \
     --planner api --model MODEL_ID

For dashboard control, add ``--dashboard`` and keep ``--env-endpoint``: the
LeKiwi environment is external, so the dashboard requires the endpoint and
never spawns its own server.

Tools and artifacts
-------------------

The planner sees three tools: ``drive(velocity, duration_s)``,
``stop()``, and the read-only ``view_env_state(step)``. ``duration_s`` is
bounded (at most 10 s per call); requested velocities are clipped to the
configured limits. Every step records ``front.png`` (the synchronized
front-camera frame) and ``camera_meta.json`` next to the run's step records;
``view_env_state`` returns the record blob plus the front image.

Safety
------

The base has no bumpers or onboard planner: clearance is the operator's
responsibility. Clear the floor around the base before every run, keep the
velocity limits in the site YAML conservative, and keep an operator within
reach of the base's power switch. If the camera view or the reported velocity
looks wrong, stop the run — an ``env.stop`` is one RPC call away from the
dashboard or terminal.

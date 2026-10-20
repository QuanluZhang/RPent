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

The extra installs headless OpenCV, matching ``lerobot[feetech]``: the robot
computer has no display, and the GUI OpenCV build would additionally require a
system ``libgl1`` package. Do not add GUI ``opencv-python`` to the same venv —
the two distributions unpack into the same ``cv2`` package and corrupt it.

The agent machine needs only RPent itself (``uv pip install -e '.[lekiwi]'``)
plus network reachability to the robot's computer.

Robot and task configuration
----------------------------

Run the detection script on the robot's computer to generate a ready-to-use
site YAML:

.. code-block:: bash

   python -m robots.lekiwi.detect --output /path/to/robot.yaml

The script never commands motion: it pings the serial bus at every baud rate
(lerobot's ``FeetechMotorsBus.scan_port``) and opens each camera once to
confirm it produces a frame. It writes the detected wheel-bus port and camera
devices plus conservative defaults for everything else, then prints the fields
a human must still review: ``wheel_radius`` and ``base_radius`` (measure them
with a ruler, in meters) and the ``drive`` limits (policy, not discovery —
start from the defaults and raise them gradually). Pass ``--skip-motors`` or
``--skip-cameras`` to probe only one side.

To fill the fields by hand instead, find each value on the robot's computer:

.. code-block:: bash

   # Wheel bus: the STS3215 board enumerates as a USB serial adapter.
   # /dev/serial/by-id names survive replugging; plain tty nodes renumber.
   ls -l /dev/serial/by-id/
   lerobot-find-port            # pings the bus; the wheels answer at IDs 7, 8, 9

   # Front camera: pick the *-index0 node (the rest are metadata nodes).
   ls -l /dev/v4l2/by-id/
   v4l2-ctl --list-devices      # human-readable camera names

``robot.port`` accepts either spelling, but the ``/dev/serial/by-id`` path is
the stable one. The camera ``serial`` accepts the full ``/dev/v4l2/by-id/...``
path, which pins the device across replugs; a numeric camera index also works
but renumbers with plug order. Copy ``robots/lekiwi/config/example.yaml``
outside versioned source for the overall shape. Exactly one camera device must
set ``main: true``; its name becomes ``main_image_key``. The ``drive``
section's keys are RLinf environment fields and are validated against the
RLinf ``LeKiwiEnvConfig``: an unknown key fails at startup instead of being
silently ignored.

The fields no command can discover are deliberate choices: geometry
(``wheel_radius``, ``base_radius``) comes from a ruler, ``max_raw`` (default
3000) is the servo's raw-speed cap, and the ``drive`` limits are the
operator's velocity envelope — keep them conservative, since the tasks bound
them further and the environment clips every command to them.

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

Workspace mapping
-----------------

LeKiwi can build a map of a workspace while an operator teleoperates the base.
The capability is robot-agnostic and lives in shared RPent components; the
LeKiwi toolkit adopts it when the two mapping services are reachable.

**Services.** ``rpent.robots.components.lingbotmap_server`` owns a LingBot-Map
mapping session (start, push frames, build, stop) over RPC. By default it
archives pushed frames and poses as the map artifact; when the ``lingbotmap``
package is installed it is used instead.
``rpent.robots.components.gdino_sam2_server`` wraps Grounding DINO (open-
vocabulary detection) and SAM 2 (mask refinement) behind one
``gdino_sam2.detect_segment`` call. Both servers are optional: they activate
only when their endpoints are configured.

**Activation.** Pass ``--map-endpoint`` to expose ``build_map``; pass
``--gdino-sam2-endpoint`` additionally to expose ``detect_objects`` and map
labeling. Both flags accept an existing server URL or spawn a local server
when no endpoint is given. For the perception server, export
``GDINO_CHECKPOINT_PATH`` and ``SAM2_CHECKPOINT_PATH`` and install the
``grounding-sam2`` extra in a CUDA environment first.

**Flow.** The agent calls ``build_map(map_name, labels)``. The tool opens a
mapping session, asks the operator to teleoperate the base slowly around the
workspace (a poller thread pushes synchronized front frames and the base pose
while they drive), and finalizes the map when the operator replies. With
``labels`` given, each label is detected and segmented on saved keyframes and
annotated onto the map. ``detect_objects(labels, step)`` runs the same
perception standalone on any persisted camera frame.

Since the LeKiwi environment reports measured velocity but no odometry pose,
frames are pushed without one and the mapping service estimates pose from
frames alone. The safety rules above apply unchanged during the teleoperation
sweep.

Start the Mapping Service Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the mapping server yourself when you want to inspect it in isolation or
run it on a different machine:

.. code-block:: bash

   # Optional: reconstruct real maps with the LingBot-Map GCT model
   # (add HF_ENDPOINT=https://hf-mirror.com for mainland China users)
   hf download robbyant/lingbot-map lingbot-map.pt --local-dir /path/to/checkpoints
   pip install -e '.[lingbot-map]'
   export LINGBOTMAP_CHECKPOINT_PATH=/path/to/checkpoints/lingbot-map.pt

   LINGBOTMAP_OUTPUT_DIR=/path/to/maps \
       python -m rpent.robots.components.lingbotmap_server \
       --transport http --host 127.0.0.1 --port 8116

With the checkpoint exported, :meth:`build_map` returns camera poses and
per-frame depth from the GCT streaming model, plus a world point cloud
unprojected from that depth.
Without it (the default) the server archives frames and poses as the map
artifact — same tool flow, no reconstruction. RPent passes its environment
through to the servers it spawns, so exporting the variables above before
launching RPent activates the real backend in spawned servers too.

Built maps are written under ``LINGBOTMAP_OUTPUT_DIR`` (or ``--output-dir``),
one directory per map with ``frames/*.png``, ``poses.json``, and ``map.json``.
Attach RPent to the running server by passing its address:

.. code-block:: bash

   rpent --robot lekiwi ... --map-endpoint http://127.0.0.1:8116

Prepare the Perception Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The perception server needs a CUDA GPU, the ``grounding-sam2`` extra, and two
checkpoints:

.. code-block:: bash

   uv pip install -e '.[grounding-sam2]'

   # Grounding DINO: the original checkpoint from the official release. The
   # transformers-format model.safetensors on Hugging Face will not load.
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -P /path/to/checkpoints

   # SAM 2: add HF_ENDPOINT=https://hf-mirror.com for mainland China users
   hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt \
       --local-dir /path/to/checkpoints

   export GDINO_CHECKPOINT_PATH=/path/to/checkpoints/groundingdino_swint_ogc.pth
   export SAM2_CHECKPOINT_PATH=/path/to/checkpoints/sam2.1_hiera_large.pt

The first start also downloads the ``bert-base-uncased`` tokenizer from
Hugging Face (Grounding DINO's text encoder), so the server needs network
access then; set ``HF_ENDPOINT=https://hf-mirror.com`` where huggingface.co
is slow or unreachable.

Keep the checkpoint filenames unchanged: the server picks the SAM 2 model
config from the filename. Then start it and attach RPent with
``--gdino-sam2-endpoint http://127.0.0.1:8115``:

.. code-block:: bash

   python -m rpent.robots.components.gdino_sam2_server \
       --transport http --host 127.0.0.1 --port 8115

Test the Services Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both servers answer plain RPC calls, so a few lines of Python are enough to
confirm one is alive before wiring it into a run. For the mapping server,
push a handful of synthetic frames and build:

.. code-block:: python

   import numpy as np
   from rpent.robots.components.lingbotmap_client import LingBotMapClient
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = LingBotMapClient(HttpRpcClient("http://127.0.0.1:8116"))
   client.start_session(map_name="smoke_test")
   for index in range(10):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       frame[:, :, index % 3] = 60 + index  # synthetic frames; a real run pushes camera frames
       client.push_frame(frame, pose=None, timestamp_s=index * 0.5)
   result = client.build()
   print(result.map_path, result.frames)  # .../maps/smoke_test 10

A successful build prints the map directory and the frame count; the
directory contains the archived frames and ``map.json``. Call
``client.stop_session()`` instead of ``build`` to discard a session.

For the perception server, run detection on any local photo:

.. code-block:: python

   import imageio
   from rpent.robots.components.gdino_sam2_client import GdinoSam2Client
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = GdinoSam2Client(HttpRpcClient("http://127.0.0.1:8115"))
   image = imageio.imread("photo.jpg")
   result = client.detect_segment(image, labels=["chair", "desk"])
   for detection in result.detections:
       print(detection.label, round(detection.score, 2), detection.box)

Each detection prints the matched label, its score, and its ``[x1, y1, x2,
y2]`` pixel box; ``result.detections`` is empty when the frame matches none of
the labels. The first call is slow while both models run their first
inference.

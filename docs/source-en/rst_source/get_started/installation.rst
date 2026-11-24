Installation
============

Install an isolated RPent environment for a robot, simulator, or standalone
model service from a source checkout, and verify that its dependencies work.
For a complete first LIBERO-PRO run with assets and checkpoints, follow
:doc:`quickstart`.

Requirements
------------

- Linux ``x86_64`` and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.
- Python 3.10–3.12. ``rpent-install`` automatically selects the newest supported
  minor accepted by all requested components and asks ``uv`` to provision it.
- An NVIDIA GPU and driver compatible with the selected component's PyTorch build.

Common Environments
-------------------

Each RPent component lives in a directory with a ``component.toml`` manifest
(``robots/<name>/`` for robots, ``components/{sim,models,tools,infra}/<name>/``
for shared services and test stacks). Installing a ``robot-*`` component pulls
in the planner layer, tools, simulator, and required model services into one
virtual environment.

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - Environment
     - Install command
     - Python & guide
   * - LIBERO / LIBERO-PRO
     - ``./rpent-install.sh install robot-libero --venv .venv-libero``
     - 3.12 · :doc:`../simulators/libero`
   * - LIBERO-plus
     - ``./rpent-install.sh install robot-libero_plus --venv .venv-libero-plus``
     - 3.12 · :doc:`../simulators/libero`
   * - RoboCasa365
     - ``./rpent-install.sh install robot-robocasa --venv .venv-robocasa``
     - 3.10 · :doc:`../simulators/robocasa`
   * - RoboTwin
     - ``./rpent-install.sh install robot-robotwin --venv .venv-robotwin``
     - 3.11 · :doc:`../simulators/robotwin`
   * - RoboLab
     - ``./rpent-install.sh install robot-robolab --venv .venv-robolab``
     - 3.11 · :doc:`../simulators/robolab`
   * - RoboDojo
     - ``./rpent-install.sh install robot-robodojo --venv .venv-robodojo``
     - 3.11 · :doc:`../simulators/robodojo`
   * - Single-Arm Franka
     - ``./rpent-install.sh install robot-franka --venv .venv-franka``
     - 3.12 · :doc:`../real_world_robots/franka`
   * - Dual-Arm Franka
     - ``./rpent-install.sh install robot-dual_franka --venv .venv-dual-franka``
     - 3.12 · :doc:`../real_world_robots/dual_franka`
   * - YAM (control machine)
     - ``./rpent-install.sh install robot-yam --venv .venv-yam-real``
     - 3.11 · :doc:`../real_world_robots/yam`
   * - Standalone Pi0.5 VLA
     - ``./rpent-install.sh install model-openpi infra-rlinf --venv .venv-openpi``
     - 3.12 · :doc:`../guides/advanced_deployment`
   * - Standalone Molmo
     - ``./rpent-install.sh install model-molmo --venv .venv-molmo``
     - 3.12 · :doc:`../guides/flash`
   * - Minimal core
     - ``./rpent-install.sh install infra-minimal --venv .venv-minimal``
     - 3.12 · CLI & discovery check
   * - Development / unit tests
     - ``./rpent-install.sh install infra-dev --venv .venv-dev``
     - 3.12 · ``infra-unit_cpu`` adds CPU Torch

Install an Environment
----------------------

From a fresh checkout, use ``./rpent-install.sh`` to bootstrap the installer
and create a virtual environment:

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero
   source .venv-libero/bin/activate

Once activated, ``rpent-install`` is available directly inside the environment.
Pass multiple component names to compose one environment (for example,
``model-openpi infra-rlinf`` for a standalone Pi0.5 VLA host, or
``robot-libero infra-dev`` for development with test tools). Pass ``--dry-run``
to inspect the generated requirements without installing.

**PyPI Mirrors and CUDA PyTorch Backends**

Pass ``--index-url`` to install through a PyPI mirror, or ``--torch-backend``
to select a PyTorch wheel index (``auto``, ``cpu``, ``cu124``, ``cu126``,
``cu128``):

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero \
     --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
     --torch-backend cu126

You can also set ``UV_INDEX_URL``, ``UV_DEFAULT_INDEX``, or
``UV_TORCH_BACKEND`` in the shell environment before invoking the installer.

.. dropdown:: Migrating from Legacy ``pyproject.toml`` Extras

   Per-robot ``pyproject.toml`` extras (``uv pip install -e ".[...]"``) are
   replaced by ``rpent-install`` components:

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - Former extra
        - Current command
      * - ``.[libero]``, ``.[libero-pro]``
        - ``./rpent-install.sh install robot-libero --venv .venv-libero``
      * - ``.[libero-plus]``
        - ``./rpent-install.sh install robot-libero_plus --venv .venv-libero-plus``
      * - ``.[robocasa]``
        - ``./rpent-install.sh install robot-robocasa --venv .venv-robocasa``
      * - ``.[robotwin]``
        - ``./rpent-install.sh install robot-robotwin --venv .venv-robotwin``
      * - ``.[robodojo]``
        - ``./rpent-install.sh install robot-robodojo --venv .venv-robodojo``
      * - ``.[franka]``
        - ``./rpent-install.sh install robot-franka --venv .venv-franka``
      * - ``.[yam]``
        - ``./rpent-install.sh install robot-yam --venv .venv-yam-real``
      * - ``.[flywheel]``
        - ``./rpent-install.sh install tool-flywheel --venv .venv-flywheel``
      * - ``.[molmo]``
        - ``./rpent-install.sh install model-molmo --venv .venv-molmo``
      * - ``.[sam3]``
        - ``./rpent-install.sh install model-sam3 --venv .venv-sam3``
      * - ``.[test]``
        - ``./rpent-install.sh install infra-dev --venv .venv-dev``

.. dropdown:: Environment Reuse, Overlays, and Wheel Scope

   - **Default path:** omitting ``--venv`` installs into
     ``.rpent/venvs/<components joined by +>`` inside the checkout.
   - **Reuse and overlays:** re-running ``install`` for the same component
     closure updates packages in place and preserves overlaid packages (such as
     ``modelscope``, ``pre-commit``, or editable RLinf checkouts). Pass
     ``--force`` to recreate the virtual environment cleanly; if recreation
     fails, the previous environment is restored.
   - **Foreign virtual environments:** ``rpent-install`` refuses to install
     into a non-empty directory that is not a virtual environment, or into a
     virtual environment created by another tool without ``--force``. For
     externally managed environments (such as Harbor in :doc:`../simulators/efe`
     or the YAM inference machine in :doc:`../real_world_robots/yam`), install
     RPent directly with ``uv pip install -e .``.
   - **PyPI wheel vs. checkout:** ``pip install rpent`` installs the core
     runtime only (``rpent[planner]`` adds planner adapters and the dashboard)
     without ``robots/`` or ``components/``. Run robots from a source checkout
     built with ``./rpent-install.sh``.

List and Check Components
-------------------------

Inspect all available components and validate manifests before installing:

.. code-block:: bash

   ./rpent-install.sh list
   ./rpent-install.sh check --all

``check`` computes each component's dependency closure without downloading
packages, verifying Python ranges, exact-pin conflicts, and ``env_conflicts``.
Across a closure, ``extra-index-urls`` are unioned, while conflicting
``index-url`` or ``index-strategy`` declarations are rejected. Passing several
component names checks whether they can share one environment.

.. dropdown:: Dependency Resolution Check (``check --resolve``)

   Add ``--resolve`` to run ``uv pip compile`` against Linux ``x86_64``
   (default ``manylinux_2_31``, or ``manylinux_2_35`` when required by a
   component such as ``sim-robodojo``) and discard the temporary output.
   Checking environments that build ``nvidia-curobo`` requires a Linux host
   with ``CUDA_HOME`` set.

Diagnose an Environment
-----------------------

Verify that an installed virtual environment matches its manifests, passes
``uv pip check``, and can import every component in its closure:

.. code-block:: bash

   ./rpent-install.sh doctor --venv .venv-libero
   ./rpent-install.sh doctor --venv .venv-libero --fingerprint

``--fingerprint`` prints the stored ``rpent-install.json`` metadata and its
hash without running checks. If you overlaid extra packages into the
environment (such as ``modelscope`` or a custom CUDA PyTorch wheel) and both
``uv pip check`` and all component probes succeed, ``doctor`` reports a
``warn`` line instead of failing.

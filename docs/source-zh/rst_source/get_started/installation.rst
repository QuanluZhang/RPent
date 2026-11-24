安装
====

在 RPent 源码目录中为机器人、仿真器或独立模型服务安装隔离的虚拟环境，并检查依赖是否可用。若需包含资产下载与模型权重的首次 LIBERO-PRO 运行流程，请先阅读 :doc:`quickstart`。

环境要求
--------

- Linux ``x86_64`` 与 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_。
- Python 3.10–3.12。``rpent-install`` 会自动选择所选组件共同支持的最新 Python 次版本，并由 ``uv`` 准备对应解释器。
- 与所选组件 PyTorch 构建兼容的 NVIDIA GPU 与驱动。

常用环境
--------

每个组件位于包含 ``component.toml`` 清单的目录中（机器人在 ``robots/<name>/``，共享仿真器、模型、工具和基础设施组件在 ``components/{sim,models,tools,infra}/<name>/``）。安装 ``robot-*`` 组件时，会自动将规划器依赖、工具、仿真器和所需模型服务装入同一个虚拟环境。

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - 环境
     - 安装命令
     - Python 与对应指南
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
   * - 单臂 Franka
     - ``./rpent-install.sh install robot-franka --venv .venv-franka``
     - 3.12 · :doc:`../real_world_robots/franka`
   * - 双臂 Franka
     - ``./rpent-install.sh install robot-dual_franka --venv .venv-dual-franka``
     - 3.12 · :doc:`../real_world_robots/dual_franka`
   * - YAM（控制机）
     - ``./rpent-install.sh install robot-yam --venv .venv-yam-real``
     - 3.11 · :doc:`../real_world_robots/yam`
   * - 独立 Pi0.5 VLA 服务
     - ``./rpent-install.sh install model-openpi infra-rlinf --venv .venv-openpi``
     - 3.12 · :doc:`../guides/advanced_deployment`
   * - 独立 Molmo 服务
     - ``./rpent-install.sh install model-molmo --venv .venv-molmo``
     - 3.12 · :doc:`../guides/flash`
   * - 最小核心环境
     - ``./rpent-install.sh install infra-minimal --venv .venv-minimal``
     - 3.12 · CLI 与机器人发现检查
   * - 开发 / 单元测试
     - ``./rpent-install.sh install infra-dev --venv .venv-dev``
     - 3.12 · ``infra-unit_cpu`` 额外包含 CPU Torch

安装环境
--------

在刚克隆的源码目录中，通过 ``./rpent-install.sh`` 引导安装器并创建虚拟环境：

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero
   source .venv-libero/bin/activate

激活环境后即可直接运行 ``rpent-install``。一次指定多个组件可将它们组合到同一环境中（例如独立 Pi0.5 VLA 主机使用 ``model-openpi infra-rlinf``，开发测试使用 ``robot-libero infra-dev``）。添加 ``--dry-run`` 可只打印生成的依赖清单而不执行安装。

**PyPI 镜像与 CUDA PyTorch 后端**

可通过 ``--index-url`` 指定 PyPI 镜像，或通过 ``--torch-backend`` 选择 PyTorch 轮子源（如 ``auto``、``cpu``、``cu124``、``cu126``、``cu128``）：

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero \
     --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
     --torch-backend cu126

也可以在运行安装命令前通过环境变量 ``UV_INDEX_URL``、``UV_DEFAULT_INDEX`` 或 ``UV_TORCH_BACKEND`` 配置。

.. dropdown:: 从旧版 ``pyproject.toml`` Extras 迁移

   原有的各机器人 extras（``uv pip install -e ".[...]"``）已统一由 ``rpent-install`` 组件替代：

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - 旧 extra
        - 当前命令
      * - ``.[libero]``、``.[libero-pro]``
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

.. dropdown:: 环境复用、叠加包与 Wheel 范围

   - **默认路径：** 省略 ``--venv`` 时，环境创建在源码目录的 ``.rpent/venvs/<组件名以 + 连接>`` 下。
   - **复用与叠加包：** 对相同组件集合重复执行 ``install`` 会在原环境中原地更新依赖，保留用户额外安装的包（如 ``modelscope``、``pre-commit`` 或可编辑安装的 RLinf）。需要彻底重建时传入 ``--force``；若重建失败，原有环境会被恢复。
   - **外部管理的虚拟环境：** ``rpent-install`` 会拒绝覆盖非空的普通目录，也会在未传 ``--force`` 时拒绝写入非本工具创建的已有虚拟环境。对于由外部工具维护的环境（例如 :doc:`../simulators/efe` 中的 Harbor 环境，或 :doc:`../real_world_robots/yam` 中的推理机环境），请直接执行 ``uv pip install -e .`` 安装 RPent。
   - **PyPI wheel 与源码目录：** ``pip install rpent`` 只包含核心运行时（``rpent[planner]`` 额外包含规划器适配器与 Dashboard），不含 ``robots/`` 与 ``components/``。运行机器人请在源码目录中通过 ``./rpent-install.sh`` 创建环境。

查看与检查组件
--------------

在安装前列出源码目录中的全部组件，并校验清单配置：

.. code-block:: bash

   ./rpent-install.sh list
   ./rpent-install.sh check --all

``check`` 不下载依赖，仅根据清单计算组件依赖集合，检查 Python 版本范围、互相冲突的精确版本固定以及 ``env_conflicts``。同一环境内多个组件的 ``extra-index-urls`` 会取并集，而 ``index-url`` 或 ``index-strategy`` 不一致时会直接报错。同时指定多个组件名可检查它们能否共用同一环境。

.. dropdown:: 依赖可解性检查（``check --resolve``）

   添加 ``--resolve`` 会调用 ``uv pip compile`` 验证依赖在 Linux ``x86_64`` 目标上是否可解（默认 ``manylinux_2_31``，当包含 ``sim-robodojo`` 等声明更高基线的组件时自动提升至 ``manylinux_2_35``），并在检查后清理临时文件。检查包含 ``nvidia-curobo`` 的环境需要在设置了 ``CUDA_HOME`` 的 Linux 主机上运行。

环境诊断
--------

检查已安装的虚拟环境是否与当前清单一致、是否通过 ``uv pip check``，以及各组件能否正常导入：

.. code-block:: bash

   ./rpent-install.sh doctor --venv .venv-libero
   ./rpent-install.sh doctor --venv .venv-libero --fingerprint

``--fingerprint`` 只打印环境中记录的 ``rpent-install.json`` 指纹及其哈希，不执行检查。如果在环境中额外叠加安装了其他包（例如 ``modelscope`` 或自定义 CUDA 版本的 PyTorch），只要 ``uv pip check`` 与全部组件导入探针均通过，``doctor`` 只会输出 ``warn`` 提示而不会判定失败。

LeKiwi
======

LeKiwi 是一个三轮全向移动底盘：三个 Feetech STS3215 轮电机挂在一条串行总线上，外加一个 USB 前置摄像头。RPent 通过 RLinf 的 LeKiwi 真实世界环境控制它：规划器下发一个有界的机体速度 ``[x, y, theta]``，单位为底盘坐标系下的 m/s、m/s、deg/s，并在每次命令后读取实测轮速和同步的前置摄像头画面。

底盘只在 ``drive`` 调用执行期间运动——环境在每一个控制步结束时都会让轮子停转，因此运行被中断或进程崩溃都不会让底盘继续滑行。底盘自身没有成功信号，任务是否成功由规划器依据任务证据判断。

环境服务端运行在机器人的板载计算机（树莓派）上，RPent 客户端从另一台机器通过 HTTP 接入。LeKiwi 的 Dashboard 配置将环境标记为外部环境：Dashboard 运行必须提供 ``--env-endpoint``，且不会自行拉起环境进程。普通终端运行既可以用 ``--env-endpoint`` 接入，也可以让 RPent 在本地拉起一个环境服务端。

依赖与安装
----------

在机器人板载计算机上使用 Linux 和 Python 3.11。先安装带 ``lekiwi`` extra 的 RPent，再从支持 LeKiwi 的 RLinf 修订版本安装——已发布的 ``rlinf`` extra 尚不包含 LeKiwi 环境层，在合入发布前请使用 ``dev/lekiwi`` 分支：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git /path/to/RLinf
   git -C /path/to/RLinf checkout dev/lekiwi
   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]' -e /path/to/RLinf
   uv pip check

该 extra 安装的是 headless 版 OpenCV，与 ``lerobot[feetech]`` 保持一致：板载计算机没有显示器，而 GUI 版 OpenCV 还需要系统安装 ``libgl1``。不要在同一个 venv 中再加装 GUI 版 ``opencv-python``——两个发行版会解包到同一个 ``cv2`` 包里并互相覆盖。

控制端机器只需安装 RPent 本身（``uv pip install -e '.[lekiwi]'``），并能通过网络访问机器人板载计算机即可。

机器人与任务配置
----------------

在机器人板载计算机上运行检测脚本，直接生成可用的站点 YAML：

.. code-block:: bash

   python -m robots.lekiwi.detect --output /path/to/robot.yaml

脚本不会让底盘运动：它只按各个波特率对串行总线执行 ping，即 lerobot 的
``FeetechMotorsBus.scan_port``，并打开每个摄像头确认能取到一帧。它会写入
检测到的轮电机串口和摄像头设备，其余字段填保守默认值，并在最后列出仍需人工
确认的字段：``wheel_radius`` 和 ``base_radius``（用尺子量，单位为米），以及
``drive`` 的速度上限（这是人为策略，无法探测——从默认值起步，逐步放宽）。
只探测其中一侧时可以加 ``--skip-motors`` 或 ``--skip-cameras``。

如果想手工填写，可以在机器人板载计算机上用下面的命令找出各个值：

.. code-block:: bash

   # 轮电机串口：STS3215 控制板枚举为一个 USB 串行适配器。
   # /dev/serial/by-id 名称在重新插拔后不变；裸 tty 节点可能重新编号。
   ls -l /dev/serial/by-id/
   lerobot-find-port            # 对总线执行 ping；三个轮电机应答 ID 为 7、8、9

   # 前置摄像头：选 *-index0 节点（其余是元数据节点）。
   ls -l /dev/v4l2/by-id/
   v4l2-ctl --list-devices      # 人类可读的摄像头名称

``robot.port`` 两种写法都接受，但 ``/dev/serial/by-id`` 路径更稳定。摄像头的
``serial`` 可以填完整的 ``/dev/v4l2/by-id/...`` 路径，重新插拔后设备不会漂移；
填数字摄像头索引也可以，但索引会随插拔顺序变化。整体结构可以参考
``robots/lekiwi/config/example.yaml``。有且仅有一个摄像头设备可以设置
``main: true``，它的名字会成为 ``main_image_key``。``drive`` 一节的键就是
RLinf 环境字段，会按 RLinf 的 ``LeKiwiEnvConfig`` 校验：未知键会在启动时报
错，而不是被静默忽略。

命令探测不到的字段都是人为选择：几何参数 ``wheel_radius``、``base_radius``
来自量尺，``max_raw`` 是舵机的原始速度上限、默认值为 3000，``drive`` 的速度
上限是操作者的速度包络——请保持保守，任务会进一步收紧约束，环境也会把每条命
令裁剪到上限内。

任务注册在 ``robots/lekiwi/tasks.py`` 中。任务 0（``drive_smoke_test``）依次执行一次保守的前进、一次原地旋转，并在每次动作后检查状态；任务 1（``room_survey``）按有界分段行进并汇报摄像头看到的内容。用 ``--task-id`` 选择任务。

启动环境服务端
--------------

在机器人板载计算机上，先导出节点序号再启动 Ray——Ray 会在启动时捕获环境变量——然后在能被控制端访问的网卡上运行环境服务端：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --transport http --host 0.0.0.0 --port 8120

服务端会在一个 Ray 分组里拉起 RLinf 的 ``RealWorldEnv`` worker，连接轮电机串口和摄像头，并提供 ``env.*`` 的 RPC 方法。可以用 ``--print-config`` 打印解析后的 RLinf 配置并退出，不接触硬件。用 SIGINT 停止服务端；worker 在退出时会停止轮子。

从控制端接入
------------

在控制端机器（例如同一网络下的一台笔记本）上接入正在运行的服务端，让规划器驱动底盘：

.. code-block:: bash

   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/robot.yaml \
     --planner api --model MODEL_ID

若要使用 Dashboard，加上 ``--dashboard`` 并保留 ``--env-endpoint``：LeKiwi 的环境是外部环境，Dashboard 必须指定端点，且不会自行拉起服务端。

工具与产物
----------

规划器可见三个工具：``drive(velocity, duration_s)``、``stop()`` 和只读的 ``view_env_state(step)``。``duration_s`` 有界（每次调用最多 10 秒）；请求的速度会被裁剪到配置的上限内。每一步都会在运行记录旁保存 ``front.png``（同步的前置摄像头帧）和 ``camera_meta.json``；``view_env_state`` 返回该步记录的完整数据以及前置图像。

安全事项
--------

底盘没有碰撞传感器，也没有板载避障：清场是操作者的责任。每次运行前清理底盘周围的地面，把站点 YAML 中的速度上限保持保守，并确保操作者随时够得到底盘的电源开关。如果摄像头画面或上报的轮速看起来不对，立即停止运行——从 Dashboard 或终端发起一次 ``env.stop`` 只有一次 RPC 调用的距离。

工作空间建图
------------

LeKiwi 可以在操作者遥控底盘的同时构建工作空间的地图。该能力是跨机器人的，位于 RPent 的共享组件中；只要两个建图服务可达，LeKiwi 的工具箱就会启用它。

**服务。** ``rpent.robots.components.lingbotmap_server`` 通过 RPC 持有一个 LingBot-Map 建图会话（开始、推送帧、构建、停止）。默认情况下它把推入的帧和位姿归档为地图产物；当安装了 ``lingbotmap`` 包时则改用该包。``rpent.robots.components.gdino_sam2_server`` 把 Grounding DINO（开放词汇检测）和 SAM 2（掩码细化）封装为一次 ``gdino_sam2.detect_segment`` 调用。两个服务端都是可选的：只有配置了端点才会启用。

**启用方式。** 传入 ``--map-endpoint`` 以暴露 ``build_map``；再传入 ``--gdino-sam2-endpoint`` 可额外暴露 ``detect_objects`` 和地图标注。两个参数既接受已运行服务端的 URL，也可以在未给出端点时在本地拉起服务端。对于感知服务端，需要先在 CUDA 环境中安装 ``grounding-sam2`` extra 并导出 ``GDINO_CHECKPOINT_PATH`` 和 ``SAM2_CHECKPOINT_PATH``。

**流程。** 规划器调用 ``build_map(map_name, labels)``。该工具会开启建图会话，请操作者缓慢地遥控底盘绕行工作空间（一个轮询线程在操作者驾驶期间推送同步的前置帧和底盘位姿），并在操作者回复后完成地图构建。若给出了 ``labels``，每个标签都会在保存的关键帧上检测并分割，然后标注到地图上。``detect_objects(labels, step)`` 可以在任意已保存的相机帧上独立运行同一感知能力。

由于 LeKiwi 环境只上报实测速度而没有里程计位姿，帧在推送时不携带位姿，由建图服务从帧本身估计位姿。上述安全规则在遥控扫描期间同样适用。

手动启动建图服务
~~~~~~~~~~~~~~~~

想单独检查建图服务端，或把它运行在另一台机器上时，可以自行启动：

.. code-block:: bash

   # 可选：用 LingBot-Map GCT 模型做真实重建
   # （中国大陆用户可在前面加 HF_ENDPOINT=https://hf-mirror.com）
   hf download robbyant/lingbot-map lingbot-map.pt --local-dir /path/to/checkpoints
   pip install -e '.[lingbot-map]'
   export LINGBOTMAP_CHECKPOINT_PATH=/path/to/checkpoints/lingbot-map.pt

   LINGBOTMAP_OUTPUT_DIR=/path/to/maps \
       python -m rpent.robots.components.lingbotmap_server \
       --transport http --host 127.0.0.1 --port 8116

导出该权重后，``build_map`` 会返回由 GCT 流式模型计算出的相机位姿、逐帧深度和世界点云。未导出时（默认），服务端把帧和位姿归档为地图产物——工具流程相同，只是没有重建。RPent 会把自身环境透传给它拉起的服务端，因此在启动 RPent 前导出上述变量即可在自动拉起的服务端中启用真实后端。

构建出的地图会写入 ``LINGBOTMAP_OUTPUT_DIR``（或 ``--output-dir``）之下，每张地图一个目录，内含 ``frames/*.png``、``poses.json`` 和 ``map.json``。把服务端地址传给 RPent 即可接入：

.. code-block:: bash

   rpent --robot lekiwi ... --map-endpoint http://127.0.0.1:8116

准备感知服务
~~~~~~~~~~~~

感知服务端需要 CUDA GPU、``grounding-sam2`` extra 和两个模型权重：

.. code-block:: bash

   uv pip install -e '.[grounding-sam2]'

   # Grounding DINO：必须使用官方 release 的原始权重。
   # Hugging Face 上 transformers 格式的 model.safetensors 无法加载。
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -P /path/to/checkpoints

   # SAM 2：中国大陆用户可在前面加 HF_ENDPOINT=https://hf-mirror.com
   hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt \
       --local-dir /path/to/checkpoints

   export GDINO_CHECKPOINT_PATH=/path/to/checkpoints/groundingdino_swint_ogc.pth
   export SAM2_CHECKPOINT_PATH=/path/to/checkpoints/sam2.1_hiera_large.pt

不要改动权重文件名：服务端会根据文件名选择 SAM 2 的模型配置。然后启动服务端，并用 ``--gdino-sam2-endpoint http://127.0.0.1:8115`` 接入 RPent：

.. code-block:: bash

   python -m rpent.robots.components.gdino_sam2_server \
       --transport http --host 127.0.0.1 --port 8115

手动测试服务
~~~~~~~~~~~~

两个服务端响应的都是普通 RPC 调用，接入运行之前用几行 Python 就能确认服务是否正常。对建图服务端，推入若干合成帧并构建：

.. code-block:: python

   import numpy as np
   from rpent.robots.components.lingbotmap_client import LingBotMapClient
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = LingBotMapClient(HttpRpcClient("http://127.0.0.1:8116"))
   client.start_session(map_name="smoke_test")
   for index in range(10):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       frame[:, :, index % 3] = 60 + index  # 合成帧；真实运行推入的是相机帧
       client.push_frame(frame, pose=None, timestamp_s=index * 0.5)
   result = client.build()
   print(result.map_path, result.frames)  # .../maps/smoke_test 10

构建成功会打印地图目录和帧数；目录中是归档的帧和 ``map.json``。若要丢弃会话，改调 ``client.stop_session()``。

对感知服务端，在任意本地照片上运行检测：

.. code-block:: python

   import imageio
   from rpent.robots.components.gdino_sam2_client import GdinoSam2Client
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = GdinoSam2Client(HttpRpcClient("http://127.0.0.1:8115"))
   image = imageio.imread("photo.jpg")
   result = client.detect_segment(image, labels=["chair", "desk"])
   for detection in result.detections:
       print(detection.label, round(detection.score, 2), detection.box)

每个检测结果打印匹配的标签、置信度和 ``[x1, y1, x2, y2]`` 像素框；帧中没有匹配标签时 ``result.detections`` 为空。第一次调用会较慢，因为两个模型要完成首次推理。

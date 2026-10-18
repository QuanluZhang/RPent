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

控制端机器只需安装 RPent 本身（``uv pip install -e '.[lekiwi]'``），并能通过网络访问机器人板载计算机即可。

机器人与任务配置
----------------

将 ``robots/lekiwi/config/example.yaml`` 复制到版本库之外并填入轮电机串口、前置摄像头的 UVC 序列号和速度上限。摄像头的序列号可以用 ``ls /dev/v4l2/by-id/`` 或 ``v4l2-ctl --list-devices`` 查找。有且仅有一个摄像头设备可以设置 ``main: true``，它的名字会成为 ``main_image_key``。``drive`` 一节的键就是 RLinf 环境字段，会按 RLinf 的 ``LeKiwiEnvConfig`` 校验：未知键会在启动时报错，而不是被静默忽略。

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

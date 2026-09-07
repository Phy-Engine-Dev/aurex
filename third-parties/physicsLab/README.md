# 内置 PhysicsLab SDK

本目录保存 `physicsLab` 2.0.6 的完整 Python 源码，用于 Aurex 的官方物理实验室社区 API、PLSAV 读写和发布兼容层。

- 上游：<https://github.com/GoodenoughPhysicsLab/physicsLab>
- 固定版本：2.0.6
- 许可证：MIT，完整许可证见 `LICENSE`

Aurex 在启动登录流程前将本目录置于 Python 导入路径最前，因此不会依赖或悄悄采用环境中未固定版本的 PyPI `physicsLab` 包。该目录不是 Git 子模块；源码随 Aurex 仓库提交。

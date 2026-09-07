# HDL 工作区

数字设计优先保存到会话所属的工作区，再读取、局部编辑和验证；不必每次重新输出完整 CPU。
这是 agent 的源码工具，不是网页 IDE。现有 Web 任务记录会显示工具调用和版本。

## 工作流程

1. `hdl_workspace_create` 创建 `.v` / `.sv` 文件；设计文件用 `role: "source"`，测试台用 `role: "testbench"`。
2. `hdl_workspace_read` 查看文件和 SHA；指定 `name` 读取原文。`find` 可定位精确字符串，`offset` / `length` 按字符分页。省略工作区 ID 可列出当前会话的工作区。
3. `hdl_workspace_edit` 带上 `expected_revision`，提交 `name` / `old_text` / `new_text`。替换必须唯一，多个文件的修改作为一个事务提交。歧义、旧版本或任何一处匹配失败时，全部不改。
4. 新文件用 `hdl_workspace_write`，`expected_sha256: null`；覆盖现有整个文件必须提供读取时的 SHA。局部修正优先使用 edit。
5. `hdl_simulate` 使用准确的 `workspace_id` / `workspace_revision`。自定义验证必须指定测试台 `top`，如需要导出，再指定设计模块 `design_top`。
6. 验证通过后，`verilog_to_sav` 只接收该次 `hdl_report_path`。只导出设计文件，不把测试台当作电路。编辑后的新版本必须重新验证。

例如，创建返回工作区 ID 和版本 1 后，局部修改可使用：

```json
{
  "workspace_id": "创建工具返回的ID",
  "expected_revision": 1,
  "edits": [{"name": "dut.v", "old_text": "assign y=a-b;", "new_text": "assign y=a+b;"}]
}
```

之后验证该次编辑返回的版本，而不是继续使用 1：

```json
{
  "workspace_id": "创建工具返回的ID",
  "workspace_revision": 2,
  "profile": "custom",
  "top": "tb",
  "design_top": "dut"
}
```

## 数据与验证边界

- SQLite 位于配置的 cache_dir 下 `hdl-workspaces/workspaces.sqlite3`，源文件及历史版本持久化；重启和同一会话的新任务可继续读取。备份时应包含该目录以及 `hdl` 验证报告目录，不要作为临时缓存清理。
- 不同会话不能读写彼此的工作区；工具不能指定服务器文件路径。
- 每次验证复制不可变源文件快照并记录 SHA。并发编辑不会改变正在编译的源码，也不会把旧验证记成新版本通过。
- 压缩索引保留工作区 ID / 版本。需要代码时读取准确版本，不依据摘要重写源码。
- custom 的 PASS 仅表示调用方测试台执行成功；不等于测试覆盖充分、完整 RV32I 合规或门级等价。
- 导出仍执行原有严格语义检查。例如当前引擎无法无损导出部分算术表达式的 X/Z 语义时，必须报错，不能放宽检查伪称成功。

## 回归

在安装了现有 Icarus 工具链的 Linux 环境：

```bash
PYTHONPATH=src:tests python -m unittest test_hdl_workspace test_hdl -q
```

工作区测试覆盖重启恢复、跨会话隔离、事务回滚、并发旧版本拒绝、精确替换、分页检索、压缩来源、沙箱验证、真实 HDL 失败后局部修复通过，以及旧报告不能导出新版。

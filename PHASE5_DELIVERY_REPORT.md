# Phase 5 交付状态报告

日期：2026-08-20

## 已完成

- 可复现安装、API启动、Gradio启动、批量推理和白名单打包 PowerShell 脚本。
- 生产配置样例；真实密钥只允许由运行环境或本机文件路径注入。
- 官方模板方案 Markdown 源和 DOCX 生成版；内容覆盖需求、架构、核心模块、创新点、效果评估与任务耗时。
- 模型卡、部署手册、4–5分钟演示分镜、截图清单、第三方组件说明和最终检查清单。
- `build-delivery`：只复制代码白名单、方案和四折权重，验证官方三目录结构、结果 Schema、身份占位符、敏感文件名、文本密钥赋值和文件哈希。
- 交付构建器成功路径、拒绝路径、缓存排除、身份占位符和密钥赋值均有自动测试。
- 参赛者已在本机完成官方初赛测试图片的四模型集成推理；`result/result.json`共695条记录，独立Schema校验返回`valid=true`且错误为0。
- 队伍名已定稿为`X-M`，方案文档身份占位符已清零；最终归档按`code/ design/ result/`白名单结构构建。
- 修复了Gradio界面静默启用Fake模型的问题：`start_ui.ps1`现在默认启动四折真实YOLO集成，固定320尺寸并在页面顶部显示运行模式；Qwen改为显式可选开关，评委无密钥时仍可完成视觉识别。
- 已按用户实际启动命令在7867端口验收真实页面，无需FastAPI；更新后的真实Agent对`00000.jpg`返回“有漂浮物”98.5%，与官方训练标签一致。

## 质量证据

- Ruff：通过。
- pytest：41 项通过。
- `pip check`：无依赖冲突。
- PowerShell：5 个脚本全部通过语法解析。
- DOCX结构：82个顶层段落、4个表格、1个A4节、14个包部件；可访问性审计0项问题。
- 原始官方模板 SHA-256：`8e8148db47a1fd7050c11027105f85299c72596ffaefd4c9a700824afc9877d0`，生成过程前后不变。
- 当前生成版方案 SHA-256：`5adeb5adfa39637ab726f393ab048185d2694684e4a043e21df094ca2cdcb6e1`。
- 4张表格均通过精确几何审计：`tblW`、`tblInd`、`tblGrid`和所有`tcW`一致。
- 现场验证未传 `--confirm-real-result` 时构建器返回错误并确认没有产生归档。

## 尚需参赛者完成的外部操作

1. DOCX视觉复核：当前环境缺少 LibreOffice/`soffice`，仍需在安装 Word/LibreOffice 的机器上逐页检查是否存在截断、溢出、异常分页或字体替换。
2. 平台提交：上传最终`.tar.gz`前，再核对官方群最新通知、平台文件大小限制和当日提交次数；上传后保存榜单分数与归档SHA-256。

## 收口命令

队伍名已定稿；如需重新生成方案，使用：

```powershell
.venv\Scripts\python.exe scripts\build_design_document.py `
  --template 'AI智能体设计方案.docx' `
  --source 'design\水域智巡智能体设计方案.md' `
  --output 'design\水域智巡智能体设计方案.docx' `
  --team-name 'X-M' `
  --date '2026-08-20' `
  --force
```

如需重新生成并验证真实结果（现有正式结果已通过校验，不要直接覆盖）：

```powershell
scripts\generate_submission.ps1 -TestImageDir '<由参赛者填写的测试图片目录>'
.venv\Scripts\water-agent.exe validate-submission --result 'result\result.json'
```

最终打包：

```powershell
scripts\build_delivery.ps1 `
  -ResultPath 'result\result.json' `
  -OutputPath '水域智巡_submission.tar.gz' `
  -ConfirmRealResult
```

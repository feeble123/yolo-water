# 部署与复现手册

本文面向评委、决赛平台运维人员和参赛队员，覆盖“安装—配置—单图智能体—批量推理—提交打包”的完整闭环。项目不预置测试集路径，也不会在启动时扫描测试图片。

## 1. 环境基线

- Windows 10/11 或兼容 Linux；Python 3.11（项目支持 3.11–3.12）。
- 推荐 NVIDIA GPU 与可用 CUDA；Phase 4 验收设备为 RTX 4070 Laptop 8GB。
- CPU 可运行，但四折集成延迟会明显增加。
- Qwen 解释功能需要能访问赛事方 OpenAI 兼容接口；视觉分类和批量结果生成可在 Qwen 不可用时独立运行。

## 2. 安装

在 `code/` 或项目根目录执行：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.production.example .env
```

也可运行 `scripts\install.ps1`。`.env`只存在于运行机器，严禁进入压缩包。

## 3. 运行配置

提交包把四个折模型放在 `models/fold0.pt` 至 `models/fold3.pt`。确认 `.env` 中：

```text
WATER_AGENT_ENSEMBLE_WEIGHTS=["models/fold0.pt","models/fold1.pt","models/fold2.pt","models/fold3.pt"]
WATER_AGENT_USE_FAKE_MODEL=false
```

默认配置会启动四折真实YOLO集成且关闭Qwen，因此即使评委没有密钥，也可以打开页面并完成视觉识别。Qwen密钥推荐通过进程环境变量注入；若使用文件，只填写运行机器上的路径：

```powershell
$env:WATER_AGENT_QWEN_API_KEY = '<在当前终端临时注入，不写入脚本>'
```

不要在录屏、日志、命令历史或异常截图中展示真实密钥。

## 4. 启动与健康检查

API：

```powershell
scripts\start_api.ps1
Invoke-RestMethod http://127.0.0.1:8000/health
```

演示界面：

```powershell
scripts\start_ui.ps1
```

浏览器访问 `http://127.0.0.1:7860`。该脚本默认启动四折真实YOLO集成，并固定为320尺寸和设备0；页面顶部会显示实际视觉模式。该Gradio服务不依赖FastAPI。

需要Qwen解释时，使用本机密钥文件显式启动：

```powershell
scripts\start_ui.ps1 -EnableQwen -QwenApiKeyFile '<本机官方API Key文件路径>'
```

仅当需要为其他系统提供REST接口时，才启动FastAPI：

```powershell
scripts\start_api.ps1
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 5. 参赛者本机生成结果

以下操作只能由参赛者在本机明确传入测试图片目录后执行：

```powershell
scripts\generate_submission.ps1 -TestImageDir '<测试图片目录>'
.venv\Scripts\water-agent.exe validate-submission --result result\result.json
```

程序拒绝覆盖已有结果，并另写概率审计文件。开发、测试和服务启动均不会自动进入测试目录。

## 6. 生成最终提交包

真实 `result.json` 通过校验后，执行：

```powershell
scripts\build_delivery.ps1 `
  -ResultPath 'result\result.json' `
  -OutputPath '水域智巡_submission.tar.gz' `
  -ConfirmRealResult
```

构建器只复制白名单文件，并在归档前验证 `code/`、`design/`、`result/result.json`、敏感文件名和结果 Schema。未显式确认真实结果时会拒绝生成归档。

## 7. 决赛平台迁移

- 视觉工具通过 `VisionClassifier` 接口隔离，可替换权重路径或推理后端。
- Qwen 通过 OpenAI 兼容适配器隔离；迁移时只调整基地址、模型名和密钥注入，不修改业务协议。
- Gradio、FastAPI 和批量 CLI 共用同一分类器与六类 Schema，避免演示链和打榜链结论不一致。
- 平台若限制联网，可关闭 Qwen，视觉识别、复核策略、审计和批量 JSON 仍可运行。

## 8. 故障排查

- `未配置Qwen API密钥`：不启用`-EnableQwen`时可忽略；若需要Qwen解释，确认显式传入`-QwenApiKeyFile`且文件路径正确。
- 模型文件不存在：确认四个 `models/fold*.pt` 均在位，路径相对当前工作目录。
- CUDA 显存不足：降低批量命令的 `--batch`，单图服务保持低并发。
- 审计库不可写：为 `runtime/`授予当前用户写权限，或暂时关闭审计；视觉核心服务不会因此停止。
- 远程大模型超时：Agent 会保留 YOLO 结果并降级解释，不要把接口故障误判为视觉分类失败。
# 智能体化升级运行补充（2026-08-21）

正式演示请启用赛事官方Qwen：

```powershell
.\scripts\start_ui.ps1 -EnableQwen -QwenApiKeyFile '.\api key.txt'
```

网页会显示“任务类型、计划来源、执行步骤、工具知识与核查依据、Top-3和工具轨迹”。Qwen先输出受校验的任务计划，本地白名单执行器再调用YOLO、类别知识、复核和报告工具；它不能改写YOLO分类标签。若Qwen网络服务失败，轨迹会显示`fallback:*`，系统仍使用确定性计划和本地解释完成识别。

不上传图片时，可输入例如“乱堆是什么意思？”进行类别咨询；此任务不会调用YOLO。`start_api.ps1`对应新增`POST /api/v1/consult`和`GET /api/v1/guidance/{label}`，原有分析和分类接口保持兼容。

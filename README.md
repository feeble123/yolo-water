# 水域综合异常识别智能体

赛事升级版工程：YOLO四模型集成负责六类水域综合异常识别；赛事官方Qwen负责生成受控任务计划和业务报告；本地白名单执行器调用工具并保留证据。批量赛事预测直接调用视觉模型，避免大模型随机性影响提交结果。

## 项目根目录与文件边界

唯一项目根目录：`E:\GPT-Codex\Yolo-water\水域综合异常识别`

| 分区 | 内容 | 规则 |
|---|---|---|
| 官方原始资料 | `训练集/`、`初赛测试集/`、`train.json`、赛事DOCX/TXT、官方ZIP | 训练集只读；程序开发与自动化不得访问初赛测试集 |
| 项目源码 | `src/`、`tests/`、`configs/`、`pyproject.toml` | 所有实现与测试代码 |
| 项目文档 | `README.md`、`architecture.md`、`findings.md`、`progress.md`、`task_plan.md` | 需求、架构、发现与进度 |
| 生成产物 | `artifacts/`、`result/` | 数据副本、指标、模型、日志和用户生成的提交结果 |
| 本地环境 | `.venv/`、`.env` | 不进入赛事提交包；`.env`和密钥严禁外发 |
| 预训练权重 | `yolo26n-cls.pt`、`yolo26n.pt` | 开源YOLO冒烟权重，最终权重写入`artifacts/training/` |

所有新增文件必须位于该项目根目录内。衍生训练图必须是独立副本，禁止硬链接到官方训练集。

## 当前能力

- 固定六分类结果协议：乱采、乱建、乱堆、乱占、有漂浮物、正常。
- 受控多工具智能体：支持图片异常识别、巡查报告、结果可靠性说明与无图片的类别知识咨询；Qwen计划必须通过本地Schema校验，失败时自动回退确定性计划。
- YOLO是唯一的视觉标签来源；Qwen不能修改标签、伪造目标框、数量、位置或违法事实。
- 页面展示任务类型、计划来源、工具步骤、工具状态、类别知识/核查依据、分类结果与人工复核建议。
- Fake视觉模型仅保留给自动化测试；`scripts\start_ui.ps1`默认启动四折真实YOLO集成，并在页面顶部显示实际运行模式。
- Ultralytics分类权重适配器。
- 四折YOLO分类模型等权概率集成，Agent单图工具与赛事批量链共用同一实现。
- 分组四折评估器：Accuracy、Balanced Accuracy、Macro-F1、逐类指标、混淆矩阵、Top-2及概率审计。
- OOF完整性检查与温度校准评估（NLL、Brier、ECE）。
- FastAPI单图分析与直接分类接口。
- Gradio演示界面：展示自然语言问题、分类结论、Top-3、复核提示和工具轨迹。
- SQLite脱敏审计日志：记录模型版本、结果与耗时，不记录原图、图片路径、问题或密钥。
- 官方格式 `result.json` 的离线校验器。
- Qwen3.6-27B官方接口适配层，密钥仅从环境变量或显式密钥文件读取。
- 训练折目录使用独立图片副本，隔离第三方库的JPEG自动修复行为，严禁硬链接回原始训练集。
- `verify-folds`可强制验证衍生数据无缺失、无大小异常且不与原训练集共享文件实体。
- 白名单`build-delivery`构建器：只组装`code/ design/ result/`，拒绝缺少真实结果确认、敏感文件和非官方顶层条目。
- 可复现部署脚本、模型卡、演示分镜、最终检查清单和官方模板方案文档。

## 安装与运行

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
.venv\Scripts\python -m uvicorn water_agent.api.main:app --reload
```

发布包中的评委/参赛者默认启动方式是`start_ui.ps1`：它自动选择四个真实模型、固定与打榜一致的320尺寸，并且不要求Qwen密钥。`FakeClassifier`仅用于测试，必须显式传入`-UseFakeModel`才会启用。

如需从Python命令直接启动，先在本地`.env`中设置真实模型路径：

```text
WATER_AGENT_USE_FAKE_MODEL=false
WATER_AGENT_ENSEMBLE_WEIGHTS=["artifacts/training/ablation_fold0_weighted_p05_c8_e15/weights/best.pt","artifacts/training/fold1_weighted_p05_c8_e15/weights/best.pt","artifacts/training/fold2_weighted_p05_c8_e15/weights/best.pt","artifacts/training/fold3_weighted_p05_c8_e15/weights/best.pt"]
WATER_AGENT_ENSEMBLE_TEMPERATURE=1.0
WATER_AGENT_VISION_IMAGE_SIZE=320
WATER_AGENT_VISION_DEVICE=0
WATER_AGENT_USE_QWEN=false
WATER_AGENT_QWEN_API_KEY_FILE=<本地密钥文件路径>
WATER_AGENT_QWEN_ENABLE_THINKING=false
WATER_AGENT_FORCE_REVIEW_LABELS=["乱建","正常"]
```

不要提交 `.env`、密钥文件或模型日志。API文档启动后位于 `http://127.0.0.1:8000/docs`。

先验证官方接口，再运行单图智能体：

```powershell
.venv\Scripts\python -m water_agent.cli verify-qwen --explanation
.venv\Scripts\python -m water_agent.cli analyze-image `
  --image '<由参赛者填写的单张图片路径>' `
  --prompt '请分析这张图片中是否存在水域异常，并说明是否需要人工复核。' `
  --output 'artifacts\e2e\agent_result.json'
```

启动真实演示界面（不需要FastAPI后端）：

```powershell
scripts\start_ui.ps1
```

浏览器访问 `http://127.0.0.1:7860`。审计记录默认写入
`artifacts/logs/agent_audit.sqlite3`；数据库不保存图片和问题正文。

若需要页面调用官方Qwen生成解释，则在运行机器上显式提供密钥文件路径：

```powershell
scripts\start_ui.ps1 -EnableQwen -QwenApiKeyFile '<本机官方API Key文件路径>'
```

Gradio页面本身就是完整的交互式智能体服务：它会在同一进程中创建Agent、调用YOLO工具并返回解释，不需要先启动FastAPI。`scripts\start_api.ps1`仅在需要供其他系统通过HTTP REST接口调用时使用。

### 智能体演示方式

建议使用官方Qwen启动网页，使任务计划和巡查报告由国产大模型参与：

```powershell
scripts\start_ui.ps1 -EnableQwen -QwenApiKeyFile '.\api key.txt'
```

上传图片后，可分别输入以下任务观察不同工具链：

| 输入示例 | 预期计划 |
|---|---|
| `请分析这张图片中有什么水域异常。` | YOLO识别 → 简洁分析 |
| `请生成这张图片的水域巡查报告，并说明是否需要人工复核。` | YOLO识别 → 类别知识查询 → 巡查报告 |
| `这张结果可靠吗？为什么需要复核？` | YOLO识别 → 复核策略 → 简洁分析 |
| 不上传图片，输入`乱堆是什么意思？` | 类别知识查询（不调用YOLO） |

页面中的“智能体任务计划”和“智能体工具轨迹”是可展示的受控决策证据，并非模型隐式思维过程。若Qwen网络调用失败，页面会显示`fallback:*`状态，且YOLO识别仍会稳定返回。

API新增：`POST /api/v1/consult`（无图片类别咨询）和`GET /api/v1/guidance/{label}`（受审核类别知识）。既有`POST /api/v1/analyze`和`POST /api/v1/classify`保持兼容。

## 当前模型验证结果

四折分组OOF共覆盖1,144张训练图片，每张只由未见过它的折模型预测：

| 指标 | 结果 |
|---|---:|
| Accuracy | 97.73% |
| Balanced Accuracy | 71.11% |
| Macro-F1 | 73.43% |
| Top-2 Accuracy | 98.95% |

正常类只有7张且OOF Recall为0，是当前最主要短板。以上是训练集的分组交叉验证结果，不代表初赛测试集成绩。

这些 Recall 来自 `train.json` 的整图标签和四折 OOF 混淆矩阵：乱采24/24、乱建2/4、
乱堆19/23、乱占57/60、有漂浮物1016/1026、正常0/7。当前模型是 YOLO 图像分类
（YOLO-CLS），没有目标框标注，因此不宣称能够定位异常物体的位置。

## 由参赛者执行批量预测

以下命令只有在参赛者明确填写`--input`后才会读取该目录。程序生成赛事提交JSON和同名`.audit.json`概率审计文件；为防止误覆盖，目标文件已存在时会停止。

```powershell
.venv\Scripts\water-agent.exe predict-batch `
  --input '<由参赛者填写的测试图片目录>' `
  --weights `
    'artifacts\training\ablation_fold0_weighted_p05_c8_e15\weights\best.pt' `
    'artifacts\training\fold1_weighted_p05_c8_e15\weights\best.pt' `
    'artifacts\training\fold2_weighted_p05_c8_e15\weights\best.pt' `
    'artifacts\training\fold3_weighted_p05_c8_e15\weights\best.pt' `
  --output 'result\result.json' `
  --image-size 320 `
  --batch 16 `
  --device 0 `
  --temperature 1
```

随后执行提交校验：

```powershell
.venv\Scripts\water-agent.exe validate-submission --result result\result.json
```

初赛测试集没有默认路径，项目开发过程不会扫描该目录。最终批量预测命令由参赛者在本机明确传入输入路径后执行。

## 交付与演示材料

- `DEPLOYMENT.md`：安装、配置、启动、批量推理和决赛平台迁移。
- `MODEL_CARD.md`：数据、四折 OOF 指标、性能、限制和安全使用边界。
- `DEMO_SCRIPT.md`：4–5分钟演示分镜、截图清单和禁止表述。
- `RELEASE_CHECKLIST.md`：队伍身份、密钥、真实结果和归档复核。
- `THIRD_PARTY_NOTICES.md`：核心依赖来源、许可证风险与参赛前合规动作。
- `PHASE5_DELIVERY_REPORT.md`：当前已完成项、质量证据、待用户收口项和最终命令。
- `LOCAL_BATCH_INFERENCE_GUIDE.md`：参赛者本机从环境检查到四折推理、结果核对和最终打包的逐步命令。
- `design/水域智巡智能体设计方案.docx`：基于官方模板生成，参赛队伍已填写为`X-M`且身份占位符审计为0。

参赛者完成真实批量推理并校验结果后，运行：

```powershell
scripts\build_delivery.ps1 `
  -ResultPath 'result\result.json' `
  -OutputPath '水域智巡_submission.tar.gz' `
  -ConfirmRealResult
```

未显式确认真实结果、结果JSON不合法、文档/权重缺失或归档包含非官方条目时，构建器会停止且不生成最终压缩包。

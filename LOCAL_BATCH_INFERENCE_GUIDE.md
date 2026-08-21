# 本地批量推理操作手册

本手册用于由参赛者本人在本机生成初赛 `result/result.json`。批量推理只调用四折 YOLO-CLS，不调用 Qwen，也不需要 API Key。项目开发自动化不会进入测试图片目录；只有你显式传入 `-TestImageDir` 后，脚本才会递归读取该目录。

## 1. 打开 PowerShell 并进入项目根目录

```powershell
Set-Location -LiteralPath 'E:\GPT-Codex\Yolo-water\水域综合异常识别'
$ProjectRoot = (Get-Location).Path
```

后续命令均在此 PowerShell 窗口执行。

## 2. 检查或安装运行环境

先检查项目命令是否已安装：

```powershell
Test-Path -LiteralPath '.\.venv\Scripts\water-agent.exe'
```

返回 `True` 即可继续。若返回 `False`，执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\install.ps1
```

安装过程需要 Python 3.11 和网络。只对当前 PowerShell 进程放宽脚本策略，关闭窗口后自动恢复。

检查依赖：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

正常结果应为 `No broken requirements found.`。

## 3. 检查 GPU 和四个模型权重

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -X utf8 -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

当前项目在 RTX 4070 Laptop 8GB 上验证，推荐 `Batch=16`。四个权重必须全部返回 `True`：

```powershell
$Weights = @(
  'artifacts\training\ablation_fold0_weighted_p05_c8_e15\weights\best.pt',
  'artifacts\training\fold1_weighted_p05_c8_e15\weights\best.pt',
  'artifacts\training\fold2_weighted_p05_c8_e15\weights\best.pt',
  'artifacts\training\fold3_weighted_p05_c8_e15\weights\best.pt'
)
$Weights | ForEach-Object { [pscustomobject]@{ Path = $_; Exists = Test-Path -LiteralPath $_ } }
```

## 4. 由你填写测试图片目录

推荐让 PowerShell 提示输入，避免把路径写入脚本：

```powershell
$TestImageDir = Read-Host '请粘贴测试图片文件夹的完整路径'
if (-not (Test-Path -LiteralPath $TestImageDir -PathType Container)) {
    throw '测试图片目录不存在'
}
```

程序会递归处理 `.jpg`、`.jpeg`、`.png`、`.webp`，忽略其他文件。先只统计数量，不打开图片：

```powershell
$SupportedExtensions = @('.jpg', '.jpeg', '.png', '.webp')
$InputFiles = @(
    Get-ChildItem -LiteralPath $TestImageDir -Recurse -File |
        Where-Object { $SupportedExtensions -contains $_.Extension.ToLowerInvariant() }
)
$ExpectedCount = $InputFiles.Count
Write-Host "待推理图片数：$ExpectedCount"
if ($ExpectedCount -eq 0) {
    throw '指定目录中没有JPG/JPEG/PNG/WebP图片'
}
```

赛事结果只保留文件名，因此不同子目录不能出现同名图片：

```powershell
$DuplicateNames = @($InputFiles | Group-Object Name | Where-Object Count -gt 1)
if ($DuplicateNames.Count -gt 0) {
    $DuplicateNames | Select-Object Name, Count
    throw '存在同名图片，请先确认赛事文件组织方式'
}
```

## 5. 处理旧结果

推理程序拒绝覆盖 `result/result.json` 或 `result/result.audit.json`。若这是第一次运行，无需处理。若已有旧结果，先备份而不是删除：

```powershell
$RunStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$HistoryDir = Join-Path $ProjectRoot "result\history\$RunStamp"
$ExistingResults = @('result\result.json', 'result\result.audit.json') |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
if ($ExistingResults.Count -gt 0) {
    New-Item -ItemType Directory -Path $HistoryDir -Force | Out-Null
    foreach ($ExistingResult in $ExistingResults) {
        Move-Item -LiteralPath $ExistingResult -Destination $HistoryDir
    }
    Write-Host "旧结果已备份到：$HistoryDir"
}
```

## 6. 推荐方式：运行批量脚本

GPU 默认配置：

```powershell
.\scripts\generate_submission.ps1 `
  -TestImageDir $TestImageDir `
  -Batch 16 `
  -Device '0'
```

脚本会完成：

1. 检查虚拟环境、输入目录和四个权重；
2. 四个 YOLO-CLS 模型依次推理；
3. 对同一图片的六类概率做等权平均；
4. 读取图片真实宽高并生成官方四字段记录；
5. 写出 `result/result.json`；
6. 写出本地审计文件 `result/result.audit.json`；
7. 自动执行一次结果 Schema 校验。

推理期间终端可能一段时间没有逐图输出，这是正常现象。最终会打印 `sample_count`、`model_count=4`、耗时和校验结果。

若显存不足，降低 batch 后重试：

```powershell
.\scripts\generate_submission.ps1 -TestImageDir $TestImageDir -Batch 8 -Device '0'
```

没有 NVIDIA GPU 时可以使用 CPU，但会慢很多：

```powershell
.\scripts\generate_submission.ps1 -TestImageDir $TestImageDir -Batch 4 -Device 'cpu'
```

## 7. 二次校验结果

```powershell
.\.venv\Scripts\water-agent.exe validate-submission `
  --result 'result\result.json'
```

必须看到：

```text
"valid": true
```

再核对输入数和结果数：

```powershell
$Submission = @(Get-Content -LiteralPath 'result\result.json' -Raw -Encoding UTF8 | ConvertFrom-Json)
Write-Host "输入图片数：$ExpectedCount"
Write-Host "结果记录数：$($Submission.Count)"
if ($Submission.Count -ne $ExpectedCount) {
    throw '输入图片数与结果记录数不一致'
}
```

核对六类标签：

```powershell
$AllowedLabels = @('乱采', '乱建', '乱堆', '乱占', '有漂浮物', '正常')
$InvalidLabels = @($Submission | Where-Object { $AllowedLabels -notcontains $_.label })
if ($InvalidLabels.Count -gt 0) {
    throw '结果中存在非法标签'
}
```

核对输入与结果文件名集合；命令没有输出即完全一致：

```powershell
$InputNames = @($InputFiles.Name | Sort-Object)
$ResultNames = @($Submission.filename | Sort-Object)
Compare-Object -ReferenceObject $InputNames -DifferenceObject $ResultNames
```

## 8. 审计文件与正式结果的区别

- `result/result.json`：正式赛事结果，只含 `filename`、`width`、`height`、`label`。
- `result/result.audit.json`：本地核查文件，含绝对输入路径、权重路径、各类概率、置信度和耗时。

不要手工修改正式结果。审计文件含本机路径，不应上传；白名单交付构建器只复制 `result/result.json`。

## 9. 可选：直接运行底层 CLI

当脚本需要调试时，可直接执行：

```powershell
.\.venv\Scripts\water-agent.exe predict-batch `
  --input $TestImageDir `
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

推荐正常情况下使用 `generate_submission.ps1`，因为它会自动选择开发版/提交版权重并紧接着执行校验。

## 10. 生成最终提交包

确认 `result.json`真实生成、记录数一致且 `valid=true` 后执行：

```powershell
.\scripts\build_delivery.ps1 `
  -ResultPath 'result\result.json' `
  -OutputPath 'X-M_水域智巡_submission.tar.gz' `
  -ConfirmRealResult
```

检查归档：

```powershell
tar -tzf 'X-M_水域智巡_submission.tar.gz'
Get-FileHash -Algorithm SHA256 -LiteralPath 'X-M_水域智巡_submission.tar.gz'
```

归档顶层只能有 `code/`、`design/`、`result/`，并且 `result/`中只能有 `result.json`。

## 11. 常见问题

- PowerShell 禁止运行脚本：先执行 `Set-ExecutionPolicy -Scope Process Bypass`，只影响当前窗口。
- 报告输出已存在：按第5节备份 `result.json`和`result.audit.json`，不要强制覆盖。
- 输入目录没有支持图片：确认格式属于 JPG/JPEG/PNG/WebP，且目录填写正确。
- 提示存在同名图片：结果 Schema 只保存 basename，不能区分不同子目录的同名文件。
- CUDA out of memory：把 `-Batch 16`降为8或4；关闭其他占用显存的软件后再运行。
- CUDA 不可用：确认当前窗口使用项目 `.venv`，再运行第3节检查；必要时使用 `-Device 'cpu'`。
- 模型类别不一致：不要替换四个正式 `best.pt`；该错误会阻止生成结果。
- Qwen或API Key未配置：不影响批量推理；只有智能体自然语言解释需要Qwen。
- 推理中断：输出在全部模型完成后才写入；先检查是否存在半成品，再重新执行。

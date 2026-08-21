param(
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 7860,
    [switch]$UseFakeModel,
    [switch]$EnableQwen,
    [string]$QwenApiKeyFile
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot '.venv\Scripts\water-agent.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'Project virtual environment not found. Run scripts\install.ps1 first.'
}
$RuntimeRoot = Join-Path $ProjectRoot 'runtime'
$YoloConfig = Join-Path $RuntimeRoot 'ultralytics_config'
$MatplotlibConfig = Join-Path $RuntimeRoot 'matplotlib_config'
New-Item -ItemType Directory -Path $YoloConfig -Force | Out-Null
New-Item -ItemType Directory -Path $MatplotlibConfig -Force | Out-Null
$env:YOLO_CONFIG_DIR = $YoloConfig
$env:MPLCONFIGDIR = $MatplotlibConfig
if (-not $env:WINDIR) {
    $env:WINDIR = 'C:\Windows'
}

if ($UseFakeModel) {
    if ($EnableQwen) {
        throw 'Qwen cannot be enabled together with UseFakeModel.'
    }
    $env:WATER_AGENT_USE_FAKE_MODEL = 'true'
    $env:WATER_AGENT_USE_QWEN = 'false'
} else {
    $PackagedWeights = @('models\fold0.pt', 'models\fold1.pt', 'models\fold2.pt', 'models\fold3.pt')
    $DevelopmentWeights = @(
        'artifacts\training\ablation_fold0_weighted_p05_c8_e15\weights\best.pt',
        'artifacts\training\fold1_weighted_p05_c8_e15\weights\best.pt',
        'artifacts\training\fold2_weighted_p05_c8_e15\weights\best.pt',
        'artifacts\training\fold3_weighted_p05_c8_e15\weights\best.pt'
    )
    $Weights = if (@($PackagedWeights | Where-Object { Test-Path -LiteralPath (Join-Path $ProjectRoot $_) -PathType Leaf }).Count -eq 4) {
        $PackagedWeights
    } elseif (@($DevelopmentWeights | Where-Object { Test-Path -LiteralPath (Join-Path $ProjectRoot $_) -PathType Leaf }).Count -eq 4) {
        $DevelopmentWeights
    } else {
        throw 'Four YOLO model weights were not found in models or artifacts\\training.'
    }
    $env:WATER_AGENT_USE_FAKE_MODEL = 'false'
    $env:WATER_AGENT_ENSEMBLE_WEIGHTS = ConvertTo-Json -InputObject @($Weights) -Compress
    $env:WATER_AGENT_VISION_IMAGE_SIZE = '320'
    $env:WATER_AGENT_VISION_DEVICE = '0'
    if ($EnableQwen) {
        if (-not $QwenApiKeyFile) {
            throw 'Pass -QwenApiKeyFile with the local official API key file path.'
        }
        if (-not (Test-Path -LiteralPath $QwenApiKeyFile -PathType Leaf)) {
            throw 'The specified Qwen API key file does not exist.'
        }
        $env:WATER_AGENT_USE_QWEN = 'true'
        $env:WATER_AGENT_QWEN_API_KEY_FILE = (Resolve-Path -LiteralPath $QwenApiKeyFile).Path
    } else {
        $env:WATER_AGENT_USE_QWEN = 'false'
    }
}
Set-Location -LiteralPath $ProjectRoot
& $Executable serve-ui --host $HostAddress --port $Port

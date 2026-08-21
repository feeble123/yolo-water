param(
    [Parameter(Mandatory = $true)]
    [string]$TestImageDir,
    [string]$OutputPath = 'result\result.json',
    [int]$Batch = 16,
    [string]$Device = '0'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot '.venv\Scripts\water-agent.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'Project virtual environment not found. Run scripts\install.ps1 first.'
}
if (-not (Test-Path -LiteralPath $TestImageDir -PathType Container)) {
    throw 'The specified image directory does not exist.'
}
$ResolvedOutput = Join-Path $ProjectRoot $OutputPath
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "Refusing to overwrite an existing result: $ResolvedOutput"
}
$PackagedWeights = @(
    'models\fold0.pt', 'models\fold1.pt', 'models\fold2.pt', 'models\fold3.pt'
)
$DevelopmentWeights = @(
    'artifacts\training\ablation_fold0_weighted_p05_c8_e15\weights\best.pt',
    'artifacts\training\fold1_weighted_p05_c8_e15\weights\best.pt',
    'artifacts\training\fold2_weighted_p05_c8_e15\weights\best.pt',
    'artifacts\training\fold3_weighted_p05_c8_e15\weights\best.pt'
)
$Weights = $PackagedWeights
$MissingPackaged = @(
    $PackagedWeights | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $ProjectRoot $_) -PathType Leaf)
    }
)
if ($MissingPackaged.Count -gt 0) {
    $Weights = $DevelopmentWeights
}
foreach ($Weight in $Weights) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $Weight) -PathType Leaf)) {
        throw "Missing model weight: $Weight"
    }
}

Set-Location -LiteralPath $ProjectRoot
& $Executable predict-batch `
    --input $TestImageDir `
    --weights $Weights `
    --output $ResolvedOutput `
    --image-size 320 `
    --batch $Batch `
    --device $Device `
    --temperature 1
& $Executable validate-submission --result $ResolvedOutput

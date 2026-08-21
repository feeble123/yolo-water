param(
    [Parameter(Mandatory = $true)]
    [string]$ResultPath,
    [string]$OutputPath = 'water-agent-submission.tar.gz',
    [switch]$ConfirmRealResult
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
if (-not $ConfirmRealResult) {
    throw 'Pass -ConfirmRealResult after confirming the result came from real inference.'
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot '.venv\Scripts\water-agent.exe'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'Project virtual environment not found. Run scripts\install.ps1 first.'
}
Set-Location -LiteralPath $ProjectRoot
& $Executable build-delivery `
    --project-root $ProjectRoot `
    --result $ResultPath `
    --output $OutputPath `
    --confirm-real-result

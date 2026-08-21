$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'

Write-Host 'Install complete. Copy .env.production.example to .env and inject secrets only at runtime.'

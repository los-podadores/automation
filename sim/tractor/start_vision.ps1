param(
    [string]$Distribution = 'Ubuntu-24.04',
    [string]$Weights = ''
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Falta el entorno Python de Windows. En la carpeta automation ejecuta: uv sync'
}
$visionArgs = @('--launch', '--distribution', $Distribution)
if ($Weights) { $visionArgs += '--weights'; $visionArgs += $Weights }
& $pythonPath (Join-Path $PSScriptRoot 'vision.py') @visionArgs
if ($LASTEXITCODE -ne 0) {
    throw 'El visor terminó con un error. Revisa el mensaje anterior y sim/tractor/README.md.'
}

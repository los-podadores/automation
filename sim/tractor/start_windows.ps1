param(
    [string]$Distribution = 'Ubuntu-24.04',
    [switch]$Check,
    [switch]$Obstacles,
    [switch]$Demo,
    [switch]$Route,
    [double]$TimeScale = 1
)
$ErrorActionPreference = 'Stop'
$simWindowsPath = $PSScriptRoot
$simLinuxPath = (& wsl.exe -d $Distribution -- wslpath -u $simWindowsPath)
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo abrir $Distribution. Instala primero: wsl --install -d Ubuntu-24.04. Luego sigue sim/tractor/README.md."
}
$simLinuxPath = $simLinuxPath.Trim()
$simArgs = @()
if ($Check) { $simArgs += '--check' }
if ($Obstacles) { $simArgs += '--obstacles' }
if ($Demo) { $simArgs += '--demo' }
if ($Route) { $simArgs += '--route' }
if ($TimeScale -ne 1) { $simArgs += '--time-scale'; $simArgs += $TimeScale.ToString([System.Globalization.CultureInfo]::InvariantCulture) }
& wsl.exe -d $Distribution -- bash "$simLinuxPath/run.sh" @simArgs
if ($LASTEXITCODE -ne 0) {
    throw 'La simulación no pudo iniciarse. Revisa el mensaje anterior y sim/tractor/README.md.'
}

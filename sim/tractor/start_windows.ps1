param(
    [string]$Distribution = 'Ubuntu-24.04',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$simWindowsPath = $PSScriptRoot
$simLinuxPath = (& wsl.exe -d $Distribution -- wslpath -u $simWindowsPath)
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo abrir $Distribution. Instala primero: wsl --install -d Ubuntu-24.04. Luego sigue sim/tractor/README.md."
}
$simLinuxPath = $simLinuxPath.Trim()
if ($Check) {
    & wsl.exe -d $Distribution -- bash "$simLinuxPath/run.sh" --check
} else {
    & wsl.exe -d $Distribution -- bash "$simLinuxPath/run.sh"
}
if ($LASTEXITCODE -ne 0) {
    throw 'La simulación no pudo iniciarse. Revisa el mensaje anterior y sim/tractor/README.md.'
}

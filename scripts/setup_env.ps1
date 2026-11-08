# Reproduces the orbitopt conda environment from the pinned lockfiles.
# Run from PowerShell:  .\scripts\setup_env.ps1          (CPU-only env "orbitopt")
#                       .\scripts\setup_env.ps1 -Gpu     (CuPy/CUDA 13 env "orbitopt-gpu")
#
# The lockfiles (envs/*.conda-lock.yml) pin every package, including numpy
# (>=1.26.4,<2 -- tudatpy 1.0's own requirement) and scipy (<1.14,
# pykep still calls the removed scipy.interpolate.interp2d). No solve, no
# pip step: install is reproducible. See envs/README.md.

param([switch]$Gpu)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $installer = "$env:TEMP\Miniforge3-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $installer
    Start-Process -FilePath $installer -ArgumentList "/InstallationType=JustMe","/RegisterPython=0","/S","/D=$env:USERPROFILE\miniforge3" -Wait
    $env:Path = "$env:USERPROFILE\miniforge3;$env:USERPROFILE\miniforge3\Scripts;$env:Path"
}

$flavor = if ($Gpu) { "gpu" } else { "cpu" }
$envName = if ($Gpu) { "orbitopt-gpu" } else { "orbitopt" }
$lockfile = Join-Path $PSScriptRoot "..\envs\$flavor.conda-lock.yml"

if (-not (Get-Command conda-lock -ErrorAction SilentlyContinue)) {
    conda install -n base -c conda-forge "conda-lock>=4" -y
}
conda-lock install -n $envName $lockfile

$envPy = "$env:USERPROFILE\miniforge3\envs\$envName\python.exe"
& $envPy -m pip install -e (Join-Path $PSScriptRoot "..") --no-deps

Write-Host "Done. Activate with: conda activate $envName"
Write-Host "Verify with:         $envPy -m pytest tests/ -v"

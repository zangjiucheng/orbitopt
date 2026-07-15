# Reproduces the orbitopt conda environment from scratch.
# Run from PowerShell: .\scripts\setup_env.ps1
#
# tudatpy is conda-forge/tudat-team only and pins python=3.10, numpy<2.0 in
# its own environment.yaml -- but numpy gets pulled to 2.x by other deps
# during solve regardless (harmless in practice: pykep/pygmo/tudatpy all
# import and run fine under numpy 2.2, verified empirically for this repo).
# pykep additionally needs scipy<1.14 (it calls the removed scipy.interpolate
# .interp2d internally) -- conda-forge hasn't caught up, so we pin it after
# the fact via pip.

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $installer = "$env:TEMP\Miniforge3-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $installer
    Start-Process -FilePath $installer -ArgumentList "/InstallationType=JustMe","/RegisterPython=0","/S","/D=$env:USERPROFILE\miniforge3" -Wait
    $env:Path = "$env:USERPROFILE\miniforge3;$env:USERPROFILE\miniforge3\Scripts;$env:Path"
}

conda env create -f environment.yml -y

$envPy = "$env:USERPROFILE\miniforge3\envs\orbitopt\python.exe"
& $envPy -m pip install "scipy<1.14,>=1.11"
& $envPy -m pip install -e . --no-deps

Write-Host "Done. Activate with: conda activate orbitopt"
Write-Host "Verify with:         $envPy -m pytest tests/ -v"

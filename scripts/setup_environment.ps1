param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Python..."
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $pythonCommand) {
    Write-Error "Python was not found. Install Python 3.9+ and ensure it is available as 'python' or 'py'."
}

$pythonExecutable = $pythonCommand.Source
Write-Host "Using Python command: $pythonExecutable"

Write-Host "Creating virtual environment at $VenvPath..."
& $pythonExecutable -m venv $VenvPath

$venvPython = Join-Path $VenvPath "Scripts/python.exe"

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing development dependencies..."
& $venvPython -m pip install -r requirements/dev.txt

Write-Host "Running environment check..."
& $venvPython scripts/check_environment.py

Write-Host ""
Write-Host "Activate the environment with:"
Write-Host "$VenvPath\Scripts\Activate.ps1"

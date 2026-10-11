# SignLang Windows Zero-Prerequisite Bootstrap Installer
# Usage: irm https://raw.githubusercontent.com/RAZOR-NINJAS/signlang/main/install.ps1 | iex
# Or run locally: .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         SignLang Windows Bootstrap Installer            " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Determine Target Directory
$repoUrl = "https://github.com/RAZOR-NINJAS/signlang.git"
$zipUrl = "https://github.com/RAZOR-NINJAS/signlang/archive/refs/heads/main.zip"

if (Test-Path "$PSScriptRoot\pyproject.toml") {
    $installDir = $PSScriptRoot
} elseif (Test-Path ".\pyproject.toml") {
    $installDir = (Get-Item ".").FullName
} else {
    $installDir = "$HOME\Projects\signlang"
    if (-not (Test-Path "$HOME\Projects")) {
        New-Item -ItemType Directory -Path "$HOME\Projects" -Force | Out-Null
    }
}

Write-Host "[1/5] Checking installation directory: $installDir" -ForegroundColor Green

# 2. Check or Install Python 3.11
function Find-Python {
    $candidates = @("py", "python", "python3")
    foreach ($cmd in $candidates) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            try {
                $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                $major, $minor = $ver.Split(".")
                if ([int]$major -eq 3 -and [int]$minor -ge 11 -and [int]$minor -le 12) {
                    return $cmd
                }
            } catch {}
        }
    }
    # Check default AppData locations
    $appDataPy = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (Test-Path $appDataPy) {
        return $appDataPy
    }
    return $null
}

$pythonCmd = Find-Python

if (-not $pythonCmd) {
    Write-Host "[!] Python 3.11 not found. Attempting automatic user-level install..." -ForegroundColor Yellow
    $installed = $false

    # Try winget first (non-admin currentUser scope)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host "    Installing Python 3.11 via winget..." -ForegroundColor Gray
        try {
            winget install --id Python.Python.3.11 -e --scope currentUser --accept-package-agreements --accept-source-agreements --silent
            $installed = $true
        } catch {
            Write-Host "    winget install failed; falling back to direct download." -ForegroundColor Yellow
        }
    }

    if (-not $installed) {
        Write-Host "    Downloading official Python 3.11 installer (user-level, no admin needed)..." -ForegroundColor Gray
        $pyUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        $pyInstaller = "$env:TEMP\python-3.11.9-installer.exe"
        Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller -UseBasicParsing
        Write-Host "    Running Python installer..." -ForegroundColor Gray
        Start-Process -FilePath $pyInstaller -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0" -Wait
        Remove-Item $pyInstaller -Force -ErrorAction SilentlyContinue
    }

    # Refresh PATH environment variable in current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
    $pythonCmd = Find-Python
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Could not automatically locate or install Python 3.11." -ForegroundColor Red
    Write-Host "Please download and install Python 3.11 from: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    Write-Host "Be sure to check 'Add python.exe to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

Write-Host "[2/5] Using Python: $pythonCmd" -ForegroundColor Green

# 3. Check or Clone Repository Source
if (-not (Test-Path "$installDir\pyproject.toml")) {
    Write-Host "[3/5] Fetching SignLang repository..." -ForegroundColor Green
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd) {
        Write-Host "    Cloning via git..." -ForegroundColor Gray
        git clone $repoUrl $installDir
    } else {
        Write-Host "    Git not found. Downloading repository ZIP..." -ForegroundColor Gray
        $zipFile = "$env:TEMP\signlang.zip"
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile -UseBasicParsing
        $tempExtract = "$env:TEMP\signlang_extracted"
        Expand-Archive -Path $zipFile -DestinationPath $tempExtract -Force
        $innerDir = Get-ChildItem -Path $tempExtract -Directory | Select-Object -First 1
        Move-Item -Path $innerDir.FullName -Destination $installDir -Force
        Remove-Item $zipFile, $tempExtract -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "[3/5] Repository files detected in: $installDir" -ForegroundColor Green
}

Set-Location $installDir

# 4. Create Virtual Environment and Install Dependencies
Write-Host "[4/5] Setting up virtual environment (.venv)..." -ForegroundColor Green
$venvPy = "$installDir\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    & $pythonCmd -m venv "$installDir\.venv"
}

Write-Host "    Installing dependencies into virtual environment..." -ForegroundColor Gray
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -e $installDir --quiet
& $venvPy -m pip install rich --quiet

# 5. Hand off to Python TUI Installer
Write-Host "[5/5] Launching interactive setup TUI..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& $venvPy "$installDir\scripts\install_tui.py" @args

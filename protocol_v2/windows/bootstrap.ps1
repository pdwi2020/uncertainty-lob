param(
    [string]$Root = "G:\DL_Paper_v2",
    [string]$CodeZip = "$env:USERPROFILE\Downloads\dl_paper_v2_code.zip",
    [string]$DataZip = "$env:USERPROFILE\Downloads\dl_paper_v2_data.zip"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

if (-not (Test-Path $CodeZip)) {
    throw "Code archive not found: $CodeZip"
}
if (-not (Test-Path $DataZip)) {
    throw "Data archive not found: $DataZip"
}

Write-Host "[1/8] Extracting Protocol V2 code"
Expand-Archive -Path $CodeZip -DestinationPath $Root -Force

Write-Host "[2/8] Extracting cached FI-2010 matrices"
$DataStage = Join-Path $Root "data_stage"
Remove-Item $DataStage -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $DataZip -DestinationPath $DataStage -Force
$ZScoreRoot = Join-Path $Root "data\fi-2010\BenchmarkDatasets\NoAuction\1.NoAuction_Zscore"
$TrainDir = Join-Path $ZScoreRoot "NoAuction_Zscore_Training"
$TestDir = Join-Path $ZScoreRoot "NoAuction_Zscore_Testing"
New-Item -ItemType Directory -Force -Path $TrainDir, $TestDir | Out-Null
Get-ChildItem $DataStage -Filter "Train_Dst_NoAuction_ZScore_CF_*.npy" -Recurse |
    Copy-Item -Destination $TrainDir -Force
Get-ChildItem $DataStage -Filter "Test_Dst_NoAuction_ZScore_CF_*.npy" -Recurse |
    Copy-Item -Destination $TestDir -Force
Remove-Item $DataStage -Recurse -Force

Write-Host "[3/8] Locating Python environment tooling"
$CondaCandidates = @(
    "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "C:\Anaconda3\Scripts\conda.exe"
)
$Conda = $CondaCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Conda) {
    throw "Conda was not found."
}

Write-Host "[4/8] Creating isolated Python 3.11 environment"
$Environment = Join-Path $Root ".conda"
if (-not (Test-Path (Join-Path $Environment "python.exe"))) {
    & $Conda create --yes --prefix $Environment python=3.11 pip
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Conda failed; installing official Python 3.11 per-project instead."
        Remove-Item $Environment -Recurse -Force -ErrorAction SilentlyContinue
        $Installer = Join-Path $Root "python-3.11.9-amd64.exe"
        Invoke-WebRequest `
            -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" `
            -OutFile $Installer `
            -UseBasicParsing
        $InstallProcess = Start-Process `
            -FilePath $Installer `
            -ArgumentList "/quiet InstallAllUsers=0 TargetDir=$Environment Include_pip=1 Include_test=0 PrependPath=0 Shortcuts=0" `
            -PassThru `
            -Wait
        if ($InstallProcess.ExitCode -ne 0) {
            throw "Python installer failed with exit code $($InstallProcess.ExitCode)"
        }
        Remove-Item $Installer -Force
    }
}
$Python = Join-Path $Environment "python.exe"

Write-Host "[5/8] Installing PyTorch CUDA runtime"
& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }

Write-Host "[6/8] Installing Protocol V2 dependencies"
& $Python -m pip install -r (Join-Path $Root "protocol_v2\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Write-Host "[7/8] Running tests and protocol audit"
$env:DL_PAPER_ROOT = $Root
Set-Location $Root
& $Python -m unittest discover -s protocol_v2\tests -v
if ($LASTEXITCODE -ne 0) { throw "Protocol V2 tests failed" }
& $Python -m protocol_v2.audit `
    --data-root (Join-Path $Root "data\fi-2010") `
    --cache-root (Join-Path $Root "cache\fi2010") `
    --output (Join-Path $Root "runs\protocol_audit.json")
if ($LASTEXITCODE -ne 0) { throw "FI-2010 protocol audit failed" }

Write-Host "[8/8] Verifying CUDA"
& $Python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed" }

Write-Host "Bootstrap complete: $Root"

param(
    [string]$Root = "G:\DL_Paper_v2",
    [string]$Campaign = "protocol_v2\config\campaign_long.yaml"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Root ".conda\python.exe"
$Runs = Join-Path $Root "runs"
$Logs = Join-Path $Root "service_logs"
$ServicesPath = Join-Path $Runs "services.json"
$CampaignPath = Join-Path $Root $Campaign

if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path $CampaignPath)) {
    throw "Campaign configuration not found: $CampaignPath"
}

$Services = if (Test-Path $ServicesPath) {
    Get-Content $ServicesPath -Raw | ConvertFrom-Json
} else {
    [pscustomobject]@{}
}

if ($Services.campaign) {
    $Existing = Get-Process -Id $Services.campaign -ErrorAction SilentlyContinue
    if ($Existing) {
        throw "Campaign PID $($Services.campaign) is still running. Refusing to create a duplicate controller."
    }
}

$TrainingChildren = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object CommandLine -Match "protocol_v2.train"
if ($TrainingChildren) {
    $Ids = ($TrainingChildren | Select-Object -ExpandProperty ProcessId) -join ", "
    throw "Training child process still running ($Ids). Let it finish before restarting the controller."
}

$env:DL_PAPER_ROOT = $Root
$env:DL_PAPER_RUNS_ROOT = $Runs
Set-Location $Root

$Process = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "protocol_v2.campaign", "--config", $CampaignPath
) -RedirectStandardOutput (Join-Path $Logs "campaign.out.log") `
  -RedirectStandardError (Join-Path $Logs "campaign.err.log") `
  -WindowStyle Hidden -PassThru

$Services | Add-Member -NotePropertyName campaign -NotePropertyValue $Process.Id -Force
$Services | Add-Member -NotePropertyName started_at -NotePropertyValue (Get-Date).ToString("o") -Force
$Services | ConvertTo-Json | Set-Content $ServicesPath

Write-Host "Campaign controller PID: $($Process.Id)"
Write-Host "Run status.ps1 to verify progress and failed count."

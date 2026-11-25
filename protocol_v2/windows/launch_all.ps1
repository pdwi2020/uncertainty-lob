param(
    [string]$Root = "G:\DL_Paper_v2",
    [string]$Campaign = "protocol_v2\config\campaign_long.yaml",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Root ".conda\python.exe"
$Runs = Join-Path $Root "runs"
$Logs = Join-Path $Root "service_logs"
New-Item -ItemType Directory -Force -Path $Runs, $Logs | Out-Null

if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. Run bootstrap.ps1 first."
}

$env:DL_PAPER_ROOT = $Root
$env:DL_PAPER_RUNS_ROOT = $Runs
Set-Location $Root

$CollectorOut = Join-Path $Logs "collector.out.log"
$CollectorErr = Join-Path $Logs "collector.err.log"
$DashboardOut = Join-Path $Logs "dashboard.out.log"
$DashboardErr = Join-Path $Logs "dashboard.err.log"
$CampaignOut = Join-Path $Logs "campaign.out.log"
$CampaignErr = Join-Path $Logs "campaign.err.log"

$Collector = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "protocol_v2.monitor.collector", "--runs-root", $Runs, "--interval", "5"
) -RedirectStandardOutput $CollectorOut -RedirectStandardError $CollectorErr -PassThru

$Dashboard = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "streamlit", "run", "protocol_v2\monitor\app.py",
    "--server.address", "127.0.0.1", "--server.port", "$Port",
    "--server.headless", "true", "--browser.gatherUsageStats", "false"
) -RedirectStandardOutput $DashboardOut -RedirectStandardError $DashboardErr -PassThru

$CampaignPath = Join-Path $Root $Campaign
$CampaignProcess = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "protocol_v2.campaign", "--config", $CampaignPath
) -RedirectStandardOutput $CampaignOut -RedirectStandardError $CampaignErr -PassThru

@{
    collector = $Collector.Id
    dashboard = $Dashboard.Id
    campaign = $CampaignProcess.Id
    dashboard_url = "http://127.0.0.1:$Port"
    started_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content (Join-Path $Runs "services.json")

Write-Host "Collector PID: $($Collector.Id)"
Write-Host "Dashboard PID: $($Dashboard.Id)"
Write-Host "Campaign PID:  $($CampaignProcess.Id)"
Write-Host "Dashboard: http://127.0.0.1:$Port"

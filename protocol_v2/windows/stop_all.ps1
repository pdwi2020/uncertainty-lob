param([string]$Root = "G:\DL_Paper_v2")

$ServicesPath = Join-Path $Root "runs\services.json"
if (-not (Test-Path $ServicesPath)) {
    Write-Host "No services file found."
    exit 0
}

$Services = Get-Content $ServicesPath -Raw | ConvertFrom-Json
foreach ($Name in @("campaign", "dashboard", "collector")) {
    $Id = $Services.$Name
    if ($Id) {
        Stop-Process -Id $Id -ErrorAction SilentlyContinue
        Write-Host "Stopped $Name PID $Id"
    }
}

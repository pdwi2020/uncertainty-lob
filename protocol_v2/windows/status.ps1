param([string]$Root = "G:\DL_Paper_v2")

$ServicesPath = Join-Path $Root "runs\services.json"
if (Test-Path $ServicesPath) {
    $Services = Get-Content $ServicesPath -Raw | ConvertFrom-Json
    $Services.PSObject.Properties | ForEach-Object {
        if ($_.Name -in @("collector", "dashboard", "campaign")) {
            $Process = Get-Process -Id $_.Value -ErrorAction SilentlyContinue
            [pscustomobject]@{
                Service = $_.Name
                PID = $_.Value
                Running = [bool]$Process
                CPU = if ($Process) { [math]::Round($Process.CPU, 1) } else { $null }
            }
        }
    } | Format-Table -AutoSize
}

Get-ChildItem (Join-Path $Root "runs") -Filter campaign_status.json -Recurse |
    ForEach-Object { Get-Content $_.FullName -Raw | ConvertFrom-Json } |
    Select-Object campaign, state, current_index, total_runs, completed, failed, current_run, updated_at |
    Format-Table -AutoSize

& nvidia-smi

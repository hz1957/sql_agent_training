$Root = Split-Path -Parent $PSScriptRoot
$LogPath = Join-Path $Root "logs\agent.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
if (-not (Test-Path $LogPath)) {
  New-Item -ItemType File -Path $LogPath | Out-Null
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Get-Content '$LogPath' -Wait -Tail 200"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 2"

$ServiceCommand = @'
while ($true) {
  Clear-Host
  Get-Date
  Get-Process python,ray,vllm,sglang -ErrorAction SilentlyContinue |
    Select-Object ProcessName,Id,CPU,WS |
    Format-Table
  Start-Sleep 2
}
'@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServiceCommand

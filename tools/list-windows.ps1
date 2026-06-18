ocs = Get-Process | Where-Object { $_.MainWindowTitle -ne '' } | Select-Object Id, ProcessName, MainWindowTitle
foreach ($p in $procs) {
    Write-Host ("PID=" + $p.Id + " Name=" + $p.ProcessName + " Title='" + $p.MainWindowTitle + "'")
}
$lv = Get-Process -Name "*lightvc*" -ErrorAction SilentlyContinue
if ($lv) {
    Write-Host "---"
    Write-Host ("lightvc-app PID: " + $lv.Id + " Title: '" + $lv.MainWindowTitle + "'")
}

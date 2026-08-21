$real = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
'--- resolucao de PATH ---'
foreach ($n in 'python','python3') {
  $c = Get-Command $n -ErrorAction SilentlyContinue
  '{0,-8} {1}' -f $n, $(if ($c) { $c.Source } else { '(ausente)' })
}
'--- binario real instalado ---'
if (Test-Path $real) { & $real --version } else { "NAO ENCONTRADO: $real" }
'--- self-checks com o binario real ---'
Push-Location 'C:\projects\tare.tools.local-labs\src'
& $real -m model_lifecycle.collectors.host
& $real -m model_lifecycle.analysis.statistics
Pop-Location

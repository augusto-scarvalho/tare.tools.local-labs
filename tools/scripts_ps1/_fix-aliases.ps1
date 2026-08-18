# Remove the Microsoft Store "app execution alias" stubs that hijack python/python3,
# then provide a REAL python3.exe.
#
# Safety: this folder holds aliases for other apps too (winget lives here), so nothing
# is deleted unless it is confirmed to be a 0-byte reparse-point stub. A real binary is
# never touched.
$ErrorActionPreference = 'Stop'
$apps = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps'
$py   = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'

if (-not (Test-Path $py)) { throw "real python missing at $py - refusing to touch aliases" }

foreach ($name in 'python.exe','python3.exe') {
    $stub = Join-Path $apps $name
    if (-not (Test-Path $stub)) { "skip  $name (nao existe)"; continue }
    $item = Get-Item $stub -Force
    $isStub = ($item.Length -eq 0) -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
    if (-not $isStub) { "KEEP  $name (nao e stub: $($item.Length) bytes) - nao removido"; continue }
    Remove-Item $stub -Force
    "removed  $name (stub de $($item.Length) bytes)"
}

# Windows Python ships python.exe only; scripts and habits expect python3. A copy in
# the install dir behaves identically (the interpreter locates its home from its own
# path), and needs no admin or developer mode -- unlike a symlink.
$py3 = Join-Path (Split-Path $py) 'python3.exe'
if (-not (Test-Path $py3)) {
    Copy-Item $py $py3
    "created  $py3"
} else {
    "exists   $py3"
}

'--- resolucao final ---'
foreach ($n in 'python','python3','winget') {
    $c = Get-Command $n -ErrorAction SilentlyContinue
    '{0,-8} {1}' -f $n, $(if ($c) { $c.Source } else { '(ausente)' })
}

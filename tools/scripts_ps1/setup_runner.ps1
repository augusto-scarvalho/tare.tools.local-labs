param(
    [string]$Token = "AFA6C6A3Z5UX7X77D4NNHKDKQMGIM",
    [string]$RepoUrl = "https://github.com/tare-research/tare.tools"
)

$RunnerDir = "C:\actions-runner"
if (-not (Test-Path $RunnerDir)) {
    New-Item -ItemType Directory -Path $RunnerDir -Force | Out-Null
}
Set-Location $RunnerDir

if (-not (Test-Path "$RunnerDir\config.cmd")) {
    Write-Host "Downloading GitHub Actions Runner v2.322.0..."
    $DownloadUrl = "https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-win-x64-2.322.0.zip"
    $ZipPath = "$RunnerDir\actions-runner.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath
    Write-Host "Extracting runner..."
    Expand-Archive -Path $ZipPath -DestinationPath $RunnerDir -Force
    Remove-Item $ZipPath -Force
}

Write-Host "Configuring GitHub Actions Runner for $RepoUrl..."
cmd.exe /c ".\config.cmd --url $RepoUrl --token $Token --name aaaaa-desktop --work _work --unattended --replace --runasservice"

Write-Host "Starting runner service..."
cmd.exe /c ".\actions.runner.service.bat start"
Write-Host "GitHub Actions Runner setup complete!"

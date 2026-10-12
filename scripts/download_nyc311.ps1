# scripts/download_nyc311.ps1
$ErrorActionPreference = "Stop"

$targetDir = "data\raw\nyc311"
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$url = "https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv?accessType=DOWNLOAD&bom=true&query=select+*+limit+100000"
$output = Join-Path $targetDir "nyc311_sample.csv"

Write-Host "Downloading NYC 311 sample..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing

$size = (Get-Item $output).Length / 1MB
Write-Host "Downloaded $([math]::Round($size, 2)) MB to $output" -ForegroundColor Green
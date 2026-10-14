
# scripts/download_metr_la.ps1
$ErrorActionPreference = "Stop"

$dataRoot = "C:\Users\Janak Mallik\ml-data"
$targetDir = Join-Path $dataRoot "metr-la"

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

Write-Host "Downloading METR-LA dataset to $targetDir" -ForegroundColor Cyan

# Source files from the DCRNN project.
# The HDF5 file uses a verified Hugging Face mirror because the
# original GitHub download connection failed in this environment.
$files = @(
    @{
        Name = "metr-la.h5"
        Url  = "https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/metr-la.h5"
        MinBytes = 1000000
    },
    @{
        Name = "adj_mx.pkl"
        Url  = "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl"
        MinBytes = 10000
    },
    @{
        Name = "graph_sensor_locations.csv"
        Url  = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv"
        MinBytes = 1000
    }
)

$failed = @()

foreach ($file in $files) {
    $name = $file.Name
    $url = $file.Url
    $out = Join-Path $targetDir $name
    $temp = "$out.download"
    $validExistingFile = $false

    # Keep an existing file only if it meets the minimum size check.
    if (Test-Path $out) {
        $existing = Get-Item $out

        if ($existing.Length -ge $file.MinBytes) {
            Write-Host "  Existing file found: $name ($($existing.Length) bytes)" -ForegroundColor Yellow
            $validExistingFile = $true
        }
        else {
            Write-Host "  Existing file is too small; downloading again: $name" -ForegroundColor Yellow
            Remove-Item $out -Force
        }
    }

    if ($validExistingFile) {
        continue
    }

    Remove-Item $temp -Force -ErrorAction SilentlyContinue

    $downloaded = $false

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Write-Host "  Downloading $name (attempt $attempt of 3)..." -ForegroundColor Cyan

            # Use a fresh temporary file so failed downloads cannot
            # leave a partial file under the final filename.
            Remove-Item $temp -Force -ErrorAction SilentlyContinue

            Invoke-WebRequest `
                -Uri $url `
                -OutFile $temp `
                -UseBasicParsing `
                -TimeoutSec 180

            if (-not (Test-Path $temp)) {
                throw "The download did not create an output file."
            }

            $size = (Get-Item $temp).Length

            if ($size -lt $file.MinBytes) {
                throw "Downloaded file is too small ($size bytes)."
            }

            Move-Item $temp $out -Force

            Write-Host "  Downloaded $name ($size bytes)" -ForegroundColor Green
            $downloaded = $true
            break
        }
        catch {
            Write-Host "  Attempt $attempt failed: $($_.Exception.Message)" -ForegroundColor Red
            Remove-Item $temp -Force -ErrorAction SilentlyContinue

            if ($attempt -lt 3) {
                Start-Sleep -Seconds (2 * $attempt)
            }
        }
    }

    if (-not $downloaded) {
        $failed += $name
        Write-Host "  FAILED: $name" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Download summary" -ForegroundColor Cyan
Get-ChildItem $targetDir -File |
    Select-Object Name, Length |
    Format-Table -AutoSize

if ($failed.Count -gt 0) {
    Write-Host "Download failed for: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "All required files are present and passed minimum-size checks." -ForegroundColor Green
Write-Host "Note: minimum-size checks do not replace content validation." -ForegroundColor Yellow

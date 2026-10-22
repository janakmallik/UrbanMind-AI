# Download UCF-Crime — manual guide.
#
# UCF-Crime is ~120 GB uncompressed. The project only uses the Test
# split (~10 GB). The official site requires a request form.

$ErrorActionPreference = "Stop"
$TargetDir = "C:\Users\Janak Mallik\ml-data\ucf_crime"
$Marker = Join-Path $TargetDir ".downloaded"

if (Test-Path $Marker) {
    Write-Host "UCF-Crime already marked at $TargetDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host ""
Write-Host "UCF-Crime requires a manual request:"
Write-Host ""
Write-Host "  Official: https://www.crcv.ucf.edu/projects/real-world/"
Write-Host "  Fill out the request form. Download link arrives in 1-2 days."
Write-Host "  Extract the Test split into: $TargetDir\Test\"
Write-Host ""
Write-Host "  Alternative: search Kaggle for 'UCF Crime dataset' — several"
Write-Host "  mirrors exist (~10 GB). Only use mirrors you trust."
Write-Host ""

$choice = Read-Host "Type 'y' when downloaded manually: "
if ($choice -ne "y") {
    Write-Host "Exiting. Run again when ready."
    exit 1
}

New-Item -ItemType File -Force -Path $Marker | Out-Null
Write-Host ""
Write-Host "Done. Next:"
Write-Host "  uv run python scripts/build_ucf_clips.py"
Write-Host "  uv run python scripts/train_ucf_crime_4class.py"
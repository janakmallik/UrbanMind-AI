# Download BDD100K — manual download guide.
#
# BDD100K's official server (https://bdd-data.berkeley.edu/) requires
# a login, so this script cannot auto-download the full 100K dataset.
# This script explains how to get the data and marks it as done once
# the user confirms.

$ErrorActionPreference = "Stop"
$TargetDir = "C:\Users\Janak Mallik\ml-data\bdd100k"
$Marker = Join-Path $TargetDir ".downloaded"

if (Test-Path $Marker) {
    Write-Host "BDD100K already marked as downloaded at $TargetDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host ""
Write-Host "BDD100K requires a free account at https://bdd-data.berkeley.edu/"
Write-Host ""
Write-Host "Steps:"
Write-Host "  1. Sign up / log in at the Berkeley site"
Write-Host "  2. Download '100K Images' and 'Labels'"
Write-Host "  3. Extract into: $TargetDir"
Write-Host "     (expect subfolders 'bdd100k/images/100k/train' and"
Write-Host "      'bdd100k/labels/100k/train' after extraction)"
Write-Host ""

$choice = Read-Host "Type 'y' when done (or 'n' to exit): "
if ($choice -ne "y") {
    Write-Host "Exiting. Run again when ready."
    exit 1
}

New-Item -ItemType File -Force -Path $Marker | Out-Null
Write-Host ""
Write-Host "Done. Next:"
Write-Host "  uv run python scripts/train_bdd100k_resnet50.py    # Week 5"
Write-Host "  uv run python scripts/train_bdd100k_segformer.py   # Week 6"
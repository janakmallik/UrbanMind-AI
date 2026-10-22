# Download HMDB-51 — manual guide.
#
# The original MPII/SERRE-Lab site is unreliable. The project's loaders
# expect the HF datasets mirror or a manual .avi download.
# This script explains both.

$ErrorActionPreference = "Stop"
$TargetDir = "C:\Users\Janak Mallik\ml-data\hmdb51"
$Marker = Join-Path $TargetDir ".downloaded"

if (Test-Path $Marker) {
    Write-Host "HMDB-51 already at $TargetDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host ""
Write-Host "HMDB-51 options:"
Write-Host ""
Write-Host "  [1] HuggingFace loader (recommended, ~2 GB download)"
Write-Host "      Downloads to ~/.cache/huggingface/ automatically."
Write-Host "  [2] Manual .avi download from SERRE-Lab"
Write-Host ""
$choice = Read-Host "Choose [1=HF, 2=manual]: "

if ($choice -eq "1") {
    Write-Host "Running HF download (this can take 5-10 min)..."
    uv run python -c "from datasets import load_dataset; load_dataset('Serrelab/hmdb51', split='train'); print('Done. Cached in ~/.cache/huggingface/')"
    New-Item -ItemType File -Force -Path $Marker | Out-Null
} elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "Download .rar files from:"
    Write-Host "  https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/"
    Write-Host "Extract into: $TargetDir\raw\"
    $done = Read-Host "Type 'y' when done: "
    if ($done -eq "y") { New-Item -ItemType File -Force -Path $Marker | Out-Null }
} else {
    Write-Host "Invalid choice."
    exit 1
}

Write-Host ""
Write-Host "Done. Next: uv run python scripts/train_hmdb51_r2plus1d.py"
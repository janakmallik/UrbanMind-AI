# Download UrbanSound8K parquet shards from HuggingFace mirror.
#
# Source: https://huggingface.co/datasets/jaimon/UrbanSound8K
# Target: C:\Users\Janak Mallik\ml-data\urbansound8k
#
# After this script runs, `scripts/precompute_audio.py` builds the
# mel-spectrogram cache (~3 GB, ~3-4 min).

$ErrorActionPreference = "Stop"
$TargetDir = "C:\Users\Janak Mallik\ml-data\urbansound8k"
$Marker = Join-Path $TargetDir ".downloaded"

if (Test-Path $Marker) {
    Write-Host "UrbanSound8K already at $TargetDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path "$TargetDir\data" | Out-Null
$base = "https://huggingface.co/datasets/jaimon/UrbanSound8K/resolve/main/data"

foreach ($i in 0..15) {
    $f = "train-{0:d5}-of-00016.parquet" -f $i
    $out = Join-Path $TargetDir "data\$f"
    if (Test-Path $out) { Write-Host "  skip $f"; continue }
    Write-Host "  downloading $f"
    Invoke-WebRequest -Uri "$base/$f" -OutFile $out -UseBasicParsing
}

New-Item -ItemType File -Force -Path $Marker | Out-Null
Write-Host ""
Write-Host "Done. Next: uv run python scripts/precompute_audio.py"
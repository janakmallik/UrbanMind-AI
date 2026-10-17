"""GPU capability test for RTX 4050 (6.44 GB VRAM).

Run one model at a time to avoid CUDA context corruption:
    uv run python scripts/gpu_capability_test.py --model resnet50
    uv run python scripts/gpu_capability_test.py --model vit
    uv run python scripts/gpu_capability_test.py --model segformer
    uv run python scripts/gpu_capability_test.py --model 3dcnn
    uv run python scripts/gpu_capability_test.py --model tinyllama
    uv run python scripts/gpu_capability_test.py --model diffusion
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_gpu_info() -> None:
    if DEVICE != "cuda":
        print("No CUDA device available.")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"  Total VRAM:    {props.total_memory / 1e9:.2f} GB")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  CUDA:          {torch.version.cuda}\n")


def try_fit(model, input_shape, start_batch=1, max_batch=128, steps=3, backward=True, label=""):
    print(f"[{label}]")
    best_batch = 0
    batch = start_batch
    while batch <= max_batch:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.train() if backward else model.eval()

            x = torch.randn(batch, *input_shape, device=DEVICE)
            if backward:
                out = model(x)
                loss = out.sum()
                loss.backward()
                model.zero_grad()
            else:
                with torch.no_grad():
                    model(x)
            del x

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            for _ in range(steps):
                x = torch.randn(batch, *input_shape, device=DEVICE)
                if backward:
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    model.zero_grad()
                else:
                    with torch.no_grad():
                        model(x)
                del x
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / steps
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  batch={batch:>3}  ok  peak={peak:.2f} GB  {batch / elapsed:.1f} it/s")
            best_batch = batch
            batch *= 2
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"  batch={batch:>3}  OOM")
            else:
                print(f"  batch={batch:>3}  error: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            break
    print(f"  -> max batch that fits: {best_batch}\n")


def test_resnet50():
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
    try_fit(model, (3, 224, 224), label="ResNet-50 @ 224x224 (fine-tune)")


def test_vit():
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1).to(DEVICE)
    try_fit(model, (3, 384, 384), label="ViT-B/16 @ 384x384 (fine-tune)")


class SegFormerWrapper(nn.Module):
    """Wraps SegFormer so forward returns logits tensor, not dataclass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def test_segformer():
    from transformers import SegformerForSemanticSegmentation

    base = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=3,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model = SegFormerWrapper(base)
    try_fit(model, (3, 512, 512), label="SegFormer-B0 @ 512x512 (fine-tune)")


def test_3dcnn():
    class Tiny3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(64, 5),
            )

        def forward(self, x):
            return self.conv(x)

    model = Tiny3DCNN().to(DEVICE)
    try_fit(model, (3, 16, 112, 112), label="Tiny 3D CNN @ 16×112×112 (video proxy)")


def test_tinyllama():
    from transformers import AutoModelForCausalLM

    print("[TinyLlama 1.1B — inference only]")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model.eval()
        input_ids = torch.randint(0, 32000, (1, 256), device=DEVICE)
        with torch.no_grad():
            model(input_ids=input_ids)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  forward seq=256: ok, peak={peak:.2f} GB")
        print(f"  LoRA fine-tune likely feasible at seq<=256, batch=1\n")
        del model
    except Exception as e:
        print(f"  failed: {e}\n")


def test_diffusion():
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    model = TinyUNet().to(DEVICE)
    try_fit(model, (3, 32, 32), label="Tiny U-Net @ 32x32 (diffusion proxy)")


MODELS = {
    "resnet50": test_resnet50,
    "vit": test_vit,
    "segformer": test_segformer,
    "3dcnn": test_3dcnn,
    "tinyllama": test_tinyllama,
    "diffusion": test_diffusion,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()
    print_gpu_info()
    MODELS[args.model]()


if __name__ == "__main__":
    main()

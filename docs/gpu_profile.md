# GPU Profile — RTX 4050 Laptop (6.44 GB VRAM)

Measured 2026-10-17 on PyTorch 2.6.0+cu124, CUDA 12.4.
Feasibility assessment for UrbanMind-AI weeks 6–12.

## Capability matrix

| Model | Input | Max batch | Peak VRAM @ max | Safe batch | Safe VRAM | Throughput |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | 224×224 | 64 | 5.70 GB | **32** | **2.92 GB** | 143 it/s |
| ViT-B/16 | 384×384 | 32* | 12.32 GB | **8** | **3.40 GB** | 13.7 it/s |
| SegFormer-B0 | 512×512 | 32* | 11.53 GB | **8** | **2.91 GB** | 26.7 it/s |
| Tiny 3D CNN | 16×112×112 | 128 | 5.26 GB | **64** | **2.64 GB** | 285 it/s |
| TinyLlama 1.1B | seq 256 | 1 | 2.25 GB | **1** | **2.25 GB** | — |
| Tiny U-Net | 32×32 | 128 | 0.38 GB | **128** | **0.38 GB** | 3072 it/s |

*Batches 16 (ViT) and 32 (SegFormer) exceed physical VRAM and swap to system RAM, dropping throughput 5–10×. Use the safe batch column for actual training.

## Implications for Week 6–12

| Week | Task | Model | Safe batch | Est. time/epoch |
|---|---|---|---:|---:|
| 6 | Segmentation | SegFormer-B0 @ 512 | 8 | ~26 min (69k) |
| 6 | Detection | YOLOv8n @ 640 | 8 | ~30 min (7k) |
| 7 | Video | Tiny 3D CNN @ 16×112×112 | 64 | ~40 min (2.5k clips) |
| 8 | LLM LoRA | TinyLlama 1.1B | 1, seq 256 | ~2h (5k examples) |
| 9 | Multimodal fusion | Frozen embeddings | n/a | ~10 min |
| 10 | Uncertainty | Post-hoc | n/a | ~2 min |
| 11 | Diffusion | Tiny U-Net @ 32 | 64 | ~15 min |
| 12 | Deployment | FastAPI | CPU only | n/a |

All weeks feasible. Nothing on the roadmap is blocked by hardware.

## Notes

- **ResNet-50 is 10× faster than ViT** at 224×224 (143 vs 13.7 it/s). Use ResNet for quick experiments.
- **ViT needs batch 8 max** to stay under 6.44 GB.
- **SegFormer-B0 at 512×512** fits batch 8 comfortably; batch 16 is near the limit.
- **TinyLlama** at seq 256 uses 2.25 GB — LoRA fine-tune feasible.
- **Tiny U-Net** is nearly free — diffusion at 32×32 is trivial.

## Measurement script

`scripts/gpu_capability_test.py --model <name>` can re-run these tests.

Known caveat: CUDA context corruption on OOM means each model should be run in a separate process invocation.
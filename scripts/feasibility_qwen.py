"""Feasibility check — does Qwen2.5-1.5B fit in 6 GB VRAM?

Loads the model, runs forward passes at increasing sequence lengths,
reports peak VRAM. If it OOMs at a needed size, we know before
committing to the generator.
"""

from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(
        f"VRAM free at start: "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB"
    )
    print()

    print(f"Loading {args.model} (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
    )
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() / 1e9
    print(f"After load: {used:.2f} GB")

    for seq_len in [256, 512, 1024, 2048]:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device="cuda")
            with torch.no_grad():
                _ = model(input_ids)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  seq_len={seq_len:>4}  OK    peak VRAM={peak:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  seq_len={seq_len:>4}  OOM")
            break

    print()
    print("Feasibility summary:")
    print(f"  - Qwen2.5-1.5B loads in 4-bit at {used:.2f} GB")
    print(f"  - Supports sequences up to at least {seq_len} tokens")
    print(f"  - Peak VRAM at max tested length: {peak:.2f} GB")
    print(f"  - Headroom: {(torch.cuda.get_device_properties(0).total_memory / 1e9) - peak:.2f} GB")


if __name__ == "__main__":
    main()

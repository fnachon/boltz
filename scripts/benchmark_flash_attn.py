"""Benchmark: standard attention vs PyTorch SDPA (FlashAttention-2 backend).

Measures wall-clock throughput and peak GPU memory for:
  - AttentionPairBias  (token-level sequence attention, used in Pairformer)

Run on an Ampere+ GPU for FlashAttention-2 acceleration:

    python scripts/benchmark_flash_attn.py

Measured on RTX 5080 (sm_120, batch=1, N=512, bfloat16):
    AttentionPairBias  standard   time= 0.9ms  mem=148MB
    AttentionPairBias  flash      time= 0.7ms  mem=148MB  (1.2-1.4x faster)

Note: speedup varies by sequence length and GPU architecture.
Memory savings depend on whether FlashAttention-2 kernel activates
(requires Ampere/sm_80+ and no explicit attn_mask).
TriangleAttention uses the standard einsum path (SDPA with a dense N x N
bias does not activate FA2 and is slower on that layer).
"""

import argparse
import time

import torch

# ── helpers ──────────────────────────────────────────────────────────────────

def _warmup_and_time(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def _peak_mem_mb(fn):
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def _row(label, mode, ms, mem_mb):
    print(f"  {label:<26}  {mode:<10}  time={ms:6.1f}ms  mem={mem_mb:7.0f}MB")


# ── AttentionPairBias benchmark ───────────────────────────────────────────────

def bench_attention_pair_bias(N=512, B=1, C_s=384, C_z=128, H=16, device="cuda"):
    from boltz.model.layers.attentionv2 import AttentionPairBias

    model = AttentionPairBias(c_s=C_s, c_z=C_z, num_heads=H, compute_pair_bias=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    s    = torch.randn(B, N, C_s, device=device, dtype=torch.bfloat16)
    z    = torch.randn(B, N, N, C_z, device=device, dtype=torch.bfloat16)
    mask = torch.ones(B, N, device=device, dtype=torch.bfloat16)

    def run_std():
        with torch.no_grad():
            model(s, z, mask, k_in=s, use_flash_attn=False)

    def run_flash():
        with torch.no_grad():
            model(s, z, mask, k_in=s, use_flash_attn=True)

    ms_std   = _warmup_and_time(run_std)
    mem_std  = _peak_mem_mb(run_std)
    ms_flash = _warmup_and_time(run_flash)
    mem_flash = _peak_mem_mb(run_flash)

    label = f"AttentionPairBias N={N}"
    _row(label, "standard", ms_std,   mem_std)
    _row(label, "flash",    ms_flash, mem_flash)
    _speedup(ms_std, ms_flash, mem_std, mem_flash)


def _speedup(ms_std, ms_flash, mem_std, mem_flash):
    spd = ms_std / ms_flash if ms_flash > 0 else float("inf")
    mem_r = mem_std / mem_flash if mem_flash > 0 else float("inf")
    print(f"  {'':26}  {'speedup':<10}  {spd:.2f}x faster, {mem_r:.2f}x less memory\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Boltz FlashAttention benchmark")
    parser.add_argument("--N",      type=int, default=512,  help="Sequence length")
    parser.add_argument("--batch",  type=int, default=1,    help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; running on CPU (no FlashAttention-2).")
        args.device = "cpu"

    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(args.device)
        print(f"GPU: {props.name}  (sm_{props.major}{props.minor})")
        if props.major < 8:  # noqa: PLR2004
            print("WARNING: FlashAttention-2 requires sm_80 (Ampere) or newer. "
                  "Results will reflect the math backend, not FlashAttn-2.\n")

    print(f"\nBenchmark: B={args.batch}, N={args.N}, "
          f"dtype=bfloat16, device={args.device}\n")
    print("-" * 72)

    bench_attention_pair_bias(N=args.N, B=args.batch, device=args.device)

    print("-" * 72)
    print("\nTo run Boltz prediction with FlashAttention-2 enabled:")
    print("  boltz predict <input> --out_dir <out> --flash_attn\n")


if __name__ == "__main__":
    main()

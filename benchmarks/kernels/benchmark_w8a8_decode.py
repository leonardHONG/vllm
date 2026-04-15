# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the W8A8 decode-stage slowdown reported in vLLM #38697.

Context: https://github.com/vllm-project/vllm/issues/38697
    llm-compressor W8A8 is reportedly slower than FP16/BF16 at the decode
    stage (M=1). This benchmark measures cutlass_scaled_mm (int8 and FP8)
    against F.linear across M ∈ {1, 2, 4, 8, 16, 32} on real Llama-3 shapes
    to answer three scouting questions:

        1. Is W8A8 slower than the unquantized baseline at M=1 on this GPU?
        2. Is the slowdown only at M=1, or does it persist across M <= 8?
        3. At what M does W8A8 start beating the baseline?

Usage (from repo root, in the vLLM venv):

    .venv/bin/python benchmarks/kernels/benchmark_w8a8_decode.py
    .venv/bin/python benchmarks/kernels/benchmark_w8a8_decode.py --dtype float16
    .venv/bin/python benchmarks/kernels/benchmark_w8a8_decode.py --reps 500

The script only times existing kernels — it does not modify any kernel code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEVICE = "cuda"

# Real Llama-3 linear shapes; (label, N, K) where out = act @ weight.T, weight [N, K].
SHAPES: list[tuple[str, int, int]] = [
    ("llama3-8B-qkv", 6144, 4096),        # GQA fused (q=4096, k=1024, v=1024)
    ("llama3-8B-o", 4096, 4096),
    ("llama3-8B-gate_up", 28672, 4096),
    ("llama3-8B-down", 4096, 14336),
    ("llama3-70B-o", 8192, 8192),
]

M_LIST = [1, 2, 4, 8, 16, 32]


def _time_kernel(fn, warmup: int, reps: int, rounds: int) -> float:
    """Return best-of-`rounds` mean latency in microseconds.

    Bulk timing (N iters / total) avoids per-iter event overhead dominating
    sub-10µs measurements; min-of-K filters scheduling noise.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best_us = float("inf")
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / reps
        best_us = min(best_us, us)
    return best_us


def _make_int8(m: int, n: int, k: int):
    a = (torch.randn(m, k, device=DEVICE) * 5).clamp(-128, 127).to(torch.int8)
    b = (torch.randn(n, k, device=DEVICE) * 5).clamp(-128, 127).to(torch.int8).t()
    scale_a = torch.rand(m, 1, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    scale_b = torch.rand(1, n, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    return a, b, scale_a, scale_b


def _make_fp8(m: int, n: int, k: int):
    fp8 = torch.float8_e4m3fn
    a = torch.randn(m, k, device=DEVICE).clamp(-448, 448).to(fp8)
    b = torch.randn(n, k, device=DEVICE).clamp(-448, 448).to(fp8).t()
    scale_a = torch.rand(m, 1, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    scale_b = torch.rand(1, n, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    return a, b, scale_a, scale_b


def _make_unquant(m: int, n: int, k: int, dtype: torch.dtype):
    x = torch.randn(m, k, device=DEVICE, dtype=dtype)
    w = torch.randn(n, k, device=DEVICE, dtype=dtype)
    return x, w


def _fmt(v: float | None, width: int = 10, prec: int = 2) -> str:
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{prec}f}"


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--reps", type=int, default=200, help="timed iters per round")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=5, help="take best-of-N rounds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert current_platform.is_cuda(), "W8A8 CUTLASS benchmark requires CUDA."
    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    cap = current_platform.get_device_capability()
    cap_int = cap[0] * 10 + cap[1]
    int8_ok = cap_int >= 75
    fp8_ok = ops.cutlass_scaled_mm_supports_fp8(cap_int) and hasattr(
        torch, "float8_e4m3fn"
    )

    print(f"# device : {torch.cuda.get_device_name()} (sm{cap_int})")
    print(f"# dtype  : {args.dtype}  (baseline F.linear uses this)")
    print(f"# reps   : {args.reps} iters × {args.rounds} rounds (min), warmup {args.warmup}")
    print(f"# int8   : {'enabled' if int8_ok else 'DISABLED (sm<75)'}")
    print(f"# fp8    : {'enabled' if fp8_ok else 'DISABLED (sm<89 or no float8_e4m3fn)'}")
    print()

    hdr = (
        f"{'shape':<20} {'M':>3} "
        f"{'int8 µs':>10} {'fp8 µs':>10} {args.dtype + ' µs':>14} "
        f"{'int8/base':>10} {'fp8/base':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, n, k in SHAPES:
        for m in M_LIST:
            t_base = None
            t_int8 = None
            t_fp8 = None

            x, w = _make_unquant(m, n, k, dtype)
            t_base = _time_kernel(
                lambda: F.linear(x, w), args.warmup, args.reps, args.rounds
            )

            if int8_ok:
                a8, b8, sa, sb = _make_int8(m, n, k)
                t_int8 = _time_kernel(
                    lambda: ops.cutlass_scaled_mm(a8, b8, sa, sb, out_dtype=dtype),
                    args.warmup,
                    args.reps,
                    args.rounds,
                )

            if fp8_ok:
                af, bf, saf, sbf = _make_fp8(m, n, k)
                t_fp8 = _time_kernel(
                    lambda: ops.cutlass_scaled_mm(af, bf, saf, sbf, out_dtype=dtype),
                    args.warmup,
                    args.reps,
                    args.rounds,
                )

            r_int8 = None if t_int8 is None else t_int8 / t_base
            r_fp8 = None if t_fp8 is None else t_fp8 / t_base
            print(
                f"{name:<20} {m:>3} "
                f"{_fmt(t_int8)} {_fmt(t_fp8)} {_fmt(t_base, width=14)} "
                f"{_fmt(r_int8)} {_fmt(r_fp8)}"
            )
        print()

    print(
        "# A ratio > 1.0 means W8A8 is SLOWER than the unquantized baseline.\n"
        "# Decoding is M=1; the issue reports ~1.5× (25µs vs 16µs) on that row."
    )


if __name__ == "__main__":
    main()

import torch
from torch.utils.cpp_extension import load
import time


torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="hist_lib",
    sources=["histogram.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

# small helper to run warmup + timed iterations
def run_benchmark(perf_func, a: torch.Tensor, tag: str, warmup: int = 10, iters: int = 100):
    # warmup
    for _ in range(warmup):
        _ = perf_func(a)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = perf_func(a)
    torch.cuda.synchronize()
    end = time.time()

    total_ms = (end - start) * 1000.0
    mean_ms = total_ms / iters
    # show a small sanity-check of output values
    out_cpu = out.detach().cpu()
    vals = out_cpu.flatten().tolist()[:8]
    print(f"{tag:>12}: mean_time={mean_ms:.6f} ms (over {iters} iters), sample={vals}")
    return out, mean_ms


if __name__ == "__main__":
    a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).cuda()

    print("-" * 80)
    out1, t1 = run_benchmark(lib.histogram_i32, a, "hist_i32", warmup=10, iters=100)

    print("-" * 80)
    out2, t2 = run_benchmark(lib.histogram_i32x4, a, "hist_i32x4", warmup=10, iters=100)

    # optional verification
    eq = torch.equal(out1, out2)
    print(f"-" * 80)
    print(f"Outputs equal: {eq}")

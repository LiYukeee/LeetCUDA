#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// Histogram
// Histogram 计数内核（逐元素版本）
// 说明：对输入数组 a 中的值做计数，结果写入全局直方图 y（使用 atomicAdd）。
// 线程布局：grid = (N+255)/256, block.x = 256（每个线程处理 1 个元素）
// 要求：a 中的值为非负整数且不超过 y 的索引上限（调用端通过 max(a) 分配 y）
// 注意：使用 atomicAdd 会在高争用桶上带来较大性能开销。
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4
// 向量化版本（每线程一次性读取 4 个 int）
// 优点：减少全局内存读取次数（一次 16 字节载入代替 4 次 4 字节载入），提升带宽利用。
// 风险/注意：必须保证对 a 的读取不会越界；否则一次性读取 4 个元素可能访问非法内存。
// 另外，虽然读取更宽，但 atomicAdd 仍是性能瓶颈，若大量元素落入少数桶，原子争用将限制性能。
// grid(N/256), block(256/4)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int *a, int *y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0)                                        \
  if (((T).size(0) != (S0))) {                                                 \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// 包装函数宏：生成 histogram_<packed_type> 的 Python 绑定函数
// 说明：
// - 检查输入 dtype；计算输入长度 N 和输入的最大值 M；
// - 在 GPU 上分配长度为 M+1 的 int32 输出 y（注意：当前将 y 分配到 CUDA device 0），
//   然后根据 n_elements 选择每个 block 的线程数，使每个 block 处理 256 个元素的工作量。
// - 最后 launch 对应的 kernel 并返回 y。
// 注意事项：
// - 如果 a 中的最大值 M 很大，会导致分配巨大的 y，可能 OOM；调用者需保证输入值域合理或先进行约束；
// - 更稳健的做法是将 y 分配到与 a 相同的 device（可用 a.device()），否则在多 GPU/重映射情形下可能出问题；
// - 为避免尾部越界，请确保内核实现对尾部访问做了保护（i32x4 内核中已加边界检查）。
#define TORCH_BINDING_HIST(packed_type, th_type, element_type, n_elements)     \
  torch::Tensor histogram_##packed_type(torch::Tensor a) {                     \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);   \
    const int N = a.size(0);                                                   \
    std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);         \
    torch::Tensor max_val = std::get<0>(max_a).cpu();                          \
    const int M = max_val.item().to<int>();                                    \
    auto y = torch::zeros({M + 1}, options);                                   \
    static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);               \
    const int NUM_BLOCKS = (N + 256 - 1) / 256;                                \
    dim3 block(NUM_THREADS_PER_BLOCK);                                         \
    dim3 grid(NUM_BLOCKS);                                                     \
    histogram_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<element_type *>(a.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
    return y;                                                                  \
  }

TORCH_BINDING_HIST(i32, torch::kInt32, int, 1)
TORCH_BINDING_HIST(i32x4, torch::kInt32, int, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}

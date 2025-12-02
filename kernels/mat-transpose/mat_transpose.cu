#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// FP32
// 两种一维索引的转置实现说明：
// - col2row（读连续，写不连续）:
//     线程按行优先遍历输入 (x[global_idx])，对此读是连续的，
//     但是写入 y 时地址为 y[global_col * row + global_row]，相邻线程写地址间隔为 row，
//     当 row 很大时写操作会成为非合并（non-coalesced）的大跨度访问，导致大量独立的全局内存事务，性能下降。
// - row2col（读不连续，写连续）:
//     线程按列优先写入 (y[global_idx])，写是连续的（写合并友好），
//     读取 x 时为跨行的有步长访问，但读取通常可以被 L1/L2 缓存或控制器合并/预取部分缓解，
//     因此整体开销通常低于非合并写。
// 结论：如果目标是高吞吐，优先保证写入地址连续（或使用 shared memory/tiling 消除步长）。
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx < row * col) {
    y[global_col * row + global_row] = x[global_idx];
  }
}

// 该实现写连续、读有步长，通常在 GPU 上比上面的实现快（因为写合并效率高）。
__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col) {
    y[global_idx] = x[global_row * col + global_col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_col = (global_idx * 4) % col;
  int global_row = (global_idx * 4) / col;

  if (global_row < row && global_col + 3 < col) {
    float4 x_val = reinterpret_cast<float4 *>(x)[global_idx];

    y[global_col * row + global_row] = x_val.x;
    y[(global_col + 1) * row + global_row] = x_val.y;
    y[(global_col + 2) * row + global_row] = x_val.z;
    y[(global_col + 3) * row + global_row] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = (global_idx * 4) / row;
  const int global_row = (global_idx * 4) % row;

  if (global_row < row && global_col < col) {
    float4 x_val;
    x_val.x = x[global_row * col + global_col];
    x_val.y = x[(global_row + 1) * col + global_col];
    x_val.z = x[(global_row + 2) * col + global_col];
    x_val.w = x[(global_row + 3) * col + global_col];
    reinterpret_cast<float4 *>(y)[global_idx] = FLOAT4(x_val);
  }
}

// work for row == col
__global__ void mat_transpose_f32_diagonal2d_kernel(float *x, float *y, int row,
                                                    int col) {
  const int block_y = blockIdx.x;
  const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  const int global_col = threadIdx.x + blockDim.x * block_x;
  const int global_row = threadIdx.y + blockDim.y * block_y;
  if (global_col < col && global_row < row) {
    y[global_row * col + global_col] = x[global_col * row + global_row];
  }
}

__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < col && global_y < row) {
    y[global_x * row + global_y] = x[global_y * col + global_x];
  }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
  const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y < col && global_x < row) {
    y[global_y * row + global_x] = x[global_x * col + global_y];
  }
}

__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x * 4 + 3 < col && global_y < row) {
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
    y[(global_x * 4) * row + global_y] = x_val.x;
    y[(global_x * 4 + 1) * row + global_y] = x_val.y;
    y[(global_x * 4 + 2) * row + global_y] = x_val.z;
    y[(global_x * 4 + 3) * row + global_y] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col2d_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y * 4 + 3 < row && global_x < col) {
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    reinterpret_cast<float4 *>(y)[global_x * row / 4 + global_y] =
        FLOAT4(x_val);
  }
}

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  // 使用 shared memory 的 tile 来做转置：
  // - 每个 block 负责一个 tile，tile 大小由 WARP_SIZE_S 和向量化宽度决定。
  // - 这里 tile 的布局是 tile[WARP_SIZE_S][WARP_SIZE_S * 4]，因为每个 thread 在 x 方向上
  //   负责一个 float4（4 个连续 float），所以在 shared memory 每列占 4 个连续位置。
  // 目的：先把全局内存的非合并访问聚合到 shared memory（一次向量化 load），
  // 在 tile 内以更友好的模式重排后再一次向量化写回，消除读/写的 stride 问题。
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];

  // 边界检查：确保读取的 global_x*4..global_x*4+3 不会越界（col + 3 用于容错）
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // 从全局内存把一组 float4 读到寄存器
    // 这里使用 reinterpret_cast<float4*> 来做对齐的向量化 load。
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];

    // 把向量化读取的 4 个 float 存到 shared memory 的连续位置
    // 使用 FLOAT4 宏可以将 float4 存放到对应的 shared memory 起始地址。
    FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
    __syncthreads();

    // 从 shared memory 按转置后的顺序取回 4 个元素，准备一次向量化写回全局内存
    float4 smem_val;
    // STRIDE 用于处理不同 block 尺寸下的映射，使得 local_x/local_y 能够正确索引 tile
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
    smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];

    // 计算写入到全局内存的位置：
    // - out_y 表示目标矩阵 y 的列索引（以元素为单位）
    // - out_x 表示目标矩阵 y 的行索引（以元素为单位）
    // 最后通过 reinterpret_cast<float4*> 把 4 个 float 一次性向量化写回。
    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    // map index n*n to (n/4)*(n*4)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // load value from x to shared memory
    float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
    tile[local_y][local_x * 4] = x_val.x;
    tile[local_y][local_x * 4 + 1] = x_val.y;
    tile[local_y][local_x * 4 + 2] = x_val.z;
    tile[local_y][local_x * 4 + 3] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
    smem_val.y =
        tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z =
        tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w =
        tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
    // map index n*n to (n/4)*(n*4)
    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  // 使用 shared memory 的 tile 进行转置（BCF 版本，按行向量化读取）
  // 布局说明：tile 的形状为 [WARP_SIZE_S * 4][WARP_SIZE_S + PAD]
  // 每个线程在 y 方向上负责一个位置（local_y），在 x 方向上负责 4 个连续的行（通过 local_y*4 存放），
  // 因此 shared memory 的第一维放大了 4 倍以保存这些按列展开的数据。
  // 目标：把连续的全局读取（每个线程读取 4 行同一列）写入 shared memory 的不同行，
  // 然后在 shared memory 内对数据重新布局，使得写回全局内存时有更好的合并性。
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];

  // 边界条件：保证读取的 (global_y*4 .. global_y*4+3) 行都在矩阵范围内，且 global_x 在列范围内
  if (global_y * 4 < row && global_x < col) {
    // 从全局内存按列向量化读取 4 个元素（相同行偏移）到寄存器
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];

    // 将读取到的 4 个元素按行分别写入 shared memory 的连续行中，列索引为 local_x
    // 这样 shared memory 中每一列对应一个 global_x，而每 4 行存储同一列的 4 个元素。
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();

    // 从 shared memory 按转置后的顺序读取出 4 个元素，准备写回全局内存
    float4 smem_val;
    // STRIDE 用于在 local_x/local_y 与 tile 索引之间做映射以适配不同 block 大小
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];

    // 计算写回的全局坐标映射：
    // - bid_x/ bid_y 为 block 的起始全局坐标基数（单位为线程）
    // - out_y/out_x 是写入 y 时的行/列（以元素为单位），随后按 row 计算线性索引
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);

    // 将 4 个元素分别写回全局内存（这四个写在设计上是连续/可合并的，取决于 out_y/out_x 的对齐）
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel(
    float *x, float *y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    smem_val.x = tile[local_x * 4][local_y];
    smem_val.y = tile[local_x * 4 + 1][local_y];
    smem_val.z = tile[local_x * 4 + 2][local_y];
    smem_val.w = tile[local_x * 4 + 3][local_y];

    const int gid_x = blockIdx.x * blockDim.x;
    const int gid_y = blockIdx.y * blockDim.y * 4;
    const int out_y = gid_y + local_x * 4;
    const int out_x = gid_x + local_y;
    reinterpret_cast<float4 *>(y)[(out_x * row + out_y) / 4] = FLOAT4(smem_val);
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


  /*
   * 将 CUDA kernel 绑定到 Python 的小宏：
   * - tag: 内核后缀名
   * - th_type: PyTorch 的 dtype (用于校验)
   * - element_type: C++ 层面的元素类型 (float/half 等)
   * - n_pack: 每个线程处理的元素打包数（例如 4 表示 float4 向量化）
   *
   * 计算 grid 时的说明：这里以一维索引展开矩阵，线程数为 blockDim.x（= WARP_SIZE），
   * total 元素数为 N*M。除以 n_pack 是因为当使用向量化（例如 float4）时，每个线程一次处理 n_pack 个元素，
   * 因此需要的线程数 / warp 数相应减少。注意：
   * - 当使用向量化时，必须保证输入在读取/写入时满足对齐与边界条件（代码中各 kernel 会检查边界）。
   * - grid 计算为简化形式，针对非常不规则的形状可能需要更精细的 block/grid 配置。 
   * - 如果在 CPU/PyTorch 端做了 device 指定，建议在此宏生成的函数中使用 x.device() 来分配/检查输出，
   *   以避免硬编码设备 0 的情况（当前代码假设输入/输出已分配于正确设备）。
   */
#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)        \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(WARP_SIZE);                                                     \
    /* 一维 grid：总元素数 / 每线程处理元素数 / 每个 block 的线程数 */            \
    dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));                 \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                             \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }

#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,              \
                                      n_element_row, n_element_col)            \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                      \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),           \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));          \
    mat_transpose_##tag##2d_kernel<<<grid, block>>>(                           \
                   reinterpret_cast<element_type *>(x.data_ptr()),             \
                   reinterpret_cast<element_type *>(y.data_ptr()), M, N);      \
  }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2d index. easier for diagonal
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)
// diagonal index method.
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)
// shared memory
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1,
                              4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_row2col, torch::kFloat32, float, 4,
                              1)
// shared memory with bcf
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float,
                              1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_row2col, torch::kFloat32, float,
                              4, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_merge_write_row2col,
                              torch::kFloat32, float, 4, 1)

// CuTe implentations
extern void mat_transpose_cute_col2row_reg(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row2col_reg(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_col_smem(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_smem(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_col_smem_swizzled(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_smem_swizzled(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_cvectorized(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_rvectorized(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor,
                                                        torch::Tensor);
extern void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor,
                                                        torch::Tensor);
extern void
    mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor,
                                                          torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col2d)
  // diagonal index method.
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)
  // shared memory optimize
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_row2col2d)
  // shared memory optimize with bcf
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(
      mat_transpose_f32x4_shared_bcf_merge_write_row2col2d)
  // CuTe implentations
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col2row_reg)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row2col_reg)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(
      mat_transpose_cute_row_rvectorized_swizzled_optimized)
}

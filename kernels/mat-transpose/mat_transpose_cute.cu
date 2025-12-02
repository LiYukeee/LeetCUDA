#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;

#define UNIT_BLK_SIZE 16

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      /* Optionally, you could also call cudaDeviceReset here */               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA,
          typename ThreadLayoutB>
__global__ void mat_transpose_cute_reg_kernel(const T *pA, T *pB, int M, int N,
                                              ThreadLayoutA tA,
                                              ThreadLayoutB tB) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  // CUTE 风格的基于 tile 的寄存器/共享内存转置实现（寄存器版本）
  // 说明：此 kernel 使用 cute 库的抽象来把全局矩阵视作 tensor，并对其进行局部 tile 的划分，
  // 然后把每个 tile 的子区域按照线程布局分配到线程，最后执行按条件的拷贝（copy_if）完成转置。
  // 关键概念：
  //  - make_tensor(make_gmem_ptr(pA), make_layout(...))：把全局内存指针绑定为可索引的 tensor，指定矩阵行优先/列优先布局。
  //  - local_tile(tensor, shape, coord)：取出位于 grid 坐标 (bx,by) 的局部 tile（大小 BLK_M x BLK_N）。
  //  - local_partition(tile, layout, tx)：根据线程布局把 tile 划分给线程 tx（产生线程负责的子块视图）。
  //  - make_identity_tensor(mA.shape()) + local_tile：用于生成 tile 的全局下标坐标，便于构造 predicate（边界判定）。
  //  - copy_if(predicate_tensor, src_partition, dst_partition)：按 predicate 条件把 src 的有效元素拷贝到 dst（用于处理边界越界）。

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // 全局 A：形状 (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // 全局 B（目标）：形状 (N, M)

  // gA/gB 表示位于 block (bx,by) 的全局 tile 视图，gA 的形状是 (BLK_M, BLK_N)
  // 注意 gB 的 coord 是 (by, bx)，即转置时 block 在全局位置互换
  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  // cA 是用于生成全局坐标的 tile（identity tensor），通过它我们能得到每个元素在全局矩阵中的索引
  auto cA = local_tile(make_identity_tensor(mA.shape()),
                       make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)

  // 把每个 tile 根据线程布局 partition 给线程 tx，得到线程局部负责的子视图
  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tBgB = local_partition(gB, tB, tx);
  Tensor tAcA = local_partition(cA, tA, tx);

  // 构造 predicate mask（布尔型 tensor），表示对应位置是否在矩阵边界内（用于处理矩阵尺寸不能整除 tile 的情况）
  Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tApA); i++) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tApA); j++) {
      // get<0>/get<1> 从 identity tensor 中取到该元素在原矩阵的 (row, col) 全局坐标
      tApA(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;
    }
  }

  // 根据 predicate 把 gA（源 tile）拷贝到 gB（目标 tile），copy_if 会跳过越界位置
  // 这是一个高层封装，内部会按线程分配做高效的内存访问（可能使用寄存器/共享内存/向量化）
  copy_if(tApA, tAgA, tBgB);
}

void mat_transpose_cute_row2col_reg(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_col2row_reg(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA,
          typename ThreadLayoutB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void
mat_transpose_cute_smem_kernel(const T *pA, T *pB, int M, int N,
                               ThreadLayoutA tA, ThreadLayoutB tB,
                               SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)
  auto cA = local_tile(make_identity_tensor(mA.shape()),
                       make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto cB = local_tile(make_identity_tensor(mB.shape()),
                       make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(make_smem_ptr(smem),
                        sA_layout); // (BM, BN)
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

  Tensor tAgA = local_partition(gA, tA, tx);
  Tensor tBgB = local_partition(gB, tB, tx);
  Tensor tAsA = local_partition(sA, tA, tx);
  Tensor tBsB = local_partition(sB, tB, tx);
  Tensor tAcA = local_partition(cA, tA, tx);
  Tensor tBcB = local_partition(cB, tB, tx);

  Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
  Tensor tBpB = make_tensor<bool>(tBcB.shape(), tBcB.stride());
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tApA); i++) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tApA); j++) {
      tApA(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;
    }
  }
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tBpB); i++) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tBpB); j++) {
      tBpB(i, j) = get<0>(tBcB(i, j)) < N && get<1>(tBcB(i, j)) < M;
    }
  }
  copy_if(tApA, tAgA, tAsA);
  __syncthreads();
  copy_if(tBpB, tBsB, tBgB);
}

constexpr int log2(int x) {
  assert(x > 0);
  return (x & (x - 1)) == 0 ? __builtin_ctz(x)
                            : (throw "x is not a power of 2", 0);
}

void mat_transpose_cute_col_smem(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_smem(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_col_smem_swizzled(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
  const int S = log2(BM);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_smem_swizzled(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  const int M = x.size(0);
  const int N = x.size(1);
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  const int S = log2(BM);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));
  static_assert(size(tA) == size(tB));
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                                 decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB,
                        sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

__host__ __device__ inline bool is_aligned_128(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA,
          typename TiledCopyB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_kernel(
    const T *pA, T *pB, int M, int N, TiledCopyA copy_a, TiledCopyB copy_b,
    SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, N)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sA = make_tensor(make_smem_ptr(smem),
                        sA_layout); // (BM, BN)
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

  auto thr_copy_a = copy_a.get_slice(tx);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);

  auto thr_copy_b = copy_b.get_slice(tx);
  Tensor tBsB = thr_copy_b.partition_S(sB);
  Tensor tBgB = thr_copy_b.partition_D(gB);

  copy(copy_a, tAgA, tAsA);
  __syncthreads();
  copy(copy_b, tBsB, tBgB);
}

void mat_transpose_cute_row_cvectorized(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE * 4;
  const int BN = UNIT_BLK_SIZE;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor x,
                                                 torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE * 4;
  const int BN = UNIT_BLK_SIZE;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  const int S = log2(BN);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_rvectorized(torch::Tensor x, torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE * 4;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM>{}, Int<BN / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN / 4>{}, Int<BM>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor x,
                                                 torch::Tensor y) {
  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE * 4;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM>{}, Int<BN / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN / 4>{}, Int<BM>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
  const int S = log2(BM);
  auto swizzle_func = Swizzle<S, 0, S>{};
  auto sA_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_transpose_cute_smem_vectorized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_b),
      decltype(sA_layout), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA,
          typename TiledCopyTrans, typename TiledCopyB, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_optimized_kernel(
    const T *pA, T *pB, int M, int N, TiledCopyA copy_a,
    TiledCopyTrans copy_trans, TiledCopyB copy_b, SmemLayoutB sB_layout) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  auto mA = make_tensor(make_gmem_ptr(pA),
                        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
  auto mB = make_tensor(make_gmem_ptr(pB),
                        make_layout(make_shape(N, M), GenRowMajor{})); // (N, N)

  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                       make_coord(bx, by)); // (BM, BN)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
                       make_coord(by, bx)); // (BN, BM)

  __shared__ T smem[BLK_M * BLK_N];
  auto sB = make_tensor(make_smem_ptr(smem),
                        sB_layout); // (BN, BM)

  auto thr_copy_a = copy_a.get_slice(tx);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  auto tAsA = make_tensor_like(tAgA);
  Tensor tAsA_view = thr_copy_a.retile_D(tAsA);
  copy(copy_a, tAgA, tAsA_view);

  auto thr_copy_trans = copy_trans.get_slice(tx);
  auto tAsB = thr_copy_trans.retile_S(tAsA);
  auto tBsB_trans = thr_copy_trans.partition_D(sB);
  copy(copy_trans, tAsB, tBsB_trans);

  auto thr_copy_b = copy_b.get_slice(tx);
  Tensor tBsB = thr_copy_b.partition_S(sB);
  Tensor tBgB = thr_copy_b.partition_D(gB);

  copy(copy_b, tBsB, tBgB);
}

void mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor x,
                                                           torch::Tensor y) {
  const int BM = 8;
  const int BN = 16 * 8;
  auto ptr_A = x.data_ptr<float>();
  auto ptr_B = y.data_ptr<float>();
  const int M = x.size(0);
  const int N = x.size(1);

  // sanity checks
  assert(M % 4 == 0);
  assert(N % 4 == 0);
  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  assert(is_aligned_128(ptr_A));
  assert(is_aligned_128(ptr_B));

  // 一次性加载8*16大小的矩阵
  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM>{}, make_shape(Int<4>{}, Int<BN / 16>{})),
                  make_stride(Int<4>{}, make_stride(Int<1>{}, Int<32>{}))),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

  // 转换数据
  auto tile_copy_trans = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(make_shape(Int<4>{}, Int<BN / 16>{}), Int<BM>{}),
                  make_stride(make_stride(Int<1>{}, Int<32>{}), Int<4>{})),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

  // 一次性存储16*8大小的矩阵
  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

  auto swizzle_func = Swizzle<2, 3, 2>{};
  auto sB_layout =
      composition(swizzle_func,
                  make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{}));

  static_assert(size(tile_copy_a) == size(tile_copy_b));

  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  mat_transpose_cute_smem_vectorized_optimized_kernel<
      float, BM, BN, decltype(tile_copy_a), decltype(tile_copy_trans),
      decltype(tile_copy_b), decltype(sB_layout)><<<grid, block>>>(
      ptr_A, ptr_B, M, N, tile_copy_a, tile_copy_trans, tile_copy_b, sB_layout);
  CUDA_CHECK(cudaGetLastError());
}

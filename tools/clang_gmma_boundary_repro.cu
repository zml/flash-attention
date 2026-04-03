#include <cute/tensor.hpp>

#include "cutlass/numeric_types.h"

using namespace cute;

namespace {

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> b;
};

template <class ThrMma, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_leaf(ThrMma const& thr_mma, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  cute::gemm(thr_mma, tCrA(_, _, _, 0), tCrB(_, _, _, 0), tCrC);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  axpby(1.0f, tCrC, 0.0f, tCgC);
}

template <class ThrMma, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_wrapper(ThrMma const& thr_mma, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
  gmma_leaf(thr_mma, sA, sB, gC);
}

template <int Repeat, class ThrMma, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_step(ThrMma const& thr_mma, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
  if constexpr (Repeat >= 0) {
    gmma_wrapper(thr_mma, sA, sB, gC);
  }
}

template <class ThrMma, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_mainloop(ThrMma const& thr_mma, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
#pragma unroll 1
  for (int i = 0; i < 2; ++i) {
    gmma_step<0>(thr_mma, sA, sB, gC);
  }
}

}  // namespace

extern "C" __global__ __launch_bounds__(128)
void clang_gmma_boundary_repro(cutlass::half_t const* src_a,
                               cutlass::half_t const* src_b,
                               float* dst_c) {
  using TA = cutlass::half_t;
  using TB = cutlass::half_t;

  auto blk_m = Int<128>{};
  auto blk_n = Int<128>{};
  auto blk_k = Int<64>{};
  auto pipes = Int<1>{};

  auto smem_layout_a = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(blk_m, blk_k, pipes));
  auto smem_layout_b = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(blk_n, blk_k, pipes));
  using SmemLayoutA = decltype(smem_layout_a);
  using SmemLayoutB = decltype(smem_layout_b);
  using Storage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;

  extern __shared__ char shared_bytes[];
  auto& storage = *reinterpret_cast<Storage*>(shared_bytes);

  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), smem_layout_a);
  Tensor sB = make_tensor(make_smem_ptr(storage.b.begin()), smem_layout_b);

  constexpr int kSmemElemsA = cosize_v<SmemLayoutA>;
  constexpr int kSmemElemsB = cosize_v<SmemLayoutB>;
  for (int idx = threadIdx.x; idx < kSmemElemsA; idx += blockDim.x) {
    storage.a.begin()[idx] = src_a ? src_a[idx & 15] : TA(0);
  }
  for (int idx = threadIdx.x; idx < kSmemElemsB; idx += blockDim.x) {
    storage.b.begin()[idx] = src_b ? src_b[idx & 15] : TB(0);
  }
  __syncthreads();

  auto tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor mC = make_tensor(make_gmem_ptr(dst_c), make_shape(blk_m, blk_n), make_stride(Int<1>{}, blk_m));
  gmma_mainloop(thr_mma, sA, sB, mC);
}

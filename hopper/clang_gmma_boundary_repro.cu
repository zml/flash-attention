#include <cute/tensor.hpp>

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

using namespace cute;

namespace {

template<typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
  static_assert(decltype(rank<0>(acc_layout))::value == 3);
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
  return make_layout(
      make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
      get<1>(acc_layout),
      coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
}

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
  static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0);
  Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
  Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
  cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
  #pragma unroll
  for (int i = 0; i < size(frag); ++i) {
    out_frg[i] = convert_op(frag[i]);
  }
}

template<bool zero_init=false, int wg_wait=0, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gmma(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  if constexpr (zero_init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  }
  static constexpr int kNumKIters = CUTE_STATIC_V(size<2>(tCrA));
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < kNumKIters; ++k_block) {
    cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
  warpgroup_fence_operand(tCrC);
}

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> b;
};

template <class TiledMma, class MmaSlice, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_leaf(TiledMma& tiled_mma, MmaSlice const& mma_slice, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
  Tensor tCrA = mma_slice.partition_fragment_A(sA);
  Tensor tCrB = mma_slice.partition_fragment_B(sB);
  Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<128>, Int<128>>{});
  clear(acc_s);

  gmma</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tCrA(_, _, _, 0), tCrB(_, _, _, 0), acc_s);

  Tensor tOrP = make_tensor(acc_s.data(), convert_layout_acc_Aregs<TiledMma>(acc_s.layout()));
  Tensor rP = make_tensor_like<cutlass::half_t>(tOrP);
  convert_type_out(tOrP, rP);

  auto dst = mma_slice.partition_C(gC);
  axpby(1.0f, rP, 0.0f, dst);
}

template <class TiledMma, class MmaSlice, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_wrapper(TiledMma& tiled_mma, MmaSlice const& mma_slice, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
  auto scoremod = [&](auto& acc) {
    #pragma unroll
    for (int i = 0; i < size(acc); ++i) {
      acc(i) = acc(i);
    }
  };

  auto fwd_step = [&](auto const& run_gemm) {
    auto tmp_acc = partition_fragment_C(tiled_mma, Shape<Int<128>, Int<128>>{});
    clear(tmp_acc);
    run_gemm(tmp_acc);
    scoremod(tmp_acc);
  };

  fwd_step([&](auto&) {
    gmma_leaf(tiled_mma, mma_slice, sA, sB, gC);
  });
}

template <int Repeat, class TiledMma, class MmaSlice, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_step(TiledMma& tiled_mma, MmaSlice const& mma_slice, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
  if constexpr (Repeat >= 0) {
    gmma_wrapper(tiled_mma, mma_slice, sA, sB, gC);
  }
}

template <class TiledMma, class MmaSlice, class SmemTensorA, class SmemTensorB, class GmemTensorC>
CUTE_DEVICE void gmma_mainloop(TiledMma& tiled_mma, MmaSlice const& mma_slice, SmemTensorA const& sA, SmemTensorB const& sB, GmemTensorC const& gC) {
#pragma unroll 1
  for (int i = 0; i < 2; ++i) {
    auto body = [&](auto iter_tag) {
      (void) iter_tag;
      gmma_step<0>(tiled_mma, mma_slice, sA, sB, gC);
    };
    body(Int<0>{});
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
  auto mma_slice = tiled_mma.get_slice(threadIdx.x);

  Tensor mC = make_tensor(make_gmem_ptr(dst_c), make_shape(blk_m, blk_n), make_stride(Int<1>{}, blk_m));
  gmma_mainloop(tiled_mma, mma_slice, sA, sB, mC);
}

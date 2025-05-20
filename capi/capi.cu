#include <stdio.h>
#include <stdint.h>

#include "capi.h"

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

#define CAPI_CHECK(cond, ...)                                                                 \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            fprintf(stderr, "Check failed (%s:%d): %s\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while(0)

namespace FLASH_NAMESPACE {

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      int64_t q_batch_stride,
                      int64_t q_row_stride,
                      int64_t q_head_stride,
                      int64_t k_batch_stride,
                      int64_t k_row_stride,
                      int64_t k_head_stride,
                      int64_t v_batch_stride,
                      int64_t v_row_stride,
                      int64_t v_head_stride,
                      int64_t o_batch_stride,
                      int64_t o_row_stride,
                      int64_t o_head_stride,
                      DataType q_dtype,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q_dtype == CAPI_BFLOAT16;

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_ptr = out;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q_batch_stride;
        params.k_batch_stride = k_batch_stride;
        params.v_batch_stride = v_batch_stride;
        params.o_batch_stride = o_batch_stride;
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    //TORCH_CHECK(p_dropout < 1.f);
    //#ifdef FLASHATTENTION_DISABLE_DROPOUT
    //    TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    //#endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int num_sm,
    void* softmax_lse_accum, const int allocated_softmax_lse_accum_size, 
    void* out_accum, const int allocated_out_accum_size) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            const int float_byte_size = 4;
            const int softmax_lse_accum_size = params.num_splits * batch_size * num_heads * max_seqlen_q * float_byte_size;
            CAPI_CHECK(softmax_lse_accum_size <= allocated_softmax_lse_accum_size, "Tensor allocated for softmax lse accum must be big enough");

            const int out_accum_size = params.num_splits * batch_size * num_heads * max_seqlen_q * head_size_rounded * float_byte_size;
            CAPI_CHECK(out_accum_size <= allocated_out_accum_size, "Tensor allocated for out accum must be big enough");

            //softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            //out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));

            params.softmax_lseaccum_ptr = softmax_lse_accum;
            params.oaccum_ptr = out_accum;
        }
        CAPI_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
}

void flashattn_mha_varlen_fwd(
        void* q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* cu_seqlens_q, // b+1
        void* cu_seqlens_k, // b+1
        void* seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        // leftpad_k_ (disabled because I don't know what it is lol)
        void* block_table, // batch_size x max_num_blocks_per_seq
        void* softmax_lse, // num_heads x total_q
        void* alibi_slopes_, // num_heads or b x num_heads
        void* softmax_lse_accum,
        void* out_accum,
        DataType q_dtype,
        int64_t q_batch_stride,
        int64_t q_row_stride,
        int64_t q_head_stride,
        int64_t k_batch_stride,
        int64_t k_row_stride,
        int64_t k_head_stride,
        int64_t v_batch_stride,
        int64_t v_row_stride,
        int64_t v_head_stride,
        int64_t o_batch_stride,
        int64_t o_row_stride,
        int64_t o_head_stride,
        int64_t block_table_batch_stride,
        const int softmax_lse_accum_size,
        const int out_accum_size,
        int max_seqlen_q,
        const int max_seqlen_k,
        bool is_causal,
        const float softmax_scale,
        int window_size_left,
        int window_size_right,
        uint32_t total_q,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t num_heads_k,
        uint32_t head_size,
        uint32_t page_block_size,
        void* stream) {

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    CAPI_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    if (max_seqlen_q == 1 && alibi_slopes_ == nullptr) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    const bool paged_KV = true;

    // TODO(Corentin):
    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size % 8 == 0 && alibi_slopes_ == nullptr;
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        // NOTE(Corentin): Suppose q is already in the correct shape
        //q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q = nullptr;
    }

    CAPI_CHECK(batch_size > 0, "batch size must be positive");
    CAPI_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    CAPI_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    CAPI_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    //CHECK_SHAPE(q, total_q, num_heads, head_size);
    //if (!paged_KV) {
    //    const int total_k = k.size(0);
    //    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    //    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    //} else {
    //    CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
    //    CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
    //    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    //}

    //CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    //CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    //if (seqused_k.has_value()){
    //    auto seqused_k_ = seqused_k.value();
    //    CAPI_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    //    CAPI_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
    //    CAPI_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
    //    CHECK_SHAPE(seqused_k_, batch_size);
    //}

    //at::Tensor out;
    //if (out_.has_value()) {
    //    out = out_.value();
    //    CAPI_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    //    CHECK_DEVICE(out);
    //    CAPI_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    //    CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
    //    if (seqlenq_ngroups_swapped) {
    //        out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
    //    }
    //} else {
    //    out = torch::empty_like(q);
    //}

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     q_batch_stride,
                     q_row_stride,
                     q_head_stride,
                     k_batch_stride,
                     k_row_stride,
                     k_head_stride,
                     v_batch_stride,
                     v_row_stride,
                     v_head_stride,
                     o_batch_stride,
                     o_row_stride,
                     o_head_stride,
                     q_dtype,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     seqused_k,
                     nullptr,
                     softmax_lse,
                     0.0,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     0.0f,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true);
    params.total_q = total_q;

    params.block_table = static_cast<int*>(block_table);
    params.block_table_batch_stride = block_table_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    //at::Tensor softmax_lse_accum, out_accum;
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        set_params_splitkv(params, batch_size, num_heads, head_size,
                               max_seqlen_k, max_seqlen_q, head_size_rounded,
                               0.f, /*num_splits*/ 0, get_num_sm(get_current_device()), 
                               softmax_lse_accum, softmax_lse_accum_size,
                               out_accum, out_accum_size);
    }

    //if (leftpad_k_.has_value()) {
    //    auto leftpad_k = leftpad_k_.value();
    //    CAPI_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
    //    CAPI_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    //    CHECK_DEVICE(leftpad_k);
    //    CHECK_CONTIGUOUS(leftpad_k);
    //    CHECK_SHAPE(leftpad_k, batch_size);
    //    params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    //}

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    //int64_t counter_offset = params.b * params.h * 32;
    //auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    //auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    //params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    //if (p_dropout > 0.0)  {
    //    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    //    // See Note [Acquire lock when using random generators]
    //    std::lock_guard<std::mutex> lock(gen->mutex_);
    //    params.philox_args = gen->philox_cuda_state(counter_offset);
    //}

    //set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_k > 0) {
        run_mha_fwd(params, reinterpret_cast<cudaStream_t>(stream), paged_KV);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        //out.zero_();
        //softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        // NOTE(Corentin): We assume that out already has the correct shape
        //int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        //int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        //out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        //q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        //softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
    }
}

}
void flashattn_mha_varlen_fwd(
        void* q,
        void* k,
        void* v,
        void* out,
        void* cu_seqlens_q,
        void* cu_seqlens_k,
        void* seqused_k,
        void* block_table,
        void* softmax_lse,
        void* alibi_slopes_,
        void* softmax_lse_accum,
        void* out_accum,
        FlashattnMhaVarlenFwdParams params,
        void* stream) {
    FLASH_NAMESPACE::flashattn_mha_varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table,
        softmax_lse,
        alibi_slopes_,
        softmax_lse_accum,
        out_accum,
        params.q_dtype,
        params.q_batch_stride,
        params.q_row_stride,
        params.q_head_stride,
        params.k_batch_stride,
        params.k_row_stride,
        params.k_head_stride,
        params.v_batch_stride,
        params.v_row_stride,
        params.v_head_stride,
        params.o_batch_stride,
        params.o_row_stride,
        params.o_head_stride,
        params.block_table_batch_stride,
        params.softmax_lse_accum_size,
        params.out_accum_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.is_causal,
        params.softmax_scale,
        params.window_size_left,
        params.window_size_right,
        params.total_q,
        params.batch_size,
        params.num_heads,
        params.num_heads_k,
        params.head_size,
        params.page_block_size,
        stream);
}

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
                      int64_t q_row_stride,
                      int64_t q_head_stride,
                      int64_t k_row_stride,
                      int64_t k_head_stride,
                      int64_t v_row_stride,
                      int64_t v_head_stride,
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

    //if (cu_seqlens_q_d == nullptr) {
    //    params.q_batch_stride = q.stride(0);
    //    params.k_batch_stride = k.stride(0);
    //    params.v_batch_stride = v.stride(0);
    //    params.o_batch_stride = out.stride(0);
    //    if (seqlenq_ngroups_swapped) {
    //         params.q_batch_stride *= seqlen_q;
    //         params.o_batch_stride *= seqlen_q;
    //    }
    //}

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

void flashattn_batched_prefill_with_kvcache(
        void* q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* cu_seqlens_q, // b+1
        void* cu_seqlens_k, // b+1
        void* seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        // leftpad_k_ (disabled because I don't know what it is lol)
        void* block_table, // batch_size x max_num_blocks_per_seq
        void* alibi_slopes_, // num_heads or b x num_heads
        int max_seqlen_q,
        const int max_seqlen_k,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,

        int64_t q_row_stride,
        int64_t q_head_stride,
        int64_t k_row_stride,
        int64_t k_head_stride,
        int64_t v_row_stride,
        int64_t v_head_stride,
        int64_t o_row_stride,
        int64_t o_head_stride,
        int64_t block_table_batch_stride,
        int64_t k_batch_stride,
        int64_t v_batch_stride,
        DataType q_dtype,
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
    const int seqlenq_ngroups_swapped = 0;
    //const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    //const int ngroups = num_heads / num_heads_k;
    //if (seqlenq_ngroups_swapped) {
    //    q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
    //    max_seqlen_q = ngroups;
    //    num_heads = num_heads_k;
    //    cu_seqlens_q_d = nullptr;
    //}

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
                     q_row_stride,
                     q_head_stride,
                     k_row_stride,
                     k_head_stride,
                     v_row_stride,
                     v_head_stride,
                     o_row_stride,
                     o_head_stride,
                     q_dtype,

                     cu_seqlens_q,
                     cu_seqlens_k,
                     seqused_k,
                     nullptr,
                     nullptr,
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
    //if (seqlenq_ngroups_swapped) {
    //    // Only apply split-k for decoding
    //    std::tie(softmax_lse_accum, out_accum) =
    //        set_params_splitkv(params, batch_size, num_heads, head_size,
    //                           max_seqlen_k, max_seqlen_q, head_size_rounded,
    //                           p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);
    //}

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
    params.rng_state = static_cast<uint64_t*>(nullptr);

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
        //int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        //int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        //out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        //q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        //softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
    }
}

}

void flashattn_batched_prefill_with_kvcache(
        void* q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* cu_seqlens_q, // b+1
        void* cu_seqlens_k, // b+1
        void* seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        // leftpad_k_ (disabled because I don't know what it is lol)
        void* block_table, // batch_size x max_num_blocks_per_seq
        void* alibi_slopes_, // num_heads or b x num_heads
        int max_seqlen_q,
        const int max_seqlen_k,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,

        int64_t q_row_stride,
        int64_t q_head_stride,
        int64_t k_row_stride,
        int64_t k_head_stride,
        int64_t v_row_stride,
        int64_t v_head_stride,
        int64_t o_row_stride,
        int64_t o_head_stride,
        int64_t block_table_batch_stride,
        int64_t k_batch_stride,
        int64_t v_batch_stride,
        DataType q_dtype,
        uint32_t total_q,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t num_heads_k,
        uint32_t head_size,
        uint32_t page_block_size,
        void* stream) {
    FLASH_NAMESPACE::flashattn_batched_prefill_with_kvcache(
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        cu_seqlens_q, // b+1
        cu_seqlens_k, // b+1
        seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        block_table, // batch_size x max_num_blocks_per_seq
        alibi_slopes_, // num_heads or b x num_heads
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,

        q_row_stride,
        q_head_stride,
        k_row_stride,
        k_head_stride,
        v_row_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        block_table_batch_stride,
        k_batch_stride,
        v_batch_stride,
        q_dtype,
        total_q,
        batch_size,
        num_heads,
        num_heads_k,
        head_size,
        page_block_size,
        stream);
}

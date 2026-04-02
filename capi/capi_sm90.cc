#include <cutlass/numeric_types.h>
#include <cstdio>
#include <cstdlib>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"

#include "fa_utils.h"

namespace FLASH_NAMESPACE {

cudaDeviceProp getCurrentDeviceProperties() {
    static cudaDeviceProp prop;
    static bool initialized = false;
    if (initialized) {
        return prop;
    }
    int device = 0;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    initialized = true;

    return prop;
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
                      const FlashattnTensor *q,
                      const FlashattnTensor *k,
                      const FlashattnTensor *v,
                      const FlashattnTensor *out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      int attention_chunk,
                      const float softcap=0.f,
                      const int sm_margin=0) {
    // Reset the parameters
    params = {};

    params.is_bf16 = q->dtype == CAPI_BFLOAT16;
    params.is_e4m3 = q->dtype == CAPI_F8E4M3FN;

    // Set the pointers and strides.
    params.q_ptr = q->ptr;
    params.k_ptr = k->ptr;
    params.v_ptr = v->ptr;
    // All stride are in elements, not bytes.
    params.q_row_stride = getStride(q, -3);
    params.k_row_stride = getStride(k, -3);
    params.v_row_stride = getStride(v, -3);
    params.q_head_stride = getStride(q, -2);
    params.k_head_stride = getStride(k, -2);
    params.v_head_stride = getStride(v, -2);
    params.v_dim_stride = getStride(v, -1);
    params.o_ptr = out->ptr;
    params.o_row_stride = getStride(out, -3);
    params.o_head_stride = getStride(out, -2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = getStride(q, 0);
        params.o_batch_stride = getStride(out, 0);
    }
    if (cu_seqlens_k_d == nullptr) {
        params.k_batch_stride = getStride(k, 0);
        params.v_batch_stride = getStride(v, 0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    //TORCH_CHECK(p_dropout < 1.f);
    //#ifdef FLASHATTENTION_DISABLE_DROPOUT
    //    TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    //#endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    cudaDeviceProp prop = getCurrentDeviceProperties();

    params.arch = prop.major * 10 + prop.minor;
    params.num_sm = prop.multiProcessorCount - sm_margin;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    CAPI_CHECK(params.num_splits >= 1, "");
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        if (!params.is_e4m3) {
                            if (params.is_bf16) {
                                #ifndef FLASHATTENTION_DISABLE_HDIM64
                                if (params.d <= 64) {
                                    if (params.dv > 256 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else if (params.dv > 64 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM192
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                            } else {
                                #ifndef FLASHATTENTION_DISABLE_FP16
                                #ifndef FLASHATTENTION_DISABLE_HDIM64
                                if (params.d <= 64) {
                                    if (params.dv > 256 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else if (params.dv > 64 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM192
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #else
                                CAPI_CHECK(false, "This flash attention build does not support FP16.");
                                #endif
                            }
                        } else {
                            #ifndef FLASHATTENTION_DISABLE_FP8
                            #ifndef FLASHATTENTION_DISABLE_HDIM64
                            if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM96
                            if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM128
                            if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM192
                            if (params.d <= 192) {
                                if (params.dv <= 128 && Arch == 90) {
                                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                } else {
                                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                }
                            }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM256
                            if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #else
                            CAPI_CHECK(false, "This flash attention build does not support FP8.");
                            #endif
                        }
                    });
                });
            });
        });
    });
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl=false) {
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
    #else
    CAPI_CHECK(false, "This flash attention build does not support combine kernels.");
    #endif
}

inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
    // disable for local since we move k_ptr to start of sliding window by m_block
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr || params.is_local) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f, use_one_mma_wg(params));
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
    // at least for MLA.
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    // Always enable PackGQA for special case of hdim = 64, qheads/kvheads = 8, local attention
    // TODO: investigate more cases where PackGQA improves perf due to better tile quantization
    bool const packgqa_override = params.arch >= 90 && (params.h / params.h_k) == 8 && 
                                  params.is_local && 
                                  params.d == 64 && (params.dv == params.d);
    if (packgqa_override) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f, use_one_mma_wg(params));
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f, use_one_mma_wg(params));
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
    #endif
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

inline bool fa3_repro_debug_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("FA3_REPRO_DEBUG");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

inline bool fa3_repro_skip_sched_prep_enabled() {
    static bool enabled = [] {
        const char* env = std::getenv("FA3_REPRO_SKIP_SCHED_PREP");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

inline int round_up_headdimv(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
void
mha_fwd(const FlashattnTensor *q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        const FlashattnTensor *k, // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        const FlashattnTensor *v, // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        //std::optional<const at::Tensor> &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        //std::optional<const at::Tensor> &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        //std::optional<const at::Tensor> &q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        const FlashattnTensor *out, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        const FlashattnTensor *cu_seqlens_q_, // b+1
        const FlashattnTensor *cu_seqlens_k_, // b+1
        /* std::optional<const at::Tensor> &cu_seqlens_k_new_,  // b+1 */
        const FlashattnTensor *seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        const FlashattnTensor *seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        int max_seqlen_q,
        // TODO: check if we need max_seqlen_k
        int max_seqlen_k,
        const FlashattnTensor *page_table_, // (b_k, max_num_pages_per_seq)
        //std::optional<const at::Tensor> &kv_batch_idx_, // b. indices to index into the KV cache
        //std::optional<const at::Tensor> &leftpad_k_, // b
        //std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        //std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        //std::optional<const at::Tensor> &seqlens_rotary_, // b
        const FlashattnTensor *q_descale_, // (b, h_k), not (b, h)
        const FlashattnTensor *k_descale_, // (b, h_k)
        const FlashattnTensor *v_descale_, // (b, h_k)
        const FlashattnTensor *softmax_lse_,
        const FlashattnTensor *softmax_lse_accum_,
        const FlashattnTensor *out_accum_,
        float const softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        float const softcap,
        bool const is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        const FlashattnTensor *scheduler_metadata_, // (b + 1)
        int num_splits,
        //std::optional<bool> pack_gqa_,
        int const sm_margin,
        const FlashattnTensor *s_aux_, // (h)
        int const cp_world_size, // context parallelism (cp) world size
        int const cp_rank, // cp rank
        const FlashattnTensor *cp_tot_seqused_k_, // (b) total seqused_k in cp world
        void* stream) {
    cudaDeviceProp dprops = getCurrentDeviceProperties();
    bool is_sm8x = dprops.major >= 8;
    CAPI_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    CAPI_CHECK(q->dtype == CAPI_FLOAT16 || q->dtype == CAPI_BFLOAT16 || q->dtype == CAPI_F8E4M3FN,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    if (dprops.major < 9) {
        CAPI_CHECK(q->dtype == CAPI_FLOAT16 || q->dtype == CAPI_BFLOAT16,
                    "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type");
    }
    CAPI_CHECK(k->dtype == q->dtype, "query and key must have the same dtype");
    CAPI_CHECK(v->dtype == q->dtype, "query and value must have the same dtype");

    //CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    CAPI_CHECK(getStride(q, -1) == 1, "Input tensor must have contiguous last dimension");
    CAPI_CHECK(getStride(k, -1) == 1, "Input tensor must have contiguous last dimension");
    CAPI_CHECK(getStride(v, -1) == 1, "Input tensor must have contiguous last dimension");

    const FlashattnTensor *page_table{nullptr};
    const bool paged_KV = page_table_ != nullptr;
    if (paged_KV) {
        page_table = page_table_;
        CAPI_CHECK(page_table->dtype == CAPI_INT32, "page_table must have dtype int32");
        CAPI_CHECK(getStride(page_table, -1) == 1, "page_table must have contiguous last dimension");
    }

    const FlashattnTensor* cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_ != nullptr;
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_;
        CAPI_CHECK(cu_seqlens_q->dtype == CAPI_INT32, "cu_seqlens_q must have dtype int32");
        CAPI_CHECK(max_seqlen_q != -1, "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }

    const FlashattnTensor *cu_seqlens_k{nullptr};
    bool const is_varlen_k = cu_seqlens_k_ != nullptr;
    if (is_varlen_k) {
        cu_seqlens_k = cu_seqlens_k_;
        CAPI_CHECK(cu_seqlens_k->dtype == CAPI_INT32, "cu_seqlens_k must have dtype int32");
        CAPI_CHECK(max_seqlen_k != -1, "max_seqlen_k must be provided if cu_seqlens_k is provided");
        CAPI_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
        //CAPI_CHECK(!kv_batch_idx_.has_value(), "If cu_seqlens_k is passed in, then page table is not supported");
    }

    const int batch_size = !is_varlen_q ? getDim(q, 0) : getDim(cu_seqlens_q, 0) - 1;
    int seqlen_q = !is_varlen_q ? getDim(q, 1) : max_seqlen_q;
    int total_q = !is_varlen_q ? batch_size * getDim(q, 1) : getDim(q, 0);
    int num_heads = getDim(q, -2);
    int const head_size = getDim(q, -1);
    int const head_size_v = getDim(v, -1);
    int const max_num_pages_per_seq = !paged_KV ? 0 : getDim(page_table, 1);
    int const num_pages = !paged_KV ? 0 : getDim(k, 0);
    int const page_size = !paged_KV ? 1 : getDim(k, 1);
    int const seqlen_k = max_seqlen_k == -1 ? (!paged_KV ? getDim(k, 1) : max_num_pages_per_seq * page_size) : max_seqlen_k;
    int const num_heads_k = getDim(k, -2);
    int const batch_size_k = !paged_KV ? (!is_varlen_k ? getDim(k, 0) : getDim(cu_seqlens_k, 0) - 1) : getDim(page_table, 0);
    int const total_k = paged_KV
        ? batch_size_k * seqlen_k
        : (!is_varlen_k ? batch_size * getDim(k, 1) : getDim(k, 0));
    //if (!kv_batch_idx_.has_value()) {
    CAPI_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
    //}
    int const max_headdim = get_max_headdim();
    CAPI_CHECK(head_size <= max_headdim, ("FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim)).c_str());
    CAPI_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (head_size_v != head_size) {
        CAPI_CHECK((head_size > 128 && head_size <= 192 && head_size_v > 96 && head_size_v <= 128) ||
                   (head_size <= 64 && head_size_v <= 512),
                   "If V headdim is different from Q/K dim, we only support Q/K headdim in (128, 192] and V headdim in (96, 128], "
                   "or (Q/K <= 64 and V <= 512).");
        CAPI_CHECK(dprops.major == 9, "Only Hopper supports different V headdim");
        if (head_size_v > 256) {
            CAPI_CHECK(q->dtype == CAPI_FLOAT16 || q->dtype == CAPI_BFLOAT16, "HeaddimV > 256 requires fp16 and bf16 data type");
        }
    }

    // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
    // TODO: check this
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
        // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
        if ((head_size <= 64 || head_size > 128) || !paged_KV) {
            is_causal = false;
        }
    }
    if (is_causal) { window_size_right = 0; }
    // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_fprop will set params.is_causal=true.
    // If we don't have is_causal here matching params.is_causal, we might get the wrong kBlockM.
    is_causal = window_size_left < 0 && window_size_right == 0;

    if (!is_varlen_q) {
        CAPI_CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    } else {
        CAPI_CHECK_SHAPE(q, total_q, num_heads, head_size);
        CAPI_CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!paged_KV) {
        if (!is_varlen_k) {
            CAPI_CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
            CAPI_CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
        } else {
            CAPI_CHECK_SHAPE(k, total_k, num_heads_k, head_size);
            CAPI_CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
            CAPI_CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
        }
    } else {
        CAPI_CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
        CAPI_CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
        CAPI_CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
    }

    if (seqused_q_ != nullptr){
        CAPI_CHECK(seqused_q_->dtype == CAPI_INT32, "seqused_q must have dtype int32");
        CAPI_CHECK_SHAPE(seqused_q_, batch_size);
    }
    if (seqused_k_ != nullptr) {
        CAPI_CHECK(seqused_k_->dtype == CAPI_INT32, "seqused_k must have dtype int32");
        CAPI_CHECK_SHAPE(seqused_k_, batch_size);
    }

    //if (leftpad_k_.has_value()) {
    //    auto leftpad_k = leftpad_k_.value();
    //    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    //    CHECK_DEVICE(leftpad_k); CHECK_CONTIGUOUS(leftpad_k);
    //    CHECK_SHAPE(leftpad_k, batch_size);
    //}

    // This is what we will template on
    bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_ != nullptr || seqused_k_ != nullptr /*|| leftpad_k_.has_value()*/;
    #ifdef FLASHATTENTION_DISABLE_VARLEN
        CAPI_CHECK(!is_varlen, "This flash attention build does not support varlen.");
    #endif

    int const alignment = q->dtype == CAPI_F8E4M3FN ? 16 : 8;
    CAPI_CHECK(head_size % alignment == 0, ("head_size should be a multiple of " + std::to_string(alignment)).c_str());
    CAPI_CHECK(head_size_v % alignment == 0, ("head_size_v should be a multiple of " + std::to_string(alignment)).c_str());

    //auto opts = q.options();
    //auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
    //at::Tensor out;
    //if (out_.has_value()) {
    //    out = out_.value();
    //    TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
    //    CHECK_DEVICE(out);
    //    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    //    if (!is_varlen_q) {
    //        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
    //    } else {
    //        CHECK_SHAPE(out, total_q, num_heads, head_size_v);
    //    }
    //} else {
    //    out = !is_varlen_q
    //        ? torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(out_type))
    //        : torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type));
    //}

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded = round_up_headdim(head_size);
    int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdimv(head_size_v);
    int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    //at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    //at::Tensor softmax_lse;
    //if (!is_varlen_q) {
    //    softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    //} else {
    //    softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    //}

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     !is_varlen_q ? nullptr : cu_seqlens_q->ptr,
                     !is_varlen_k ? nullptr : cu_seqlens_k->ptr,
                     seqused_q_ != nullptr ? seqused_q_->ptr : nullptr,
                     seqused_k_ != nullptr ? seqused_k_->ptr : nullptr,
                     softmax_lse_->ptr,
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     sm_margin);
    params.total_q = total_q;
    params.total_k = total_k;
    params.b_k = batch_size_k;
    params.dv = head_size_v;
    params.dv_rounded = head_size_v_rounded;
    //if (leftpad_k_.has_value()) {  // This needs to be set before get_pagedkv_tma
    //    params.leftpad_k = static_cast<int *>(leftpad_k_.value().data_ptr());
    //}
    if (paged_KV) {
        params.page_table = static_cast<int*>(page_table->ptr);
        params.page_table_batch_stride = getStride(page_table, 0);
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    //if (k_new_.has_value()) {  // This needs to be set before get_pagedkv_tma
    //    at::Tensor k_new, v_new;
    //    TORCH_CHECK(v_new_.has_value(), "If k_new is supplied, v_new must also be passed in");
    //    TORCH_CHECK(seqused_k_.has_value(), "If k_new is supplied, seqlens_k must also be passed in");
    //    TORCH_CHECK(seqlen_q <= seqlen_k, "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache");
    //    at::Tensor cu_seqlens_k_new;
    //    bool const is_varlen_k_new = cu_seqlens_k_new_.has_value();
    //    if (is_varlen_k_new) {
    //        cu_seqlens_k_new = cu_seqlens_k_new_.value();
    //        CHECK_DEVICE(cu_seqlens_k_new); CHECK_CONTIGUOUS(cu_seqlens_k_new);
    //        TORCH_CHECK(cu_seqlens_k_new.dtype() == torch::kInt32, "cu_seqlens_k_new must have dtype torch.int32");
    //    }
    //    k_new = k_new_.value();
    //    v_new = v_new_.value();
    //    TORCH_CHECK(k_new.dtype() == q_type, "k_new must have the same dtype as query");
    //    TORCH_CHECK(v_new.dtype() == q_type, "v_new must have the same dtype as query");
    //    CHECK_DEVICE(k_new); CHECK_DEVICE(v_new);
    //    TORCH_CHECK(k_new.stride(-1) == 1, "k_new tensor must have contiguous last dimension");
    //    TORCH_CHECK(v_new.stride(-1) == 1, "v_new tensor must have contiguous last dimension");
    //    // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when is_varlen_k_new
    //    int seqlen_k_new = !is_varlen_k_new ? k_new.size(1) : 0;
    //    int total_k_new = !is_varlen_k_new ? batch_size * k_new.size(1): k_new.size(0);
    //    if (!is_varlen_k_new) {
    //        CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
    //        CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
    //    } else {
    //        CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
    //        CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
    //        CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
    //    }
    //    params.seqlen_knew = seqlen_k_new;
    //    params.total_knew = total_k_new;
    //    params.knew_ptr = k_new.data_ptr();
    //    params.vnew_ptr = v_new.data_ptr();
    //    // All stride are in elements, not bytes.
    //    params.knew_row_stride = k_new.stride(-3);
    //    params.vnew_row_stride = v_new.stride(-3);
    //    params.knew_head_stride = k_new.stride(-2);
    //    params.vnew_head_stride = v_new.stride(-2);
    //    if (!is_varlen_k_new) {
    //        params.knew_batch_stride = k_new.stride(0);
    //        params.vnew_batch_stride = v_new.stride(0);
    //    }
    //    if (is_varlen_k_new) {
    //        params.cu_seqlens_knew = static_cast<int*>(cu_seqlens_k_new.data_ptr());
    //    }
    //}

    // 992 = 32 * 31 is the max supported batch in prepare_varlen_num_blocks kernel
    bool const use_dynamic_split = is_varlen && params.b <= 992 && num_splits != 1;
    // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
    params.num_splits_dynamic_ptr = !use_dynamic_split ? nullptr : reinterpret_cast<int*>(1);

    params.pagedkv_tma = get_pagedkv_tma(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    // Determine if we should pack GQA before num_splits since it impacts use_one_mma_wg (in get_num_splits)
    //params.pack_gqa = pack_gqa_.has_value() ? pack_gqa_.value() : get_pack_gqa(params);
    params.pack_gqa = get_pack_gqa(params);
    // Always enable PackGQA for Split
    params.pack_gqa |= params.num_splits > 1;

    // This needs to be set after get_num_splits
    //at::Tensor tile_count_semaphore;  // Contains the semaphore and optionally num_splits_dynamic
    const FlashattnTensor* tile_count_semaphore{nullptr};
    // We don't use the persistent scheduler if Split and not Varlen
    bool const scheduler_needs_semaphore = params.arch >= 90
        ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
        : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));

    if (scheduler_needs_semaphore || use_dynamic_split) {
        int metadata_size = int(scheduler_needs_semaphore) + int(use_dynamic_split) * params.b;
        params.skip_scheduler_metadata_computation = fa3_repro_skip_sched_prep_enabled();
        //if (numElements(scheduler_metadata_) != metadata_size) {
        //    fprintf(stderr, "scheduler_metadata_ requires %d bytes, allocated %ld bytes\n", metadata_size, numElements(scheduler_metadata_));
        //    CAPI_CHECK(false, "");
        //}
        //CAPI_CHECK(scheduler_metadata_->dtype == CAPI_INT32, "scheduler_metadata_ dtype should be int32");
        CAPI_CHECK(scheduler_metadata_ != nullptr, "scheduler_metadata_ needs to be passed");
        //if (scheduler_metadata_ != nullptr) {
            CAPI_CHECK(getDim(scheduler_metadata_, 0) >= metadata_size, "scheduler_metadata must have sufficient size");
            CAPI_CHECK(scheduler_metadata_->dtype == CAPI_INT32, "scheduler_metadata must have dtype int32");
            tile_count_semaphore = scheduler_metadata_;
        //} else {
        //    tile_count_semaphore = torch::empty({metadata_size}, opts.dtype(torch::kInt32));
        //}
        //if (scheduler_needs_semaphore && !use_dynamic_split) {
        //    tile_count_semaphore.zero_();  // If varlen we'll manually do the zero-ing
        //}
        params.tile_count_semaphore = scheduler_needs_semaphore ? static_cast<int*>(tile_count_semaphore->ptr) : nullptr;
        params.num_splits_dynamic_ptr = use_dynamic_split ? static_cast<int*>(tile_count_semaphore->ptr) + 1 : nullptr;
    }

    if (fa3_repro_debug_enabled()) {
        std::fprintf(stderr,
                     "fa3_debug arch=%d b=%d h=%d h_k=%d d=%d dv=%d seqlen_q=%d seqlen_k=%d "
                     "varlen_q=%d varlen_k=%d paged_kv=%d pagedkv_tma=%d num_splits=%d "
                     "use_dynamic_split=%d scheduler_needs_semaphore=%d pack_gqa=%d "
                     "skip_sched_meta=%d page_size=%d num_pages=%d total_q=%d total_k=%d\n",
                     params.arch,
                     params.b,
                     params.h,
                     params.h_k,
                     params.d,
                     params.dv,
                     params.seqlen_q,
                     params.seqlen_k,
                     is_varlen_q,
                     is_varlen_k,
                     paged_KV,
                     params.pagedkv_tma,
                     params.num_splits,
                     use_dynamic_split,
                     scheduler_needs_semaphore,
                     params.pack_gqa,
                     params.skip_scheduler_metadata_computation,
                     params.page_size,
                     params.num_pages,
                     params.total_q,
                     params.total_k);
    }

    //if (q_v_.has_value()) {
    //    TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
    //    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
    //                "q_v is only supported for fp16 and bf16 data type");
    //    TORCH_CHECK(params.arch == 90, "q_v is only supported for Hopper GPUs");
    //    at::Tensor q_v = q_v_.value();
    //    TORCH_CHECK(q_v.dtype() == q_type, "q_v must have the same dtype as query");
    //    CHECK_DEVICE(q_v);
    //    TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
    //    if (!is_varlen_q) {
    //        CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
    //    } else {
    //        CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    //    }
    //    params.qv_ptr = q_v.data_ptr();
    //    // All stride are in elements, not bytes.
    //    params.qv_row_stride = q_v.stride(-3);
    //    params.qv_head_stride = q_v.stride(-2);
    //    if (!is_varlen_q) {
    //        params.qv_batch_stride = q_v.stride(0);
    //    }
    //}

    //if (rotary_cos_.has_value()) {
    //    TORCH_CHECK(k_new_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
    //    auto rotary_cos = rotary_cos_.value();
    //    CHECK_DEVICE(rotary_cos); CHECK_CONTIGUOUS(rotary_cos);
    //    params.rotary_dim = rotary_cos.size(1) * 2;
    //    TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
    //    TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    //    const int seqlen_ro = rotary_cos.size(0);
    //    if (paged_KV) {
    //        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    //    }
    //    CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
    //    TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

    //    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    //    auto rotary_sin = rotary_sin_.value();
    //    CHECK_DEVICE(rotary_sin); CHECK_CONTIGUOUS(rotary_sin);
    //    CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
    //    TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
    //    params.rotary_cos_ptr = rotary_cos.data_ptr();
    //    params.rotary_sin_ptr = rotary_sin.data_ptr();
    //    params.is_rotary_interleaved = is_rotary_interleaved;
    //    if (seqlens_rotary_.has_value()) {
    //        at::Tensor seqlens_rotary = seqlens_rotary_.value();
    //        CHECK_DEVICE(seqlens_rotary); CHECK_CONTIGUOUS(seqlens_rotary);
    //        TORCH_CHECK(seqlens_rotary.dtype() == torch::kInt32, "seqlens_rotary must have dtype torch.int32");
    //        CHECK_SHAPE(seqlens_rotary, batch_size);
    //        params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
    //    }
    //} else {
    params.rotary_dim = 0;
    //}

    //if (kv_batch_idx_.has_value()) {
    //    auto kv_batch_idx = kv_batch_idx_.value();
    //    CHECK_DEVICE(kv_batch_idx); CHECK_CONTIGUOUS(kv_batch_idx);
    //    TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
    //    params.kv_batch_idx = reinterpret_cast<int *>(kv_batch_idx.data_ptr());
    //}

    const FlashattnTensor *out_accum;
    const FlashattnTensor *softmax_lse_accum;
    //auto outaccum_type = at::ScalarType::Float;
    if (params.num_splits > 1) {
        CAPI_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        out_accum = out_accum_;
        softmax_lse_accum = softmax_lse_accum_;
        if (!is_varlen_q) {
            //out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_v}, opts.dtype(outaccum_type));
            //softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        
            const int64_t float_size = 4;
            const int64_t out_accum_size = (int64_t)params.num_splits * (int64_t)batch_size * (int64_t)num_heads * (int64_t)seqlen_q * (int64_t)head_size_v * (int64_t)float_size;
            const int64_t allocated_out_accum_size = byteSize(out_accum);
            if (out_accum_size > allocated_out_accum_size) {
                fprintf(stderr, "out_accum requires %ld bytes, allocated %ld bytes\n", out_accum_size, allocated_out_accum_size);
                CAPI_CHECK(false, "");
            }
            const int64_t softmax_lse_accum_size = (int64_t)params.num_splits * (int64_t)batch_size * (int64_t)num_heads * (int64_t)seqlen_q * (int64_t)float_size;
            const int64_t allocated_softmax_lse_accum_size = byteSize(softmax_lse_accum);
            if (softmax_lse_accum_size > allocated_softmax_lse_accum_size) {
                fprintf(stderr, "softmax_lse_accum requires %ld bytes, allocated %ld bytes\n", softmax_lse_accum_size, allocated_softmax_lse_accum_size);
                CAPI_CHECK(false, "");
            }

            params.oaccum_batch_stride = num_heads * seqlen_q * head_size_v;
            params.lseaccum_batch_stride = num_heads * seqlen_q;
            params.oaccum_split_stride = batch_size * num_heads * total_q * head_size_v;
            params.oaccum_row_stride = head_size_v;
            params.oaccum_head_stride = seqlen_q * head_size_v;
            params.lseaccum_split_stride = num_heads * total_q;
            params.lseaccum_head_stride = seqlen_q;
        } else {
            //out_accum = torch::empty({params.num_splits, num_heads, total_q, head_size_v}, opts.dtype(outaccum_type));
            //softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q}, opts.dtype(at::kFloat));

            const int64_t float_size = 4;
            const int64_t out_accum_size = (int64_t)params.num_splits * (int64_t)num_heads * (int64_t)total_q * (int64_t)head_size_v * (int64_t)float_size;
            const int64_t allocated_out_accum_size = byteSize(out_accum);
            if (out_accum_size > allocated_out_accum_size) {
                fprintf(stderr, "out_accum requires %ld bytes, allocated %ld bytes\n", out_accum_size, allocated_out_accum_size);
                CAPI_CHECK(false, "");
            }
            const int64_t softmax_lse_accum_size = (int64_t)params.num_splits * (int64_t)num_heads * (int64_t)total_q * (int64_t)float_size;
            const int64_t allocated_softmax_lse_accum_size = byteSize(softmax_lse_accum);
            if (softmax_lse_accum_size > allocated_softmax_lse_accum_size) {
                fprintf(stderr, "softmax_lse_accum requires %ld bytes, allocated %ld bytes\n", softmax_lse_accum_size, allocated_softmax_lse_accum_size);
                CAPI_CHECK(false, "");
            }


            params.oaccum_split_stride = num_heads * total_q * head_size_v;
            params.oaccum_row_stride = head_size_v;
            params.oaccum_head_stride = total_q * head_size_v;
            params.lseaccum_split_stride = num_heads * total_q;
            params.lseaccum_head_stride = total_q;
        }
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum->ptr;
        params.softmax_lseaccum_ptr = softmax_lse_accum->ptr;
    }

    if (q->dtype == CAPI_F8E4M3FN) {
        if (q_descale_ != nullptr) {
            CAPI_CHECK_SHAPE(q_descale_, batch_size, num_heads_k);
            params.q_descale_ptr = static_cast<float*>(q_descale_->ptr);
            params.q_descale_batch_stride = getStride(q_descale_, 0);
            params.q_descale_head_stride = getStride(q_descale_, 1);
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_ != nullptr) {
            CAPI_CHECK_SHAPE(k_descale_, batch_size, num_heads_k);
            params.k_descale_ptr = static_cast<float*>(k_descale_->ptr);
            params.k_descale_batch_stride = getStride(k_descale_, 0);
            params.k_descale_head_stride = getStride(k_descale_, 1);
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_ != nullptr) {
            CAPI_CHECK_SHAPE(v_descale_, batch_size, num_heads_k);
            params.v_descale_ptr = static_cast<float*>(v_descale_->ptr);
            params.v_descale_batch_stride = getStride(v_descale_, 0);
            params.v_descale_head_stride = getStride(v_descale_, 1);
        } else {
            params.v_descale_ptr = nullptr;
        }
    }
    
    if(s_aux_ != nullptr) {
        CAPI_CHECK(params.arch == 90, "S aux is currently only supported for Hopper GPUs");
        CAPI_CHECK(num_heads <= 64, "We only support query heads <= 64 with S aux");
        CAPI_CHECK(head_size == head_size_v, "We don't support S aux with hdim != hdim_v");
        CAPI_CHECK(s_aux_->dtype == CAPI_BFLOAT16,
            "We only support bf16 dtype for S aux.");
        CAPI_CHECK_SHAPE(s_aux_, num_heads);
        params.s_aux_ptr = s_aux_->ptr;
    } else {
        params.s_aux_ptr = nullptr;
    }

    params.cp_world_size = cp_world_size;
    params.cp_rank = cp_rank;
    params.cp_tot_seqused_k = cp_tot_seqused_k_ != nullptr ? reinterpret_cast<int*>(cp_tot_seqused_k_->ptr) : nullptr;
    CAPI_CHECK(cp_world_size > 0, "cp_world_size must be positive, required by downstream unified code path. Use 1 if CP is not enabled.");
    CAPI_CHECK(cp_world_size != 1 || cp_rank == 0, "When context parallelism is disabled, cp_rank must be zero");
    CAPI_CHECK(cp_world_size == 1 || cp_tot_seqused_k_->ptr != nullptr, "cp_tot_seqused_k_ must be provided when context parallelism is enabled.");
    CAPI_CHECK(!(params.is_local && cp_world_size > 1), 
        "Local attention (sliding window) is not currently supported with context parallelism (cp_world_size > 1)."
        "Requires proper n_offset handling in block boundary calculations in mainloop and block.h");



    //#ifdef FLASHATTENTION_DISABLE_LOCAL
    //TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    //#endif
    //#ifdef FLASHATTENTION_DISABLE_SOFTCAP
    //TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
    //#endif
    //#ifdef FLASHATTENTION_DISABLE_SPLIT
    //TORCH_CHECK(params.num_splits == 1, "This flash attention build does not support splits.");
    //#endif
    //#ifdef FLASHATTENTION_DISABLE_PACKGQA
    //TORCH_CHECK(!params.pack_gqa || params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1, "This flash attention build does not support pack_gqa.");
    //#endif
    //#ifdef FLASHATTENTION_DISABLE_PAGEDKV
    //TORCH_CHECK(!(params.page_table && !params.pagedkv_tma), "This flash attention build does not support paged KV.");
    //#endif
    //#ifdef FLASHATTENTION_DISABLE_APPENDKV
    //TORCH_CHECK(!k_new_.has_value(), "This flash attention build does not support appending KV.");
    //#endif

    auto out_type = q->dtype;
    if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
        run_mha_fwd(params, reinterpret_cast<cudaStream_t>(stream));
        if (params.num_splits > 1) {
            if (out_type == CAPI_BFLOAT16) {
                // Since we want output in BF16. Otherwise fwd_combine will output to FP16
                params.is_bf16 = true;
            }
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            // However, with dynamic split, each row needs to know which batch it belongs to
            // to read the number of splits, so we just use the varlen version of combine kernel.
            // if (is_varlen_q && !seqused_q_.has_value()) {
            // if (is_varlen_q) {
            //     params.b = 1;
            //     params.seqlen_q = total_q;
            // }
            // This will zero out the semaphore if needed
            run_mha_fwd_combine(params, reinterpret_cast<cudaStream_t>(stream), true /*enable_pdl*/);
        } else if (scheduler_needs_semaphore && params.skip_scheduler_metadata_computation) {
            // need to zero out the semaphore in this case
            //tile_count_semaphore.index({torch::indexing::Slice(0, 1)}).zero_();
        }
    } else if (total_q > 0 && num_heads_k > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        //out.zero_();
        //softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // return {out, softmax_lse};
    //return {out, softmax_lse, out_accum, softmax_lse_accum};
}
}


void fa3_mha_fwd(
        const FlashattnTensor *q,
        const FlashattnTensor *k,
        const FlashattnTensor *v,
        const FlashattnTensor *out,
        const FlashattnTensor *cu_seqlens_q,
        const FlashattnTensor *cu_seqlens_k,
        const FlashattnTensor *seqused_q,
        const FlashattnTensor *seqused_k,
        const FlashattnTensor *page_table,
        const FlashattnTensor *q_descale_,
        const FlashattnTensor *k_descale_,
        const FlashattnTensor *v_descale_,
        const FlashattnTensor *softmax_lse,
        const FlashattnTensor *softmax_lse_accum,
        const FlashattnTensor *out_accum,
        const FlashattnTensor *scheduler_metadata,
        const FlashattnTensor *s_aux,
        const FlashattnTensor *cp_tot_seqused_k,
        const FA3MhaFwdParams *params,
        void* stream) {
    FLASH_NAMESPACE::mha_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        params->max_seqlen_q,
        params->max_seqlen_k,
        page_table,
        q_descale_,
        k_descale_,
        v_descale_,
        softmax_lse,
        softmax_lse_accum,
        out_accum,
        params->softmax_scale,
        params->is_causal,
        params->window_size_left,
        params->window_size_right,
        params->softcap,
        params->is_rotary_interleaved,
        scheduler_metadata,
        params->num_splits,
        params->sm_margin,
        s_aux,
        params->cp_world_size,
        params->cp_rank,
        cp_tot_seqused_k,
        stream
    );
}

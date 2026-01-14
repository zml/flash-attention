#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CAPI_F8E4M3FN,
    CAPI_FLOAT,
    CAPI_FLOAT16,
    CAPI_BFLOAT16,
    CAPI_INT32,
    CAPI_INT8,
} DataType;

typedef struct FA2MhaVarlenFwdParams {
    int max_seqlen_q;
    int max_seqlen_k;

    bool is_causal;
    float softmax_scale;

    int window_size_left;
    int window_size_right;

    int num_splits;
    int num_heads;
} FA2MhaVarlenFwdParams;

typedef struct FA2MhaFwdParams {
    bool is_causal;
    float softmax_scale;

    int window_size_left;
    int window_size_right;
} FA2MhaFwdParams;

typedef struct FA3MhaFwdParams {
    int max_seqlen_q;
    int max_seqlen_k;

    float softcap;
    bool is_rotary_interleaved;
    int num_splits;
    int const sm_margin;

    bool is_causal;
    float softmax_scale;

    int window_size_left;
    int window_size_right;

    int cp_world_size;
    int cp_rank;
} FA3MhaFwdParams;

typedef struct FlashattnTensor {
    int64_t strides[8];
    int64_t dims[8];
    uint32_t rank;
    void *ptr;
    DataType dtype;
} FlashattnTensor;

void fa2_mha_fwd(
        const FlashattnTensor *q, // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const FlashattnTensor *k, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const FlashattnTensor *v, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const FlashattnTensor *out, // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const FlashattnTensor *softmax_lse, // batch_size x num_heads x seqlen_q
        const FlashattnTensor *alibi_slopes_, // num_heads or batch_size x num_heads
        const FlashattnTensor *softmax_lse_accum,
        const FlashattnTensor *out_accum,
        const FA2MhaFwdParams *params,
        void* stream);

void fa2_mha_varlen_fwd(
        const FlashattnTensor *q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const FlashattnTensor *k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        const FlashattnTensor *v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        const FlashattnTensor *out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const FlashattnTensor *cu_seqlens_q, // b+1
        const FlashattnTensor *cu_seqlens_k, // b+1
        const FlashattnTensor *seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        const FlashattnTensor *block_table, // batch_size x max_num_blocks_per_seq
        const FlashattnTensor *softmax_lse, // num_heads x total_q
        const FlashattnTensor *alibi_slopes_, // num_heads or b x num_heads
        const FlashattnTensor *softmax_lse_accum,
        const FlashattnTensor *out_accum,
        const FA2MhaVarlenFwdParams *params,
        void* stream);

void fa3_mha_fwd(
        const FlashattnTensor *q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        const FlashattnTensor *k, // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        const FlashattnTensor *v, // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        const FlashattnTensor *out, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        const FlashattnTensor *cu_seqlens_q, // b+1
        const FlashattnTensor *cu_seqlens_k, // b+1
        const FlashattnTensor *seqused_q, // b. If given, only this many elements of each batch element's queries and outputs are used.
        const FlashattnTensor *seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        const FlashattnTensor *page_table, // (b_k, max_num_pages_per_seq)
        const FlashattnTensor *q_descale_,  // (b, h_k), not (b, h)
        const FlashattnTensor *k_descale_,  // (b, h_k)
        const FlashattnTensor *v_descale_,  // (b, h_k)
        const FlashattnTensor *softmax_lse,
        const FlashattnTensor *softmax_lse_accum,
        const FlashattnTensor *out_accum,
        const FlashattnTensor *scheduler_metadata, // (b + 1)
        const FlashattnTensor *s_aux, // (h)
        const FlashattnTensor *cp_tot_seqused_k, // (b) total seqused_k in cp world
        const FA3MhaFwdParams *params,
        void* stream);

#ifdef __cplusplus
}
#endif

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
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

    int num_heads;
    int num_splits;
} FA2MhaVarlenFwdParams;

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

void fa2_mha_varlen_fwd(
        FlashattnTensor q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        FlashattnTensor k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        FlashattnTensor v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        FlashattnTensor out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        FlashattnTensor cu_seqlens_q, // b+1
        FlashattnTensor cu_seqlens_k, // b+1
        FlashattnTensor seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        FlashattnTensor block_table, // batch_size x max_num_blocks_per_seq
        FlashattnTensor softmax_lse, // num_heads x total_q
        FlashattnTensor alibi_slopes_, // num_heads or b x num_heads
        FlashattnTensor softmax_lse_accum,
        FlashattnTensor out_accum,
        FA2MhaVarlenFwdParams params,
        void* stream);

void fa3_mha_fwd(FlashattnTensor q,
        FlashattnTensor k,
        FlashattnTensor v,
        FlashattnTensor out,
        FlashattnTensor cu_seqlens_q,
        FlashattnTensor seqused_k,
        FlashattnTensor page_table,
        FlashattnTensor softmax_lse,
        FlashattnTensor softmax_lse_accum,
        FlashattnTensor out_accum,
        FlashattnTensor scheduler_metadata,
        FlashattnTensor s_aux,
        FlashattnTensor cp_tot_seqused_k,
        FA3MhaFwdParams params,
        void* stream);

#ifdef __cplusplus
}
#endif

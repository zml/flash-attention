#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CAPI_FLOAT16,
    CAPI_BFLOAT16,
} DataType;

typedef struct FlashattnMhaVarlenFwdParams {
    DataType q_dtype;

    int64_t q_batch_stride;
    int64_t q_row_stride;
    int64_t q_head_stride;

    int64_t k_batch_stride;
    int64_t k_row_stride;
    int64_t k_head_stride;

    int64_t v_batch_stride;
    int64_t v_row_stride;
    int64_t v_head_stride;

    int64_t o_batch_stride;
    int64_t o_row_stride;
    int64_t o_head_stride;

    int64_t block_table_batch_stride;

    int softmax_lse_accum_size;
    int out_accum_size;

    int max_seqlen_q;
    int max_seqlen_k;

    bool is_causal;
    float softmax_scale;

    int window_size_left;
    int window_size_right;

    uint32_t total_q;
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t num_heads_k;
    uint32_t head_size;
    uint32_t page_block_size;
} FlashattnMhaVarlenFwdParams;

void flashattn_mha_varlen_fwd(
        void* q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        void* out, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void* cu_seqlens_q, // b+1
        void* cu_seqlens_k, // b+1
        void* seqused_k, // b. If given, only this many elements of each batch element's keys are used.
        void* block_table, // batch_size x max_num_blocks_per_seq
        void* softmax_lse, // num_heads x total_q
        void* alibi_slopes_, // num_heads or b x num_heads
        void* softmax_lse_accum,
        void* out_accum,
        FlashattnMhaVarlenFwdParams params,
        void* stream);

#ifdef __cplusplus
}
#endif

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
        void* stream);

#ifdef __cplusplus
}
#endif

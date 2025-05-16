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
                      int64_t v_dim_stride,
                      int64_t o_row_stride,
                      int64_t o_head_stride,
                      DataType q_dtype,
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

    params.is_bf16 = q_dtype == CAPI_BFLOAT16;
    params.is_e4m3 = false;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.v_dim_stride = v_dim_stride;
    params.o_ptr = out;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;

    //if (cu_seqlens_q_d == nullptr) {
    //    params.q_batch_stride = q.stride(0);
    //    params.o_batch_stride = out.stride(0);
    //}
    //if (cu_seqlens_k_d == nullptr) {
    //    params.k_batch_stride = k.stride(0);
    //    params.v_batch_stride = v.stride(0);
    //}

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
    if (window_size_left < 0) { window_size_left = seqlen_k - 1; }
    if (window_size_right < 0) { window_size_right = seqlen_q - 1; }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = attention_chunk;

    int device = 0;
    CUDA_CALL(cudaGetDevice());
    cudaDeviceProp prop = {};
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    params.arch = prop.major * 10 + prop.minor;
    params.num_sm = prop.multiProcessorCount - sm_margin;
}


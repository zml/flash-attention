#include "capi/capi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t err__ = (expr);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                 \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

struct Options {
    int batch = 1;
    int seqlen_q = 8;
    int seqlen_k = 8;
    int num_heads = 4;
    int num_heads_k = 4;
    int head_dim = 128;
    bool causal = true;
    int iters = 3;
    int seed = 123;
    bool bf16 = true;
    int num_splits = 1;
    bool varlen = true;
    bool paged_kv = true;
    int page_size = 16;
    int seqused_k = -1;
    bool skip_ref = false;
    float atol = 3e-2f;
    float rtol = 3e-2f;
};

struct Stats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float mean_abs = 0.0f;
    int max_idx = -1;
    float ref_at_max = 0.0f;
    float got_at_max = 0.0f;
};

bool ParseBool(std::string_view value) {
    return value == "1" || value == "true" || value == "True";
}

bool ParseArg(std::string_view arg, std::string_view name, std::string* value) {
    std::string prefix = "--";
    prefix += name;
    prefix += "=";
    if (!arg.starts_with(prefix)) {
        return false;
    }
    *value = std::string(arg.substr(prefix.size()));
    return true;
}

Options ParseOptions(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string value;
        std::string_view arg(argv[i]);
        if (ParseArg(arg, "batch", &value)) opts.batch = std::stoi(value);
        else if (ParseArg(arg, "seqlen_q", &value)) opts.seqlen_q = std::stoi(value);
        else if (ParseArg(arg, "seqlen_k", &value)) opts.seqlen_k = std::stoi(value);
        else if (ParseArg(arg, "num_heads", &value)) opts.num_heads = std::stoi(value);
        else if (ParseArg(arg, "num_heads_k", &value)) opts.num_heads_k = std::stoi(value);
        else if (ParseArg(arg, "head_dim", &value)) opts.head_dim = std::stoi(value);
        else if (ParseArg(arg, "causal", &value)) opts.causal = ParseBool(value);
        else if (ParseArg(arg, "iters", &value)) opts.iters = std::stoi(value);
        else if (ParseArg(arg, "seed", &value)) opts.seed = std::stoi(value);
        else if (ParseArg(arg, "bf16", &value)) opts.bf16 = ParseBool(value);
        else if (ParseArg(arg, "num_splits", &value)) opts.num_splits = std::stoi(value);
        else if (ParseArg(arg, "varlen", &value)) opts.varlen = ParseBool(value);
        else if (ParseArg(arg, "paged_kv", &value)) opts.paged_kv = ParseBool(value);
        else if (ParseArg(arg, "page_size", &value)) opts.page_size = std::stoi(value);
        else if (ParseArg(arg, "seqused_k", &value)) opts.seqused_k = std::stoi(value);
        else if (ParseArg(arg, "skip_ref", &value)) opts.skip_ref = ParseBool(value);
        else if (ParseArg(arg, "atol", &value)) opts.atol = std::stof(value);
        else if (ParseArg(arg, "rtol", &value)) opts.rtol = std::stof(value);
        else if (arg == "--help") {
            std::cout
                << "Usage: bazel run //:fa3_sm90_repro -- "
                << "[--batch=N] [--seqlen_q=N] [--seqlen_k=N] [--num_heads=N] "
                << "[--num_heads_k=N] [--head_dim=N] [--causal=0|1] [--iters=N] "
                << "[--seed=N] [--bf16=0|1] [--num_splits=N] [--varlen=0|1] "
                << "[--paged_kv=0|1] [--page_size=N] [--seqused_k=N] [--skip_ref=0|1] "
                << "[--atol=F] [--rtol=F]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << arg << std::endl;
            std::exit(1);
        }
    }
    if (opts.num_heads % opts.num_heads_k != 0) {
        std::cerr << "num_heads must be divisible by num_heads_k\n";
        std::exit(1);
    }
    if (opts.page_size <= 0 || opts.page_size % 16 != 0) {
        std::cerr << "page_size must be positive and divisible by 16\n";
        std::exit(1);
    }
    if (opts.seqused_k > opts.seqlen_k) {
        std::cerr << "seqused_k must be <= seqlen_k\n";
        std::exit(1);
    }
    return opts;
}

size_t Product(const std::vector<int64_t>& dims) {
    size_t n = 1;
    for (int64_t d : dims) n *= static_cast<size_t>(d);
    return n;
}

FlashattnTensor MakeTensor(void* ptr, DataType dtype, const std::vector<int64_t>& dims) {
    FlashattnTensor t{};
    t.ptr = ptr;
    t.dtype = dtype;
    t.rank = static_cast<uint32_t>(dims.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        t.dims[i] = dims[i];
        t.strides[i] = stride;
        stride *= dims[i];
    }
    return t;
}

uint16_t FloatToBf16Bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float Bf16BitsToFloat(uint16_t x) {
    uint32_t u = static_cast<uint32_t>(x) << 16;
    float out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

uint16_t FloatToF16Bits(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float F16BitsToFloat(uint16_t x) {
    __half h;
    std::memcpy(&h, &x, sizeof(x));
    return __half2float(h);
}

void FillInput(std::vector<uint16_t>* out, const Options& opts) {
    std::mt19937 rng(opts.seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    for (uint16_t& v : *out) {
        float x = dist(rng);
        v = opts.bf16 ? FloatToBf16Bits(x) : FloatToF16Bits(x);
    }
}

std::vector<float> DecodeToFloat(const std::vector<uint16_t>& src, bool bf16) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = bf16 ? Bf16BitsToFloat(src[i]) : F16BitsToFloat(src[i]);
    }
    return out;
}

std::vector<float> CpuReference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const Options& opts) {
    const int b = opts.batch;
    const int sq = opts.seqlen_q;
    const int sk = opts.seqlen_k;
    const int h = opts.num_heads;
    const int hk = opts.num_heads_k;
    const int d = opts.head_dim;
    const int gqa_ratio = h / hk;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> out(static_cast<size_t>(b) * sq * h * d, 0.0f);
    std::vector<float> scores(sk, 0.0f);
    std::vector<float> probs(sk, 0.0f);

    auto q_idx = [=](int ib, int iq, int ih, int id) {
        return (((ib * sq + iq) * h + ih) * d + id);
    };
    auto kv_idx = [=](int ib, int ik, int ih, int id) {
        return (((ib * sk + ik) * hk + ih) * d + id);
    };

    for (int ib = 0; ib < b; ++ib) {
        for (int iq = 0; iq < sq; ++iq) {
            for (int ih = 0; ih < h; ++ih) {
                const int ikh = ih / gqa_ratio;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int ik = 0; ik < sk; ++ik) {
                    if (opts.causal && ik > iq) {
                        scores[ik] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float dot = 0.0f;
                    for (int id = 0; id < d; ++id) {
                        dot += q[q_idx(ib, iq, ih, id)] * k[kv_idx(ib, ik, ikh, id)];
                    }
                    scores[ik] = dot * scale;
                    max_score = std::max(max_score, scores[ik]);
                }
                float denom = 0.0f;
                for (int ik = 0; ik < sk; ++ik) {
                    probs[ik] = std::isinf(scores[ik]) ? 0.0f : std::exp(scores[ik] - max_score);
                    denom += probs[ik];
                }
                for (int ik = 0; ik < sk; ++ik) {
                    probs[ik] /= denom;
                }
                for (int id = 0; id < d; ++id) {
                    float acc = 0.0f;
                    for (int ik = 0; ik < sk; ++ik) {
                        acc += probs[ik] * v[kv_idx(ib, ik, ikh, id)];
                    }
                    out[q_idx(ib, iq, ih, id)] = acc;
                }
            }
        }
    }
    return out;
}

void FillPagedKv(
    std::vector<uint16_t>* paged,
    const std::vector<uint16_t>& logical,
    const Options& opts,
    int max_pages_per_seq) {
    const int hk = opts.num_heads_k;
    const int d = opts.head_dim;
    const int logical_stride = hk * d;
    const int page_stride = opts.page_size * logical_stride;
    std::fill(paged->begin(), paged->end(), 0);
    for (int ib = 0; ib < opts.batch; ++ib) {
        for (int ik = 0; ik < opts.seqlen_k; ++ik) {
            const int page_idx = ik / opts.page_size;
            const int page_offset = ik % opts.page_size;
            const int phys_page = ib * max_pages_per_seq + page_idx;
            const size_t dst_base =
                static_cast<size_t>(phys_page) * page_stride +
                static_cast<size_t>(page_offset) * logical_stride;
            const size_t src_base =
                static_cast<size_t>(ib * opts.seqlen_k + ik) * logical_stride;
            std::copy_n(logical.data() + src_base, logical_stride, paged->data() + dst_base);
        }
    }
}

Stats Compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Stats s;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float abs_err = std::abs(ref[i] - got[i]);
        const float rel_err = abs_err / std::max(std::abs(ref[i]), 1e-6f);
        s.mean_abs += abs_err;
        if (abs_err > s.max_abs) {
            s.max_abs = abs_err;
            s.max_rel = rel_err;
            s.max_idx = static_cast<int>(i);
            s.ref_at_max = ref[i];
            s.got_at_max = got[i];
        }
    }
    s.mean_abs /= static_cast<float>(ref.size());
    return s;
}

bool HasNaN(const std::vector<float>& x) {
    for (float v : x) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseOptions(argc, argv);
    const int effective_seqlen_k = opts.seqused_k >= 0 ? opts.seqused_k : opts.seqlen_k;
    const int max_pages_per_seq = std::max(1, (opts.seqlen_k + opts.page_size - 1) / opts.page_size);
    const int num_pages = opts.batch * max_pages_per_seq;

    const std::vector<int64_t> q_dims = opts.varlen
        ? std::vector<int64_t>{opts.batch * opts.seqlen_q, opts.num_heads, opts.head_dim}
        : std::vector<int64_t>{opts.batch, opts.seqlen_q, opts.num_heads, opts.head_dim};
    const std::vector<int64_t> kv_dims = opts.paged_kv
        ? std::vector<int64_t>{num_pages, opts.page_size, opts.num_heads_k, opts.head_dim}
        : opts.varlen
        ? std::vector<int64_t>{opts.batch * opts.seqlen_k, opts.num_heads_k, opts.head_dim}
        : std::vector<int64_t>{opts.batch, opts.seqlen_k, opts.num_heads_k, opts.head_dim};
    const std::vector<int64_t> lse_dims = opts.varlen
        ? std::vector<int64_t>{opts.num_heads, opts.batch * opts.seqlen_q}
        : std::vector<int64_t>{opts.batch, opts.num_heads, opts.seqlen_q};
    const std::vector<int64_t> sched_dims = {opts.batch + 1};
    const std::vector<int64_t> cu_dims = {opts.batch + 1};
    const std::vector<int64_t> page_table_dims = {opts.batch, max_pages_per_seq};
    const std::vector<int64_t> seqused_dims = {opts.batch};

    const size_t q_elems = Product(q_dims);
    const size_t kv_elems = Product(kv_dims);
    const size_t out_elems = q_elems;
    const size_t lse_elems = Product(lse_dims);
    const size_t cu_elems = Product(cu_dims);
    const size_t sched_elems = Product(sched_dims);
    const size_t page_table_elems = Product(page_table_dims);
    const size_t seqused_elems = Product(seqused_dims);

    std::vector<uint16_t> h_q(q_elems);
    std::vector<uint16_t> h_k_logical(
        static_cast<size_t>(opts.batch) * opts.seqlen_k * opts.num_heads_k * opts.head_dim);
    std::vector<uint16_t> h_v_logical(
        static_cast<size_t>(opts.batch) * opts.seqlen_k * opts.num_heads_k * opts.head_dim);
    std::vector<uint16_t> h_k(kv_elems);
    std::vector<uint16_t> h_v(kv_elems);
    FillInput(&h_q, opts);
    FillInput(&h_k_logical, opts);
    FillInput(&h_v_logical, opts);
    if (opts.paged_kv) {
        FillPagedKv(&h_k, h_k_logical, opts, max_pages_per_seq);
        FillPagedKv(&h_v, h_v_logical, opts, max_pages_per_seq);
    } else {
        h_k = h_k_logical;
        h_v = h_v_logical;
    }

    std::vector<float> ref;
    if (!opts.skip_ref) {
        const std::vector<float> q_f = DecodeToFloat(h_q, opts.bf16);
        const std::vector<float> k_f = DecodeToFloat(h_k_logical, opts.bf16);
        const std::vector<float> v_f = DecodeToFloat(h_v_logical, opts.bf16);
        Options ref_opts = opts;
        ref_opts.seqlen_k = effective_seqlen_k;
        ref = CpuReference(q_f, k_f, v_f, ref_opts);
    }
    std::vector<int32_t> h_cu_q(cu_elems);
    std::vector<int32_t> h_cu_k(cu_elems);
    std::vector<int32_t> h_page_table(page_table_elems);
    std::vector<int32_t> h_seqused_k(seqused_elems, effective_seqlen_k);
    for (int i = 0; i <= opts.batch; ++i) {
        h_cu_q[i] = i * opts.seqlen_q;
        h_cu_k[i] = i * opts.seqlen_k;
    }
    for (int ib = 0; ib < opts.batch; ++ib) {
        for (int page = 0; page < max_pages_per_seq; ++page) {
            h_page_table[static_cast<size_t>(ib) * max_pages_per_seq + page] =
                ib * max_pages_per_seq + page;
        }
    }

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_out = nullptr;
    float* d_lse = nullptr;
    int32_t* d_cu_q = nullptr;
    int32_t* d_cu_k = nullptr;
    int32_t* d_page_table = nullptr;
    int32_t* d_seqused_k = nullptr;
    int* d_sched = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cu_q, cu_elems * sizeof(int32_t)));
    if (!opts.paged_kv) {
        CUDA_CHECK(cudaMalloc(&d_cu_k, cu_elems * sizeof(int32_t)));
    }
    if (opts.paged_kv) {
        CUDA_CHECK(cudaMalloc(&d_page_table, page_table_elems * sizeof(int32_t)));
    }
    if (opts.seqused_k >= 0) {
        CUDA_CHECK(cudaMalloc(&d_seqused_k, seqused_elems * sizeof(int32_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_sched, sched_elems * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cu_q, h_cu_q.data(), cu_elems * sizeof(int32_t), cudaMemcpyHostToDevice));
    if (!opts.paged_kv) {
        CUDA_CHECK(cudaMemcpy(d_cu_k, h_cu_k.data(), cu_elems * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    if (opts.paged_kv) {
        CUDA_CHECK(cudaMemcpy(d_page_table, h_page_table.data(), page_table_elems * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    if (opts.seqused_k >= 0) {
        CUDA_CHECK(cudaMemcpy(d_seqused_k, h_seqused_k.data(), seqused_elems * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sched, 0, sched_elems * sizeof(int)));

    const DataType dtype = opts.bf16 ? CAPI_BFLOAT16 : CAPI_FLOAT16;
    FlashattnTensor q = MakeTensor(d_q, dtype, q_dims);
    FlashattnTensor k = MakeTensor(d_k, dtype, kv_dims);
    FlashattnTensor v = MakeTensor(d_v, dtype, kv_dims);
    FlashattnTensor out = MakeTensor(d_out, dtype, q_dims);
    FlashattnTensor lse = MakeTensor(d_lse, CAPI_FLOAT, lse_dims);
    FlashattnTensor cu_q = MakeTensor(d_cu_q, CAPI_INT32, cu_dims);
    FlashattnTensor cu_k = MakeTensor(d_cu_k, CAPI_INT32, cu_dims);
    FlashattnTensor page_table = MakeTensor(d_page_table, CAPI_INT32, page_table_dims);
    FlashattnTensor seqused_k = MakeTensor(d_seqused_k, CAPI_INT32, seqused_dims);
    FlashattnTensor sched = MakeTensor(d_sched, CAPI_INT32, sched_dims);

    const FA3MhaFwdParams params{
        .max_seqlen_q = opts.seqlen_q,
        .max_seqlen_k = opts.seqlen_k,
        .softcap = 0.0f,
        .is_rotary_interleaved = false,
        .num_splits = opts.num_splits,
        .sm_margin = 0,
        .is_causal = opts.causal,
        .softmax_scale = 1.0f / std::sqrt(static_cast<float>(opts.head_dim)),
        .window_size_left = -1,
        .window_size_right = -1,
        .cp_world_size = 1,
        .cp_rank = 0,
    };

    std::vector<uint16_t> h_out(out_elems);
    std::vector<float> got(out_elems);
    std::vector<float> prev_run(out_elems, 0.0f);
    float max_repeat_delta = 0.0f;

    for (int iter = 0; iter < opts.iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sched, 0, sched_elems * sizeof(int)));
        fa3_mha_fwd(
            &q,
            &k,
            &v,
            &out,
            opts.varlen ? &cu_q : nullptr,
            !opts.paged_kv && opts.varlen ? &cu_k : nullptr,
            nullptr,
            opts.seqused_k >= 0 ? &seqused_k : nullptr,
            opts.paged_kv ? &page_table : nullptr,
            nullptr,
            nullptr,
            nullptr,
            &lse,
            nullptr,
            nullptr,
            &sched,
            nullptr,
            nullptr,
            &params,
            nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        got = DecodeToFloat(h_out, opts.bf16);
        if (iter > 0) {
            for (size_t i = 0; i < got.size(); ++i) {
                max_repeat_delta = std::max(max_repeat_delta, std::abs(got[i] - prev_run[i]));
            }
        }
        prev_run = got;
    }

    const Stats stats = opts.skip_ref ? Stats{} : Compare(ref, got);
    const bool has_nan = HasNaN(got);
    const bool compare_pass = opts.skip_ref ||
        stats.max_abs <= opts.atol + opts.rtol * std::abs(stats.ref_at_max);
    const bool pass = !has_nan && compare_pass && max_repeat_delta == 0.0f;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "config"
              << " batch=" << opts.batch
              << " seqlen_q=" << opts.seqlen_q
              << " seqlen_k=" << opts.seqlen_k
              << " num_heads=" << opts.num_heads
              << " num_heads_k=" << opts.num_heads_k
              << " head_dim=" << opts.head_dim
              << " causal=" << opts.causal
              << " dtype=" << (opts.bf16 ? "bf16" : "fp16")
              << " num_splits=" << opts.num_splits
              << " varlen=" << opts.varlen
              << " paged_kv=" << opts.paged_kv
              << " page_size=" << opts.page_size
              << " effective_seqlen_k=" << effective_seqlen_k
              << " skip_ref=" << opts.skip_ref
              << " iters=" << opts.iters
              << "\n";
    if (opts.skip_ref) {
        std::cout << "compare skipped\n";
    } else {
        std::cout << "compare"
                  << " max_abs=" << stats.max_abs
                  << " max_rel=" << stats.max_rel
                  << " mean_abs=" << stats.mean_abs
                  << " max_idx=" << stats.max_idx
                  << " ref=" << stats.ref_at_max
                  << " got=" << stats.got_at_max
                  << "\n";
    }
    std::cout << "repeatability max_delta=" << max_repeat_delta
              << " has_nan_or_inf=" << has_nan << "\n";
    std::cout << "sample";
    for (int i = 0; i < std::min<int>(8, got.size()); ++i) {
        std::cout << " " << got[i];
    }
    std::cout << "\n";

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_lse));
    CUDA_CHECK(cudaFree(d_cu_q));
    if (d_cu_k != nullptr) CUDA_CHECK(cudaFree(d_cu_k));
    if (d_page_table != nullptr) CUDA_CHECK(cudaFree(d_page_table));
    if (d_seqused_k != nullptr) CUDA_CHECK(cudaFree(d_seqused_k));
    CUDA_CHECK(cudaFree(d_sched));

    if (!pass) {
        std::cerr << "FAIL\n";
        return 2;
    }
    std::cout << "PASS\n";
    return 0;
}

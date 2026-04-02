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
        else if (ParseArg(arg, "atol", &value)) opts.atol = std::stof(value);
        else if (ParseArg(arg, "rtol", &value)) opts.rtol = std::stof(value);
        else if (arg == "--help") {
            std::cout
                << "Usage: bazel run //:fa3_sm90_repro -- "
                << "[--batch=N] [--seqlen_q=N] [--seqlen_k=N] [--num_heads=N] "
                << "[--num_heads_k=N] [--head_dim=N] [--causal=0|1] [--iters=N] "
                << "[--seed=N] [--bf16=0|1] [--num_splits=N] [--atol=F] [--rtol=F]\n";
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

    const std::vector<int64_t> q_dims = {
        opts.batch, opts.seqlen_q, opts.num_heads, opts.head_dim};
    const std::vector<int64_t> kv_dims = {
        opts.batch, opts.seqlen_k, opts.num_heads_k, opts.head_dim};
    const std::vector<int64_t> lse_dims = {
        opts.batch, opts.num_heads, opts.seqlen_q};
    const std::vector<int64_t> sched_dims = {1};

    const size_t q_elems = Product(q_dims);
    const size_t kv_elems = Product(kv_dims);
    const size_t out_elems = q_elems;
    const size_t lse_elems = Product(lse_dims);

    std::vector<uint16_t> h_q(q_elems);
    std::vector<uint16_t> h_k(kv_elems);
    std::vector<uint16_t> h_v(kv_elems);
    FillInput(&h_q, opts);
    FillInput(&h_k, opts);
    FillInput(&h_v, opts);

    const std::vector<float> q_f = DecodeToFloat(h_q, opts.bf16);
    const std::vector<float> k_f = DecodeToFloat(h_k, opts.bf16);
    const std::vector<float> v_f = DecodeToFloat(h_v, opts.bf16);
    const std::vector<float> ref = CpuReference(q_f, k_f, v_f, opts);

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_out = nullptr;
    float* d_lse = nullptr;
    int* d_sched = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_lse, lse_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sched, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d_lse, 0, lse_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sched, 0, sizeof(int)));

    const DataType dtype = opts.bf16 ? CAPI_BFLOAT16 : CAPI_FLOAT16;
    FlashattnTensor q = MakeTensor(d_q, dtype, q_dims);
    FlashattnTensor k = MakeTensor(d_k, dtype, kv_dims);
    FlashattnTensor v = MakeTensor(d_v, dtype, kv_dims);
    FlashattnTensor out = MakeTensor(d_out, dtype, q_dims);
    FlashattnTensor lse = MakeTensor(d_lse, CAPI_FLOAT, lse_dims);
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
        CUDA_CHECK(cudaMemset(d_sched, 0, sizeof(int)));
        fa3_mha_fwd(
            &q,
            &k,
            &v,
            &out,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
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

    const Stats stats = Compare(ref, got);
    const bool has_nan = HasNaN(got);
    const bool pass = !has_nan &&
        stats.max_abs <= opts.atol + opts.rtol * std::abs(stats.ref_at_max) &&
        max_repeat_delta == 0.0f;

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
              << " iters=" << opts.iters
              << "\n";
    std::cout << "compare"
              << " max_abs=" << stats.max_abs
              << " max_rel=" << stats.max_rel
              << " mean_abs=" << stats.mean_abs
              << " max_idx=" << stats.max_idx
              << " ref=" << stats.ref_at_max
              << " got=" << stats.got_at_max
              << "\n";
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
    CUDA_CHECK(cudaFree(d_sched));

    if (!pass) {
        std::cerr << "FAIL\n";
        return 2;
    }
    std::cout << "PASS\n";
    return 0;
}

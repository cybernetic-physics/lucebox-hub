#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];
};

// Must mirror LoraSet in kernel.cu exactly.
struct LoraSet {
    const __nv_bfloat16 *fa_q_A,    *fa_q_B;
    const __nv_bfloat16 *fa_k_A,    *fa_k_B;
    const __nv_bfloat16 *fa_v_A,    *fa_v_B;
    const __nv_bfloat16 *fa_o_A,    *fa_o_B;
    const __nv_bfloat16 *fa_gate_A, *fa_gate_B;
    const __nv_bfloat16 *fa_up_A,   *fa_up_B;
    const __nv_bfloat16 *fa_down_A, *fa_down_B;
    const __nv_bfloat16 *dn_qkv_A,  *dn_qkv_B;
    const __nv_bfloat16 *dn_z_A,    *dn_z_B;
    const __nv_bfloat16 *dn_out_A,  *dn_out_B;
    const __nv_bfloat16 *dn_gate_A, *dn_gate_B;
    const __nv_bfloat16 *dn_up_A,   *dn_up_B;
    const __nv_bfloat16 *dn_down_A, *dn_down_B;
};

extern "C" void launch_prefill_megakernel(
    const int *token_ids, int *out_token,
    const void *embed, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    LoraSet lora,
    int lora_rank, float lora_scaling, float *lora_h_ws,
    void *fa_k_cache, void *fa_v_cache,
    float *dn_states, float *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2,
    void *attn_buf, void *mlp_buf, void *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    void *final_normed,
    float *lm_bmv, int *lm_bmi,
    int S, int max_seq,
    cudaStream_t stream);

static const __nv_bfloat16 *bf16_ptr_or_null(const torch::Tensor &t) {
    if (!t.defined() || t.numel() == 0) return nullptr;
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, "LoRA tensor must be bf16");
    return (const __nv_bfloat16 *)t.data_ptr();
}

void train_mega_forward(
    torch::Tensor out_token, torch::Tensor token_ids,
    torch::Tensor embed, torch::Tensor layers_packed,
    torch::Tensor final_norm_w, torch::Tensor lm_head_w,
    // LoRA tensors as one flat list, all bf16 or empty.
    // Order: fa_{q,k,v,o,gate,up,down} × {A, B}, dn_{qkv,z,out,gate,up,down} × {A, B}.
    torch::TensorList lora_tensors,
    int64_t lora_rank, double lora_scaling, torch::Tensor lora_h_ws,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf, torch::Tensor dn_out_buf,
    torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    int64_t max_seq)
{
    int S = (int)token_ids.size(0);

    LoraSet lora{};
    TORCH_CHECK(lora_tensors.size() == 26,
        "Expected 26 LoRA tensors (7 FA + 6 DN projections × A, B). Got ", lora_tensors.size());
    #define SET(idx, name) lora.name = bf16_ptr_or_null(lora_tensors[idx])
    SET(0,  fa_q_A);    SET(1,  fa_q_B);
    SET(2,  fa_k_A);    SET(3,  fa_k_B);
    SET(4,  fa_v_A);    SET(5,  fa_v_B);
    SET(6,  fa_o_A);    SET(7,  fa_o_B);
    SET(8,  fa_gate_A); SET(9,  fa_gate_B);
    SET(10, fa_up_A);   SET(11, fa_up_B);
    SET(12, fa_down_A); SET(13, fa_down_B);
    SET(14, dn_qkv_A);  SET(15, dn_qkv_B);
    SET(16, dn_z_A);    SET(17, dn_z_B);
    SET(18, dn_out_A);  SET(19, dn_out_B);
    SET(20, dn_gate_A); SET(21, dn_gate_B);
    SET(22, dn_up_A);   SET(23, dn_up_B);
    SET(24, dn_down_A); SET(25, dn_down_B);
    #undef SET

    launch_prefill_megakernel(
        (const int *)token_ids.data_ptr(), (int *)out_token.data_ptr(),
        embed.data_ptr(),
        reinterpret_cast<const LayerWeights *>(layers_packed.data_ptr()),
        final_norm_w.data_ptr(), lm_head_w.data_ptr(),
        lora,
        (int)lora_rank, (float)lora_scaling, (float *)lora_h_ws.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        (float *)dn_states.data_ptr(), (float *)conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(), dn_out_buf.data_ptr(),
        (float *)beta_buf.data_ptr(), (float *)alpha_buf.data_ptr(),
        final_normed.data_ptr(),
        (float *)lm_bmv.data_ptr(), (int *)lm_bmi.data_ptr(),
        S, (int)max_seq,
        c10::cuda::getCurrentCUDAStream().stream());
}

// ---------------------------------------------------------------------------
// Fused multi-param AdamW — single kernel dispatch updates every LoRA param.
// ---------------------------------------------------------------------------
extern "C" void launch_fused_adamw(
    void *params_bf16, float *m, float *v, const float *grad,
    long long numel, int step,
    float lr, float beta1, float beta2, float eps, float wd,
    cudaStream_t stream);

extern "C" void launch_bwd_ce_lm_head(
    const void *final_normed, const void *lm_head_w,
    int target_token,
    float *grad_final_normed_out, float *loss_out,
    cudaStream_t stream);

extern "C" void launch_bwd_rmsnorm(
    const void *x, const void *w, const float *dy, float *dx,
    int S, int H, float eps, cudaStream_t stream);

extern "C" void launch_bwd_swiglu(
    const void *gate, const void *up, const float *dy,
    float *dgate, float *dup, int N, cudaStream_t stream);

extern "C" void launch_bwd_lora_linear(
    const void *x, const void *A, const void *B,
    const float *grad_y,
    float *grad_x, float *grad_A, float *grad_B,
    float *workspace_lora_h, float *workspace_grad_lora_h,
    int S, int K_in, int K_out, int R, float scaling,
    cudaStream_t stream);

extern "C" cudaError_t launch_dn_fwd_with_delta_save(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *decay, const float *state_in,
    __nv_bfloat16 *y, float *state_out, __nv_bfloat16 *delta_save,
    float *state_history,
    int S, int H, cudaStream_t stream);

extern "C" cudaError_t launch_dn_bwd(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *decay,
    const float *state_in,
    const __nv_bfloat16 *delta_save,
    const __nv_bfloat16 *dy,
    const float *state_history,
    __nv_bfloat16 *dq, __nv_bfloat16 *dk, __nv_bfloat16 *dv,
    float *dbeta, float *ddecay, float *dstate_init,
    int S, int H, cudaStream_t stream);

extern "C" cudaError_t launch_dn_chunked_fwd(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *g, const float *state_in,
    __nv_bfloat16 *y, float *state_out,
    int S, int H, cudaStream_t stream);

void fused_adamw_step(
    torch::Tensor params,
    torch::Tensor m, torch::Tensor v,
    torch::Tensor grad,
    int64_t step,
    double lr, double beta1, double beta2, double eps, double wd)
{
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(m.scalar_type() == torch::kFloat32);
    TORCH_CHECK(v.scalar_type() == torch::kFloat32);
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32);
    TORCH_CHECK(params.numel() == m.numel() && m.numel() == v.numel()
                && v.numel() == grad.numel());
    launch_fused_adamw(
        params.data_ptr(), (float *)m.data_ptr(), (float *)v.data_ptr(),
        (const float *)grad.data_ptr(),
        (long long)params.numel(), (int)step,
        (float)lr, (float)beta1, (float)beta2, (float)eps, (float)wd,
        c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, ops) {
    ops.def("train_mega_forward(Tensor out_token, Tensor token_ids, "
            "Tensor embed, Tensor layers_packed, "
            "Tensor final_norm_w, Tensor lm_head_w, "
            "Tensor[] lora_tensors, "
            "int lora_rank, float lora_scaling, Tensor lora_h_ws, "
            "Tensor fa_k_cache, Tensor fa_v_cache, "
            "Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, "
            "Tensor attn_buf, Tensor mlp_buf, Tensor dn_out_buf, "
            "Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, "
            "Tensor lm_bmv, Tensor lm_bmi, "
            "int max_seq) -> ()");
    ops.impl("train_mega_forward", torch::kCUDA, &train_mega_forward);

    ops.def("fused_adamw_step(Tensor(a!) params, Tensor(b!) m, Tensor(c!) v, "
            "Tensor grad, int step, float lr, float beta1, float beta2, "
            "float eps, float wd) -> ()");
    ops.impl("fused_adamw_step", torch::kCUDA, &fused_adamw_step);

    ops.def("bwd_ce_lm_head(Tensor final_normed, Tensor lm_head_w, "
            "int target_token, Tensor(a!) grad_final_normed, Tensor(b!) loss) -> ()");
    ops.impl("bwd_ce_lm_head", torch::kCUDA, +[](
        torch::Tensor final_normed, torch::Tensor lm_head_w,
        int64_t target_token,
        torch::Tensor grad_final_normed, torch::Tensor loss) {
            TORCH_CHECK(final_normed.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(lm_head_w.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(grad_final_normed.scalar_type() == torch::kFloat32);
            TORCH_CHECK(loss.scalar_type() == torch::kFloat32);
            launch_bwd_ce_lm_head(
                final_normed.data_ptr(), lm_head_w.data_ptr(),
                (int)target_token,
                (float *)grad_final_normed.data_ptr(),
                (float *)loss.data_ptr(),
                c10::cuda::getCurrentCUDAStream().stream());
        });

    ops.def("bwd_rmsnorm(Tensor x, Tensor w, Tensor dy, Tensor(a!) dx, "
            "int S, int H, float eps) -> ()");
    ops.impl("bwd_rmsnorm", torch::kCUDA, +[](
        torch::Tensor x, torch::Tensor w, torch::Tensor dy, torch::Tensor dx,
        int64_t S, int64_t H, double eps) {
            TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(w.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(dy.scalar_type() == torch::kFloat32);
            TORCH_CHECK(dx.scalar_type() == torch::kFloat32);
            launch_bwd_rmsnorm(
                x.data_ptr(), w.data_ptr(),
                (const float *)dy.data_ptr(), (float *)dx.data_ptr(),
                (int)S, (int)H, (float)eps,
                c10::cuda::getCurrentCUDAStream().stream());
        });

    ops.def("bwd_swiglu(Tensor gate, Tensor up, Tensor dy, "
            "Tensor(a!) dgate, Tensor(b!) dup, int N) -> ()");
    ops.impl("bwd_swiglu", torch::kCUDA, +[](
        torch::Tensor gate, torch::Tensor up, torch::Tensor dy,
        torch::Tensor dgate, torch::Tensor dup, int64_t N) {
            TORCH_CHECK(gate.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(up.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(dy.scalar_type() == torch::kFloat32);
            TORCH_CHECK(dgate.scalar_type() == torch::kFloat32);
            TORCH_CHECK(dup.scalar_type() == torch::kFloat32);
            launch_bwd_swiglu(
                gate.data_ptr(), up.data_ptr(),
                (const float *)dy.data_ptr(),
                (float *)dgate.data_ptr(), (float *)dup.data_ptr(),
                (int)N,
                c10::cuda::getCurrentCUDAStream().stream());
        });

    ops.def("dn_fwd_save(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor decay, "
            "Tensor state_in, Tensor(a!) y, Tensor(b!) state_out, "
            "Tensor(c!) delta_save, Tensor(d!) state_history) -> ()");
    ops.impl("dn_fwd_save", torch::kCUDA, +[](
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor beta, torch::Tensor decay,
        torch::Tensor state_in,
        torch::Tensor y, torch::Tensor state_out, torch::Tensor delta_save,
        torch::Tensor state_history) {
            int S = (int)q.size(0);
            int H = (int)q.size(1);
            cudaError_t err = launch_dn_fwd_with_delta_save(
                (const __nv_bfloat16 *)q.data_ptr(),
                (const __nv_bfloat16 *)k.data_ptr(),
                (const __nv_bfloat16 *)v.data_ptr(),
                (const float *)beta.data_ptr(),
                (const float *)decay.data_ptr(),
                (const float *)state_in.data_ptr(),
                (__nv_bfloat16 *)y.data_ptr(),
                state_out.numel() > 0 ? (float *)state_out.data_ptr() : nullptr,
                (__nv_bfloat16 *)delta_save.data_ptr(),
                state_history.numel() > 0 ? (float *)state_history.data_ptr() : nullptr,
                S, H, c10::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(err == cudaSuccess, "dn_fwd_save: ",
                        cudaGetErrorString(err));
        });

    ops.def("dn_bwd(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor decay, "
            "Tensor state_in, Tensor delta_save, Tensor dy, Tensor state_history, "
            "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, "
            "Tensor(d!) dbeta, Tensor(e!) ddecay, Tensor(f!) dstate_init) -> ()");
    ops.impl("dn_bwd", torch::kCUDA, +[](
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor beta, torch::Tensor decay,
        torch::Tensor state_in, torch::Tensor delta_save, torch::Tensor dy,
        torch::Tensor state_history,
        torch::Tensor dq, torch::Tensor dk, torch::Tensor dv,
        torch::Tensor dbeta, torch::Tensor ddecay, torch::Tensor dstate_init) {
            int S = (int)q.size(0);
            int H = (int)q.size(1);
            cudaError_t err = launch_dn_bwd(
                (const __nv_bfloat16 *)q.data_ptr(),
                (const __nv_bfloat16 *)k.data_ptr(),
                (const __nv_bfloat16 *)v.data_ptr(),
                (const float *)beta.data_ptr(),
                (const float *)decay.data_ptr(),
                (const float *)state_in.data_ptr(),
                (const __nv_bfloat16 *)delta_save.data_ptr(),
                (const __nv_bfloat16 *)dy.data_ptr(),
                (const float *)state_history.data_ptr(),
                (__nv_bfloat16 *)dq.data_ptr(),
                (__nv_bfloat16 *)dk.data_ptr(),
                (__nv_bfloat16 *)dv.data_ptr(),
                (float *)dbeta.data_ptr(),
                (float *)ddecay.data_ptr(),
                dstate_init.numel() > 0 ? (float *)dstate_init.data_ptr() : nullptr,
                S, H, c10::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(err == cudaSuccess, "dn_bwd: ",
                        cudaGetErrorString(err));
        });

    ops.def("dn_chunked_fwd(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor g, "
            "Tensor state_in, Tensor(a!) y, Tensor(b!) state_out) -> ()");
    ops.impl("dn_chunked_fwd", torch::kCUDA, +[](
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor beta, torch::Tensor g,
        torch::Tensor state_in,
        torch::Tensor y, torch::Tensor state_out) {
            int S = (int)q.size(0);
            int H = (int)q.size(1);
            cudaError_t err = launch_dn_chunked_fwd(
                (const __nv_bfloat16 *)q.data_ptr(),
                (const __nv_bfloat16 *)k.data_ptr(),
                (const __nv_bfloat16 *)v.data_ptr(),
                (const float *)beta.data_ptr(),
                (const float *)g.data_ptr(),
                (const float *)state_in.data_ptr(),
                (__nv_bfloat16 *)y.data_ptr(),
                state_out.numel() > 0 ? (float *)state_out.data_ptr() : nullptr,
                S, H, c10::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(err == cudaSuccess, "dn_chunked_fwd: ",
                        cudaGetErrorString(err));
        });

    ops.def("bwd_lora_linear(Tensor x, Tensor A, Tensor B, Tensor grad_y, "
            "Tensor(a!) grad_x, Tensor(b!) grad_A, Tensor(c!) grad_B, "
            "Tensor(d!) ws_lora_h, Tensor(e!) ws_grad_lora_h, "
            "int S, int K_in, int K_out, int R, float scaling) -> ()");
    ops.impl("bwd_lora_linear", torch::kCUDA, +[](
        torch::Tensor x, torch::Tensor A, torch::Tensor B,
        torch::Tensor grad_y,
        torch::Tensor grad_x, torch::Tensor grad_A, torch::Tensor grad_B,
        torch::Tensor ws_lora_h, torch::Tensor ws_grad_lora_h,
        int64_t S, int64_t K_in, int64_t K_out, int64_t R, double scaling) {
            TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(A.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(B.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32);
            TORCH_CHECK(grad_x.scalar_type() == torch::kFloat32);
            TORCH_CHECK(grad_A.scalar_type() == torch::kFloat32);
            TORCH_CHECK(grad_B.scalar_type() == torch::kFloat32);
            launch_bwd_lora_linear(
                x.data_ptr(), A.data_ptr(), B.data_ptr(),
                (const float *)grad_y.data_ptr(),
                (float *)grad_x.data_ptr(), (float *)grad_A.data_ptr(),
                (float *)grad_B.data_ptr(),
                (float *)ws_lora_h.data_ptr(), (float *)ws_grad_lora_h.data_ptr(),
                (int)S, (int)K_in, (int)K_out, (int)R, (float)scaling,
                c10::cuda::getCurrentCUDAStream().stream());
        });
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

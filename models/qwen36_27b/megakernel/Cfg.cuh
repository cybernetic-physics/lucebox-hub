/**
 * Model-config tag structs for the Qwen3.x hybrid megakernel.
 *
 * The 0.8B and 27B variants share the same architecture family (Hybrid
 * Gated DeltaNet + Gated Attention, head_dim=256, RoPE 64/1e7, vocab
 * 248320). Only seven dimensions actually differ — captured here as
 * compile-time `static constexpr` members of a tag struct so nvcc can
 * fully specialize each kernel per model and keep the inner loops
 * branch-free.
 *
 * To add a new Qwen3.x variant: write a new tag struct here, write a
 * specialization of the templated decode_kernel<Cfg> in `kernel_decode.cu`,
 * and register a corresponding torch op.
 */
#pragma once

#include <stdint.h>

namespace lucebox::qwen3x {

// ---------------------------------------------------------------------------
// Family invariants — same for every Qwen3.x model we support.
// ---------------------------------------------------------------------------
struct FamilyInvariants {
    static constexpr int   FA_HEAD_DIM      = 256;
    static constexpr int   DN_KEY_DIM       = 128;
    static constexpr int   DN_VALUE_DIM     = 128;
    static constexpr int   DN_CONV_KERNEL   = 4;
    static constexpr int   FA_ROTARY_DIM    = 64;
    static constexpr float FA_ROPE_THETA    = 1.0e7f;
    static constexpr int   VOCAB_SIZE       = 248320;
    // The hybrid pattern is 3 DN + 1 FA, repeating. Layer i is FA iff
    // ((i + 1) % 4) == 0.
    static constexpr bool  is_fa_layer(int i) { return ((i + 1) % 4) == 0; }
};

// ---------------------------------------------------------------------------
// Qwen3.5-0.8B   — the existing megakernel target.
// ---------------------------------------------------------------------------
struct Cfg_0p8B : FamilyInvariants {
    static constexpr const char *name      = "Qwen3.5-0.8B";
    static constexpr int   NUM_LAYERS      = 24;
    static constexpr int   HIDDEN          = 1024;
    static constexpr int   INTERMEDIATE    = 3584;
    static constexpr int   FA_NUM_Q_HEADS  = 8;
    static constexpr int   FA_NUM_KV_HEADS = 2;
    static constexpr int   DN_NUM_V_HEADS  = 16;
    static constexpr int   DN_NUM_QK_HEADS = 16;   // unified V/QK
    static constexpr bool  USE_YARN        = false;
    static constexpr int   MAX_CONTEXT     = 65536;

    // Derived.
    static constexpr int FA_GQA_RATIO   = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
    static constexpr int FA_Q_SIZE      = FA_NUM_Q_HEADS  * FA_HEAD_DIM;
    static constexpr int FA_KV_SIZE     = FA_NUM_KV_HEADS * FA_HEAD_DIM;
    static constexpr int FA_QPROJ_SIZE  = 2 * FA_Q_SIZE;        // q + gate
    static constexpr int DN_QK_SIZE     = DN_NUM_QK_HEADS * DN_KEY_DIM;
    static constexpr int DN_V_SIZE      = DN_NUM_V_HEADS  * DN_VALUE_DIM;
    static constexpr int DN_CONV_CH     = DN_QK_SIZE * 2 + DN_V_SIZE;
    // GQA across the DN V/QK split: how many V heads each QK head serves.
    static constexpr int DN_V_PER_QK    = DN_NUM_V_HEADS / DN_NUM_QK_HEADS;  // = 1
};

// ---------------------------------------------------------------------------
// Qwen3.6-27B    — target of the Phase 1+ port.
// ---------------------------------------------------------------------------
struct Cfg_27B : FamilyInvariants {
    static constexpr const char *name      = "Qwen3.6-27B";
    static constexpr int   NUM_LAYERS      = 64;
    static constexpr int   HIDDEN          = 5120;
    static constexpr int   INTERMEDIATE    = 17408;
    static constexpr int   FA_NUM_Q_HEADS  = 24;
    static constexpr int   FA_NUM_KV_HEADS = 4;
    static constexpr int   DN_NUM_V_HEADS  = 48;
    static constexpr int   DN_NUM_QK_HEADS = 16;   // split V/QK -- new code needed
    static constexpr bool  USE_YARN        = true;
    static constexpr int   MAX_CONTEXT     = 262144;

    static constexpr int FA_GQA_RATIO   = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
    static constexpr int FA_Q_SIZE      = FA_NUM_Q_HEADS  * FA_HEAD_DIM;
    static constexpr int FA_KV_SIZE     = FA_NUM_KV_HEADS * FA_HEAD_DIM;
    static constexpr int FA_QPROJ_SIZE  = 2 * FA_Q_SIZE;
    static constexpr int DN_QK_SIZE     = DN_NUM_QK_HEADS * DN_KEY_DIM;
    static constexpr int DN_V_SIZE      = DN_NUM_V_HEADS  * DN_VALUE_DIM;
    static constexpr int DN_CONV_CH     = DN_QK_SIZE * 2 + DN_V_SIZE;
    static constexpr int DN_V_PER_QK    = DN_NUM_V_HEADS / DN_NUM_QK_HEADS;  // = 3
};

// ---------------------------------------------------------------------------
// Sanity checks at compile time.
// ---------------------------------------------------------------------------
static_assert(Cfg_0p8B::FA_NUM_Q_HEADS % Cfg_0p8B::FA_NUM_KV_HEADS == 0,
              "FA GQA must divide cleanly (0.8B)");
static_assert(Cfg_27B::FA_NUM_Q_HEADS  % Cfg_27B::FA_NUM_KV_HEADS  == 0,
              "FA GQA must divide cleanly (27B)");
static_assert(Cfg_0p8B::DN_NUM_V_HEADS % Cfg_0p8B::DN_NUM_QK_HEADS == 0,
              "DN V/QK must divide cleanly (0.8B)");
static_assert(Cfg_27B::DN_NUM_V_HEADS  % Cfg_27B::DN_NUM_QK_HEADS  == 0,
              "DN V/QK must divide cleanly (27B)");
static_assert(Cfg_0p8B::FA_HEAD_DIM == Cfg_27B::FA_HEAD_DIM,
              "FA head_dim must be 256 across the Qwen3.x family");
static_assert(Cfg_0p8B::DN_KEY_DIM == Cfg_27B::DN_KEY_DIM,
              "DN key_dim must be 128 across the Qwen3.x family");
static_assert(Cfg_0p8B::VOCAB_SIZE == Cfg_27B::VOCAB_SIZE,
              "vocab must be 248320 across the Qwen3.x family");

}  // namespace lucebox::qwen3x

/**
 * Tree-verify primitives for MTP-style speculative decoding.
 *
 * DDTree (Ringel & Romano 2026) extends chain-style speculative decoding
 * with a TREE of candidate continuations: the draft proposes N branches
 * of length L, and the target verifies them with ONE forward pass over
 * all N*L nodes (using a tree-aware attention mask).
 *
 * Data layout:
 *
 *   tree_descriptor[N_NODES] = { int32 parent_idx; int32 depth; int32 token_id; int32 _pad; }
 *
 *   - parent_idx: -1 for the root (= last committed token before this
 *     speculation round). Other nodes: index in [0, N_NODES) of their
 *     parent in the tree.
 *   - depth: 0 for the root, parent.depth + 1 otherwise.
 *   - token_id: the candidate token at this node.
 *
 * Tree-aware FA attention (per query head, per node):
 *   For each k_t at node t, find the path from root -> t (length depth+1).
 *   The query at node t attends only to k_v for v in that path (causal
 *   within the tree). Online softmax over the path length, exactly the
 *   same arithmetic as linear-causal — only the indexing changes.
 *
 * Tree-aware DN state forking:
 *   The DN recurrence is stateful — branching forks the state. Each leaf's
 *   state is reachable by replaying the recurrence along the path. For
 *   the verify forward, we therefore process nodes in DEPTH-FIRST order
 *   so the recurrent state at each node is just (parent.state after
 *   applying parent's input). State storage:
 *       dn_states_tree[N_NODES, N_DN_LAYERS, V_HEADS, VAL, KEY]
 *   For 27B at N_NODES=22, that's 22 * 48 * 48 * 128 * 128 * 4 bytes
 *   = ~3.2 GB — too much. Mitigation: only store states for ACTIVE
 *   leaves (~the budget=22 nodes the tree-verify exercises), reusing
 *   memory once the tree is rolled back.
 *
 * This header defines the tree descriptor + the attention-mask
 * primitive. The persistent megakernel that drives a tree-verify
 * forward is `prefill_megakernel_tree<Cfg>` (TODO -- next session).
 */
#pragma once

#include "Cfg.cuh"
#include "helpers.cuh"

namespace lucebox::qwen3x {

// Per-node tree descriptor, packed into a uint4 for coalesced load.
struct TreeNode {
    int32_t parent_idx;   // -1 for root, else index in [0, N_NODES)
    int32_t depth;
    int32_t token_id;
    int32_t _pad;
};
static_assert(sizeof(TreeNode) == 16, "TreeNode must be 16 bytes for uint4 load");

// Maximum supported tree size — matches the DDTree budget hyperparameter.
constexpr int TREE_BUDGET_MAX = 64;   // production target ~22; cap at 64

// Build a parent-chain iterator at compile-time bounded depth. Given a
// node idx and the tree descriptor, walk ancestors and call `visit(t)`
// for each ancestor t in [root, node-1]. The visit callable returns
// true to continue, false to terminate early.
template<typename Visit>
__device__ __forceinline__ void walk_ancestors(
    const TreeNode *__restrict__ tree, int node_idx, Visit&& visit)
{
    // Bounded loop to avoid infinite traversal on malformed trees.
    #pragma unroll 1
    for (int hop = 0; hop < TREE_BUDGET_MAX; ++hop) {
        int p = tree[node_idx].parent_idx;
        if (p < 0) return;
        if (!visit(p)) return;
        node_idx = p;
    }
}

// Tree-aware FA scoring for one (qh, leaf_node) pair. Walks ancestors,
// computes softmax over their K vectors. Same online-softmax math as
// the linear-causal attention scan in fa_layer.cuh.
//
// k_cache layout: [N_NODES, KV_HEADS, HEAD_DIM] bf16 (NOT [KV_HEADS,
// MAX_SEQ, HEAD]). Each tree node gets a slot.
template<typename Cfg, typename QLocalT>
__device__ float tree_attention_score(
    const TreeNode *tree, int leaf_idx, int kvh,
    const QLocalT *q_local, const __nv_bfloat16 *k_cache,
    float attn_scale, int lane_id)
{
    constexpr int D = Cfg::FA_HEAD_DIM;
    constexpr int EPL = D / WARP_SIZE;
    float partial_max = -INFINITY, partial_sum = 0.0f;

    // Include the leaf itself in the path (Q attends to leaf's own K).
    auto score_node = [&](int t) -> float {
        const __nv_bfloat16 *kp = k_cache + (size_t)t * Cfg::FA_NUM_KV_HEADS * D
                                   + (size_t)kvh * D;
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < EPL; ++e)
            s += q_local[e] * __bfloat162float(__ldg(kp + lane_id * EPL + e));
        s = warp_reduce_sum_x(s) * attn_scale;
        return __shfl_sync(0xffffffff, s, 0);
    };

    {
        float s = score_node(leaf_idx);
        partial_max = s;
        partial_sum = 1.0f;
    }
    walk_ancestors(tree, leaf_idx, [&](int t) -> bool {
        float s = score_node(t);
        float old_max = partial_max;
        partial_max = fmaxf(partial_max, s);
        float exp_diff = fast_exp(old_max - partial_max);
        partial_sum = partial_sum * exp_diff + fast_exp(s - partial_max);
        return true;
    });
    return partial_max + logf(partial_sum);  // LSE
}

// ---------------------------------------------------------------------------
// Host-side helper: validate tree topology (called from the torch op).
// ---------------------------------------------------------------------------
__host__ bool tree_descriptor_is_valid(const TreeNode *tree, int n_nodes) {
    // Root must be at index 0 with parent_idx = -1, depth = 0.
    if (n_nodes <= 0 || tree[0].parent_idx != -1 || tree[0].depth != 0) return false;
    for (int i = 1; i < n_nodes; ++i) {
        int p = tree[i].parent_idx;
        if (p < 0 || p >= i) return false;  // parent must come before child
        if (tree[i].depth != tree[p].depth + 1) return false;
    }
    return true;
}

}  // namespace lucebox::qwen3x

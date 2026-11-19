// CzechSpeech language model: GPT-2 inference on ggml for variant selection (see czechspeech_lm.h).
//
// Written for CzechSpeech; the graph follows ggml's GPT-2 example (examples/gpt-2 in github.com/ggml-org/ggml, MIT
// licence, Copyright (c) 2023-2026 The ggml authors; see THIRD_PARTY_NOTICES). Tensor names and metadata keys are
// llama.cpp's for the gpt2 architecture, as written by scripts/lm/convert_gpt2.py.
//
// Privacy: token ids come in and sums of log-probabilities go out; nothing is logged but the reason a file cannot be
// loaded. No file or network I/O except reading the model file.

#include "czechspeech_lm.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef __ANDROID__
#include <android/log.h>
#define CS_LM_WARN(msg) __android_log_write(ANDROID_LOG_WARN, "CzechSpeech", (msg))
#else
#define CS_LM_WARN(msg) fprintf(stderr, "CzechSpeech: %s\n", (msg))
#endif

// Tokens evaluated in one graph: bounds the compute buffer (the logits of a chunk are n_vocab x 32 floats, 6.4 MB).
#define CS_LM_CHUNK 32
#define CS_LM_GRAPH_SIZE 2048
// Most layers one chunk's graph holds: cs_lm_graph makes 42 tensors per layer and 11 more, CS_LM_GRAPH_SIZE in all.
// A file with more is refused instead of overflowing the graph (a ggml assertion, which aborts the process).
#define CS_LM_MAX_LAYERS 48

struct cs_lm_layer {
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;
    struct ggml_tensor * qkv_w;
    struct ggml_tensor * qkv_b;
    struct ggml_tensor * out_w;
    struct ggml_tensor * out_b;
    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;
    struct ggml_tensor * up_w;
    struct ggml_tensor * up_b;
    struct ggml_tensor * down_w;
    struct ggml_tensor * down_b;
};

struct cs_lm {
    int n_vocab;
    int n_ctx;
    int n_embd;
    int n_head;
    int n_layer;
    float eps;

    struct ggml_context * weights;      // the file's tensors
    ggml_backend_buffer_t weights_buf;  // their data
    struct ggml_tensor * wte;           // token embedding, also the output matrix (tied)
    struct ggml_tensor * wpe;
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;
    struct cs_lm_layer * layers;

    ggml_backend_t backend;
    ggml_gallocr_t galloc;              // keeps its compute buffer, sized for the largest chunk so far
    void * meta;                        // tensor and graph metadata of one chunk's graph
    size_t meta_size;

    // key/value cache of the sequences being scored, n_kv positions per layer (only during cs_lm_score)
    struct ggml_context * kv_ctx;
    ggml_backend_buffer_t kv_buf;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    int n_kv;
};

static bool cs_lm_u32(const struct gguf_context * g, const char * key, int * out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        return false;
    }
    *out = (int) gguf_get_val_u32(g, id);
    return *out > 0;
}

// The tensor [name] with the given shape (ne1 = 1 for a vector) and a type the graph handles, or NULL: f32, f16 or
// q8_0 for a [matrix] (the operand of a matrix product or of get_rows), f32 for everything else (norms, biases, the
// position embedding), which goes into element-wise operations with f32 activations.
static struct ggml_tensor * cs_lm_tensor(struct ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1,
                                         bool matrix) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == NULL || t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != 1 || t->ne[3] != 1) {
        return NULL;
    }
    const bool type_ok = t->type == GGML_TYPE_F32 ||
                         (matrix && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0));
    return type_ok ? t : NULL;
}

static struct ggml_tensor * cs_lm_block_tensor(struct ggml_context * ctx, int il, const char * what, int64_t ne0,
                                               int64_t ne1, bool matrix) {
    char name[64];
    snprintf(name, sizeof(name), "blk.%d.%s", il, what);
    return cs_lm_tensor(ctx, name, ne0, ne1, matrix);
}

static void cs_lm_kv_free(struct cs_lm * lm) {
    if (lm->kv_buf != NULL) {
        ggml_backend_buffer_free(lm->kv_buf);
    }
    if (lm->kv_ctx != NULL) {
        ggml_free(lm->kv_ctx);
    }
    lm->kv_buf = NULL;
    lm->kv_ctx = NULL;
    lm->k = lm->v = NULL;
    lm->n_kv = 0;
}

void cs_lm_free(struct cs_lm * lm) {
    if (lm == NULL) {
        return;
    }
    cs_lm_kv_free(lm);
    if (lm->galloc != NULL) {
        ggml_gallocr_free(lm->galloc);
    }
    if (lm->weights_buf != NULL) {
        ggml_backend_buffer_free(lm->weights_buf);
    }
    if (lm->weights != NULL) {
        ggml_free(lm->weights);
    }
    if (lm->backend != NULL) {
        ggml_backend_free(lm->backend);
    }
    free(lm->layers);
    free(lm->meta);
    free(lm);
}

// Reads every tensor's data from the file into its (host) buffer; false when the file ends early.
static bool cs_lm_read_data(const char * path, const struct gguf_context * g, struct ggml_context * ctx) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return false;
    }
    const size_t base = gguf_get_data_offset(g);
    bool ok = true;
    for (int64_t i = 0; ok && i < gguf_get_n_tensors(g); i++) {
        struct ggml_tensor * t = ggml_get_tensor(ctx, gguf_get_tensor_name(g, i));
        const size_t n = t != NULL ? ggml_nbytes(t) : 0;
        ok = t != NULL && fseek(f, (long) (base + gguf_get_tensor_offset(g, i)), SEEK_SET) == 0 &&
             fread(t->data, 1, n, f) == n;
    }
    fclose(f);
    return ok;
}

// Checks the hyperparameters and every tensor the graph uses (name, shape, type); NULL when all is right.
static const char * cs_lm_check(struct cs_lm * lm, const struct gguf_context * g) {
    const int64_t arch = gguf_find_key(g, "general.architecture");
    const int64_t vocab = gguf_find_key(g, "tokenizer.ggml.tokens");
    const int64_t eps = gguf_find_key(g, "gpt2.attention.layer_norm_epsilon");
    int n_ff = 0;
    if (arch < 0 || gguf_get_kv_type(g, arch) != GGUF_TYPE_STRING || strcmp(gguf_get_val_str(g, arch), "gpt2") != 0) {
        return "language model: not a gpt2 model";
    }
    if (!cs_lm_u32(g, "gpt2.context_length", &lm->n_ctx) || !cs_lm_u32(g, "gpt2.embedding_length", &lm->n_embd) ||
        !cs_lm_u32(g, "gpt2.block_count", &lm->n_layer) || !cs_lm_u32(g, "gpt2.attention.head_count", &lm->n_head) ||
        !cs_lm_u32(g, "gpt2.feed_forward_length", &n_ff) || eps < 0 ||
        gguf_get_kv_type(g, eps) != GGUF_TYPE_FLOAT32 || lm->n_embd % lm->n_head != 0) {
        return "language model: missing or invalid hyperparameters";
    }
    if (lm->n_layer > CS_LM_MAX_LAYERS) {
        return "language model: more layers than one graph holds";
    }
    if (vocab < 0 || gguf_get_kv_type(g, vocab) != GGUF_TYPE_ARRAY || gguf_get_arr_n(g, vocab) == 0 ||
        gguf_get_arr_n(g, vocab) > (size_t) INT32_MAX) {
        return "language model: no vocabulary";
    }
    lm->eps = gguf_get_val_f32(g, eps);
    lm->n_vocab = (int) gguf_get_arr_n(g, vocab);
    const int64_t n_embd = lm->n_embd;   // 64 bits: 3 * n_embd must not overflow for an absurd embedding size
    struct ggml_context * w = lm->weights;
    lm->wte = cs_lm_tensor(w, "token_embd.weight", n_embd, lm->n_vocab, true);
    lm->wpe = cs_lm_tensor(w, "position_embd.weight", n_embd, lm->n_ctx, false);
    lm->ln_f_g = cs_lm_tensor(w, "output_norm.weight", n_embd, 1, false);
    lm->ln_f_b = cs_lm_tensor(w, "output_norm.bias", n_embd, 1, false);
    lm->layers = (struct cs_lm_layer *) calloc((size_t) lm->n_layer, sizeof(struct cs_lm_layer));
    if (lm->layers == NULL) {
        return "language model: out of memory";
    }
    bool ok = lm->wte && lm->wpe && lm->ln_f_g && lm->ln_f_b;
    for (int il = 0; ok && il < lm->n_layer; il++) {
        struct cs_lm_layer * L = &lm->layers[il];
        L->ln_1_g = cs_lm_block_tensor(w, il, "attn_norm.weight", n_embd, 1, false);
        L->ln_1_b = cs_lm_block_tensor(w, il, "attn_norm.bias", n_embd, 1, false);
        L->qkv_w  = cs_lm_block_tensor(w, il, "attn_qkv.weight", n_embd, 3 * n_embd, true);
        L->qkv_b  = cs_lm_block_tensor(w, il, "attn_qkv.bias", 3 * n_embd, 1, false);
        L->out_w  = cs_lm_block_tensor(w, il, "attn_output.weight", n_embd, n_embd, true);
        L->out_b  = cs_lm_block_tensor(w, il, "attn_output.bias", n_embd, 1, false);
        L->ln_2_g = cs_lm_block_tensor(w, il, "ffn_norm.weight", n_embd, 1, false);
        L->ln_2_b = cs_lm_block_tensor(w, il, "ffn_norm.bias", n_embd, 1, false);
        L->up_w   = cs_lm_block_tensor(w, il, "ffn_up.weight", n_embd, n_ff, true);
        L->up_b   = cs_lm_block_tensor(w, il, "ffn_up.bias", n_ff, 1, false);
        L->down_w = cs_lm_block_tensor(w, il, "ffn_down.weight", n_ff, n_embd, true);
        L->down_b = cs_lm_block_tensor(w, il, "ffn_down.bias", n_embd, 1, false);
        ok = L->ln_1_g && L->ln_1_b && L->qkv_w && L->qkv_b && L->out_w && L->out_b && L->ln_2_g && L->ln_2_b &&
             L->up_w && L->up_b && L->down_w && L->down_b;
    }
    return ok ? NULL : "language model: a tensor is missing or has the wrong shape or type";
}

struct cs_lm * cs_lm_load(const char * path) {
    // Metadata first: the data is read after every check has passed, into a buffer whose allocation can fail cleanly.
    struct ggml_context * meta = NULL;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    struct gguf_context * g = gguf_init_from_file(path, gp);
    if (g == NULL) {
        CS_LM_WARN("language model: not a readable GGUF file");
        return NULL;
    }
    struct cs_lm * lm = (struct cs_lm *) calloc(1, sizeof(struct cs_lm));
    if (lm == NULL) {
        gguf_free(g);
        ggml_free(meta);
        return NULL;
    }
    lm->weights = meta;
    const char * error = cs_lm_check(lm, g);
    if (error == NULL) {
        lm->backend = ggml_backend_cpu_init();
        lm->weights_buf = lm->backend != NULL ? ggml_backend_alloc_ctx_tensors(lm->weights, lm->backend) : NULL;
        if (lm->weights_buf == NULL) {
            error = "language model: out of memory";
        } else if (!cs_lm_read_data(path, g, lm->weights)) {
            error = "language model: the file ends early";
        }
    }
    gguf_free(g);
    if (error == NULL) {
        lm->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(lm->backend));
        lm->meta_size = ggml_tensor_overhead() * CS_LM_GRAPH_SIZE + ggml_graph_overhead_custom(CS_LM_GRAPH_SIZE, false);
        lm->meta = malloc(lm->meta_size);
        if (lm->meta == NULL) {
            error = "language model: out of memory";
        }
    }
    if (error != NULL) {
        CS_LM_WARN(error);
        cs_lm_free(lm);
        return NULL;
    }
    return lm;
}

int cs_lm_n_vocab(const struct cs_lm * lm) {
    return lm->n_vocab;
}

int cs_lm_n_ctx(const struct cs_lm * lm) {
    return lm->n_ctx;
}

void cs_lm_buffer_sizes(const struct cs_lm * lm, size_t * weights, size_t * compute) {
    *weights = ggml_backend_buffer_get_size(lm->weights_buf);
    *compute = ggml_gallocr_get_buffer_size(lm->galloc, 0);
}

// Key/value cache for sequences of up to n_kv tokens (f32, both n_layer x n_kv x n_embd); false when out of memory.
static bool cs_lm_kv_init(struct cs_lm * lm, int n_kv) {
    cs_lm_kv_free(lm);
    struct ggml_init_params p = { /*.mem_size =*/ 2 * ggml_tensor_overhead(), /*.mem_buffer =*/ NULL, /*.no_alloc =*/ true };
    lm->kv_ctx = ggml_init(p);
    const int64_t n = (int64_t) lm->n_layer * n_kv * lm->n_embd;
    lm->k = ggml_new_tensor_1d(lm->kv_ctx, GGML_TYPE_F32, n);
    lm->v = ggml_new_tensor_1d(lm->kv_ctx, GGML_TYPE_F32, n);
    lm->kv_buf = ggml_backend_alloc_ctx_tensors(lm->kv_ctx, lm->backend);
    lm->n_kv = n_kv;
    return lm->kv_buf != NULL;
}

// Graph for n tokens at positions n_past.. : stores their keys and values in the cache and, for the first n_out of
// them, outputs the next-token distribution ("probs", n_vocab x n_out).
static struct ggml_cgraph * cs_lm_graph(struct cs_lm * lm, struct ggml_context * ctx0, int n_past, int n, int n_out) {
    const int n_embd = lm->n_embd;
    const int n_head = lm->n_head;
    const int head_dim = n_embd / n_head;
    const size_t es = ggml_element_size(lm->k);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, CS_LM_GRAPH_SIZE, false);

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    struct ggml_tensor * inpL = ggml_add(ctx0, ggml_get_rows(ctx0, lm->wte, embd), ggml_get_rows(ctx0, lm->wpe, pos));

    for (int il = 0; il < lm->n_layer; il++) {
        const struct cs_lm_layer * L = &lm->layers[il];

        struct ggml_tensor * cur = ggml_norm(ctx0, inpL, lm->eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L->ln_1_g), L->ln_1_b);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L->qkv_w, cur), L->qkv_b);   // [3 n_embd, n]

        struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, n, cur->nb[1], 0 * sizeof(float) * n_embd);
        struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, n, cur->nb[1], 1 * sizeof(float) * n_embd);
        struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, n, cur->nb[1], 2 * sizeof(float) * n_embd);

        // keys and values of the new tokens into the cache (run before the reads below: graph order)
        const size_t layer_offs = es * n_embd * ((size_t) il * lm->n_kv);
        struct ggml_tensor * k = ggml_view_1d(ctx0, lm->k, (int64_t) n * n_embd, layer_offs + es * n_embd * n_past);
        struct ggml_tensor * v = ggml_view_1d(ctx0, lm->v, (int64_t) n * n_embd, layer_offs + es * n_embd * n_past);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));

        const int n_all = n_past + n;
        struct ggml_tensor * Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, ggml_cont(ctx0, Qcur), head_dim, n_head, n),
                                              0, 2, 1, 3);                                   // [hd, n, heads]
        struct ggml_tensor * K = ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, ggml_view_1d(ctx0, lm->k, (int64_t) n_all * n_embd, layer_offs),
                            head_dim, n_head, n_all), 0, 2, 1, 3);                           // [hd, n_all, heads]
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);                                  // [n_all, n, heads]
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf((float) head_dim));
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);
        struct ggml_tensor * Vt = ggml_cont(ctx0, ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, ggml_view_1d(ctx0, lm->v, (int64_t) n_all * n_embd, layer_offs),
                            head_dim, n_head, n_all), 1, 2, 0, 3));                          // [n_all, hd, heads]
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, Vt, KQ);                               // [hd, n, heads]
        cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3)), n_embd, n);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L->out_w, cur), L->out_b);
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpL);

        cur = ggml_norm(ctx0, inpFF, lm->eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L->ln_2_g), L->ln_2_b);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L->up_w, cur), L->up_b);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L->down_w, cur), L->down_b);
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    if (n_out > 0) {
        struct ggml_tensor * h = ggml_view_2d(ctx0, inpL, n_embd, n_out, inpL->nb[1], 0);
        h = ggml_norm(ctx0, h, lm->eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, lm->ln_f_g), lm->ln_f_b);
        struct ggml_tensor * probs = ggml_soft_max_inplace(ctx0, ggml_mul_mat(ctx0, lm->wte, h));
        ggml_set_name(probs, "probs");
        ggml_set_output(probs);
        ggml_build_forward_expand(gf, probs);
    }
    return gf;
}

// Evaluates seq[start .. len) (positions start..), with the cache valid before start; adds log P(seq[q + 1]) for
// q = start .. len - 2 into lp[q + 1].
static bool cs_lm_eval(struct cs_lm * lm, const int32_t * seq, int start, int len, double * lp) {
    int32_t pos[CS_LM_CHUNK];
    for (int p0 = start; p0 < len; p0 += CS_LM_CHUNK) {
        const int n = len - p0 < CS_LM_CHUNK ? len - p0 : CS_LM_CHUNK;
        const int n_out = (len - 1 - p0) < n ? (len - 1 - p0) : n;   // positions whose next token is in seq
        struct ggml_init_params p = { /*.mem_size =*/ lm->meta_size, /*.mem_buffer =*/ lm->meta, /*.no_alloc =*/ true };
        struct ggml_context * ctx0 = ggml_init(p);   // in lm->meta: never fails
        struct ggml_cgraph * gf = cs_lm_graph(lm, ctx0, p0, n, n_out);
        if (!ggml_gallocr_alloc_graph(lm->galloc, gf)) {
            ggml_free(ctx0);
            return false;
        }
        for (int i = 0; i < n; i++) {
            pos[i] = p0 + i;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "embd"), seq + p0, 0, (size_t) n * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), pos, 0, (size_t) n * sizeof(int32_t));
        if (ggml_backend_graph_compute(lm->backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx0);
            return false;
        }
        if (n_out > 0) {
            const struct ggml_tensor * probs = ggml_graph_get_tensor(gf, "probs");
            for (int i = 0; i < n_out; i++) {
                const int q = p0 + i;
                float prob = 0.0f;
                ggml_backend_tensor_get(probs, &prob, ((size_t) i * lm->n_vocab + (size_t) seq[q + 1]) * sizeof(float),
                                        sizeof(float));
                lp[q + 1] = log(prob > FLT_MIN ? (double) prob : (double) FLT_MIN);
            }
        }
        ggml_free(ctx0);
    }
    return true;
}

// Lexicographic order of sequences i and j (then by index), for the insertion sort in cs_lm_score.
static int cs_lm_compare(const int32_t * tokens, const int32_t * offsets, int i, int j) {
    const int32_t * x = tokens + offsets[i];
    const int32_t * y = tokens + offsets[j];
    const int nx = offsets[i + 1] - offsets[i];
    const int ny = offsets[j + 1] - offsets[j];
    for (int k = 0; k < nx && k < ny; k++) {
        if (x[k] != y[k]) {
            return x[k] < y[k] ? -1 : 1;
        }
    }
    if (nx != ny) {
        return nx < ny ? -1 : 1;
    }
    return i < j ? -1 : (i > j ? 1 : 0);
}

bool cs_lm_score(struct cs_lm * lm, const int32_t * tokens, const int32_t * offsets, int n_seqs, int n_threads,
                 double * out) {
    if (lm == NULL || n_seqs < 0 || (n_seqs > 0 && (tokens == NULL || offsets == NULL || out == NULL))) {
        return false;
    }
    int max_len = 1;
    for (int i = 0; i < n_seqs; i++) {
        const int len = offsets[i + 1] - offsets[i];
        if (offsets[i] < 0 || len < 1 || len > lm->n_ctx) {
            return false;
        }
        for (int k = offsets[i]; k < offsets[i + 1]; k++) {
            if (tokens[k] < 0 || tokens[k] >= lm->n_vocab) {
                return false;
            }
        }
        if (len > max_len) {
            max_len = len;
        }
    }
    if (n_seqs == 0) {
        return true;
    }

    int * order = (int *) malloc((size_t) n_seqs * sizeof(int));
    double * lp = (double *) calloc((size_t) max_len, sizeof(double));
    bool ok = order != NULL && lp != NULL && cs_lm_kv_init(lm, max_len);
    if (ok) {
        // Sorted, each sequence shares the longest possible prefix with the one before it (a handful of sequences:
        // insertion sort).
        for (int i = 0; i < n_seqs; i++) {
            int r = i;
            while (r > 0 && cs_lm_compare(tokens, offsets, order[r - 1], i) > 0) {
                order[r] = order[r - 1];
                r--;
            }
            order[r] = i;
        }
        ggml_backend_cpu_set_n_threads(lm->backend, n_threads > 0 ? n_threads : 1);

        const int32_t * prev = NULL;
        int prev_len = 0;
        for (int r = 0; ok && r < n_seqs; r++) {
            const int i = order[r];
            const int32_t * seq = tokens + offsets[i];
            const int len = offsets[i + 1] - offsets[i];
            int common = 0;   // lp[1 .. common - 1] and the cache of positions < common are valid for seq
            while (prev != NULL && common < len && common < prev_len && prev[common] == seq[common]) {
                common++;
            }
            if (common < len) {
                const int start = common > 0 ? common - 1 : 0;   // the last shared token predicts the first new one
                ok = cs_lm_eval(lm, seq, start, len, lp);
            }
            double sum = 0.0;
            for (int j = 1; j < len; j++) {
                sum += lp[j];
            }
            out[i] = sum;
            prev = seq;
            prev_len = len;
        }
    }
    cs_lm_kv_free(lm);   // only held during the call
    free(order);
    free(lp);
    return ok;
}

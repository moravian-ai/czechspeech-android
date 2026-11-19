// CzechSpeech language model for variant selection: GPT-2 on ggml (the ggml that whisper.cpp brings), loaded from a
// GGUF file written by scripts/lm/convert_gpt2.py. It only scores token sequences; the tokenizer is in Kotlin.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cs_lm;

// Loads the model; NULL (with a warning in the log that names the reason, never text) when the file is not a GPT-2
// GGUF file this code can run: wrong shapes or tensor types, more layers than one graph holds, a cut file, or no
// memory for the weights. The file's SHA-256 is not checked here: it is checked when the file is downloaded.
struct cs_lm * cs_lm_load(const char * path);
void cs_lm_free(struct cs_lm * lm);

int cs_lm_n_vocab(const struct cs_lm * lm);
int cs_lm_n_ctx(const struct cs_lm * lm);

// Bytes held while the model is loaded: the weights, and the compute buffer kept between calls (sized for the
// largest chunk scored so far; 0 before the first call).
void cs_lm_buffer_sizes(const struct cs_lm * lm, size_t * weights, size_t * compute);

// Scores n_seqs token sequences: sequence i is tokens[offsets[i] .. offsets[i + 1]), starting with the start token.
// out[i] = sum over j >= 1 of log P(token j | tokens before j), 0 for a sequence of the start token alone. Sequences
// are evaluated in sorted order and share the key/value cache of their common prefix, so variants of one sentence cost
// little more than one. Returns false on invalid input (a token out of range, a sequence longer than the context,
// an empty sequence), when the key/value cache cannot be allocated, or when ggml fails.
bool cs_lm_score(struct cs_lm * lm, const int32_t * tokens, const int32_t * offsets, int n_seqs, int n_threads,
                 double * out);

#ifdef __cplusplus
}
#endif

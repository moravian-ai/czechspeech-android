// CzechSpeech JNI glue between ai.moravian.czechspeech.internal.NativeBridge and whisper.cpp.
//
// Written for CzechSpeech; parts of the structure (context handles as jlong, the parameter set for whisper_full)
// follow whisper.cpp examples/whisper.android/lib/src/main/jni/whisper/jni.c, MIT licence,
// Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.
//
// Privacy rules implemented here:
//   - Audio arrives as a Java float[]; it is copied into a malloc'd buffer with Get<Type>ArrayRegion (no pinned
//     or borrowed Java memory), and that buffer is zeroed and freed as soon as the native call is done.
//   - Transcript text is returned as UTF-8 bytes; the native copy is zeroed and freed. Nothing is logged except
//     whisper.cpp/ggml warnings and errors, which never contain transcript text (INFO and DEBUG are dropped:
//     DEBUG output of whisper.cpp includes decoded tokens).
//   - No file or network I/O except reading the model file the caller passes in.
//   - A ggml assertion aborts the process with its message in logcat; it never forks or execs a debugger.

#include <jni.h>

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "whisper.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <android/set_abort_message.h>
#else
#include <stdio.h>
#endif

#define CS_TAG "CzechSpeech"
#define CS_STACK_SAMPLES 4096

// ---------------------------------------------------------------------------------------------------------------------
// helpers

static void cs_secure_zero(void *p, size_t n) {
    if (p == NULL || n == 0) {
        return;
    }
    memset(p, 0, n);
    // Keep the compiler from dropping the memset as a dead store before free().
    __asm__ __volatile__("" : : "r"(p) : "memory");
}

static void cs_throw(JNIEnv *env, const char *class_name, const char *message) {
    jclass cls = (*env)->FindClass(env, class_name);
    if (cls != NULL) {
        (*env)->ThrowNew(env, cls, message);
        (*env)->DeleteLocalRef(env, cls);
    }
}

// Validates n against the array and copies the first n samples into dst (stack) or a new malloc'd buffer.
// Returns NULL with a pending Java exception on error. *heap tells the caller whether to free the result.
static float *cs_copy_samples(JNIEnv *env, jfloatArray samples, jint n, float *stack_buf, bool *heap) {
    *heap = false;
    if (samples == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "samples");
        return NULL;
    }
    const jsize len = (*env)->GetArrayLength(env, samples);
    if (n < 0 || n > len) {
        cs_throw(env, "java/lang/IllegalArgumentException", "sample count out of range");
        return NULL;
    }
    float *buf = stack_buf;
    if (n > CS_STACK_SAMPLES || stack_buf == NULL) {
        buf = (float *) malloc((size_t) (n > 0 ? n : 1) * sizeof(float));
        if (buf == NULL) {
            cs_throw(env, "java/lang/OutOfMemoryError", "native audio buffer");
            return NULL;
        }
        *heap = true;
    }
    if (n > 0) {
        (*env)->GetFloatArrayRegion(env, samples, 0, n, buf);
    }
    return buf;
}

static void cs_release_samples(float *buf, jint n, bool heap) {
    if (buf == NULL) {
        return;
    }
    cs_secure_zero(buf, (size_t) (n > 0 ? n : 0) * sizeof(float));
    if (heap) {
        free(buf);
    }
}

// whisper.cpp and ggml log through this callback (set in JNI_OnLoad). Only warnings and errors pass.
static void cs_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void) user_data;
    static _Thread_local int keep_continuation = 0;
    int keep;
    if (level == GGML_LOG_LEVEL_CONT) {
        keep = keep_continuation;
    } else {
        keep = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
        keep_continuation = keep;
    }
    if (!keep || text == NULL) {
        return;
    }
#ifdef __ANDROID__
    __android_log_write(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN, CS_TAG, text);
#else
    fputs(text, stderr);
#endif
}

// GGML_ASSERT and GGML_ABORT call this with "file:line: message" and then abort(). Without it, ggml writes the message
// to stderr, which nobody sees on Android, and calls ggml_print_backtrace(). The message names a source location and
// a condition, never audio or transcript text.
static void cs_ggml_abort(const char *message) {
    if (message == NULL) {
        return;
    }
#ifdef __ANDROID__
    __android_log_write(ANDROID_LOG_FATAL, CS_TAG, message);
    android_set_abort_message(message);  // shown in the tombstone and in ApplicationExitInfo
#else
    fprintf(stderr, "%s: ggml abort: %s\n", CS_TAG, message);
#endif
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void) vm;
    (void) reserved;
    // ggml_print_backtrace() forks the process and execs gdb, then lldb, to print a backtrace: never inside an app.
    // It returns at once when GGML_NO_BACKTRACE is set. The Kotlin loader sets it on Android before loading; this
    // covers the host build and any other way of loading the library. An existing value is kept.
    setenv("GGML_NO_BACKTRACE", "1", 0);
    ggml_set_abort_callback(cs_ggml_abort);
    whisper_log_set(cs_log_callback, NULL);
    return JNI_VERSION_1_6;
}

// ---------------------------------------------------------------------------------------------------------------------
// whisper

struct cs_whisper {
    struct whisper_context *ctx;
    atomic_int abort_generation;  // a transcription with this generation stops at the next check
    // Language of the last transcription: a pointer into whisper.cpp's static whisper_lang_str table (never freed),
    // NULL before the first call, after a failed or cancelled one and while a call runs. Read on the thread that
    // called nativeWhisperTranscribe, right after it, so it needs no lock.
    const char *language;
};

struct cs_abort {
    struct cs_whisper *w;
    int generation;
};

// whisper.cpp (pinned commit) checks abort_callback after the encoder pass and after each decoder step, not inside
// the encoder graph; encoder_begin_callback adds a check before the encoder starts.
static bool cs_abort_callback(void *data) {
    const struct cs_abort *a = (const struct cs_abort *) data;
    return atomic_load_explicit(&a->w->abort_generation, memory_order_relaxed) == a->generation;
}

static bool cs_encoder_begin_callback(struct whisper_context *ctx, struct whisper_state *state, void *data) {
    (void) ctx;
    (void) state;
    return !cs_abort_callback(data);
}

// Encoder window trimmed to the utterance: 20 ms mel frames + 32, rounded up to a multiple of 16; the full
// window (0) from 1500 frames (30 s) up. Never shorter than CS_MIN_AUDIO_CTX: the model was never trained on a
// window that short, and below 4 s of speech it then invents or repeats sentences (15-46 % of utterances).
// A floor of 512 frames (~10 s) cut WER on short utterances from 6.28 to 3.90 % (clean) and 10.82 to 9.89 %
// (noise) and removed the invented-word bursts, against the trimmed window on the same short utterances (CHANGELOG,
// "The encoder window is never shorter than 512 frames").
#define CS_MIN_AUDIO_CTX 512

static int cs_audio_ctx(jint n_samples) {
    int frames = (int) ((n_samples + 319) / 320) + 32;
    frames = (frames + 15) / 16 * 16;
    if (frames < CS_MIN_AUDIO_CTX) frames = CS_MIN_AUDIO_CTX;
    return frames < 1500 ? frames : 0;
}

JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeAudioCtx(JNIEnv *env, jclass clazz, jint n_samples) {
    (void) env;
    (void) clazz;
    return cs_audio_ctx(n_samples);
}

JNIEXPORT jstring JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeSystemInfo(JNIEnv *env, jclass clazz) {
    (void) clazz;
    return (*env)->NewStringUTF(env, whisper_print_system_info());
}

JNIEXPORT jlong JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperInit(JNIEnv *env, jclass clazz, jstring model_path) {
    (void) clazz;
    if (model_path == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "modelPath");
        return 0;
    }
    const char *path = (*env)->GetStringUTFChars(env, model_path, NULL);
    if (path == NULL) {
        return 0;
    }
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);
    (*env)->ReleaseStringUTFChars(env, model_path, path);
    if (ctx == NULL) {
        return 0;
    }
    struct cs_whisper *w = (struct cs_whisper *) calloc(1, sizeof(struct cs_whisper));
    if (w == NULL) {
        whisper_free(ctx);
        cs_throw(env, "java/lang/OutOfMemoryError", "whisper handle");
        return 0;
    }
    w->ctx = ctx;
    atomic_init(&w->abort_generation, 0);
    return (jlong) (intptr_t) w;
}

JNIEXPORT void JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperFree(JNIEnv *env, jclass clazz, jlong handle) {
    (void) env;
    (void) clazz;
    struct cs_whisper *w = (struct cs_whisper *) (intptr_t) handle;
    if (w == NULL) {
        return;
    }
    whisper_free(w->ctx);
    free(w);
}

JNIEXPORT void JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperAbort(JNIEnv *env, jclass clazz, jlong handle,
                                                                       jint generation) {
    (void) env;
    (void) clazz;
    struct cs_whisper *w = (struct cs_whisper *) (intptr_t) handle;
    if (w != NULL) {
        atomic_store_explicit(&w->abort_generation, generation, memory_order_relaxed);
    }
}

#define CS_MAX_LANGUAGES 128

// Splits a comma-separated list of two- or three-letter lowercase whisper language codes ("cs", "pl,en") into whisper.cpp language ids. Returns the count,
// or -1 for an empty list, an empty item (a leading, trailing or doubled comma), a code whisper.cpp does not know,
// a code of 8 characters or more, or more than max entries.
static int cs_parse_languages(const char *languages, int *ids, int max) {
    int n = 0;
    const char *p = languages;
    while (p != NULL && *p != '\0') {
        const char *end = strchr(p, ',');
        const size_t len = end != NULL ? (size_t) (end - p) : strlen(p);
        char code[8];
        if (len == 0 || len >= sizeof(code) || n >= max) {
            return -1;
        }
        memcpy(code, p, len);
        code[len] = '\0';
        const int id = whisper_lang_id(code);
        if (id < 0) {
            return -1;
        }
        ids[n++] = id;
        if (end != NULL && end[1] == '\0') {
            return -1;  // a trailing comma
        }
        p = end != NULL ? end + 1 : p + len;
    }
    return n > 0 ? n : -1;
}

// Returns the transcript as UTF-8 bytes (possibly empty), or null when whisper_full failed or was aborted.
// languages: comma-separated two- or three-letter lowercase whisper language codes the model speaks and the caller allows ("cs", "pl,en"); a malformed or
// unknown list throws IllegalArgumentException before anything is allocated. One code forces that language, exactly
// as before (no detection pass). Several: the language is detected first (mel spectrogram, one encoder pass and one
// decoder step on a state of its own) and the most probable one of the list is used. whisper.cpp's detection cannot
// be limited to a list and takes no abort callback, so the abort is checked before it and again right after it:
// a call cancelled before it starts, or during the detection, returns null without decoding. nativeWhisperLanguage
// then tells the language of the transcript.
JNIEXPORT jbyteArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperTranscribe(JNIEnv *env, jclass clazz, jlong handle,
                                                                            jint generation, jfloatArray samples,
                                                                            jint n_samples, jint n_threads,
                                                                            jstring languages) {
    (void) clazz;
    struct cs_whisper *w = (struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->ctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "whisper context not loaded");
        return NULL;
    }
    w->language = NULL;
    if (languages == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "languages");
        return NULL;
    }
    int ids[CS_MAX_LANGUAGES];
    const char *list = (*env)->GetStringUTFChars(env, languages, NULL);
    if (list == NULL) {
        return NULL;  // OutOfMemoryError pending
    }
    const int n_languages = cs_parse_languages(list, ids, CS_MAX_LANGUAGES);
    (*env)->ReleaseStringUTFChars(env, languages, list);
    if (n_languages < 0) {
        cs_throw(env, "java/lang/IllegalArgumentException", "invalid language list");
        return NULL;
    }
    bool heap = false;
    float *pcm = cs_copy_samples(env, samples, n_samples, NULL, &heap);
    if (pcm == NULL) {
        return NULL;
    }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads        = n_threads > 0 ? n_threads : 1;
    params.translate        = false;
    params.language         = whisper_lang_str(ids[0]);
    params.detect_language  = false;
    params.no_context       = true;   // no text carried over between utterances
    params.no_timestamps    = true;
    // whisper.cpp keeps suppress_blank on, which forbids the end of the text as the first token. A model that learned to
    // answer a segment without speech with an empty transcript then has to write something into silence:
    // 100 % of 360 such segments with it on, 0 % off; speech transcripts of the CzechSpeech model unchanged (measured
    // on 12 Nov 2026). The empty result is dropped by the callers.
    params.suppress_blank   = false;
    params.single_segment   = false;
    params.offset_ms        = 0;
    params.print_special    = false;
    params.print_progress   = false;
    params.print_realtime   = false;  // would printf the text
    params.print_timestamps = false;
    params.audio_ctx        = cs_audio_ctx(n_samples);

    struct cs_abort abort_data = { w, generation };
    params.abort_callback                   = cs_abort_callback;
    params.abort_callback_user_data         = &abort_data;
    params.encoder_begin_callback           = cs_encoder_begin_callback;
    params.encoder_begin_callback_user_data = &abort_data;

    if (n_languages > 1) {
        if (cs_abort_callback(&abort_data)) {
            cs_release_samples(pcm, n_samples, heap);
            return NULL;  // already cancelled: no state, mel or encoder pass
        }
        // A state of its own per call, deliberately: the context's default state keeps the audio_ctx that the
        // previous whisper_full set (whisper.cpp never resets it), which the detection would inherit; a fresh one
        // reads the whole 30 s window. It is freed again below, so no second state stays in memory.
        struct whisper_state *state = whisper_init_state(w->ctx);
        if (state == NULL) {
            cs_release_samples(pcm, n_samples, heap);
            cs_throw(env, "java/lang/OutOfMemoryError", "whisper state");
            return NULL;
        }
        int best = -1;
        float *probs = (float *) calloc((size_t) whisper_lang_max_id() + 1, sizeof(float));
        if (probs != NULL && whisper_pcm_to_mel_with_state(w->ctx, state, pcm, n_samples, params.n_threads) == 0 &&
            whisper_lang_auto_detect_with_state(w->ctx, state, 0, params.n_threads, probs) >= 0) {
            for (int i = 0; i < n_languages; i++) {
                if (best < 0 || probs[ids[i]] > probs[best]) {
                    best = ids[i];
                }
            }
        }
        const bool out_of_memory = probs == NULL;
        free(probs);
        whisper_free_state(state);
        if (cs_abort_callback(&abort_data)) {
            cs_release_samples(pcm, n_samples, heap);
            return NULL;  // cancelled during the detection, which cannot be interrupted: drop the result
        }
        if (best < 0) {
            cs_release_samples(pcm, n_samples, heap);
            if (out_of_memory) {
                cs_throw(env, "java/lang/OutOfMemoryError", "language probabilities");
            }
            return NULL;
        }
        params.language = whisper_lang_str(best);
    }

    const int rc = whisper_full(w->ctx, params, pcm, n_samples);
    cs_release_samples(pcm, n_samples, heap);
    if (rc != 0) {
        return NULL;
    }

    const int n_segments = whisper_full_n_segments(w->ctx);
    size_t total = 0;
    for (int i = 0; i < n_segments; i++) {
        const char *seg = whisper_full_get_segment_text(w->ctx, i);
        if (seg != NULL) {
            total += strlen(seg);
        }
    }
    char *text = (char *) malloc(total > 0 ? total : 1);
    if (text == NULL) {
        cs_throw(env, "java/lang/OutOfMemoryError", "transcript buffer");
        return NULL;
    }
    size_t pos = 0;
    for (int i = 0; i < n_segments; i++) {
        const char *seg = whisper_full_get_segment_text(w->ctx, i);
        if (seg != NULL) {
            const size_t len = strlen(seg);
            memcpy(text + pos, seg, len);
            pos += len;
        }
    }
    jbyteArray out = (*env)->NewByteArray(env, (jsize) total);
    if (out != NULL && total > 0) {
        (*env)->SetByteArrayRegion(env, out, 0, (jsize) total, (const jbyte *) text);
    }
    cs_secure_zero(text, total);
    free(text);
    if (out != NULL) {
        w->language = params.language;
    }
    return out;
}

// Language of the last transcript (a whisper language code; a static string owned by whisper.cpp), or null when the last call
// failed, was cancelled or has not run.
JNIEXPORT jstring JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperLanguage(JNIEnv *env, jclass clazz, jlong handle) {
    (void) clazz;
    const struct cs_whisper *w = (const struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->language == NULL) {
        return NULL;
    }
    return (*env)->NewStringUTF(env, w->language);
}

// ---------------------------------------------------------------------------------------------------------------------
// Silero VAD

struct cs_mem_reader {
    const uint8_t *data;
    size_t size;
    size_t pos;
};

static size_t cs_mem_read(void *ctx, void *output, size_t read_size) {
    struct cs_mem_reader *r = (struct cs_mem_reader *) ctx;
    const size_t left = r->size - r->pos;
    const size_t n = read_size < left ? read_size : left;
    memcpy(output, r->data + r->pos, n);
    r->pos += n;
    return n;
}

static bool cs_mem_eof(void *ctx) {
    const struct cs_mem_reader *r = (const struct cs_mem_reader *) ctx;
    return r->pos >= r->size;
}

static void cs_mem_close(void *ctx) {
    (void) ctx;
}

JNIEXPORT jlong JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeVadInit(JNIEnv *env, jclass clazz, jbyteArray model,
                                                                  jint n_threads) {
    (void) clazz;
    if (model == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "model");
        return 0;
    }
    const jsize size = (*env)->GetArrayLength(env, model);
    uint8_t *bytes = (uint8_t *) malloc(size > 0 ? (size_t) size : 1);
    if (bytes == NULL) {
        cs_throw(env, "java/lang/OutOfMemoryError", "VAD model buffer");
        return 0;
    }
    (*env)->GetByteArrayRegion(env, model, 0, size, (jbyte *) bytes);

    struct cs_mem_reader reader = { bytes, (size_t) size, 0 };
    struct whisper_model_loader loader = {
        .context = &reader,
        .read    = cs_mem_read,
        .eof     = cs_mem_eof,
        .close   = cs_mem_close,
    };
    struct whisper_vad_context_params cparams = whisper_vad_default_context_params();
    cparams.n_threads = n_threads > 0 ? n_threads : 1;
    cparams.use_gpu   = false;
    struct whisper_vad_context *vctx = whisper_vad_init_with_params(&loader, cparams);
    free(bytes);
    return (jlong) (intptr_t) vctx;
}

JNIEXPORT void JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeVadFree(JNIEnv *env, jclass clazz, jlong handle) {
    (void) env;
    (void) clazz;
    struct whisper_vad_context *vctx = (struct whisper_vad_context *) (intptr_t) handle;
    if (vctx != NULL) {
        whisper_vad_free(vctx);
    }
}

JNIEXPORT void JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeVadReset(JNIEnv *env, jclass clazz, jlong handle) {
    (void) env;
    (void) clazz;
    struct whisper_vad_context *vctx = (struct whisper_vad_context *) (intptr_t) handle;
    if (vctx != NULL) {
        whisper_vad_reset_state(vctx);
    }
}

// Speech probabilities for consecutive 512-sample windows (the last one zero-padded), continuing the LSTM state of
// earlier calls. Writes up to out.length values and returns the number of windows, or -1 on error.
JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeVadProbs(JNIEnv *env, jclass clazz, jlong handle,
                                                                   jfloatArray samples, jint n_samples,
                                                                   jfloatArray out) {
    (void) clazz;
    struct whisper_vad_context *vctx = (struct whisper_vad_context *) (intptr_t) handle;
    if (vctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "VAD context not loaded");
        return -1;
    }
    if (out == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "out");
        return -1;
    }
    float stack_buf[CS_STACK_SAMPLES];
    bool heap = false;
    float *pcm = cs_copy_samples(env, samples, n_samples, stack_buf, &heap);
    if (pcm == NULL) {
        return -1;
    }
    const bool ok = whisper_vad_detect_speech_no_reset(vctx, pcm, n_samples);
    cs_release_samples(pcm, n_samples, heap);
    if (!ok) {
        return -1;
    }
    const int n_probs = whisper_vad_n_probs(vctx);
    const jsize cap = (*env)->GetArrayLength(env, out);
    const int n_out = n_probs < cap ? n_probs : cap;
    if (n_out > 0) {
        (*env)->SetFloatArrayRegion(env, out, 0, n_out, whisper_vad_probs(vctx));
    }
    return n_probs;
}

// Number of speech segments in the samples by whisper_vad_segments_from_samples (fresh LSTM state, default
// segment parameters except the threshold), or -1 on error.
JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeVadSpeechSegments(JNIEnv *env, jclass clazz, jlong handle,
                                                                            jfloatArray samples, jint n_samples,
                                                                            jfloat threshold) {
    (void) clazz;
    struct whisper_vad_context *vctx = (struct whisper_vad_context *) (intptr_t) handle;
    if (vctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "VAD context not loaded");
        return -1;
    }
    bool heap = false;
    float *pcm = cs_copy_samples(env, samples, n_samples, NULL, &heap);
    if (pcm == NULL) {
        return -1;
    }
    struct whisper_vad_params params = whisper_vad_default_params();
    params.threshold = threshold;
    struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(vctx, params, pcm, n_samples);
    cs_release_samples(pcm, n_samples, heap);
    if (segments == NULL) {
        return -1;
    }
    const int n = whisper_vad_segments_n_segments(segments);
    whisper_vad_free_segments(segments);
    return n;
}

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

// sched_getaffinity, sched_setaffinity and the CPU_* macros (glibc and bionic declare them for _GNU_SOURCE).
#define _GNU_SOURCE

#include <jni.h>

#include <math.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "whisper.h"
#include "czechspeech_lm.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <android/set_abort_message.h>
#else
#include <stdio.h>
#endif

#define CS_TAG "CzechSpeech"
#define CS_STACK_SAMPLES 4096
#define CS_MAX_LANGUAGES 128

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
// CPU affinity of the calling thread (thread placement, see ThreadPlacement.kt)
//
// The SDK narrows the affinity of its recognition thread for a recognition and unpins it afterwards (every CPU, never
// a mask read before, see nativeUnpinThread). ggml (pinned commit) starts its
// worker threads for every graph from the thread that computes it (a threadpool per graph: ggml_graph_compute without
// cplan->threadpool, ggml-cpu.c), with an empty cpumask, so it never sets their affinity and they inherit the
// caller's; whisper.cpp's own threads (mel spectrogram, beam search decoders) are std::threads of the caller too, and
// GGML_OPENMP is off. Linux copies a thread's affinity to every thread it creates.

// The CPUs of a mask (bit i = CPU i); CPUs from 64 up do not fit and make the mask unknown (0).
static jlong cs_mask_of(const cpu_set_t *set) {
    uint64_t mask = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, set)) {
            continue;
        }
        if (cpu >= 64) {
            return 0;
        }
        mask |= (uint64_t) 1 << cpu;
    }
    return (jlong) mask;
}

// sched_getaffinity of the calling thread: the CPUs it may use and that are active now; 0 on failure.
JNIEXPORT jlong JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeThreadCpus(JNIEnv *env, jclass clazz) {
    (void) env;
    (void) clazz;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    return cs_mask_of(&set);
}

// sched_setaffinity of the calling thread (pid 0) to the CPUs of mask; false when the kernel refuses it (no CPU of the
// mask in the thread's cpuset, say), and then nothing changes.
JNIEXPORT jboolean JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeSetThreadCpus(JNIEnv *env, jclass clazz, jlong mask) {
    (void) env;
    (void) clazz;
    if (mask == 0) {
        return JNI_FALSE;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu = 0; cpu < 64 && cpu < CPU_SETSIZE; cpu++) {
        if (((uint64_t) mask >> cpu) & 1u) {
            CPU_SET(cpu, &set);
        }
    }
    return sched_setaffinity(0, sizeof(set), &set) == 0 ? JNI_TRUE : JNI_FALSE;
}

// sched_setaffinity of the calling thread to every CPU: the kernel keeps only those of the thread's cpuset (bits past
// the last possible CPU are ignored), so the thread may again use whatever its cpuset allows, now and after the cpuset
// changes. Linux 6.2+ keeps a sched_setaffinity mask as the thread's own request (user_cpus_ptr), intersects every
// later cpuset with it and copies it to the threads the thread starts: putting back a mask read earlier would keep it
// narrowed, every CPU drops the request. False when refused.
JNIEXPORT jboolean JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeUnpinThread(JNIEnv *env, jclass clazz) {
    (void) env;
    (void) clazz;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        CPU_SET(cpu, &set);
    }
    return sched_setaffinity(0, sizeof(set), &set) == 0 ? JNI_TRUE : JNI_FALSE;
}

// ---------------------------------------------------------------------------------------------------------------------
// whisper

struct cs_whisper {
    struct whisper_context *ctx;
    atomic_int abort_generation;  // a transcription with this generation stops at the next check
    int min_audio_ctx;            // floor of the encoder window, in frames; belongs to the model (see cs_audio_ctx)
    // Language of the last transcription: a pointer into whisper.cpp's static whisper_lang_str table (never freed),
    // NULL before the first call, after a failed or cancelled one and while a call runs. Read on the thread that
    // called nativeWhisperTranscribe, right after it, so it needs no lock.
    const char *language;
    // Shares of the languages of the last list recognition, rescaled to sum 1 (index as in the list), the language
    // selection's input; n_shares is 0 for a forced language, before the first call, after a failed or cancelled one
    // and at the start of a call. The selection callback sets them in the middle of a call, so read them only after
    // the call returns, like language.
    float shares[CS_MAX_LANGUAGES];
    int n_shares;
};

struct cs_abort {
    struct cs_whisper *w;
    int generation;
};

// whisper.cpp (pinned commit) checks abort_callback after the encoder pass and after each decoder step;
// encoder_begin_callback adds a check before the encoder starts, and native/patches/whisper.cpp-encoder-abort.patch
// hands the callback to ggml for the encoder's graphs, which checks it after every node: an abort during the encoder
// returns within one node's time, not at the end of the pass (seconds on a phone). ggml calls it on the calling
// thread (thread 0 of the graph).
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
// window (0) from 1500 frames (30 s) up. Never shorter than a floor (min_frames): a model was never trained on a
// window much shorter than what it saw, and below 4 s of speech it then invents or repeats sentences (the published Czech model
// trimmed freely: 15-46 % of utterances). The floor belongs to the model, so it is kept with its context (cs_whisper,
// nativeWhisperSetMinAudioCtx). The default, CS_MIN_AUDIO_CTX, is the 512 frames (~10 s) that the published Czech model needs: it cut WER on
// short utterances from 6.28 to 3.90 % (clean) and 10.82 to 9.89 % (noise) and removed the invented-word bursts
// (measured on short utterances, Nov 2026). The Czech+Slovak+English model bears 384
// (measured against the published model on the same recordings).
#define CS_MIN_AUDIO_CTX 512
#define CS_MIN_AUDIO_CTX_LOWEST 256
#define CS_MIN_AUDIO_CTX_HIGHEST 1500

static int cs_audio_ctx(jint n_samples, int min_frames) {
    int frames = (int) ((n_samples + 319) / 320) + 32;
    frames = (frames + 15) / 16 * 16;
    if (frames < min_frames) frames = min_frames;
    return frames < 1500 ? frames : 0;
}

// The window for n_samples with the floor min_frames (the formula alone: tests and the Kotlin copy for the log).
JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeAudioCtx(JNIEnv *env, jclass clazz, jint n_samples,
                                                                   jint min_frames) {
    (void) env;
    (void) clazz;
    return cs_audio_ctx(n_samples, min_frames);
}

// Sets the floor of the encoder window for every recognition on this context (final text, preview, n-best).
// A multiple of 16 from 256 to 1500 frames; anything else is refused, before the context is looked at.
JNIEXPORT void JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperSetMinAudioCtx(JNIEnv *env, jclass clazz,
                                                                                jlong handle, jint min_frames) {
    (void) clazz;
    if (min_frames < CS_MIN_AUDIO_CTX_LOWEST || min_frames > CS_MIN_AUDIO_CTX_HIGHEST || min_frames % 16 != 0) {
        cs_throw(env, "java/lang/IllegalArgumentException", "minimum encoder window out of range");
        return;
    }
    struct cs_whisper *w = (struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->ctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "whisper context not loaded");
        return;
    }
    w->min_audio_ctx = min_frames;
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
    w->min_audio_ctx = CS_MIN_AUDIO_CTX;
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

// Parses the language list argument of a recognition (see cs_parse_languages) into ids; an invalid list throws
// IllegalArgumentException (a null one NullPointerException) and returns -1. Called before anything is allocated.
static int cs_language_ids(JNIEnv *env, jstring languages, int *ids, int max) {
    if (languages == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "languages");
        return -1;
    }
    const char *list = (*env)->GetStringUTFChars(env, languages, NULL);
    if (list == NULL) {
        return -1;  // OutOfMemoryError pending
    }
    const int n = cs_parse_languages(list, ids, max);
    (*env)->ReleaseStringUTFChars(env, languages, list);
    if (n < 0) {
        cs_throw(env, "java/lang/IllegalArgumentException", "invalid language list");
    }
    return n;
}

// The language rule of one recognition, the same as pickLanguage in LanguageRule.kt (a test compares them on a
// grid). shares: the languages' shares rescaled to sum 1; thresholds: one per language (element 0 is not read), NaN =
// not a candidate, otherwise 0..1; NULL = no rule, the most probable language (the first of equals). Else the
// candidate is the most probable language with a threshold other than the first, and it is chosen when its share
// reaches its threshold; otherwise the first language. Returns the index.
static int cs_pick_language(const float *shares, const float *thresholds, int n) {
    int best = 0;
    if (thresholds == NULL) {
        for (int i = 1; i < n; i++) {
            if (shares[i] > shares[best]) {
                best = i;
            }
        }
        return best;
    }
    int candidate = -1;
    for (int i = 1; i < n; i++) {
        if (!isnan(thresholds[i]) && (candidate < 0 || shares[i] > shares[candidate])) {
            candidate = i;
        }
    }
    return candidate >= 0 && shares[candidate] >= thresholds[candidate] ? candidate : 0;
}

// What the language selection callback of whisper_full needs: the languages of the call (whisper.cpp ids), the
// thresholds (NULL: the most probable) and the context to leave the shares in.
struct cs_language_choice {
    struct cs_whisper *w;
    const int *ids;
    int n;
    const float *thresholds;
};

// whisper_full calls it with the probabilities of all languages once it has detected the language on the window of
// the call. The shares of the call's languages, rescaled to sum 1, go to the context; the rule picks the language.
static int cs_language_select(struct whisper_context *ctx, struct whisper_state *state, const float *probs,
                              void *data) {
    (void) ctx;
    (void) state;
    struct cs_language_choice *choice = (struct cs_language_choice *) data;
    float sum = 0.0f;
    for (int i = 0; i < choice->n; i++) {
        sum += probs[choice->ids[i]];
    }
    for (int i = 0; i < choice->n; i++) {
        choice->w->shares[i] = sum > 0.0f ? probs[choice->ids[i]] / sum : 0.0f;
    }
    choice->w->n_shares = choice->n;
    return choice->ids[cs_pick_language(choice->w->shares, choice->thresholds, choice->n)];
}

// Reads the thresholds argument of a recognition into out (CS_MAX_LANGUAGES floats). Returns 0 for null (no rule),
// n_languages for a valid array. Another length, a value outside NaN and 0..1, or an array with a single language
// throws IllegalArgumentException and returns -1. Called before anything is allocated.
static int cs_thresholds(JNIEnv *env, jfloatArray thresholds, int n_languages, float *out) {
    if (thresholds == NULL) {
        return 0;
    }
    const jsize len = (*env)->GetArrayLength(env, thresholds);
    if (n_languages < 2 || len != n_languages) {
        cs_throw(env, "java/lang/IllegalArgumentException", "thresholds do not fit the language list");
        return -1;
    }
    (*env)->GetFloatArrayRegion(env, thresholds, 0, len, out);
    for (int i = 0; i < n_languages; i++) {
        if (!isnan(out[i]) && !(out[i] >= 0.0f && out[i] <= 1.0f)) {
            cs_throw(env, "java/lang/IllegalArgumentException", "threshold out of range");
            return -1;
        }
    }
    return n_languages;
}

// Rule of one recognition, for tests: the index cs_pick_language gives for these shares and thresholds.
JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativePickLanguage(JNIEnv *env, jclass clazz, jfloatArray shares,
                                                                       jfloatArray thresholds) {
    (void) clazz;
    if (shares == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "shares");
        return -1;
    }
    const jsize n = (*env)->GetArrayLength(env, shares);
    if (n < 1 || n > CS_MAX_LANGUAGES) {
        cs_throw(env, "java/lang/IllegalArgumentException", "shares out of range");
        return -1;
    }
    float th[CS_MAX_LANGUAGES];
    const int n_th = cs_thresholds(env, thresholds, n, th);
    if (n_th < 0) {
        return -1;
    }
    float sh[CS_MAX_LANGUAGES];
    (*env)->GetFloatArrayRegion(env, shares, 0, n, sh);
    return cs_pick_language(sh, n_th > 0 ? th : NULL, n);
}

// The whisper_full parameters of every recognition: greedy decoding with the temperature fallback (variant selection
// switches the final text to beam search; a preview turns the fallback off, see cs_transcribe). The encoder window
// (audio_ctx) is set by the callers, from the context's floor.
static struct whisper_full_params cs_full_params(jint n_threads, jint n_samples) {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads        = n_threads > 0 ? n_threads : 1;
    params.translate        = false;
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
    return params;
}

// Greedy decoding; returns the transcript as UTF-8 bytes (possibly empty), or null when whisper_full failed or was
// aborted. With fallback (the final text without variant selection), whisper.cpp decodes a result that looks wrong
// again at higher temperatures (0.2 steps up to 1.0, best of 5); without (a preview), it decodes once, greedily: audio
// cut mid-word would trigger the fallback often, for text that is replaced anyway. Never beam search, never the
// language model: those are the final text's (nativeWhisperTranscribeNBest).
// languages: comma-separated two- or three-letter lowercase whisper language codes the model speaks and the caller
// allows ("cs", "pl,en"); a malformed or unknown list throws IllegalArgumentException before anything is allocated.
// One code forces that language (no detection, no callback); several: whisper_full detects the language on the
// window of the call and the rule of cs_pick_language picks one of the list (thresholds: one per language, NaN = not a
// candidate, null = the most probable). nativeWhisperLanguage tells the language of the transcript and
// nativeWhisperLanguageShares the shares the pick was made from.
static jbyteArray cs_transcribe(JNIEnv *env, jlong handle, jint generation, jfloatArray samples, jint n_samples,
                                jint n_threads, bool fallback, jstring languages, jfloatArray thresholds) {
    struct cs_whisper *w = (struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->ctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "whisper context not loaded");
        return NULL;
    }
    w->language = NULL;
    w->n_shares = 0;
    int ids[CS_MAX_LANGUAGES];
    const int n_languages = cs_language_ids(env, languages, ids, CS_MAX_LANGUAGES);
    if (n_languages < 0) {
        return NULL;
    }
    float thresholds_buf[CS_MAX_LANGUAGES];
    const int n_thresholds = cs_thresholds(env, thresholds, n_languages, thresholds_buf);
    if (n_thresholds < 0) {
        return NULL;
    }
    bool heap = false;
    float *pcm = cs_copy_samples(env, samples, n_samples, NULL, &heap);
    if (pcm == NULL) {
        return NULL;
    }

    struct whisper_full_params params = cs_full_params(n_threads, n_samples);
    params.audio_ctx = cs_audio_ctx(n_samples, w->min_audio_ctx);
    if (!fallback) {
        params.temperature_inc = 0.0f;
    }

    struct cs_abort abort_data = { w, generation };
    params.abort_callback                   = cs_abort_callback;
    params.abort_callback_user_data         = &abort_data;
    params.encoder_begin_callback           = cs_encoder_begin_callback;
    params.encoder_begin_callback_user_data = &abort_data;

    struct cs_language_choice choice = { w, ids, n_languages, n_thresholds > 0 ? thresholds_buf : NULL };
    if (n_languages > 1) {
        params.language                           = "auto";
        params.language_select_callback           = cs_language_select;
        params.language_select_callback_user_data = &choice;
    } else {
        params.language = whisper_lang_str(ids[0]);
    }

    const int rc = whisper_full(w->ctx, params, pcm, n_samples);
    cs_release_samples(pcm, n_samples, heap);
    if (rc != 0) {
        w->n_shares = 0;
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
        w->n_shares = 0;
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
        w->language = n_languages > 1 ? whisper_lang_str(whisper_full_lang_id(w->ctx)) : params.language;
    } else {
        w->n_shares = 0;
    }
    return out;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperTranscribe(JNIEnv *env, jclass clazz, jlong handle,
                                                                            jint generation, jfloatArray samples,
                                                                            jint n_samples, jint n_threads,
                                                                            jstring languages,
                                                                            jfloatArray thresholds) {
    (void) clazz;
    return cs_transcribe(env, handle, generation, samples, n_samples, n_threads, true, languages, thresholds);
}

// A preview: one greedy pass (cs_transcribe without the fallback), whatever the final text uses.
JNIEXPORT jbyteArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperPreview(JNIEnv *env, jclass clazz, jlong handle,
                                                                         jint generation, jfloatArray samples,
                                                                         jint n_samples, jint n_threads,
                                                                         jstring languages, jfloatArray thresholds) {
    (void) clazz;
    return cs_transcribe(env, handle, generation, samples, n_samples, n_threads, false, languages, thresholds);
}

static size_t cs_put(char *buf, size_t pos, const void *src, size_t n) {
    if (buf != NULL) {
        memcpy(buf + pos, src, n);
    }
    return pos + n;
}

// Serialises the hypotheses of the last whisper_full call (little-endian, read by NBest.kt), into buf, or only counts
// the bytes when buf is NULL. Returns the size:
//   int32 windows; per window: int32 hypotheses; per hypothesis: float64 sum_logprobs, float64 score,
//   float32 temperature, int32 result_len, int32 best (0/1), int32 text bytes, UTF-8 text.
static size_t cs_put_hypotheses(struct whisper_context *ctx, char *buf) {
    const int n_windows = whisper_full_n_hypothesis_windows(ctx);
    size_t pos = cs_put(buf, 0, &n_windows, sizeof(int32_t));
    for (int w = 0; w < n_windows; w++) {
        const int n_hyps = whisper_full_n_hypotheses(ctx, w);
        pos = cs_put(buf, pos, &n_hyps, sizeof(int32_t));
        for (int i = 0; i < n_hyps; i++) {
            const whisper_hypothesis h = whisper_full_get_hypothesis(ctx, w, i);
            const int32_t best = h.best ? 1 : 0;
            int32_t text_len = 0;
            for (int k = 0; k < h.n_tokens; k++) {
                text_len += (int32_t) strlen(whisper_token_to_str(ctx, h.tokens[k]));
            }
            pos = cs_put(buf, pos, &h.sum_logprobs, sizeof(double));
            pos = cs_put(buf, pos, &h.score, sizeof(double));
            pos = cs_put(buf, pos, &h.temperature, sizeof(float));
            pos = cs_put(buf, pos, &h.result_len, sizeof(int32_t));
            pos = cs_put(buf, pos, &best, sizeof(int32_t));
            pos = cs_put(buf, pos, &text_len, sizeof(int32_t));
            for (int k = 0; k < h.n_tokens; k++) {
                const char *piece = whisper_token_to_str(ctx, h.tokens[k]);
                pos = cs_put(buf, pos, piece, strlen(piece));
            }
        }
    }
    return pos;
}

// Beam search with beam_size beams (whisper.cpp patched: the beams take the most likely tokens, see
// native/patches/whisper.cpp-czechspeech.patch) and every beam's hypothesis per window, serialised as in
// cs_put_hypotheses; null when whisper_full failed or was aborted. The buffer holding the text is zeroed.
JNIEXPORT jbyteArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperTranscribeNBest(JNIEnv *env, jclass clazz,
                                                                                 jlong handle, jint generation,
                                                                                 jfloatArray samples, jint n_samples,
                                                                                 jint n_threads, jint beam_size,
                                                                                 jstring languages,
                                                                                 jfloatArray thresholds) {
    (void) clazz;
    struct cs_whisper *w = (struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->ctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "whisper context not loaded");
        return NULL;
    }
    w->language = NULL;
    w->n_shares = 0;
    if (beam_size < 1 || beam_size > 8) {
        cs_throw(env, "java/lang/IllegalArgumentException", "beam size out of range");
        return NULL;
    }
    int ids[CS_MAX_LANGUAGES];
    const int n_languages = cs_language_ids(env, languages, ids, CS_MAX_LANGUAGES);
    if (n_languages < 0) {
        return NULL;
    }
    float thresholds_buf[CS_MAX_LANGUAGES];
    const int n_thresholds = cs_thresholds(env, thresholds, n_languages, thresholds_buf);
    if (n_thresholds < 0) {
        return NULL;
    }
    bool heap = false;
    float *pcm = cs_copy_samples(env, samples, n_samples, NULL, &heap);
    if (pcm == NULL) {
        return NULL;
    }

    struct whisper_full_params params = cs_full_params(n_threads, n_samples);
    params.audio_ctx              = cs_audio_ctx(n_samples, w->min_audio_ctx);
    params.strategy               = WHISPER_SAMPLING_BEAM_SEARCH;
    params.beam_search.beam_size  = beam_size;   // greedy.best_of (5) stays for the temperature fallback

    struct cs_abort abort_data = { w, generation };
    params.abort_callback                   = cs_abort_callback;
    params.abort_callback_user_data         = &abort_data;
    params.encoder_begin_callback           = cs_encoder_begin_callback;
    params.encoder_begin_callback_user_data = &abort_data;

    struct cs_language_choice choice = { w, ids, n_languages, n_thresholds > 0 ? thresholds_buf : NULL };
    if (n_languages > 1) {
        params.language                           = "auto";
        params.language_select_callback           = cs_language_select;
        params.language_select_callback_user_data = &choice;
    } else {
        params.language = whisper_lang_str(ids[0]);
    }

    const int rc = whisper_full(w->ctx, params, pcm, n_samples);
    cs_release_samples(pcm, n_samples, heap);
    if (rc != 0) {
        w->n_shares = 0;
        return NULL;
    }

    const size_t total = cs_put_hypotheses(w->ctx, NULL);
    char *buf = (char *) malloc(total);
    if (buf == NULL) {
        w->n_shares = 0;
        cs_throw(env, "java/lang/OutOfMemoryError", "hypotheses buffer");
        return NULL;
    }
    cs_put_hypotheses(w->ctx, buf);
    jbyteArray out = (*env)->NewByteArray(env, (jsize) total);
    if (out != NULL) {
        (*env)->SetByteArrayRegion(env, out, 0, (jsize) total, (const jbyte *) buf);
    }
    cs_secure_zero(buf, total);
    free(buf);
    if (out != NULL) {
        w->language = n_languages > 1 ? whisper_lang_str(whisper_full_lang_id(w->ctx)) : params.language;
    } else {
        w->n_shares = 0;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Language model (variant selection), czechspeech_lm.c

JNIEXPORT jlong JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeLmInit(JNIEnv *env, jclass clazz, jstring model_path) {
    (void) clazz;
    if (model_path == NULL) {
        cs_throw(env, "java/lang/NullPointerException", "modelPath");
        return 0;
    }
    const char *path = (*env)->GetStringUTFChars(env, model_path, NULL);
    if (path == NULL) {
        return 0;
    }
    struct cs_lm *lm = cs_lm_load(path);
    (*env)->ReleaseStringUTFChars(env, model_path, path);
    return (jlong) (intptr_t) lm;
}

JNIEXPORT void JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeLmFree(JNIEnv *env, jclass clazz, jlong handle) {
    (void) env;
    (void) clazz;
    cs_lm_free((struct cs_lm *) (intptr_t) handle);
}

JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeLmVocabSize(JNIEnv *env, jclass clazz, jlong handle) {
    (void) clazz;
    const struct cs_lm *lm = (const struct cs_lm *) (intptr_t) handle;
    if (lm == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "language model not loaded");
        return 0;
    }
    return cs_lm_n_vocab(lm);
}

JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeLmContextLength(JNIEnv *env, jclass clazz, jlong handle) {
    (void) clazz;
    const struct cs_lm *lm = (const struct cs_lm *) (intptr_t) handle;
    if (lm == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "language model not loaded");
        return 0;
    }
    return cs_lm_n_ctx(lm);
}

// Bytes the language model holds: {weights, compute buffer kept between calls} (tests and diagnostics).
JNIEXPORT jlongArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeLmBufferBytes(JNIEnv *env, jclass clazz, jlong handle) {
    (void) clazz;
    const struct cs_lm *lm = (const struct cs_lm *) (intptr_t) handle;
    if (lm == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "language model not loaded");
        return NULL;
    }
    size_t weights = 0;
    size_t compute = 0;
    cs_lm_buffer_sizes(lm, &weights, &compute);
    const jlong sizes[2] = { (jlong) weights, (jlong) compute };
    jlongArray out = (*env)->NewLongArray(env, 2);
    if (out != NULL) {
        (*env)->SetLongArrayRegion(env, out, 0, 2, sizes);
    }
    return out;
}

// Sum of token log-probabilities for each sequence tokens[offsets[i] .. offsets[i + 1]) (each starting with the start
// token); null on invalid input or a ggml failure.
JNIEXPORT jdoubleArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeLmScore(JNIEnv *env, jclass clazz, jlong handle,
                                                                  jintArray tokens, jintArray offsets, jint n_threads) {
    (void) clazz;
    struct cs_lm *lm = (struct cs_lm *) (intptr_t) handle;
    if (lm == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "language model not loaded");
        return NULL;
    }
    if (tokens == NULL || offsets == NULL) {
        cs_throw(env, "java/lang/NullPointerException", tokens == NULL ? "tokens" : "offsets");
        return NULL;
    }
    const jsize n_tokens = (*env)->GetArrayLength(env, tokens);
    const jsize n_offsets = (*env)->GetArrayLength(env, offsets);
    if (n_offsets < 1) {
        cs_throw(env, "java/lang/IllegalArgumentException", "offsets");
        return NULL;
    }
    const int n_seqs = n_offsets - 1;
    int32_t *tok = (int32_t *) malloc((size_t) (n_tokens > 0 ? n_tokens : 1) * sizeof(int32_t));
    int32_t *off = (int32_t *) malloc((size_t) n_offsets * sizeof(int32_t));
    double *sums = (double *) malloc((size_t) (n_seqs > 0 ? n_seqs : 1) * sizeof(double));
    jdoubleArray out = NULL;
    if (tok == NULL || off == NULL || sums == NULL) {
        cs_throw(env, "java/lang/OutOfMemoryError", "language model buffers");
    } else {
        (*env)->GetIntArrayRegion(env, tokens, 0, n_tokens, (jint *) tok);
        (*env)->GetIntArrayRegion(env, offsets, 0, n_offsets, (jint *) off);
        bool valid = off[0] == 0 && off[n_seqs] <= n_tokens;
        for (int i = 0; valid && i < n_seqs; i++) {
            valid = off[i] <= off[i + 1];
        }
        if (valid && cs_lm_score(lm, tok, off, n_seqs, n_threads, sums)) {
            out = (*env)->NewDoubleArray(env, n_seqs);
            if (out != NULL && n_seqs > 0) {
                (*env)->SetDoubleArrayRegion(env, out, 0, n_seqs, sums);
            }
        }
    }
    if (tok != NULL) {
        cs_secure_zero(tok, (size_t) (n_tokens > 0 ? n_tokens : 0) * sizeof(int32_t));
    }
    free(tok);
    free(off);
    free(sums);
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

// Shares of the languages of the list in the last recognition, rescaled to sum 1 and in the list's order: what the
// language was picked from. Null when the language was forced, or the last call failed, was cancelled or has not run.
JNIEXPORT jfloatArray JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperLanguageShares(JNIEnv *env, jclass clazz,
                                                                                jlong handle) {
    (void) clazz;
    const struct cs_whisper *w = (const struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->n_shares <= 0) {
        return NULL;
    }
    jfloatArray out = (*env)->NewFloatArray(env, w->n_shares);
    if (out != NULL) {
        (*env)->SetFloatArrayRegion(env, out, 0, w->n_shares, w->shares);
    }
    return out;
}

// Encoder runs of this context since it was loaded (tests and logs): a language selection and the first window share
// one.
JNIEXPORT jint JNICALL
Java_ai_moravian_czechspeech_internal_NativeBridge_nativeWhisperEncodeRuns(JNIEnv *env, jclass clazz, jlong handle) {
    (void) clazz;
    const struct cs_whisper *w = (const struct cs_whisper *) (intptr_t) handle;
    if (w == NULL || w->ctx == NULL) {
        cs_throw(env, "java/lang/IllegalStateException", "whisper context not loaded");
        return 0;
    }
    return whisper_full_n_encode(w->ctx);
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

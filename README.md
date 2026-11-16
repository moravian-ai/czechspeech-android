# CzechSpeech for Android

Offline Czech speech-to-text for Android apps. CzechSpeech is Whisper large-v3-turbo fine-tuned on Czech by
[Moravian AI](https://moravian.ai/czechspeech/), quantised to 4 bits (474 MB). It runs on the phone through
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), with [Silero VAD](https://github.com/snakers4/silero-vad)
to find speech.

You add one dependency, call `listen()` and collect Czech transcripts. To stop, cancel the collection. Audio never
leaves the device.

Version 0.2.0, the first public release. See [What has been tested](#what-has-been-tested).

## Install

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

// app/build.gradle.kts
dependencies {
    implementation("ai.moravian:czechspeech-android:0.2.0")
}
```

Requirements: `minSdk` 26, a 64-bit device (arm64-v8a, or the x86_64 emulator), RAM for the 474 MB model plus
whisper.cpp's working buffers while a session runs, and 474 MB of storage for the model.

## Permissions

The SDK's manifest declares **no permissions**. Your app declares:

```xml
<!-- for listen(); request it at runtime before collecting -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<!-- only if you use CzechSpeechModel.download() -->
<uses-permission android:name="android.permission.INTERNET" />
```

If you ship the model another way (for example in your own download flow or an asset pack), you don't need
INTERNET for CzechSpeech at all.

## The model

The model is downloaded once to `noBackupFilesDir/czechspeech/`, which is private to the app and excluded from
backups:

```kotlin
val modelFile: File = CzechSpeechModel.download(context) { done, total ->
    progress = done.toFloat() / total            // called on a background thread
}
```

- **Resumable:** the download goes to a `.part` file, and an interrupted or cancelled download continues on the
  next call with an HTTP Range request.
- **Checked for space first:** before any request, the storage must have room for what is left to download plus
  100 MB, so the model never fills the device. Otherwise `download()` throws an `IOException` saying how much is
  needed and how much is free. The figure is `StorageManager.getAllocatableBytes()`, which also counts cache the
  system can clear, or the usable space; when the device cannot tell, the download goes ahead.
- **Verified:** the size and SHA-256 are checked before the file gets its final name.
- **Idempotent:** a verified file is returned at once, without network or rehashing.
- **Pinned:** it comes from a fixed revision of
  [`moravianai/CzechSpeech`](https://huggingface.co/moravianai/CzechSpeech) on Hugging Face and follows its redirect
  to the CDN. It never follows a redirect from HTTPS to HTTP.
- **Mirrorable:** a company can serve it from its own server. The hash is still enforced:

  ```kotlin
  CzechSpeechModel.download(context, CzechSpeechModel.DEFAULT.copy(url = "https://models.example.com/czechspeech-q4_0.bin"))
  ```

You can also pass any `File` to `CzechSpeech` yourself. `CzechSpeechModel.isDownloaded()`,
`CzechSpeechModel.file()` and `CzechSpeechModel.delete()` cover the rest.

### Models and languages

`CzechSpeechModel.CATALOG` maps a model id to an `Entry(id, source, languages, tokensPerSecond)`: where the file
comes from, which languages it speaks (two- or three-letter lowercase whisper language codes such as `cs`, `pl` or `yue`) and how many tokens per second of
speech it needs (informational on Android: the native layer does not limit the token count). Today the catalog holds
one model, `"cs"`: the pinned Czech model above, languages `["cs"]`, 25 tokens per second.
`CzechSpeechModel.entry(id)` returns an entry, or throws `IllegalArgumentException("unknown model: <id>")`.
`CzechSpeechModel.DEFAULT_ID` (`"cs"`) names the model used when no id is given, and
`CzechSpeechModel.DEFAULT` is that entry's source, unchanged, and a file downloaded by an earlier version is found in
the cache (it is keyed by SHA-256) without a new download.

Give the app the model id: `CzechSpeechModel.download(context, id)` (and `file`, `isDownloaded`, `delete` by id) fetches
the entry's source, and the `CzechSpeech` constructor that takes the entry gets the languages from it, so they are never
passed separately and a model cannot be forced into the wrong language. An unknown id throws
`IllegalArgumentException` at once, before any I/O. `download(context)` without an id is still the Czech model.

```kotlin
val speech = CzechSpeech(context, CzechSpeechModel.download(context, "cs"), CzechSpeechModel.entry("cs"))
```

For a model file outside the catalog, tell `CzechSpeech` its languages yourself with `languages` (default
`listOf("cs")`; a repeated code counts once).

- **One language** (the Czech model): it is forced, exactly as before. Whisper does not detect the language, so there
  is no extra work and the text is the same as in earlier versions.
- **Several languages:** `listen(language)` and `transcribe(samples, language)` force one of them. Without a
  choice, the recogniser detects the language among the list, never outside it, and the transcript says which it
  picked in `Transcript.language`. A detection that returns no language of the list fails that recognition (`SpeechEvent.Error`); no language is made up.
- **A language outside the list** (`listen("de")` on a `["cs"]` model) throws `IllegalArgumentException` at the
  call, not later while collecting. An empty list or a code that is not two or three lowercase letters is rejected when `CzechSpeech` is created.
- **What detection costs:** it runs before every utterance and adds the mel spectrogram, one encoder pass over the
  whole 30 s window and one decoder step to its recognition, whatever the length of the utterance. Forced
  languages run the encoder only over the length of the utterance. Measured on a 6-thread x86-64 laptop, with a 7 s
  Czech utterance: 2.1 s forced, 10.7 s with detection among `cs` and `en` (model load included in both, the
  difference is the detection). Phones are slower; measure on yours before offering detection there. The detection state
  is created for the call and freed again, so it does not stay in memory. Whisper's detection cannot be interrupted,
  so cancelling the collection or `close()` during it takes effect when it returns, and the result is discarded. `stop()`
  never cancels a recognition: a detection under way finishes and its transcript is delivered. Forced languages skip
  all of this.

## Listen

```kotlin
val speech = CzechSpeech(context, modelFile)          // cheap: nothing is loaded yet

val job = lifecycleScope.launch {
    speech.listen().collect { event ->               // opens the microphone, loads the model
        when (event) {
            is SpeechEvent.SpeechStarted -> showListening()
            is SpeechEvent.SpeechEnded -> showRecognising()
            is SpeechEvent.Transcript -> append(event.transcript.text)
            is SpeechEvent.Error -> log("one utterance failed: ${event.cause}")
        }
    }
}
speech.stop()      // finishes: the last utterance is still transcribed, then the flow completes and frees the model
// job.cancel()    // discards instead: closes the microphone at once, speech not yet transcribed is lost
```

`Transcript.language` is the language of the text: the forced one, or the detected one for a model with several
languages (see [Models and languages](#models-and-languages)). `listen()` and `transcripts()` take an optional `language`.

If you only want the text, use `speech.transcripts()`. It returns `Flow<Transcript>` and is the same as
`listen().filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }`.

How a session behaves:

- **The flow is cold.** Collecting starts a session, and you can run one collection at a time per `CzechSpeech`.
- **The model loads with the microphone.** It loads in parallel with opening the microphone, and speech captured
  meanwhile waits in a queue. Loading takes about 0.5 to 3 s from flash.
- **Order of events.** Each `SpeechStarted` is followed by a `SpeechEnded`. The `Transcript` comes after
  `SpeechEnded`, because recognition starts once the utterance is complete. No `Transcript` follows if the speech
  check or the recogniser found no words.
- **`SpeechStarted` can fire on noise.** Background noise can trigger it; the speech check then stops it before
  it produces text.
- **Fatal errors are thrown from the flow:** `SecurityException` without RECORD_AUDIO, and `CzechSpeechException`
  when the microphone, the model or the native library fails. `SpeechEvent.Error` only reports one utterance
  whose recognition failed or that was skipped (next point), and the session continues.
- **Falling behind.** Utterances wait for the recogniser in a queue that holds at most 60 s of audio (about
  3.8 MB). On a phone where recognition is slower than speech, long continuous talk can fill it; an utterance that
  does not fit is then skipped (when nothing is waiting, an utterance is always taken). Its audio is zeroed at once, and a `SpeechEvent.Error` whose
  `CzechSpeechException` says which speech was skipped takes its place. Speech is never dropped without that
  event. The microphone is not paused for the recogniser instead, because audio that is not read is lost without
  any signal.
- **Stopping with `stop()`.** Use this for a Stop button. It finishes the word in progress: if someone is speaking,
  the microphone stays on until a short pause (256 ms), at most 1 s, and that audio is part of the utterance (cut
  off mid-word, Whisper tends to invent an ending); without speech it closes at once. The utterance in progress
  then ends as if silence followed, and everything still waiting is transcribed and emitted. Then the flow
  completes normally and the model is freed (unless you [keep it loaded](#keeping-the-model-loaded)). `stop()`
  returns immediately and does nothing without an active session.
- **Cancelling.** Speech in progress or waiting in the queue is discarded. The microphone closes at the end of the
  frame being read, and the collection completes at once. Native work that whisper.cpp cannot interrupt finishes
  in the background, and the model is freed right after it (with `keepModelLoaded` it stays loaded):
  - a recognition running at that moment stops at whisper.cpp's next check, which comes only between stages, so it
    can take a few seconds for a long utterance;
  - a model load runs to its end.

  A session cancelled before its load started never loads the model.
- **`close()`** returns at once and never blocks, so it is safe on the main thread. Once it has returned:
  - no new session or call starts: `listen()` and `transcribe()` throw
    `IllegalStateException("CzechSpeech is closed")`, also a collection or call that starts while `close()` runs;
  - the active collection is cancelled (the collector sees a `CancellationException`), and every `transcribe()` in
    progress resumes at once with that `IllegalStateException`;
  - the microphone closes at the end of the frame being read;
  - native work already running (a model load, the speech check, an encoder pass) finishes in the background, and
    the model is freed right after, also a model kept with `keepModelLoaded`.

  `awaitClosed()` suspends until that background work is done and the model is freed, for example before you
  create another instance, so that two models are never held at once.
- **Timestamps.** `Transcript.startMs` and `endMs` are milliseconds of audio since the session started.
  `processingMs` is the time spent on recognition.

## Transcribe audio you already have

```kotlin
val text: String = speech.transcribe(samples)   // 16 kHz mono FloatArray in [-1, 1]
```

The same speech check and filter apply, and the result is `""` when there is no speech. If Silero VAD finds no
speech, the model is not even loaded. Otherwise the model is loaded for the call and freed afterwards, unless a
`listen()` session already holds it or `keepModelLoaded` keeps it.

**You own `samples`.** The SDK reads the array and never modifies it, so zero it yourself when you are done. Unlike
`listen()`, `transcribe()` processes the whole buffer in one call; longer inputs run through Whisper's 30 s window.

**Input is checked first.** `transcribe()` throws `IllegalArgumentException`, before any work, for:

- more than 10 minutes of audio (9,600,000 samples): the call holds the array, a native copy and Whisper's mel
  spectrogram at once (about 110 MB at that length), so split longer audio at pauses;
- a NaN or infinite sample: a single NaN can make the voice detector report speech;
- a sample of magnitude above 2. From 256 up, it almost always means 16-bit PCM passed without scaling, and the
  message says to divide by 32768: unscaled, the voice detector finds no speech and the result would be `""` with
  no hint. Below 256, it is float audio that clips, and the message says to clip or normalise it.

Invalid input is rejected rather than repaired, and the message names the first offending sample.

## Keeping the model loaded

By default every `listen()` session and every `transcribe()` call loads the model and frees it when it ends.
Loading takes about 0.5 to 3 s (0.4 to 0.9 s on a Pixel 9a). For dictation, where sessions follow each other, you
can keep the model loaded instead, as with `keepModelLoaded` in the web SDK:

```kotlin
val speech = CzechSpeech(context, modelFile, keepModelLoaded = true)

speech.listen().collect { /* … */ }   // the first session loads the model; later sessions and calls reuse it
speech.releaseModel()                 // frees it, for example after two minutes without dictation
```

- **From the first load until `releaseModel()` or `close()`.** The model is loaded by the first session or call that
  needs it; a `transcribe()` of audio without speech still does not load it. It stays loaded when that session or
  call ends, and also when a session is cancelled: a load still running then completes, and the model is kept.
- **`releaseModel()` frees it once the sessions and calls running at that moment have ended.** When none is running,
  the model is freed at once. Otherwise the last of them frees it when it ends: a session that ends normally
  (`stop()`, the end of the audio) before its flow completes, a `transcribe()` call before it returns, and a
  cancelled session or call right after, in the background.
  - A session or call that starts after `releaseModel()` keeps the model again. If the model has been freed by then,
    it loads it again; if the release is still waiting for the work that was running, the release is dropped and the
    model stays loaded.
  - `releaseModel()` returns at once and never blocks, so it is safe on the main thread. The model is freed on the
    SDK's recognition thread, after native work already running there, such as the model load of a cancelled
    session.
  - It does nothing without `keepModelLoaded`, when no model is loaded and nothing is running, and after `close()`.
- **`close()` frees it**, like everything else; `awaitClosed()` waits for that. `close()` ends the instance;
  `releaseModel()` only frees the model, and the instance stays usable.
- **One instance per process.** Each instance with `keepModelLoaded` keeps a model of its own, so two instances hold
  two models (about 1 GB). Share one `CzechSpeech`, for example from your `Application` or a singleton.
- **The price is memory.** The model (474 MB) and whisper.cpp's working buffers stay allocated between sessions.
  Android moves memory that nobody touches to zram, and a model swapped out there makes the next recognition take
  up to a minute. Call `releaseModel()` when no speech is expected for a while, and when the system asks for memory:

  ```kotlin
  override fun onTrimMemory(level: Int) {   // in your Application, Service or Activity (ComponentCallbacks2)
      super.onTrimMemory(level)
      speech.releaseModel()                 // during a session, the model is freed when it ends
  }
  ```
- **Privacy.** whisper.cpp's buffers derived from the last audio, the mel spectrogram and the encoder activations,
  also stay in RAM until the model is freed (see [Privacy](#privacy)).

## Configuration

```kotlin
CzechSpeech(context, modelFile, CzechSpeechConfig(minSilenceMs = 800))
```

| field | default | meaning |
|---|---|---|
| `threshold` | 0.25 | speech starts at a frame with Silero probability ≥ this; also the threshold of the speech check before Whisper |
| `negThreshold` | 0.10 | speech continues while probability ≥ this |
| `minSilenceMs` | 600 | the utterance ends after this much audio below `negThreshold` |
| `preRollMs` | 300 | audio before the start of speech that is passed to the recogniser |
| `minSpeechMs` | 250 | utterances with less audio ≥ `threshold` are dropped (clicks, coughs) |
| `maxUtteranceMs` | 28 000 | longer speech is cut at the quietest 32 ms frame of the last 2 s and continues as the next utterance |
| `threads` | 0 | recogniser threads; 0 = number of big CPU cores (2 to 8) |

**Why the threshold is 0.25 and not Silero's usual 0.5.** We tested both on 2,181 recordings with speech. At 0.25
no recording was dropped. At 0.5, 5 of the 729 recordings with reverberation and babble noise were dropped. Every
synthetic non-speech signal was still rejected at 0.25: silence, white, pink and brown noise, 50 Hz hum, clicks,
bursts and beeps.

## Privacy

What the SDK guarantees:

- **No audio on disk:** audio is never written to disk. There is no temporary WAV and no cache file.
- **No network calls:** the library contains no network code except `CzechSpeechModel.download`, which only
  downloads the model. Its manifest declares no permissions, so it cannot add INTERNET to your app.
- **No telemetry:** there is no analytics, no crash reporting and no usage pings.
- **Audio is zeroed:** every SDK buffer that held audio is filled with zeros as soon as it is no longer needed.
  That covers the capture frame, the pre-roll ring, the utterance buffer and queued utterances (including those
  discarded on cancel). In native code, audio is copied into a `malloc`'d buffer, which is zeroed and freed when
  the call returns. The Java arrays are never pinned.
- **No transcript text in logs:** logcat (tag `CzechSpeech`) only shows timings, counts and states.
  `Transcript.toString()` does not print the text either. whisper.cpp's own debug output, which includes decoded
  tokens, is suppressed; only its warnings and errors are passed through.
- **The model is held only while needed:** it is loaded only during `listen()` sessions and `transcribe()` calls,
  then freed, unless you choose `keepModelLoaded`, which keeps it until `releaseModel()` or `close()`. An idle 0.5 GB
  model would be swapped to zram by Android, and the next transcription would then take up to a minute.
- **No emotion or tone analysis:** the SDK does not analyse emotion, tone of voice or speaker identity, only the
  words.

**Limit.** whisper.cpp keeps internal working buffers derived from the audio while the model is loaded: the mel
spectrogram and encoder activations. They stay in RAM and are released when the model is freed: at the end of the
session, or with `keepModelLoaded` at `releaseModel()` or `close()`. The SDK does not zero them, because that
would mean patching whisper.cpp.

## Devices and CPU variants

The AAR contains the native library built several times, each for a different instruction-set level. At runtime
the SDK picks the best one that the CPU supports, according to the `Features` line of `/proc/cpuinfo`. If loading
a library fails, it falls back to the next one down. A library built for instructions the CPU lacks would crash
with SIGILL, which is why there is no single "fast" build.

| library | ABI | built with | picked when the CPU has |
|---|---|---|---|
| `libczechspeech_armv82_dotprod_fp16_i8mm.so` | arm64-v8a | `-march=armv8.2-a+dotprod+fp16+i8mm` | `asimddp`, `asimdhp`, `fphp` and `i8mm` (e.g. Cortex-X2/A710/A510 generation and newer: Snapdragon 8 Gen 1+, Tensor G3+) |
| `libczechspeech_armv82_dotprod_fp16.so` | arm64-v8a | `-march=armv8.2-a+dotprod+fp16` | `asimddp`, `asimdhp` and `fphp` (Cortex-A55/A75/A76/A77/A78/X1 class) |
| `libczechspeech_armv8a.so` | arm64-v8a | `-march=armv8-a` | anything else (Cortex-A53/A73 class) |
| `libczechspeech_x86_64.so` | x86_64 | x86-64 + SSE4.2 | the Android emulator |

The build script checks every library: no dotprod, i8mm or fp16 arithmetic instructions in the baseline, no i8mm
in the dotprod build, and no AVX in the x86_64 build. Every library has 16 KB-aligned LOAD segments, which Google
Play requires for Android 15+ devices.

32-bit devices (armeabi-v7a, x86) are not supported; `listen()` and `transcribe()` throw `CzechSpeechException`
there.

**Size.** The AAR is 3.8 MB. In an APK the native libraries are stored uncompressed: about 1.65 MB per arm64
variant (5 MB for all three) plus 2 MB for x86_64. App bundles deliver only the device's ABI. Exclude x86_64 from
release builds if you don't need the emulator.

## Why whole utterances and not words as you speak

Whisper is not a streaming model. It transcribes a finished piece of audio. Showing partial results would mean
running the recogniser again and again on the growing buffer while the person speaks, which costs a lot of battery
on a phone for text that is then replaced. So CzechSpeech waits for the end of an utterance (600 ms of silence)
and transcribes it once. **Partial results are a non-goal for 0.2.**

## The hallucination gate

On silence and noise, Whisper models invent text. Without a gate, this model "transcribed" all 66 of 66 test
signals without speech, producing things like "Titulky vytvořil JohnnyX", "," or "... ...". CzechSpeech therefore
applies two checks, in `listen()` and in `transcribe()` alike:

1. **Before Whisper:** the utterance must contain at least one speech segment according to Silero VAD
   (`whisper_vad_segments_from_samples`, threshold 0.25). If it doesn't, Whisper does not run.
2. **After Whisper:** a result with no letter or digit is dropped.
   A model that has learned to answer a segment without speech with an empty transcript can do so: the decoder
   is not forbidden to end the text as its first token (`suppress_blank` is off). An empty result is dropped without
   an event or an error, and is not decoded again.

With these checks no test signal without speech produced text, and no recording with speech was lost (see the
threshold note above). The JVM integration tests repeat this on recorded silence, pink noise, mains hum, clicks
and 10 s of digital zeros.

**Repetition loops:** audio that ends mid-word (`stop()`, the length cap) can make Whisper repeat a phrase until its
token limit, so a phrase of 2 to 12 words repeated three or more times in a row is kept once, and a cut-off copy at
the end is dropped.

## Building from source

```bash
git clone --recurse-submodules <this repository>
native/build-native.sh            # Linux or WSL: NDK r27c, CMake 3.22; builds all variants + the host library
./gradlew :czechspeech:assembleRelease :sample:assembleRelease
./gradlew :czechspeech:testReleaseUnitTest
./gradlew :czechspeech:publishToLocalFolder      # static Maven repository in build/maven-repo
```

- **Toolchain:** JDK 17 and the Android SDK with platform 36 and build tools 36.0.0. The build uses the Android
  Gradle plugin 8.9.3, Gradle 8.14.4 (the wrapper) and Kotlin 2.0.21; `compileSdk` is 36, `minSdk` 26.
- **whisper.cpp:** pinned as the submodule `third_party/whisper.cpp` at commit `482279b`.
- **Native libraries:** prebuilt by `native/build-native.sh` into `native/out/` and not committed. The AAR build
  fails if they are missing. The last build (15 Nov 2026) used NDK r27c (27.2.12479018), CMake 3.22.1 and whisper.cpp
  `482279b8a18855ba741d7b88c30fcb3874691f59`; the SHA-256 of each library is in the CHANGELOG.
- **JVM tests:** they use the host build `native/out/host/linux-x86_64/libczechspeech_host.so`. The integration
  tests read their inputs from environment variables only and skip, naming the variable, when one is missing:
  - `CZECHSPEECH_MODEL`: a CzechSpeech `.bin` (whisper.cpp) model.
  - `CZECHSPEECH_TEST_AUDIO`: a folder of 16 kHz mono WAV files: `a-silence-5s`, `b-pink-noise-2s`, `c-hum-2s` and
    `d-clicks-6s` (synthetic silence, pink noise, mains hum, clicks), `e-rec` (ParCzech4Speech segment
    `test_0000-0000000333`, CC BY 4.0) and `f-rec-plus-silence15s` (`e-rec` followed by 15 s of digital silence).
  - `CZECHSPEECH_TEST_MANIFEST`: the ParCzech4Speech manifest (`.jsonl`) with the reference transcript of `e-rec`.
  - `CZECHSPEECH_TEST_THREADS`: whisper threads for the tests (default 6).

  Never commit models or audio; `.gitignore` blocks them.
- **Build output:** `CZECHSPEECH_BUILD_ROOT` moves Gradle's build directories elsewhere, for example off `/mnt/c`
  when building in WSL.

The `sample` module is a one-screen app with Start/Stop and a list of transcripts. It uses a `.bin` model pushed to
`/sdcard/Android/data/ai.moravian.czechspeech.sample/files/models/`, or downloads one.

## Publishing to Maven Central

The release goes to the Central Portal (central.sonatype.com) under the verified namespace `ai.moravian`. The
script builds and signs the bundle, and with `--upload` sends it; it never publishes. You press Publish in the
portal. Run it in WSL or Linux (bash, python3, gpg, curl).

**Once:**

- Give the signing key a passphrase if it has none: `gpg --passwd E40DE78E91BEDC18`.
- Put the Central Portal user token (generated under your account in the portal) in `~/.m2/settings.xml`, or point
  `SETTINGS_XML` at another file, as a server with the id `central`:

  ```xml
  <server><id>central</id><username>TOKEN-USERNAME</username><password>TOKEN-PASSWORD</password></server>
  ```

  The script stops with a message naming what is missing or still a `${...}` placeholder. It never prints the
  token, and gives it to curl through a private temporary file.

**Each release:**

1. Set `version` in `czechspeech/build.gradle.kts` to a release version (no `-SNAPSHOT`), give it a heading in
   `CHANGELOG.md`, and commit everything: the script refuses a dirty tree.
2. `scripts/release-maven-central.sh` builds the signed bundle in a fresh temporary directory, checks that every
   artifact has `.asc`, `.md5` and `.sha1` files and that every signature verifies, and prints the zip path and its
   SHA-256. Look at it before going on.
3. `scripts/release-maven-central.sh --upload` does the same, uploads the bundle as `USER_MANAGED` and waits until
   Central reports `VALIDATED` or `FAILED`, with the errors if it failed.
4. Open <https://central.sonatype.com/publishing/deployments>, check the deployment and click Publish.

`GPG_KEY_NAME` overrides the key. The signing in Gradle is off unless `-Pczechspeech.sign=true`, so ordinary
builds need no gpg. The javadoc jar Central requires is generated by AGP from the Kotlin sources.

**A published version can never be deleted or replaced.** Check the bundle first; if something is wrong after
publishing, the fix is a new version number. `scripts/test-release-maven-central.sh` tests the script's guards with
a throwaway key and a stub curl; it touches neither the network nor your keyring.

## What has been tested

Tested so far:

- **Unit tests:** endpointer, CPU detection, download (resume, redirect, hash) and the pipeline lifecycle with a
  fake native layer.
- **Integration tests on the JVM:** the real whisper.cpp, Silero VAD and CzechSpeech model on Linux x86-64.
- **Build checks:** the Android libraries pass the alignment and instruction-set checks above.

- **On a phone (5 Nov 2026):** the sample app on a Pixel 9a (Tensor G4), live microphone, Czech speech.
  - The SDK picked `libczechspeech_armv82_dotprod_fp16_i8mm.so`, with 4 recogniser threads.
  - The model loaded in 0.4–0.9 s.
  - Utterances of 2.2–13.5 s were transcribed in 0.8–5.1 s, that is 0.32–0.38× real time.
  - Speech detection took 1.5 ms per 32 ms frame.
  - The transcripts were checked by ear only, without a reference, so there is no WER.
  - `stop()` in the middle of speech: the word in progress was finished, and the last utterance was transcribed.

**Not yet tested:** the emulator, other phones and CPU variants, long sessions, and `keepModelLoaded` on a phone
(only with the fake native layer).

## Enterprise edition

This SDK and the free Czech model are all you need for Czech. For companies, Moravian AI also offers an enterprise
edition on request:

- **CzechSpeech Enterprise:** one model for Czech, Slovak and English. Its Czech word error rate is equal or lower than
  the free model's on our three Czech test sets; Slovak FLEURS 6.5 %, English FLEURS 6.6 %. Force the language per
  call (see [Models and languages](#models-and-languages)).
- **CzechSpeech Voiceprint:** speaker recognition (who said what), fine-tuned on Czech speech. Support in this SDK is
  in preparation.
- **Fine-tuning on your vocabulary and your recordings,** with the accuracy measured on your own data before you
  decide.
- **Help with integration,** e-mail support and model updates.

It uses this same public SDK. An enterprise model is a file that you host yourself or ship in your app, and you pass
it with its SHA-256 as in [The model](#the-model). There is no licence server and no telemetry: audio and transcripts
stay on the device, as with the free model.

Write to info@moravian.ai.

*Česky:* toto SDK s volným českým modelem stačí na češtinu. Firmám nabízí Moravian AI na vyžádání podnikovou verzi:
- jeden model pro češtinu, slovenštinu a angličtinu (čeština má na našich třech českých testech stejnou nebo nižší
  chybovost než volný model);
- rozpoznávání řečníků doladěné na češtině (podporu v SDK připravujeme);
- doladění na váš slovník a vaše nahrávky s přesností změřenou na vašich datech;
- pomoc s nasazením.

Funguje se stejným veřejným SDK: model hostujete sami a předáte ho s kontrolním součtem SHA-256. Žádný licenční server
ani telemetrie, zvuk zůstává v zařízení. Napište na info@moravian.ai.

## Licence

**Dual licence: free for noncommercial use (PolyForm Noncommercial 1.0.0); commercial licence: info@moravian.ai**

- **This SDK:** © 2026 Jan Rosecký (Moravian AI), under the dual licence above. See [LICENSE](LICENSE).
- **The model weights:** licensed separately (free for non-commercial use under CC BY-NC 4.0; commercial licence:
  info@moravian.ai), on
  [Hugging Face](https://huggingface.co/moravianai/CzechSpeech). The base model, Whisper large-v3-turbo, is © 2022
  OpenAI, MIT licence.
- **Bundled third-party components:** whisper.cpp and ggml (MIT, © The ggml authors), the Silero VAD v6.2.0
  model (MIT, © Silero Team) and the Android NDK's LLVM C++ runtime linked into the native libraries (Apache 2.0
  with LLVM exceptions). See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES); a copy of both files is inside the AAR
  under `META-INF/czechspeech-android/`.

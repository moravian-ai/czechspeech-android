# CzechSpeech for Android

Offline Czech speech-to-text for Android apps. CzechSpeech is Whisper large-v3-turbo fine-tuned on Czech by
[Moravian AI](https://moravian.ai/czechspeech/), quantised to 4 bits (474 MB). It runs on the phone through
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), with [Silero VAD](https://github.com/snakers4/silero-vad)
to find speech.

You add one dependency, call `listen()` and collect Czech transcripts. To stop, cancel the collection. Audio never
leaves the device.

Version 0.3.0, the first public release. See [What has been tested](#what-has-been-tested).

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
    implementation("ai.moravian:czechspeech-android:0.3.0")
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

`CzechSpeechModel.CATALOG` maps a model id to an `Entry(id, source, languages, tokensPerSecond, languageSwitch = null)`: where the file
comes from, which languages it speaks (two- or three-letter lowercase whisper language codes such as `cs`, `pl` or `yue`), how many tokens per second of
speech it needs (informational on Android: the native layer does not limit the token count) and, for a model of several
languages, the rule that lets it leave its first language (`languageSwitch`, below). Today the catalog holds
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
`listOf("cs")`; a repeated code counts once) and, for several, its `languageSwitch`.

- **One language** (the Czech model): it is forced, exactly as before. Whisper does not recognise the language, so there
  is no extra work and the text is the same as in earlier versions.
- **Several languages:** `listen(language)`, `listenWithPreviews(previews, language)` and `transcribe(samples, language)`
  force one of them (one code: no recognition of the language, no rescue; this is also how an app offers a language the
  rule never chooses by itself, such as Slovak). Without a choice, the recogniser decides the language among the list,
  never outside it, and the transcript says which it picked in `Transcript.language`. A result with no language of the
  list fails that recognition (`SpeechEvent.Error`); no language is made up.
- **A language outside the list** (`listen("de")` on a `["cs"]` model) throws `IllegalArgumentException` at the
  call, not later while collecting. An empty list or a code that is not two or three lowercase letters is rejected when `CzechSpeech` is created.
- **Without a `languageSwitch`**, a model of several languages takes the most probable language of the list for every
  recognition on its own: no state between utterances, no rescue.
- **With a `languageSwitch`** the first language of the list is preferred.
  `LanguageSwitch(thresholds = mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)` means:
  - another language is chosen only when its share of the recogniser's probability among the languages of the list reaches
    its threshold (0.5 up to but not including 1), and only for audio of at least `minLeaveMs`; shorter audio stays in the
    first language;
  - a language without a threshold is never chosen automatically, only when the app asks for it (`listen("sk")`);
  - the language of a `listen()` collection is kept: it starts in the first language and every transcript sent sets the
    language the next utterance is expected in (an empty or dropped result, a preview and a rescue do not). In another
    language, audio shorter than `minReturnMs` stays in it; from `minReturnMs` the threshold has to be reached again or the
    collection returns to the first language (so a short answer in Czech after English returns, and a short English one after
    Czech does not leave). `transcribe()` keeps no language between calls;
  - the numbers belong to one model file, measured on it: the ones above are placeholders until they are measured on the
    file you ship;
  - checked when the entry or `CzechSpeech` is created: the model needs at least two languages and the thresholds name
    only languages other than the first.
- **Rescue:** when a final recognition comes out empty and another language the rule allows (the first, or one with a
  threshold) has at least half of the shares, one more run is made forced in that language on the same encoder window; it
  goes before the repeats on a longer window (`retryEmptyFinals`), which then run in the language of the last attempt, and it does not depend on that
  option. It runs for `listen()` and `transcribe()`, not for early start, previews or streaming passes, and the language of
  the collection stays as it was. Slovak is never a rescue language unless it has a threshold.
- **What the decision costs:** the recogniser decides inside the recognition itself, on the (shortened) window of the
  utterance and with the encoder output the decoding then reuses, so it adds one step of the decoder (milliseconds), not
  a pass over the whole 30 s window, and it stops at a cancellation like any recognition (cancelling the collection or `close()`
  discards the result; `stop()` never cancels a recognition, the one under way is delivered). The cost on a phone is not
  measured yet; measure on yours. Variant selection (`variantSelection`) and sentence joining use their language model in the
  first language only; in another language Whisper's own choice stays and the text is not joined with the one before.
- **With previews:** a preview runs in the language of the collection and never changes it.
- **With streaming:** every recognition of a session follows the same rule. A piece of speech starts in the language of the
  collection and keeps it until two passes in a row recognise another one. A pass commits words only when it and the pass before
  ran in the same language and the piece has had a pass long enough to decide the language (`minLeaveMs` in the first
  language, `minReturnMs` in another), so a short pass over foreign speech forced into the wrong language writes nothing. The
  final recognition of a piece that committed words is forced in the piece's language, and at its end the collection takes
  the language of the piece's last text. A larger `minLeaveMs` delays the first word of a piece by about a second per extra
  second. A `Transcript` sent while speaking carries the language of the recognition its words come from.

### Your own or a licensed model

The public catalogue holds the free Czech model only. A model that is not in it, for example a model of several languages
that a company licensed from Moravian AI (the Czech, Slovak and English enterprise model is gated on Hugging Face and
its files go only to companies with a licence), is a file you host yourself: on your server, your CDN or in your
app. You pass its URL and SHA-256 to the SDK, which downloads the file and refuses it when the size or the hash does not
match. No licence server is involved and nothing is sent to us.

```kotlin
val entry = CzechSpeechModel.Entry(
    id = "my-model",
    source = CzechSpeechModel.Source(
        url = "https://models.example.com/my-model.bin",
        sizeBytes = 474_000_000L,                 // the size of your file
        sha256 = "<64 hex digits of your file>",
        fileName = "my-model.bin",                   // its own name: the default is the free model's file
    ),
    languages = listOf("cs", "sk", "en"),         // first = the default language
    tokensPerSecond = 25,
    languageSwitch = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f)),   // numbers measured on your file; null keeps the most probable language
)
val file = CzechSpeechModel.download(context, entry.source)
val speech = CzechSpeech(context, file, entry)
```

The entry does not have to be in `CzechSpeechModel.CATALOG`: `CzechSpeech` takes the languages and the `languageSwitch`
from the entry you pass, exactly as it does for a catalogue entry (see "Models and languages" above for the rule and
for forcing a language per call). The enterprise model and its licence are available on request at info@moravian.ai.

#### Encrypting the hosted file

If the file sits at a public URL that you do not want to advertise, host it encrypted so that a found URL is useless
without the key. Pass the key with the source; the SDK decrypts while it downloads (chunk by chunk, resume with `Range`
works, memory stays constant) and the file on the device is the plain model:

```kotlin
val source = CzechSpeechModel.Source(
    url = "https://models.example.com/my-model.bin.enc",
    sizeBytes = 474_000_000L,                       // the size of the file (encrypted = plain size)
    sha256 = "<sha256 of the PLAIN model>",         // verified after decryption, as always
    fileName = "my-model.bin",                   // its own name: the default is the free model's file
    encryption = CzechSpeechModel.Encryption.aes256Ctr(keyHex = "<64 hex digits>", ivHex = "<32 hex digits>"),
)
```

The format is AES-256-CTR over the whole file: the ciphertext is as long as the plaintext, the 16-byte IV is the initial
counter block, incremented as one 128-bit big-endian integer per 16-byte block. There is no MAC; the plaintext SHA-256
is the integrity check, and a wrong key or a damaged file is a "SHA-256 mismatch" that deletes the download. To encrypt
with OpenSSL, keeping the key and the IV in a file only you can read:

```
key=$(openssl rand -hex 32); iv=$(openssl rand -hex 16)
openssl enc -aes-256-ctr -K "$key" -iv "$iv" -in model.bin -out model.bin.enc
(umask 077; printf 'key=%s\niv=%s\n' "$key" "$iv" > model.key)
```

`scripts/encrypt-model.sh [options] <in|-> <out|-> <keyfile>` does this: it draws a fresh key and IV, writes the key file with
mode 600 (never over an existing one), leaves no partial output, prints the two hashes and the size, and streams
(`-` for stdin and stdout, so a model can go from storage to storage without a local copy). **Give it the known hash and
size of the model** (`--expect-sha256 <hash> --expect-size <bytes>`, from the catalogue, Hugging Face or your build) and
use `set -o pipefail` in a pipeline: an input that breaks off looks like a shorter file, and the hash the script prints
for it is not the model's. `Source.sha256` and `sizeBytes` come from the known hash; the printed one only confirms it.
Run it on a machine you do not share, as OpenSSL takes the key on its command line. `CzechSpeechModel.Encryption` and
`Source.toString()` never show the key.

**Never encrypt two different files with the same key and IV** (a new version of a model gets a new key, or at least a new
IV): the two ciphertexts would reveal the XOR of the two plaintexts. The key and the IV belong to the hosted file: publish
a re-encrypted file under a new URL, and never replace the file behind a URL that released apps use.

**What this protects, exactly:** the file on your server or CDN. Someone who finds the URL gets noise. It does not protect
the model on the device: the decrypted model lives in the app's private storage, and the key is part of your app, so
anyone who can take your APK apart, or who has a rooted device, can get both. It is access control for hosting, not
copy protection.

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

`Transcript.language` is the language of the text: the forced one, or the one the recogniser chose for a model with several
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

## Previews while speaking

Optional. In long continuous speech the first text otherwise comes only after a pause, up to 28 s in. With
`listenWithPreviews()` the utterance in progress is also recognised every few seconds, and its text so far comes
as a `SpeechPreview` between the usual events:

```kotlin
speech.listenWithPreviews().collect { event ->         // PreviewConfig(firstAfterMs = 5_000, intervalMs = 5_000)
    when (event) {
        is SpeechPreview -> showGrey(event.startMs, event.text)        // "" withdraws it
        is SpeechEvent.Transcript -> commit(event.transcript)          // replaces the preview with the same startMs
        is SpeechEvent -> Unit                                         // started, ended, error: as with listen()
    }
}
```

**A preview is unstable text.** Show it as provisional, in grey or as composing text, and never insert it as
finished text. Its words can change in the next preview and in the final text. The final text is recognised from the
whole utterance exactly as before, so it is as accurate as without previews.

- **Replaced or withdrawn.** The `Transcript` with the same `startMs` replaces the preview. If the utterance gives no
  transcript (no words, a failed or skipped recognition), a `SpeechPreview` with the same `startMs` and empty text
  withdraws it. A preview with text never comes after the end of its speech or after the transcript that replaces it.
- **When.** The first preview of an utterance comes once it has lasted 5 s, the next ones at the earliest 5 s of audio
  after the previous one started, one at a time. `PreviewConfig` changes both times.
- **Final text first.** No preview starts while an utterance waits for its final recognition or is in it. A preview
  still running when its utterance ends is aborted and its result is dropped. The abort stops whisper.cpp between
  two steps of its computation, also inside the encoder (`native/patches/whisper.cpp-encoder-abort.patch`; unpatched,
  whisper.cpp only looked after the encoder pass, and the final text waited 2 to 4 s behind a dropped preview on a
  phone). How long each abort took is in logcat (`stopped N ms after the abort`, and `aborts: N, the slowest stopped
  the recogniser after M ms` at the end of a session that had any); that is the most a preview holds up the final
  text. No preview
  starts in a pause (which may be the end), nor when, judged by the speed of the last preview, it would still run
  when the utterance reaches the 28 s cap.
- **The stable part.** A preview's audio ends wherever the speaker is, often inside a word, so its last words are the
  least reliable. `SpeechPreview.stableText` is the start of `text` that the previous preview of the same utterance
  also started with (the same words, whatever their case and punctuation: LocalAgreement-2 of whisper-streaming);
  `unstableText` is the rest, and `text` is both, as before. Show the uncertain end lighter if you like. The stable
  part is still a preview: nothing is final before the `Transcript`.
- **Continuously.** `PreviewConfig.continuous()` previews as densely as the recogniser allows: the first after 2 s,
  the next as soon as the previous one has ended and at least 1 s of new audio has come. The text then grows about
  once per pass (2 to 3.5 s on a Pixel 9a) also in speech without any pause, and the recogniser works nearly all the
  time someone speaks: expect several times the battery and heat of the default. Not yet measured on a phone.
- **The cost.** Each preview recognises the whole utterance so far, in one greedy pass: without the speech check and
  without whisper.cpp's temperature fallback, which the final text keeps (audio cut mid-word would trigger it often,
  and the cost of a preview would swing with it). That holds with [variant selection](#variant-selection-optional-off-by-default)
  on too: a preview never runs beam search or the language model, only the final text does. With the
  defaults, a 28 s utterance gets up to 5 previews over about 75 s of audio, 2.7 times its own, on top of its final
  recognition. Shorter utterances cost less: one of 10 s gets a single preview of 5 s. This is recogniser time and
  battery; previews are off unless you call `listenWithPreviews()`.
- **Stopping.** After `stop()` no preview starts, and one that is running is aborted at once, not when the word in
  progress ends (up to 1 s later): the final text of that utterance comes next. Every preview shown is then replaced
  or withdrawn before the flow completes. Cancelling the collection or `close()` aborts a running preview; clear what
  you show yourself.
- **Privacy.** The same rules as for transcripts: the audio of a preview is a copy in memory, zeroed after use; preview
  text is never logged, and `SpeechPreview.toString()` does not print it.

`listen()` itself has not changed: without `listenWithPreviews()` nothing of this runs. Previews come as a separate
call, so that an exhaustive `when` over `SpeechEvent` in existing code needs no new branch.

## Early start of the final recognition (optional, off by default)

An utterance ends after 600 ms of silence, and only then is it recognised: the text comes 0.6 s plus the recognition
after the last word. With `CzechSpeechConfig(earlyStart = EarlyStart())` the final recognition starts once the pause
has lasted 256 ms (`EarlyStart.silenceMs`): the same recognition, with everything the configuration asks for.

- **The pause becomes the end:** that recognition gives the `Transcript`, up to 350 ms sooner (minSilenceMs −
  silenceMs, in 32 ms frames: 608 − 256).
- **The speech goes on:** the recognition is aborted and dropped; nothing of it reaches the app, the sentence joiner
  never hears of it, its audio is zeroed, and the utterance is recognised at its end as usual. In a
  `listenWithPreviews()` session it is not aborted but finishes and comes as a `SpeechPreview`
  (`PreviewConfig.fromPauses`, on by default): it was started anyway, it is a full recognition, and its audio ends at
  a pause instead of inside a word. It gives way as soon as the next pause comes, or the utterance ends, before it has
  finished: the recognition that may give the final text starts at once and never waits behind a preview. Timed
  previews then count from the pause, so they come only when no pause has come for `intervalMs`.
- **One recognition at a time**, on the one whisper thread, as before. Nothing is started early while another
  utterance still waits in the queue.
- **The audio of the pause is zeros.** At the early start the rest of the pause has not been recorded, so the
  recogniser gets zeros in its place, up to the usual length. Every utterance ended by silence gets them while early
  start is on, whether its early recognition was used or had to be repeated: what the recogniser gets depends on the
  audio alone, never on timing. It is not what it gets without early start, where the last ~350 ms are the
  microphone's own silence, so **the text can differ from the text without early start**. Whisper pads every
  utterance with zeros up to its window anyway, so the difference is where the zeros begin; how often it changes a
  word has not been measured.
- **The cost** is the recognitions dropped: every pause between 256 and 600 ms inside an utterance starts one. Logcat
  counts them per session: `early start: N used, M ms saved, L too late; P shown as a preview; D discarded, A aborted,
  T ms of recogniser time`, and each `utterance:` line says `early start used, N ms saved` when its text came from
  one. "Too late" are utterances whose recognition began only after they had ended (the recogniser was still busy
  with the one before): recognised as without early start, nothing gained.
- **With variant selection,** an early recognition dropped while the language model scores a window finishes that
  one run (it cannot be interrupted) and scores no further window; the final text waits that long. How long one run
  takes on a phone has not been measured (the `utterance:` line gives `language model N ms in R run(s)`).
- **`silenceMs` and `minSilenceMs` are counted in 32 ms frames, rounded up.** `CzechSpeechConfig` throws
  `IllegalArgumentException` when the early start would fall on the frame that ends the utterance (with the default
  600 ms, `silenceMs` above 576): nothing could ever start early, and a silent no-op would hide that.
- **Off by default** until both have been measured on phones: the change of the text, and the battery.

## Streaming transcription (optional, off by default)

With `CzechSpeechConfig(streaming = StreamingConfig())` the text of words is sent while the person is still speaking,
in continuous speech, instead of after the pause that ends the utterance. Documented in full in the KDoc of
`StreamingConfig`; in short:

- **How:** while speech is in progress the SDK recognises its not yet committed audio again and again, one pass at a
  time (the first one second after the speech started, each next one when the previous has finished and a second of
  audio has come). Words on which two consecutive passes agree exactly (letters, case and punctuation) are committed
  and sent at once as a `SpeechEvent.Transcript`. When the uncommitted audio is longer than 6 s, its start moves to the
  quietest place before committed words, so passes stay short. When the speech ends, the final recognition covers only
  the audio after the committed words.
- **Events:** the text of one utterance can come in several `Transcript`s, the first ones before its `SpeechEnded`,
  the last ones after it (a chunk found just before the end is sent when the recogniser has it, and the final
  recognition adds the rest); concatenate them with one space. A chunk is final (never changed or withdrawn), chunks of
  an utterance share its `startMs`, `endMs` never decreases and never passes the utterance's end (the last chunk ends
  there when the final recognition added words). When everything was committed while speaking, the final recognition
  adds no `Transcript`. In `listenWithPreviews()` a `SpeechPreview`
  carries only the words after the last chunk (`stableLength` 0, all of it uncertain: these are the words the next `Transcript`
  will most likely send, and any `Transcript` replaces the preview); previews with text come after a pass, never
  after `SpeechEnded` (the empty preview that withdraws one can). `listen()` gets the same `Transcript`s without previews. With sentence joining, the first chunk
  of an utterance is decided as a transcript is today, later chunks only extend it. A piece that starts where a cut
  inside speech ended (a forced cut, the length cap) on a frame of speech (the cut is at the quietest frame of its window)
  continues the sentence: no sentence decision, its first `Transcript` has `sentenceJoin` with `continuesPrevious` true, `previousEnd`
  `REMOVE_PERIOD` when the previous text ends in a period (a question or exclamation mark stays, and then `sentenceJoin`
  is null) and `continuingText` with the first letter lowered (an acronym keeps its letters). With sentence joining the
  language model is asked once per piece about that capital only: the first word keeps its capital unless the small letter wins by more than 1.0 (a name stays)
  (`continuingText` is then the capital form); when it fails, or when the word has no capital to lower, the letter is lowered
  as before. Logcat says `sentence boundary: continued (cut)`, with `, lower-case|capital kept, language model N ms` when it
  was asked (50 to 360 ms on a phone). A cut on a frame below the endpointer's `negThreshold` is a pause shorter than
  `minSilenceMs`, typically a comma or a breath, not a sentence end: the language model decides the boundary as after a pause
  but leans to going on. The joined variants (the plain continuation, the comma, a name with its capital) win unless the end
  of the sentence is clearly likelier (by more than 2.0 in natural log P, where after a real pause a join needs 5.6 in its
  favour), and the first word is lowered when the model prefers the small letter by more than 1.0 (2.5 after a pause). The
  `Transcript.sentenceJoin` is the same as for a pause (`previousEnd` may be `PERIOD_TO_COMMA`); Logcat says
  `sentence boundary: join|comma|split (quiet cut), language model N ms`. After real silence (the endpointer ended the
  utterance) nothing changes. Without streaming nothing changes.
- **`SpeechEvent.UtteranceDone(startMs)`:** exactly one for every utterance (piece of speech), after its last  `Transcript`, also when its final recognition found no words or failed; `startMs` equals the `startMs` of its  `Transcript`s, and for the first utterance of a speech the `atMs` of its `SpeechStarted`. A speech cut by the length  cap or the lag limit is several utterances, each with its own `UtteranceDone` and no `SpeechStarted` after the  first. It can come after the `SpeechStarted` of the next speech. With the default `CutSeam.CONTEXT_AND_HELD_WORD` the `UtteranceDone` of a piece cut inside speech is deferred until its held word is resolved: it comes right before the next piece's first `Transcript` (or after the one more `Transcript` that sends the word as it was); `CutSeam.CONTEXT` sends it at once. Never emitted without streaming.
- **Guarantees:** every word's text is sent at most `maxLagMs` (default 15 s) after it was spoken, on a phone for
  which a recognition of 8 s of audio takes at most `maxLagMs` minus 9 s; the safeguard measures the recogniser and
  cuts the piece (at the quietest place of the last 2 s, as the length cap does) before the delay would break the
  bound, and pauses the passes on a phone where a pass over 6 s would take more than 2 s. The text once sent never
  changes. A cut of a piece (the length cap, a forced cut) never falls before the audio whose words were sent, so no
  word is written twice there. On a phone where the passes are paused (a pass over 6 s takes more than 2 s) the text
  comes from the final recognitions of pieces, and a piece is cut as soon as the phone keeps up with it and there is a
  good place: the target length is the shortest one from 5.5 s (up to 12 s) at which the recognition of
  the shortest possible piece (the target less 1.5 s, never under 4 s) is estimated at 60 % of it or less; from that
  floor on (the target less 1.5 s) the piece is cut at once on the first quiet stretch of 256 ms (eight frames below the
  endpointer's `negThreshold`, the pause with which `stop()` decides that a word has finished: a breath or a comma, not a
  hesitation between a preposition and its word), INSIDE that stretch, keeping 128 ms of it in the piece so that no word is left
  behind; with no quiet stretch
  it is cut at the target, at the quietest frame of the last 1.5 s before it. So the pieces are 4 to 8 s by the speed of
  the phone (recomputed every frame, so a phone that heats up gets longer ones) and the first words come about a second
  earlier than with the cut at the target only. A shorter piece
  would not be cheaper (the encoder window has a floor: a recognition of 2 s of audio takes as long as one of 7 s), it
  would only add cuts, load and split words. A phone too slow for that is cut at the lag limit only, after 8 s of audio at
  the earliest. **What is not guaranteed:** where the buffer is cut inside a piece, a
  word can be repeated or lost (the overlap is heard differently the second time; the measurement's simulation has this
  too), and a forced cut at the lag limit, which has little room to choose a quiet place, can split a word. With the context at a cut (`CutSeam`), the final of a piece after a cut covers up to 14.5 s of audio, not 12 s, and a fallback adds a second recognition (about 2 s on a phone), so for such pieces the rest does not always come within `minSilenceMs` plus one recognition of at most 12 s, and a word can come later than `maxLagMs`, as after the retry of an empty final. The bound is thin
  on a phone that is slow and gets slower as it heats up: with a short pass of 6.6 s (twice the cool floating
  microphone) the simulated largest delay is 15.0 s of the 15 s.
- **Context at a cut (`StreamingConfig.cutSeam`):** a piece cut inside speech may split a word, and the piece after it starts
  with no sound before it. `StreamingConfig.cutSeam` is `CutSeam.CONTEXT_AND_HELD_WORD` by default (the held word, below, on top of the
  context); `CutSeam.CONTEXT` is the context alone, with no change in the events, and `CutSeam.NONE` neither. With the context (both of the first two) the final recognition of the piece after such a cut also gets the
  last 2.5 s of the audio of the piece before it. Its text then starts with words that were already sent: the last words sent
  are lined up with the first words of the text (letters and digits only, no case, no punctuation), the anchor is accepted when
  at least two words are assigned and at least two of them agree, and only the words after the anchor are sent. Nothing already
  sent changes; that no word is sent twice is a heuristic, not a guarantee (a misaligned anchor can repeat or drop a word of the seam;
  measured on a laptop: 0 repeated and 4 lost words in 196 seams). With no anchor (an empty run, no two agreeing words, nothing after
  the anchor) the piece is recognised from its own audio alone, as with `CutSeam.NONE` (only the ladder of retries after an empty result
  differs, see below), at the price of a second recognition (about 2 s on a phone, in about 5 % of the pieces after a cut). Not after a cut on a quiet frame, nor after silence, nor when the piece before sent fewer than two words, nor
  for a piece whose passes already sent words. **Language:** the run is forced in the language of the piece before it (the session language does not change at a cut), so in a
  session with several languages a language change inside continuous speech is not noticed at a cut: after a cut inside speech the next
  piece is recognised in the previous piece's language, and a change is noticed again after a pause (a new utterance), a quiet cut, or
  an anchor fallback (the piece's own audio is then recognised with the language rule). A piece whose text came from a rescue in another
  language hands no context on. **Cost:** the context adds 2.5 s to the audio of the final. Up to 7 s of audio the encoder window is
  the model's floor (384 frames) and costs the same, but the decoder writes about 2.5 s more words: on a laptop the time per tail
  went from a median of 1904 to 2009 ms (+7 %) and from a p90 of 2205 to 2490 ms. Above 7 s the window grows with the audio (8 s: 432
  frames). The guard counts the context in its estimate of when the buffer must be cut (lag limit) but not in the target length of the
  pieces (the 60 % duty): a continuation final therefore has a real duty above 60 %, the flat-cost median of the guard mixes runs with
  and without the context, and a fallback doubles one final; this is not a runaway (the context is a fixed addend, and lag cuts are
  bounded below by 8 s of buffer) but it is not measured on a phone: check the median piece length, the lag-limit cuts per minute
  against `CutSeam.NONE` and the p90 delay there. After such a final that comes out empty the retry is on the full
  encoder window only (no step on 512 frames), and only if it would still end within the lag limit. `CutSeam.NONE` keeps
  everything as it was. Measured on a laptop (196 tails after a forced cut): the text anchor was found in 95 % of them, the
  errors (substitutions, deletions, insertions) in the part after the cut fell from 12.9 % to 8.9 % of its words (about a third; with the held word
  to 6.1 %), the deletions alone by about two thirds (0.9 % to 0.3 %); not measured on a phone. Logcat (counts and times, never text):
  `utterance: seam context C ms + tail T ms, window W frames, anchor M of N word(s) matched, new text after P word(s)` or
  `utterance: seam fallback (no anchor|empty|gate|empty rest), tail-only run follows`, `streaming: seam context released unused (<why>)`,
  and in the session line `seams: N with context (A anchored, F fallback (reason n, ...), R ms of fallback runs, windows above 384: W), U context(s) never used`.
- **The held word (`CutSeam.CONTEXT_AND_HELD_WORD`, the default):** the word a cut splits is the last one of the piece before it. With
  this setting (the default; `StreamingConfig(cutSeam = CutSeam.CONTEXT)` turns the holding off and keeps the context, with the plain order of the events), while the phone is in chunk mode (the passes are paused) a piece that ends at a cut inside speech and has at least three
  words is delivered **without its last word**. The final recognition of the next piece (with the context) decodes that word
  again, and the text of the next piece then starts with its new version: the old one is never sent. What an app sees: the first
  `Transcript` of a piece after a cut starts with a word that was not at the end of the piece before (`sentenceJoin` says
  `continuesPrevious`, `previousEnd` KEEP and `continuingText` is the text as Whisper wrote it, case included; no language model is asked, and
  the previous text has no period from the cut to remove), and the `UtteranceDone` of the cut piece is **deferred** until then: it comes
  right before that `Transcript`, about one piece (a few seconds) after the last `Transcript` of the cut piece. Where the word
  cannot be replaced it is sent as it was, once, as one more `Transcript` of the cut piece (same `startMs`, `endMs` of the piece,
  `processingMs` 0, no `sentenceJoin`) before anything else of the next piece, and the `UtteranceDone` follows it: when the text has no
  two-word anchor on the delivered words (the stage-1 anchor on all the words, the held one included, is tried next, with no new
  recognition, and only if that fails the piece is recognised again from its own audio), the next piece was dropped or skipped, the
  next speech starts after silence, capture ended, or the replacement would come later than `maxLagMs` allows. With a quiet cut, after
  silence, with fewer than three words and while passes run nothing is held. A held word is never shown as a preview. **Delay:** the guarantee of `maxLagMs` holds with the held word too, because a replacement that would come later is refused and the word is sent as it was; in the simulation on the Pixel's measured curve (5 minutes of continuous speech) the largest delay of a word grew by 0.5 s at most (7.27 to 7.77 s at x1.0, 9.54 to 10.04 s at x1.4) and not at all on the slower phones that are cut at the lag limit (14.39 s at x1.9, 15.00 s at x2.2); not measured on a phone. A cancelled
  session sends nothing, as for any text waiting. Logcat: `streaming: held word replaced|kept (<why>), delay N ms`, `utterance: seam context
  ... held word replaced|kept`, and in the session line `held H: replaced X, kept Y (<why> n, ...), held-word delay median M ms max X ms`.
- **Cost:** while passes run, the recogniser computes about four times as much as without streaming: battery and heat
  grow with the time spent speaking. Early start is ignored while streaming is on (one logcat line), and so are the
  timing and `fromPauses` of `PreviewConfig`. `transcribe()` is not affected.
- **Measured in simulation only,** on the Czech+Slovak+English model with the encoder window floor 384; nothing has been
  measured on a phone. With a model that needs the floor 512 (the published Czech model) passes are dearer and more of the speech is closed
  by the safeguards.
- **Logcat** (counts and times, never text): `pass: A ms of audio from S ms, recogniser R ms, window W frames, …; C
  word(s) committed, P pending` (or `recogniser cancelled after …`, or `dropped, <why: its utterance has ended, its buffer was cut meanwhile, cancelled after it finished, …>`),
  `streaming: forced cut at …, reason: target L ms, duty D %` (the piece reached its target length; `target L ms, quiet point, …`: it was cut earlier on a quiet stretch) or `reason: lag limit M ms`, `streaming: passes paused, a pass over 6 s would take P ms` / `passes resumed`,
  `utterance: empty result in xx, rescue in yy: rescued|still empty` (a final recognition that came out empty was run once more
  forced in another language of the rule; see "Language rule"),
  `utterance: language decision of the first run: chose xx, current xx, audio N ms, shares xx 0.811 yy 0.174 zz 0.015, thresholds zz 0.900|too short to decide, no candidate, rescue in zz|no rescue: <why>`
  (after a final whose first run chose its language by the rule came out empty: what it decided from and why no rescue ran),
  `utterance: … no letters or digits, dropped (model returned: empty|bracketed|punctuation|symbols, N part(s); language en (forced|chosen))`
  for a recognition whose text has no letter or digit and is dropped (the kind of nothing and the number of its parts,
  never the text or its length; whisper.cpp's probability of no speech, the average log-probability and the number of
  tokens would need a change of the native library; the line says the language the recogniser ran in and whether it was forced or chosen),
  `streaming: utterance of D ms: K passes (A aborted), W words while speaking, F after it ended, C cuts` after each
  piece, and one `streaming: K passes …` line per session with the passes, transcripts, cuts, forced cuts, pauses, the
  largest buffer, the median length of the pieces, the delay of the text after its audio (median and maximum, a lower bound) and the counts of what the
  algorithm does not guard: seams decided by a final recognition, empty finals and the pending words they lose, stale
  passes and internal errors.

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
| `minSilenceMs` | 600 | the utterance ends after this much audio below `negThreshold` (sensible 400 to 2000; accepted 32 to 10000). Longer: fewer sentences split where the speaker paused to think (a pause of 900 ms is two utterances at 600 and one at 1200), the text later by the same amount; with `earlyStart` more of its starts are wasted (it must stay a frame below), with streaming a shorter pause is a quiet cut |
| `preRollMs` | 300 | audio before the start of speech that is passed to the recogniser |
| `minSpeechMs` | 250 | utterances with less audio ≥ `threshold` are dropped (clicks, coughs) |
| `maxUtteranceMs` | 28 000 | longer speech is cut at the quietest 32 ms frame of the last 2 s and continues as the next utterance (so a piece is between the limit less 2 s and the limit: 2 s at the smallest limit, 4000) |
| `threads` | 0 | most recogniser threads; 0 = number of big CPU cores (2 to 8) |
| `autoThreads` | true | before each recognition, threads and cores from the cores the app may use at that moment, see [Threads and cores](#threads-and-cores); false = always `threads` threads, placed by Android |
| `variantSelection` | null | beam search and a Czech language model choosing among the variants, see [Variant selection](#variant-selection-optional-off-by-default) |
| `sentenceJoining` | null | the Czech language model decides whether a transcript continues the sentence of the previous one, see [Sentences split by a pause](#sentences-split-by-a-pause-optional-off-by-default) |
| `streaming` | null | send the text of words while the person is still speaking, in continuous speech, see [Streaming transcription](#streaming-transcription-optional-off-by-default) |
| `retryEmptyFinals` | true | repeat a final recognition that came out without a letter or a digit on a longer encoder window, see [Empty results](#empty-results-are-repeated-on-a-longer-window) |

**Why the threshold is 0.25 and not Silero's usual 0.5.** We tested both on 2,181 recordings with speech. At 0.25
no recording was dropped. At 0.5, 5 of the 729 recordings with reverberation and babble noise were dropped. Every
synthetic non-speech signal was still rejected at 0.25: silence, white, pink and brown noise, 50 Hz hum, clicks,
bursts and beeps.

## Variant selection (optional, off by default)

Whisper can propose several variants of an utterance (beam search), and a small Czech language model can pick the
one that reads best. This mostly fixes word endings, i/y and diacritics that the sound alone does not decide ("velký
psy" → "velké psy"). Measured on parliament speech, read sentences (FLEURS) and parliament speech in noise, with five
beams, it lowered the word error rate by 0.34 to 1.60 points (16 to 20 %) and fixed a quarter of such confusions.

```kotlin
val lm = File(…)   // the language model, see below
val speech = CzechSpeech(context, modelFile, CzechSpeechConfig(variantSelection = VariantSelection(lm)))
```

Without `variantSelection` nothing changes: greedy decoding, the same native call and the same transcripts as
before. [Previews](#previews-while-speaking) never use it: they stay one greedy pass, so only the final text pays
for the beams and the language model.

- **The language model** is Czech GPT-2 small ([lchaloupsky/czech-gpt2-oscar](https://huggingface.co/lchaloupsky/czech-gpt2-oscar),
  MIT, 124 M parameters), converted to GGUF q8_0 by `scripts/lm/convert_gpt2.py`. The conversion is reproducible,
  so the file has a fixed size and SHA-256 (`CzechLanguageModel.SIZE_BYTES`, `SHA256`, `FILE_NAME`), and it is
  downloaded and verified like the CzechSpeech model:
  `CzechSpeechModel.download(context, CzechLanguageModel.source(url))`. As with the model, the size and SHA-256 are
  checked when the file is downloaded, not each time it is loaded; loading checks its structure (shapes, tensor types,
  size) and refuses a file it cannot run. Moravian AI does not publish it yet, so there is no default URL: serve it
  yourself or push it with adb.
- **How a variant is chosen:** score = log P_Whisper + λ · log P_LM + β · length in Whisper tokens, the highest wins
  (λ = `languageModelWeight`, β = `lengthBonus`, both 0.3, the measured optimum for five beams; β 0.75 for three).
  Whisper's choice stays when the variants differ only in case and punctuation (the language model is not run), when
  the language model fails, and in a window Whisper decoded at a fallback temperature (its log-probabilities are then
  on another scale than the weights were tuned for). A variant without a letter or digit (a beam that ended the text
  at once) is never chosen unless Whisper chose it: the language model would give the empty text its highest score.
  The other way round, when Whisper's choice has no letter or digit (no speech, says Whisper), it stays: the text of
  a beam Whisper turned down never overrides it, and neither the language model nor the formula is run.
  `languageModelWeight = 0f` turns the language model off (beam search alone); `languageModelFile` can then be null.
- **Beam search is fixed.** whisper.cpp 1.9.5 drew the beams' candidates at random instead of taking the most likely
  tokens, and started every beam with the same first token, so a confident model gave five copies of one text. The
  SDK builds whisper.cpp with `native/patches/whisper.cpp-czechspeech.patch`, which fixes both (at temperature 0;
  the temperature fallback still samples) and lists every beam's hypothesis. Greedy decoding is not affected.
- **Cost.** Beam search repeats Whisper's decoder for every beam (estimated 2.2× the time of greedy for five beams
  on a Pixel 9a, not yet measured; the Benchmark screen of the `sample` app's debug build measures it) and keeps a
  7-sequence key/value cache (+63 MB). The language model holds, while it is loaded: 135 MB of weights, a compute
  buffer kept between utterances (2.7 MB measured on short sentences; a 32-token chunk needs about 8 MB, an estimate)
  and the tokenizer's tables on the Java heap (10.8 MB measured on the JVM). While it scores, a key/value cache of
  about 74 KB per token (7 MB for 100 tokens) comes on top. It is loaded with the CzechSpeech model, stays with it
  under `keepModelLoaded` and is freed with it.
- **Privacy:** the language model only sees the variants' text in memory. Nothing is logged but counts and times, and
  the tokenizer forgets the words of an utterance once it is scored.

## Sentences split by a pause (optional, off by default)

Each utterance ends at a pause of about 0.6 s and is transcribed on its own. A pause in the middle of a sentence then
often comes out as two sentences: Whisper ends the first part with a period and starts the second with a capital
("…můžeš nějak. S tebou spojit…", or "…nějak. s tebou…"), or writes a capital without a period. With sentence joining,
the Czech language model of [variant selection](#variant-selection-optional-off-by-default) decides at each such
boundary whether the new transcript continues the previous one:

```kotlin
val lm = File(…)   // the language model, as for variant selection
val speech = CzechSpeech(context, modelFile, CzechSpeechConfig(sentenceJoining = SentenceJoining(lm)))
```

It works with greedy decoding as well as with variant selection; with both, give them the same file, which is loaded
once. Without `sentenceJoining` nothing changes.

**What you get.** `Transcript.text` is always exactly what you get without sentence joining. The decision comes in
`Transcript.sentenceJoin`, null when the transcript stays as Whisper wrote it:

| field | meaning |
|---|---|
| `continuesPrevious` | this transcript continues the sentence of the previous transcript of the session |
| `previousEnd` | what to do with the end of the previous text: `KEEP`, `REMOVE_PERIOD`, `PERIOD_TO_COMMA` and, added later, `REMOVE_ELLIPSIS` ("..." or "…" at the end goes), `ELLIPSIS_TO_COMMA` and `REMOVE_COMMA` (a trailing comma goes); `previousEnd.applyTo(text)` does it, closing quotes and brackets and whitespace after the end stay |
| `continuingText` | this transcript after the previous one, first letter as decided ("s tebou spojit…", but "Honzu a Petru" after "Pozdravuj") |
| `standaloneText` | this transcript on its own, with a capital first letter |

A client that can still change the end of the previous text (nothing typed or moved since it inserted it) applies
`previousEnd` and appends `continuingText`; otherwise it appends `standaloneText`, which is what a sentence end
needs. When `continuesPrevious` is false, both texts are the same: Whisper wrote "A. b" and the language model kept
the sentence end, so the capital is added ("A. B").

**An ellipsis or a comma at the end of the previous text.** A recogniser that is cut at a pause often writes "…" or "..." (or
a comma) there. The language model decides these boundaries too. After an ellipsis ("A… b") the candidates are the plain
continuation (`REMOVE_ELLIPSIS`, the next word small unless the model prefers its capital, a name), a comma
(`ELLIPSIS_TO_COMMA`), the ellipsis kept with a small letter (`KEEP`, `continuingText` small) and a new sentence as written
(`sentenceJoin` null, or `continuesPrevious` false when the capital is added). When the two speeches are less than 2.0 s apart
(`SentenceBoundary.ELLIPSIS_PAUSE_LIMIT_MS`, and always after a cut on a quiet frame) the prior leans to removing the ellipsis:
another reading wins only when the model prefers it by more than 2.0 (a comma 1.5) in natural log P. After a longer real pause a
trailing off is likelier real, so the readings are compared without that prior (the comma still needs its 1.5). After a trailing comma the text without it
("A b") is a candidate next to "A, b" and "A, B": it wins (`REMOVE_COMMA`) when the model prefers it by more than 1.5, so a pause
in the wrong place does not force a comma. A real trailing off at the very end of the dictation is not touched: no piece follows
it. The constants are not measured (no labelled data); they are conservative and meant to be tuned on logs. Logcat:
`sentence boundary: ellipsis removed|ellipsis to comma|ellipsis kept|comma removed, language model N ms` (with ` (quiet cut)`
after a cut on a quiet frame), never text. A piece after a cut on speech removes the ellipsis of the previous piece without asking.

A keyboard that commits each transcript followed by a space:

```kotlin
private var lastCommitted: String? = null       // the last transcript as committed, with its space

fun onTranscript(t: Transcript, ic: InputConnection) {
    val join = t.sentenceJoin
    val previous = lastCommitted
    val untouched = previous != null && ic.getSelectedText(0).isNullOrEmpty() &&
        ic.getTextBeforeCursor(previous.length, 0)?.toString() == previous
    ic.beginBatchEdit()
    val text = when {
        join == null -> t.text                                   // as without sentence joining
        join.continuesPrevious && untouched -> {
            val edited = join.previousEnd.applyTo(previous!!)    // "…nějak. " → "…nějak "
            val same = previous.commonPrefixWith(edited).length
            ic.deleteSurroundingText(previous.length - same, 0)
            ic.commitText(edited.substring(same), 1)
            join.continuingText                                  // "s tebou spojit ve video hovoru."
        }
        else -> join.standaloneText                              // the previous text stays as it is
    }
    ic.commitText("$text ", 1)
    ic.endBatchEdit()
    lastCommitted = "$text "
}
```

Keep any history of inserted texts in step (an "undo the last sentence" button, say): the previous entry now ends as
`edited`. Your own capitalisation rules should not change `continuingText` back.

- **The rule** is the one measured on parliament speech and read sentences with the same model (12 Nov 2026): the language model scores the end of the sentence
  ("A. B"), the joins with a small letter ("A b") and with the capital kept ("A B", a name) and a join by a comma
  ("A, b"), on the last 20 words of the previous transcript and the first 12 of the new one. It joins when a join
  beats the end of the sentence by more than 5.56 (natural log P; 3.66 when Whisper wrote a small letter after the
  period, which makes a false sentence end about 7 times likelier). A period is never added: after "A B" or "A, B" it
  only lowers the capital. A capital is lowered only when the small letter wins by more than 2.5: names of people
  kept theirs in the measurement, but multi-word names and institutions lost it in 2 to 6 % of the cases
  ("Ministerstva zemědělství"). On real transcripts this fixed 42.5 % of the false periods (17 of 40) and took the
  period of 0.9 % of the true sentence ends (1 of 117; 0.4 % of 770 in a simulation); it lowered none of the 6 names
  at such a boundary. The SDK's rules are checked decision by decision against the measurement's code
  (`scripts/sentences/make_reference.py`, `SentenceBoundaryReferenceTest`).
- **Never joined:** across two `listen()` sessions, across a pause longer than 3 s (from the end of one transcript's
  speech to the start of the next), across speech that was lost (an utterance that failed or was skipped) or gave no
  words, a boundary that looks like no false sentence end (a question mark, an abbreviation, an ordinal, "A b" as it should
  be), previews, and `transcribe()`.
- **Time:** the first transcript comes as soon as without sentence joining; only the one after a boundary waits for
  the language model, a few short texts sharing their beginning (0.1 s on 6 threads of the build machine, 0.4 s on
  one; see [What has been tested](#what-has-been-tested); the target on a phone is under 0.3 s). It counts in that transcript's `processingMs`.
- **Loading.** The language model (135 MB of weights, 11 MB of tokenizer tables on the Java heap) loads right after
  the first utterance of a session, so that its transcript never waits for it; an utterance that ends while it loads
  waits, once. It does not load when no other utterance can come (one sentence ended by `stop()`). It is freed with
  the CzechSpeech model, and `transcribe()` never loads it for sentence joining. With variant selection on too, it
  loads with the model instead, before the first utterance, which needs it there.
- **Without the language model** (the file does not load, or it fails on a boundary) the transcript comes without
  `sentenceJoin`, and the log says why. A load that failed is tried again in the next session, also when the model
  stays loaded (`keepModelLoaded`), so a file downloaded meanwhile is picked up.
- **`PreviousEnd.applyTo()`** changes only a period that may end a sentence; text that ends otherwise (an ellipsis,
  "atd.", "17.", a question mark, nothing) comes back as it is.
- **Privacy:** as with variant selection, the language model only sees the text in memory; the log says what was
  decided and how long it took, never the text, and `SentenceJoin.toString()` does not print it.
- **Not yet measured on dictation.** The test sets were parliament speech and read sentences; forms of address and
  short sentences were rare there ("Tomáši. | Zdraví Honza." stays as it is). Try it on real dictation before turning
  it on for everyone.

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
- **The audio kept at a cut (streaming, `CutSeam.CONTEXT` and `CutSeam.CONTEXT_AND_HELD_WORD`):** the last 2.5 s of a piece cut inside speech stay in memory
  only, in one array of 160 KB at most, until the final recognition of the next piece has used them; they are zeroed right
  after that run, and also when no piece continues them (the piece after was dropped or skipped, the next final is of another speech,
  capture ended, the session ended or was cancelled). They are never written anywhere and never logged.
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

### Threads and cores

Android lets an app use different cores depending on what it is doing, and changes that while it runs. On a Pixel 9a
(Tensor G4: cpu0–3 Cortex-A520, cpu4–6 A720, cpu7 X4) an app on screen may use all eight, a service under another
app's window (a keyboard's floating button) only 0–6, and an app with the screen off only the four little cores.
whisper.cpp splits its work evenly between its threads, so the slowest thread sets the pace: four threads on the three
A720s and one A520 took 5.0 s for a 6 s utterance that four threads on the X4 and the A720s took 2.6–2.8 s.

The rule is simple and conservative, since it cannot be measured on phones other than the Pixel 9a: it changes the
behaviour only where the gain is large. Before every recognition (final text, preview, and each run of the language
model) the SDK reads the cores its recognition thread may use now (its mask, `Cpus_allowed` of
`/proc/thread-self/status`, else `sched_getaffinity`) and the speed of each core
(`/sys/devices/system/cpu/cpuN/cpu_capacity`, else `cpufreq/cpuinfo_max_freq`, read once). It then compares two plans
by threads × the capacity of the slowest core a thread ends up on:

- **As before:** `threads` threads (by default the big core count), wherever Android puts them, which is the fastest
  allowed cores first; but never more threads than allowed cores, since ggml's threads wait for each other spinning,
  without yielding, and a thread that shares a core stalls every step.
- **Fast cores:** the allowed cores faster than the device's little cores (its slowest cores and any at most a fifth
  faster), at most `threads` threads, the thread pinned to them. whisper.cpp and ggml start their worker threads from
  the pinned thread during the call, and Linux gives them the same cores.

The fast cores are taken only when they do at least 1.5 times better by that estimate. The margin is there because
ggml shares out matrix products dynamically, so "the slowest core sets the pace" overstates a slow core's delay: on the
Pixel 9a under the floating button the measured gap, 1.9 times, lay between that estimate's 3.9 times and 1.35 times
of an estimate that adds the cores up. With all cores allowed the plans tie (the fast cores are where Android puts the
threads anyway), so the plan is as before, unpinned.

Cores are counted from the thread's mask, never from those running at the moment: phones park idle big cores
(Qualcomm's core_ctl) and wake them under load. Instead, at the end of a pinned computation the SDK reads the pinned
cores that ran (`sched_getaffinity`); when fewer ran than there were threads (cores isolated for heat, which no load
wakes), pinning pauses for 30 s, with everything exactly as before meanwhile, and is then tried again.

After a pinned call the thread is unpinned: it may use every CPU again, which the kernel bounds by the cpuset of the
moment. Its mask from before is never put back: Linux 6.2+ keeps that mask as the thread's own request, intersects
every later cpuset with it and gives it to every thread the thread starts, so a mask read under the floating button
(0–6) would keep the recognition off cpu7 after the app comes on screen.

| layout (cpu_capacity, limit) | on screen | 0–6 (floating button) | little cores only |
|---|---|---|---|
| Tensor G4: 4× 191, 3× 741, 1× 1024 (4) | 4, as before | 3 on 4–6 (2.9×) | 4, as before |
| 4+4: 4× 382, 4× 1024 (4) | 4, as before | 3 on 4–6 (2.0×) | 4, as before |
| 1+3+4 (Snapdragon 865): 4× 220, 3× 855, 1× 1024 (4) | 4, as before | 3 on 4–6 (2.9×) | 4, as before |
| 2+6 (Helio G99): 6× 391, 2× 1024 (2) | 2, as before | 2, as before (1 on 6 is 1.31×) | 2, as before |
| 1+5+2 (Snapdragon 8 Gen 3): 2× 325, 5× 820–870, 1× 1024 (6) | 6, as before | 5 on 2–6 (2.1×) | 2 on the 2 cores (before: 6) |
| Dimensity 9300: 4× 560 (A720), 3× 900, 1× 1024 (4) | 4, as before | 4, as before (3 on 4–6 is 1.2×) | 4, as before |
| 2+2+4 (Tensor G1): 4× 160, 2× 498, 2× 1024 (4) | 4, as before | 3 on 4–6 (2.3×) | 4, as before |
| 8 alike (8) | 8, as before | 7 (before: 8) | 4 (before: 8) |
| emulator, no speeds | `threads`, as before | the same | the same |

So a single fast core among stronger little ones (4+4 with the cpuset 0–4: 1 × 1024 against 4 × 382) stays as before,
and a strong slowest tier loses no thread. With fewer allowed cores than `threads` (alike cores in a smaller cpuset,
say), there are as many threads as allowed cores, unpinned: a decision, since the estimate cannot see the spinning
barriers. With the clock fallback, which cannot tell an A720 from an A520 at a similar clock, the estimate is
conservative (on Tensor G4's clocks the floating button stays as before). A failure to read or set the affinity never
fails a recognition: a refused pin runs the plan as before, and a failed unpin is logged and retried before the next
plan, which meanwhile runs no more threads than the cores of the thread's narrowed mask, pinning nothing.
`CzechSpeechConfig(autoThreads = false)` turns all of this off for diagnostics.

Logcat (tag `CzechSpeech`) shows, numbers only, the cores and their capacities once per session (`cpus: capacity
191,191,191,191,741,741,741,1024 (cpu_capacity), at most 4 threads, placed per recognition`), and with each recognition
the threads and the core masks: `threads 3 on cpus 0x70 (allowed 0x7f, fast cores)`, `threads 4, not pinned (allowed
0xff, as before)`; a pause shows as `pinning paused`.

## Why whole utterances and not words as you speak

Whisper is not a streaming model. It transcribes a finished piece of audio. Showing partial results means running
the recogniser again and again on the growing buffer while the person speaks, which costs battery on a phone for
text that is then replaced. So CzechSpeech waits for the end of an utterance (600 ms of silence) and transcribes it
once. Words do not appear as they are spoken; [previews](#previews-while-speaking) are the opt-in middle ground:
the text so far every few seconds, at the price above, and [streaming transcription](#streaming-transcription-optional-off-by-default) sends the words themselves while the person speaks.

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

## The encoder window and the model

The encoder's time grows with the audio window it works on, so the SDK trims the window to the utterance (whisper.cpp's
`audio_ctx`): the audio in 20 ms frames plus 32, rounded up to a multiple of 16, and the whole 30 s window from 1500
frames up. A model fails on windows much shorter than it was trained on, inventing and repeating sentences in short
utterances, so the window never goes under a floor, `CzechSpeechConfig.minEncoderWindow`. **The floor belongs to the
model file:**

| model | `minEncoderWindow` | why |
|---|---|---|
| CzechSpeech (Czech), the published one | 512 (10.24 s), the default | with 384 it gets worse; from 256 down it invents text |
| Czech+Slovak+English (the enterprise model, not in the public catalogue) | 384 (7.68 s) | no loss in the measurement, and 27 % less recognition time for utterances up to 7 s |

The measurement compared the two models on the same recordings. What it does not cover: 384 is close to where the model starts to fail (256 is too low for the Czech model,
and nothing between 256 and 384 was tried for the new model), the times are from x86, not from a phone, and the
speech was parliament speech, not dictation. Measure on the phone before relying on it: the sample app's Benchmark
takes `--ei min_audio_ctx 384`, and logcat (tag `CzechSpeech`) gives the floor in `model loaded … encoder window at
least N frames` and the window of every recognition in `utterance: … window N frames`. The floor applies to the
final text, previews and early recognitions alike; a value that is not a multiple of 16 from 256 to 1500 is refused
(`IllegalArgumentException`). It is a setting of `CzechSpeechConfig` because this version has no description of a
model beyond its file; a model catalogue should carry it.

### Empty results are repeated on a longer window

Whether the model writes anything for a stretch of speech is a flip on the edge that depends on the window. In the
measurement of the cs+sk+en model (noisy speech, 1.6 to 3.6 s of speech per utterance) the next longer window gave text for
12 of 16 empty results; whisper.cpp's temperature fallback gave none; and on audio without speech the model wrote nothing at
384 frames or more, so a repeat does not invent text. So a final recognition that would be dropped for having no letter or
digit is repeated when the endpointer saw clear speech in it (at least 1 s of frames above `threshold`): first on 512
frames when the run was below that, then on the longest window the library takes (a floor of 1488 frames, which is the
whole 1500 within 0.8 %). The first result with words is the final text (same `startMs`, same events, sentence joining as
usual); at most two repeats. The full-window repeat (the dear one) starts only when nothing else waits for the
recogniser, checked right before it starts: the queue is empty, in streaming no other final is pending, and `stop()` has not
been called (after `stop()` no repeat of any kind starts); a repeat whose window is less than 25 % longer than the first
run's is skipped; a streamed piece that already wrote words is repeated only when words were still pending (else what
remains is the tail of the piece). The audio stays in RAM and is zeroed after the last attempt. Early-start recognitions are not
repeated. **Cost:** one more recognition per empty result, only then: on a Pixel-like phone (2.3 s at 384 frames) about
4 s on 512 and about 12 s on the full window, estimated from the measured cost per window (not measured on the full window),
so that utterance's text comes up to ~16 s later in the worst case. **This can exceed `maxLagMs` for that one stretch**
(streaming's 15 s) by about 4 s for the repeat on 512 frames and, when nothing else waits, by up to 12 to 16 s more for
the full window, so by up to about 20 s in all (a warm phone is 1.4 to 1.9 times slower). With streaming, the full-window repeat is skipped while
two finals are pending, and the repeats are booked apart from the guard's cost estimates.

With a language rule ([Models and languages](#models-and-languages)) an empty result is first rescued in another language,
one run on the same window that does not depend on `retryEmptyFinals` or on the second of clear speech; the repeats then follow
in the language of the last attempt.

Logcat (no text, no length): `utterance: empty result, retry with window N: rescued` or `still empty`, and for the language
rescue `utterance: empty result in xx, rescue in yy: rescued|still empty` (the languages of the first run and of the rescue)
and `utterance: language decision of the first run: …` (the shares, the thresholds and the audio length the rule decided from, and the rescue or why none ran).
The line `…, no letters or digits, dropped (…)` of a run that another run follows is held back only when the repeats can
follow: `retryEmptyFinals` is on, the utterance has at least 1 s of clear speech, it is not an early start and not the rest of a
streamed piece that wrote words without any pending. Then only the line of the last run comes, once, after the last attempt,
and none when a later run gave words. In every other case (under 1 s of clear speech, `CzechSpeechConfig(retryEmptyFinals =
false)`, which turns the repeat off and drops the result at once as before) every run writes its line at once, so a language
rescue that comes out empty too gives two lines, the first run's and the rescue's.

**`SpeechEvent.UtteranceNotWritten(startMs)`** is emitted once for an utterance (or, with streaming, a piece that wrote no
words) of clear speech that ends up without text after the repeats, or with the repeat off, so that the app can show that
something was said and is missing. `startMs` is that of the `Transcript` it would have had; with streaming it comes before
the `UtteranceDone` of the same `startMs`. For a piece that a forced cut or the length cap ended inside speech it comes
**before** the `SpeechEnded` of the whole utterance, else after the `SpeechEnded`. Nothing is emitted for a stretch without
clear speech (a click, a cough); it is emitted also when `stop()` kept a repeat from starting.

## Building from source

```bash
git clone --recurse-submodules <this repository>
native/build-native.sh            # Linux or WSL: NDK r27c, CMake 3.22; builds all variants + the host library
./gradlew :czechspeech:assembleRelease :sample:assembleRelease
./gradlew :czechspeech:testReleaseUnitTest
./gradlew :czechspeech:publishToLocalFolder      # static Maven repository in build/maven-repo
```

Packaging the AAR (`bundleReleaseAar`, so `assembleRelease` and publishing) first runs `checkNativeLibs`, which fails with the
command to run when `native/out` is missing or was built from other sources than this tree (`native/source-stamp.sh` hashes
`native/src`, `native/patches`, `native/CMakeLists.txt`, `native/build-native.sh` and the whisper.cpp commit; the build writes
the stamp next to the libraries). The unit tests do not depend on it.

- **Toolchain:** JDK 17 and the Android SDK with platform 36 and build tools 36.0.0. The build uses the Android
  Gradle plugin 8.9.3, Gradle 8.14.4 (the wrapper) and Kotlin 2.0.21; `compileSdk` is 36, `minSdk` 26.
- **whisper.cpp:** pinned as the submodule `third_party/whisper.cpp` at commit `482279b`, built with
  `native/patches/*.patch` applied: `native/build-native.sh` extracts the commit into its build directory and patches
  it there, so the submodule stays untouched. To move whisper.cpp, update the submodule and `WHISPER_COMMIT`, then
  refresh the patches until they apply.
- **Language model (variant selection):** `scripts/lm/convert_gpt2.py` converts lchaloupsky/czech-gpt2-oscar to the
  GGUF file (numpy and safetensors). `scripts/lm/make_reference.py` writes the reference data of the tests from
  PyTorch: token ids and log-probabilities of invented sentences, and a tiny random GPT-2 that the JVM tests run
  without the real model. `CZECHSPEECH_LM` points the tests at the converted file.
- **Native libraries:** prebuilt by `native/build-native.sh` into `native/out/` and not committed. The AAR build
  fails if they are missing or were built from other sources (the stamp, see above). The last build (17 Nov 2026, the language list together with streaming and the abort inside the encoder) used NDK r27c (27.2.12479018), CMake 3.22.1 and whisper.cpp
  `482279b8a18855ba741d7b88c30fcb3874691f59` with `native/patches/*.patch`; the SHA-256 of each
  library is in the CHANGELOG.
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
`/sdcard/Android/data/ai.moravian.czechspeech.sample/files/models/`, or downloads one. Its debug build has a
Benchmark screen (`BenchmarkActivity`, in `sample/src/debug`, so the release build does not contain it) that times
recognition per decoding mode (greedy, beam search, beam search with the language model
pushed as a `.gguf` to the same folder) on invented sentences spoken by the phone's text-to-speech into memory, and
shows and logs only numbers (logcat tag `CzechSpeechBench`). `--ez auto_threads false` runs it without
[thread placement](#threads-and-cores), for A/B runs; its RESULT line gives the recogniser threads Linux saw at once,
their core masks and the cores they ran on, the app's cores at the start and the end, whether the screen went off (the
screen is kept on while it runs), and `text_hash`, a short SHA-256 fingerprint of the transcripts, so that two runs
(`--ei threads 3 --ez auto_threads false` against 4 threads, say) can be compared without the text.

## Publishing to Maven Central

The release goes to the Central Portal (central.sonatype.com) under the verified namespace `ai.moravian`. The
script builds and signs the bundle, and with `--upload` sends it; it never publishes. You press Publish in the
portal. Run it in WSL or Linux (bash, python3, gpg, curl).

**Once:**

- Use a dedicated signing key (`GPG_KEY_NAME`, default the Moravian AI release key); a key with a passphrase is fine, gpg asks for it.
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

- **Unit tests:** endpointer, CPU detection, download (resume, redirect, hash), the pipeline lifecycle and previews
  (their rules, and sessions with a fake native layer, also with variant selection on: a preview stays one greedy
  pass and never frees or loads the language model).
- **Sentence joining (12 Nov 2026, JVM):** the rules are tested against the measurement's own tests, and sessions with a
  fake native layer; with the real language model (q8_0) the report's eight invented pairs decide as reported there
  (the test prints their margins next to the report's, PyTorch float32). One boundary with the full window (20 + 12 words, four
  candidates) took 104 ms on 6 threads and 415 ms on one, on the build machine (x86-64); not yet measured on a phone.
- **Integration tests on the JVM:** the real whisper.cpp, Silero VAD and CzechSpeech model on Linux x86-64.
- **Build checks:** the Android libraries pass the alignment and instruction-set checks above.
- **Variant selection (12 Nov 2026, JVM):**
  - the tokenizer gives exactly the Hugging Face token ids on 19 invented sentences, with the real vocabulary and
    with a tiny one; GPT-2 on ggml matches PyTorch within 0.2 % + 0.02 nats per sentence in f32 (a tiny random
    model) and within 0.33 nats (0.6 %) per sentence in q8_0 (the real model), keeping the order of the variants of
    a sentence;
  - the patched beam search gives five different variants of the parliament recording (the measurement found copies
    of one text with whisper.cpp unpatched), and the greedy transcripts of every integration recording are the same
    as with the previous native library;
  - on the build machine (x86-64, 6 threads) a 6.9 s utterance took 2.08 s greedy, 2.41 s with five beams and
    2.46 s with the language model (91 ms for its five variants). A phone's slower decoder makes beams relatively
    dearer there; the Benchmark screen of the `sample` app's debug build measures it.
- **Thread placement (14 Nov 2026, JVM):** the rule on the layouts in the table of [Threads and cores](#threads-and-cores)
  (with the 1.5 margin) and on unknown or partial cores; with a fake native layer that models the kernel's affinity
  (Linux 6.2+ keeping a thread's own mask across cpuset changes, a cpuset change during a computation, older kernels,
  cores parked while idle that the load wakes, cores isolated for good), every computation (final text, preview, beam
  search, language model) runs pinned on the whisper thread only, the thread is unpinned afterwards, also after a
  failure or a cancellation, top-app plans 4 threads again after the foreground, parked cores keep the plan, isolated
  ones pause pinning for 30 s, a nested placement leaves the outer pin alone, and a refused or failing affinity call
  changes nothing else. With the
  host library on Linux: only the calling thread is pinned, unpinning widens even a mask a thread was started with,
  every worker thread of a real transcription was seen on the pinned cores only (126–127 threads
  seen for 4 threads on 2 cores, 83–85 for 3 on 3: ggml starts new ones for every graph), greedy decoding and beam
  search give exactly the same on 6 threads unpinned, 4 on 2 cores and 3 on 3, and the language model's scores are
  exactly the same on 1 thread and on 2, 3 and 6 pinned to 2 cores. Not yet measured on a phone.

- **On a phone (5 Nov 2026):** the sample app on a Pixel 9a (Tensor G4), live microphone, Czech speech.
  - The SDK picked `libczechspeech_armv82_dotprod_fp16_i8mm.so`, with 4 recogniser threads.
  - The model loaded in 0.4–0.9 s.
  - Utterances of 2.2–13.5 s were transcribed in 0.8–5.1 s, that is 0.32–0.38× real time.
  - Speech detection took 1.5 ms per 32 ms frame.
  - The transcripts were checked by ear only, without a reference, so there is no WER.
  - `stop()` in the middle of speech: the word in progress was finished, and the last utterance was transcribed.

**Not yet tested:** the emulator, other phones and CPU variants, long sessions, `keepModelLoaded` and previews
on a phone (only with the fake native layer), and thread placement on a phone.

## Enterprise edition

This SDK and the free Czech model are all you need for Czech. For companies, Moravian AI also offers an enterprise
edition on request:

- **CzechSpeech Enterprise cs+sk+en (beta):** one model for Czech, Slovak and English. Its Czech word error rate is the free model's
  on Czech parliament speech (1.9 %), lower on FLEURS (7.3 % against 7.8 %) and 6.8 % against 6.7 % on noisy speech,
  a difference within the margin of the measurement; Slovak FLEURS 6.6 %, English FLEURS 6.6 %. It is a beta: see
  its model card at [huggingface.co/moravianai/CzechSpeech-Enterprise](https://huggingface.co/moravianai/CzechSpeech-Enterprise)
  for where it falls short. Force the language per call (see [Models and languages](#models-and-languages)).
- **CzechSpeech Voiceprint:** speaker recognition (who said what), fine-tuned on Czech speech. Support in this SDK is
  in preparation.
- **Fine-tuning on your vocabulary and your recordings,** with the accuracy measured on your own data before you
  decide.
- **Help with integration,** e-mail support and model updates.

It uses this same public SDK. An enterprise model is a file that you host yourself or ship in your app, and you pass
it as described in [Your own or a licensed model](#your-own-or-a-licensed-model). There is no licence server and no telemetry: audio and transcripts
stay on the device, as with the free model.

Write to info@moravian.ai.

*Česky:* toto SDK s volným českým modelem stačí na češtinu. Firmám nabízí Moravian AI na vyžádání podnikovou verzi:
- jeden model pro češtinu, slovenštinu a angličtinu, beta (čeština: v parlamentních záznamech stejná chybovost jako volný
  model, na FLEURS nižší, v šumu 6,8 % proti 6,7 %, rozdíl v mezích měření);
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

# Changelog

All notable changes to CzechSpeech for Android. The format follows [Keep a Changelog](https://keepachangelog.com/).
Until 1.0, a minor version may change the API.

## [0.3.0] - 2026-11-19

The first published version. `0.1.0` (staged locally on 4 Nov 2026) and `0.2.0` (prepared on 16-17 Nov 2026) were never
published; everything below is one list, and the `0.2.0` entries of the earlier draft are part of it. (The date above is
filled in on the day of the release, see `RELEASING.md`.)

- **API:** `StreamingConfig(maxLagMs, cutSeam)` with the `CutSeam` enum (`NONE`, `CONTEXT`, `CONTEXT_AND_HELD_WORD`; default `CONTEXT_AND_HELD_WORD`, see below; `CONTEXT` is the context alone). With
  `CONTEXT` the final recognition of a piece after a cut inside speech also gets the last 2.5 s of the piece before it, the
  words that repeat what was already sent are found by a two-word text anchor and left out, and without an anchor the piece is
  recognised from its own audio as before. The audio is kept in memory only, zeroed after that run or when nobody continues it.
  The guard counts it in the lag estimate (not in the target length of the pieces); an empty final of such a piece is repeated
  on the full window only (no repeat on 512 frames), and only if the projected cost (raised by 30 %, plus a margin of 1 s) still ends within the lag limit. The run is in the language of the piece before (a language change inside continuous speech is noticed again after a pause, a quiet cut or an anchor fallback), and a piece whose text came from a rescue hands no context on. A fallback adds a second recognition and a continuation final covers up to 14.5 s of audio: the delay guarantee has these two exceptions, as for the retry of an empty final. `CutSeam.NONE` keeps the behaviour without the context. New logcat lines and a `seams:` part of the
  session line, counts only. See the README, "Streaming transcription".
- **API, additive:** optional encryption of a self-hosted model file: `CzechSpeechModel.Encryption.aes256Ctr(keyHex, ivHex)` and a last parameter `encryption: Encryption? = null` of `CzechSpeechModel.Source` (and `Entry.source`). AES-256-CTR over the whole file, same length; the download is decrypted chunk by chunk (resume with `Range` works) and the file on the device and `sha256` are the plaintext. Source-compatible for existing calls (`@JvmOverloads` keeps the four-argument Java constructor); the `copy` and `componentN` of the `Source` data class gain the new parameter, which does not matter for 0.3.0, the first published version. It protects the file on the server only, not on the device. `scripts/encrypt-model.sh` encrypts a file or a stream and checks it against the known hash and size. Never reuse a key and IV for a second file. A JCA provider that buffers `AES/CTR` output is refused with an IOException instead of corrupting the stream. See the README, "Encrypting the hosted file".
- **API, additive:** `CutSeam.CONTEXT_AND_HELD_WORD` (the default of `StreamingConfig.cutSeam`; `CutSeam.CONTEXT` opts out): in chunk mode a piece cut inside speech is delivered without its
  last word, the next piece's text (recognised with the context) starts with its new version, and the `UtteranceDone` of the cut piece
  waits for that (it comes right before that text). A word that cannot be replaced is sent as it was, once, as one more `Transcript` of its
  piece before anything else of the next piece; the exactly-once delivery of the tracker is covered by a randomised test of event orders (it drives the tracker's bookkeeping, not the engine; the engine is covered by session tests). The text of a
  replaced word has `SentenceJoin(continuesPrevious = true, previousEnd = KEEP)`. A held word is never a preview. The `maxLagMs` guarantee holds with it (a later replacement is refused); simulated largest delay of a word at most 0.5 s longer than with `CONTEXT`.
- Fix (crash on Android): a diagnostic added to the log of a dropped recognition built a regular expression with an unescaped `}`
  that the desktop JVM accepts and Android's regex engine rejects, so every final recognition without a letter or a digit ended
  the host app with an ExceptionInInitializerError. The pattern is gone (a hand-written scan), the other patterns live in one
  place (`SdkPatterns`), unit tests refuse syntax Android does not take, and the engine compiles them all when it is created
  (a defect is an `Error` event at the start of `listen()`, a failure of `transcribe()`). Whatever the work of one utterance
  throws, an `Error` too (an OutOfMemoryError excepted: it ends the session, as before), is now an `Error` event for that utterance (and `UtteranceNotWritten` after clear speech) and the
  session goes on; it no longer reaches the collector. `UtteranceNotWritten` is now also sent for a failed recognition after
  clear speech.
- **API, additive:** `PreviousEnd` has three new values after `PERIOD_TO_COMMA`: `REMOVE_ELLIPSIS`, `ELLIPSIS_TO_COMMA` and
  `REMOVE_COMMA` (`applyTo` handles "..." and "…" and closing quotes or brackets). A client that handles `PreviousEnd` with an
  exhaustive `when` needs three branches (the rest of `SentenceJoin` is unchanged). `CzechSpeechConfig.minSilenceMs`, public
  since the first version (default 600, accepted 32..10000), now has its trade-off documented (sensible 400..2000); no change
  of range or default.
- Sentence joining: an ellipsis ("..." or "…") at the end of the previous piece is decided by the language model instead of
  being left as it is: plain continuation, comma, the ellipsis kept with a small letter, or a new sentence; the prior leans to
  removing it (another reading wins by more than 2.0, a comma by more than 1.5 over the plain continuation). A trailing comma at
  the end of the previous piece may go ("A, b" → "A b") when the model prefers the text without it by more than 1.5. After a
  quiet cut the comma is chosen only when it beats the plain continuation by more than 1.5. A piece after a cut on speech removes
  the ellipsis of the previous piece. Logcat `sentence boundary: ellipsis removed|ellipsis to comma|ellipsis kept|comma removed,
  language model N ms`; never text. All margins are unmeasured (no labelled data) and conservative.

- Review fixes: the quiet-point cut falls inside the quiet stretch that triggered it (an earlier dip of the window can no longer
  take the cut and leave the last words in a dropped remainder), keeps 128 ms of the stretch in the piece, and needs 256 ms of
  quiet (a hesitation between a preposition and its word does not cut). After a quiet cut the new sentence is compared with the
  best continuing candidate by the margin, then the comma with the plain continuation. An ellipsis is decided with the prior to
  removal only when the speeches are less than 2.0 s apart or at a quiet cut. "atd..." keeps its abbreviation period when the
  ellipsis goes. No public API change.
- Streaming, chunk mode (passes paused on a slow phone): a piece is cut as soon as a good place exists. From the floor of
  the cut window (the target less 1.5 s, never under 4 s) the first quiet stretch of 256 ms (eight frames below the endpointer's
  `negThreshold`) cuts, inside that stretch and keeping 128 ms of it in the piece, instead of waiting for the target length; without one the cut is at the target as before. The duty
  bound (60 % of the shortest possible piece), the 4 s minimum, the two-finals rule and the lag rule are unchanged. Logcat
  `streaming: forced cut …, reason: target L ms, quiet point, duty D %`.
- Streaming with sentence joining: a piece that follows a cut on a quiet frame of continuous speech (not real silence) is
  decided with a prior for going on: the plain continuation, a comma (`PreviousEnd.PERIOD_TO_COMMA`, existing) or a name
  with its capital win unless the end of the sentence is likelier by more than 2.0 (natural log P); the first word is
  lowered when the language model prefers it by more than 1.0. A piece after a cut on speech asks the language model once
  about the capital of its first word (a name keeps it) instead of always lowering it. Logcat
  `sentence boundary: join|comma|split (quiet cut), language model N ms` and
  `sentence boundary: continued (cut), lower-case|capital kept, language model N ms`; never text. After real silence nothing
  changes. No public API change (`SentenceJoin` and `PreviousEnd` are as they were).

- A final recognition without a letter or a digit is repeated on a longer encoder window when the endpointer saw clear
  speech in it (at least 1 s): 512 frames, then 1488 (the whole window within 0.8 %), at most twice, the first result with
  words wins; in streaming mode the full-window repeat waits while two finals are pending, and the repeats are booked
  apart from the guard's cost estimates. `CzechSpeechConfig.retryEmptyFinals` (true by default) turns it off. New event
  `SpeechEvent.UtteranceNotWritten(startMs)` for clear speech that ends up without text; logcat
  `utterance: empty result, retry with window N: rescued|still empty`. Cost: one more recognition per empty result; the
  full-window repeat starts only when nothing else waits (empty queue, no other final pending, no `stop()`), a repeat that
  grows the window by less than 25 % is skipped, a streamed piece that wrote words is repeated only when words were pending,
  and a repeat can exceed `maxLagMs` for that one stretch (about 4 s on 512 frames, up to 12 to 16 s more on the full window).

- Logcat: the line of a recognition dropped for having no letter or digit says what the model returned (`empty`, `bracketed`
  like `[ ... ]`, `punctuation`, `symbols`) and in how many parts, and that the language is forced; never the text or its
  length. The behaviour is unchanged.
- Model with a language list, with or without a language rule: the log line of every recognition (`utterance: ...`,
  `transcribe: ...`) now carries `language xx, ` before the thread placement (also `(forced)`, `(rescue)` or the shares, see
  "Language rule"); a one-language model writes the line as before. Variant selection (the language model choosing among the
  beam's hypotheses) and sentence joining use the language model **only for text in the first language**, with a switch or
  without: another language's text stays as Whisper wrote it, is not joined with the one before it and the joiner starts over.
  This holds for streaming too, for every kind of chunk (the first of an utterance, the first after a cut inside speech, a
  later one): such a chunk is never given to the language model.
- Build: `checkNativeLibs` (a dependency of `bundleReleaseAar` and `bundleDebugAar`, not of the unit tests) also fails when
  the libraries in `native/out` were built from other sources than this tree; `native/build-native.sh android` writes the stamp
  (`native/out/source-stamp`, made by `native/source-stamp.sh`) next to them.

### Language rule: the first language is preferred

For a model of several languages. The numbers a model needs (`LanguageSwitch`) are measured on its file; none is shipped for
any model yet.

- **API, additive:** `CzechSpeechModel.LanguageSwitch(thresholds: Map<String, Float>, minLeaveMs = 2_000, minReturnMs = 1_000)`
  (a regular class, no `copy`; keys are language codes other than the first, each threshold from 0.5 up to but not
  including 1, both times 0 to 30 000 ms, the map is copied and read-only), `CzechSpeechModel.Entry(..., languageSwitch: LanguageSwitch? = null)`
  and a last parameter `languageSwitch` of the `CzechSpeech` constructor (also a property; the constructor from an `Entry` takes it
  from the entry). A switch needs at least two languages and names only languages other than the first
  (`IllegalArgumentException` at construction otherwise). `Entry` gets a new parameter, so its generated `copy` has one
  more parameter (the four-argument constructor stays for Java and Kotlin callers: `@JvmOverloads`).
  `CzechSpeech.listenWithPreviews(previews, language: String? = null)` takes a language like `listen` (a language outside the
  list throws `IllegalArgumentException` at the call). A language asked for always wins: one code, no recognition, no rescue.
- **The rule.** With a switch, a recognition without a forced language takes the first language unless another language reaches
  its threshold, and another language is a candidate only for audio of at least `minLeaveMs`. A language without a threshold
  (Slovak, in the setting the author intends) is never chosen automatically, only when the app asks for it. Without a switch
  a model of several languages takes the most probable language of the list for every recognition, as the earlier draft did.
- **The language of the collection.** `listen()` starts in the first language and every transcript sent to the app sets the
  language the next utterance is expected in; an empty, dropped, failed or skipped result and a preview change nothing. While
  the collection is in another language, audio shorter than `minReturnMs` stays in it, longer audio has to reach the threshold
  again or the collection returns to the first language. `transcribe()` has no such state (every call starts in the first language).
- Log: a final that came out empty after its language was chosen by the rule logs `utterance: language decision of the first run: …`
  (shares, thresholds or "too short to decide", audio ms, and the rescue or why none ran; codes and numbers only).
- **Rescue after an empty result.** When a final recognition comes out empty (no letter or digit) and another language the rule
  allows has at least half of the shares, one more run is made forced in that language on the same encoder window, before the
  repeats on a longer window (which then run in the language of the last attempt). It does not depend on
  `retryEmptyFinals` or on a second of clear speech, it runs for `listen` and `transcribe`, not for early start, previews or
  streaming passes, and its text does not change the language of the collection.
- **Streaming.** A piece of speech starts in the language of the collection and keeps it until two passes in a row recognise
  another one. Passes use the rule with the piece's language and their own length; a pass commits words only when it and the
  pass before ran in the same language and the piece has had a pass long enough to decide (`minLeaveMs` from the first
  language, `minReturnMs` from another), so a short pass over foreign speech forced into the wrong language writes nothing. The
  final recognition of a piece that committed words is forced in the piece's language (no rescue); at the end of a piece the
  collection takes the language of its last text. With `minLeaveMs` over 2 000 ms the first word of a piece comes about one
  pass (a second) later for each further second; a `minLeaveMs` near or above 12 000 ms (the longest piece) lets a Czech piece
  never settle, and streaming degrades to one transcript per cut. Without a language rule nothing changes.
- **Variant selection and sentence joining work in the first language only**, with a switch or without one: see the entry
  "Model with a language list" above this heading. In another language the beam search keeps Whisper's own choice.
- **Native:** the language is decided inside `whisper_full` (a third patch of whisper.cpp,
  `native/patches/whisper.cpp-language-select.patch`): on the shortened window of the call, interruptible like any recognition,
  and the decoding reuses the encoder output, so the decision costs one step of the decoder instead of a pass of
  the encoder over the whole 30 s window. `nativeWhisperTranscribe`, `nativeWhisperPreview` and `nativeWhisperTranscribeNBest`
  take `thresholds: FloatArray?` (NaN: not a candidate; null: the most probable language) and the new
  `nativeWhisperLanguageShares` returns the shares of the last call. A list without a switch can decide a borderline
  utterance differently than the whole-window recognition of the earlier draft did.
- Logcat (languages and numbers, never text): `language en (cs 0.002, sk 0.001, en 0.997)`, `language cs (forced)` or
  `language en (rescue)` in the line of a recognition, `language: N switches, M rescues` at the end of a collection (with
  streaming also the changes of a piece's language and the passes held by the language), `language en` in a streaming
  pass line (`language en (too short to decide)` when the pass could not change its piece's language), and the line of a dropped transcript says the language and whether it was forced or chosen
  (`…; language en (chosen)`).

### Added

- **The floor of the encoder window is a setting of the model** (`CzechSpeechConfig.minEncoderWindow`, default 512 =
  no change): until now the constant `CS_MIN_AUDIO_CTX` in the native code. The Czech+Slovak+English model bears 384 frames
  without loss and recognises utterances up to 7 s in 27 % less time; the published Czech model must stay at 512
  (worse with 384, invented text from 256 down), so the floor goes with the model file
  (measured on x86 and
  on parliament speech, 384 is close to the edge: measure on the phone). A multiple of 16 from 256 to 1500, else
  IllegalArgumentException. The engine sets it on the model's context when it loads (`nativeWhisperSetMinAudioCtx`),
  so the final text, previews and early recognitions get the same; the formula of the window is unchanged (a test
  compares the default with the old constant for every length). Logcat: `model loaded … encoder window at least N
  frames`, and `window N frames` in every `utterance:` and `preview:` line. The sample app's Benchmark takes
  `--ei min_audio_ctx 384`. `nativeAudioCtx` now takes the floor as its second argument.
- **Early start of the final recognition, opt-in** (`CzechSpeechConfig.earlyStart = EarlyStart()`, default null =
  off). The final recognition of an utterance starts once the pause that may end it has lasted 256 ms
  (`EarlyStart.silenceMs`) instead of the 600 ms that end the utterance. When the pause becomes the end, that
  recognition gives the transcript, up to 350 ms sooner; when the speech goes on, it is aborted and dropped (zeroed,
  never logged, never seen by sentence joining), and the utterance is recognised at its end as usual. One recognition
  at a time, as before; none is started early behind a backlog. The recogniser gets zeros for the part of the pause
  not recorded yet, and so does every utterance ended by silence while this is on, so the text never depends on
  timing; it can differ from the text without early start (the microphone's own silence there), which has not been
  measured, hence off by default. Logcat: `utterance: … early start used, N ms saved` and, per session, `early start:
  N used, M ms saved, L too late; P shown as a preview; D discarded, A aborted, T ms of recogniser time` ("too late":
  the recognition began only after the utterance had ended, so nothing was gained; not counted as used).
  `CzechSpeechConfig` gains the parameter at the end and refuses an `EarlyStart.silenceMs` that, rounded up to 32 ms
  frames like `minSilenceMs`, would fall on the frame that ends the utterance (IllegalArgumentException: nothing
  could ever start early). With variant selection, a recognition dropped during a run of the language model
  finishes that run and scores no further window.
- **Streaming transcription, opt-in** (with the event `SpeechEvent.UtteranceDone(startMs)`: one per utterance, after its last `Transcript`) (`CzechSpeechConfig.streaming = StreamingConfig()`, default null = off). In
  continuous speech the text of words is sent while the person is still speaking: the SDK recognises the not yet
  committed audio again and again on the whisper thread (one pass at a time, the first after 1 s), commits the words
  two consecutive passes agree on and sends them as `SpeechEvent.Transcript`s; the buffer is cut at the quietest
  place when it is longer than 6 s, and the final recognition covers only the audio after the committed words. One
  utterance can therefore give several `Transcript`s (the first ones before its `SpeechEnded`): concatenate them with
  one space. In `listenWithPreviews` a `SpeechPreview` shows the words after the last chunk (`stableLength` 0).
  A piece that starts where a forced cut or the length cap ended on a frame of speech continues the one before it without the
  language model (a cut on a quiet frame, a pause shorter than `minSilenceMs`, is decided by the language model as after a pause)
  (`sentenceJoin`: `REMOVE_PERIOD`, a lowered first letter; a question or exclamation mark stays and nothing changes),
  so text is no longer cut into sentences at a cut; logcat `sentence boundary: continued (cut)`.
  Safeguard (`StreamingConfig.maxLagMs`, 10 000 to 30 000, default 15 000): the recogniser's speed is measured, the
  piece is cut and closed before a word would be sent later than that, and passes pause on a phone where a pass
  over 6 s would take more than 2 s. Early start is ignored while it is on. A word can be repeated or lost at the
  seams of the buffer inside a piece (known limit, inherited from the simulation); no cut of a piece (length cap, forced
  cut) writes a word twice. On a phone whose passes pause, the text comes in pieces cut as soon as the phone keeps up
  (cut at the quietest frame before a target, the shortest length of 5.5 to 12 s whose shortest possible piece, the target
  less 1.5 s and never under 4 s, takes at most 60 % of it to recognise: pieces of 4 to 8 s by the speed of the
  phone, longer ones when it heats up; no shorter piece is cheaper, the encoder window has a floor), the lag limit cuts
  only a phone too slow for that (8 s at the earliest, where it can split a word). The speed estimate is kept across the sessions of one
  engine. On a slow and heating phone the 15 s bound is thin (15.0 s in the simulation at 2.2 times the measured phone). Logcat
  (counts and times, never text): `pass: …` per pass, `streaming: forced cut at …` (with its reason), `streaming: passes paused …`,
  `streaming: utterance of … ms: …` per piece, `streaming: N passes …` per session. Measured in simulation on the
  cs+sk+en model only; not measured on a phone.
- **Previews from pauses** (`PreviewConfig.fromPauses`, default true; only with `earlyStart`): in a
  `listenWithPreviews()` session an early recognition that the speech ran over is not aborted but finishes and comes
  as a `SpeechPreview`; it is aborted when the next pause comes or its utterance ends before it has finished, so
  that the recognition that may give the final text starts at once. Timed previews count from the pause.
- **The stable part of a preview:** `SpeechPreview.stableLength`, `stableText`, `unstableText`: the words that two
  previews in a row of the same utterance start with (LocalAgreement-2; case and punctuation ignored). `text` is
  unchanged; the constructor gains `stableLength = 0` at the end. For previews only: nothing is final before the
  `Transcript`. **A change for code that compares previews:** `stableLength` is set in every `listenWithPreviews()`
  session, with or without early start, from the second preview of an utterance on, and it is part of `equals`,
  `hashCode` and `copy`: `SpeechPreview(startMs, text)` no longer equals a preview with the same text and a stable
  part. Compare `startMs` and `text`.
- **Constructors for Java and for code compiled against an older build:** `SpeechPreview`, `PreviewConfig` and
  `CzechSpeechConfig` keep the constructors without their newer trailing parameters (`@JvmOverloads`).
- **`PreviewConfig.continuous()`**: the first preview after 2 s, the next as soon as the previous one has ended and
  1 s of audio has come. The defaults (5 s, 5 s) are unchanged.

- **Variant selection** (`CzechSpeechConfig.variantSelection`, default null = off): Whisper proposes the five best
  variants of an utterance by beam search, and a Czech GPT-2 small (lchaloupsky/czech-gpt2-oscar, MIT, GGUF q8_0)
  picks one by log P_Whisper + λ · log P_LM + β · Whisper tokens (`VariantSelection`: `beamSize` 5,
  `languageModelWeight` λ 0.3, `lengthBonus` β 0.3, as measured; λ 0 = beam search alone). In the measurement of
  12 Nov 2026 it lowered the word error
  rate by 0.34 to 1.60 points on four test sets and fixed a quarter of the confusions of endings, i/y and diacritics.
  The language model is a separate file, loaded and freed with the CzechSpeech model; `CzechLanguageModel` gives its
  file name, size and SHA-256 for `CzechSpeechModel.download(context, CzechLanguageModel.source(url))`. It is not
  published yet. GPT-2 runs on the ggml that whisper.cpp brings (`native/src/czechspeech_lm.c`), its tokenizer in
  Kotlin (`Gpt2Tokenizer`, the same token ids as Hugging Face's). `scripts/lm/convert_gpt2.py` makes the file
  reproducibly, `scripts/lm/make_reference.py` the reference data of the tests (PyTorch). Without
  `variantSelection` the SDK makes the same native call as before (a test checks it). Time on a phone: not yet
  measured; the new Benchmark screen of the sample app's debug build measures it. Whisper's own choice stays in a
  window decoded at a fallback temperature (another scale of log-probabilities), where the variants differ only in
  case and punctuation (also with λ 0) and when the language model fails; a variant without a letter or digit is
  never chosen unless Whisper chose it (the language model gives the empty text its highest score; the measurement's
  rescore.py had the same flaw), and when Whisper's own choice has none (no speech), no other beam's text overrides it. The language model file is checked for structure when it loads (shapes, tensor
  types, at most 48 layers, a cut file) and refused with a `CzechSpeechException` instead of aborting in ggml; its
  SHA-256 is checked when it is downloaded. `languageModelFile` is optional with λ 0. `CzechSpeechConfig` gains the
  parameter at the end:
  source compatible, but its constructor and `copy` change signature for code compiled against the old one (nothing
  has been published).
- **Previews while speaking, opt-in:** `CzechSpeech.listenWithPreviews(PreviewConfig())`. While an utterance is still
  being spoken, it is recognised every few seconds (first after 5 s, then at most every 5 s, one at a time), and its
  text so far comes as a `SpeechPreview(startMs, text)` between the same `SpeechEvent`s as from `listen()`. A preview
  is unstable text: show it as provisional and never insert it as finished text. The `Transcript` with the same
  `startMs` replaces it; a preview with empty text withdraws it when the utterance gives no transcript. Final text
  goes first: no preview starts while an utterance waits for or is in its final recognition, and a preview running
  when its utterance ends is aborted and dropped, so a preview never comes after the end of its speech or after the
  transcript that replaces it; none starts in a pause, nor when it would still run at the 28 s cap, judged by the
  speed of the last preview; after `stop()` none starts and a running one is aborted at once. A preview decodes once,
  greedily, without whisper.cpp's temperature fallback (a new native call, `nativeWhisperPreview`; the final text
  keeps the fallback) and without the speech check, and previews cost recogniser time: a 28 s
  utterance gets up to 5 previews over about 75 s of audio, 2.7 times its own. Their audio is a copy in memory, zeroed
  after use, and their text is never logged. The new types are `ListenEvent` (what the new call emits: a
  `SpeechEvent` or a `SpeechPreview`; `SpeechEvent` now extends it), `SpeechPreview` and `PreviewConfig`. `listen()`
  is unchanged and runs none of it; an exhaustive `when` over `SpeechEvent` needs no new branch. With variant
  selection on, a preview is still that one greedy pass: no beam search, no language model (a test checks it with the
  fake native layer); only the final text uses them.
- **Sentence joining, opt-in** (`CzechSpeechConfig.sentenceJoining = SentenceJoining(languageModelFile)`, default
  null = off). A pause in the middle of a sentence splits it into two utterances, and Whisper often ends the first
  with a period and starts the second with a capital (reported from a phone: "…můžeš nějak. s tebou spojit…"). The
  Czech GPT-2 of variant selection now decides at each such boundary whether the new transcript continues the
  previous one of the same session, by the rule measured on 12 Nov 2026: candidates "A. B", "A b", "A B" and "A, b" on a window of 20 + 12 words; join
  when a join beats the sentence end by more than 5.56 (3.66 after "A. b"); a small letter only when it wins by more
  than 2.5; a period is never added. It fixed 42.5 % of the false periods and took the period of 0.9 % of the true
  sentence ends. Nothing is joined across sessions, across a pause over 3 s or across lost speech or an utterance
  without words, nor in previews or `transcribe()`, nor after `close()`. `Transcript.text` does not change; the decision comes in the new `Transcript.sentenceJoin`
  (`SentenceJoin`: `continuesPrevious`, `previousEnd` = `KEEP`/`REMOVE_PERIOD`/`PERIOD_TO_COMMA` with `applyTo()`,
  `continuingText` and `standaloneText`), so a client that does not read it gets exactly the text it got before,
  and one that cannot change the previous text inserts `standaloneText`; `applyTo()` changes only a period that may
  end a sentence. Only the transcript after a boundary waits, for the language model's time. The language model
  loads after the first utterance of a session (never delaying its transcript, and not at all when no other
  utterance can come, as for one sentence ended by `stop()`; variant selection, which needs it for the first
  utterance, still loads it with the model), and a failed load is tried again in the next session; without it (not
  loaded, or failing on a boundary) transcripts come without the field. Works with greedy decoding and with variant
  selection, which then share one file and one loaded model (`CzechSpeechConfig` refuses two different files). The text is never logged, and
  `SentenceJoin.toString()` hides it. The rules are `SentenceBoundary`, a port of the measurement's
  `research/dekodovani/hranice.py` with its tests, checked against that code on 176 invented boundaries with fixed
  scores (`scripts/sentences/make_reference.py`, `SentenceBoundaryReferenceTest`); an integration test checks the report's invented pairs with the
  real q8_0 model (all eight as reported) and times one boundary of 20 + 12 words: 104 ms on
  6 threads, 415 ms on one, on the build machine; not yet measured on a phone. `Transcript` and `CzechSpeechConfig` gain a parameter at the end, with a
  default: source compatible, but their constructors and `copy` change signature for code compiled against the old
  ones (nothing has been published). `SpeechEvent` gains no subtype, so an exhaustive `when` over it needs no new
  branch.
- The sample app's Benchmark screen (debug builds): `--ez auto_threads false` turns thread placement off. Its RESULT
  line gives what Linux saw of the recogniser's threads (most at once, their core masks, the cores they ran on, from
  `/proc` every 25 ms), the app's cores at the start and the end, how often the screen went off, and `text_hash`, a
  short SHA-256 fingerprint of the transcripts, to compare runs (3 against 4 threads, say) without the text. It keeps
  the screen on while it runs, and when adb starts a measurement (the `mode` extra) on a dark or locked phone it turns
  the screen on and shows over the lock screen: before, such a run measured the background's little cores.
- `CzechSpeechModel.DEFAULT_ID` (`"cs"`): the id of the model used when none is given. `DEFAULT` is that entry's
  source, and `CzechSpeech` without an entry or `languages` assumes that entry's languages.
- **A catalog of models with languages.** `CzechSpeechModel.CATALOG` maps an id to an
  `Entry(id, source, languages, tokensPerSecond)`, and `CzechSpeechModel.entry(id)` returns one or throws
  `IllegalArgumentException("unknown model: <id>")`. It holds one model, `"cs"`: the pinned Czech source
  (`CzechSpeechModel.DEFAULT`, unchanged, so a model file downloaded earlier is found in the cache), languages
  `["cs"]`, 25 tokens per second. `tokensPerSecond` is information only: the native layer does not limit tokens.
- `CzechSpeech(..., languages = listOf("cs"))`, a new last constructor parameter (also a property): the languages of
  the model, two- or three-letter lowercase whisper language codes (`cs`, `pl`, `yue`, `haw`; the rule is `[a-z]{2,3}`). The list is
  copied and a repeated code counts once (`["cs", "cs"]` is one language: no recognition of the language); an empty list or a
  code that does not follow the rule is rejected at construction.
- `listen(language: String? = null)` and `transcribe(samples, language: String? = null)` take a language of the
  model. A language outside `languages` throws `IllegalArgumentException` at the call (for `listen`, when it is
  called, not when the flow is collected). The Java signatures without it still exist (`@JvmOverloads`).
- **The language among several languages.** With one language (the Czech model) it is forced, as before: no
  recognition of the language, the same text. With several and no choice, whisper.cpp decides the language among the list
  inside the recognition itself (see "Language rule" above: the first language preferred, the thresholds, the
  collection's language). Cancelling the collection or calling `close()` stops the recognition like any other, and its
  result is discarded. `stop()` never cancels a recognition: one under way finishes and its transcript is
  delivered. A result with no language of the list fails that recognition (a `SpeechEvent.Error`); no
  language is made up.
- **Languages while streaming, with previews and with variant selection.** Every recognition of a session (the final text,
  a preview, a streaming pass, the repeat of an empty final) is made in the language of the call, by the same rule as
  `listen()` alone: with one language it is forced, nothing else runs; with several, the rule above decides, on the shortened
  window and at the cost of one decoder step. A chunk sent while speaking carries the language of the recognition its words
  come from, so does every `Transcript`.
- **A model by id.** `CzechSpeechModel.file(context, id)`, `isDownloaded(context, id)`, `download(context, id,
  onProgress)` and `delete(context, id)` take a catalog id (an unknown id throws `IllegalArgumentException("unknown
  model: <id>")` at once, before any I/O). `CzechSpeech(context, modelFile, model: CzechSpeechModel.Entry, config,
  keepModelLoaded)` takes the languages from the entry, so the app never passes them separately:
  `CzechSpeech(context, CzechSpeechModel.download(context, "cs"), CzechSpeechModel.entry("cs"))`. The constructor with
  `languages` stays for model files outside the catalog. `download(context)` with no id still downloads `DEFAULT`.
- `transcripts(language: String? = null)` passes the language to `listen(language)`, like the web `transcripts()`.
- `CzechSpeech.awaitClosed()`: suspends until the work `close()` left running has finished. That is the
  microphone, native work already under way (a model load, the speech check, an encoder pass) and the freeing of
  the model.
- `keepModelLoaded`, a new parameter of the `CzechSpeech` constructor (default `false`, also a property). The model
  then stays loaded from the first session or call that loads it until `releaseModel()` or `close()`, also when a
  session is cancelled, instead of being loaded and freed by every `listen()` session and `transcribe()` call. It is
  the web SDK's `keepModelLoaded`. Without it nothing changes.
- `CzechSpeech.releaseModel()`: frees a model kept by `keepModelLoaded` once the sessions and calls running at that
  moment have ended, at once when none is running. It never blocks. A session or call that starts afterwards keeps
  the model again; a release still waiting is then dropped. It does nothing when no model is loaded and nothing is
  running. Unlike `close()`, it does not end the instance.
- Release tooling for Maven Central: the POM now has an `scm` block, the release publication carries a javadoc jar
  (generated by AGP from the Kotlin sources), and `-Pczechspeech.sign=true` signs it with gpg (off by default).
  `scripts/release-maven-central.sh` builds and checks the signed bundle and, with `--upload`, sends it to the
  Central Portal for review; see "Publishing to Maven Central" in the README.

### Changed

- **Recogniser threads follow the cores Android lets the app use at that moment** (`CzechSpeechConfig.autoThreads`,
  default true; a new parameter at the end, source compatible, but the constructor and `copy` change signature for
  code compiled against the old ones, and nothing has been published). Android gives an app different cores by its
  state and changes them while it runs: on a Pixel 9a (Tensor G4) all eight on screen, 0–6 for a service under another
  app's window (the keyboard's floating button) and the four little cores with the screen off. The SDK always ran 4
  threads; with 0–6 one of them landed on a little core, four times slower, and held the others back, since ggml splits
  the work evenly: a 6 s utterance took 5.0 s instead of 2.6–2.8 s. Now, before every recognition (final text,
  preview, each run of the language model), the SDK reads the cores its recognition thread may use (its mask:
  `/proc/thread-self/status`, else `sched_getaffinity`) and the cores' speeds (`cpu_capacity`, else
  `cpuinfo_max_freq`, read once per instance), and compares two plans by threads × the capacity of the slowest core a
  thread ends up on: as before (`threads` threads, unpinned, but never more than allowed cores, since ggml's barriers
  spin without yielding) and the allowed cores faster than the device's little cores (the slowest and any at most a
  fifth faster), pinned. It pins only when that is at least 1.5 times better (ggml shares out matrix products
  dynamically, so the estimate overstates a slow core's delay: on the Tensor G4 the measured gap was 1.9 times, between
  the estimate's 3.9 and 1.35 times of one that adds the cores up): on the Pixel 9a 3 threads on 4–6 under the floating
  button (2.9 times), as before on screen (a tie) and with the screen off. Close cases stay as before (2+6 under 0–6:
  1.31 times; Dimensity 9300, three X4 against four threads: 1.2 times), and so does a single fast core among stronger
  little ones. Cores are counted from the mask, never from those running at a quiet moment, since phones park idle big
  cores (Qualcomm's core_ctl) and wake them under load; instead, when fewer of the pinned cores ran at the end of a
  pinned computation than there were threads (cores isolated for heat), pinning pauses for 30 s, exactly as before
  meanwhile. whisper.cpp's and ggml's worker threads, which the pinned thread starts during the call (ggml makes a
  threadpool per graph, with no cpumask of its own), inherit the pin. Afterwards the thread is unpinned to every CPU,
  which the kernel bounds by the cpuset of the moment; a mask read before is never put back, since Linux 6.2+ keeps a
  sched_setaffinity mask as the thread's own request across cpuset changes and gives it to the threads the thread
  starts. A failed unpin is logged and retried before the next plan, which meanwhile runs no more threads than the
  cores of the narrowed mask and pins nothing; a placement nested in another on the same thread leaves the pin to the
  outer one; a refused pin runs the plan as before. Devices that report no speeds (an emulator) keep `threads` threads
  wherever Android puts them. A failure to read or set the affinity never fails a recognition. `threads` stays the
  ceiling. Logcat shows the cores and their capacities once per session and the threads and core masks with each
  recognition, numbers only. The rule is `ThreadPlanner`, tested on the layouts of the Tensor G4 and G1, 4+4, 1+3+4,
  2+6, Snapdragon 8 Gen 3 (1+5+2), Dimensity 9300, eight alike and an emulator; the placement with a fake native layer
  that models the kernel's affinity (6.2+ and older, a cpuset change during a computation, cores parked while idle and
  cores isolated for good, failures, refusals, cancellation, nesting, the calling thread of every affinity call) and
  with the host library, where the
  worker threads of a real transcription were seen on the pinned cores only. Transcripts do not depend on the threads
  or the cores: the integration recordings give the same text as before, and greedy decoding and beam search on 4
  threads over 2 cores and 3 over 3 give the same as 6 unpinned. Not yet measured on a phone; the Benchmark screen's
  `--ez auto_threads false` runs without it, for A/B runs.
- **The encoder window is never shorter than 512 frames (about 10 s).** Whisper's encoder window was trimmed to the
  utterance, and the CzechSpeech model was never trained on a window that short: under 4 s of speech it invented or
  repeated sentences in 15 to 46 % of the utterances. Now the window is the utterance's length rounded up as before,
  but at least 512 frames; from 1500 frames (30 s) up it is still the full window. On the measured short utterances
  the word error rate fell from 6.28 to 3.90 % on clean audio (change -2.38 points, 95 % interval -4.14 to -1.03)
  and from 10.82 to 9.89 % in noise (-0.93 points, -1.51 to -0.43), the invented-word bursts went from 8 to 0 (clean)
  and 2 to 0 (noise), and the total processing time is 0.97 times the old one, because the temperature fallback and
  its loops no longer happen; the latency hardly changes. Needs the rebuilt native libraries (see "Native
  libraries" below). A test covers the window (`NativeAudioCtxTest`, through the new internal
  `NativeBridge.nativeAudioCtx`).

- **Built against Android 16 (API 36).** `compileSdk` is 36, with the Android Gradle plugin 8.9.3 (8.9 is the first
  line that supports API 36; 8.9.3 its last patch, with R8 fixes) and build tools 36.0.0. Gradle 8.14.4 and Kotlin 2.0.21 stay. `minSdk` stays 26. The API and the
  behaviour do not change: the release classes are the same as before, code included (`javap -c -p` of every class),
  and the AAR asks apps for no particular `compileSdk` or plugin version. The sample app targets API 36, as Google Play
  requires since 31 Aug 2026. Its screen pads itself by the action bar, the navigation bar and a display cutout, which
  Android 15 lays over apps that target API 35+ (edge to edge).
- **Cancelling a `listen()` collection never waits for native work.** The microphone closes at the end of the frame
  being read, and the collection completes at once. A recognition or model load already running finishes in the
  background, and the model is freed right after it. Before, the collection completed only after a running encoder
  pass, after a model load, or after another caller's `transcribe()`. A session cancelled before its model load
  started no longer loads the model at all. A session that ends normally, through `stop()` or the end of the audio,
  still frees the model before the flow completes.
- **`close()` guarantees, and documents, exactly this:**
  - no new session or call starts;
  - the active session is cancelled;
  - `transcribe()` calls in progress fail at once with `IllegalStateException("CzechSpeech is closed")`, and their
    recognition is aborted;
  - the microphone closes at the end of the current frame;
  - native work already running finishes in the background.

  A `listen()` collection starting while `close()` runs is either cancelled or fails with that exception; before,
  it could run on after `close()`. `close()` never blocks.
- **`transcribe()` rejects invalid input with `IllegalArgumentException`**, before any work:
  - more than 10 minutes of audio (9,600,000 samples);
  - a NaN or infinite sample;
  - a sample of magnitude above 2. From 256 up, the message says to scale 16-bit PCM (divide by 32768); below that,
    to clip or normalise the audio.
- **The utterance queue holds at most 60 s of audio.** When recognition falls behind, an utterance that does not fit
  is skipped, its audio is zeroed, and a `SpeechEvent.Error` (a `CzechSpeechException`) says which speech was
  skipped. An utterance arriving at an empty queue is always taken. Before, the queue was unbounded.
- **Microphone failures from `AudioRecord` come out as `CzechSpeechException`**, with a message saying what to do
  and the platform exception as the cause. That covers an unsupported format, a microphone that is unavailable or
  in use, and a microphone lost while recording. A permission revoked after the check stays a `SecurityException`.
  The capture thread's priority is best effort.
- **`CzechSpeechModel.download()` checks the free storage first.** What is left to download plus 100 MB must be free,
  per `StorageManager.getAllocatableBytes()` or else the usable space. Otherwise it throws an `IOException` before
  any request. When the device cannot tell the free space, the download goes ahead.
- **A ggml assertion no longer forks the app process to exec gdb or lldb.** `GGML_NO_BACKTRACE` is set before the
  native library loads. `JNI_OnLoad` also installs an abort callback that writes the assertion to logcat (FATAL,
  tag `CzechSpeech`) and to the tombstone. The native libraries were rebuilt with this on 6 Nov 2026 (see
  "Native libraries" below).
- **`suppress_blank` is off.** whisper.cpp forbids the end of the text as the first token, so a model that learned to
  answer a segment without speech with an empty transcript had to write something into silence:
  100 % of 360 segments without speech with it on, 0 % with it off (past the speech gate 12.2 % against 0 %).
  Speech transcripts do not change (parliament, FLEURS cs and short utterances the same; 0 of 1,952 utterances
  dropped), and the CzechSpeech model is not affected. An empty result is dropped without a `Transcript` event, an
  error or a retry: an empty decode ends at its first token with a log-probability near 0, which is not a failure for
  whisper.cpp's temperature fallback. Covered by `SpeechEngineTest` and `NativeParamsTest`.

- **Breaking:** `Transcript` has a fifth property, `language` (the language of the text: the forced one, or the
  detected one; `Transcript(text, startMs, endMs, processingMs, language, sentenceJoin = null)`: `language` comes before
  `sentenceJoin`, which keeps its default). Code that constructs or copies a
  `Transcript` (the four-argument constructor, `copy`, `componentN`) must now pass `language`; test fakes that build
  transcripts are affected. Code that only reads transcripts is source and binary compatible.
- Language codes (in `CzechSpeechModel.Entry`, `CzechSpeech(languages = ...)`, `listen` and `transcribe`) must be two
  or three lowercase letters, as in the web SDK; anything else (`"czech"`, `"CS"`, `"cs "`) throws
  `IllegalArgumentException` naming the code, before it reaches the native layer.
- The JNI functions `nativeWhisperTranscribe`, `nativeWhisperPreview` and `nativeWhisperTranscribeNBest` (a `languages`
  argument) and `nativeWhisperLanguage` changed with the
  libraries below, so the libraries and the AAR must be from the same build.

### Native libraries

On 19 Nov 2026 the comment text (and nothing else) in `native/src/czechspeech_jni.c` was rewritten. Two rebuilds in the environment of that day, one with and one without the comment change, produced identical libraries, so the comment does not reach the binaries. The five libraries in the table below are the ones built earlier (17 Nov 2026) and are kept as they are; the source stamp (`native/source-stamp.sh`) was renewed for the comment-only change. The rebuilds were not compared with the table.

Built on 17 Nov 2026 with `native/build-native.sh all` (NDK r27c 27.2.12479018, CMake 3.22.1, Ninja), the one build of
the release branch's language list and this branch's streaming work: the language list (`nativeWhisperTranscribe`,
`nativeWhisperPreview` and `nativeWhisperTranscribeNBest` take the comma-separated languages, `nativeWhisperLanguage` is
new), the abort inside the encoder, the floor of the encoder window kept with the context
(`nativeWhisperSetMinAudioCtx`; `nativeAudioCtx` takes the floor), thread placement (`nativeThreadCpus`,
`nativeSetThreadCpus`, `nativeUnpinThread`), the n-best call and GPT-2 (`native/src/czechspeech_lm.c`) for variant
selection, and `nativeWhisperPreview` (one greedy pass without the temperature fallback). The greedy calls and the n-best
call share `cs_full_params`; one language is forced as before (no recognition of the language, no callback), several are
decided inside `whisper_full` by `cs_language_select` and `cs_pick_language` (the third patch, see "Language rule"). The
Android libraries below were rebuilt on 18 Nov 2026 from the sources of this tree, with that patch
(`native/build-native.sh android`, which also writes `native/out/source-stamp` for `checkNativeLibs`); the host library
(JVM tests only) is the one in `native/out/host` on that day. They keep the
floor on the encoder window (`CS_MIN_AUDIO_CTX`), `suppress_blank = false`, and the ggml crash fix of 6 Nov 2026
(`JNI_OnLoad` sets `GGML_NO_BACKTRACE` and installs the abort callback). whisper.cpp is still commit
`482279b8a18855ba741d7b88c30fcb3874691f59`, built with `native/patches/whisper.cpp-czechspeech.patch` and
`native/patches/whisper.cpp-encoder-abort.patch` applied (in the build directory; the submodule is untouched). The script's
checks passed: 16 KB LOAD alignment, no instructions above each variant's level, no build-machine paths. SHA-256 of the
stripped libraries:

| library | SHA-256 |
|---|---|
| `libczechspeech_armv8a.so` | `52f146af9633df634b8d5bfb2a55f1381bdfe9bd2cf4972c52b63cdd0a599338` |
| `libczechspeech_armv82_dotprod_fp16.so` | `ad848b90a05a94e8220a5edf6e28dba776a831da69bd3447dffb52888d5f8c90` |
| `libczechspeech_armv82_dotprod_fp16_i8mm.so` | `196398f36e0e130cf745ecf2bdd7cb495fda9f9973ae191ecbcea275aa18f636` |
| `libczechspeech_x86_64.so` | `d3f91c4c963dcff4ae8741431cdb8f65834c429c0a068cc7da4e494cb82b248e` |
| `libczechspeech_host.so` (JVM tests only) | `d8e72b5e903f834ae6222b747967a02a376d93805fd887915bd69b7717fc4652` |

Libraries built earlier (4, 6, 9, 12, 14 or 15 Nov 2026) do not match these JNI signatures; do not mix them with this AAR.

### Tests

- The recordings the integration tests read from `CZECHSPEECH_TEST_AUDIO` have English names: `a-silence-5s`,
  `b-pink-noise-2s`, `c-hum-2s`, `d-clicks-6s`, `e-rec` and `f-rec-plus-silence15s` (before: `a-ticho-5s`,
  `b-ruzovy-sum-2s`, `c-brum-2s`, `d-cvaky-6s`, `f-rec-plus-ticho15s`). The audio is the same; rename the files.

### Fixed

- **An aborted preview no longer holds up the final text for the rest of its encoder pass.** whisper.cpp looked at
  the abort callback only after the encoder pass and after each decoder step, so a preview aborted during the encoder
  (seconds on a phone) kept the whisper thread until the pass ended: the final text waited 2 to 4 s behind it.
  `native/patches/whisper.cpp-encoder-abort.patch` hands the callback to ggml for the encoder's three graphs, where
  it is checked after every node. Logcat now says how long each abort took (`stopped N ms after the abort`) and, at
  the end of a session that had any, their number and the slowest (`aborts: N, the slowest stopped the recogniser
  after M ms`); only aborts that stopped whisper.cpp at work are counted. The native libraries have to be rebuilt for
  this.
- An utterance handed to a recognition that was cancelled before it ran was never zeroed. It now is, like
  queued and skipped utterances.
- A session that ended before capture started (a model load that failed at once) left its audio source unclosed.
- An `AudioRecord` that would not start recording stayed allocated until the session ended.
- Android lint warned about `UsableSpace` in `ModelDownloader`. Its `File.getUsableSpace` default is gone: the
  download is always told the free space, which `CzechSpeechModel.download()` takes from
  `StorageManager.getAllocatableBytes()` first.

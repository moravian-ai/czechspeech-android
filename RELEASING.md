# Releasing CzechSpeech for Android

Both SDKs are released together and with the same version number: this one to Maven Central, the web SDK
(`czechspeech-web`, its own `RELEASING.md`) to npm. A version published to Maven Central can never be deleted or
replaced; a mistake is fixed by a new version number.

The release needs things only the maintainer has: the GitHub organisation, the Central Portal account and its user
token, and the GPG signing key with its passphrase. Nothing here is done by CI.

## What must already be true

- The model the SDK downloads is public: `CzechSpeechModel.DEFAULT` points at a pinned revision of
  <https://huggingface.co/moravianai/CzechSpeech> (size and SHA-256 in `CzechSpeechModel.kt`).
- The namespace `ai.moravian` shows as verified at <https://central.sonatype.com/publishing/namespaces>.
- `~/.m2/settings.xml` has the Central Portal user token as the server `central`, and the signing key has a
  passphrase (README, "Publishing to Maven Central", "Once").

## 1. Prepare the release commit (WSL or Linux)

```bash
git switch main && git merge --ff-only vydani-0.2.0      # or merge the release branch the usual way
git submodule update --init
native/build-native.sh all                               # NDK r27c, CMake 3.22.1
sha256sum native/out/jniLibs/*/*.so native/out/host/*/*.so
```

- The SHA-256 of the five libraries must equal the table in `CHANGELOG.md` ("Native libraries"). If they differ,
  find out why before going on; if the difference is intended, update the table.
- `CHANGELOG.md` has the heading `## [<version>] - <date>` with the date of the release.
- The version is the same in `czechspeech/build.gradle.kts` (`version`), `ModelDownloader.USER_AGENT`,
  `sample/build.gradle.kts` (`versionName`) and the README (the first paragraph and the install snippet).

```bash
./gradlew :czechspeech:testReleaseUnitTest :czechspeech:lintRelease :sample:assembleRelease
scripts/test-release-maven-central.sh                    # the release script's guards; no network, no real key
git status --short                                       # must print nothing: the release script refuses a dirty tree
```

## 2. Make the repository public and push

The POM's `scm` block and the README link to <https://github.com/moravian-ai/czechspeech-android>. Make the
repository public in GitHub's settings (Settings, General, Danger Zone, Change visibility) after a last look at
the history, the branches on the remote and the issues, then:

```bash
git push origin main
```

## 3. Maven Central

```bash
scripts/release-maven-central.sh             # builds, signs (gpg asks for the passphrase) and checks the bundle
```

It prints the bundle's path and SHA-256 and uploads nothing. Look into the zip: the AAR with the four
`jni/*/libczechspeech_*.so`, the POM, the sources and javadoc jars and the Gradle module file, each with `.asc`,
`.md5` and `.sha1`.

```bash
scripts/release-maven-central.sh --upload    # the same, then uploads as USER_MANAGED and waits for VALIDATED
```

Then open <https://central.sonatype.com/publishing/deployments>, check the deployment and press **Publish**. The
script never publishes. The artifact appears on `repo1.maven.org` some minutes later.

## 4. Check the published artifact from an empty project

```bash
curl -sI https://repo1.maven.org/maven2/ai/moravian/czechspeech-android/<version>/czechspeech-android-<version>.pom | head -1
```

Create a new Android project (or use any app) with only `google()` and `mavenCentral()` as repositories, add
`implementation("ai.moravian:czechspeech-android:<version>")`, build it, and on a phone run
`CzechSpeechModel.download(context)` followed by one `listen()` session: the model downloads from Hugging Face, its
SHA-256 is verified by the SDK, and a spoken sentence comes back as a transcript.

## 5. Tag

```bash
git tag -a v<version> -m "CzechSpeech for Android <version>"
git push origin v<version>
```

Then release the web SDK (or do its steps in parallel), and update <https://moravian.ai/czechspeech/> if it names
versions.

## What the script does, and what it leaves to you

| step | who |
|---|---|
| refuses a dirty tree, a SNAPSHOT version and a version without a CHANGELOG heading | script |
| builds the release AAR, the sources and javadoc jars, the POM and the module file in a fresh temporary folder | script |
| signs every file with the GPG key (`GPG_KEY_NAME`, default the Moravian AI key); you type the passphrase | script and you |
| checks that every file has `.asc`, `.md5`, `.sha1` and that every signature verifies | script |
| with `--upload`: reads the token from `settings.xml` (never printed), uploads, waits for `VALIDATED` or `FAILED` | script |
| rebuilding the native libraries and comparing their SHA-256 with the CHANGELOG | you (step 1) |
| making the repository public, `git push`, pressing Publish in the portal, the tag | you |

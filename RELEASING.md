# Releasing CzechSpeech for Android

Both SDKs are released together and with the same version number: this one to Maven Central, the web SDK
(`czechspeech-web`, its own `RELEASING.md`) to npm. A version published to Maven Central can never be deleted or
replaced; a mistake is fixed by a new version number.

The release needs things only the maintainer has: the GitHub organisation, the Central Portal account and its user
token, and the GPG signing key. Nothing here is done by CI.

The first published version is **0.3.0**. `0.1.0` and `0.2.0` were prepared and never published; their deployments in
the Central Portal (if any are still there as "Validated") are dropped before the first upload (step 0).

## Where things live

- `origin` is a private development repository with the whole history.
- `public` is `moravian-ai/czechspeech-android`, the repository the POM and the README point to. It holds a squashed
  history: one commit per release, written by `Moravian AI <info@moravian.ai>` with a tree equal to the release branch,
  each on top of the one before, so it is pushed without force. The development history stays private.
- The release branch is `vydani-<version>`, its public form `vydani-<version>-squash`. A `vydani-<version>-squash` that
  already exists on `origin` is only a preview: step 2 makes it again after the date is filled in.
- **Never** `git push public --tags`, `--follow-tags`, `--all` or `--mirror`: the development repository has local tags on
  development commits. Push one named branch or one named tag.

## What must already be true

- The model the SDK downloads is public: `CzechSpeechModel.DEFAULT` points at a pinned revision of
  <https://huggingface.co/moravianai/CzechSpeech> (size and SHA-256 in `CzechSpeechModel.kt`).
- The namespace `ai.moravian` shows as verified at <https://central.sonatype.com/publishing/namespaces>.
- `~/.m2/settings.xml` has the Central Portal user token as the server `central`; the signing key is in the WSL keyring
  (`GPG_KEY_NAME`, default the Moravian AI release key; gpg asks for its passphrase when it has one) and its public part
  is on a keyserver Central reads (<https://keys.openpgp.org> or `keyserver.ubuntu.com`).

## 0. Drop the old deployments

In <https://central.sonatype.com/publishing/deployments> every deployment of an earlier, unpublished version
(`czechspeech-android-0.2.0`, "Validated") is dropped with **Drop**. Nothing of them was published; a deployment left
there only confuses the next check.

## 1. Fill in the release commit

All of it is committed on `vydani-<version>`, then the public form is made again (step 2). Git runs in Git Bash (a
worktree does not work in WSL: its `.git` points at `C:/…`); Gradle and the tests run in WSL (a JDK and the Android SDK).

- **The model catalogue.** The public catalogue holds the free Czech model only. No enterprise model (the Czech, Slovak
  and English one) goes into `CzechSpeechModel.CATALOG`: it is gated on Hugging Face and a licensed customer hosts the
  file himself and passes an `Entry` (README, "Your own or a licensed model"). Nothing to add here for the release.
- **The date.** `CHANGELOG.md` has the heading `## [<version>] - YYYY-MM-DD`; on the day of the release the placeholder
  becomes the date, and it is committed (Git Bash):

  ```bash
  sed -i "s/^## \[0.3.0\] - YYYY-MM-DD/## [0.3.0] - $(date +%F)/" CHANGELOG.md
  git diff --stat                                          # only CHANGELOG.md, one line
  git commit -am "CHANGELOG: 0.3.0 dated"
  ```

  `scripts/release-maven-central.sh` refuses a heading without a real date.
- **The version** is the same in `czechspeech/build.gradle.kts` (`version`), `ModelDownloader.USER_AGENT`,
  `sample/build.gradle.kts` (`versionName`) and the README (the first paragraph and the install snippet).
- **The native libraries.** `git submodule update --init; native/build-native.sh all` (NDK r27c, CMake 3.22.1), then
  `sha256sum native/out/jniLibs/*/*.so native/out/host/*/*.so`: the five SHA-256 must equal the table in `CHANGELOG.md`
  ("Native libraries"). If they differ, find out why before going on; if the difference is intended, update the table.
  (Libraries already built and checked can be copied into `native/out` of the worktree; the folder is not tracked.)

Tests, in WSL (one heavy job at a time, offline):

```bash
./gradlew --offline --max-workers=4 :czechspeech:testReleaseUnitTest :czechspeech:lintRelease :sample:assembleRelease
scripts/test-release-maven-central.sh                    # the release script's guards and its whole run; no network, no real key
```

`PublicApiTest` inside the unit tests compares the public API with `czechspeech/api/public-api.txt` (a change of the
API makes it fail; after a deliberate change run it once with `CZECHSPEECH_UPDATE_API=1` and commit the listing).
`git status --short` must then print nothing: the release script refuses a dirty tree.

## 2. The public form of the release

One commit on top of the public history, with the tree of the release branch. It stops at the first failed check and
moves no branch before they all pass (Git Bash):

```bash
bash -euo pipefail <<'EOF'
V=0.3.0
git fetch public
c=$(GIT_AUTHOR_NAME="Moravian AI" GIT_AUTHOR_EMAIL=info@moravian.ai GIT_COMMITTER_NAME="Moravian AI" GIT_COMMITTER_EMAIL=info@moravian.ai \
    git commit-tree "vydani-$V^{tree}" -p public/main -F <message file>)
[ -n "$c" ] && [ "$c" != "$(git rev-parse "vydani-$V")" ]                     # a new commit, not the development head
[ "$(git rev-parse "$c^{tree}")" = "$(git rev-parse "vydani-$V^{tree}")" ]    # the tree of the release branch
[ "$(git rev-parse "$c^")" = "$(git rev-parse public/main)" ]                  # on top of the public history
[ "$(git log -1 --format='%ae%n%ce%n%an%n%cn' "$c" | sort -u | tr '\n' ' ')" = "Moravian AI info@moravian.ai " ]   # Moravian AI only
git branch -f "vydani-$V-squash" "$c"
git push public "vydani-$V-squash:main"                  # a plain push; the repository may still be private
EOF
```

The alternative with a new root (no `-p public/main`, then `git push --force public …`) only after the same checks (the
parent one dropped), and only while the repository is private.

## 3. Maven Central

The script runs in WSL from a clean clone of the squash commit, made from the **main clone, not a worktree** (the branch
is shared with the worktree, as all local branches are), with `native/out` copied in from the checked build:

```bash
rm -rf ~/czechspeech-release && git clone --branch vydani-<version>-squash <main clone> ~/czechspeech-release/android
test "$(git -C ~/czechspeech-release/android rev-parse HEAD)" = "$(git -C <main clone> rev-parse public/main)"   # the public commit
mkdir -p ~/czechspeech-release/android/native/out && cp -r <worktree>/native/out/. ~/czechspeech-release/android/native/out/   # check the SHA-256 again
cd ~/czechspeech-release/android
scripts/release-maven-central.sh --upload    # builds, signs, checks the bundle, uploads as USER_MANAGED and waits for VALIDATED
```

The script prints the bundle's path and SHA-256: look into that zip (the one that was uploaded; the signatures carry a
time stamp, so a second build would be another file): the AAR with the `jni/*/libczechspeech_*.so`, the POM, the sources and
javadoc jars and the Gradle module file, each with `.asc`, `.md5` and `.sha1`. Without `--upload` it only builds.

Then open <https://central.sonatype.com/publishing/deployments>, check the deployment and press **Publish**. The
script never publishes (and, since 0.3.0, ends with exit code 0 after a successful upload). The artifact appears on
`repo1.maven.org` some minutes later. A failed validation is fixed in the release branch and the squash is made again
(step 2); that is why the repository is made public only after the publication (step 4).

## 4. Make the repository public

The POM's `scm` block and the README link to <https://github.com/moravian-ai/czechspeech-android>. After a last look at
the files and the branches (`git ls-remote public`): Settings, General, Danger Zone, Change visibility, or
`gh repo edit moravian-ai/czechspeech-android --visibility public`.

## 5. Check the published artifact from an empty project

```bash
curl -sI https://repo1.maven.org/maven2/ai/moravian/czechspeech-android/<version>/czechspeech-android-<version>.pom | head -1
```

Create a new Android project (or use any app) with only `google()` and `mavenCentral()` as repositories, add
`implementation("ai.moravian:czechspeech-android:<version>")`, build it, and on a phone run
`CzechSpeechModel.download(context)` followed by one `listen()` session: the model downloads from Hugging Face, its
SHA-256 is verified by the SDK, and a spoken sentence comes back as a transcript.

## 6. Tag

The tagger of an annotated tag is the git identity of whoever runs it, so it is set explicitly and checked before the push:

```bash
bash -euo pipefail <<'EOF'
git fetch public
[ "$(git rev-parse vydani-0.3.0-squash)" = "$(git rev-parse public/main)" ]       # the commit that was released
GIT_COMMITTER_NAME="Moravian AI" GIT_COMMITTER_EMAIL=info@moravian.ai git tag -a v<version> -m "CzechSpeech for Android <version>" vydani-<version>-squash
git cat-file -p v<version> | grep '^tagger'              # must print: tagger Moravian AI <info@moravian.ai> …
git push public v<version>
EOF
```

Then release the web SDK (or do its steps in parallel), and publish the GitHub release (its notes link to the npm
package: after `npm publish`). Update <https://moravian.ai/czechspeech/> if it names versions.

## What the script does, and what it leaves to you

| step | who |
|---|---|
| refuses a dirty tree, a SNAPSHOT version, a version without a CHANGELOG heading and a heading without a date | script |
| builds the release AAR, the sources and javadoc jars, the POM and the module file in a fresh temporary folder | script |
| signs every file with the GPG key (`GPG_KEY_NAME`, default the Moravian AI release key); you type the passphrase if it has one | script and you |
| checks that every file has `.asc`, `.md5`, `.sha1` and that every signature verifies | script |
| with `--upload`: reads the token from `settings.xml` (never printed), uploads, waits for `VALIDATED` or `FAILED` | script |
| rebuilding the native libraries and comparing their SHA-256 with the CHANGELOG | you (step 1) |
| the date in the CHANGELOG, the public commit, pressing Publish in the portal, making the repository public, the tag | you |

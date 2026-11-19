#!/usr/bin/env bash
# Tests for release-maven-central.sh: the guards, the settings.xml parser, the bundle checker and the upload
# request handling (against a stub curl). Nothing here touches the network, the real gpg keyring or the real
# settings.xml: gpg runs with a throwaway GNUPGHOME and every token below is a fake.
#
#   scripts/test-release-maven-central.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/release-maven-central.sh"
WORK="$(mktemp -d)"
export GNUPGHOME="$WORK/gnupg"
mkdir -m 700 "$GNUPGHOME"
trap 'gpgconf --kill gpg-agent >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

FAKE_USER="fakeuser-7Qx9"
FAKE_PASS="fakepass-Zk3L2"
PASS=0
FAIL=0

ok() { PASS=$((PASS + 1)); echo "  ok   $1"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL $1"; }

check() { # name, command...
    local name="$1"; shift
    if "$@"; then ok "$name"; else bad "$name"; fi
}

no_secret_in() { # files...
    ! grep -q -e "$FAKE_USER" -e "$FAKE_PASS" "$@"
}

if [[ ! -f "$SCRIPT" ]]; then
    echo "missing $SCRIPT"
    exit 1
fi
# shellcheck source=release-maven-central.sh
source "$SCRIPT"
set +e +u   # the script under test sets -e; the harness wants to see failures

settings() { # file, body of <servers>
    printf '<?xml version="1.0" encoding="UTF-8"?>\n<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0">\n<servers>\n%s\n</servers>\n</settings>\n' "$2" >"$1"
}

echo "settings.xml parser"
S="$WORK/settings"
mkdir -p "$S"

settings "$S/ok.xml" "<server><id>other</id><username>x</username><password>y</password></server>
<server><id>central</id><username>$FAKE_USER</username><password>$FAKE_PASS</password></server>"
CENTRAL_USER=""; CENTRAL_PASS=""
SETTINGS_XML="$S/ok.xml" load_credentials >"$WORK/o" 2>"$WORK/e"
check "valid file: exit 0" test $? -eq 0
check "valid file: user parsed" test "$CENTRAL_USER" = "$FAKE_USER"
check "valid file: password parsed" test "$CENTRAL_PASS" = "$FAKE_PASS"
check "valid file: nothing printed" no_secret_in "$WORK/o" "$WORK/e"

settings "$S/noserver.xml" "<server><id>other</id><username>$FAKE_USER</username><password>$FAKE_PASS</password></server>"
(SETTINGS_XML="$S/noserver.xml" load_credentials) >"$WORK/o" 2>"$WORK/e"
check "no central server: exit 1" test $? -eq 1
check "no central server: message names the server id" grep -q "central" "$WORK/e"
check "no central server: no token on output" no_secret_in "$WORK/o" "$WORK/e"

settings "$S/placeholder.xml" "<server><id>central</id><username>$FAKE_USER</username><password>\${env.CENTRAL_PASSWORD}</password></server>"
(SETTINGS_XML="$S/placeholder.xml" load_credentials) >"$WORK/o" 2>"$WORK/e"
check "placeholder password: exit 1" test $? -eq 1
check "placeholder password: message names password" grep -q "password" "$WORK/e"
check "placeholder: no token on output" no_secret_in "$WORK/o" "$WORK/e"
check "placeholder: value not echoed" bash -c "! grep -q CENTRAL_PASSWORD '$WORK/e'"

settings "$S/emptyuser.xml" "<server><id>central</id><username>  </username><password>$FAKE_PASS</password></server>"
(SETTINGS_XML="$S/emptyuser.xml" load_credentials) >"$WORK/o" 2>"$WORK/e"
check "empty username: exit 1" test $? -eq 1
check "empty username: message names username" grep -q "username" "$WORK/e"
check "empty username: no token on output" no_secret_in "$WORK/o" "$WORK/e"

settings "$S/nopass.xml" "<server><id>central</id><username>$FAKE_USER</username></server>"
(SETTINGS_XML="$S/nopass.xml" load_credentials) >"$WORK/o" 2>"$WORK/e"
check "missing password element: exit 1" test $? -eq 1
check "missing password: no token on output" no_secret_in "$WORK/o" "$WORK/e"

printf 'not xml at all %s' "$FAKE_PASS" >"$S/broken.xml"
(SETTINGS_XML="$S/broken.xml" load_credentials) >"$WORK/o" 2>"$WORK/e"
check "unparseable file: exit 1" test $? -eq 1
check "unparseable file: content not echoed" no_secret_in "$WORK/o" "$WORK/e"

(SETTINGS_XML="$S/does-not-exist.xml" load_credentials) >"$WORK/o" 2>"$WORK/e"
check "missing file: exit 1" test $? -eq 1

echo "repository guards"
R="$WORK/repo"
mkdir -p "$R/czechspeech"
git -C "$R" init -q
git -C "$R" config user.email t@example.invalid
git -C "$R" config user.name t
printf 'version = "1.2.3"\n' >"$R/czechspeech/build.gradle.kts"
printf '# Changelog\n\n## Unreleased\n\n## [1.2.3] - 2026-11-07\n' >"$R/CHANGELOG.md"
cat >"$R/gradlew" <<EOS
#!/usr/bin/env bash
touch "$WORK/gradlew-ran"
exit 1
EOS
chmod +x "$R/gradlew"
git -C "$R" add -A && git -C "$R" commit -qm init

check "project_version reads build.gradle.kts" test "$(REPO_DIR="$R" project_version)" = "1.2.3"

(REPO_DIR="$R" check_clean_tree) >"$WORK/o" 2>"$WORK/e"
check "clean tree accepted" test $? -eq 0
echo x >"$R/untracked.txt"
(REPO_DIR="$R" check_clean_tree) >"$WORK/o" 2>"$WORK/e"
check "dirty tree (untracked file) refused" test $? -eq 1
rm "$R/untracked.txt"
echo y >>"$R/CHANGELOG.md"
(REPO_DIR="$R" check_clean_tree) >"$WORK/o" 2>"$WORK/e"
check "dirty tree (modified file) refused" test $? -eq 1
git -C "$R" checkout -q CHANGELOG.md

(REPO_DIR="$R" check_version 1.2.3-SNAPSHOT) >"$WORK/o" 2>"$WORK/e"
check "SNAPSHOT version refused" test $? -eq 1
(REPO_DIR="$R" check_version 9.9.9) >"$WORK/o" 2>"$WORK/e"
check "version without CHANGELOG heading refused" test $? -eq 1
(REPO_DIR="$R" check_version 1.2.3) >"$WORK/o" 2>"$WORK/e"
check "version with CHANGELOG heading accepted" test $? -eq 0
(REPO_DIR="$R" check_version 1.2) >"$WORK/o" 2>"$WORK/e"
check "version that is only a prefix of a heading refused" test $? -eq 1
printf '# Changelog\n\n## [1.2.5] - YYYY-MM-DD\n' >"$R/CHANGELOG.md"
(REPO_DIR="$R" check_version 1.2.5) >"$WORK/o" 2>"$WORK/e"
check "heading with the placeholder date refused" test $? -eq 1
check "placeholder date: message asks for the date" grep -q "no date" "$WORK/e"
printf '# Changelog\n\n## [1.2.5]\n' >"$R/CHANGELOG.md"
(REPO_DIR="$R" check_version 1.2.5) >"$WORK/o" 2>"$WORK/e"
check "heading without any date refused" test $? -eq 1
git -C "$R" checkout -q CHANGELOG.md

echo "whole script refuses before building"
rm -f "$WORK/gradlew-ran"
echo x >"$R/untracked.txt"
REPO_DIR="$R" bash "$SCRIPT" >"$WORK/o" 2>"$WORK/e"
check "script on a dirty tree: exit 1" test $? -eq 1
check "script on a dirty tree: gradle not started" test ! -e "$WORK/gradlew-ran"
rm "$R/untracked.txt"
sed -i 's/1.2.3/1.2.3-SNAPSHOT/' "$R/czechspeech/build.gradle.kts"
git -C "$R" commit -qam snapshot
REPO_DIR="$R" bash "$SCRIPT" >"$WORK/o" 2>"$WORK/e"
check "script on a SNAPSHOT version: exit 1" test $? -eq 1
check "script on a SNAPSHOT version: gradle not started" test ! -e "$WORK/gradlew-ran"
REPO_DIR="$R" bash "$SCRIPT" --bogus >"$WORK/o" 2>"$WORK/e"
check "unknown option: exit 2" test $? -eq 2

echo "bundle checker"
gpg --batch --passphrase '' --quick-gen-key 'Test <test@example.invalid>' rsa2048 sign never >/dev/null 2>&1
KEY="$(gpg --list-keys --with-colons 2>/dev/null | awk -F: '/^fpr/ {print $10; exit}')"
check "throwaway key created" test -n "$KEY"

B="$WORK/bundle/ai/moravian/demo/1.2.3"
mkdir -p "$B"
for f in demo-1.2.3.aar demo-1.2.3.pom; do
    echo "content of $f" >"$B/$f"
    gpg --batch --yes --local-user "$KEY" --armor --detach-sign -o "$B/$f.asc" "$B/$f" 2>/dev/null
    for f2 in "$f" "$f.asc"; do
        md5sum "$B/$f2" | cut -d' ' -f1 >"$B/$f2.md5"
        sha1sum "$B/$f2" | cut -d' ' -f1 >"$B/$f2.sha1"
    done
done
(check_bundle_dir "$B") >"$WORK/o" 2>"$WORK/e"
check "complete bundle accepted" test $? -eq 0
(verify_signatures "$B") >"$WORK/o" 2>"$WORK/e"
check "good signatures verify" test $? -eq 0

rm "$B/demo-1.2.3.pom.asc"
(check_bundle_dir "$B") >"$WORK/o" 2>"$WORK/e"
check "missing .asc refused" test $? -eq 1
check "missing .asc: message names the file" grep -q "demo-1.2.3.pom" "$WORK/e"
gpg --batch --yes --local-user "$KEY" --armor --detach-sign -o "$B/demo-1.2.3.pom.asc" "$B/demo-1.2.3.pom" 2>/dev/null

rm "$B/demo-1.2.3.aar.sha1"
(check_bundle_dir "$B") >"$WORK/o" 2>"$WORK/e"
check "missing .sha1 refused" test $? -eq 1
sha1sum "$B/demo-1.2.3.aar" | cut -d' ' -f1 >"$B/demo-1.2.3.aar.sha1"

rm "$B/demo-1.2.3.aar.md5"
(check_bundle_dir "$B") >"$WORK/o" 2>"$WORK/e"
check "missing .md5 refused" test $? -eq 1
md5sum "$B/demo-1.2.3.aar" | cut -d' ' -f1 >"$B/demo-1.2.3.aar.md5"

echo "tampered" >>"$B/demo-1.2.3.aar"
(verify_signatures "$B") >"$WORK/o" 2>"$WORK/e"
check "tampered file fails gpg --verify" test $? -eq 1

mkdir -p "$WORK/empty"
(check_bundle_dir "$WORK/empty") >"$WORK/o" 2>"$WORK/e"
check "empty bundle directory refused" test $? -eq 1

echo "zip layout"
echo "content of demo-1.2.3.aar" >"$B/demo-1.2.3.aar"
make_zip "$WORK/bundle" ai/moravian/demo/1.2.3 "$WORK/out.zip" >"$WORK/o" 2>"$WORK/e"
check "zip made" test -s "$WORK/out.zip"
check "zip keeps the Maven path" python3 -c "
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
assert 'ai/moravian/demo/1.2.3/demo-1.2.3.aar' in names, names
assert all(n.startswith('ai/moravian/demo/1.2.3/') for n in names), names
" "$WORK/out.zip"

echo "upload and status (stub curl, no network)"
STUB="$WORK/stub"
mkdir -p "$STUB"
cat >"$STUB/curl" <<'EOS'
#!/usr/bin/env bash
# Stub: records its argv, checks the -K config (mode, header) and answers like Central would.
echo "$@" >>"$STUB_LOG.argv"
cfg=""
out=""
prev=""
for a in "$@"; do
    [[ "$prev" == "-K" ]] && cfg="$a"
    [[ "$prev" == "-o" ]] && out="$a"
    prev="$a"
done
if [[ -n "$cfg" ]]; then
    stat -c '%a' "$cfg" >>"$STUB_LOG.mode"
    expected="$(printf '%s:%s' "$STUB_USER" "$STUB_PASS" | base64 -w0)"
    grep -q "Authorization: Bearer $expected" "$cfg" && echo token-correct >>"$STUB_LOG.cfg"
fi
case "$*" in
    *publisher/upload*) body="deploy-1234" ;;
    *publisher/status*)
        n=$(($(cat "$STUB_LOG.n" 2>/dev/null || echo 0) + 1)); echo "$n" >"$STUB_LOG.n"
        if [[ "$n" -lt 3 ]]; then body='{"deploymentId":"deploy-1234","deploymentState":"VALIDATING"}'
        else body="{\"deploymentId\":\"deploy-1234\",\"deploymentState\":\"${STUB_FINAL:-VALIDATED}\",\"errors\":{\"pkg:maven/ai.moravian/x@1\":[\"boom\"]}}"; fi ;;
esac
[[ -n "$out" ]] && printf '%s' "$body" >"$out"
printf '200'
EOS
chmod +x "$STUB/curl"
export STUB_USER="$FAKE_USER" STUB_PASS="$FAKE_PASS"

export STUB_LOG="$WORK/stublog"
export SETTINGS_XML="$S/ok.xml"
CENTRAL_USER=""; CENTRAL_PASS=""
load_credentials
(PATH="$STUB:$PATH" POLL_INTERVAL=0 upload_and_wait "$WORK/out.zip" 1.2.3) >"$WORK/o" 2>"$WORK/e"
check "upload and poll to VALIDATED: exit 0" test $? -eq 0
check "reports state VALIDATED" grep -q "VALIDATED" "$WORK/o"
check "prints the portal URL" grep -q "https://central.sonatype.com/publishing/deployments" "$WORK/o"
check "token never in argv" no_secret_in "$STUB_LOG.argv"
check "token never on stdout or stderr" no_secret_in "$WORK/o" "$WORK/e"
check "token only in the config file, correctly encoded" grep -q token-correct "$STUB_LOG.cfg"
check "config file is 0600" bash -c "! grep -qv '^600$' '$STUB_LOG.mode'"
check "upload URL carries USER_MANAGED and the name" grep -q "publishingType=USER_MANAGED&name=czechspeech-android-1.2.3" "$STUB_LOG.argv"
check "no verbose or header dump flags" bash -c "! grep -q -e ' -v' -e ' -i' -e '--dump-header' -e ' -D' '$STUB_LOG.argv'"

rm -f "$STUB_LOG".*
(PATH="$STUB:$PATH" POLL_INTERVAL=0 STUB_FINAL=FAILED upload_and_wait "$WORK/out.zip" 1.2.3) >"$WORK/o" 2>"$WORK/e"
check "FAILED deployment: exit 1" test $? -eq 1
check "FAILED deployment: errors printed" grep -q "boom" "$WORK/o" "$WORK/e"

echo "whole script as it runs for real (set -euo pipefail; stub gradle and curl, throwaway key)"
R2="$WORK/repo2"
mkdir -p "$R2/czechspeech"
git -C "$R2" init -q
git -C "$R2" config user.email t@example.invalid
git -C "$R2" config user.name t
printf 'version = "1.2.3"\n' >"$R2/czechspeech/build.gradle.kts"
printf '# Changelog\n\n## [1.2.3] - 2026-11-07\n' >"$R2/CHANGELOG.md"
# the stub of the Gradle run writes a signed demo bundle where the script asked for the Maven repository
cat >"$R2/gradlew" <<EOS
#!/usr/bin/env bash
for a in "\$@"; do case "\$a" in -Pczechspeech.mavenRepoDir=*) repo="\${a#*=}" ;; esac; done
d="\$repo/ai/moravian/czechspeech-android/1.2.3"
mkdir -p "\$d"
for f in czechspeech-android-1.2.3.aar czechspeech-android-1.2.3.pom; do
    echo "content of \$f" >"\$d/\$f"
    gpg --batch --yes --local-user "$KEY" --armor --detach-sign -o "\$d/\$f.asc" "\$d/\$f"
    for g in "\$f" "\$f.asc"; do
        md5sum "\$d/\$g" | cut -d" " -f1 >"\$d/\$g.md5"
        sha1sum "\$d/\$g" | cut -d" " -f1 >"\$d/\$g.sha1"
    done
done
EOS
chmod +x "$R2/gradlew"
git -C "$R2" add -A && git -C "$R2" commit -qm init
rm -f "$STUB_LOG".*
PATH="$STUB:$PATH" POLL_INTERVAL=0 GPG_KEY_NAME="$KEY" REPO_DIR="$R2" bash "$SCRIPT" >"$WORK/o" 2>"$WORK/e"
check "script without --upload: exit 0" test $? -eq 0
check "script without --upload: prints the bundle, uploads nothing" bash -c "grep -q '^bundle: ' '$WORK/o' && grep -q 'Not uploaded' '$WORK/o' && test ! -e '$STUB_LOG.argv'"
rm -f "$STUB_LOG".*
PATH="$STUB:$PATH" POLL_INTERVAL=0 GPG_KEY_NAME="$KEY" REPO_DIR="$R2" bash "$SCRIPT" --upload >"$WORK/o" 2>"$WORK/e"
check "script with --upload to VALIDATED: exit 0 (the exit trap must not need a local variable)" test $? -eq 0
check "script with --upload: nothing on stderr" test ! -s "$WORK/e"
check "script with --upload: reports VALIDATED" grep -q "VALIDATED" "$WORK/o"
check "script with --upload: token not on output" no_secret_in "$WORK/o" "$WORK/e"

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

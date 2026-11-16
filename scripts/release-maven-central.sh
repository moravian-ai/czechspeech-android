#!/usr/bin/env bash
# Builds the signed Maven Central bundle for CzechSpeech for Android and, with --upload, sends it to the Central
# Portal as a USER_MANAGED deployment. The upload stops at "validated": you review the deployment in the portal
# and press Publish yourself. A published version can never be deleted or replaced.
#
# Run it in WSL or Linux (bash, python3, gpg, curl):
#
#   scripts/release-maven-central.sh             build, sign and check the bundle; print the zip and its SHA-256
#   scripts/release-maven-central.sh --upload    the same, then upload the bundle and wait for validation
#
# Environment:
#   GPG_KEY_NAME   the signing key (default: Moravian AI's key, long ID E40DE78E91BEDC18)
#   SETTINGS_XML   Maven settings holding the Central Portal user token as <server><id>central</id> with
#                  <username> and <password> (default: ~/.m2/settings.xml, or on WSL the Windows one)
#   POLL_INTERVAL  seconds between status requests after an upload (default 10)
#   POLL_MAX       status requests before giving up (default 180)
#   REPO_DIR       the repository to release (default: the one this script lives in)
#
# The token is never printed and never put on a command line: curl reads it from a 0600 temporary config file
# that is deleted on exit. The script never runs with `set -x` and never logs response headers.
set -euo pipefail

DEFAULT_KEY="E40DE78E91BEDC18"
CENTRAL_URL="https://central.sonatype.com"
PORTAL_URL="$CENTRAL_URL/publishing/deployments"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

die() {
    echo "error: $*" >&2
    exit 1
}

project_version() {
    sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO_DIR/czechspeech/build.gradle.kts" | head -n 1
}

check_clean_tree() {
    local status
    status="$(git -C "$REPO_DIR" status --porcelain)" || die "cannot run git status in $REPO_DIR"
    if [[ -n "$status" ]]; then
        echo "$status" >&2
        die "the working tree is not clean (commit or stash the changes above first)"
    fi
}

check_version() { # version
    local version="$1"
    [[ -n "$version" ]] || die "no version found in czechspeech/build.gradle.kts"
    [[ "$version" != *-SNAPSHOT ]] || die "version $version is a SNAPSHOT; Central takes releases only"
    local escaped="${version//./\\.}"
    grep -Eq "^##[[:space:]]+\[?${escaped}\]?([[:space:]]|\$)" "$REPO_DIR/CHANGELOG.md" ||
        die "CHANGELOG.md has no heading for version $version (expected \"## [$version] - <date>\")"
}

# Reads the user token of <server><id>central</id> from settings.xml into CENTRAL_USER and CENTRAL_PASS.
# On any problem it prints what is missing (never what the file contains) and returns 1.
load_credentials() {
    local file="${SETTINGS_XML:-}"
    if [[ -z "$file" ]]; then
        file="$HOME/.m2/settings.xml"
        if [[ ! -f "$file" && -n "${USER:-}" && -f "/mnt/c/Users/$USER/.m2/settings.xml" ]]; then
            file="/mnt/c/Users/$USER/.m2/settings.xml"
        fi
    fi
    local status="" user="" pass=""
    {
        IFS= read -r -d '' status || true
        IFS= read -r -d '' user || true
        IFS= read -r -d '' pass || true
    } < <(SETTINGS_FILE="$file" python3 -I - <<'PY'
import os, re, sys
import xml.etree.ElementTree as ET

def out(status, user="", password=""):
    sys.stdout.write(status + "\0" + user + "\0" + password + "\0")

path = os.environ["SETTINGS_FILE"]
if not os.path.isfile(path):
    out("settings.xml not found (set SETTINGS_XML)")
    sys.exit(0)
try:
    root = ET.parse(path).getroot()
except Exception:
    out("settings.xml cannot be parsed as XML")
    sys.exit(0)

def local(tag):
    return tag.rsplit("}", 1)[-1]

def child_text(node, name):
    for child in node:
        if local(child.tag) == name:
            return (child.text or "").strip()
    return None

server = None
for node in root.iter():
    if local(node.tag) == "server" and child_text(node, "id") == "central":
        server = node
        break
if server is None:
    out("settings.xml has no <server> with id \"central\"")
    sys.exit(0)

values = {}
for name in ("username", "password"):
    value = child_text(server, name)
    if value is None or value == "":
        out("the \"central\" server has no " + name + " (it is missing or empty)")
        sys.exit(0)
    if re.search(r"\$\{[^}]*\}", value):
        out("the \"central\" server's " + name + " is still a placeholder")
        sys.exit(0)
    values[name] = value
out("OK", values["username"], values["password"])
PY
)
    if [[ "$status" != "OK" ]]; then
        echo "error: ${status:-could not read the Central token}" >&2
        return 1
    fi
    CENTRAL_USER="$user"
    CENTRAL_PASS="$pass"
}

# Every artifact in the version directory needs .asc, .md5 and .sha1 files next to it.
check_bundle_dir() { # directory of one version
    local dir="$1" count=0 problems=0 f
    [[ -d "$dir" ]] || die "no such directory: $dir"
    for f in "$dir"/*; do
        [[ -f "$f" ]] || continue
        case "$f" in
            *.asc | *.md5 | *.sha1 | *.sha256 | *.sha512) continue ;;
        esac
        count=$((count + 1))
        for ext in asc md5 sha1; do
            if [[ ! -s "$f.$ext" ]]; then
                echo "error: missing $(basename "$f").$ext" >&2
                problems=$((problems + 1))
            fi
        done
    done
    [[ "$count" -gt 0 ]] || die "no artifacts in $dir"
    [[ "$problems" -eq 0 ]] || die "$problems signature or checksum file(s) missing in $dir"
    echo "checked $count artifacts: .asc, .md5 and .sha1 present"
}

verify_signatures() { # directory of one version
    local dir="$1" asc failed=0 n=0
    for asc in "$dir"/*.asc; do
        [[ -f "$asc" ]] || continue
        n=$((n + 1))
        if ! gpg --batch --verify "$asc" "${asc%.asc}" >/dev/null 2>&1; then
            echo "error: gpg --verify failed for $(basename "$asc")" >&2
            failed=$((failed + 1))
        fi
    done
    [[ "$n" -gt 0 ]] || die "no .asc files in $dir"
    [[ "$failed" -eq 0 ]] || die "$failed signature(s) did not verify"
    echo "verified $n signatures"
}

make_zip() { # root, relative directory kept inside the zip, output file
    python3 -I - "$1" "$2" "$3" <<'PY'
import os, sys, zipfile
root, rel, out = sys.argv[1:4]
base = os.path.join(root, rel)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for dirpath, _dirs, files in sorted(os.walk(base)):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            z.write(full, os.path.relpath(full, root).replace(os.sep, "/"))
PY
}

# curl with the Authorization header read from a private config file, never from the command line.
CURL_CFG=""
curl_central() { # curl arguments
    curl -sS -K "$CURL_CFG" "$@"
}

json_field() { # file, field
    python3 -I - "$1" "$2" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    value = data.get(sys.argv[2], "")
except Exception:
    value = ""
print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
PY
}

upload_and_wait() { # zip, version
    local zip="$1" version="$2" code body id state tries=0
    [[ -n "${CENTRAL_USER:-}" && -n "${CENTRAL_PASS:-}" ]] || die "no Central credentials loaded"
    CURL_CFG="$(mktemp "${TMPDIR:-/tmp}/curl-cfg.XXXXXX")"
    body="$(mktemp "${TMPDIR:-/tmp}/central-body.XXXXXX")"
    chmod 600 "$CURL_CFG" "$body"
    trap 'rm -f "$CURL_CFG" "$body"' EXIT
    trap 'exit 130' INT TERM
    printf 'header = "Authorization: Bearer %s"\n' "$(printf '%s:%s' "$CENTRAL_USER" "$CENTRAL_PASS" | base64 | tr -d '\n')" >"$CURL_CFG"

    echo "uploading $(basename "$zip") as czechspeech-android-$version (USER_MANAGED, no automatic publish)"
    code="$(curl_central -o "$body" -w '%{http_code}' -X POST \
        --form "bundle=@$zip" \
        "$CENTRAL_URL/api/v1/publisher/upload?publishingType=USER_MANAGED&name=czechspeech-android-$version")" ||
        die "the upload request failed"
    case "$code" in
        200 | 201) ;;
        *) die "the upload was refused (HTTP $code); response: $(head -c 500 "$body" | tr -d '\r')" ;;
    esac
    id="$(tr -d '\r\n"' <"$body")"
    [[ -n "$id" ]] || die "Central returned no deployment id"
    echo "deployment id: $id"

    while true; do
        code="$(curl_central -o "$body" -w '%{http_code}' -X POST \
            "$CENTRAL_URL/api/v1/publisher/status?id=$id")" || die "the status request failed"
        [[ "$code" == 200 ]] || die "the status request was refused (HTTP $code)"
        state="$(json_field "$body" deploymentState)"
        echo "state: $state"
        case "$state" in
            VALIDATED)
                rm -f "$CURL_CFG" "$body"
                echo "Validated. Review it and press Publish at: $PORTAL_URL"
                echo "(deployment id $id; a published version cannot be deleted or replaced)"
                return 0
                ;;
            FAILED)
                echo "Deployment FAILED. Errors:" >&2
                json_field "$body" errors >&2
                rm -f "$CURL_CFG" "$body"
                echo "Portal: $PORTAL_URL" >&2
                return 1
                ;;
        esac
        tries=$((tries + 1))
        if [[ "$tries" -ge "${POLL_MAX:-180}" ]]; then
            rm -f "$CURL_CFG" "$body"
            die "still \"$state\" after $tries status requests; look at $PORTAL_URL (deployment id $id)"
        fi
        sleep "${POLL_INTERVAL:-10}"
    done
}

main() {
    local upload=0 arg
    for arg in "$@"; do
        case "$arg" in
            --upload) upload=1 ;;
            -h | --help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d;s/^# \{0,1\}//'; return 0 ;;
            *) echo "usage: $0 [--upload]" >&2; return 2 ;;
        esac
    done

    local version key
    key="${GPG_KEY_NAME:-$DEFAULT_KEY}"
    version="$(project_version)"
    check_clean_tree
    check_version "$version"
    if [[ "$upload" -eq 1 ]]; then
        load_credentials || exit 1
    fi

    # Prime gpg-agent with the passphrase here, in this terminal, so the Gradle run does not need a prompt.
    GPG_TTY="$(tty 2>/dev/null || true)"
    export GPG_TTY
    echo "signing check with key $key"
    echo "signing check" | gpg --local-user "$key" --clearsign >/dev/null ||
        die "gpg cannot sign with key $key (no such key, or the passphrase was refused)"

    local work repo
    work="$(mktemp -d "${TMPDIR:-/tmp}/czechspeech-central.XXXXXX")"
    repo="$work/repo"
    echo "building $version into $repo"
    (cd "$REPO_DIR" && ./gradlew --max-workers=4 :czechspeech:publishToLocalFolder \
        -Pczechspeech.sign=true "-Psigning.gnupg.keyName=$key" "-Pczechspeech.mavenRepoDir=$repo")

    local rel="ai/moravian/czechspeech-android/$version"
    [[ -d "$repo/$rel" ]] || die "the build produced no $rel"
    check_bundle_dir "$repo/$rel"
    verify_signatures "$repo/$rel"

    local zip="$work/czechspeech-android-$version-bundle.zip"
    make_zip "$repo" "$rel" "$zip"
    echo
    echo "bundle: $zip"
    echo "sha256: $(sha256sum "$zip" | cut -d' ' -f1)"

    if [[ "$upload" -eq 1 ]]; then
        echo
        upload_and_wait "$zip" "$version"
    else
        echo "Not uploaded. Run again with --upload to send it to the Central Portal."
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi

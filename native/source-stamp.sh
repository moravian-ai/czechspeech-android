#!/usr/bin/env bash
# The stamp of the native sources: a hash over everything the prebuilt libraries are made from, so that an AAR is never
# packaged with libraries that predate the native code (an UnsatisfiedLinkError on the first call that is missing).
#
#   native/source-stamp.sh                print the hash
#   native/source-stamp.sh write FILE     write the hash to FILE (native/build-native.sh does, next to the libraries)
#   native/source-stamp.sh check FILE     exit 0 when FILE holds the hash; otherwise exit 1 and say what to run
#
# The hash covers native/src, native/patches, native/CMakeLists.txt, native/build-native.sh and the whisper.cpp commit
# pinned in it (WHISPER_COMMIT). NATIVE_STAMP_ROOT replaces the repository root (the unit test uses a copy).
set -euo pipefail

ROOT=${NATIVE_STAMP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
# FILE is relative to the caller's directory, not to the repository root the script moves to
FILE=${2:-}
case $FILE in "" | /*) ;; *) FILE=$PWD/$FILE ;; esac
cd "$ROOT"

sha() {
    if command -v sha256sum > /dev/null 2>&1; then sha256sum; else shasum -a 256; fi | cut -d' ' -f1
}

stamp() {
    local commit f
    local -a files=()
    commit=$(sed -n 's/^WHISPER_COMMIT=//p' native/build-native.sh)
    [ -n "$commit" ] || { echo "ERROR: no WHISPER_COMMIT in native/build-native.sh" >&2; exit 2; }
    # In a git checkout only tracked files count (editor swap, .orig and .rej files must not make the stamp stale);
    # elsewhere (a copy of the tree) every file does. NUL-separated, so names with spaces or globs are safe.
    if [ -z "$(git rev-parse --show-prefix 2> /dev/null || echo x)" ] \
        && git ls-files --error-unmatch native/build-native.sh > /dev/null 2>&1; then
        while IFS= read -r -d '' f; do files+=("$f"); done < <(git ls-files -z native/src native/patches | LC_ALL=C sort -z)
    else
        while IFS= read -r -d '' f; do files+=("$f"); done < <(find native/src native/patches -type f -print0 | LC_ALL=C sort -z)
    fi
    files+=(native/CMakeLists.txt native/build-native.sh)
    {
        echo "whisper.cpp $commit"
        for f in "${files[@]}"; do
            echo "$f $(sha < "$f")"
        done
    } | sha
}

case ${1:-print} in
    print) stamp ;;
    write) [ -n "$FILE" ] || { echo "usage: $0 write FILE" >&2; exit 2; }; stamp > "$FILE" ;;
    check)
        [ -n "$FILE" ] || { echo "usage: $0 check FILE" >&2; exit 2; }
        want=$(stamp)
        if [ ! -f "$FILE" ]; then
            echo "The native libraries have no stamp ($2 is missing): they were built before the stamp existed, so they may predate the native sources. Run native/build-native.sh android first."
            exit 1
        fi
        if [ "$(tr -d '[:space:]' < "$FILE")" != "$want" ]; then
            echo "The native libraries were built from other sources than the ones in this tree ($2 differs from the stamp of native/src, native/patches, native/build-native.sh and the whisper.cpp commit). Run native/build-native.sh android first."
            exit 1
        fi
        ;;
    *) echo "usage: $0 [print | write FILE | check FILE]" >&2; exit 2 ;;
esac

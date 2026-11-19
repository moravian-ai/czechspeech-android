#!/usr/bin/env bash
# Builds the CzechSpeech JNI library (whisper.cpp + ggml, statically linked) as whole-library CPU variants:
#
#   native/out/jniLibs/arm64-v8a/libczechspeech_armv8a.so                    armv8-a (any 64-bit ARM phone)
#   native/out/jniLibs/arm64-v8a/libczechspeech_armv82_dotprod_fp16.so       armv8.2-a+dotprod+fp16
#   native/out/jniLibs/arm64-v8a/libczechspeech_armv82_dotprod_fp16_i8mm.so  armv8.2-a+dotprod+fp16+i8mm
#   native/out/jniLibs/x86_64/libczechspeech_x86_64.so                       x86-64 + SSE4.2 (emulator)
#   native/out/host/linux-x86_64/libczechspeech_host.so                      build machine's CPU, JVM tests only
#
# The Kotlin loader (NativeLoader) picks one at runtime from /proc/cpuinfo, so no phone gets instructions it lacks.
# Every Android library is checked for 16 KB LOAD alignment and for instructions above its variant's level.
#
# whisper.cpp is the pinned submodule with native/patches/*.patch applied (the beam search fix and every beam's
# hypothesis, for variant selection; the abort callback inside the encoder; the language selection inside
# whisper_full). The submodule itself is never modified: its commit is extracted with git archive
# into $CZECHSPEECH_NATIVE_BUILD_DIR/whisper.cpp-patched and patched there, again only when the commit or a patch
# changes. To update whisper.cpp, move the submodule and WHISPER_COMMIT, then refresh the patches until they apply.
#
# Usage (Linux or WSL):  native/build-native.sh [all|android|host]      (default: all)
# Environment:
#   ANDROID_NDK_HOME              NDK r27c        (default ~/android/android-ndk-r27c)
#   CMAKE_BIN_DIR                 cmake + ninja   (default ~/android/sdk/cmake/3.22.1/bin)
#   JAVA_HOME                     JDK for jni.h in the host build (default ~/android/jdk17)
#   JOBS                          parallel compile jobs (default 8)
#   CZECHSPEECH_NATIVE_BUILD_DIR  CMake build trees (default ~/.cache/czechspeech-native; keep it off /mnt/c)
set -euo pipefail

WHAT=${1:-all}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NDK=${ANDROID_NDK_HOME:-$HOME/android/android-ndk-r27c}
CMAKE_BIN_DIR=${CMAKE_BIN_DIR:-$HOME/android/sdk/cmake/3.22.1/bin}
CMAKE=$CMAKE_BIN_DIR/cmake
NINJA=$CMAKE_BIN_DIR/ninja
export JAVA_HOME=${JAVA_HOME:-$HOME/android/jdk17}
JOBS=${JOBS:-8}
BUILD=${CZECHSPEECH_NATIVE_BUILD_DIR:-$HOME/.cache/czechspeech-native}
OUT=$ROOT/native/out
WHISPER=$ROOT/third_party/whisper.cpp
WHISPER_COMMIT=482279b8a18855ba741d7b88c30fcb3874691f59
LLVM_BIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
# Strip local paths from __FILE__ (ggml asserts) so the libraries carry no build-machine paths and are reproducible.
COMMON_FLAGS="-ffunction-sections -fdata-sections -ffile-prefix-map=$ROOT/= -ffile-prefix-map=$BUILD/=build/"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$WHISPER/include/whisper.h" ] || die "third_party/whisper.cpp is empty; run: git submodule update --init"
actual=$(git -C "$WHISPER" rev-parse HEAD 2>/dev/null || echo unknown)
[ "$actual" = "$WHISPER_COMMIT" ] || die "third_party/whisper.cpp is at $actual, expected $WHISPER_COMMIT"
[ -x "$CMAKE" ] && [ -x "$NINJA" ] || die "cmake/ninja not found in $CMAKE_BIN_DIR"

PATCHED=$BUILD/whisper.cpp-patched

# The patches took: the patched header declares their API. An unpatched tree is never stamped nor reused.
patched_ok() {
    local header=$PATCHED/include/whisper.h
    grep -q 'whisper_full_get_hypothesis' "$header" 2>/dev/null && grep -q 'language_select_callback' "$header" 2>/dev/null
}

# The pinned whisper.cpp with native/patches/*.patch applied, in $PATCHED; redone only when the commit or a patch changes.
prepare_whisper() {
    local want stamp p
    want="$WHISPER_COMMIT $(cat "$ROOT"/native/patches/*.patch | sha256sum | cut -d' ' -f1)"
    stamp=$(cat "$PATCHED.stamp" 2>/dev/null || true)
    if [ "$stamp" = "$want" ] && patched_ok; then
        echo "== whisper.cpp ${WHISPER_COMMIT:0:7} + native/patches (unchanged)"
        return
    fi
    echo "== whisper.cpp ${WHISPER_COMMIT:0:7} + native/patches"
    rm -rf "$PATCHED" "$PATCHED.stamp"
    mkdir -p "$PATCHED"
    git -C "$WHISPER" archive --format=tar "$WHISPER_COMMIT" | tar -x -C "$PATCHED"
    for p in "$ROOT"/native/patches/*.patch; do
        (cd "$PATCHED" && git apply --whitespace=nowarn "$p") || die "patch $(basename "$p") does not apply"
        echo "   applied $(basename "$p")"
    done
    patched_ok || die "native/patches applied, but $PATCHED/include/whisper.h lacks whisper_full_get_hypothesis or language_select_callback"
    echo "$want" > "$PATCHED.stamp"
}

build_android() {  # name abi ggml-options...
    local name=$1 abi=$2
    shift 2
    [ -f "$NDK/build/cmake/android.toolchain.cmake" ] || die "NDK not found at $NDK"
    local dir=$BUILD/$name
    echo "== $name ($abi): $*"
    "$CMAKE" -S "$ROOT/native" -B "$dir" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$abi" -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_static \
        -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="$COMMON_FLAGS" -DCMAKE_CXX_FLAGS="$COMMON_FLAGS" \
        -DCZECHSPEECH_LIB_NAME="$name" -DWHISPER_CPP_DIR="$PATCHED" -DGGML_NATIVE=OFF "$@" > "$dir.configure.log" 2>&1 \
        || { tail -40 "$dir.configure.log"; die "configure $name failed"; }
    "$CMAKE" --build "$dir" --target czechspeech -j "$JOBS" > "$dir.build.log" 2>&1 \
        || { tail -60 "$dir.build.log"; die "build $name failed"; }
    mkdir -p "$OUT/jniLibs/$abi"
    "$LLVM_BIN/llvm-strip" --strip-unneeded -o "$OUT/jniLibs/$abi/lib$name.so" "$dir/lib$name.so"
}

check_android() {  # file abi level
    local so=$1 abi=$2 level=$3
    # 16 KB page alignment of every LOAD segment
    local aligns
    aligns=$("$LLVM_BIN/llvm-readelf" -lW "$so" | awk '$1 == "LOAD" { print $NF }' | sort -u | tr '\n' ' ')
    [ "$aligns" = "0x4000 " ] || die "$(basename "$so"): LOAD alignment is '$aligns', expected 0x4000"
    # instructions above the variant's level would crash (SIGILL) on phones without them
    local dis
    dis=$("$LLVM_BIN/llvm-objdump" -d --no-show-raw-insn "$so")
    if [ "$abi" = arm64-v8a ]; then
        local dot mmla fp16
        dot=$(grep -cE $'\t(sdot|udot|usdot|sudot)\t' <<< "$dis" || true)
        mmla=$(grep -cE $'\t(smmla|ummla|usmmla)\t' <<< "$dis" || true)
        fp16=$(grep -cE $'\tf(add|sub|mul|mla|mls|div|max|min|maxnm|minnm|abs|neg|sqrt|fma|cvtzs|cvtzu)\t(v[0-9]+\\.[48]h|h[0-9]+)' <<< "$dis" || true)
        echo "   $(basename "$so"): LOAD align 0x4000; dotprod=$dot i8mm=$mmla fp16-arith=$fp16"
        case $level in
            base) [ "$dot" = 0 ] && [ "$mmla" = 0 ] && [ "$fp16" = 0 ] || die "baseline library uses dotprod/i8mm/fp16" ;;
            dotprod) [ "$mmla" = 0 ] || die "dotprod library uses i8mm" ;;
            i8mm) ;;
        esac
    else
        local avx
        avx=$(grep -cE $'\tv[a-z0-9]+\t|%ymm|%zmm' <<< "$dis" || true)
        echo "   $(basename "$so"): LOAD align 0x4000; avx=$avx"
        [ "$avx" = 0 ] || die "x86_64 library uses AVX instructions"
    fi
    # no build-machine paths (user names, home directories) in shipped binaries
    if "$LLVM_BIN/llvm-strings" "$so" | grep -qF -e "$ROOT" -e "$BUILD" -e "$HOME"; then
        die "$(basename "$so") contains local build paths"
    fi
}

build_host() {
    local name=czechspeech_host dir=$BUILD/czechspeech_host
    [ -f "$JAVA_HOME/include/jni.h" ] || die "JAVA_HOME=$JAVA_HOME has no include/jni.h"
    echo "== $name (linux-x86_64, host CPU)"
    "$CMAKE" -S "$ROOT/native" -B "$dir" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="$COMMON_FLAGS" -DCMAKE_CXX_FLAGS="$COMMON_FLAGS" \
        -DCZECHSPEECH_LIB_NAME="$name" -DWHISPER_CPP_DIR="$PATCHED" -DGGML_NATIVE=ON > "$dir.configure.log" 2>&1 \
        || { tail -40 "$dir.configure.log"; die "configure $name failed"; }
    "$CMAKE" --build "$dir" --target czechspeech -j "$JOBS" > "$dir.build.log" 2>&1 \
        || { tail -60 "$dir.build.log"; die "build $name failed"; }
    mkdir -p "$OUT/host/linux-x86_64"
    strip --strip-unneeded -o "$OUT/host/linux-x86_64/lib$name.so" "$dir/lib$name.so"
    echo "   $(ls -la "$OUT/host/linux-x86_64/lib$name.so" | awk '{print $5}') bytes"
}

# Stamp of the sources as they are NOW, before anything is prepared or compiled: sources edited during a long build
# then differ from it, so libraries built from the old ones are never stamped as current.
SOURCE_STAMP=$(bash "$ROOT/native/source-stamp.sh")
mkdir -p "$BUILD"
prepare_whisper
case $WHAT in
    all|android)
        rm -rf "$OUT/jniLibs" "$OUT/source-stamp"
        build_android czechspeech_armv8a                   arm64-v8a -DGGML_CPU_ARM_ARCH=armv8-a
        build_android czechspeech_armv82_dotprod_fp16      arm64-v8a -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
        build_android czechspeech_armv82_dotprod_fp16_i8mm arm64-v8a -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm
        build_android czechspeech_x86_64                   x86_64    -DGGML_SSE42=ON
        echo "== checks"
        check_android "$OUT/jniLibs/arm64-v8a/libczechspeech_armv8a.so"                   arm64-v8a base
        check_android "$OUT/jniLibs/arm64-v8a/libczechspeech_armv82_dotprod_fp16.so"      arm64-v8a dotprod
        check_android "$OUT/jniLibs/arm64-v8a/libczechspeech_armv82_dotprod_fp16_i8mm.so" arm64-v8a i8mm
        check_android "$OUT/jniLibs/x86_64/libczechspeech_x86_64.so"                      x86_64    base
        # the stamp of the sources these libraries are made from; checkNativeLibs of czechspeech/build.gradle.kts compares it
        echo "$SOURCE_STAMP" > "$OUT/source-stamp"
        ls -la "$OUT"/jniLibs/*/
        ;;
esac
case $WHAT in
    all|host) build_host ;;
esac
case $WHAT in
    all|android|host) echo DONE ;;
    *) die "unknown target '$WHAT' (all|android|host)" ;;
esac

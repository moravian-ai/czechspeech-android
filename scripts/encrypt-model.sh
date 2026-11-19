#!/usr/bin/env bash
# Encrypts a model file for CzechSpeechModel.Encryption (AES-256-CTR, whole file, same length), streaming.
#
# Usage: scripts/encrypt-model.sh [options] <in|-> <out|-> <keyfile>
#
#   <in>, <out>   a path, or - for stdin / stdout (so a model can be piped from storage to storage without a local copy)
#   <keyfile>     receives "key=<64 hex>" and "iv=<32 hex>", mode 600; the key is never written to stdout
#                 unless --print-key is given. An existing key file or output file is never overwritten (--force).
# Options:
#   --expect-sha256 <hex>  the KNOWN SHA-256 of the plain model (catalogue, Hugging Face, build report) and
#   --expect-size <bytes>  its size: if what was read does not match, the script fails and removes its output.
#                          Always give them when the input is a pipe: an input that breaks off looks like a shorter
#                          file, and its hash would otherwise be printed as if it were the model's.
#   --print-key            also print key and IV (to stdout, or to stderr when <out> is -)
#   --force                overwrite an existing key file or output file
#   --reuse-iv-i-know      allow ENC_IV from the environment (see below)
#   ENC_KEY (environment, hex)  use this key instead of generating one with `openssl rand -hex 32`
#   ENC_IV  (environment, hex)  use this IV instead of a random one; needs --reuse-iv-i-know
#
# NEVER encrypt two different files with the same key and IV (a new version of a model gets a new key, or at least a
# new IV): the two ciphertexts would reveal the XOR of the two plaintexts. By default every run draws a fresh IV.
# The key and IV belong to the hosted file: publish a re-encrypted file under a new URL, never over the URL of a
# released app.
#
# Prints (to stdout, or to stderr when <out> is -): plaintext sha256 (what Source.sha256 takes, once it equals the
# hash you know), ciphertext sha256 and size.
# In a pipeline use `set -o pipefail`, so that a failure of the upstream command fails the whole pipeline.
# Decrypt to check: openssl enc -d -aes-256-ctr -K <key> -iv <iv> -in <out>
# Note: openssl takes -K on its command line, so the key is visible to other users of this machine while it runs;
# run it on a machine you do not share.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# *//' >&2; exit 2; }

print_key=0 force=0 reuse_iv=0 expect_sha="" expect_size=""
while [ "$#" -gt 0 ]; do
  case $1 in
    --print-key) print_key=1 ;;
    --force) force=1 ;;
    --reuse-iv-i-know) reuse_iv=1 ;;
    --expect-sha256) [ "$#" -ge 2 ] || usage; expect_sha=$(printf '%s' "$2" | tr 'A-F' 'a-f'); shift ;;
    --expect-size) [ "$#" -ge 2 ] || usage; expect_size=$2; shift ;;
    --) shift; break ;;
    -?*) usage ;;
    *) break ;;
  esac
  shift
done
[ "$#" -eq 3 ] || usage
in=$1 out=$2 keyfile=$3

[ "$keyfile" != "-" ] || { echo "the key file must be a path" >&2; exit 2; }
[ -z "$expect_sha" ] || [[ $expect_sha =~ ^[0-9a-f]{64}$ ]] || { echo "--expect-sha256 must be 64 hex digits" >&2; exit 2; }
[ -z "$expect_size" ] || [[ $expect_size =~ ^[0-9]+$ ]] || { echo "--expect-size must be a number of bytes" >&2; exit 2; }
command -v openssl >/dev/null || { echo "openssl is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required (GNU coreutils)" >&2; exit 2; }

if [ -n "${ENC_IV:-}" ] && [ "$reuse_iv" != 1 ]; then
  echo "ENC_IV is refused without --reuse-iv-i-know: reusing a key and IV for a second file breaks the encryption" >&2
  exit 2
fi
if [ "$in" != "-" ] && [ "$out" != "-" ] && [ "$in" -ef "$out" ]; then echo "input and output are the same file" >&2; exit 2; fi
if [ "$force" != 1 ]; then
  [ ! -e "$keyfile" ] || { echo "$keyfile exists; it holds the key of another file (--force to overwrite)" >&2; exit 2; }
  if [ "$out" != "-" ] && [ -e "$out" ]; then echo "$out exists (--force to overwrite)" >&2; exit 2; fi
fi

key=${ENC_KEY:-$(openssl rand -hex 32)}
iv=${ENC_IV:-$(openssl rand -hex 16)}
[[ $key =~ ^[0-9a-fA-F]{64}$ ]] || { echo "ENC_KEY must be 64 hex digits" >&2; exit 2; }
[[ $iv =~ ^[0-9a-fA-F]{32}$ ]] || { echo "ENC_IV must be 32 hex digits" >&2; exit 2; }

tmp=$(mktemp -d)
part=""
[ "$out" = "-" ] || part="$out.part"
cleanup() { rm -rf "$tmp"; if [ -n "$part" ]; then rm -f "$part"; fi; }
trap cleanup EXIT

# the key goes to a file only this user can read, and to its place only after everything else succeeded
(umask 077; printf 'key=%s\niv=%s\n' "$key" "$iv" > "$tmp/key")
chmod 600 "$tmp/key"

src() { if [ "$in" = "-" ]; then cat; else cat -- "$in"; fi; }
dst() { if [ "$out" = "-" ]; then cat; else cat > "$part"; fi; }

src | tee >(sha256sum | cut -d' ' -f1 > "$tmp/plain") \
  | openssl enc -aes-256-ctr -K "$key" -iv "$iv" \
  | tee >(sha256sum | cut -d' ' -f1 > "$tmp/cipher") >(wc -c | tr -d ' ' > "$tmp/size") \
  | dst

# the process substitutions finish a moment after the pipeline
for _ in $(seq 1 600); do
  [ -s "$tmp/plain" ] && [ -s "$tmp/cipher" ] && [ -s "$tmp/size" ] && break
  sleep 0.1
done
{ [ -s "$tmp/plain" ] && [ -s "$tmp/cipher" ] && [ -s "$tmp/size" ]; } || { echo "hashing did not finish" >&2; exit 1; }

plain_sha=$(cat "$tmp/plain") cipher_sha=$(cat "$tmp/cipher") size=$(cat "$tmp/size")
if [ -n "$expect_sha" ] && [ "$plain_sha" != "$expect_sha" ]; then
  echo "FAILED: the input has sha256 $plain_sha, not the expected $expect_sha (a broken-off input?)" >&2
  exit 3
fi
if [ -n "$expect_size" ] && [ "$size" != "$expect_size" ]; then
  echo "FAILED: the input has $size bytes, not the expected $expect_size (a broken-off input?)" >&2
  exit 3
fi

if [ -n "$part" ]; then mv -f "$part" "$out"; fi
mv -f "$tmp/key" "$keyfile"
mode_note="(mode 600)"
if [ "$(stat -c %a "$keyfile" 2>/dev/null || echo ?)" != "600" ]; then
  mode_note="(WARNING: could not set mode 600 on this file system; protect it yourself)"
fi

report() {
  echo "plaintext sha256:  $plain_sha"
  echo "ciphertext sha256: $cipher_sha"
  echo "size:              $size"
  if [ -z "$expect_sha" ]; then echo "NOTE: no --expect-sha256 given; the hash above is only trustworthy if the input was read completely"; fi
  echo "key and IV written to $keyfile $mode_note"
  if [ "$print_key" = 1 ]; then echo "key=$key"; echo "iv=$iv"; fi
}
if [ "$out" = "-" ]; then report >&2; else report; fi

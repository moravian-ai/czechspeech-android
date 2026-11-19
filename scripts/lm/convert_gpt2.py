"""Converts a Hugging Face GPT-2 checkpoint to the GGUF file that CzechSpeech's variant selection loads.

The SDK's language model is lchaloupsky/czech-gpt2-oscar (GPT-2 small, 124 M parameters, MIT), converted to q8_0:

    python scripts/lm/convert_gpt2.py --hf <directory> --out ggml-czech-gpt2-oscar-q8_0.gguf --type q8_0

<directory> holds model.safetensors, config.json, vocab.json and merges.txt of revision
2b5b1c4f483886485c6c6703bb1d94b1547730f7 (huggingface.co/lchaloupsky/czech-gpt2-oscar). The output is byte for byte
reproducible: the same input gives the same file, whose size and SHA-256 the SDK pins (CzechLanguageModel).

The file follows llama.cpp's GGUF layout for the gpt2 architecture (tensor names, metadata keys), so llama.cpp can
read it too. Linear weights are stored as [out, in] (ggml ne = [in, out]); the token embedding doubles as the output
matrix (tied, as in GPT-2). q8_0 quantises the five weight matrices of every block and the token embedding with
ggml's reference rounding; position embedding, layer norms and biases stay f32.

Needs numpy and safetensors only (no PyTorch).
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

# gguf value types
T_UINT32, T_FLOAT32, T_STRING, T_ARRAY = 4, 6, 8, 9
# ggml tensor types
GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8
QK8_0 = 32
# llama.cpp general.file_type
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7}


def quantize_q8_0(x: np.ndarray) -> bytes:
    """ggml's quantize_row_q8_0_ref: blocks of 32, d = amax / 127 as f16, q = round half away from zero of x / d."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.shape[-1] % QK8_0 == 0, x.shape
    blocks = x.reshape(-1, QK8_0)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / np.float32(127)).astype(np.float32)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0).astype(np.float32)
    y = (blocks * inv[:, None]).astype(np.float32)
    q = np.round(y)
    t = np.trunc(y)
    half = np.abs(y - t) == np.float32(0.5)
    q[half] = (t + np.sign(y))[half]
    out = np.empty(len(blocks), dtype=[("d", "<f2"), ("qs", "i1", (QK8_0,))])
    out["d"] = d.astype(np.float16)
    out["qs"] = q.astype(np.int8)
    return out.tobytes()


def tensor_bytes(x: np.ndarray, ggml_type: int) -> bytes:
    if ggml_type == GGML_F32:
        return np.ascontiguousarray(x, dtype="<f4").tobytes()
    if ggml_type == GGML_F16:
        return np.ascontiguousarray(x, dtype="<f2").tobytes()
    if ggml_type == GGML_Q8_0:
        return quantize_q8_0(x)
    raise ValueError(ggml_type)


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _value(v) -> bytes:
    if isinstance(v, str):
        return struct.pack("<I", T_STRING) + _str(v)
    if isinstance(v, float):
        return struct.pack("<If", T_FLOAT32, v)
    if isinstance(v, int):
        return struct.pack("<II", T_UINT32, v)
    if isinstance(v, list):
        assert v and all(isinstance(e, str) for e in v), "only non-empty string arrays"
        return struct.pack("<IIQ", T_ARRAY, T_STRING, len(v)) + b"".join(_str(e) for e in v)
    raise TypeError(type(v))


def write_gguf(path: Path, kv: dict, tensors: list[tuple[str, np.ndarray, int]]) -> None:
    """GGUF v3: header, metadata, tensor infos, then the tensor data, each aligned to 32 bytes."""
    datas, infos, offset = [], [], 0
    for name, x, ggml_type in tensors:
        data = tensor_bytes(x, ggml_type)
        ne = list(reversed(x.shape))                      # ggml: ne[0] is the fastest (last numpy) axis
        infos.append(_str(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
                     + struct.pack("<IQ", ggml_type, offset))
        datas.append(data)
        offset += len(data) + (-len(data)) % ALIGNMENT
    head = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kv))
    head += b"".join(_str(k) + _value(v) for k, v in kv.items()) + b"".join(infos)
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % ALIGNMENT))
        for data in datas:
            f.write(data + b"\0" * ((-len(data)) % ALIGNMENT))


def load_vocab(hf: Path) -> tuple[list[str], list[str]]:
    vocab = json.loads((hf / "vocab.json").read_text(encoding="utf-8"))
    tokens = [""] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    assert sorted(vocab.values()) == list(range(len(vocab))), "vocabulary ids are not 0..n-1"
    merges = [m for m in (hf / "merges.txt").read_text(encoding="utf-8").split("\n")
              if m and not m.startswith("#version")]
    assert all(len(m.split(" ")) == 2 for m in merges), "a merge is not two symbols"
    return tokens, merges


def convert(hf: Path, out: Path, kind: str, name: str, source: str) -> None:
    from safetensors.numpy import load_file

    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "gpt2" and cfg.get("activation_function", "gelu_new") == "gelu_new", cfg
    w = {k.removeprefix("transformer."): v for k, v in load_file(str(hf / "model.safetensors")).items()}
    if "lm_head.weight" in w:
        assert np.array_equal(w["lm_head.weight"], w["wte.weight"]), "untied output matrix is not supported"
    tokens, merges = load_vocab(hf)
    n_vocab, n_embd = w["wte.weight"].shape
    assert n_vocab == len(tokens), (n_vocab, len(tokens))
    big = {"f32": GGML_F32, "f16": GGML_F16, "q8_0": GGML_Q8_0}[kind]
    small = GGML_F32
    eos = tokens.index("<|endoftext|>") if "<|endoftext|>" in tokens else cfg["eos_token_id"]

    t = [("token_embd.weight", w["wte.weight"], big),
         ("position_embd.weight", w["wpe.weight"], small)]
    for i in range(cfg["n_layer"]):
        p = f"h.{i}."
        t += [(f"blk.{i}.attn_norm.weight", w[p + "ln_1.weight"], small),
              (f"blk.{i}.attn_norm.bias", w[p + "ln_1.bias"], small),
              (f"blk.{i}.attn_qkv.weight", w[p + "attn.c_attn.weight"].T, big),      # Conv1D [in, out] -> [out, in]
              (f"blk.{i}.attn_qkv.bias", w[p + "attn.c_attn.bias"], small),
              (f"blk.{i}.attn_output.weight", w[p + "attn.c_proj.weight"].T, big),
              (f"blk.{i}.attn_output.bias", w[p + "attn.c_proj.bias"], small),
              (f"blk.{i}.ffn_norm.weight", w[p + "ln_2.weight"], small),
              (f"blk.{i}.ffn_norm.bias", w[p + "ln_2.bias"], small),
              (f"blk.{i}.ffn_up.weight", w[p + "mlp.c_fc.weight"].T, big),
              (f"blk.{i}.ffn_up.bias", w[p + "mlp.c_fc.bias"], small),
              (f"blk.{i}.ffn_down.weight", w[p + "mlp.c_proj.weight"].T, big),
              (f"blk.{i}.ffn_down.bias", w[p + "mlp.c_proj.bias"], small)]
    t += [("output_norm.weight", w["ln_f.weight"], small), ("output_norm.bias", w["ln_f.bias"], small)]

    kv = {
        "general.architecture": "gpt2",
        "general.name": name,
        "general.license": "MIT",
        "general.source.url": source,
        "general.file_type": FILE_TYPE[kind],
        "general.quantization_version": 2,
        "gpt2.context_length": int(cfg["n_positions"]),
        "gpt2.embedding_length": int(n_embd),
        "gpt2.feed_forward_length": int(w["h.0.mlp.c_fc.weight"].shape[1]),
        "gpt2.block_count": int(cfg["n_layer"]),
        "gpt2.attention.head_count": int(cfg["n_head"]),
        "gpt2.attention.layer_norm_epsilon": float(cfg["layer_norm_epsilon"]),
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": int(eos),
        "tokenizer.ggml.eos_token_id": int(eos),
    }
    write_gguf(out, kv, [(n, np.asarray(x, dtype=np.float32), ty) for n, x, ty in t])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", type=Path, required=True, help="directory with model.safetensors, config.json, vocab.json, merges.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--type", choices=sorted(FILE_TYPE), default="q8_0")
    ap.add_argument("--name", default="czech-gpt2-oscar")
    ap.add_argument("--source", default="https://huggingface.co/lchaloupsky/czech-gpt2-oscar/tree/"
                                        "2b5b1c4f483886485c6c6703bb1d94b1547730f7")
    a = ap.parse_args()
    convert(a.hf, a.out, a.type, a.name, a.source)
    import hashlib
    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(f"{a.out}: {a.out.stat().st_size} bytes, sha256 {h}")


if __name__ == "__main__":
    main()

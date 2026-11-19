"""Reference data for the language-model tests: token ids from the Hugging Face tokenizer and log-probabilities
from PyTorch (float32, CPU), computed the way the measurement scored hypotheses: the text is stripped, prefixed with
the start token (<|endoftext|>), and the log-probabilities of its tokens are summed (no end token).

    python scripts/lm/make_reference.py tiny  --out czechspeech/src/test/resources/lm
    python scripts/lm/make_reference.py czech --hf <czech-gpt2-oscar directory> --out czechspeech/src/test/resources/lm

`tiny` trains a small byte-level BPE tokenizer on the test sentences and builds a two-layer GPT-2 with random weights
(seed 0), then writes it as GGUF in f32 and q8_0 (tiny-gpt2-f32.gguf, tiny-gpt2-q8_0.gguf) with reference-tiny.tsv:
the JVM tests check the SDK's tokenizer and its ggml GPT-2 against these without the real model. `czech` writes
reference-czech-gpt2.tsv for the real model (tests run when CZECHSPEECH_LM points at the converted file).

TSV columns: sum of log-probabilities, token ids without the start token (space separated), text.
Needs torch, transformers, tokenizers, safetensors, numpy. Runs on the CPU only.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Invented for these tests (no outside data). Groups of variants share prefixes, as beam-search hypotheses do.
SENTENCES = [
    "Dobrý den, posílám vám zprávu z vlaku.",
    "Zítra v 7:45 odjíždíme na chatu do Beskyd.",
    "Ve středu jsem koupil 3 kila jablek a 12 rohlíků.",
    "Paní Nováková říkala, že přijde až po obědě.",
    "Můžeš mi prosím zavolat zpátky? Díky!",
    "To je ale překvapení – úplně jsem na to zapomněl.",
    "Firma O'Brien & syn otevřela pobočku v Ostravě.",
    "Šumění, čeřící se hladina… a pak ticho.",
    "Dvě  mezery uvnitř věty a tabulátor\tnavíc.",
    "Cena je 1 299,50 Kč (bez DPH).",
    "Viděl jsem velké psy u řeky.",
    "Viděl jsem velký psy u řeky.",
    "Viděl jsem velké psi u řeky.",
    "Viděli jsme velké psy u řeky.",
    "Petře, zdraví tě Jana.",
    "Petr, zdravý tě Jana.",
    "Petře zdraví tě, Jana.",
    "a",
    "Ahoj",
]


def logprobs(model, bos: int, ids: list[int]) -> tuple[float, list[float]]:
    import torch

    seq = torch.tensor([[bos] + ids])
    with torch.no_grad():
        logits = model(input_ids=seq).logits.float()
    lp = torch.log_softmax(logits[0, :-1], dim=-1).gather(1, seq[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(lp.sum()), [float(x) for x in lp]


def write_tsv(path: Path, tok, model, bos: int) -> None:
    rows = []
    for s in SENTENCES:
        ids = tok(s.strip(), add_special_tokens=False)["input_ids"]
        total, _ = logprobs(model, bos, ids)
        rows.append(f"{total!r}\t{' '.join(map(str, ids))}\t{s.strip()}")
    # The text is everything after the second tab, so a tab inside a sentence is kept.
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{path}: {len(rows)} sentences")


def cmd_tiny(out: Path) -> None:
    import json

    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_gpt2 import convert

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(SENTENCES * 3, vocab_size=420, min_frequency=2,
                                special_tokens=["<|endoftext|>"], show_progress=False)
        bpe.save_model(str(d))
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        bos = vocab["<|endoftext|>"]
        # The files the real model has, so that AutoTokenizer builds the same GPT-2 tokenizer for both.
        special = {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
                   "pad_token": "<|endoftext|>"}
        (d / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
        (d / "tokenizer_config.json").write_text(json.dumps({"pad_token": "<|endoftext|>"}), encoding="utf-8")

        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=32, n_layer=2, n_head=4,
                         activation_function="gelu_new", layer_norm_epsilon=1e-5, initializer_range=0.3,
                         resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=bos, eos_token_id=bos)
        model = GPT2LMHeadModel(cfg).eval()
        with torch.no_grad():   # non-trivial layer norms and biases, so that the test sees them applied
            for name, p in model.named_parameters():
                if name.endswith(".bias"):
                    p.copy_(0.2 * torch.randn_like(p))
                elif ".ln_" in name or name.startswith("transformer.ln_f"):
                    p.copy_(1.0 + 0.2 * torch.randn_like(p))
        model.save_pretrained(str(d), safe_serialization=True)
        tok = AutoTokenizer.from_pretrained(str(d))
        assert len(tok) == len(vocab), (len(tok), len(vocab))
        out.mkdir(parents=True, exist_ok=True)
        for kind in ("f32", "q8_0"):
            convert(d, out / f"tiny-gpt2-{kind}.gguf", kind, "tiny-gpt2-test", "make_reference.py tiny, seed 0")
            print(out / f"tiny-gpt2-{kind}.gguf", (out / f"tiny-gpt2-{kind}.gguf").stat().st_size, "bytes")
        write_tsv(out / "reference-tiny.tsv", tok, model, bos)


def cmd_czech(hf: Path, out: Path) -> None:
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(str(hf))
    model = GPT2LMHeadModel.from_pretrained(str(hf), dtype=torch.float32).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "reference-czech-gpt2.tsv", tok, model, bos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("czech")
    c.add_argument("--hf", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "tiny":
        cmd_tiny(a.out)
    else:
        cmd_czech(a.hf, a.out)


if __name__ == "__main__":
    main()

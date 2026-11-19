#!/usr/bin/env python3
"""Makes `replay-synthetic.tsv`, the sample of StreamingReplayTest that runs everywhere (no recordings, no model).

The sample is the output of the measurement's own simulation (`simulate_utt` of `dekodovani/prubezny.py`, config
`b-pauza`) over invented speech: made-up words (no real speech, no recordings) in utterances of 7 to 26 s, heard by a
fake recogniser that returns the words of the window it is given, with the faults that matter to the committer: the
first word of a window cut in half, the last one cut off, a letter misheard, empty passes, a pass of punctuation only,
repeated words, and a rewritten overlap right after a cut. The format is the one of `scripts/streaming/export_replay.py`
(U, S and P lines), so the test replays it like the real data and demands the simulation's decisions.

Run (the measurement's code is only read; numpy is needed, e.g. the research environment's Python):

    python make_synthetic_replay.py --research <RESEARCH> --out replay-synthetic.tsv

where <RESEARCH> is the `research` directory that holds `dekodovani/prubezny.py` (it is not modified, nor anything
else there). Output is deterministic: the same inputs give the same bytes.

One deliberate difference from the simulation's own `quiet_point`: without a recording it returns the end of the range,
which is not a whole millisecond and lies outside the window the SDK's quiet point can pick; here a stand-in picks a
whole-millisecond point, 15 ms or more from both ends of the range, from a hash of the range (as the SDK's does).
"""
import argparse
import json
import math
import random
import sys
import tempfile
import zlib
from pathlib import Path

SR = 16000
SEED = 20261115
SYLLABLES = ["ba", "ko", "le", "mu", "ri", "so", "ta", "ve", "ni", "do", "ža", "če", "ho", "pu", "ře", "ky"]


def make_vocab(rng: random.Random) -> list[str]:
    words = set()
    while len(words) < 34:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.choice([1, 2, 2, 3, 3, 4]))))
    return sorted(words) + ["no", "a", "to"]


def make_stream(rng: random.Random, vocab: list[str], n_utts: int) -> list[dict]:
    """Invented utterances on one timeline: words with start/end in ms, then the endpointer's numbers."""
    utts, t = [], 2_000
    for _ in range(n_utts):
        length_s = rng.choice([7, 9, 12, 15, 18, 22, 26])
        start = t
        words, at = [], start
        while at < start + length_s * 1000:
            w = rng.choice(vocab)
            dur = rng.randint(230, 460)
            words.append({"w": w, "s": at, "e": at + dur})
            at += dur + rng.randint(30, 120)
            if rng.random() < 0.09:                     # a repeated word, "no no"
                words.append({"w": w, "s": at, "e": at + dur})
                at += dur + rng.randint(30, 90)
        end = words[-1]["e"]
        for i, w in enumerate(words):                    # punctuation of the "truth": a comma, a sentence end
            w["comma"] = i % 7 == 3
            w["stop"] = i % 13 == 12
        a0 = (start - 320) * SR // 1000 // 16 * 16
        emit = end + 600
        a1 = math.ceil(emit * SR / 1000 / 1600) * 1600
        utts.append({"words": words, "a0": a0, "a1": a1, "start_ms": start, "end_ms": end, "emit_ms": emit, "cut": None})
        t = emit + rng.randint(1_500, 4_000)
    return utts


def render(w: dict, first: bool, prev_stop: bool, rng: random.Random, last_of_window: bool) -> str:
    text = w["w"]
    if first or prev_stop:
        text = text.capitalize()
    if w["comma"] and rng.random() > 0.1:
        text += ","
    if w["stop"] or (last_of_window and rng.random() < 0.25):
        text += "."
    return text


class FakeRunner:
    def __init__(self, utts: list[dict]):
        self.words = [w for u in utts for w in u["words"]]

    def run(self, wav, start: int, end: int, floor: int, fallback: bool, mode: str, prompt: str = "") -> dict:
        rng = random.Random(zlib.crc32(repr((SEED, start, end, fallback)).encode()))
        s_ms, e_ms = start / 16, end / 16
        out: list[str] = []
        roll = rng.random()
        if roll < (0.0 if fallback else 0.05):
            out = []                                     # an empty pass
        elif roll < (0.0 if fallback else 0.07):
            out = ["…"]                                  # nothing but punctuation
        else:
            inside = [w for w in self.words if w["e"] > s_ms and w["s"] < e_ms]
            prev_stop = False
            for k, w in enumerate(inside):
                cut_start = w["s"] < s_ms - 20
                cut_end = w["e"] > e_ms + 20
                if (cut_start or cut_end) and not fallback and rng.random() < 0.5:
                    continue                             # the half-heard word is dropped
                last = k == len(inside) - 1
                text = render(w, not out, prev_stop, rng, last and not cut_end)
                prev_stop = w["stop"]
                if cut_start and len(w["w"]) > 2:
                    text = text[len(text) // 2:]         # "alší" for "další"
                elif cut_end and len(w["w"]) > 2:
                    text = text[: max(2, len(text) // 2)]
                elif rng.random() < 0.04 and len(text) > 2:
                    i = rng.randrange(len(text) - 1)
                    text = text[:i] + rng.choice("aeiou") + text[i + 1:]
                elif len(out) < 2 and rng.random() < 0.12:
                    text = rng.choice(["no", "a", "to", "je"])   # the overlap after a cut is rewritten
                out.append(text)
        tokens = [[100 + i, (" " + w).encode("utf-8").hex(), -1, 0.9] for i, w in enumerate(out)]
        return {"tokens": tokens, "token_beg": 50_000, "audio_ctx": max(384, math.ceil((end - start) / 320)),
                "text": (" " + " ".join(out)) if out else "", "temps": [0.0], "ms_total": 0}


def quiet_stand_in(wav, lo: int, hi: int) -> int:
    """Whole millisecond, at least 240 samples (15 ms) from both ends, chosen from a hash of the range."""
    slots = (hi - lo - 480) // 16
    return lo + 240 + 16 * (zlib.crc32(repr((SEED, lo, hi)).encode()) % (slots + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", required=True, help="the directory with dekodovani/prubezny.py (read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--utterances", type=int, default=22)
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(a.research).resolve()))
    from dekodovani import prubezny as P

    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "streaming"))
    import export_replay

    P.quiet_point = quiet_stand_in
    rng = random.Random(SEED)
    utts = make_stream(rng, make_vocab(rng), a.utterances)
    stream = {"sid": "S00", "cond": "synthetic", "wav": "no-such-recording.wav",
              "utts": [{k: v for k, v in u.items() if k != "words"} for u in utts]}
    result = P.simulate(stream, P.config("b-pauza"), FakeRunner(utts))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "sim").mkdir()
        with (work / "sim" / "synthetic.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"sid": "S00", "cond": "synthetic", "utts": result["utts"]}) + "\n")
        print("streams=%d utterances=%d passes=%d cuts=%d" % export_replay.export(work, "synthetic"))
        Path(a.out).write_bytes((work / "sdk" / "replay-synthetic.tsv").read_bytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())

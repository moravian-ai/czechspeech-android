#!/usr/bin/env python3
"""Makes `replay-anchor-synthetic.tsv`, the sample of CutSeamReplayTest that runs everywhere (no recordings, no model).

Rows in the format of `zbytek_po_strihu.py export-anchor`: id, arm (L2.5 or H1), words of the piece before, its length in ms,
held words h, the piece's text, the context run's text, and the decision of the measurement's own functions
(`anchor_overlap`, `matched_words`): in_buf, prefix, ok, the text after the anchor. Invented words only. The context run is
made from the end of the piece's words as a recogniser would hear them with the faults that matter to the anchor: the first
word cut in half, a letter misheard, a word dropped, a repeated word, a different ending, nothing in common, then new words.

    python make_synthetic_anchor_replay.py --research <RESEARCH> --out replay-anchor-synthetic.tsv

<RESEARCH> is the `research` directory that holds `dekodovani/zbytek_po_strihu.py` (only read). Deterministic.
"""
import argparse
import importlib.util
import random
import sys
from pathlib import Path

SYLLABLES = ["ba", "ko", "le", "mu", "ri", "so", "ta", "ve", "ni", "do", "ža", "če", "ho", "pu", "ře", "ky"]


def load_research(research: Path):
    sys.path.insert(0, str(research))
    spec = importlib.util.spec_from_file_location("zbytek_po_strihu", research / "dekodovani" / "zbytek_po_strihu.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["zbytek_po_strihu"] = mod
    spec.loader.exec_module(mod)
    return mod


def word(rng):
    w = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice([1, 2, 2, 3, 3, 4])))
    if rng.random() < 0.15:
        w += rng.choice([",", ".", "?"])
    if rng.random() < 0.1:
        w = w.capitalize()
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = load_research(Path(a.research))
    rng = random.Random(20261118)
    rows = []
    for n in range(60):
        count = rng.randint(2, 13)
        piece = [word(rng) for _ in range(count)]
        ms = rng.choice([3000, 4160, 4480, 5000, 5120, 5500, 6400])
        share = min(1.0, 2.5 / (ms / 1000))
        covered = max(1, min(count, int(round(count * share)) + rng.choice([-1, 0, 0, 1])))
        for arm, h in (("L2.5", 0), ("H1", 1)):
            hyp = list(piece[count - covered:])
            fault = rng.choice(["none", "none", "half", "letter", "drop", "repeat", "other", "nothing", "short"])
            if fault == "half" and hyp:
                hyp[0] = hyp[0][len(hyp[0]) // 2:] or hyp[0]
            elif fault == "letter" and hyp:
                i = rng.randrange(len(hyp))
                hyp[i] = "ž" + hyp[i][1:]
            elif fault == "drop" and len(hyp) > 1:
                del hyp[rng.randrange(len(hyp))]
            elif fault == "repeat" and hyp:
                i = rng.randrange(len(hyp))
                hyp.insert(i, hyp[i])
            elif fault == "other":
                hyp = [word(rng) for _ in hyp]
            elif fault == "nothing":
                hyp = []
            elif fault == "short":
                hyp = hyp[-1:]
            hyp += [word(rng) for _ in range(rng.randint(0, 6))]
            committed = piece[:max(0, len(piece) - h)]
            expected = max(0, int(round(len(piece) * share)) - h)
            in_buf, prefix = z.anchor_overlap(hyp, committed, expected, [])
            c = committed[len(committed) - in_buf:] if in_buf > 0 else []
            match = z.matched_words([z.key(w) for w in c], [z.key(w) for w in hyp[:prefix]])
            ok = prefix >= 2 and match >= 2
            rows.append("\t".join(["synthetic/%03d" % n, arm, str(count), str(ms), str(h), " ".join(piece), " ".join(hyp),
                                   str(in_buf), str(prefix), "1" if ok else "0", " ".join(hyp[prefix:])]))
    Path(a.out).write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print("%d rows, %d accepted" % (len(rows), sum(r.split("\t")[9] == "1" for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

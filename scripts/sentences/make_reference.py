"""Reference decisions of sentence joining, from the measurement's own code, for SentenceBoundaryReferenceTest.

The rules of SentenceBoundary are a port of the measurement's own code (the research repository). This script runs that module on invented boundaries with fixed candidate scores (no language
model: the scores are input), so the JVM test checks the port against the original, decision by decision:

    python -m make_reference --research <research repository>/research --out <tsv>      (run from scripts/sentences)

The parameters are those of the measured rule (hranice_eval.py): theta = the tuned margin 5.5557098388671875, theta
for "A. b" lower by log((16.5 / 32.5) / (15.5 / 203.5)), case margin 2.5, the comma candidate, window 20 + 12 words.

TSV columns (tab separated, one row per boundary and score set): A, B, score set, raw_class, the candidates' scores
(name=value, comma separated), decide_class, margin (best join minus split, "-" without split), new A, new B, and the
language model texts of split, join_keep, join and comma ("-" when absent).
Needs only the Python standard library; writes no bytecode into the research tree.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

THETA = 5.5557098388671875
THETA_PL = THETA - math.log((16.5 / 32.5) / (15.5 / 203.5))
CASE = 2.5
USE = ("join", "join_keep", "comma")
NAMES = ("split", "join_keep", "join", "comma")

LONG_A = ("Na schůzi jsme probrali rozpočet na příští rok, plán oprav ve škole a také to, kdo pojede na výlet "
          "s dětmi do hor, protože.")
LONG_B = "Bez toho bychom nic nestihli a rodiče by se zlobili, že jim nikdo nic neřekl včas a znovu."

# Invented boundaries (A, B): every way Whisper writes one, and the cases the rules leave alone.
PAIRS = [
    ("Jasně, ale můžeš nějak.", "s tebou zavolat večer."),       # "A. b"
    ("Zítra ráno pojedeme.", "Do Brna autem."),                   # "A. B"
    ("Přijdu, jakmile", "Skončí porada."),                        # "A B"
    ("Řekl jsem mu,", "Že to stihneme."),                         # "A, B"
    ("Opravdu?", "Ano, opravdu."),                                # question mark
    ("Sešli jsme se 17.", "Listopadu u kašny."),                  # ordinal
    ("Koupil jablka, hrušky atd.", "Pak šel domů."),              # abbreviation
    ("Hlasovali jsme pro", "ODS a lidovce."),                     # acronym after "A B"
    ("Odjeli jsme v roce.", "2019 na jih."),                      # a number: no small letter
    ("Řekla „ano.“", "Pak odešla."),                              # period before a closing quote
    ("To bylo všechno...", "Další bod."),                         # ellipsis
    (LONG_A, LONG_B),                                             # longer than the window
    ("Pozdravuj.", "Honzu a Petru."),                             # a name
    ("Byl jsem v.", "Praze na úřadě."),                           # a place
    ("Ozvi se", "Jakmile dorazíš."),                              # "A B"
    ("Napiš mi, prosím.", "„Kdy přijdeš?“"),                      # opening quote in B
    ("Ahoj (dneska.)", "Jdeme ven."),                             # period before a bracket
    ("Já nevím.", "ČEZ zvýšil ceny."),                            # an acronym after a period
    ("", "Ahoj."),                                                # nothing before
    ("Mám hotovo.", "– a ty?"),                                   # B starts without a letter
    ("Ing.", "Novák přišel."),                                    # abbreviation alone
    ("Bylo to tak…", "Že jsme odešli."),                          # one-character ellipsis
]

# Candidate scores (natural log P), chosen around the thresholds.
SCORE_SETS = [
    {"split": -50.0, "join": -40.0, "join_keep": -45.0, "comma": -47.0},   # a small-letter join wins clearly
    {"split": -50.0, "join": -44.4, "join_keep": -43.0, "comma": -60.0},   # the capital wins by 1.4
    {"split": -50.0, "join": -44.5, "join_keep": -47.0, "comma": -60.0},   # margin 5.5, small letter by exactly 2.5
    {"split": -50.0, "join": -46.3, "join_keep": -48.9, "comma": -44.3},   # the comma wins, small letter by 2.6
    {"split": -40.0, "join": -50.0, "join_keep": -50.0, "comma": -50.0},   # the sentence end wins
    {"split": -50.0, "join": -46.0, "join_keep": -44.4, "comma": -60.0},   # the capital, margin 5.6
    {"split": -50.0, "join": -46.4, "join_keep": -49.0, "comma": -60.0},   # margin 3.6: below both thresholds
    {"split": -50.0, "join": -46.3, "join_keep": -49.0, "comma": -60.0},   # margin 3.7: above the one for "A. b"
]


def rows(H):
    for a, b in PAIRS:
        cls = H.raw_class(a, b)
        texts = H.lm_texts(a, b, comma=True, ka=20, kb=12)
        cands = H.candidates(a, b, comma=True)
        for i, base in enumerate(SCORE_SETS):
            scores = {k: base[k] for k in texts}
            choice = H.decide_class(cls, scores, THETA, CASE, USE, THETA_PL)
            margin = H.margin(scores, USE) if "split" in scores else None
            new_a, new_b = (a, b) if choice == "raw" else cands[choice]
            yield [a, b, str(i), cls, ",".join(f"{k}={v!r}" for k, v in scores.items()), choice,
                   "-" if margin is None else repr(margin), new_a, new_b, *(texts.get(n, "-") for n in NAMES)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", type=Path, required=True, help="the research directory (read only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.research))
    from dekodovani import hranice as H

    lines = ["\t".join(r) for r in rows(H)]
    assert all(c not in "".join(lines) for c in "\r") and all(r.count("\t") == 12 for r in lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines)} rows -> {args.out}")


if __name__ == "__main__":
    main()

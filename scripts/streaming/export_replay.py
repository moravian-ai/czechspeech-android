#!/usr/bin/env python3
"""Exports the passes of the measurement's simulation for the SDK's replay test (StreamingReplayTest).

    export_replay.py passes --research <RESEARCH> --work <WORK> --cfg b-pauza,b-pauza+t2.5

Reads <WORK>/sim/<cfg>.jsonl (rows: sid, cond, utts[].passes[], utts[].words[]) and writes <WORK>/sdk/replay-<cfg>.tsv,
UTF-8, one line per record:

    U <sid> <cond> <utt index> <a0 samples = buf_start of its first pass> <committed words joined by one space>
    S <seam word> <index of the pass after which the buffer was cut>       (zero or more, right after the U line)
    P <pass|final> <buf_start samples> <end samples> <m> <n> <text>

An S line says: the seam after the cut following pass <i> was resolved (by pass or final <i + 1>) with that word index
as the piece's first word of the new buffer (the simulation's `seams[].word`).

Tabs and newlines inside texts become spaces; aborted passes are left out. Only text of public parliament recordings
is written, and nothing of it is committed to the repository.
"""
import argparse
import json
import sys
from pathlib import Path


def clean(text: str) -> str:
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split(" ")).strip()


def export(work: Path, cfg: str) -> tuple[int, int, int, int]:
    src = work / "sim" / f"{cfg}.jsonl"
    out = work / "sdk" / f"replay-{cfg}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    streams = utts = passes = cuts = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8", newline="\n") as o:
        for line in f:
            row = json.loads(line)
            streams += 1
            for ui, u in enumerate(row["utts"]):
                recs = [p for p in u["passes"] if not p["aborted"]]
                if not recs:
                    continue
                utts += 1
                words = " ".join(clean(w["w"]) for w in u["words"])
                o.write(f"U\t{row['sid']}\t{row['cond']}\t{ui}\t{recs[0]['buf_start']}\t{words}\n")
                for s in u["seams"]:
                    if s["word"] is None:
                        raise SystemExit(f"error: unresolved seam in {row['sid']} {row['cond']} utterance {ui}")
                    o.write(f"S\t{s['word']}\t{s['pass']}\n")
                for i, p in enumerate(recs):
                    kind = "final" if p["final"] else "pass"
                    o.write(f"P\t{kind}\t{p['buf_start']}\t{p['end']}\t{p['m']}\t{p['n']}\t{clean(p['text'] or '')}\n")
                    passes += 1
                    if i + 1 < len(recs) and recs[i + 1]["buf_start"] != p["buf_start"]:
                        cuts += 1
    return streams, utts, passes, cuts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["passes"])
    ap.add_argument("--research", required=True, help="the measurement's code (read only; not needed to export passes)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cfg", required=True)
    a = ap.parse_args()
    for cfg in a.cfg.split(","):
        s, u, p, c = export(Path(a.work), cfg)
        print(f"{cfg}: streams={s} utterances={u} passes={p} cuts={c}")
        if min(s, u, p, c) == 0:
            print("error: a count is zero", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

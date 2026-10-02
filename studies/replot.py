"""Regenerate a study's figures from its saved JSON, without re-running it."""
import json, os, sys
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

def load(n):
    with open(os.path.join(OUT, f"{n}.json")) as fh:
        return json.load(fh)

which = sys.argv[1] if len(sys.argv) > 1 else "s4"
if which == "s4":
    import s4_scaling
    d = load("s4_scaling")
    s4_scaling.figure(d["N_sweep"], d["horizon_sweep"])

"""Copy the figures the manuscript includes from ``figs/`` into ``paper/Figs/``.

The studies write to ``figs/``; the manuscript reads ``paper/Figs/``, which also
holds figures that no study produces (the platform photograph, the diagrams of
the geometry).  Keeping the two directories separate means a re-run cannot
silently remove a figure the paper needs, and this script is the one step that
moves the regenerated ones across.
"""

from __future__ import annotations

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "figs")
PAPER_FIGS = os.path.join(ROOT, "paper", "Figs")
MANUSCRIPT = os.path.join(ROOT, "paper", "manuscript.tex")


def wanted() -> list[str]:
    """Figure file names the manuscript includes."""
    with open(MANUSCRIPT) as fh:
        return sorted({os.path.basename(m)
                       for m in re.findall(r"Figs/([^}]+)", fh.read())})


def main() -> None:
    if not os.path.exists(MANUSCRIPT):
        print("  no paper/manuscript.tex in this checkout; nothing to sync")
        return
    os.makedirs(PAPER_FIGS, exist_ok=True)
    copied, missing = [], []
    for name in wanted():
        src = os.path.join(FIGS, name)
        if os.path.exists(src):
            dst = os.path.join(PAPER_FIGS, name)
            if os.path.exists(dst) and not os.access(dst, os.W_OK):
                os.chmod(dst, 0o644)
            shutil.copy2(src, dst)
            copied.append(name)
        elif not os.path.exists(os.path.join(PAPER_FIGS, name)):
            missing.append(name)
    print(f"  synced {len(copied)} figures into paper/Figs")
    if missing:
        print("  MISSING (neither generated nor in paper/Figs): " + ", ".join(missing))


if __name__ == "__main__":
    main()

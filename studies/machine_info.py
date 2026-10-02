"""Record the machine and software the studies ran on, for the paper.

The solve times reported in the paper are only meaningful together with the
machine that measured them, so the runners record it in ``out/machine.json``
and ``make_results_tex.py`` writes it into the text.
"""

from __future__ import annotations

import os
import platform

import casadi

from common import dump


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return " ".join(line.split(":", 1)[1].split())
    except OSError:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    info = {"cpu": cpu_model(), "cores": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "casadi": casadi.__version__}
    print("  " + ", ".join(f"{k}: {v}" for k, v in info.items()), flush=True)
    dump(info, "machine")

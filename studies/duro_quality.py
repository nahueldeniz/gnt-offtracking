"""Fix quality of the secondary (Swift Duro) receiver on the last trailer.

The previous version of this work described the Duro's output as ground truth
for the last trailer. The logs do not support that. This script reads the
``MSG_POS_LLH`` records (SBP message 522) from the receiver's own logs and
reports, per file and over the campaign, the distribution of the fix-mode flag
and the receiver's self-reported horizontal accuracy.

In SBP the low three bits of the ``MSG_POS_LLH`` flags field encode the fix
mode: 0 invalid, 1 single-point positioning, 2 DGNSS, 3 float RTK, 4 fixed RTK,
5 dead reckoning, 6 SBAS. Only modes 3 and 4 give the centimetre-level accuracy
that a ground-truth claim needs.

The logs are not in the repository (they are part of the released data set);
point ``--root`` at the experiment folder to re-run this.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics

from common import OUT, dump

FLAG_NAME = {0: "invalid", 1: "spp", 2: "dgnss", 3: "float_rtk",
             4: "fixed_rtk", 5: "dead_reckoning", 6: "sbas"}
MSG_POS_LLH = 522


def scan(path):
    """Fix-mode counts, horizontal accuracies and satellite counts of one log."""
    flags = collections.Counter()
    acc, sats = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            # cheap pre-filter: parsing every line of a multi-megabyte log is
            # an order of magnitude slower than the substring test
            if '"msg_type":522' not in line.replace(" ", ""):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("msg_type") != MSG_POS_LLH or "flags" not in rec:
                continue
            flags[rec["flags"] & 0x7] += 1
            if "h_accuracy" in rec:
                acc.append(float(rec["h_accuracy"]))
            if "n_sats" in rec:
                sats.append(int(rec["n_sats"]))
    return flags, acc, sats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="folder holding the experiment subfolders")
    args = ap.parse_args()
    root = args.root or os.environ.get("GNT_FIELD_LOGS")
    if root is None:
        raise SystemExit("pass --root, or set GNT_FIELD_LOGS: the receiver logs "
                         "are part of the released data set, not of this repository")
    root = os.path.abspath(os.path.expanduser(root))

    files = sorted(glob.glob(os.path.join(root, "*", "*.sbp.json")))
    if not files:
        raise SystemExit(f"no .sbp.json logs under {root}")

    total = collections.Counter()
    all_acc, all_sats, per_file = [], [], []
    for f in files:
        flags, acc, sats = scan(f)
        total.update(flags)
        all_acc += acc
        all_sats += sats
        per_file.append({
            "file": os.path.relpath(f, root),
            "epochs": sum(flags.values()),
            "flags": {FLAG_NAME.get(k, str(k)): v for k, v in sorted(flags.items())},
            "h_acc_median_mm": statistics.median(acc) if acc else None,
            "n_sats_median": statistics.median(sats) if sats else None,
        })
        print(f"{per_file[-1]['file']:34s} {per_file[-1]['epochs']:6d} epochs  "
              f"{per_file[-1]['flags']}  median h_acc "
              f"{per_file[-1]['h_acc_median_mm']} mm")

    n = sum(total.values())
    all_acc.sort()
    rtk = total[3] + total[4]
    out = {
        "n_files": len(files),
        "n_epochs": n,
        "flags": {FLAG_NAME.get(k, str(k)): v for k, v in sorted(total.items())},
        "rtk_fraction": rtk / n if n else None,
        "spp_fraction": total[1] / n if n else None,
        "h_acc_median_m": statistics.median(all_acc) / 1000.0 if all_acc else None,
        "h_acc_p10_m": all_acc[len(all_acc) // 10] / 1000.0 if all_acc else None,
        "h_acc_p90_m": all_acc[9 * len(all_acc) // 10] / 1000.0 if all_acc else None,
        "n_sats_median": statistics.median(all_sats) if all_sats else None,
        "per_file": per_file,
    }
    print(f"\ncampaign: {n} epochs over {len(files)} logs, "
          f"{100*out['spp_fraction']:.1f}% single-point, "
          f"{100*out['rtk_fraction']:.1f}% RTK; "
          f"median horizontal accuracy {out['h_acc_median_m']:.2f} m "
          f"(p10 {out['h_acc_p10_m']:.2f}, p90 {out['h_acc_p90_m']:.2f})")
    dump(out, "duro_quality")


if __name__ == "__main__":
    main()

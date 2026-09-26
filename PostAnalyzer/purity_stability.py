"""
purity_stability.py
───────────────────
Purity і stability у бінах перерізу для всіх методів реконструкції, разом із дослідницькими
моделями lkrnn* (nn_apply.py --study).

  purity_i    = N(gen у біні i і reco у біні i) / N(reco у біні i)
  stability_i = N(gen у біні i і reco у біні i) / N(gen у біні i)

Рахуються події, що пройшли відбір (det) і мають фізичний розв'язок методу. Алгоритми — на всіх
подіях, нейромережі (lkrnn*) — лише на подіях поза тренуванням (nn_split.eval_sample), як в
integral_res.py. Похибки біноміальні.

Біни за замовчуванням — біни перерізу з ttbarMakeHist.cxx (гістограми h_*_cs, аналіз CMS 7 ТеВ);
--binning fine бере дрібні біни рисунків (kinreco_eff_v4.BINS).

  python3 purity_stability.py
  python3 purity_stability.py --input-pattern 'studies/<прогін>/apply/ttbar_output_study_{ch}.root' \
      --outdir studies/<прогін>/purity
  python3 purity_stability.py --level gen
Вихід у --outdir: purity_stability_<рівень>.txt, .json, .png, .pdf
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

import kinreco_eff_v4 as k4
import nn_split

CS_BINS = {   # біни перерізу з ttbarMakeHist.cxx (h_mtt_cs, h_pttt_cs, h_ytt_cs)
    "mtt": [0., 345., 400., 470., 550., 650., 800., 1100., 1600.],
    "pttt": [0., 20., 60., 120., 300.],
    "ytt": [-2.5, -1.5, -0.7, 0.0, 0.7, 1.5, 2.5],
}
VARS = ["mtt", "pttt", "ytt"]
BASE_METHODS = ["lkr", "lkrv2", "lkrv3", "lkrnn"]
NICE = {"lkr": "LKR", "lkrv2": "LKRv2", "lkrv3": "LKRv3", "lkrnn": "LKRnn"}
COLORS = {"lkr": "tab:blue", "lkrv2": "tab:red", "lkrv3": "tab:orange", "lkrnn": "tab:green"}
EXTRA_COLORS = ["tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray", "black"]


def study_methods(fname):
    """Дослідницькі методи lkrnn* у файлі (гілки, записані nn_apply.py --study)."""
    keys = set(uproot.open(fname)["ttbarTree"].keys())
    return sorted(k for k in keys if k.startswith("lkrnn") and k not in ("lkrnn", "lkrnn_gen")
                  and not k.endswith("_gen") and f"mtt_{k}" in keys)


def purity_stability(gen, reco, sel, bins):
    """Purity, stability (з похибками) і кількості подій по бінах для подій sel."""
    bins = np.asarray(bins, float)
    nb = len(bins) - 1
    g = np.digitize(gen[sel], bins) - 1          # поза межами -> -1 або nb, тобто ні в якому біні
    r = np.digitize(reco[sel], bins) - 1
    out = {k: [] for k in ("purity", "purity_err", "stability", "stability_err", "n_reco", "n_gen", "n_both")}
    for i in range(nb):
        in_g, in_r = g == i, r == i
        n_g, n_r, n_b = int(in_g.sum()), int(in_r.sum()), int((in_g & in_r).sum())
        p = n_b / n_r if n_r else float("nan")
        s = n_b / n_g if n_g else float("nan")
        out["purity"].append(p)
        out["purity_err"].append(float(np.sqrt(p * (1 - p) / n_r)) if n_r else float("nan"))
        out["stability"].append(s)
        out["stability_err"].append(float(np.sqrt(s * (1 - s) / n_g)) if n_g else float("nan"))
        out["n_reco"].append(n_r)
        out["n_gen"].append(n_g)
        out["n_both"].append(n_b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_det_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--level", choices=["det", "gen"], default="det")
    ap.add_argument("--split-model", dest="split_model", choices=["det", "gen"], default=None,
                    help="модель, чиї тренувальні події виключаються (за замовч. = --level; для gen–det: gen)")
    ap.add_argument("--binning", choices=["cs", "fine"], default="cs",
                    help="cs — біни перерізу з ttbarMakeHist.cxx; fine — біни рисунків")
    ap.add_argument("--outdir", default="purity_stability")
    args = ap.parse_args()
    sfx = "" if args.level == "det" else "_gen"
    split_model = args.split_model or args.level
    bins_all = CS_BINS if args.binning == "cs" else {v: list(k4.BINS[v]) for v in VARS}

    files = [(ch, args.input_pattern.format(ch=ch)) for ch in args.channels
             if os.path.exists(args.input_pattern.format(ch=ch))]
    if not files:
        raise SystemExit("[E] немає вхідних файлів")
    study = study_methods(files[0][1])
    k4.METHODS = BASE_METHODS + study
    k4.VARS = VARS
    parts = []
    for ch, f in files:
        a = k4.load(f)
        a["_sample"], _ = nn_split.eval_sample(f, ch, split_model)
        parts.append(a)
    a = k4.combine(parts) if len(parts) > 1 else parts[0]
    methods = [m for m in k4.METHODS if f"{m}{sfx}" in a and f"mtt_{m}{sfx}" in a]
    for i, m in enumerate(study):
        COLORS.setdefault(m, EXTRA_COLORS[i % len(EXTRA_COLORS)])
    NICE.update(nn_split.method_labels(files[0][1]))                 # короткі підписи, напр. "MLP (raw)"
    base = (a["reco_passed_selection"] == 1) if args.level == "det" else np.ones(len(a["mtt_gen"]), bool)

    result = {"level": args.level, "input_pattern": args.input_pattern, "channels": [c for c, _ in files],
              "binning": args.binning, "bins": bins_all, "methods": {}}
    for m in methods:
        result["methods"][m] = {}
        for v in VARS:
            reco, gen = a[f"{v}_{m}{sfx}"], a[f"{v}_gen"]
            sel = base & (a[f"{m}{sfx}"] == 1) & k4.physical(reco, v) & np.isfinite(gen)
            if nn_split.uses_nn(m):
                sel &= a["_sample"]
            result["methods"][m][v] = purity_stability(gen, reco, sel, bins_all[v])

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, f"purity_stability_{args.level}")
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=1)

    lines = [f"PURITY / STABILITY, рівень {args.level}, файли {args.input_pattern}, канали {result['channels']}, "
             f"біни {args.binning}",
             "Алгоритми — усі події; lkrnn* — лише події поза тренуванням NN. Формат: purity / stability"]
    for v in VARS:
        b = bins_all[v]
        lines.append(f"\n--- {v} ---")
        lines.append(f"{'бін':>16} | " + " | ".join(f"{NICE.get(m, m):>15}" for m in methods))
        for i in range(len(b) - 1):
            cells = []
            for m in methods:
                r = result["methods"][m][v]
                cells.append(f"{r['purity'][i]:.3f} / {r['stability'][i]:.3f}" if r["n_gen"][i] else "—")
            lines.append(f"{b[i]:>7g}–{b[i + 1]:<7g} | " + " | ".join(f"{c:>15}" for c in cells))
    text = "\n".join(lines)
    with open(stem + ".txt", "w") as f:
        f.write(text + "\n")
    print(text)

    fig, axes = plt.subplots(2, len(VARS), figsize=(13, 7.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.3)
    for col, v in enumerate(VARS):
        b = np.asarray(bins_all[v], float)
        lo = b[:-1].copy()
        if v == "mtt" and lo[0] == 0:           # перший бін 0–345 ГеВ фізично порожній для gen
            lo[0] = b[1] - 50
        cen, half = 0.5 * (lo + b[1:]), 0.5 * (b[1:] - lo)
        n_m = len(methods)
        for j, m in enumerate(methods):
            r = result["methods"][m][v]
            shift = (j - (n_m - 1) / 2) * 0.08 * half   # невеликий зсув, щоб точки не накладались
            for row, key in enumerate(("purity", "stability")):
                axes[row, col].errorbar(cen + shift, r[key], yerr=r[key + "_err"], xerr=half, fmt="o", ms=4,
                                        color=COLORS.get(m), label=NICE.get(m, m), capsize=0, elinewidth=1)
        for row, key in enumerate(("Purity", "Stability")):
            ax = axes[row, col]
            ax.set_xlabel(k4.LABELS[v], fontsize=11)
            ax.set_ylabel(key, fontsize=11)
            ax.grid(True, alpha=0.4)
            ax.set_ylim(0, 1)
            if col == 0:
                ax.legend(fontsize=8)
    fig.savefig(stem + ".png", dpi=150)
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print(f"\n[I] збережено: {stem}.txt, .json, .png, .pdf")


if __name__ == "__main__":
    main()

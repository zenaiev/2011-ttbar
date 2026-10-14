"""
seed_metrics.py
───────────────
Розкид метрик між тренуваннями з різними сідами: σ(M), σ(M) біля порога, |bias|(M), σ(pT), σ(y), Δχ² трьох
сценаріїв fold_check.py (той самий бінінг) і cond(A) матриці відгуку (response_matrix.py, біни перерізу).
Групи — набори гілок одного методу з різними сідами; для кожної групи — кожен сід, середнє і стандартне відхилення.

  python seed_metrics.py --input-pattern 'ttbar_output_fig17_{ch}.root' \\
      --group "MLP=lkrnnmlpl1zmlj,lkrnnmlps1,lkrnnmlps2,lkrnnmlps3,lkrnnmlps4"
"""
import argparse

import numpy as np

import fold_check as fc
import integral_res as ir
import nn_split
import response_matrix as rm

REF = ["gen", "lkr", "lkrnn", "lkrnnbndraw", "lkrnnfreeraw", "lkrnnmlpl1zmlj", "lkrnnflowmlj"]   # бінінг як у fold_check
COLS = ["σ(M)", "σ(M)thr", "|bias|M", "σ(pT)", "σ(y)", "Δχ²slope", "Δχ²res", "Δχ²thr", "cond(A)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_fig17_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--group", action="append", required=True, metavar="НАЗВА=гілка1,гілка2,...")
    args = ap.parse_args()

    loaded = []
    for ch in args.channels:
        f = args.input_pattern.format(ch=ch)
        a, _ = ir.read_arrays(f)
        a["_sample"], _ = nn_split.eval_sample(f, ch, "det")
        loaded.append(a)
    a = ir.combine_arrays(loaded)
    base = (a["reco_passed_selection"] == 1) & a["_sample"]
    truth = a["mtt_gen"]

    def sel(m):
        return base if m == "gen" else base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")

    ref = [(m, m, (truth if m == "gen" else a[f"mtt_{m}"])[sel(m)], truth[sel(m)]) for m in REF]
    edges = fc.make_edges(ref, 100, 20, 3)
    g = ref[0][2]
    for _, wf, _ in fc.SCENARIOS:
        for _ in range(5):
            fc.SCALE[id(wf)] = fc.SCALE.get(id(wf), 1.0) * 5 / np.sqrt(fc.chi2(g, fc.weight(wf, g), edges)[0])
    te = np.append(np.array(rm.TRUE_EDGES), rm.INF)
    re_ = rm.reco_edges(rm.TRUE_EDGES, 2)
    tb = truth[base]

    def metrics(m):
        s = sel(m)
        reco, tr = a[f"mtt_{m}"][s], truth[s]
        d = reco - tr
        out = [d.std(), d[(tr > 340) & (tr < 362)].std(), ir.abs_bias(a[f"mtt_{m}"], truth, s, "mtt"),
               (a[f"pttt_{m}"] - a["pttt_gen"])[s].std(), (a[f"ytt_{m}"] - a["ytt_gen"])[s].std()]
        out += [fc.chi2(reco, fc.weight(wf, tr), edges)[0] for _, wf, _ in fc.SCENARIOS]
        ok = s[base]
        _, nt, A = rm.response(np.where(s, a[f"mtt_{m}"], 0.0)[base], ok, tb, te, re_)
        return out + [rm.condition(A, nt)[0]]

    print(f"{'':28}" + "".join(f"{c:>10}" for c in COLS))
    for spec in args.group:
        name, branches = spec.split("=", 1)
        rows = []
        for m in branches.split(","):
            r = metrics(m)
            rows.append(r)
            print(f"  {m:26}" + "".join(f"{v:10.3f}" for v in r))
        R = np.array(rows)
        print(f"{name + ': середнє':28}" + "".join(f"{v:10.3f}" for v in R.mean(0)))
        print(f"{name + ': розкид (σ)':28}" + "".join(f"{v:10.3f}" for v in R.std(0, ddof=1)))
        print()


if __name__ == "__main__":
    main()

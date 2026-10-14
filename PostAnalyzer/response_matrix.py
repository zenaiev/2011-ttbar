"""
response_matrix.py
──────────────────
Матриця відгуку за M(tt̄) для кожного методу реконструкції і її число обумовленості.

  A_ij = P(reco у біні i | правда в біні j) = N_ij / N_j,
N_ij — кількість подій MC з правдою в біні j і reco в біні i, N_j — усі події з правдою в біні j (разом із тими,
де метод не дав розв'язку), тож стовпець сумується до ефективності методу в біні правди.

Число обумовленості (Blobel, PHYSTAT 2011, розд. 2.1): cond(A) = σ₁/σₙ — відношення найбільшого й найменшого
сингулярних чисел A; верхня межа коефіцієнта, з яким анфолдінг підсилює відносну похибку даних. cond ≈ 1 —
відгук майже діагональний. Друге значення — після масштабування рядків на 1/√y_i (y = A·N, як у Blobel перед SVD);
воно залежить ще й від заселеності бінів (навіть для ідеального детектора > 1).

--binning cs (за замовч.): біни правди — біни перерізу (--true-edges, останній до ∞); reco — кожен бін правди
поділений на --split частин (m > n, як радить Blobel), плюс 0–345 і 1600–∞.
--binning chi2: і правда, і reco — ті самі біни, що в fold_check.py для Δχ² (≥ 100 подій MC на бін для кожного
методу, переважно по 1 ГеВ). Біни значно вужчі за роздільну здатність, тож матриця майже вироджена і cond(A)
відображає здебільшого статистичний шум MC; вихідні файли — з суфіксом _chi2. Усі методи — на тих самих подіях поза тренуванням мереж (~15%,
eval_sample), як у fold_check.py.

Вихід у --outdir:
  response_matrix.root  — для кожного методу M: TH2D resp_M (кількості N_ij; x — правда, y — reco) і
                          prob_M (A_ij), TH1D truth (N_j); останній бін — до 10⁴ ГеВ (тобто «∞»);
  response_matrix.txt   — таблиця cond(A) і матриці A_ij у текстовому вигляді;
  response_matrix.png/.pdf — A_ij для методів --plot-methods.
  З --binning chi2 (332 × 332): ті самі файли з суфіксом _chi2, а матриці — в окремих
  response_matrix_chi2_<метод>.txt (межі бінів, N_j, кількості N_ij і ймовірності A_ij).

  python response_matrix.py --input-pattern 'ttbar_output_fig9_{ch}.root'
"""

import argparse
import os

import numpy as np
import uproot

import integral_res as ir
import nn_split
from fold_check import METHODS, make_edges

TRUE_EDGES = [345.0, 400.0, 470.0, 550.0, 650.0, 800.0, 1100.0]   # біни перерізу (ttbarMakeHist.cxx h_mtt_cs)
RECO_MAX = 1600.0                                                  # вище — один reco-бін переповнення
INF = 1e4                                                          # «∞» для збережених гістограм


def reco_edges(true_edges, split):
    """Кожен бін правди — на split частин (останній — до RECO_MAX), плюс 0–перша межа і RECO_MAX–∞."""
    T = list(true_edges) + [RECO_MAX]
    inner = np.concatenate([np.linspace(T[k], T[k + 1], split + 1)[:-1] for k in range(len(T) - 1)])
    return np.concatenate([[0.0], inner, [RECO_MAX, INF]])


def response(reco, ok, truth, t_edges, r_edges):
    """Кількості N_ij, N_j і ймовірності A_ij (події поза межами — в крайні біни)."""
    nt, nr = len(t_edges) - 1, len(r_edges) - 1
    t = np.clip(np.digitize(truth, t_edges) - 1, 0, nt - 1)
    r = np.clip(np.digitize(reco, r_edges) - 1, 0, nr - 1)
    N = np.zeros((nr, nt))
    np.add.at(N, (r[ok], t[ok]), 1)
    n_true = np.bincount(t, minlength=nt).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return N, n_true, np.nan_to_num(N / n_true)


def condition(A, n_true):
    """cond(A) = σ₁/σₙ і те саме після масштабування рядків на 1/√y_i."""
    A = A[:, n_true > 0]                                             # порожні біни правди не входять
    s = np.linalg.svd(A, compute_uv=False)
    y = A @ n_true[n_true > 0]
    sw = np.linalg.svd(A[y > 0] / np.sqrt(y[y > 0])[:, None], compute_uv=False)
    return s[0] / s[-1], sw[0] / sw[-1], s[0], s[-1]


def write_matrix_txt(path, lab, N, n_true, A, t_edges, r_edges):
    """Матриця одного методу в txt: межі бінів, N_j, кількості N_ij (цілі) і ймовірності A_ij.
    Рядок i — reco-бін, стовпець j — бін правди; останній бін — до ∞ (записано як 10⁴ ГеВ)."""
    fmt = lambda e: " ".join(f"{x:g}" for x in e)
    with open(path, "w") as f:
        f.write(f"# Матриця відгуку M(tt̄), метод: {lab}\n"
                f"# рядки — reco-біни ({len(r_edges) - 1}), стовпці — біни правди ({len(t_edges) - 1})\n"
                f"# межі бінів правди, ГеВ:\n{fmt(t_edges)}\n"
                f"# межі reco-бінів, ГеВ:\n{fmt(r_edges)}\n"
                f"# N_j — усі події MC з правдою в біні j (знаменник A_ij):\n{fmt(n_true)}\n"
                f"# N_ij — кількість подій MC (правда в біні j, reco в біні i):\n")
        np.savetxt(f, N, fmt="%d")
        f.write("# A_ij = N_ij / N_j — імовірність reco в біні i для правди в біні j:\n")
        np.savetxt(f, A, fmt="%.6g")


def label(edges, k):
    return f"{edges[k]:.0f}–" + ("∞" if edges[k + 1] >= INF else f"{edges[k + 1]:.0f}")


def plot(mats, t_edges, r_edges, outdir, stem_name="response_matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1, 4.6), squeeze=False, gridspec_kw={"wspace": 0.08})
    tl = [label(t_edges, k) for k in range(len(t_edges) - 1)]
    rl = [label(r_edges, k) for k in range(len(r_edges) - 1)]
    fine = len(tl) > 30                                              # дрібні біни: без підписів клітинок
    from matplotlib.colors import LogNorm
    for ax, (lab, A, cond) in zip(axes[0], mats):
        if fine:                                                     # ймовірності ~1e-3 — логарифмічна шкала
            im = ax.imshow(np.where(A > 0, A, np.nan), origin="lower", aspect="auto", cmap="Blues",
                           norm=LogNorm(vmin=1e-4, vmax=1), interpolation="nearest")
        else:
            im = ax.imshow(A, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{lab}\ncond(A) = {cond:.3g}", fontsize=10)
        xt = range(0, len(tl), max(1, len(tl) // 12)) if fine else range(len(tl))
        yt = range(0, len(rl), max(1, len(rl) // 20)) if fine else range(len(rl))
        ax.set_xticks(list(xt))
        ax.set_xticklabels([tl[k] for k in xt], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(list(yt))
        ax.set_yticklabels([rl[k] for k in yt] if ax is axes[0][0] else [], fontsize=6)
        ax.set_xlabel(r"true $M(t\bar t)$ [GeV]", fontsize=9)
        if ax is axes[0][0]:
            ax.set_ylabel(r"reco $M(t\bar t)$ [GeV]", fontsize=9)
        for (i, j), v in (np.ndenumerate(A) if not fine else []):
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > 0.5 else "#222222")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="P(reco bin | true bin)")
    stem = os.path.join(outdir, stem_name)
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print(f"[I] {stem}.png, .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_fig9_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--true-edges", dest="true_edges", type=float, nargs="+", default=TRUE_EDGES,
                    help="межі бінів правди, ГеВ (останній бін — до ∞)")
    ap.add_argument("--split", type=int, default=2, help="на скільки reco-бінів ділиться кожен бін правди")
    ap.add_argument("--binning", choices=["cs", "chi2"], default="cs",
                    help="cs — біни перерізу (7 правди × 16 reco); chi2 — біни Δχ² із fold_check.py для правди й reco")
    ap.add_argument("--plot-methods", dest="plot_methods", nargs="+",
                    default=["lkr", "lkrnnfreeraw", "lkrnnmlpl1zmlj", "lkrnnflowmlj", "lkrnnmlp", "lkrnnflow"],
                    help="методи на рисунку (ті, що є у файлах)")
    ap.add_argument("--outdir", default="figs_best")
    args = ap.parse_args()

    loaded = []
    for ch in args.channels:
        f = args.input_pattern.format(ch=ch)
        a, present = ir.read_arrays(f)
        a["_sample"], _ = nn_split.eval_sample(f, ch, "det")
        loaded.append(a)
    a = ir.combine_arrays(loaded)
    base = (a["reco_passed_selection"] == 1) & a["_sample"]
    truth = a["mtt_gen"][base]
    t_edges = np.append(np.array(args.true_edges, float), INF)
    r_edges = reco_edges(args.true_edges, args.split)
    stem_name = "response_matrix"
    if args.binning == "chi2":                                       # ті самі біни, що для Δχ² у fold_check.py
        sets = []
        for m, lab in METHODS:
            if m != "gen" and m not in present:
                continue
            sel = base if m == "gen" else base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")
            sets.append((m, lab, (a["mtt_gen"] if m == "gen" else a[f"mtt_{m}"])[sel], a["mtt_gen"][sel]))
        t_edges = make_edges(sets, 100, 20, 3)
        t_edges[-1] = INF
        r_edges = t_edges
        stem_name = "response_matrix_chi2"

    os.makedirs(args.outdir, exist_ok=True)
    root_file = os.path.join(args.outdir, stem_name + ".root")
    L = [f"Матриця відгуку M(tt̄): {base.sum()} подій MC (~15% поза тренуванням мереж), усі методи на тих самих подіях.",
         f"Біни правди ({len(t_edges) - 1}), ГеВ: " + " | ".join(f"{e:.0f}" for e in t_edges[:-1]) + " | ∞",
         f"Біни reco ({len(r_edges) - 1}), ГеВ: " + " | ".join(f"{e:.0f}" for e in r_edges[:-1]) + " | ∞",
         "",
         f"{'метод':<22}{'cond(A)':>12}{'(масштаб. 1/√y)':>18}{'σ₁':>9}{'σₙ':>11}{'ефективність':>14}"]
    mats, blocks, mat_files = [], [], []
    with uproot.recreate(root_file) as out:
        out["truth"] = (np.bincount(np.clip(np.digitize(truth, t_edges) - 1, 0, len(t_edges) - 2),
                                    minlength=len(t_edges) - 1).astype(float), t_edges)
        for m, lab in METHODS:
            if m != "gen" and m not in present:
                continue
            if m == "gen":
                reco, ok = truth, np.ones(len(truth), bool)
            else:
                reco = a[f"mtt_{m}"][base]
                ok = (a[m][base] == 1) & ir.physical(reco, "mtt")
                reco = np.where(ok, reco, 0.0)
            N, n_true, A = response(reco, ok, truth, t_edges, r_edges)
            c, cw, s1, sn = condition(A, n_true)
            out[f"resp_{m}"] = (N.T, t_edges, r_edges)                # x — правда, y — reco
            out[f"prob_{m}"] = (A.T, t_edges, r_edges)
            L.append(f"{lab:<22}{c:12.4g}{cw:18.4g}{s1:9.3f}{sn:11.3g}{100 * N.sum() / n_true.sum():13.1f}%")
            if m == "gen" or m in args.plot_methods:
                mats.append((lab, A, c))
            if len(t_edges) > 40:                                    # великі матриці — окремий txt на метод
                mfile = os.path.join(args.outdir, f"{stem_name}_{m}.txt")
                write_matrix_txt(mfile, lab, N, n_true, A, t_edges, r_edges)
                mat_files.append(mfile)
                continue
            blocks += ["", f"A_ij — {lab} (рядки — reco, стовпці — правда)",
                       f"{'reco \\ правда':>12}" + "".join(f"{label(t_edges, j):>10}" for j in range(len(t_edges) - 1))]
            blocks += [f"{label(r_edges, i):>12}" + "".join(f"{v:10.4f}" for v in A[i]) for i in range(len(A))]
    text = "\n".join(L + blocks)
    txt = os.path.join(args.outdir, stem_name + ".txt")
    open(txt, "w").write(text + "\n")
    print("\n".join(L))
    print(f"[I] {root_file} (resp_<метод>, prob_<метод>, truth); {txt}")
    if mat_files:
        print(f"[I] матриці в txt ({len(mat_files)} файлів): {os.path.join(args.outdir, stem_name)}_<метод>.txt")
    plot(mats, t_edges, r_edges, args.outdir, stem_name)


if __name__ == "__main__":
    main()

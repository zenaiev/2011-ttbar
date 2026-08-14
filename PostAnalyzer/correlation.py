"""
plot_correlation.py
───────────────────
Будує 2D кореляційні матриці reco vs gen для всіх алгоритмів.
Читає дані ВИКЛЮЧНО з ROOT файлу, що гарантує однакову кількість подій.
 
Запуск:
  python3 plot_correlation.py
"""
 
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import uproot
 
pi = np.pi
 
VARNAMES = ["mtt", "pttt", "ytt", "phitt"]
LABELS = {
    "mtt":   r"$M(t\bar{t})$ [GeV]",
    "pttt":  r"$p_T(t\bar{t})$ [GeV]",
    "ytt":   r"$y(t\bar{t})$",
    "phitt": r"$\phi(t\bar{t})$ [rad]",
}
RANGES = {
    "mtt":   (340,  1400),
    "pttt":  (0,    500),
    "ytt":   (-2.5, 2.5),
    "phitt": (-pi,  pi),
}
NBINS = 50
 
 
def pearson_r(x, y):
    """Коефіцієнт кореляції Пірсона."""
    xm, ym = x - x.mean(), y - y.mean()
    return np.sum(xm * ym) / np.sqrt(np.sum(xm**2) * np.sum(ym**2) + 1e-12)
 
 
def plot_correlation_matrix(preds_dict, targets, outdir, title_suffix=""):
    """
    preds_dict: {"LKR": array(N,4), "LKRV3": array(N,4), "LKRNN": array(N,4)}
    targets:    array(N,4)  — gen-рівень
    """
    os.makedirs(outdir, exist_ok=True)
    n_algos = len(preds_dict)
    n_vars  = len(VARNAMES)
 
    fig, axes = plt.subplots(
        n_vars, n_algos,
        figsize=(5 * n_algos, 4.5 * n_vars),
        squeeze=False # Захист, якщо алгоритм лише 1
    )
 
    fig.suptitle(
        r"CMS open data $pp\to t\bar{t}$, dilepton, $\sqrt{s}=7$ TeV"
        f"\nReco vs Gen correlation{title_suffix}",
        fontsize=13
    )
 
    for col, (algo_name, preds) in enumerate(preds_dict.items()):
        # targets може бути словником {алгоритм: gen-масив} — LKRnn має власний
        # (held-out) набір подій, тож gen-рівень для нього інший
        tgt = targets[algo_name] if isinstance(targets, dict) else targets
        for row, vn in enumerate(VARNAMES):
            ax  = axes[row, col]
            idx = row  # колонка в масиві [mtt, pttt, ytt, phitt]

            gen  = tgt[:, idx]
            reco = preds[:,  idx]
 
            # Фільтруємо невдалі реконструкції (значення -1000)
            mask = (reco > -999) & (reco < 9999) & np.isfinite(reco) & np.isfinite(gen)
            gen  = gen[mask]
            reco = reco[mask]
 
            vmin, vmax = RANGES[vn]
 
            # 2D гістограма
            h, xedges, yedges = np.histogram2d(
                gen, reco,
                bins=NBINS,
                range=[[vmin, vmax], [vmin, vmax]]
            )
 
            # Нормуємо по стовпцях (умовний розподіл reco|gen)
            col_sums = h.sum(axis=1, keepdims=True)
            col_sums[col_sums == 0] = 1
            h_norm = h / col_sums
 
            im = ax.pcolormesh(
                xedges, yedges, h_norm.T,
                cmap="Blues",
                norm=mcolors.PowerNorm(gamma=0.5)
            )
            fig.colorbar(im, ax=ax, label="P(reco|gen)")
 
            # Ідеальна діагональ
            ax.plot([vmin, vmax], [vmin, vmax],
                    "r--", linewidth=1.2, label="ideal")
 
            # Коефіцієнт кореляції
            r = pearson_r(gen, reco)
            ax.text(0.04, 0.93, f"r = {r:.3f}",
                    transform=ax.transAxes,
                    fontsize=10, color="darkred",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
 
            # Bias і resolution як текст
            resid = reco - gen
            if vn == "phitt" or vn == "dphitt":
                resid[resid >  pi] -= 2*pi
                resid[resid < -pi] += 2*pi
            ax.text(0.04, 0.82,
                    f"bias={resid.mean():+.2f}\nσ={resid.std():.2f}",
                    transform=ax.transAxes,
                    fontsize=8, color="navy",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
 
            if row == 0:
                ax.set_title(algo_name, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Reco  {LABELS[vn]}", fontsize=10)
            ax.set_xlabel(f"Gen  {LABELS[vn]}", fontsize=10)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.2)
 
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = os.path.join(outdir, "correlation_reco_vs_gen.pdf")
    out_png = os.path.join(outdir, "correlation_reco_vs_gen.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[I] Збережено: {out_pdf}")
    print(f"[I] Збережено: {out_png}")
    plt.close(fig)
 
 
def load_from_root(root_file, tree_name="ttbarTree", nn_trained=None, channel=None):
    """Завантажує результати LKR, LKRv3 та LKRNN алгоритмів з ROOT файлу.
    nn_trained — ROOT-файл із позначеними тренувальними подіями NN: для LKRNN такі події
    виключаються (мережа їх бачила, інакше кореляція завищена). Алгоритмічні методи —
    на всьому датасеті."""
    tree   = uproot.open(f"{root_file}:{tree_name}")
    avail  = set(tree.keys())
 
    variables = ["mtt", "pttt", "ytt", "phitt"]
    algos     = ["lkr", "lkrv3", "lkrnn"]  # Будуємо для цих трьох
 
    # Завантаження Gen-рівня
    gen_arrays = {}
    for v in variables:
        br = f"{v}_gen"
        if br in avail:
            gen_arrays[v] = tree[br].array(library="np")
        else:
            print(f"[E] Помилка: не знайдено гілку {br} у ROOT файлі")
            return {}, None
 
    N = len(next(iter(gen_arrays.values())))
    targets = np.stack([gen_arrays[v] for v in variables], axis=1)
 
    # held-out маска для LKRnn: True = подія НЕ брала участі в навчанні
    eval_ok = np.ones(N, dtype=bool)
    if nn_trained and os.path.exists(nn_trained):
        t = uproot.open(f"{nn_trained}:nn_trained").arrays(library="np")
        sel = t["channel"] == channel if channel is not None else np.ones(len(t["channel"]), bool)
        ent = t["entry"][sel]
        ent = ent[(ent >= 0) & (ent < N)]
        eval_ok[ent] = False
        print(f"[I] NN held-out: виключено {len(ent)} тренувальних подій")

    # Завантаження результатів реконструкції
    preds_dict = {}
    targets_dict = {}
    for algo in algos:
        cols = []
        found = True
        for v in variables:
            br = f"{v}_{algo}"
            if br in avail:
                cols.append(tree[br].array(library="np"))
            else:
                print(f"[W] Пропущено {algo.upper()}: немає гілки {br}")
                found = False
                break
        if found:
            pred = np.stack(cols, axis=1)
            tgt = targets
            if algo == "lkrnn":       # LKRnn — лише на подіях, яких мережа не бачила
                pred, tgt = pred[eval_ok], targets[eval_ok]
            preds_dict[algo.upper()] = pred
            targets_dict[algo.upper()] = tgt
            print(f"[I] {algo.upper()}: завантажено {len(pred)} подій")

    return preds_dict, targets_dict
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # За замовчуванням беремо ваш основний файл
    parser.add_argument("--root", default="ttbar_output_3.root", help="Шлях до ROOT файлу")
    parser.add_argument("--tree", default="ttbarTree")
    parser.add_argument("--out",  default="plots")
    parser.add_argument("--nn-trained", dest="nn_trained", default="solve_nn_output/nn_trained.root",
                        help="ROOT-файл із тренувальними подіями NN (виключаються для LKRNN)")
    parser.add_argument("--channel", type=int, default=None,
                        help="номер каналу для звірки з nn_trained (за замовч. визначається з імені файлу)")
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print(f"[E] ROOT файл не знайдено: {args.root}")
        exit(1)

    channel = args.channel
    if channel is None:
        m = re.search(r"_(\d+)\.root$", args.root)
        channel = int(m.group(1)) if m else None

    preds_dict, targets = load_from_root(args.root, args.tree, args.nn_trained, channel)
 
    if not preds_dict or targets is None:
        print("[E] Немає даних для побудови графіків")
        exit(1)
 
    plot_correlation_matrix(preds_dict, targets, args.out)
    print("\n[I] Успішно завершено!")

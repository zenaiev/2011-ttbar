"""
correlation.py
──────────────
Будує 2D кореляційні матриці reco vs gen (детекторний рівень) для всіх алгоритмів.
Алгоритмічні методи — на всьому датасеті; LKRNN — лише на подіях, не використаних
у тренуванні NN (nn_split.eval_sample), як у integral_res.py та paper_figs.py.

Запуск:
  python3 correlation.py --channels 1 2 3 --input-pattern 'ttbar_output_det_{ch}.root'
  python3 correlation.py --root ttbar_output_det_3.root
  python3 correlation.py --channels 1 2 3 --input-pattern 'ttbar_output_full_{ch}.root' --split-model gen  # gen–det
"""
 
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import uproot

import nn_split
 
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
 
 
def plot_correlation_matrix(preds_dict, targets, outdir, title_suffix="", name="correlation_reco_vs_gen"):
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
        # targets — словник {алгоритм: gen-масив}: LKRNN має власну вибірку (поза тренуванням NN)
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
    out_pdf = os.path.join(outdir, f"{name}.pdf")
    out_png = os.path.join(outdir, f"{name}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[I] Збережено: {out_pdf}")
    print(f"[I] Збережено: {out_png}")
    plt.close(fig)
 
 
def load_from_root(root_file, tree_name="ttbarTree", channel=None, split_model="det", all_events=False):
    """Завантажує результати LKR, LKRv2, LKRv3 та LKRnn з ROOT-файлу.
    LKRNN обмежується подіями, не використаними у тренуванні моделі split_model
    (nn_split.eval_sample); алгоритмічні методи — усі події. all_events=True — LKRNN теж на всіх."""
    tree   = uproot.open(f"{root_file}:{tree_name}")
    avail  = set(tree.keys())

    variables = ["mtt", "pttt", "ytt", "phitt"]
    algos     = ["lkr", "lkrv2", "lkrv3", "lkrnn"]  # пропускаються, якщо гілок немає

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

    # вибірка подій поза тренуванням NN (застосовується лише до LKRNN)
    if all_events:
        sample = np.ones(N, dtype=bool)
        print(f"[!] {root_file}: LKRNN на УСІХ подіях, разом із тренувальними (лише для порівняння)")
    else:
        if channel is None:
            print(f"[E] {root_file}: не вдалося визначити канал — задайте --channel")
            return {}, None
        sample, info = nn_split.eval_sample(root_file, channel, split_model, treename=tree_name)
        print(f"[I] {root_file}: LKRNN оцінюється на {int(sample.sum())} з {N} подій (поза тренуванням "
              f"{split_model}-моделі: тест {info['test']} + {info['extra']} непридатних)")

    # Завантаження результатів реконструкції
    preds_dict, targets_dict = {}, {}
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
            sel = sample if nn_split.uses_nn(algo) else np.ones(N, dtype=bool)
            preds_dict[algo.upper()] = np.stack(cols, axis=1)[sel]
            targets_dict[algo.upper()] = targets[sel]
            print(f"[I] {algo.upper()}: завантажено {int(sel.sum())} подій")

    return preds_dict, targets_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # За замовчуванням беремо ваш основний файл
    parser.add_argument("--root", default="ttbar_output_3.root", help="Шлях до ROOT файлу")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="об'єднати канали ttbar_output_full_<ch>.root (напр. 1 2 3)")
    parser.add_argument("--name", default="correlation_reco_vs_gen",
                        help="базове ім'я вихідного файлу")
    parser.add_argument("--tree", default="ttbarTree")
    parser.add_argument("--out",  default="plots")
    parser.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_det_{ch}.root",
                        help="шаблон файлів для --channels; за замовч. основні результати, для крос-тесту: ttbar_output_full_{ch}.root з --split-model gen")
    parser.add_argument("--split-model", dest="split_model", choices=["det", "gen"], default="det",
                        help="модель, чиї тренувальні події виключаються (gen — для файлів з krNNGen 1)")
    parser.add_argument("--all-events", dest="all_events", action="store_true",
                        help="LKRNN теж на всіх подіях, разом із тренувальними (лише для порівняння)")
    parser.add_argument("--channel", type=int, default=None,
                        help="номер каналу для --root (за замовч. визначається з імені файлу)")
    args = parser.parse_args()

    if args.channels:
        parts = []
        for ch in args.channels:
            f = args.input_pattern.format(ch=ch)
            if not os.path.exists(f):
                print(f"[W] {f} не знайдено — канал {ch} пропущено")
                continue
            parts.append(load_from_root(f, args.tree, ch, args.split_model, args.all_events))
        parts = [p for p in parts if p[0]]
        if not parts:
            print("[E] Немає вхідних файлів"); exit(1)
        preds_dict = {algo: np.concatenate([p[0][algo] for p in parts]) for algo in parts[0][0]}
        targets = {algo: np.concatenate([p[1][algo] for p in parts]) for algo in parts[0][0]}
        print(f"[I] Об'єднано канали {args.channels}: " + ", ".join(f"{k} {len(v)}" for k, v in targets.items()) + " подій")
    else:
        if not os.path.exists(args.root):
            print(f"[E] ROOT файл не знайдено: {args.root}")
            exit(1)
        channel = args.channel
        if channel is None:
            m = re.search(r"_(\d+)\.root$", args.root)
            channel = int(m.group(1)) if m else None
        preds_dict, targets = load_from_root(args.root, args.tree, channel, args.split_model, args.all_events)

    if not preds_dict or targets is None:
        print("[E] Немає даних для побудови графіків")
        exit(1)
 
    plot_correlation_matrix(preds_dict, targets, args.out, name=args.name)
    print("\n[I] Успішно завершено!")

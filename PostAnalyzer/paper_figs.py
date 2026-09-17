"""
paper_figs.py
─────────────
Рисунки для статті: пара методів (LKR + один інший) на одній фігурі,
3 рядки × 3 колонки — efficiency (upper), resolution (middle), bias (lower)
для M(ttbar), pT(ttbar), y(ttbar).

Уся фізика (бінінг, фільтри, похибки) береться з kinreco_eff_v4, щоб не дублювати реалізацію.
Ефективність усіх методів і bias/роздільна здатність LKR, LKRv2, LKRv3 — на всьому датасеті;
bias і роздільна здатність LKRnn — лише на подіях, не використаних у тренуванні NN
(nn_split.eval_sample). Так само, як у integral_res.py.
Для gen–det (файли з krNNGen 1) додайте --split-model gen.

Рисунки кожного режиму — в окремій папці (ім'я файлу з суфіксом режиму, як у статті):
  <outdir>/det/lkr_lkrnn.pdf, <outdir>/gen/lkr_lkrnn_gen.pdf, <outdir>/gennn/lkr_lkrnn_gennn.pdf

Запуск:
  python3 paper_figs.py                          # усі доступні пари, канали 1+2+3
  python3 paper_figs.py --pairs lkrv2 lkrv3      # лише ці пари
  python3 paper_figs.py --channels 3             # один канал
  python3 paper_figs.py --level gen              # генераторний рівень
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kinreco_eff_v4 as k4
import nn_split

BASE = "lkr"                       # базовий метод, з яким порівнюємо
VARS = ["mtt", "pttt", "ytt"]
NICE = {"lkr": "LKR", "lkrv2": "LKRv2", "lkrv3": "LKRv3", "lkrnn": "LKRnn",
        "fkr": "FKR", "skr": "SKR"}
COLORS = {"lkr": "tab:blue", "lkrv2": "tab:red", "lkrv3": "tab:orange",
          "lkrnn": "tab:green", "fkr": "tab:purple", "skr": "tab:brown"}


def make_figure(a, methods, level, out_pdf, tag="", eff_min=None, height=9.5):
    """Одна фігура: рядки = efficiency/resolution/bias, колонки = змінні."""
    suffix = "" if level == "det" else "_gen"
    n = len(a["mtt_gen"])
    base_all = a["reco_passed_selection"] == 1 if level == "det" else np.ones(n, dtype=bool)

    eff_all = []          # ефективність з усіх панелей -> спільна вісь Y
    fig, axes = plt.subplots(3, len(VARS), figsize=(13, height))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07,
                        wspace=0.28, hspace=0.28)

    for col, var in enumerate(VARS):
        gen = a[f"{var}_gen"]
        span = {"eff": [], "res": [], "bias": []}
        for m in methods:
            if f"{var}_{m}{suffix}" not in a:
                continue
            reco = a[f"{var}_{m}{suffix}"]
            valid = base_all & (a[f"{m}{suffix}"] == 1)          # ефективність — на всьому датасеті
            # bias/роздільна здатність LKRnn — лише на подіях поза тренуванням NN
            res_mask = a["eval_sample"] if nn_split.uses_nn(m) else None
            cx, eff, eff_u, bias, bias_u, res, res_u = k4.calc_bin_metrics(
                reco, gen, valid, base_all, k4.BINS[var], var, res_mask)

            style = dict(marker="o", markersize=4, color=COLORS.get(m), label=NICE.get(m, m.upper()),
                         linewidth=1.2, capsize=2)
            axes[0, col].errorbar(cx, eff,  eff_u,  **style)
            axes[1, col].errorbar(cx, res,  res_u,  **style)
            axes[2, col].errorbar(cx, bias, bias_u, **style)
            span["eff"]  += [v for v in eff  if v == v]
            eff_all      += [v for v in eff  if v == v]
            span["res"]  += [v for v in res  if v == v]
            span["bias"] += [v for v in bias if v == v]

        for row, (key, ylab) in enumerate([("eff", "Efficiency"),
                                           ("res", "Resolution"),
                                           ("bias", "Bias")]):
            ax = axes[row, col]
            ax.set_xlabel(k4.LABELS[var], fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            ax.grid(True, alpha=0.4)
            ax.tick_params(labelsize=9)
            if col == 0:
                ax.legend(fontsize=9)
            # масштаб за значеннями, не за (роздутими у хвостах) похибками
            vals = span[key]
            if key == "eff" and eff_min is not None:
                continue          # задано --eff-min: спільна вісь для ефективності — ставиться нижче
            if vals:
                lo, hi = min(vals), max(vals)
                pad = 0.12 * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1e-6)
                ax.set_ylim(lo - pad, hi + pad)

    # За замовчуванням (як на рисунках статті) вісь ефективності кожної панелі масштабується за її точками.
    # Якщо задано --eff-min, ефективність усіх панелей — на спільній осі [eff_min, 1.005].
    if eff_all and eff_min is not None:
        lo = eff_min
        for c in range(len(VARS)):
            axes[0, c].set_ylim(lo, 1.005)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"[I] {out_pdf}  ({' vs '.join(NICE.get(m, m) for m in methods)}, {level}{tag})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="методи для порівняння з LKR (за замовч. усі наявні)")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--level", choices=["det", "gen"], default="det")
    ap.add_argument("--outdir", default="figs",
                    help="базова папка; рисунки йдуть у <outdir>/det, <outdir>/gen або <outdir>/gennn")
    ap.add_argument("--split-model", dest="split_model", choices=["det", "gen"], default=None,
                    help="модель, чиї тренувальні події виключаються з оцінки (за замовч. = --level; для gen–det: gen)")
    ap.add_argument("--split-file", dest="split_file", default=None,
                    help="dataset_split.root відповідної моделі (за замовч. визначається з --split-model або з файлу nn_apply)")
    ap.add_argument("--all-events", dest="all_events", action="store_true",
                    help="LKRnn теж на всіх подіях, разом із тренувальними (лише для порівняння)")
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_det_{ch}.root",
                    help="шаблон вхідних файлів; за замовч. основні результати (det-модель), для крос-тесту: ttbar_output_full_{ch}.root з --split-model gen")
    ap.add_argument("--per-channel", action="store_true",
                    help="окрема фігура на кожен канал (за замовч. усі канали разом)")
    ap.add_argument("--summary", action="store_true",
                    help="додатково одна фігура з УСІМА методами разом (figs/lkr_all*.pdf)")
    ap.add_argument("--eff-min", dest="eff_min", type=float, default=None,
                    help="спільна вісь ефективності від цього значення до 1.005; за замовч. — власна вісь "
                         "кожної панелі за її точками, як на рисунках статті")
    ap.add_argument("--height", type=float, default=9.5,
                    help="висота фігури в дюймах (ширина 13); за замовч. 9.5 — як на рисунках статті")
    ap.add_argument("--only-summary", action="store_true",
                    help="лише зведена фігура з усіма методами, без попарних")
    args = ap.parse_args()
    # режим -> окрема папка й суфікс імені: det–det, gen–gen, gen–det (крос-тест gen-моделі на файлах з krNNGen 1)
    split_model = args.split_model or args.level
    mode = "gen" if args.level == "gen" else ("gennn" if split_model == "gen" else "det")
    sfx = {"det": "", "gen": "_gen", "gennn": "_gennn"}[mode]
    outdir = os.path.join(args.outdir, mode)

    # читаємо всі можливі методи
    k4.METHODS = ["lkr", "lkrv2", "lkrv3", "lkrnn"]

    loaded, names = [], []
    for ch in args.channels:
        f = args.input_pattern.format(ch=ch)
        if not os.path.exists(f):
            print(f"[W] {f} не знайдено — канал {ch} пропущено")
            continue
        a = k4.load(f)
        if args.all_events:
            a["eval_sample"] = np.ones(len(a["mtt_gen"]), dtype=bool)
        else:
            a["eval_sample"], info = nn_split.eval_sample(f, ch, split_model, args.split_file)
            print(f"[I] {f}: LKRnn оцінюється на {int(a['eval_sample'].sum())} з {info['n']} подій "
                  f"(тест {info['test']} + {info['extra']} непридатних до тренування, f={info['frac']:.4f})")
        loaded.append(a)
        names.append(k4.CHANNEL_NAMES.get(ch, str(ch)))
    if not loaded:
        print("[E] Немає вхідних файлів.")
        return

    datasets = list(zip(names, loaded)) if args.per_channel else \
               [("+".join(names), k4.combine(loaded) if len(loaded) > 1 else loaded[0])]

    for label, a in datasets:
        present = [m for m in ["lkrv2", "lkrv3", "lkrnn"]
                   if f"mtt_{m}{'' if args.level == 'det' else '_gen'}" in a]
        tag_ch = f"_{label}" if args.per_channel else ""

        # зведена фігура: базовий LKR + усі наявні варіанти на одних осях
        if args.summary or args.only_summary:
            out = os.path.join(outdir, f"lkr_all{sfx}{tag_ch}.pdf")
            make_figure(a, [BASE] + present, args.level, out, tag=f", {label}", eff_min=args.eff_min, height=args.height)
        if args.only_summary:
            continue

        # у крос-тесті gen–det від det–det відрізняється лише LKRnn, тож за замовчуванням лише ця пара
        pairs = args.pairs or (["lkrnn"] if mode == "gennn" else present)
        for m in pairs:
            if m not in present:
                print(f"[W] {NICE.get(m, m)} відсутній у даних — пропущено "
                      f"(увімкніть kr_{NICE.get(m, m).upper()} у конфізі та перезапустіть eventReco)")
                continue
            tag = f"_{label}" if args.per_channel else ""
            out = os.path.join(outdir, f"lkr_{m}{sfx}{tag}.pdf")
            make_figure(a, [BASE, m], args.level, out, tag=f", {label}", eff_min=args.eff_min, height=args.height)


if __name__ == "__main__":
    main()

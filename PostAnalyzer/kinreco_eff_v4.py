"""
kinreco_eff_v4.py
──────────────────
Роздільна здатність / bias / ефективність для LKR та LKRv3 на
ttbar_output_full_*.root (нова архітектура eventReco з reco+gen у одному дереві).

Детекторний рівень: reco_passed_selection == 1, гілки *_lkr / *_lkrv3.
Генераторний рівень: усі події, гілки *_lkr_gen / *_lkrv3_gen (окремо
перевіряємо прапорець lkr_gen / lkrv3_gen == 1, кілька % подій без розв'язку).

Запуск:
  python3 kinreco_eff_v4.py                      # канал emu (файл _3)
  python3 kinreco_eff_v4.py --channel 3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot

METHODS = ["lkr", "lkrv3"]
VARS = ["mtt", "pttt", "ytt"]
LABELS = {
    "mtt":  r"$M(t\bar{t})$ [GeV]",
    "pttt": r"$p_T(t\bar{t})$ [GeV]",
    "ytt":  r"$y(t\bar{t})$",
}
BINS = {
    "mtt": np.concatenate((np.linspace(340, 450, 5, endpoint=False),
                           np.logspace(np.log10(450), np.log10(700), 5, endpoint=False),
                           np.logspace(np.log10(700), np.log10(1400), 8))),
    "pttt": np.concatenate((np.linspace(0, 100, 5, endpoint=False),
                            np.logspace(np.log10(100), np.log10(250), 5, endpoint=False),
                            np.logspace(np.log10(250), np.log10(800), 8))),
    "ytt": np.array([-2.4, -2.0, -1.75] + np.linspace(-1.6, 1.6, 16, endpoint=False).tolist() + [1.75, 2.0, 2.4]),
}
COLORS = {"lkr": "tab:blue", "lkrv3": "tab:orange"}


def calc_bin_metrics(reco, gen, valid, denom_mask, bins):
    """Ефективність, bias, roздільна здатність у кожному біні gen.
    valid      — чисельник ефективності (base_valid & прапорець методу);
    denom_mask — знаменник ефективності: det -> reco_passed_selection==1, gen -> усі події.
    Ефективність КР = N(реконструйовано) / N(після препроцесингу).
    Похибки — точно як у kinreco_eff_v3.py: bias_unc = sigma/sqrt(n),
    resolution_unc = sqrt(sqrt(mu4 - sigma^4)/n)."""
    centers, eff, eff_u, bias, bias_u, res, res_u = [], [], [], [], [], [], []
    for i in range(len(bins) - 1):
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        mask_bin = (gen > bins[i]) & (gen < bins[i + 1])
        n_den = (mask_bin & denom_mask).sum()   # знаменник (відібрані / усі)
        mask_ok = mask_bin & valid              # чисельник (реконструйовані)
        n_ok = mask_ok.sum()
        if n_den == 0:
            eff.append(np.nan); eff_u.append(np.nan)
            bias.append(np.nan); bias_u.append(np.nan)
            res.append(np.nan); res_u.append(np.nan)
            continue
        e = n_ok / n_den
        eff.append(e); eff_u.append(np.sqrt(e * (1 - e) / n_den))
        # для bias/roзд. здатності відкидаємо нефінітні значення (напр. NaN у FKR)
        mask_res = mask_ok & np.isfinite(reco)
        n_res = mask_res.sum()
        if n_res < 2:
            bias.append(np.nan); bias_u.append(np.nan)
            res.append(np.nan); res_u.append(np.nan)
            continue
        r = reco[mask_res] - gen[mask_res]
        b = r.mean()
        s = np.sqrt(np.mean((r - b) ** 2))
        bias.append(b); bias_u.append(s / np.sqrt(n_res))
        res.append(s)
        mu4 = np.mean((r - b) ** 4)
        res_u.append(np.sqrt(np.sqrt(max(mu4 - s ** 4, 0.0)) / n_res))
    return (np.array(centers), np.array(eff), np.array(eff_u),
            np.array(bias), np.array(bias_u), np.array(res), np.array(res_u))


def integral_res(reco, gen, valid, denom_mask):
    eff = valid.sum() / denom_mask.sum() if denom_mask.sum() else float("nan")
    m = valid & np.isfinite(reco)
    r = reco[m] - gen[m]
    return np.sqrt(np.mean((r - r.mean()) ** 2)), r.mean(), eff


def load(fname):
    tree = uproot.open(fname)["ttbarTree"]
    keys = ["reco_passed_selection"]
    for v in VARS:
        keys.append(f"{v}_gen")
        for m in METHODS:
            keys += [f"{v}_{m}", f"{v}_{m}_gen"]
    for m in METHODS:
        keys += [m, f"{m}_gen"]
    return tree.arrays(keys, library="np")


def make_plot(a, level, out_prefix):
    """level: 'det' (reco_passed_selection==1, *_lkr) or 'gen' (усі події, *_lkr_gen)."""
    suffix = "" if level == "det" else "_gen"
    if level == "det":
        base_valid = a["reco_passed_selection"] == 1
    else:
        base_valid = np.ones(len(a["mtt_gen"]), dtype=bool)

    fig_res, axs_res = plt.subplots(1, 3, figsize=(15, 5))
    fig_bias, axs_bias = plt.subplots(1, 3, figsize=(15, 5))
    fig_eff, axs_eff = plt.subplots(1, 3, figsize=(15, 5))
    for fig in (fig_res, fig_bias, fig_eff):
        fig.subplots_adjust(0.07, 0.12, 0.98, 0.86, wspace=0.3)

    print(f"\n=== {level.upper()} рівень ({'reco_passed_selection==1' if level=='det' else 'усі події'}) ===")
    print(f"{'Змінна':<8} | {'Метод':<7} | {'Еф. (%)':<8} | {'Bias':<10} | {'Roзд.':<10}")
    print("-" * 55)

    for k, var in enumerate(VARS):
        gen = a[f"{var}_gen"]
        all_res, all_bias = [], []
        for m in METHODS:
            reco = a[f"{var}_{m}{suffix}"]
            passed = a[f"{m}{suffix}"] == 1  # прапорець успішної реконструкції методу
            valid = base_valid & passed
            cx, eff, eff_u, bias, bias_u, res, res_u = calc_bin_metrics(reco, gen, valid, base_valid, BINS[var])
            all_res += [r for r in res if r == r]
            all_bias += [b for b in bias if b == b]

            axs_res[k].errorbar(cx, res, res_u, marker='o', markersize=2, color=COLORS[m], label=m.upper())
            axs_bias[k].errorbar(cx, bias, bias_u, marker='o', markersize=2, color=COLORS[m], label=m.upper())
            axs_eff[k].errorbar(cx, eff, eff_u, marker='o', markersize=2, color=COLORS[m], label=m.upper())

            sigma, b, eff_int = integral_res(reco, gen, valid, base_valid)
            print(f"{var:<8} | {m.upper():<7} | {100*eff_int:<8.2f} | {b:<10.4f} | {sigma:<10.4f}")

        # масштаб осі Y за самими значеннями (не за роздутими похибками)
        if all_res:
            axs_res[k].set_ylim(0, max(all_res) * 1.25)
        if all_bias:
            lo, hi = min(all_bias), max(all_bias)
            pad = 0.2 * (hi - lo) if hi > lo else max(abs(hi), 1.0)
            axs_bias[k].set_ylim(lo - pad, hi + pad)

        for ax, fig, ylabel in [(axs_res[k], fig_res, "Resolution"),
                                 (axs_bias[k], fig_bias, "Bias"),
                                 (axs_eff[k], fig_eff, "Efficiency")]:
            ax.set_xlabel(LABELS[var]); ax.set_ylabel(ylabel)
            ax.grid(True); ax.legend()
    fig_res.suptitle(f"Роздільна здатність LKR vs LKRv3 ({level}-рівень)")
    fig_res.savefig(f"plots/{out_prefix}_resolution_{level}.png", dpi=150)

    fig_bias.suptitle(f"Bias LKR vs LKRv3 ({level}-рівень)")
    fig_bias.savefig(f"plots/{out_prefix}_bias_{level}.png", dpi=150)

    fig_eff.suptitle(f"Ефективність LKR vs LKRv3 ({level}-рівень)")
    fig_eff.savefig(f"plots/{out_prefix}_efficiency_{level}.png", dpi=150)

    print(f"[I] Збережено: plots/{out_prefix}_{{resolution,bias,efficiency}}_{level}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=3, help="1=ee, 2=mumu, 3=emu")
    args = ap.parse_args()
    fname = f"ttbar_output_full_{args.channel}.root"

    a = load(fname)
    print(f"[I] {fname}: {len(a['mtt_gen'])} подій")

    make_plot(a, "det", "kr_v4")
    make_plot(a, "gen", "kr_v4")


if __name__ == "__main__":
    main()

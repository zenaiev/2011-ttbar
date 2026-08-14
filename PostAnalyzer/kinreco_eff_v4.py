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
import os
import numpy as np
import matplotlib.pyplot as plt
import uproot

METHODS = ["lkr", "lkrv3", "lkrnn"]
VARS = ["mtt", "pttt", "ytt"]
CHANNEL_NAMES = {1: "ee", 2: "mumu", 3: "emu"}
# верхня межа реконструйованого значення — як у kinreco_eff_v3.py (reco < 9999),
# відсікає чисельні артефакти LKR (напр. mtt ~104 ТеВ) і дає ідентичні v3 числа
PHYS_MAX = 9999.0
PHYS_RANGE = {"mtt": (-999.0, PHYS_MAX), "pttt": (-999.0, PHYS_MAX), "ytt": (-999.0, PHYS_MAX)}
LABELS = {
    "mtt":  r"$M(t\bar{t})$ [GeV]",
    "pttt": r"$p_T(t\bar{t})$ [GeV]",
    "ytt":  r"$y(t\bar{t})$",
}
# бінінг — як у kinreco_eff_v3.py (січнева презентація)
BINS = {
    "mtt": np.concatenate((np.linspace(340, 450, 5, endpoint=False),
                           np.logspace(np.log10(450), np.log10(700), 5, endpoint=False),
                           np.logspace(np.log10(700), np.log10(1400), 8))),
    "pttt": np.concatenate((np.linspace(0, 100, 5, endpoint=False),
                            np.logspace(np.log10(100), np.log10(250), 5, endpoint=False),
                            np.logspace(np.log10(250), np.log10(800), 8))),
    "ytt": np.array([-2.4, -2.0, -1.75] + np.linspace(-1.6, 1.6, 16, endpoint=False).tolist() + [1.75, 2.0, 2.4]),
}
COLORS = {"lkr": "tab:blue", "lkrv3": "tab:orange", "lkrnn": "tab:green"}


def physical(reco, variable):
    """Маска фізично допустимих реконструйованих значень (фінітні + у діапазоні PHYS_RANGE)."""
    m = np.isfinite(reco)
    if variable in PHYS_RANGE:
        lo, hi = PHYS_RANGE[variable]
        m &= (reco > lo) & (reco < hi)
    return m


def calc_bin_metrics(reco, gen, valid, denom_mask, bins, variable, res_mask=None):
    """Ефективність, bias, roздільна здатність у кожному біні gen.
    valid      — чисельник ефективності (base_valid & прапорець методу);
    denom_mask — знаменник ефективності: det -> reco_passed_selection==1, gen -> усі події.
    res_mask   — (опційно) додаткове обмеження ЛИШЕ для bias/roзд. здатності, напр. held-out
                 події для LKRnn. Ефективність рахується на ПОВНОМУ наборі, бо це властивість
                 алгоритму (чи є розв'язок), а не моделі; інакше порівняння методів нечесне,
                 адже тренувальна вибірка складалася лише з успішних подій.
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
        # для bias/roзд. здатності відкидаємо нефінітні та нефізичні значення
        # (напр. NaN у FKR або mtt ~104 ТеВ у LKR — див. PHYS_RANGE)
        mask_res = mask_ok & physical(reco, variable)
        if res_mask is not None:
            mask_res = mask_res & res_mask
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


def integral_res(reco, gen, valid, denom_mask, variable, res_mask=None):
    """Повертає (roзд. здатність, bias, ефективність, N використаних подій).
    res_mask — додаткове обмеження лише для bias/roзд. здатності (held-out для LKRnn);
    ефективність завжди на повному наборі (властивість алгоритму, не моделі)."""
    eff = valid.sum() / denom_mask.sum() if denom_mask.sum() else float("nan")
    m = valid & physical(reco, variable)
    if res_mask is not None:
        m = m & res_mask
    n_used = int(m.sum())
    if n_used == 0:
        return float("nan"), float("nan"), eff, 0
    r = reco[m] - gen[m]
    return np.sqrt(np.mean((r - r.mean()) ** 2)), r.mean(), eff, n_used


def load(fname, channel=None, nn_trained=None):
    """Читає дерево. Якщо задано nn_trained (шлях до nn_trained.root від тренування),
    додає маску 'nn_eval_ok': False для подій, які БРАЛИ УЧАСТЬ у навчанні NN.
    Ці події виключаються ЛИШЕ з метрик LKRnn (алгоритмічні методи — на всьому датасеті)."""
    tree = uproot.open(fname)["ttbarTree"]
    keys = ["reco_passed_selection"]
    for v in VARS:
        keys.append(f"{v}_gen")
        for m in METHODS:
            keys += [f"{v}_{m}", f"{v}_{m}_gen"]
    for m in METHODS:
        keys += [m, f"{m}_gen"]
    keys = [k for k in dict.fromkeys(keys) if k in tree.keys()]
    a = tree.arrays(keys, library="np")

    n = tree.num_entries
    eval_ok = np.ones(n, dtype=bool)
    if nn_trained and os.path.exists(nn_trained):
        t = uproot.open(f"{nn_trained}:nn_trained").arrays(library="np")
        sel = t["channel"] == channel if channel is not None else np.ones(len(t["channel"]), bool)
        ent = t["entry"][sel]
        ent = ent[(ent >= 0) & (ent < n)]
        eval_ok[ent] = False
        print(f"    [NN] виключено {len(ent)} тренувальних подій -> для LKRnn лишається {int(eval_ok.sum())}")
    a["nn_eval_ok"] = eval_ok
    return a


def make_plot(a, level, out_prefix, info=""):
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

    n_den = int(base_valid.sum())
    counts = []  # (var, method, n_all, n_plot) — друкуються окремим блоком у кінці
    print(f"\n=== {level.upper()} рівень ({'reco_passed_selection==1' if level=='det' else 'усі події'}) ===")
    print(f"{'Змінна':<8} | {'Метод':<7} | {'Еф. (%)':<8} | {'Bias':<10} | {'Roзд.':<10}")
    print("-" * 55)

    for k, var in enumerate(VARS):
        gen = a[f"{var}_gen"]
        all_res, all_bias = [], []
        # діапазон бінінгу: події поза ним на графік не потрапляють
        in_range = (gen > BINS[var][0]) & (gen < BINS[var][-1])
        for m in METHODS:
            if f"{var}_{m}{suffix}" not in a:
                continue                      # метод відсутній у цьому файлі
            reco = a[f"{var}_{m}{suffix}"]
            passed = a[f"{m}{suffix}"] == 1  # прапорець успішної реконструкції методу
            valid = base_valid & passed      # ефективність — на ПОВНОМУ наборі (усі методи)
            # bias/roзд. здатність для LKRnn — лише на held-out подіях (мережа їх не бачила).
            # Ефективність держати на повному наборі критично: тренувальна вибірка складалася
            # тільки з успішних подій, тож виключення її спотворило б знаменник.
            res_mask = a["nn_eval_ok"] if m == "lkrnn" else None
            cx, eff, eff_u, bias, bias_u, res, res_u = calc_bin_metrics(
                reco, gen, valid, base_valid, BINS[var], var, res_mask)
            all_res += [r for r in res if r == r]
            all_bias += [b for b in bias if b == b]

            axs_res[k].errorbar(cx, res, res_u, marker='o', markersize=2, color=COLORS[m], label=m.upper())
            axs_bias[k].errorbar(cx, bias, bias_u, marker='o', markersize=2, color=COLORS[m], label=m.upper())
            axs_eff[k].errorbar(cx, eff, eff_u, marker='o', markersize=2, color=COLORS[m], label=m.upper())

            sigma, b, eff_int, n_all = integral_res(reco, gen, valid, base_valid, var, res_mask)
            # скільки подій реально лягло на графіки (валідні, фізичні, у діапазоні бінінгу)
            n_plot = int((valid & in_range & physical(reco, var)).sum())
            if res_mask is not None:
                n_plot = int((valid & in_range & physical(reco, var) & res_mask).sum())
            counts.append((var, m, n_all, n_plot))
            print(f"{var:<8} | {m.upper():<7} | {100*eff_int:<8.2f} | {b:<10.4f} | {sigma:<10.4f}")

        # масштаб осі Y — щільно навколо самих значень (як у січневій презентації):
        # вісь НЕ прив'язана до нуля, інакше різниця LKR/LKRv3 візуально стискається.
        # Межі рахуємо за точками, а не за (роздутими у хвостах) похибками.
        for ax, vals in ((axs_res[k], all_res), (axs_bias[k], all_bias)):
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)

        for ax, fig, ylabel in [(axs_res[k], fig_res, "Resolution"),
                                 (axs_bias[k], fig_bias, "Bias"),
                                 (axs_eff[k], fig_eff, "Efficiency")]:
            ax.set_xlabel(LABELS[var]); ax.set_ylabel(ylabel)
            ax.grid(True); ax.legend()
    methods_str = " vs ".join(m.upper() for m in METHODS)
    lvl = {"det": "detector level", "gen": "generator level"}.get(level, level)
    tail = f" ({lvl}){info}"      # info: e.g. ", emu channel, NN model: gen-trained"
    for fig, metric, name in [(fig_res, "Resolution", "resolution"),
                              (fig_bias, "Bias", "bias"),
                              (fig_eff, "Efficiency", "efficiency")]:
        fig.suptitle(f"{metric}: {methods_str}{tail}")
        fig.savefig(f"plots/{out_prefix}_{name}_{level}.png", dpi=150)

    print(f"[I] Збережено: plots/{out_prefix}_{{resolution,bias,efficiency}}_{level}.png")
    return {"level": level, "n_den": n_den, "counts": counts}


def combine(arrays_list):
    """Об'єднує масиви кількох каналів в один набір (сумарна статистика)."""
    return {k: np.concatenate([a[k] for a in arrays_list]) for k in arrays_list[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", "--channel", dest="channels", type=int, nargs="+", default=[1, 2, 3],
                    help="канали: 1=ee, 2=mumu, 3=emu (за замовчуванням усі три)")
    ap.add_argument("--combine", action="store_true",
                    help="додатково побудувати сумарний результат по всіх заданих каналах")
    ap.add_argument("--model", default="",
                    help="підпис моделі NN на графіках, напр. 'detector-trained' або 'gen-trained'")
    ap.add_argument("--nn-trained", dest="nn_trained", default="solve_nn_output/nn_trained.root",
                    help="ROOT-файл із позначеними тренувальними подіями NN (виключаються з метрик LKRnn)")
    args = ap.parse_args()
    model_note = f", NN model: {args.model}" if args.model else ""

    loaded, stats = [], []   # stats: (мітка каналу, результат make_plot) для зведення в кінці
    for ch in args.channels:
        fname = f"ttbar_output_full_{ch}.root"
        if not os.path.exists(fname):
            print(f"[W] {fname} не знайдено — канал {ch} пропущено")
            continue
        a = load(fname, channel=ch, nn_trained=args.nn_trained)
        label = f"c{ch} ({CHANNEL_NAMES.get(ch, '?')})"
        info = f", {CHANNEL_NAMES.get(ch, '?')} channel{model_note}"
        print(f"\n[I] Канал {ch} ({CHANNEL_NAMES.get(ch, '?')}): {fname}, {len(a['mtt_gen'])} подій")
        stats.append((label, make_plot(a, "det", f"kr_v4_c{ch}", info)))
        stats.append((label, make_plot(a, "gen", f"kr_v4_c{ch}", info)))
        loaded.append(a)

    if not loaded:
        print("[E] Жодного вхідного файлу не знайдено.")
        return

    if args.combine and len(loaded) > 1:
        a = combine(loaded)
        chs = "+".join(CHANNEL_NAMES.get(c, str(c)) for c in args.channels)
        info = f", channels {chs}{model_note}"
        print(f"\n[I] Сумарно ({chs}): {len(a['mtt_gen'])} подій")
        stats.append((f"comb ({chs})", make_plot(a, "det", "kr_v4_comb", info)))
        stats.append((f"comb ({chs})", make_plot(a, "gen", "kr_v4_comb", info)))

    print_counts(stats)


def print_counts(stats):
    """Зведення кількості подій — окремим блоком у кінці, щоб не заважало копіювати таблиці."""
    print("\n" + "=" * 78)
    print("КІЛЬКІСТЬ ПОДІЙ (по каналах)")
    print("=" * 78)
    print(f"{'Канал':<14} | {'Рівень':<6} | {'Змінна':<7} | {'Метод':<7} | "
          f"{'N (усі)':<9} | {'N (графік)':<10} | {'Знаменник':<9}")
    print("-" * 78)
    for label, s in stats:
        for var, m, n_all, n_plot in s["counts"]:
            print(f"{label:<14} | {s['level']:<6} | {var:<7} | {m.upper():<7} | "
                  f"{n_all:<9} | {n_plot:<10} | {s['n_den']:<9}")
        print("-" * 78)

    # сумарно по каналах (записи 'comb' не додаємо — це вже об'єднаний набір, було б подвійне рахування)
    per_channel = [(lbl, s) for lbl, s in stats if not lbl.startswith("comb")]
    if len(set(lbl for lbl, _ in per_channel)) < 2:
        return
    totals, den = {}, {}
    for _, s in per_channel:
        den[s["level"]] = den.get(s["level"], 0) + s["n_den"]
        for var, m, n_all, n_plot in s["counts"]:
            key = (s["level"], var, m)
            a0, p0 = totals.get(key, (0, 0))
            totals[key] = (a0 + n_all, p0 + n_plot)

    chans = ", ".join(dict.fromkeys(lbl for lbl, _ in per_channel))
    print("\n" + "=" * 78)
    print(f"КІЛЬКІСТЬ ПОДІЙ (сумарно: {chans})")
    print("=" * 78)
    print(f"{'Рівень':<6} | {'Змінна':<7} | {'Метод':<7} | "
          f"{'N (усі)':<9} | {'N (графік)':<10} | {'Знаменник':<9}")
    print("-" * 78)
    for level in ("det", "gen"):
        for var in VARS:
            for m in METHODS:
                if (level, var, m) not in totals:
                    continue
                n_all, n_plot = totals[(level, var, m)]
                print(f"{level:<6} | {var:<7} | {m.upper():<7} | "
                      f"{n_all:<9} | {n_plot:<10} | {den[level]:<9}")
        print("-" * 78)


if __name__ == "__main__":
    main()

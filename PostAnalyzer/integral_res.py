"""
integral_res.py
────────────────
Інтегральні метрики (ефективність, bias, roздільна здатність) для методів
кінематичної реконструкції на ttbar_output_full_*.root (нова архітектура
eventReco з reco+gen у одному дереві).

Рахує ОБИДВА рівні:
  det — детекторний: reco_passed_selection == 1, гілки *_lkr / *_lkrv3 / *_fkr
  gen — генераторний: усі події, гілки *_lkr_gen / *_lkrv3_gen / *_fkr_gen

Для кожного методу перевіряється власний прапорець успішної реконструкції
(lkr / lkrv3 / fkr, або з суфіксом _gen) — кілька % подій без розв'язку
відкидаються з розрахунку bias/roздільної здатності.

Запуск:
  python3 integral_res.py                      # канал emu (файл _3)
  python3 integral_res.py --channel 3
  python3 integral_res.py ttbar_output_full_3.root   # або явний шлях до файлу
"""

import argparse
import os
import numpy as np
import uproot

VARS = ["mtt", "pttt", "ytt", "phitt", "dphitt"]
CHANNEL_NAMES = {1: "ee", 2: "mumu", 3: "emu"}
ANGULAR = {"phitt", "dphitt"}  # періодичні змінні: залишок згортається у [-pi, pi]
# верхня межа реконструйованого значення — як у kinreco_eff_v3.py (reco < 9999),
# відсікає чисельні артефакти LKR (напр. mtt ~104 ТеВ) і дає ідентичні v3 числа
PHYS_MAX = 9999.0
PHYS_RANGE = {"mtt": (-999.0, PHYS_MAX), "pttt": (-999.0, PHYS_MAX), "ytt": (-999.0, PHYS_MAX)}

# бінінг для per-bin виводу — як у kinreco_eff_v3/v4
_pi = np.pi
BINS = {
    "mtt": np.concatenate((np.linspace(340, 450, 5, endpoint=False),
                           np.logspace(np.log10(450), np.log10(700), 5, endpoint=False),
                           np.logspace(np.log10(700), np.log10(1400), 8))),
    "pttt": np.concatenate((np.linspace(0, 100, 5, endpoint=False),
                            np.logspace(np.log10(100), np.log10(250), 5, endpoint=False),
                            np.logspace(np.log10(250), np.log10(800), 8))),
    "ytt": np.array([-2.4, -2.0, -1.75] + np.linspace(-1.6, 1.6, 16, endpoint=False).tolist() + [1.75, 2.0, 2.4]),
    "phitt": np.linspace(-_pi, _pi, 9),
    "dphitt": np.linspace(-_pi, _pi, 9),
}


def physical(reco, variable):
    """Маска фізично допустимих значень (фінітні + у діапазоні PHYS_RANGE)."""
    m = np.isfinite(reco)
    if variable in PHYS_RANGE:
        lo, hi = PHYS_RANGE[variable]
        m &= (reco > lo) & (reco < hi)
    return m


def detect_methods(tree):
    """Автовизначення методів реконструкції, увімкнених у файлі (fkr/lkr/lkrv3/…)."""
    keys = set(tree.keys())
    candidates = ["lkr", "lkrv2", "lkrv3", "lkrnn", "fkr", "skr"]
    return [m for m in candidates if m in keys]


def wrap_angle(r):
    r = r.copy()
    r[r > np.pi] -= 2 * np.pi
    r[r < -np.pi] += 2 * np.pi
    return r


def calc_metrics(reco, gen, valid, variable):
    # відкидаємо нефінітні (напр. NaN у FKR) та нефізичні (напр. mtt ~104 ТеВ у LKR)
    # значення з розрахунку bias/roзд. здатності — див. PHYS_RANGE
    valid = valid & physical(reco, variable)
    n_ok = valid.sum()
    if n_ok == 0:
        return float("nan"), float("nan"), 0
    residual = reco[valid] - gen[valid]
    if variable in ANGULAR:
        residual = wrap_angle(residual)
    return residual.mean(), residual.std(), n_ok


def binned_metrics(reco, gen, base_valid, passed, variable, bins):
    """Ефективність (%) та roздільна здатність у кожному біні gen.
    eff = N(reco успішна) / N(знаменника) у біні; res = std залишку (нефізичні відкинуто."""
    effs, ress, ndens = [], [], []
    valid = base_valid & passed
    phys = physical(reco, variable)
    for i in range(len(bins) - 1):
        mask_bin = (gen > bins[i]) & (gen < bins[i + 1])
        n_den = int((mask_bin & base_valid).sum())
        n_num = int((mask_bin & valid).sum())
        ndens.append(n_den)
        effs.append(100.0 * n_num / n_den if n_den else float("nan"))
        mres = mask_bin & valid & phys
        if int(mres.sum()) >= 2:
            r = reco[mres] - gen[mres]
            if variable in ANGULAR:
                r = wrap_angle(r)
            ress.append(np.sqrt(np.mean((r - r.mean()) ** 2)))
        else:
            ress.append(float("nan"))
    return effs, ress, ndens


def print_binned_level(a, level, methods):
    """Per-bin таблиці (ефективність та roзд. здатність у кожному біні) — окремий вивід."""
    suffix = "" if level == "det" else "_gen"
    n_total = len(a["mtt_gen"])
    base_valid = a["reco_passed_selection"] == 1 if level == "det" else np.ones(n_total, dtype=bool)

    print(f"\n{'#' * 100}")
    print(f"# {level.upper()} рівень — ПО БІНАХ  (eff = N(reco)/N(знаменника) у біні, res = roзд. здатність)")
    print("#" * 100)
    for variable in VARS:
        bins = BINS[variable]
        gen_arr = a[f"{variable}_gen"]
        per = {m: binned_metrics(a[f"{variable}_{m}{suffix}"], gen_arr, base_valid,
                                 a[f"{m}{suffix}"] == 1, variable, bins) for m in methods}
        hdr = f"{'бін (центр)':<13} | {'N_bin':<8}"
        for m in methods:
            hdr += f" | {m.upper() + ' eff%':<9} | {m.upper() + ' res':<10}"
        print(f"\n--- {variable} ---")
        print(hdr)
        print("-" * len(hdr))
        centers = 0.5 * (bins[:-1] + bins[1:])
        for bi in range(len(centers)):
            n_den = per[methods[0]][2][bi]
            row = f"{centers[bi]:<13.3f} | {n_den:<8}"
            for m in methods:
                effs, ress, _ = per[m]
                row += f" | {effs[bi]:<9.2f} | {ress[bi]:<10.4f}"
            print(row)


def print_level(a, level, methods, n_total):
    suffix = "" if level == "det" else "_gen"
    base_valid = a["reco_passed_selection"] == 1 if level == "det" else np.ones(n_total, dtype=bool)
    # знаменник ефективності кінематичної реконструкції:
    #   det — події, що пройшли детекторний відбір (reco_passed_selection==1)
    #   gen — усі події
    # (так само, як на січневій презентації: eff = N(reco успішна) / N(відібрані))
    n_denom = int(base_valid.sum())
    counts = []  # (змінна, метод, N) — друкуються окремим блоком у кінці

    sep = "=" * 122
    print(f"\n{sep}")
    print(f"{level.upper()} рівень ({'reco_passed_selection==1' if level == 'det' else 'усі події'}), "
          f"знаменник ефективності N = {n_denom}")
    print(sep)
    print(f"{'Змінна':<8} | {'Метод':<7} | {'Еф. (%)':<10} | {'Bias':<10} | {'Resolution':<12} | {'vs LKR':<10}")
    print(sep)

    for variable in VARS:
        gen_arr = a[f"{variable}_gen"]
        res_by_method = {}
        for m in methods:
            reco_arr = a[f"{variable}_{m}{suffix}"]
            passed = a[f"{m}{suffix}"] == 1
            valid = base_valid & passed
            # ефективність = частка успішно реконструйованих серед знаменника (за прапорцем методу)
            eff = 100.0 * int(valid.sum()) / n_denom if n_denom else float("nan")
            # для bias/roзд.здатності додатково відкидаємо нефінітні значення (див. calc_metrics)
            bias, res, n_ok = calc_metrics(reco_arr, gen_arr, valid, variable)
            res_by_method[m] = res
            counts.append((variable, m, n_ok))
            print(f"{variable:<8} | {m.upper():<7} | {eff:<10.2f} | {bias:<10.4f} | {res:<12.4f}", end="")
            if m != "lkr" and "lkr" in res_by_method and not np.isnan(res_by_method["lkr"]) and res_by_method["lkr"] > 0 and not np.isnan(res):
                imp = 100 * (res_by_method["lkr"] - res) / res_by_method["lkr"]
                print(f" | {imp:+.2f}%")
            else:
                print(f" | {'---':<10}")
        print("-" * 122)
    return {"level": level, "n_den": n_denom, "counts": counts}


def calculate_integral_metrics(filename, methods=None):
    tree = uproot.open(filename)["ttbarTree"]
    methods = methods or detect_methods(tree)
    if "lkr" not in methods:
        print("[Попередження] LKR відсутній у файлі — колонка 'vs LKR' буде порожня")

    keys = ["reco_passed_selection"]
    for v in VARS:
        keys.append(f"{v}_gen")
        for m in methods:
            keys += [f"{v}_{m}", f"{v}_{m}_gen"]
    for m in methods:
        keys += [m, f"{m}_gen"]
    a = tree.arrays(keys, library="np")

    n_total = len(a["mtt_gen"])
    print(f"[I] {filename}: {n_total} подій, методи: {', '.join(m.upper() for m in methods)}")

    # 1) інтегральні величини по всьому діапазону (лише детекторний рівень)
    print(f"\n{'*' * 60}\n*  ІНТЕГРАЛЬНІ (весь діапазон)\n{'*' * 60}")
    stats = [print_level(a, "det", methods, n_total)]

    # 2) окремий per-bin вивід (роздільна здатність та ефективність по бінах)
    print(f"\n{'*' * 60}\n*  ПО БІНАХ\n{'*' * 60}")
    print_binned_level(a, "det", methods)

    return stats


def print_counts(all_stats):
    """Зведення кількості подій — окремим блоком у кінці, щоб не заважало копіювати таблиці."""
    print("\n" + "=" * 70)
    print("КІЛЬКІСТЬ ПОДІЙ (по каналах)")
    print("=" * 70)
    print(f"{'Канал':<12} | {'Рівень':<6} | {'Змінна':<7} | {'Метод':<7} | {'N':<9} | {'Знаменник':<9}")
    print("-" * 70)
    for label, stats in all_stats:
        for s in stats:
            for var, m, n_ok in s["counts"]:
                print(f"{label:<12} | {s['level']:<6} | {var:<7} | {m.upper():<7} | "
                      f"{n_ok:<9} | {s['n_den']:<9}")
            print("-" * 70)

    if len(all_stats) < 2:
        return
    totals, den, methods_seen = {}, {}, []
    for _, stats in all_stats:
        for s in stats:
            den[s["level"]] = den.get(s["level"], 0) + s["n_den"]
            for var, m, n_ok in s["counts"]:
                totals[(s["level"], var, m)] = totals.get((s["level"], var, m), 0) + n_ok
                if m not in methods_seen:
                    methods_seen.append(m)

    chans = ", ".join(lbl for lbl, _ in all_stats)
    print("\n" + "=" * 70)
    print(f"КІЛЬКІСТЬ ПОДІЙ (сумарно: {chans})")
    print("=" * 70)
    print(f"{'Рівень':<6} | {'Змінна':<7} | {'Метод':<7} | {'N':<9} | {'Знаменник':<9}")
    print("-" * 70)
    for level in ("det", "gen"):
        for var in VARS:
            for m in methods_seen:
                if (level, var, m) in totals:
                    print(f"{level:<6} | {var:<7} | {m.upper():<7} | "
                          f"{totals[(level, var, m)]:<9} | {den[level]:<9}")
        print("-" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=None, help="явний шлях до ROOT-файлу (перекриває --channels)")
    ap.add_argument("--channels", "--channel", dest="channels", type=int, nargs="+", default=[1, 2, 3],
                    help="канали: 1=ee, 2=mumu, 3=emu (за замовчуванням усі три)")
    args = ap.parse_args()

    all_stats = []
    if args.file:
        all_stats.append((os.path.basename(args.file), calculate_integral_metrics(args.file)))
    else:
        for ch in args.channels:
            fname = f"ttbar_output_full_{ch}.root"
            if not os.path.exists(fname):
                print(f"[W] {fname} не знайдено — канал {ch} пропущено")
                continue
            print(f"\n{'#' * 122}\n# КАНАЛ {ch} ({CHANNEL_NAMES.get(ch, '?')})\n{'#' * 122}")
            label = f"c{ch} ({CHANNEL_NAMES.get(ch, '?')})"
            all_stats.append((label, calculate_integral_metrics(fname)))
        if not all_stats:
            print("[E] Жодного вхідного файлу не знайдено.")

    if all_stats:
        print_counts(all_stats)

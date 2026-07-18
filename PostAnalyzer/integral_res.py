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
import numpy as np
import uproot

VARS = ["mtt", "pttt", "ytt", "phitt", "dphitt"]
ANGULAR = {"phitt", "dphitt"}  # періодичні змінні: залишок згортається у [-pi, pi]


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
    # деякі методи (напр. FKR) інколи лишають NaN у реконструйованому значенні
    # навіть при passed==1 — відкидаємо такі події з розрахунку bias/roзд. здатності
    valid = valid & np.isfinite(reco)
    n_ok = valid.sum()
    if n_ok == 0:
        return float("nan"), float("nan"), 0
    residual = reco[valid] - gen[valid]
    if variable in ANGULAR:
        residual = wrap_angle(residual)
    return residual.mean(), residual.std(), n_ok


def print_level(a, level, methods, n_total):
    suffix = "" if level == "det" else "_gen"
    base_valid = a["reco_passed_selection"] == 1 if level == "det" else np.ones(n_total, dtype=bool)
    # знаменник ефективності кінематичної реконструкції:
    #   det — події, що пройшли детекторний відбір (reco_passed_selection==1)
    #   gen — усі події
    # (так само, як на січневій презентації: eff = N(reco успішна) / N(відібрані))
    n_denom = int(base_valid.sum())

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
            bias, res, _ = calc_metrics(reco_arr, gen_arr, valid, variable)
            res_by_method[m] = res
            print(f"{variable:<8} | {m.upper():<7} | {eff:<10.2f} | {bias:<10.4f} | {res:<12.4f}", end="")
            if m != "lkr" and "lkr" in res_by_method and not np.isnan(res_by_method["lkr"]) and res_by_method["lkr"] > 0 and not np.isnan(res):
                imp = 100 * (res_by_method["lkr"] - res) / res_by_method["lkr"]
                print(f" | {imp:+.2f}%")
            else:
                print(f" | {'---':<10}")
        print("-" * 122)


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

    print_level(a, "det", methods, n_total)
    print_level(a, "gen", methods, n_total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=None, help="явний шлях до ROOT-файлу")
    ap.add_argument("--channel", type=int, default=3, help="1=ee, 2=mumu, 3=emu (якщо file не задано)")
    args = ap.parse_args()
    fname = args.file or f"ttbar_output_full_{args.channel}.root"
    calculate_integral_metrics(fname)

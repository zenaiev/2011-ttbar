"""
integral_res.py
────────────────
Ефективність, bias і роздільна здатність методів кінематичної реконструкції
(LKR, LKRv2, LKRv3, LKRnn, ...) СУМАРНО по всіх каналах: інтегрально та по бінах.

Вибірки:
  * алгоритмічні методи (LKR, LKRv2, LKRv3) — на ВСЬОМУ датасеті;
  * bias і роздільна здатність LKRnn — лише на подіях, які не брали участі в тренуванні мережі
    (nn_split.eval_sample): випадкова ~15% частина повного датасету, тож порівняння коректне;
  * ефективність LKRnn — на ВСЬОМУ датасеті: прапорець lkrnn збігається з lkrv3 в усіх подіях,
    тобто від ваг мережі не залежить.

Інтегральний bias — середнє reco − gen; |Bias| — середнє |reco − gen| по подіях (зсуви різних знаків
не гасять один одного).

Режими (det–det / gen–gen / gen–det):
  python3 integral_res.py --input-pattern 'ttbar_output_det_{ch}.root'                      # det–det
  python3 integral_res.py --input-pattern 'ttbar_output_det_{ch}.root' --level gen          # gen–gen
  python3 integral_res.py --input-pattern 'ttbar_output_full_{ch}.root' --split-model gen   # gen–det (файли з krNNGen 1)
  ... --all-events    # LKRnn теж на всіх подіях, разом із тренувальними (лише для порівняння)
"""

import argparse
import os
import numpy as np
import uproot

import nn_split

VARS = ["mtt", "pttt", "ytt", "phitt", "dphitt"]
CHANNEL_NAMES = {1: "ee", 2: "mumu", 3: "emu"}
ANGULAR = {"phitt", "dphitt"}  # періодичні змінні: залишок згортається у [-pi, pi]
# верхня межа реконструйованого значення — як у kinreco_eff_v3.py (reco < 9999),
# відсікає чисельні артефакти LKR (напр. mtt ~104 ТеВ)
PHYS_MAX = 9999.0
PHYS_RANGE = {"mtt": (-999.0, PHYS_MAX), "pttt": (-999.0, PHYS_MAX), "ytt": (-999.0, PHYS_MAX)}

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
    "dphitt": np.linspace(0, _pi, 9),     # |Δφ| ∈ [0, π]
}
SEP = "=" * 123


def physical(reco, variable):
    """Маска фізично допустимих значень (фінітні + у діапазоні PHYS_RANGE)."""
    m = np.isfinite(reco)
    if variable in PHYS_RANGE:
        lo, hi = PHYS_RANGE[variable]
        m &= (reco > lo) & (reco < hi)
    return m


def detect_methods(tree):
    """Методи реконструкції, наявні у файлі, разом із дослідницькими моделями lkrnn* (nn_apply.py --study)."""
    keys = set(tree.keys())
    base = [m for m in ["lkr", "lkrv2", "lkrv3", "lkrnn", "fkr", "skr"] if m in keys]
    extra = sorted(k for k in keys if k.startswith("lkrnn") and k not in ("lkrnn", "lkrnn_gen")
                   and not k.endswith("_gen") and f"mtt_{k}" in keys)
    return base + extra


def wrap_angle(r):
    r = r.copy()
    r[r > np.pi] -= 2 * np.pi
    r[r < -np.pi] += 2 * np.pi
    return r


def calc_metrics(reco, gen, valid, variable):
    """(bias, roзд. здатність, N) по валідних фізичних подіях."""
    valid = valid & physical(reco, variable)
    n_ok = int(valid.sum())
    if n_ok == 0:
        return float("nan"), float("nan"), 0
    residual = reco[valid] - gen[valid]
    if variable in ANGULAR:
        residual = wrap_angle(residual)
    return residual.mean(), residual.std(), n_ok


def abs_bias(reco, gen, valid, variable):
    """Bias по модулю: середнє |reco − gen| по подіях (зсуви різних знаків не гасять один одного)."""
    valid = valid & physical(reco, variable)
    residual = reco[valid] - gen[valid]
    if variable in ANGULAR:
        residual = wrap_angle(residual)
    return float(np.mean(np.abs(residual))) if len(residual) else float("nan")


def binned_metrics(reco, gen, base_valid, passed, variable, bins, res_mask=None):
    """Ефективність (%) і roзд. здатність у кожному біні gen.
    Ефективність — на base_valid; res_mask (LKRnn: події поза тренуванням NN) обмежує лише roзд. здатність.
    Повертає (effs, ress, знаменники ефективності, кількості подій для roзд. здатності, зсуви)."""
    effs, ress, ndens, nres, biases = [], [], [], [], []
    valid = base_valid & passed
    phys = physical(reco, variable)
    for i in range(len(bins) - 1):
        mask_bin = (gen > bins[i]) & (gen < bins[i + 1])
        n_den = int((mask_bin & base_valid).sum())
        ndens.append(n_den)
        effs.append(100.0 * int((mask_bin & valid).sum()) / n_den if n_den else float("nan"))
        mres = mask_bin & valid & phys
        if res_mask is not None:
            mres &= res_mask
        nres.append(int(mres.sum()))
        if int(mres.sum()) >= 2:
            r = reco[mres] - gen[mres]
            if variable in ANGULAR:
                r = wrap_angle(r)
            ress.append(np.sqrt(np.mean((r - r.mean()) ** 2)))
            biases.append(float(r.mean()))
        else:
            ress.append(float("nan"))
            biases.append(float("nan"))
    return effs, ress, ndens, nres, biases


def improvement(base, val):
    """Покращення roзд. здатності відносно базового методу, у %."""
    if base is None or not np.isfinite(base) or base <= 0 or not np.isfinite(val):
        return "---"
    return f"{100 * (base - val) / base:+.2f}%"


def method_base(base, sample, method):
    """Знаменник методу: алгоритмічні — усі події, LKRnn — лише події поза тренуванням NN."""
    return base & sample if nn_split.uses_nn(method) else base


def print_integral(a, level, methods, base, sample, refs=("lkr", "lkrv3"), labels=None):
    suffix = "" if level == "det" else "_gen"
    labels = labels or {m: m.upper() for m in methods}
    counts = []
    w = max([7] + [len(labels[m]) for m in methods])
    print(f"\n{SEP}\nІНТЕГРАЛЬНО (весь діапазон), {level.upper()}-рівень\n{SEP}")
    print(f"{'Змінна':<8} | {'Метод':<{w}} | {'N знам.':<8} | {'Еф. (%)':<9} | {'Bias':<11} | {'|Bias|':<11} | {'Resolution':<11} | "
          f"{'vs LKR':<9} | {'vs LKRv3':<9}")
    print(SEP)
    for variable in VARS:
        gen_arr = a[f"{variable}_gen"]
        res_by = {}
        for r in refs:                                       # еталони — навіть якщо їх не друкуємо
            res_by[r] = calc_metrics(a[f"{variable}_{r}{suffix}"], gen_arr, base & (a[f"{r}{suffix}"] == 1), variable)[1]
        for m in methods:
            n_den = int(base.sum())
            valid = base & (a[f"{m}{suffix}"] == 1)              # ефективність — на всьому датасеті
            eff = 100.0 * int(valid.sum()) / n_den if n_den else float("nan")
            res_valid = method_base(valid, sample, m)            # bias/res LKRnn — лише поза тренуванням NN
            bias, res, n_ok = calc_metrics(a[f"{variable}_{m}{suffix}"], gen_arr, res_valid, variable)
            abias = abs_bias(a[f"{variable}_{m}{suffix}"], gen_arr, res_valid, variable)
            res_by[m] = res
            counts.append((variable, m, n_ok, n_den))
            vs_lkr = improvement(res_by.get("lkr"), res) if m != "lkr" else "---"
            vs_v3 = improvement(res_by.get("lkrv3"), res) if m not in ("lkr", "lkrv2", "lkrv3") else "---"
            print(f"{variable:<8} | {labels[m]:<{w}} | {n_den:<8} | {eff:<9.2f} | {bias:<11.4f} | {abias:<11.4f} | {res:<11.4f} | "
                  f"{vs_lkr:<9} | {vs_v3:<9}")
        print("-" * len(SEP))
    return counts


def print_binned(a, level, methods, base, sample, show_bias=False, show_eff=True, labels=None):
    suffix = "" if level == "det" else "_gen"
    labels = labels or {m: m.upper() for m in methods}
    w = max(10, max(len(labels[m]) for m in methods) + 5)         # спільна ширина колонок: назви не «з'їжджають»
    nn = [m for m in methods if nn_split.uses_nn(m)]
    print(f"\n{SEP}\nПО БІНАХ, {level.upper()}-рівень  (eff = N(reco)/N(знаменника) у біні, res = roзд. здатність;"
          f"\nN_bin — усі події (знаменник ефективності), N_res NN — події LKRnn поза тренуванням NN для res)\n{SEP}")
    for variable in VARS:
        bins = BINS[variable]
        gen_arr = a[f"{variable}_gen"]
        per = {m: binned_metrics(a[f"{variable}_{m}{suffix}"], gen_arr, base, a[f"{m}{suffix}"] == 1, variable, bins,
                                 sample if nn_split.uses_nn(m) else None) for m in methods}
        hdr = f"{'бін (центр)':<13} | {'N_bin':<8}"
        if nn:
            hdr += f" | {'N_res NN':<8}"
        for m in methods:
            if show_eff:
                hdr += f" | {labels[m] + ' eff%':<{w}}"
            hdr += f" | {labels[m] + ' res':<{w}}"
            if show_bias:
                hdr += f" | {labels[m] + ' bias':<{w}}"
        print(f"\n--- {variable} ---\n{hdr}\n{'-' * len(hdr)}")
        centers = 0.5 * (bins[:-1] + bins[1:])
        for bi, c in enumerate(centers):
            row = f"{c:<13.3f} | {per[methods[0]][2][bi]:<8}"
            if nn:
                row += f" | {per[nn[0]][3][bi]:<8}"
            for m in methods:
                if show_eff:
                    row += f" | {per[m][0][bi]:<{w}.2f}"
                row += f" | {per[m][1][bi]:<{w}.4f}"
                if show_bias:
                    row += f" | {per[m][4][bi]:<+{w}.4f}"
            print(row)


def print_counts(counts, level, labels=None):
    print(f"\n{'=' * 60}\nКІЛЬКІСТЬ ПОДІЙ ({level}-рівень, сумарно)\n{'=' * 60}")
    labels = labels or {m: m.upper() for _, m, _, _ in counts}
    w = max([7] + [len(labels[m]) for _, m, _, _ in counts])
    print(f"{'Змінна':<8} | {'Метод':<{w}} | {'N':<9} | {'Знаменник':<9}")
    print("-" * 60)
    for variable, m, n_ok, n_den in counts:
        print(f"{variable:<8} | {labels[m]:<{w}} | {n_ok:<9} | {n_den:<9}")


def read_arrays(filename):
    """Читає потрібні гілки; повертає (масиви, методи)."""
    tree = uproot.open(filename)["ttbarTree"]
    methods = detect_methods(tree)
    keys = ["reco_passed_selection"]
    for v in VARS:
        keys.append(f"{v}_gen")
        for m in methods:
            keys += [f"{v}_{m}", f"{v}_{m}_gen"]
    for m in methods:
        keys += [m, f"{m}_gen"]
    keys = [k for k in dict.fromkeys(keys) if k in tree.keys()]
    return tree.arrays(keys, library="np"), methods


def combine_arrays(arrays_list):
    """Зшиває масиви кількох каналів в один набір."""
    common = set(arrays_list[0])
    for a in arrays_list[1:]:
        common &= set(a)
    return {k: np.concatenate([a[k] for a in arrays_list]) for k in common}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_det_{ch}.root",
                    help="шаблон вхідних файлів; за замовч. основні результати (det-модель), для крос-тесту: ttbar_output_full_{ch}.root з --split-model gen")
    ap.add_argument("--channels", "--channel", dest="channels", type=int, nargs="+", default=[1, 2, 3],
                    help="канали, що об'єднуються: 1=ee, 2=mumu, 3=emu")
    ap.add_argument("--level", choices=["det", "gen"], default="det", help="рівень оцінки")
    ap.add_argument("--split-model", dest="split_model", choices=["det", "gen"], default=None,
                    help="модель, чиї тренувальні події виключаються (за замовч. = --level; для gen–det: gen)")
    ap.add_argument("--split-file", dest="split_file", default=None,
                    help="dataset_split.root відповідної моделі (за замовч. визначається з --split-model або з файлу nn_apply)")
    ap.add_argument("--all-events", dest="all_events", action="store_true",
                    help="LKRnn теж на всіх подіях, разом із тренувальними (лише для порівняння)")
    ap.add_argument("--no-binned", dest="no_binned", action="store_true", help="без виводу по бінах")
    ap.add_argument("--binned-bias", dest="binned_bias", action="store_true",
                    help="у таблиці по бінах додати колонку зсуву для кожного методу")
    ap.add_argument("--label", action="append", default=[], metavar="МЕТОД=НАЗВА",
                    help="своя коротка назва методу в таблицях, напр. --label lkrnnmlpl1zraw='MLP L1'")
    ap.add_argument("--no-binned-eff", dest="binned_eff", action="store_false",
                    help="не друкувати колонки ефективності в таблиці по бінах (у мереж вона однакова)")
    ap.add_argument("--methods", nargs="+", default=None, metavar="МЕТОД",
                    help="друкувати лише ці методи (напр. lkrnn lkrnnmlpl1z); «vs LKR» і «vs LKRv3» рахуються й так")
    args = ap.parse_args()
    split_model = args.split_model or args.level

    loaded, names, infos, methods = [], [], [], None
    for ch in args.channels:
        fname = args.input_pattern.format(ch=ch)
        if not os.path.exists(fname):
            print(f"[W] {fname} не знайдено — канал {ch} пропущено")
            continue
        a, m_ch = read_arrays(fname)
        methods = m_ch if methods is None else [m for m in methods if m in m_ch]
        if args.all_events or not any(nn_split.uses_nn(m) for m in m_ch):
            a["_sample"] = np.ones(len(a["mtt_gen"]), dtype=bool)
        else:
            a["_sample"], info = nn_split.eval_sample(fname, ch, split_model, args.split_file)
            infos.append((ch, info))
        loaded.append(a)
        names.append(CHANNEL_NAMES.get(ch, str(ch)))
    if not loaded:
        print("[E] Жодного вхідного файлу не знайдено.")
        return

    a = combine_arrays(loaded)
    suffix = "" if args.level == "det" else "_gen"
    methods = [m for m in methods if f"{m}{suffix}" in a and f"mtt_{m}{suffix}" in a]   # напр. дослідницькі — лише det
    refs = [m for m in ("lkr", "lkrv3") if m in methods]                    # еталони для колонок «vs LKR», «vs LKRv3»
    if args.methods:
        missing = [m for m in args.methods if m not in methods]
        if missing:
            raise SystemExit(f"[E] немає у файлах: {missing}; є: {methods}")
        methods = list(args.methods)
    labels = {m: m.upper() for m in methods}                                # за замовч. — назва гілки
    labels.update(nn_split.short_method_labels(args.input_pattern.format(ch=args.channels[0])))
    for s in args.label:                                                    # ручні назви мають перевагу
        k, sep, v = s.partition("=")
        if not sep:
            raise SystemExit(f"[E] очікується МЕТОД=НАЗВА, отримано {s!r}")
        labels[k] = v
    full = nn_split.method_labels(args.input_pattern.format(ch=args.channels[0]))
    base = (a["reco_passed_selection"] == 1) if args.level == "det" else np.ones(len(a["mtt_gen"]), dtype=bool)
    sample = a["_sample"]

    print(SEP)
    print(f"СУМАРНО {'+'.join(names)}  |  файли: {args.input_pattern}  |  рівень оцінки: {args.level}  |  "
          f"методи: {', '.join(m.upper() for m in methods)}")
    print("Ефективність усіх методів, bias/res алгоритмів: УСІ події")
    if args.all_events:
        print("Bias/res LKRnn: УСІ події, разом із тренувальними (лише для порівняння)")
    elif infos:
        print(f"Bias/res LKRnn: лише події поза тренуванням {split_model}-моделі (випадкова частина повного датасету)")
        print(f"  {'канал':<6} | {'подій':>9} | {'придатні':>9} | {'train/val':>9} | {'тест':>8} | "
              f"{'+ непридатні':>12} | {'частка f':>8}")
        for ch, inf in infos:
            print(f"  {CHANNEL_NAMES.get(ch, ch):<6} | {inf['n']:>9} | {inf['eligible']:>9} | {inf['trained']:>9} | "
                  f"{inf['test']:>8} | {inf['extra']:>12} | {inf['frac']:>8.4f}")
        print(f"  подій у вибірці для bias/res LKRnn: {int(sample.sum())} із {len(sample)}")
    for m in methods:
        if m in full:
            print(f"  {labels[m]:<16} = {full[m]}  ({m})")
    print(SEP)

    counts = print_integral(a, args.level, methods, base, sample, refs, labels)
    if not args.no_binned:
        print_binned(a, args.level, methods, base, sample, args.binned_bias, args.binned_eff, labels)
    print_counts(counts, args.level, labels)


if __name__ == "__main__":
    main()

"""
fold_check.py
─────────────
Direct (forward) folding: теорія не розгортається, а «згортається» відгуком детектора й реконструкції,
y_reco = R · x_теорія, і порівнюється з reco-спектром. Якість методу реконструкції — наскільки добре порівняння
на reco-рівні РОЗРІЗНЯЄ дві теорії спектра M(tt̄).

Теорія A — номінальний MC. Теорія B — той самий MC, у якому кожна подія має вагу w(M_правда) (значення w у центрі
1-ГеВ біна справжньої маси), нормований на ту саму кількість подій (порівнюється лише форма). Для кожного методу
    Δχ² = Σ_i (y_B,i − y_A,i)² / y_A,i
у reco-бінах, у кількостях подій MC. «Частка ідеалу» = Δχ² / Δχ²(reco = правда). Усі методи — на тих самих
подіях поза тренуванням мереж (~15%, eval_sample).
y_A — цілі числа: кожна подія оригінального MC має вагу 1. y_B — не цілі: кожна подія входить із вагою w(M_правда)
(дробове число), і гістограма ще множиться на Σy_A/Σy_B, щоб кількість подій була та сама.

Сценарії теорії B (амплітуди масштабуються так, щоб для ідеального детектора Δχ² = --ref-z², за замовчуванням 25):
  нахил      w = (M / 500 ГеВ)^0.1                 — плавна зміна форми;
  резонанс   w = 1 + 0.5·exp(−(M − 500)²/2·30²)    — вузька структура;
  поріг      w = 1 + 0.3·exp(−(M − 350)²/2·5²)     — надлишок біля порога (топоній).

Reco-біни — одні для всіх методів, якомога дрібніші, підбираються на оригінальному MC: сітка по 1 ГеВ від 0 до
3000 ГеВ (+ переповнення). Від піку спектра в обидва боки біни ростуть: кожен — найвужчий, у якому для кожного
методу >= --min-count (100) подій, але не вужчий за попередній (ширина від піку не спадає — без чергування
вужчих і ширших бінів). Біля країв, де 100 подій уже не набрати, до --edge-bins (3) бінів можуть мати менше, але
>= --min-edge (20); неповний залишок додається до крайнього біна. Кількості подій у кожному біні — у кінці
<outdir>/fold_check.txt.

Параметрична модель: w(M) = 1 + a·t + b·t², t = (M − 500)/500; очікувані точності σ(a), σ(b) з форми reco-спектра
(фішерівська інформація, з поправкою на шум скінченного MC).

Матриця відгуку і її число обумовленості — у response_matrix.py.

Рисунок <outdir>/fold_check.pdf: рядок — сценарій, стовпець — метод; звичайний і перезважений розподіли на рівні
правди (синій, однаковий для всіх стовпців; на власних бінах за правдою, тим самим правилом) і після реконструкції
(червоний, на reco-бінах) та їхні відношення. Δχ² і для правди, і для reco рахується на reco-бінах.
Масштаб осі y задають лише звичайні біни: широкі крайні (напр. 0–361 ГеВ у LKR) виходять за верх осі.

  python fold_check.py --input-pattern 'ttbar_output_fig9_{ch}.root'
"""

import argparse
import os

import numpy as np

import integral_res as ir
import nn_split

METHODS = [("gen", "ideal (reco = true)"), ("lkr", "LKR"), ("lkrnn", "LKRnn bounded (C++)"),
           ("lkrnnbndraw", "LKRnn bounded"), ("lkrnnfreeraw", "LKRnn unbounded"),
           ("lkrnnmlpl1zmlj", "MLP"), ("lkrnnflowmlj", "Flow"),
           ("lkrnnmlp", "MLP"), ("lkrnnflow", "Flow")]          # назви гілок із train_best.sh / README
# (назва, вага теорії B, діапазон осі x на рисунку)
SCENARIOS = [("нахил", lambda m: (np.maximum(m, 1) / 500.0) ** 0.1, (250, 1000)),
             ("резонанс", lambda m: 1 + 0.5 * np.exp(-0.5 * ((m - 500) / 30) ** 2), (250, 1000)),
             ("поріг", lambda m: 1 + 0.3 * np.exp(-0.5 * ((m - 350) / 5) ** 2), (250, 700))]
TITLES = {"нахил": r"slope: $w=(M/500)^{0.1}$", "резонанс": r"resonance: $+0.5\,G(500, 30)$",
          "поріг": r"threshold: $+0.3\,G(350, 5)$"}
GEN_WIDTH = 1.0                   # біни правди, ГеВ
SCALE = {}                        # множник амплітуди кожного сценарію (однаковий еталон Δχ²)


def weight(wf, m):
    """Вага теорії B у центрі 1-ГеВ біна справжньої маси, з масштабом амплітуди: w = 1 + k·(w₀ − 1)."""
    mc = (np.floor(m / GEN_WIDTH) + 0.5) * GEN_WIDTH
    return 1.0 + SCALE.get(id(wf), 1.0) * (wf(mc) - 1.0)


def chi2(reco, w, edges):
    """Δχ² між формами теорій A (ваги 1) і B (ваги w) у бінах edges; повертає також гістограми."""
    yA = np.histogram(reco, edges)[0].astype(float)                  # цілі: кількість подій MC
    yB = np.histogram(reco, edges, weights=w)[0]
    yB *= yA.sum() / yB.sum()                                        # та сама кількість подій — лише форма
    ok = yA > 0
    return float(np.sum((yB[ok] - yA[ok]) ** 2 / yA[ok])), yA, yB


def make_edges(sets, min_count, min_edge, n_edge, empty=-1, grid=1.0, m_max=3000.0):
    """Спільні reco-біни на оригінальному MC (сітка по grid + переповнення): від піку спектра в обидва боки
    ширина бінів не спадає; у кожному біні >= min_count подій для кожного методу, у до n_edge крайніх — >= min_edge."""
    fine = np.concatenate([np.arange(0.0, m_max + grid / 2, grid), [1e5]])
    counts = np.stack([np.histogram(reco, fine)[0] for _, _, reco, _ in sets])
    peak = int(np.argmax(counts.min(0)))                             # сітковий бін із найбільшим мінімумом по методах

    def side(cnt):
        """Межі від пікового краю (у кроках сітки) для однієї сторони."""
        cum = np.concatenate([np.zeros((len(cnt), 1)), np.cumsum(cnt, 1)], axis=1)
        n, i, w, thr, n_rel, out = cnt.shape[1], 0, 1, min_count, 0, []
        while i < n and n_rel < n_edge:
            c = cum[:, i + w:] - cum[:, [i]]                         # події в біні [i, j) для кінців j >= i + w
            ok = ((c >= thr) | (c <= empty)).all(0) & (c >= thr).any(0)   # кожному вистачає або практично порожньо
            if not ok.any():
                if thr == min_count:                                 # min_count уже не набрати — крайні біни
                    thr = min_edge
                    continue
                break
            j = i + w + int(np.argmax(ok))
            out.append(j)
            w, i = j - i, j
            n_rel += thr == min_edge
        out = out or [n]
        out[-1] = n                                                  # неповний залишок — до крайнього біна
        return out

    right = [peak + c for c in side(counts[:, peak:])]
    left = [peak - c for c in side(counts[:, :peak][:, ::-1])] if peak else []
    return fine[sorted(set(left + [peak] + right))]


def fisher(reco, t, edges):
    """σ(a) (лише a) і σ(a), σ(b) (разом) для w = 1 + a·t + b·t² з форми reco-спектра."""
    nb = len(edges) - 1
    i = np.digitize(reco, edges) - 1
    f = np.stack([t, t ** 2])
    y = np.bincount(i, minlength=nb)[:nb]
    D = np.stack([np.bincount(i, weights=fk, minlength=nb)[:nb] for fk in f])
    c = D.sum(1) / y.sum()                                           # нормування форми
    Dn, fc = D - np.outer(c, y), f - c[:, None]
    noise = np.stack([[np.bincount(i, weights=fc[k] * fc[l], minlength=nb)[:nb] for l in range(2)]
                      for k in range(2)])                            # шум скінченного MC — віднімається
    ok = y > 0
    I = np.einsum("ki,li->kl", Dn[:, ok], Dn[:, ok] / y[ok]) - (noise[:, :, ok] / y[ok]).sum(-1)
    cov = np.linalg.inv(I)
    return 1 / np.sqrt(I[0, 0]), np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def plot(sets, ideal, edges, gen_edges, outdir, scenarios=None):
    """Рядок — сценарій, стовпець — метод: правда (синій, на власних gen-бінах) і reco (червоний, на reco-бінах),
    звичайний і перезважений, + відношення. Δχ² у підписах — для обох на reco-бінах."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def drawn(e):
        d = e.copy()
        d[-1] = max(1000.0, d[-2] + 50.0)                            # відкритий останній бін — до 1000 ГеВ
        return d

    draw, gen_draw = drawn(edges), drawn(gen_edges)
    scen = [sc for sc in SCENARIOS if scenarios is None or sc[0] in scenarios]
    nr, nc = len(scen), len(sets)
    fig = plt.figure(figsize=(4.8 * nc, 4.3 * nr))
    gs = fig.add_gridspec(3 * nr - 1, nc, height_ratios=sum(([2.4, 1.2, 0.8] for _ in range(nr)), [])[:-1],
                          hspace=0.06, wspace=0.26)
    for r, (sname, wf, (lo, hi)) in enumerate(scen):
        for c, (_, lab, reco, truth) in enumerate(sets):
            ax, axr = fig.add_subplot(gs[3 * r, c]), fig.add_subplot(gs[3 * r + 1, c])
            ymax = 0.0
            for k, (level, x, tr, e, d, color) in enumerate(
                    (("generator level", ideal, ideal, gen_edges, gen_draw, "#1f77b4"),
                     (f"reco: {lab}", reco, truth, edges, draw, "#d62728"))):
                d2 = chi2(x, weight(wf, tr), edges)[0]                   # Δχ² — завжди на reco-бінах
                _, yA, yB = chi2(x, weight(wf, tr), e)                   # гістограми — на бінах свого рівня
                shown = (d[:-1] < hi) & (d[1:] > lo)                     # біни у видимому діапазоні
                ax.stairs(yA, d, color=color, lw=1.6, label=f"{level}: nominal")                  # події в біні
                ax.stairs(yB, d, color=color, lw=1.2, ls=(0, (4, 2)), label=f"{level}: reweighted")
                wd = np.diff(d)
                narrow = shown & (wd <= 1.5 * np.median(wd[shown]))     # ширші біни (напр. 0–361, 361–369 ГеВ у LKR)
                ymax = max(ymax, yA[narrow].max(), yB[narrow].max())  # не задають масштаб — виходять за верх осі
                with np.errstate(all="ignore"):
                    ratio = yB / yA
                axr.stairs(ratio, d, color=color, lw=1.6, baseline=None)
                axr.text(0.02, 0.93 - 0.24 * k, rf"{level}: $\Delta\chi^2$ = {d2:.1f}", transform=axr.transAxes,
                         color=color, fontsize=10, va="top")
                if k == 0:
                    gen_ratio = ratio[shown & np.isfinite(ratio)]
                else:
                    band = 1 / np.sqrt(np.maximum(yA, 1e-9))
                    axr.fill_between(draw, np.append(1 - band, 1 - band[-1]), np.append(1 + band, 1 + band[-1]),
                                     step="post", color="#dddddd", lw=0, zorder=0,
                                     label="stat. ±1σ" if r == c == 0 else None)
            ax.set_ylim(0, 1.15 * ymax)
            ax.set_xlim(lo, hi)
            axr.set_xlim(lo, hi)
            axr.set_ylim(min(0.9, gen_ratio.min() - 0.03), max(1.1, gen_ratio.max() + 0.08))
            axr.axhline(1, color="#222222", lw=0.8)
            if hi > edges[-2]:
                ax.axvline(edges[-2], color="#999999", lw=0.8, ls=":")
            ax.tick_params(labelbottom=False, labelsize=10)
            axr.tick_params(labelsize=10)
            if r == 0:
                ax.set_title(lab, fontsize=13)
            if r == c == 0:
                ax.legend(fontsize=9.5, loc="upper right")
                axr.legend(fontsize=9.5, loc="lower right")
            if c == 0:
                ax.set_ylabel(f"{TITLES[sname]}\nevents per bin", fontsize=11)
                axr.set_ylabel("reweighted / nominal", fontsize=10)
            if r == nr - 1:
                axr.set_xlabel(r"$M(t\bar t)$ [GeV]", fontsize=12)
    stem = os.path.join(outdir, "fold_check")
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print(f"[I] {stem}.png, .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_fig9_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--plot-scenarios", dest="plot_scenarios", nargs="+", default=None,
                    choices=[sc[0] for sc in SCENARIOS], help="рядки рисунка (за замовч. усі сценарії)")
    ap.add_argument("--plot-methods", dest="plot_methods", nargs="+",
                    default=["lkr", "lkrnnfreeraw", "lkrnnmlpl1zmlj", "lkrnnflowmlj", "lkrnnmlp", "lkrnnflow"],
                    help="стовпці рисунка (ті, що є у файлах)")
    ap.add_argument("--ref-z", dest="ref_z", type=float, default=5.0,
                    help="амплітуди сценаріїв — так, щоб для ідеального детектора Δχ² = Z²")
    ap.add_argument("--min-count", dest="min_count", type=int, default=100,
                    help="мінімум подій MC у reco-біні для кожного методу")
    ap.add_argument("--min-edge", dest="min_edge", type=int, default=20,
                    help="мінімум подій MC у перших і останніх --edge-bins бінах")
    ap.add_argument("--empty", type=int, default=-1,
                    help="бін вважається порожнім для методу, якщо в ньому <= EMPTY подій (за замовч. -1 — вимкнено)")
    ap.add_argument("--edge-bins", dest="edge_bins", type=int, default=3,
                    help="скільки крайніх бінів з кожного боку можуть мати менше --min-count подій")
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
    truth = a["mtt_gen"]

    sets = []                                    # (ключ, підпис, reco, правда) — усі на тих самих подіях
    for m, lab in METHODS:
        if m != "gen" and m not in present:
            continue
        sel = base if m == "gen" else base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")
        sets.append((m, lab, (truth if m == "gen" else a[f"mtt_{m}"])[sel], truth[sel]))
    edges = make_edges(sets, args.min_count, args.min_edge, args.edge_bins, args.empty)
    gen_edges = make_edges(sets[:1], args.min_count, args.min_edge, args.edge_bins)   # лише для рисунка
    gen_edges[0] = np.floor(sets[0][2].min())                        # правди нижче 2mₜ немає — бін від першої події

    g = sets[0][2]                               # ідеальний детектор: однаковий еталон Δχ² у всіх сценаріях
    for _, wf, _ in SCENARIOS:
        for _ in range(5):                       # Δχ² не зовсім квадратичний за амплітудою (нормування) — ітерації
            SCALE[id(wf)] = SCALE.get(id(wf), 1.0) * args.ref_z / np.sqrt(chi2(g, weight(wf, g), edges)[0])

    nb = len(edges) - 1
    counts = np.stack([np.histogram(reco, edges)[0] for _, _, reco, _ in sets])
    rng = lambda b: f"{edges[b]:.0f}–" + ("∞" if b == nb - 1 else f"{edges[b + 1]:.0f}")
    L = [f"Direct folding, M(tt̄): {len(g)} подій MC (сигнал, ~15% поза тренуванням мереж), усі методи на тих самих подіях.",
         f"Reco-біни ({nb}; >= {args.min_count} подій MC для кожного методу, до {args.edge_bins} крайніх з кожного боку — "
         f">= {args.min_edge}; ширина від піку не спадає), ГеВ:",
         "  " + " | ".join(f"{e:.0f}" for e in edges[:-1]) + " | ∞",
         f"Біни генераторного рівня на рисунку (лише за правдою, {len(gen_edges) - 1}; Δχ² усюди — на reco-бінах), ГеВ:",
         "  " + " | ".join(f"{e:.0f}" for e in gen_edges[:-1]) + " | ∞",
         "",
         f"Δχ² (у дужках — частка ідеального детектора); амплітуди теорій — для ідеалу Δχ² = {args.ref_z ** 2:g}",
         f"{'метод':<22}" + "".join(f"{s:>18}" for s, _, _ in SCENARIOS)]
    ideal = {}
    for _, lab, reco, tr in sets:
        cells = []
        for sname, wf, _ in SCENARIOS:
            d2 = chi2(reco, weight(wf, tr), edges)[0]
            ideal.setdefault(sname, d2)
            cells.append(f"{d2:9.1f} ({100 * d2 / ideal[sname]:3.0f}%)")
        L.append(f"{lab:<22}" + "".join(f"{x:>18}" for x in cells))

    L += ["", "Параметрична модель w = 1 + a·t + b·t², t = (M − 500)/500: очікувані точності",
          f"{'метод':<22}{'σ(a), лише a':>14}{'σ(a), a і b':>14}{'σ(b), a і b':>14}"]
    for _, lab, reco, tr in sets:
        t = ((np.floor(tr / GEN_WIDTH) + 0.5) * GEN_WIDTH - 500.0) / 500.0
        L.append(f"{lab:<22}" + "".join(f"{v:14.4f}" for v in fisher(reco, t, edges)))

    text = "\n".join(L)
    print(text)
    B = ["", f"Подій MC у кожному reco-біні (оригінальний MC, вага 1 — цілі числа)",
         f"{'№':>4} {'бін, ГеВ':>12}" + "".join(f"{lab:>{max(len(lab), 6) + 2}}" for _, lab, _, _ in sets)]
    B += [f"{b:4d} {rng(b):>12}" + "".join(f"{v:{max(len(lab), 6) + 2}d}" for (_, lab, _, _), v in zip(sets, counts[:, b]))
          for b in range(nb)]
    os.makedirs(args.outdir, exist_ok=True)
    txt = os.path.join(args.outdir, "fold_check.txt")
    open(txt, "w").write(text + "\n" + "\n".join(B) + "\n")
    print(f"[I] {txt} (у кінці — кількості подій у кожному біні)")
    plot([s for s in sets if s[0] in args.plot_methods], g, edges, gen_edges, args.outdir, args.plot_scenarios)


if __name__ == "__main__":
    main()

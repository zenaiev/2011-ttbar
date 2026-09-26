"""
fold_check.py
─────────────
Direct (forward) folding: теорія не розгортається, а «згортається» відгуком детектора й реконструкції,
y_reco = R · x_теорія, і порівнюється з reco-спектром. Якість методу реконструкції — наскільки добре порівняння
на reco-рівні РОЗРІЗНЯЄ дві теорії спектра M(tt̄).

Теорія A — номінальний MC. Теорія B — той самий MC, у якому кожна подія має вагу w(M_правда) (значення w у центрі
1-ГеВ біна справжньої маси), нормований на ту саму кількість подій (порівнюється лише форма). Для кожного методу
    Δχ² = Σ_i (y_B,i − y_A,i)² / y_A,i
у reco-бінах, при статистиці даних 2011 (лише сигнал, 2.5 фб⁻¹). «Частка ідеалу» = Δχ² / Δχ²(reco = правда).
Усі методи — на тих самих подіях поза тренуванням мереж (~15%, eval_sample), з вагою --data-weight / 0.15.

Сценарії теорії B (амплітуди масштабуються так, щоб для ідеального детектора Δχ² = --ref-z², за замовчуванням 25):
  нахил      w = (M / 500 ГеВ)^0.1                 — плавна зміна форми;
  резонанс   w = 1 + 0.5·exp(−(M − 750)²/2·30²)    — вузька структура;
  поріг      w = 1 + 0.3·exp(−(M − 350)²/2·5²)     — надлишок біля порога (топоній).

Reco-біни — одні для всіх методів, якомога дрібніші: сітка по 1 ГеВ від 0 до 3000 ГеВ (+ переповнення), сусідні
біни зливаються, доки в кожному для кожного методу не буде >= --min-count (100) очікуваних подій даних або
практично нуль (< 1; порожній бін дає в Δχ² нуль). Межі друкуються у виводі.

Параметрична модель: w(M) = 1 + a·t + b·t², t = (M − 500)/500; очікувані точності σ(a), σ(b) з форми reco-спектра
(фішерівська інформація, з поправкою на шум скінченного MC).

Рисунок <outdir>/fold_check.pdf: рядок — сценарій, стовпець — метод; звичайний і перезважений розподіли на рівні
правди (синій, однаковий для всіх стовпців) і після реконструкції (червоний) та їхні відношення.

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
             ("резонанс", lambda m: 1 + 0.5 * np.exp(-0.5 * ((m - 750) / 30) ** 2), (250, 1000)),
             ("поріг", lambda m: 1 + 0.3 * np.exp(-0.5 * ((m - 350) / 5) ** 2), (250, 700))]
TITLES = {"нахил": r"slope: $w=(M/500)^{0.1}$", "резонанс": r"resonance: $+0.5\,G(750, 30)$",
          "поріг": r"threshold: $+0.3\,G(350, 5)$"}
GEN_WIDTH = 1.0                   # біни правди, ГеВ
SCALE = {}                        # множник амплітуди кожного сценарію (однаковий еталон Δχ²)


def weight(wf, m):
    """Вага теорії B у центрі 1-ГеВ біна справжньої маси, з масштабом амплітуди: w = 1 + k·(w₀ − 1)."""
    mc = (np.floor(m / GEN_WIDTH) + 0.5) * GEN_WIDTH
    return 1.0 + SCALE.get(id(wf), 1.0) * (wf(mc) - 1.0)


def chi2(reco, w, edges, scale):
    """Δχ² між формами теорій A (ваги 1) і B (ваги w) у бінах edges; повертає також гістограми."""
    yA = np.histogram(reco, edges)[0] * scale
    yB = np.histogram(reco, edges, weights=w)[0] * scale
    yB *= yA.sum() / yB.sum()                                        # та сама кількість подій — лише форма
    ok = yA > 0
    return float(np.sum((yB[ok] - yA[ok]) ** 2 / yA[ok])), yA, yB


def make_edges(sets, min_count, empty=1.0, grid=1.0, m_max=3000.0):
    """Спільні reco-біни: сітка по grid + переповнення; зливати сусідні, доки для кожного методу в біні
    >= min_count подій або < empty (порожній)."""
    fine = np.concatenate([np.arange(0.0, m_max + grid / 2, grid), [1e5]])
    counts = np.stack([np.histogram(reco, fine)[0] * scale for _, _, reco, _, scale in sets])
    edges, acc = [0.0], np.zeros(len(sets))
    for i in range(counts.shape[1]):
        acc += counts[:, i]
        if ((acc >= min_count) | (acc < empty)).all() and (acc >= min_count).any():
            edges.append(fine[i + 1])
            acc[:] = 0.0
    edges[-1] = fine[-1]                                             # неповний хвіст — до останнього біна
    return np.array(edges)


def fisher(reco, t, scale, edges):
    """σ(a) (лише a) і σ(a), σ(b) (разом) для w = 1 + a·t + b·t² з форми reco-спектра."""
    nb = len(edges) - 1
    i = np.digitize(reco, edges) - 1
    f = np.stack([t, t ** 2])
    y = np.bincount(i, minlength=nb)[:nb] * scale
    D = np.stack([np.bincount(i, weights=fk, minlength=nb)[:nb] * scale for fk in f])
    c = D.sum(1) / y.sum()                                           # нормування форми
    Dn, fc = D - np.outer(c, y), f - c[:, None]
    noise = np.stack([[np.bincount(i, weights=fc[k] * fc[l], minlength=nb)[:nb] * scale ** 2 for l in range(2)]
                      for k in range(2)])                            # шум скінченного MC — віднімається
    ok = y > 0
    I = np.einsum("ki,li->kl", Dn[:, ok], Dn[:, ok] / y[ok]) - (noise[:, :, ok] / y[ok]).sum(-1)
    cov = np.linalg.inv(I)
    return 1 / np.sqrt(I[0, 0]), np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def plot(sets, ideal, edges, outdir):
    """Рядок — сценарій, стовпець — метод: правда (синій) і reco (червоний), звичайний і перезважений, + відношення."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    draw = edges.copy()
    draw[-1] = max(1000.0, draw[-2] + 50.0)                          # відкритий останній бін — до 1000 ГеВ
    width = np.diff(draw)
    nr, nc = len(SCENARIOS), len(sets)
    fig = plt.figure(figsize=(4.8 * nc, 4.3 * nr))
    gs = fig.add_gridspec(3 * nr - 1, nc, height_ratios=sum(([2.4, 1.2, 0.8] for _ in range(nr)), [])[:-1],
                          hspace=0.06, wspace=0.26)
    for r, (sname, wf, (lo, hi)) in enumerate(SCENARIOS):
        for c, (_, lab, reco, truth, scale) in enumerate(sets):
            ax, axr = fig.add_subplot(gs[3 * r, c]), fig.add_subplot(gs[3 * r + 1, c])
            ymax = 0.0
            for k, (level, x, tr, sc, color) in enumerate((("generator level", ideal[0], ideal[0], ideal[1], "#1f77b4"),
                                                           (f"reco: {lab}", reco, truth, scale, "#d62728"))):
                d2, yA, yB = chi2(x, weight(wf, tr), edges, sc)
                ax.stairs(yA / width, draw, color=color, lw=1.6, label=f"{level}: nominal")       # густина: біни різні
                ax.stairs(yB / width, draw, color=color, lw=1.2, ls=(0, (4, 2)), label=f"{level}: reweighted")
                ymax = max(ymax, (yA / width).max(), (yB / width).max())
                with np.errstate(all="ignore"):
                    ratio = yB / yA
                axr.stairs(ratio, draw, color=color, lw=1.6, baseline=None)
                axr.text(0.02, 0.93 - 0.2 * k, rf"{level}: $\Delta\chi^2$ = {d2:.1f}", transform=axr.transAxes,
                         color=color, fontsize=8, va="top")
                if k == 0:
                    gen_ratio = ratio
                else:
                    band = 1 / np.sqrt(np.maximum(yA, 1e-9))
                    axr.fill_between(draw, np.append(1 - band, 1 - band[-1]), np.append(1 + band, 1 + band[-1]),
                                     step="post", color="#dddddd", lw=0, zorder=0,
                                     label="data stat. ±1σ" if r == c == 0 else None)
            vis = (draw[:-1] < hi) & (draw[1:] > lo) & np.isfinite(gen_ratio)
            ax.set_ylim(0, 1.15 * ymax)
            ax.set_xlim(lo, hi)
            axr.set_xlim(lo, hi)
            axr.set_ylim(min(0.9, gen_ratio[vis].min() - 0.03), max(1.1, gen_ratio[vis].max() + 0.08))
            axr.axhline(1, color="#222222", lw=0.8)
            if hi > edges[-2]:
                ax.axvline(edges[-2], color="#999999", lw=0.8, ls=":")
            ax.tick_params(labelbottom=False, labelsize=8)
            axr.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if r == c == 0:
                ax.legend(fontsize=7.5, loc="lower left")
                axr.legend(fontsize=7.5, loc="lower right")
            if c == 0:
                ax.set_ylabel(f"{TITLES[sname]}\nevents / GeV", fontsize=9)
                axr.set_ylabel("reweighted / nominal", fontsize=8)
            if r == nr - 1:
                axr.set_xlabel(r"$M(t\bar t)$ [GeV]", fontsize=9)
    stem = os.path.join(outdir, "fold_check")
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print(f"[I] {stem}.png, .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_fig9_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--plot-methods", dest="plot_methods", nargs="+",
                    default=["lkr", "lkrnnfreeraw", "lkrnnmlpl1zmlj", "lkrnnflowmlj", "lkrnnmlp", "lkrnnflow"],
                    help="стовпці рисунка (ті, що є у файлах)")
    ap.add_argument("--data-weight", dest="data_weight", type=float, default=0.007529,
                    help="вага події сигнального MC відносно даних 2011 (ttbarMakeHist.cxx)")
    ap.add_argument("--ref-z", dest="ref_z", type=float, default=5.0,
                    help="амплітуди сценаріїв — так, щоб для ідеального детектора Δχ² = Z²")
    ap.add_argument("--min-count", dest="min_count", type=float, default=100.0,
                    help="мінімум очікуваних подій даних у reco-біні для кожного методу")
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
    scale = args.data_weight / 0.15

    sets = []                                    # (ключ, підпис, reco, правда, вага) — усі на тих самих подіях
    for m, lab in METHODS:
        if m != "gen" and m not in present:
            continue
        sel = base if m == "gen" else base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")
        sets.append((m, lab, (truth if m == "gen" else a[f"mtt_{m}"])[sel], truth[sel], scale))
    edges = make_edges(sets, args.min_count)

    g = sets[0][2]                               # ідеальний детектор: однаковий еталон Δχ² у всіх сценаріях
    for _, wf, _ in SCENARIOS:
        SCALE[id(wf)] = args.ref_z / np.sqrt(chi2(g, weight(wf, g), edges, scale)[0])

    L = [f"Direct folding, M(tt̄): статистика даних 2011, {scale * len(g):.0f} подій сигналу; усі методи на тих самих "
         f"~15% подій MC ({len(g)}).",
         f"Reco-біни ({len(edges) - 1}, у кожному >= {args.min_count:g} подій для кожного методу або порожній), ГеВ:",
         "  " + " | ".join(f"{e:.0f}" for e in edges[:-1]) + " | ∞",
         "",
         f"Δχ² (у дужках — частка ідеального детектора); амплітуди теорій — для ідеалу Δχ² = {args.ref_z ** 2:g}",
         f"{'метод':<22}" + "".join(f"{s:>18}" for s, _, _ in SCENARIOS)]
    ideal = {}
    for _, lab, reco, tr, sc in sets:
        cells = []
        for sname, wf, _ in SCENARIOS:
            d2 = chi2(reco, weight(wf, tr), edges, sc)[0]
            ideal.setdefault(sname, d2)
            cells.append(f"{d2:9.1f} ({100 * d2 / ideal[sname]:3.0f}%)")
        L.append(f"{lab:<22}" + "".join(f"{x:>18}" for x in cells))

    L += ["", "Параметрична модель w = 1 + a·t + b·t², t = (M − 500)/500: очікувані точності",
          f"{'метод':<22}{'σ(a), лише a':>14}{'σ(a), a і b':>14}{'σ(b), a і b':>14}"]
    for _, lab, reco, tr, sc in sets:
        t = ((np.floor(tr / GEN_WIDTH) + 0.5) * GEN_WIDTH - 500.0) / 500.0
        L.append(f"{lab:<22}" + "".join(f"{v:14.4f}" for v in fisher(reco, t, sc, edges)))

    text = "\n".join(L)
    os.makedirs(args.outdir, exist_ok=True)
    open(os.path.join(args.outdir, "fold_check.txt"), "w").write(text + "\n")
    print(text)
    plot([s for s in sets if s[0] in args.plot_methods], (g, scale), edges, args.outdir)


if __name__ == "__main__":
    main()

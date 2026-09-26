"""
unfold_check.py
───────────────
Чи «лагодить» розгортка великий зсув реконструкції? Два питання, на які відповідає цей скрипт,
у бінах перерізу за масою tt̄ (CS_BINS з purity_stability.py):

1. ЦІНА В СТАТИСТИЦІ. Будуємо матрицю відгуку R[i][j] = P(reco в біні i | правда в біні j) для кожного
   методу, «вимірюємо» псевдодані з того самого MC і розгортаємо оберненням матриці:
       cov = (Rᵀ V⁻¹ R)⁻¹,  V = diag(очікувані відліки в бінах reco).
   Виводимо відносну статистичну похибку розгорнутого спектра по бінах і число обумовленості R.
   Зсув сам по собі тут не заважає — заважає розмиття подій між бінами, яке його супроводжує.

2. ЗАЛЕЖНІСТЬ ВІД МОДЕЛІ. Матриця будується з номінального MC, а псевдодані — з MC, у якому спектр
   справжньої маси перезважено (нахил ±): w(M) = (M / 500 ГеВ)^p. Якби R була властивістю лише детектора,
   розгортка відновила б перезважений спектр точно. Відхилення показує, наскільки метод «тягне» відповідь
   до спектра, на якому вчився: для стиснутої (зміщеної) відповіді воно більше.

3. МІГРАЦІЇ. Похибку розгортки визначає не RMS, а те, скільки подій переходить між бінами. Для кожного біна:
       стабільність_j = N(правда j, reco j) / N(правда j, reco будь-де)   — частка подій, що лишилася у своєму біні;
       чистота_i      = N(правда i, reco i) / N(reco i, правда будь-де)   — частка «своїх» у біні reco.
   Зведено: частки подій, що лишилися у своєму біні, пішли в сусідній, далі ніж на один бін і за межі
   345–1600 ГеВ. --matrix друкує повну матрицю P(reco | правда) у %, --plot малює її тепловою картою.

4. РЕГУЛЯРИЗОВАНА РОЗГОРТКА на статистиці даних 2011 (вага події сигнального MC --data-weight = 0.007529,
   2.5 фб⁻¹; фон не враховано, тож похибки занижені на ~20%). Як у TUnfold: мінімізується
       (y − R x)ᵀ V⁻¹ (y − R x) + τ² ‖L (x / x_MC − 1)‖²,   L — друга різниця (кривизна) по бінах правди,
   тобто штрафується кривизна відношення до спектра MC. Бінів reco може бути більше, ніж бінів правди
   (--reco-split ділить кожен бін reco на N частин, --reco-low додає бін reco 0–345 ГеВ). τ вибирається
   за мінімумом середньої глобальної кореляції (ScanTau у TUnfold) — процедура, яку можна виконати на даних.
   Виводимо τ, середню глобальну кореляцію, стат. похибку по бінах (τ = 0 — незарегуляризований розв'язок
   найменших квадратів) — абсолютного й нормованого (1/σ dσ/dM, як у CMS) спектра — і зсув розгорнутого
   спектра в сценаріях «нахил» і «резонанс».

  python3 unfold_check.py --input-pattern 'ttbar_output_fig3_{ch}.root' --plot figs_best
"""

import argparse
import os

import numpy as np

import integral_res as ir
import nn_split
from purity_stability import CS_BINS

METHODS = [("lkr", "LKR"), ("lkrnn", "LKRnn (bounded)"), ("lkrnnfree", "LKRnn (unbounded, no calib.)"),
           ("lkrnnmlpl1zraw", "MLP L1+z (no calib.)"), ("lkrnnmlpl1z", "MLP L1+z (calib. α=0.4)"),
           ("lkrnnflowraw", "Flow+z mean (no calib.)"), ("lkrnnmlpl2zraw", "MLP L2+z (no calib.)"),
           ("lkrnnmlpl1zxraw", "MLP L1+z +extra"), ("lkrnnmlpl2zxraw", "MLP L2+z +extra"),
           ("lkrnnflowxraw", "Flow+z mean +extra"), ("lkrnnmlpl1zpair", "MLP L1+z +extra +pair"),
           ("lkrnnflowpair", "Flow+z mean +extra +pair"), ("lkrnnmlpl1zcsv", "MLP L1+z +CSV"),
           ("lkrnnflowcsv", "Flow+z mean +CSV"), ("lkrnnmlpl1zmlj", "MLP L1+z +CSV +m(lj)"),
           ("lkrnnflowmlj", "Flow+z mean +CSV +m(lj)")]
METHODS = [(m, lab) for m, lab in METHODS]
EDGES = np.array(CS_BINS["mtt"][1:], float)      # 345…1600 ГеВ: перший бін 0–345 порожній на рівні правди


def response(reco, truth, sel, edges, weights):
    """R[i][j] = P(reco у біні i | правда у біні j) з ваг weights; повертає також суму ваг правди по бінах."""
    nb = len(edges) - 1
    t = np.digitize(truth, edges) - 1
    r = np.digitize(reco, edges) - 1
    R, n_true = np.zeros((nb, nb)), np.zeros(nb)
    for j in range(nb):
        in_j = (t == j)
        n_true[j] = weights[in_j].sum()
        for i in range(nb):
            R[i, j] = weights[in_j & sel & (r == i)].sum()
        R[:, j] /= max(n_true[j], 1e-9)          # умовні ймовірності: ефективність = сума стовпця <= 1
    return R, n_true


def response_rect(reco, truth, sel, r_edges, t_edges, weights):
    """Як response(), але з окремими межами бінів reco (їх може бути більше, ніж бінів правди)."""
    t = np.digitize(truth, t_edges) - 1
    r = np.digitize(reco, r_edges) - 1
    nt, nr = len(t_edges) - 1, len(r_edges) - 1
    R, n_true = np.zeros((nr, nt)), np.zeros(nt)
    for j in range(nt):
        in_j = t == j
        n_true[j] = weights[in_j].sum()
        R[:, j] = np.bincount(r[in_j & sel & (r >= 0) & (r < nr)], weights=weights[in_j & sel & (r >= 0) & (r < nr)],
                              minlength=nr)[:nr] / max(n_true[j], 1e-9)
    return R, n_true


def tikhonov(R, y, x0, tau):
    """Розв'язок min (y − R x)ᵀ V⁻¹ (y − R x) + τ² ‖L (x / x0 − 1)‖², V = diag(y); повертає x і коваріацію."""
    nt = R.shape[1]
    Vi = np.diag(1.0 / y)
    F = R.T @ Vi @ R
    D2 = np.zeros((nt - 2, nt))
    for k in range(nt - 2):
        D2[k, k:k + 3] = (1.0, -2.0, 1.0)
    L = D2 @ np.diag(1.0 / x0)
    Ainv = np.linalg.inv(F + tau ** 2 * L.T @ L)
    x = x0 + Ainv @ R.T @ Vi @ (y - R @ x0)
    return x, Ainv @ F @ Ainv


def global_corr(cov):
    """Глобальні коефіцієнти кореляції ρ_i = sqrt(1 − 1 / (C_ii (C⁻¹)_ii)); вироджена коваріація (надто велике τ,
    розв'язок повторює форму MC) — ρ = 1."""
    if np.linalg.cond(cov) > 1e12:
        return np.ones(len(cov))
    return np.sqrt(np.maximum(1.0 - 1.0 / (np.diag(cov) * np.diag(np.linalg.inv(cov))), 0.0))


def migration(reco, truth, sel, edges):
    """Відліки M[i][j]: reco в біні i, правда в біні j; індекси 0 — нижче першої межі, nb+1 — вище останньої."""
    nb = len(edges) - 1
    t = np.digitize(truth[sel], edges)
    r = np.digitize(reco[sel], edges)
    M = np.zeros((nb + 2, nb + 2))
    np.add.at(M, (r, t), 1.0)
    return M


def migration_summary(M):
    """Стабільність і чистота по бінах та зведені частки (у %) для бінів правди 1..nb."""
    nb = M.shape[0] - 2
    inner = M[:, 1:nb + 1]                                   # стовпці — біни правди в діапазоні
    diag = np.diag(M)[1:nb + 1]
    stab = 100 * diag / np.maximum(inner.sum(0), 1e-9)
    pur = 100 * diag / np.maximum(M[1:nb + 1, :].sum(1), 1e-9)
    i, j = np.indices(inner.shape)                            # i — рядок reco (0..nb+1), j + 1 — бін правди
    dist = np.abs(i - (j + 1))
    outside = (i == 0) | (i == nb + 1)
    tot = inner.sum()
    frac = {"diag": inner[dist == 0].sum(), "near": inner[(dist == 1) & ~outside].sum(),
            "far": inner[(dist >= 2) & ~outside].sum(), "out": inner[outside].sum()}
    return stab, pur, {k: 100 * v / tot for k, v in frac.items()}


def bin_labels(edges):
    return [f"{edges[j]:.0f}-{edges[j + 1]:.0f}" for j in range(len(edges) - 1)]


def plot_matrices(mats, edges, outdir):
    """Теплові карти P(reco | правда) у % для кількох методів, спільна шкала (одна послідовна гама)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = len(edges) - 1
    ticks = [f"<{edges[0]:.0f}"] + bin_labels(edges) + [f">{edges[-1]:.0f}"]
    n = len(mats)
    ncol = min(n, 3)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol + 1.2, 4.6 * nrow + 0.4), squeeze=False,
                             layout="constrained")
    for ax in axes.flat[n:]:
        ax.axis("off")
    for k, (ax, (label, M)) in enumerate(zip(axes.flat, mats)):
        P = 100 * M[:, 1:nb + 1] / np.maximum(M[:, 1:nb + 1].sum(0), 1e-9)
        im = ax.imshow(P, origin="lower", cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(P):
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 55 else "#222222")
        for j in range(nb):                                   # діагональ: подія лишилася у своєму біні
            ax.add_patch(plt.Rectangle((j - 0.5, j + 0.5), 1, 1, fill=False, ec="#d62728", lw=1.3))
        _, _, frac = migration_summary(M)
        ax.set_title(f"{label}\nown bin {frac['diag']:.1f}%, neighbour {frac['near']:.1f}%, "
                     f"further {frac['far']:.1f}%, out {frac['out']:.1f}%", fontsize=9)
        ax.set_xticks(range(nb))
        ax.set_yticks(range(nb + 2))
        row, col = divmod(k, ncol)
        ax.set_xticklabels(bin_labels(edges), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ticks if col == 0 else [], fontsize=8)
        if row == nrow - 1 or k + ncol >= n:
            ax.set_xlabel(r"true $M(t\bar t)$ bin [GeV]", fontsize=9)
        if col == 0:
            ax.set_ylabel(r"reconstructed $M(t\bar t)$ bin [GeV]", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="P(reco bin | true bin) [%]")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, "migration_mtt")
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print(f"\n[I] {stem}.png, .pdf")


def unfold(R, n_reco):
    """Розгортка оберненням матриці: значення й коваріація (пуассонівські похибки псевдоданих)."""
    V_inv = np.diag(1.0 / np.maximum(n_reco, 1e-9))
    cov = np.linalg.inv(R.T @ V_inv @ R)
    return cov @ R.T @ V_inv @ n_reco, cov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_fig_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tilt", type=float, default=0.5, help="показник p у перезважуванні w = (M/500)^p")
    ap.add_argument("--bump", type=float, nargs=3, default=[900.0, 60.0, 2.0], metavar=("M0", "SIGMA", "AMP"),
                    help="додатковий сценарій: вузький «резонанс» усередині біна, w += AMP·exp(-(M-M0)²/2σ²)")
    ap.add_argument("--matrix", nargs="*", default=None, metavar="METHOD",
                    help="надрукувати матрицю P(reco | правда) у %% (без аргументів — для всіх методів)")
    ap.add_argument("--plot", default=None, metavar="OUTDIR", help="теплові карти матриць міграцій у OUTDIR")
    ap.add_argument("--plot-methods", dest="plot_methods", nargs="+",
                    default=["lkr", "lkrnn", "lkrnnfree", "lkrnnmlpl1zraw", "lkrnnmlpl2zraw", "lkrnnflowraw"])
    ap.add_argument("--data-weight", dest="data_weight", type=float, default=0.007529,
                    help="вага події сигнального MC відносно даних 2011 (ttbarMakeHist.cxx, mcSigReco)")
    ap.add_argument("--reco-split", dest="reco_split", type=int, default=2,
                    help="на скільки частин ділити кожен бін reco в регуляризованій розгортці")
    ap.add_argument("--reco-low", dest="reco_low", action="store_true", help="додати бін reco 0–345 ГеВ")
    ap.add_argument("--tau", type=float, default=None, help="фіксоване τ замість мінімуму глобальної кореляції")
    args = ap.parse_args()

    loaded = []
    for ch in args.channels:
        f = args.input_pattern.format(ch=ch)
        a, present = ir.read_arrays(f)
        a["_sample"], _ = nn_split.eval_sample(f, ch, "det")
        loaded.append(a)
    a = ir.combine_arrays(loaded)
    global METHODS
    METHODS = [(m, lab) for m, lab in METHODS if m in present]     # лише наявні у файлах методи
    base = a["reco_passed_selection"] == 1
    truth = a["mtt_gen"]
    nb = len(EDGES) - 1
    print(f"бінів: {nb} ({EDGES[0]:.0f}–{EDGES[-1]:.0f} ГеВ), подій із відбором: {int(base.sum())}\n")

    w_flat = np.ones(len(truth))
    w_tilt = np.where(truth > 0, (np.maximum(truth, 1) / 500.0) ** args.tilt, 1.0)

    print(f"{'метод':<30} | {'число обум.':>11} | відносна стат. похибка розгортки по бінах, %")
    print(f"{'':<30} | {'':>11} | " + " ".join(f"{EDGES[j]:.0f}-{EDGES[j+1]:.0f}".rjust(9) for j in range(nb)))
    rows = {}
    for m, label in METHODS:
        sel = base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")
        if nn_split.uses_nn(m):
            sel = sel & a["_sample"]
        w = w_flat * (1.0 / 0.15 if nn_split.uses_nn(m) else 1.0)   # мережі — на ~15% подій: зрівнюємо статистику
        R, n_true = response(a[f"mtt_{m}"], truth, sel, EDGES, w)
        n_reco = R @ n_true
        val, cov = unfold(R, n_reco)
        rel = 100 * np.sqrt(np.diag(cov)) / np.maximum(n_true, 1e-9)
        cond = np.linalg.cond(R)
        rows[m] = (R, n_true, rel, cond, migration(a[f"mtt_{m}"], truth, sel, EDGES))
        print(f"{label:<30} | {cond:>11.2f} | " + " ".join(f"{v:9.2f}" for v in rel))

    head = f"{'':<30} | " + " ".join(s.rjust(9) for s in bin_labels(EDGES))
    for title, idx in (("стабільність: частка подій біна правди, що лишилася у своєму біні reco, %", 0),
                       ("чистота: частка подій біна reco, у яких правда в тому самому біні, %", 1)):
        print(f"\nміграції — {title}\n{head}")
        for m, label in METHODS:
            v = migration_summary(rows[m][4])[idx]
            print(f"{label:<30} | " + " ".join(f"{x:9.1f}" for x in v))
    print(f"\nміграції — куди потрапляють події з бінів правди {EDGES[0]:.0f}–{EDGES[-1]:.0f} ГеВ, %")
    print(f"{'метод':<30} | {'свій бін':>9} | {'сусідній':>9} | {'далі':>9} | {'за межі':>9}")
    for m, label in METHODS:
        f = migration_summary(rows[m][4])[2]
        print(f"{label:<30} | {f['diag']:>9.1f} | {f['near']:>9.1f} | {f['far']:>9.1f} | {f['out']:>9.1f}")

    if args.matrix is not None:
        ticks = [f"<{EDGES[0]:.0f}"] + bin_labels(EDGES) + [f">{EDGES[-1]:.0f}"]
        for m, label in METHODS:
            if args.matrix and m not in args.matrix:
                continue
            M = rows[m][4][:, 1:nb + 1]
            P = 100 * M / np.maximum(M.sum(0), 1e-9)
            print(f"\nматриця міграцій {label}: P(reco | правда), %; стовпці — правда, рядки — reco")
            corner = "reco \\ правда"
            print(f"{corner:>13} " + " ".join(s.rjust(9) for s in bin_labels(EDGES)))
            for i in range(nb + 1, -1, -1):
                print(f"{ticks[i]:>13} " + " ".join(f"{x:9.1f}" for x in P[i]))

    if args.plot:
        plot_matrices([(lab, rows[m][4]) for m in args.plot_methods for mm, lab in METHODS if mm == m],
                      EDGES, args.plot)

    m0, s0, amp = args.bump
    scenarios = [(f"нахил w = (M/500)^{args.tilt:g}", w_tilt),
                 (f"резонанс: +{amp:g}·exp(-(M-{m0:.0f})²/2·{s0:.0f}²)",
                  1.0 + amp * np.exp(-0.5 * ((truth - m0) / s0) ** 2))]
    for title, w_scen in scenarios:
        print(f"\nзалежність від моделі: псевдодані зі спектром «{title}», матриця — з номінального MC")
        print(f"{'метод':<30} | {'макс. відх.':>11} | відхилення розгорнутого спектра від правди по бінах, %")
        for m, label in METHODS:
            R, n_true, *_ = rows[m]
            sel = base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")
            if nn_split.uses_nn(m):
                sel = sel & a["_sample"]
            w = w_scen * (1.0 / 0.15 if nn_split.uses_nn(m) else 1.0)
            R_t, n_true_t = response(a[f"mtt_{m}"], truth, sel, EDGES, w)
            n_reco_t = R_t @ n_true_t                  # «дані» з іншим спектром, але тим самим детектором
            val, _ = unfold(R, n_reco_t)               # розгортаємо номінальною матрицею
            dev = 100 * (val - n_true_t) / np.maximum(n_true_t, 1e-9)
            print(f"{label:<30} | {np.abs(dev).max():>10.2f}% | " + " ".join(f"{v:9.2f}" for v in dev))

    # ── регуляризована розгортка на статистиці даних ────────────────────────────────────────────────
    r_edges = np.concatenate([np.linspace(EDGES[k], EDGES[k + 1], args.reco_split + 1)[:-1] for k in range(nb)]
                             + [EDGES[-1:]])
    if args.reco_low:
        r_edges = np.concatenate([[0.0], r_edges])
    print(f"\nрегуляризована розгортка (Тихонов, кривизна x/x_MC), статистика даних 2011: вага MC-події "
          f"{args.data_weight:g}; бінів reco {len(r_edges) - 1}, бінів правди {nb}; фон не враховано")
    print(f"τ — {'фіксоване' if args.tau is not None else 'мінімум середньої глобальної кореляції ρ̄'}; "
          f"похибки й відхилення у %")
    cms = [5.4, 3.8, 4.9, 6.1, 6.9, 13.3, 22.4]
    print(f"{'метод':<30} {'τ':>8} {'ρ̄':>5} | {'':<8} | " + " ".join(s.rjust(9) for s in bin_labels(EDGES)))
    print(f"{'CMS TOP-11-013 (стат.)':<30} {'':>8} {'':>5} | {'стат.':<8} | " + " ".join(f"{v:9.1f}" for v in cms))
    taus = np.concatenate([[0.0], np.logspace(-2, 3, 201)])
    for m, label in METHODS:
        sel = base & (a[m] == 1) & ir.physical(a[f"mtt_{m}"], "mtt")
        if nn_split.uses_nn(m):
            sel = sel & a["_sample"]
        scale = args.data_weight / (0.15 if nn_split.uses_nn(m) else 1.0)
        R, x0 = response_rect(a[f"mtt_{m}"], truth, sel, r_edges, EDGES, np.full(len(truth), scale))
        y0 = R @ x0
        keep = y0 > 1e-3                                  # порожні біни reco (напр. 0–345 для мереж) не беремо
        R, y0 = R[keep], y0[keep]
        if args.tau is not None:
            tau = args.tau
        else:
            rho = [np.mean(global_corr(tikhonov(R, y0, x0, t)[1])) for t in taus[1:]]
            tau = taus[1 + int(np.argmin(rho))]
        for t in ([0.0, tau] if tau > 0 else [0.0]):
            _, cov = tikhonov(R, y0, x0, t)
            rel = 100 * np.sqrt(np.diag(cov)) / x0
            S = x0.sum()                                  # нормований переріз x/Σx, як у CMS: J = (S·I − x·1ᵀ)/S²
            J = (S * np.eye(nb) - np.outer(x0, np.ones(nb))) / S ** 2
            rel_norm = 100 * np.sqrt(np.diag(J @ cov @ J.T)) / (x0 / S)
            rows_out = [("стат.", rel), ("норм.", rel_norm)]
            for sname, w_scen in (("нахил", w_tilt), ("резонанс", 1.0 + amp * np.exp(-0.5 * ((truth - m0) / s0) ** 2))):
                R_t, x_t = response_rect(a[f"mtt_{m}"], truth, sel, r_edges, EDGES, w_scen * scale)
                y_t = (R_t @ x_t)[keep]
                x_hat, _ = tikhonov(R, y_t, x0, t)
                rows_out.append((sname, 100 * (x_hat - x_t) / x_t))
            rho_mean = np.mean(global_corr(cov))
            for k, (rname, v) in enumerate(rows_out):
                lead = f"{label if t == 0 else '':<30} {t:>8.3g} {rho_mean:>5.2f}" if k == 0 else f"{'':<30} {'':>8} {'':>5}"
                print(f"{lead} | {rname:<8} | " + " ".join(f"{x:9.1f}" for x in v))


if __name__ == "__main__":
    main()

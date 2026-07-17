import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys

# Відкриття дерева
# Якщо передали ім'я файла як параметр, використовувати його, або файл за замовчуванням
filename = "ttbar_output_3.root" if len(sys.argv) < 2 else sys.argv[1]
tree = uproot.open(filename)["ttbarTree"]

# Визначення масивів для зчитування
#kinrecos = ['lkrv2']
kinrecos = ['lkr','lkrv3','lkrnn']
variables = ['mtt', 'pttt', 'ytt']
arrays = tree.arrays(
    [f"{v}_{k}" for v in variables for k in kinrecos] + [f"{v}_gen" for v in variables],
    library="np"
)

# Бінінг
pi = np.pi
mtt_bins = np.concatenate((np.linspace(340, 450, 5, endpoint=False), np.logspace(np.log10(450), np.log10(700), 5, endpoint=False), np.logspace(np.log10(700), np.log10(1400), 8)))
pttt_bins = np.concatenate((np.linspace(0, 100, 5, endpoint=False), np.logspace(np.log10(100), np.log10(250), 5, endpoint=False), np.logspace(np.log10(250), np.log10(800), 8)))
ytt_bins = [-2.4,-2.0,-1.75] + np.linspace(-1.6, 1.6, 16, endpoint=False).tolist() + [1.75,2.0,2.4]
phitt_bins = np.linspace(-pi, pi, 9)
dphitt_bins = np.linspace(-pi, pi, 9)

bins_dict = {
    'mtt': mtt_bins,
    'pttt': pttt_bins,
    'ytt': ytt_bins,
    'phitt': phitt_bins,
    'dphitt': dphitt_bins
}
labels = {
    'mtt': '$M(t\\bar{t})$ [GeV]',
    'pttt': '$p_T(t\\bar{t})$ [GeV]',
    'ytt': '$y(t\\bar{t})$',
    'phitt': '$\\phi(t\\bar{t})$',
    'dphitt': r'$\Delta\phi(t\bar{t})$'
}
positions = {'mtt': (0, 0), 'pttt': (0, 1), 'ytt': (0, 2)}
# Функція обчислення
def calculate_efficiency(reco, gen, bins, variable):
    bin_centers = []
    efficiency = []
    efficiency_unc = []
    bias = []
    bias_unc = []
    resolution = []
    resolution_unc = []
    # маска для відбору реконструйованих подій
    # перевіряємо реконструйоване значення. Для невдалої реконструкції reco = -1000
    # це працює коректно для M(ttbar), pT(ttbar), y(ttbar)
    # також відкидаємо події з нефізичними значеннями M(ttbar), pT(ttbar), y(ttbar) > 99999
    mask_reco = (reco > -999) & (reco < 9999)
    for i in range(len(bins)-1):
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        # маска для відбору подій в біні
        mask_bin = (gen > bins[i]) & (gen < bins[i+1])
        events_gen = len(gen[mask_bin])
        rec_events = len(reco[mask_bin & mask_reco])
        if events_gen == 0:
            efficiency.append(float('nan'))
            efficiency_unc.append(float('nan'))
            bias.append(float('nan'))
            bias_unc.append(float('nan'))
            resolution.append(float('nan'))
            resolution_unc.append(float('nan'))
        else:
            efficiency.append(rec_events / events_gen)
            efficiency_unc.append(np.sqrt(efficiency[-1]*(1-efficiency[-1])/events_gen))
            if rec_events == 0:
                bias.append(float('nan'))
                bias_unc.append(float('nan'))
                resolution.append(float('nan'))
                resolution_unc.append(float('nan'))
            else:
                reco_final = reco[mask_bin & mask_reco]
                gen_final = gen[mask_bin & mask_reco]
                residual = reco_final - gen_final
                if variable == 'phitt':
                    residual[residual > np.pi] = residual[residual > np.pi] - 2*np.pi
                    residual[residual < -np.pi] = residual[residual < -np.pi] + 2*np.pi
                bias.append(np.mean(residual))
                resolution.append(np.sqrt(np.mean((residual-bias[-1])**2)))
                n = len(gen_final)
                bias_unc.append(resolution[-1]/np.sqrt(n))
                mu4 = np.mean((residual-bias[-1])**4)
                resolution_unc.append(np.sqrt(max(mu4 - resolution[-1]**4, 0.0) / n))
    return bin_centers, efficiency, efficiency_unc, bias, bias_unc, resolution, resolution_unc

# Обчислення ефективності, bias, resolution у циклі
results = {}
for kinreco in kinrecos:
    results[kinreco] = {}
    for variable in variables:
        reco_array = arrays[f"{variable}_{kinreco}"]
        gen_array = arrays[f"{variable}_gen"]
        bins = bins_dict[variable]
        results[kinreco][variable] = calculate_efficiency(reco_array, gen_array, bins, variable)
        with open(f'kr_performance/kr_performance_{kinreco}_{variable}.txt', 'w') as f:
            for i in range(len(results[kinreco][variable][0])):
                for ii in range(len(results[kinreco][variable])):
                    f.write(f'{results[kinreco][variable][ii][i]:.6e} ')
                f.write('\n')

# Побудова графіків ефективності
fig_eff, axs_eff = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
fig_eff.subplots_adjust(0.07, 0.12, 0.98, 0.88, wspace=0.32)


for variable in variables:
    i, j = positions[variable]
    for kinreco in kinrecos:
        bin_centers = results[kinreco][variable][0]
        eff = results[kinreco][variable][1]
        eff_unc = results[kinreco][variable][2]
        axs_eff[i, j].errorbar(bin_centers, eff, eff_unc, marker='o', markersize=5, label=kinreco.upper())
    axs_eff[i, j].set_xlabel(labels[variable])
    axs_eff[i, j].set_ylabel('Efficiency')
    axs_eff[i, j].legend()
    axs_eff[i, j].grid(True)
    if variable == 'phitt':
        axs_eff[i, j].set_ylim(0.5, 1.0)

fig_eff.suptitle('CMS open data $pp \\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV\nEfficiency of reconstructed variables')
fig_eff.savefig('plots/efficiency.pdf')
fig_eff.savefig('plots/efficiency.png')

fig_bias, axs_bias = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
fig_bias.subplots_adjust(0.07, 0.12, 0.98, 0.88, wspace=0.32)

for variable in variables:
    i, j = positions[variable]
    all_bias = []
    for kinreco in kinrecos:
        bin_centers = results[kinreco][variable][0]
        bias = results[kinreco][variable][3]
        bias_unc = results[kinreco][variable][4]
        axs_bias[i, j].errorbar(bin_centers, bias, bias_unc, marker='o', markersize=5, label=kinreco.upper())
        all_bias += [b for b in bias if b == b]  # відкидаємо NaN
    axs_bias[i, j].set_xlabel(labels[variable])
    axs_bias[i, j].set_ylabel('Bias')
    axs_bias[i, j].legend()
    axs_bias[i, j].grid(True)
    # масштаб за самими значеннями bias, а не за (роздутими) похибками
    if all_bias:
        lo, hi = min(all_bias), max(all_bias)
        pad = 0.2 * (hi - lo) if hi > lo else max(abs(hi), 1.0)
        axs_bias[i, j].set_ylim(lo - pad, hi + pad)

fig_bias.suptitle('CMS open data $pp \\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV\nBias of reconstructed variables')
fig_bias.savefig('plots/bias.pdf')
fig_bias.savefig('plots/bias.png')


fig_res, axs_res = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
fig_res.subplots_adjust(0.07, 0.12, 0.98, 0.88, wspace=0.32)

for variable in variables:
    i, j = positions[variable]
    # 1-й прохід: масштаб за самими значеннями роздільної здатності
    all_res = []
    for kinreco in kinrecos:
        all_res += [r for r in results[kinreco][variable][5] if r == r]  # без NaN
    ymax = max(all_res) * 1.25 if all_res else None
    err_cap = 0.3 * ymax if ymax else None  # обмеження довжини похибок для відображення
    # 2-й прохід: побудова з обрізаними похибками (малі — незмінні, роздуті — обрізані)
    for kinreco in kinrecos:
        bin_centers = results[kinreco][variable][0]
        resolution = results[kinreco][variable][5]
        resolution_unc = results[kinreco][variable][6]
        if err_cap is not None:
            resolution_unc = [min(u, err_cap) if u == u else 0.0 for u in resolution_unc]
        axs_res[i, j].errorbar(bin_centers, resolution, resolution_unc, marker='o',
                               markersize=5, elinewidth=1.0, capsize=2, alpha=0.85,
                               label=kinreco.upper())
    axs_res[i, j].set_xlabel(labels[variable])
    axs_res[i, j].set_ylabel('Resolution')
    axs_res[i, j].legend()
    axs_res[i, j].grid(True)
    if ymax:
        axs_res[i, j].set_ylim(0, ymax)

fig_res.suptitle('CMS open data $pp \\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV\nResolution of reconstructed variables')
fig_res.savefig('plots/resolution.pdf')
fig_res.savefig('plots/resolution.png')

# --- Побудова графіків відношення ---
fig_ratio, axs_ratio = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
fig_ratio.subplots_adjust(0.07, 0.12, 0.98, 0.88, wspace=0.32)

for variable in variables:
    if variable not in positions: continue
    i, j = positions[variable]

    # Базовий метод для порівняння - LKR
    res_lkr = np.array(results['lkr'][variable][5])
    bin_centers = results['lkr'][variable][0]

    # Малюємо відношення для кожного методу до LKR
    for k in kinrecos:
        res_current = np.array(results[k][variable][5])
        # divide із обробкою ділення на 0 (RuntimeWarning зникне)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = res_current / res_lkr
            
        color = 'red' if k == 'lkrnn' else None
        axs_ratio[i, j].plot(bin_centers, ratio, 'o-', label=f"{k.upper()}/LKR", markersize=4, color=color)

    axs_ratio[i, j].axhline(1.0, color='black', linestyle='--', alpha=0.5)
    axs_ratio[i, j].set_xlabel(labels[variable])
    axs_ratio[i, j].set_ylabel("Resolution ratio")
    axs_ratio[i, j].set_ylim(0.5, 1.5)
    axs_ratio[i, j].grid(True)
    axs_ratio[i, j].legend(fontsize='small', ncol=2)

fig_ratio.suptitle('Resolution Ratio relative to LKR')
fig_ratio.savefig('plots/resolution_ratio.png')
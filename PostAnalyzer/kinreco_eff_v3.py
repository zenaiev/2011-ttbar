import uproot
import numpy as np
import matplotlib.pyplot as plt

# Відкриття дерева
tree = uproot.open("ttbar_output_123.root")["ttbarTree"]

# Визначення масивів для зчитування
kinrecos = ['fkr', 'skr', 'lkr']
variables = ['mtt', 'pttt', 'ytt', 'phitt']
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

bins_dict = {
    'mtt': mtt_bins,
    'pttt': pttt_bins,
    'ytt': ytt_bins,
    'phitt': phitt_bins
}
labels = {
    'mtt': '$M(t\\bar{t})$ [GeV]',
    'pttt': '$p_T(t\\bar{t})$ [GeV]',
    'ytt': '$y(t\\bar{t})$',
    'phitt': '$\\phi(t\\bar{t})$'
}
positions = {'mtt': (0, 0), 'pttt': (0, 1), 'ytt': (1, 0), 'phitt': (1, 1)}
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
            # похибка дорівнює sqrt(E*(1-E)/N_gen), E=N_rec/N-gen
            efficiency_unc.append(np.sqrt(efficiency[-1]*(1-efficiency[-1])/events_gen))
            # рахуємо зсув (bias): bias = sum(rec-gen)/N
            # N це кількість вдало реконструйованих подій (використовуємо лише їх)
            # це рахується як середнє значення (mean) всіх rec-gen значень
            reco_final = reco[mask_bin & mask_reco]
            gen_final = gen[mask_bin & mask_reco]
            # один раз рахуємо і зберігаємо різницю rec-gen
            residual = reco_final - gen_final
            if variable == 'phitt':
                residual[residual > np.pi] = residual[residual > np.pi] - 2*np.pi
                residual[residual < -np.pi] = residual[residual < -np.pi] + 2*np.pi
            bias.append(np.mean(residual))
            # рахуємо роздільну здатність (resolution): sigma = sqrt(sum((rec-gen)^2-mean((rec-gen)^2)/n))
            # роздільна здатність це квадратний корінь із варіації (variance)
            # див. https://pdg.lbl.gov/ -> "Reviews, tables, plots" -> "Mathematicl tools" -> "Statistics" -> "40.2.1Estimators for mean, variance, and median"
            #resolution.append(np.std(residual)) # це короткий метод
            # це явний вигляд:
            resolution.append(np.sqrt(np.mean((residual-bias[-1])**2)))
            # рахуємо похибки на зсув (sigma/sqrt(n)) та роздільну здатність (sigma/sqrt(2n))
            n = len(gen_final)
            bias_unc.append(resolution[-1]/np.sqrt(n))
            resolution_unc.append(np.sqrt(np.sqrt(np.mean((residual-bias[-1])**4)-resolution[-1]**4)/n))
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

# Побудова графіків ефективності
fig_eff, axs_eff = plt.subplots(2, 2, figsize=(7, 7))
fig_eff.subplots_adjust(0.11, 0.08, 0.97, 0.93, wspace=0.28)


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

fig_bias, axs_bias = plt.subplots(2, 2, figsize=(7, 7))
fig_bias.subplots_adjust(0.11, 0.08, 0.97, 0.93, wspace=0.28)

for variable in variables:
    i, j = positions[variable]
    for kinreco in kinrecos:
        bin_centers = results[kinreco][variable][0]
        bias = results[kinreco][variable][3]
        bias_unc = results[kinreco][variable][4]
        axs_bias[i, j].errorbar(bin_centers, bias, bias_unc, marker='o', markersize=5, label=kinreco.upper())
    axs_bias[i, j].set_xlabel(labels[variable])
    axs_bias[i, j].set_ylabel('Bias')
    axs_bias[i, j].legend()
    axs_bias[i, j].grid(True)

fig_bias.suptitle('CMS open data $pp \\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV\nBias of reconstructed variables')
fig_bias.savefig('plots/bias.pdf')
fig_bias.savefig('plots/bias.png')


fig_res, axs_res = plt.subplots(2, 2, figsize=(7, 7))
fig_res.subplots_adjust(0.11, 0.08, 0.97, 0.93, wspace=0.28)

for variable in variables:
    i, j = positions[variable]
    for kinreco in kinrecos:
        bin_centers = results[kinreco][variable][0]
        resolution = results[kinreco][variable][5]
        resolution_unc = results[kinreco][variable][6]
        axs_res[i, j].errorbar(bin_centers, resolution, resolution_unc, marker='o', markersize=5, label=kinreco.upper())
    axs_res[i, j].set_xlabel(labels[variable])
    axs_res[i, j].set_ylabel('Resolution')
    axs_res[i, j].legend()
    axs_res[i, j].grid(True)

fig_res.suptitle('CMS open data $pp \\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV\nResolution of reconstructed variables')
fig_res.savefig('plots/resolution.pdf')
fig_res.savefig('plots/resolution.png')


import uproot
import numpy as np
import matplotlib.pyplot as plt

# Відкриваємо файл і дерево
#tree = uproot.open("/home/nik2305/Documents/new 2011-ttbar/PostAnalyzer/ttbar_output.root")["ttbarTree"]
#tree = uproot.open("ttbar_output_em.root")["ttbarTree"]
tree = uproot.open("ttbar_output_123.root")["ttbarTree"]
#tree = uproot.open("ttbar_output_3.root")["ttbarTree"]


# Зчитуємо необхідні масиви
arrays = tree.arrays([
    "mtt_fkr", "pttt_fkr", "ytt_fkr",
    "mtt_skr", "pttt_skr", "ytt_skr",
    "mtt_lkr", "pttt_lkr", "ytt_lkr",
    "mtt_gen", "pttt_gen", "ytt_gen"
], library="np")
# FKR arrays
mtt_fkr = arrays["mtt_fkr"]
pttt_fkr = arrays["pttt_fkr"]
ytt_fkr = arrays["ytt_fkr"]

# SKR arrays
mtt_skr = arrays["mtt_skr"]
pttt_skr = arrays["pttt_skr"]
ytt_skr = arrays["ytt_skr"]

# LKR arrays
mtt_lkr = arrays["mtt_lkr"]
pttt_lkr = arrays["pttt_lkr"]
ytt_lkr = arrays["ytt_lkr"]

# Generated arrays
mtt_gen = arrays["mtt_gen"]
pttt_gen = arrays["pttt_gen"]
ytt_gen = arrays["ytt_gen"]

# Визначаємо бінування
#mtt_bins = np.linspace(340, 1400, 14)
#pttt_bins = np.linspace(0, 800, 14)
#ytt_bins = np.linspace(-2.4, 2.4, 14)
mtt_bins = np.concatenate((np.linspace(340, 450, 5, endpoint=False), np.logspace(np.log10(450), np.log10(700), 5, endpoint=False), np.logspace(np.log10(700), np.log10(1400), 8)))
pttt_bins = np.concatenate((np.linspace(0, 100, 5, endpoint=False), np.logspace(np.log10(100), np.log10(250), 5, endpoint=False), np.logspace(np.log10(250), np.log10(800), 8)))
ytt_bins = [-2.4,-2.0,-1.75] + np.linspace(-1.6, 1.6, 16, endpoint=False).tolist() + [1.75,2.0,2.4]

# Функція для обрахунку ефективності
def calculate_efficiency(reco, gen, bins):
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
            bias.append(np.mean(residual))
            # рахуємо роздільну здатність (resolution): sigma = sqrt(sum((rec-gen)-mean((rec-gen)^2)/n))
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

bin_centers_mtt_fkr, eff_mtt_fkr, eff_unc_mtt_fkr, bias_mtt_fkr, bias_unc_mtt_fkr, resol_mtt_fkr, resol_unc_mtt_fkr = calculate_efficiency(mtt_fkr, mtt_gen, mtt_bins)
bin_centers_pttt_fkr, eff_pttt_fkr, eff_unc_pttt_fkr, bias_pttt_fkr, bias_unc_pttt_fkr, resol_pttt_fkr, resol_unc_pttt_fkr = calculate_efficiency(pttt_fkr, pttt_gen, pttt_bins)
bin_centers_ytt_fkr, eff_ytt_fkr, eff_unc_ytt_fkr, bias_ytt_fkr, bias_unc_ytt_fkr, resol_ytt_fkr, resol_unc_ytt_fkr = calculate_efficiency(ytt_fkr, ytt_gen, ytt_bins)

bin_centers_mtt_skr, eff_mtt_skr, eff_unc_mtt_skr, bias_mtt_skr, bias_unc_mtt_skr, resol_mtt_skr, resol_unc_mtt_skr = calculate_efficiency(mtt_skr, mtt_gen, mtt_bins)
bin_centers_pttt_skr, eff_pttt_skr, eff_unc_pttt_skr, bias_pttt_skr, bias_unc_pttt_skr, resol_pttt_skr, resol_unc_pttt_skr = calculate_efficiency(pttt_skr, pttt_gen, pttt_bins)
bin_centers_ytt_skr, eff_ytt_skr, eff_unc_ytt_skr, bias_ytt_skr, bias_unc_ytt_skr, resol_ytt_skr, resol_unc_ytt_skr = calculate_efficiency(ytt_skr, ytt_gen, ytt_bins)

bin_centers_mtt_lkr, eff_mtt_lkr, eff_unc_mtt_lkr, bias_mtt_lkr, bias_unc_mtt_lkr, resol_mtt_lkr, resol_unc_mtt_lkr = calculate_efficiency(mtt_lkr, mtt_gen, mtt_bins)
bin_centers_pttt_lkr, eff_pttt_lkr, eff_unc_pttt_lkr, bias_pttt_lkr, bias_unc_pttt_lkr, resol_pttt_lkr, resol_unc_pttt_lkr = calculate_efficiency(pttt_lkr, pttt_gen, pttt_bins)
bin_centers_ytt_lkr, eff_ytt_lkr, eff_unc_ytt_lkr, bias_ytt_lkr, bias_unc_ytt_lkr, resol_ytt_lkr, resol_unc_ytt_lkr = calculate_efficiency(ytt_lkr, ytt_gen, ytt_bins)

# не довіряємо похибкам для роздільної здатності, не показуєм їх на малюнках
#resol_unc_mtt_fkr = None
#resol_unc_pttt_fkr = None
#resol_unc_ytt_fkr = None
#resol_unc_mtt_skr = None
#resol_unc_pttt_skr = None
#resol_unc_ytt_skr = None
#resol_unc_mtt_lkr = None
#resol_unc_pttt_lkr = None
#resol_unc_ytt_lkr = None

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 12))

# FKR
#axs[0, 0].scatter(bin_centers_mtt_fkr, eff_mtt_fkr)
#axs[0, 0].plot(bin_centers_mtt_fkr, eff_mtt_fkr)
axs[0, 0].errorbar(bin_centers_mtt_fkr, eff_mtt_fkr, eff_unc_mtt_fkr, marker='o')
axs[0, 0].set_title('FKR mass')
axs[0, 0].set_xlabel('mtt')

axs[0, 1].scatter(bin_centers_pttt_fkr, eff_pttt_fkr)
axs[0, 1].plot(bin_centers_pttt_fkr, eff_pttt_fkr)
axs[0, 1].set_title('FKR: momentum')
axs[0, 1].set_xlabel('pttt')

axs[0, 2].scatter(bin_centers_ytt_fkr, eff_ytt_fkr)
axs[0, 2].plot(bin_centers_ytt_fkr, eff_ytt_fkr)
axs[0, 2].set_title('FKR: rapidity')
axs[0, 2].set_xlabel('ytt')


# SKR
axs[1, 0].scatter(bin_centers_mtt_skr, eff_mtt_skr)
axs[1, 0].plot(bin_centers_mtt_skr, eff_mtt_skr)
axs[1, 0].set_title('SKR: mtt')
axs[1, 0].set_xlabel('mtt')

axs[1, 1].scatter(bin_centers_pttt_skr, eff_pttt_skr)
axs[1, 1].plot(bin_centers_pttt_skr, eff_pttt_skr)
axs[1, 1].set_title('SKR: pttt')
axs[1, 1].set_xlabel('pttt')

axs[1, 2].scatter(bin_centers_ytt_skr, eff_ytt_skr)
axs[1, 2].plot(bin_centers_ytt_skr, eff_ytt_skr)
axs[1, 2].set_title('SKR: ytt')
axs[1, 2].set_xlabel('ytt')

# LKR
axs[2, 0].scatter(bin_centers_mtt_lkr, eff_mtt_lkr)
axs[2, 0].plot(bin_centers_mtt_lkr, eff_mtt_lkr)
axs[2, 0].set_title('LKR: mtt')
axs[2, 0].set_xlabel('mtt')

axs[2, 1].scatter(bin_centers_pttt_lkr, eff_pttt_lkr)
axs[2, 1].plot(bin_centers_pttt_lkr, eff_pttt_lkr)
axs[2, 1].set_title('LKR: pttt')
axs[2, 1].set_xlabel('pttt')

axs[2, 2].scatter(bin_centers_ytt_lkr, eff_ytt_lkr)
axs[2, 2].plot(bin_centers_ytt_lkr, eff_ytt_lkr)
axs[2, 2].set_title('LKR: ytt')
axs[2, 2].set_xlabel('ytt')

# Formatting
for ax in axs.flat:
    ax.set_ylabel('Efficiency')
    ax.grid(True)

# Plotting efficiency
fig_eff, axs_eff = plt.subplots(2, 2, figsize=(7, 7))
fig_eff.subplots_adjust(0.11,0.08,0.97,0.93,wspace=0.28)
axs_eff[0, 0].errorbar(bin_centers_mtt_skr, eff_mtt_skr, eff_unc_mtt_skr, marker='o', markersize=5, label='SKR')
axs_eff[0, 0].errorbar(bin_centers_mtt_fkr, eff_mtt_fkr, eff_unc_mtt_fkr, marker='o', markersize=5, label='FKR')
axs_eff[0, 0].errorbar(bin_centers_mtt_lkr, eff_mtt_lkr, eff_unc_mtt_lkr, marker='o', markersize=5, label='LKR')
axs_eff[0, 0].set_xlabel('$M(t\\bar{t})$ [GeV]')
axs_eff[0, 0].set_ylabel('Efficiency')
axs_eff[0, 0].legend()
axs_eff[0, 1].errorbar(bin_centers_pttt_skr, eff_pttt_skr, eff_unc_pttt_skr, marker='o', markersize=5, label='SKR')
axs_eff[0, 1].errorbar(bin_centers_pttt_fkr, eff_pttt_fkr, eff_unc_pttt_fkr, marker='o', markersize=5, label='FKR')
axs_eff[0, 1].errorbar(bin_centers_pttt_lkr, eff_pttt_lkr, eff_unc_pttt_lkr, marker='o', markersize=5, label='LKR')
axs_eff[0, 1].set_xlabel('$p_{T}(t\\bar{t})$ [GeV]')
axs_eff[0, 1].set_ylabel('Efficiency')
axs_eff[0, 1].legend()
axs_eff[1, 0].errorbar(bin_centers_ytt_skr, eff_ytt_skr, eff_unc_ytt_skr, marker='o', markersize=5, label='SKR')
axs_eff[1, 0].errorbar(bin_centers_ytt_fkr, eff_ytt_fkr, eff_unc_ytt_fkr, marker='o', markersize=5, label='FKR')
axs_eff[1, 0].errorbar(bin_centers_ytt_lkr, eff_ytt_lkr, eff_unc_ytt_lkr, marker='o', markersize=5, label='LKR')
axs_eff[1, 0].set_xlabel('$y(t\\bar{t})$')
axs_eff[1, 0].set_ylabel('Efficiency')
axs_eff[1, 0].legend()
fig_eff.suptitle('CMS open data $pp\\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV')
fig_eff.savefig('plots/efficiency.pdf')
fig_eff.savefig('plots/efficiency.png')

# Plotting bias
fig_bias, axs_bias = plt.subplots(2, 2, figsize=(7, 7))
fig_bias.subplots_adjust(0.11,0.08,0.97,0.93,wspace=0.28)
axs_bias[0, 0].errorbar(bin_centers_mtt_fkr, bias_mtt_fkr, bias_unc_mtt_fkr, marker='o', markersize=5, label='FKR')
axs_bias[0, 0].errorbar(bin_centers_mtt_skr, bias_mtt_skr, bias_unc_mtt_skr, marker='o', markersize=5, label='SKR')
axs_bias[0, 0].errorbar(bin_centers_mtt_lkr, bias_mtt_lkr, bias_unc_mtt_lkr, marker='o', markersize=5, label='LKR')
axs_bias[0, 0].set_xlabel('$M(t\\bar{t})$ [GeV]')
axs_bias[0, 0].set_ylabel('Bias [GeV]')
axs_bias[0, 0].legend()
axs_bias[0, 1].errorbar(bin_centers_pttt_fkr, bias_pttt_fkr, bias_unc_pttt_fkr, marker='o', markersize=5, label='FKR')
axs_bias[0, 1].errorbar(bin_centers_pttt_skr, bias_pttt_skr, bias_unc_pttt_skr, marker='o', markersize=5, label='SKR')
axs_bias[0, 1].errorbar(bin_centers_pttt_lkr, bias_pttt_lkr, bias_unc_pttt_lkr, marker='o', markersize=5, label='LKR')
axs_bias[0, 1].set_xlabel('$p_{T}(t\\bar{t})$ [GeV]')
axs_bias[0, 1].set_ylabel('Bias [GeV]')
axs_bias[0, 1].legend()
axs_bias[1, 0].errorbar(bin_centers_ytt_fkr, bias_ytt_fkr, bias_unc_ytt_fkr, marker='o', markersize=5, label='FKR')
axs_bias[1, 0].errorbar(bin_centers_ytt_skr, bias_ytt_skr, bias_unc_ytt_skr, marker='o', markersize=5, label='SKR')
axs_bias[1, 0].errorbar(bin_centers_ytt_lkr, bias_ytt_lkr, bias_unc_ytt_lkr, marker='o', markersize=5, label='LKR')
axs_bias[1, 0].set_xlabel('$y(t\\bar{t})$')
axs_bias[1, 0].set_ylabel('Bias')
axs_bias[1, 0].legend()
fig_bias.suptitle('CMS open data $pp\\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV')
fig_bias.savefig('plots/bias.pdf')
fig_bias.savefig('plots/bias.png')

# Plotting resolution
fig_resol, axs_resol = plt.subplots(2, 2, figsize=(7, 7))
fig_resol.subplots_adjust(0.11,0.08,0.97,0.93,wspace=0.28)
axs_resol[0, 0].errorbar(bin_centers_mtt_fkr, resol_mtt_fkr, resol_unc_mtt_fkr, marker='o', markersize=5, label='FKR')
axs_resol[0, 0].errorbar(bin_centers_mtt_skr, resol_mtt_skr, resol_unc_mtt_skr, marker='o', markersize=5, label='SKR')
axs_resol[0, 0].errorbar(bin_centers_mtt_lkr, resol_mtt_lkr, resol_unc_mtt_lkr, marker='o', markersize=5, label='LKR')
axs_resol[0, 0].set_xlabel('$M(t\\bar{t})$ [GeV]')
axs_resol[0, 0].set_ylabel('Resolution [GeV]')
axs_resol[0, 0].legend()
axs_resol[0, 1].errorbar(bin_centers_pttt_fkr, resol_pttt_fkr, resol_unc_pttt_fkr, marker='o', markersize=5, label='FKR')
axs_resol[0, 1].errorbar(bin_centers_pttt_skr, resol_pttt_skr, resol_unc_pttt_skr, marker='o', markersize=5, label='SKR')
axs_resol[0, 1].errorbar(bin_centers_pttt_lkr, resol_pttt_lkr, resol_unc_pttt_lkr, marker='o', markersize=5, label='LKR')
axs_resol[0, 1].set_xlabel('$p_{T}(t\\bar{t})$ [GeV]')
axs_resol[0, 1].set_ylabel('Resolution [GeV]')
axs_resol[0, 1].legend()
axs_resol[1, 0].errorbar(bin_centers_ytt_fkr, resol_ytt_fkr, resol_unc_ytt_fkr, marker='o', markersize=5, label='FKR')
axs_resol[1, 0].errorbar(bin_centers_ytt_skr, resol_ytt_skr, resol_unc_ytt_skr, marker='o', markersize=5, label='SKR')
axs_resol[1, 0].errorbar(bin_centers_ytt_lkr, resol_ytt_lkr, resol_unc_ytt_lkr, marker='o', markersize=5, label='LKR')
axs_resol[1, 0].set_xlabel('$y(t\\bar{t})$')
axs_resol[1, 0].set_ylabel('Resolution')
axs_resol[1, 0].legend()
fig_resol.suptitle('CMS open data $pp\\to t\\bar{t}$, dilepton decay channel, $\\sqrt{s}=7$ TeV')
fig_resol.savefig('plots/resolution.pdf')
fig_resol.savefig('plots/resolution.png')

plt.tight_layout()
#plt.show()

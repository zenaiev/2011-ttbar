import uproot
import numpy as np
import matplotlib.pyplot as plt

# Відкриваємо файл і дерево
tree = uproot.open("/home/nik2305/Documents/new 2011-ttbar/PostAnalyzer/ttbar_output.root")["ttbarTree"]

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
mtt_bins = np.linspace(340, 1400, 14)
pttt_bins = np.linspace(0, 800, 14)
ytt_bins = np.linspace(-2.5, 4, 14)

# Функція для обрахунку ефективності
def calculate_efficiency(reco, gen, bins):
    efficiency = []
    bin_centers = []
    for i in range(len(bins)-1):
        events_gen = len(gen[(gen > bins[i]) & (gen < bins[i+1])])
        rec_events = len(reco[(gen > bins[i]) & (gen < bins[i+1]) & (reco > 0)])
        if events_gen == 0:
            efficiency.append(0)
        else:
            efficiency.append(rec_events / events_gen)
        bin_centers.append((bins[i] + bins[i+1]) / 2)
    return bin_centers, efficiency

bin_centers_mtt_fkr, eff_mtt_fkr = calculate_efficiency(mtt_fkr, mtt_gen, mtt_bins)
bin_centers_pttt_fkr, eff_pttt_fkr = calculate_efficiency(pttt_fkr, pttt_gen, pttt_bins)
bin_centers_ytt_fkr, eff_ytt_fkr = calculate_efficiency(ytt_fkr, ytt_gen, ytt_bins)

bin_centers_mtt_skr, eff_mtt_skr = calculate_efficiency(mtt_skr, mtt_gen, mtt_bins)
bin_centers_pttt_skr, eff_pttt_skr = calculate_efficiency(pttt_skr, pttt_gen, pttt_bins)
bin_centers_ytt_skr, eff_ytt_skr = calculate_efficiency(ytt_skr, ytt_gen, ytt_bins)

bin_centers_mtt_lkr, eff_mtt_lkr = calculate_efficiency(mtt_lkr, mtt_gen, mtt_bins)
bin_centers_pttt_lkr, eff_pttt_lkr = calculate_efficiency(pttt_lkr, pttt_gen, pttt_bins)
bin_centers_ytt_lkr, eff_ytt_lkr = calculate_efficiency(ytt_lkr, ytt_gen, ytt_bins)

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 12))

# FKR
axs[0, 0].scatter(bin_centers_mtt_fkr, eff_mtt_fkr)
axs[0, 0].plot(bin_centers_mtt_fkr, eff_mtt_fkr)
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

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams["font.size"] = 14

ak = np.arange(0, 0.41, 0.01)

k_mod = np.exp(ak * np.exp(ak)) ** 1.25
N_mod = np.exp(ak * np.exp(ak)) ** 0.5
g_mod = (1 - ak * np.exp(ak)) ** (-0.25)

ak_mod = g_mod * k_mod * N_mod

k_mod_log = np.log(k_mod)
N_mod_log = np.log(N_mod)
g_mod_log = np.log(g_mod)
ak_mod_log = k_mod_log + N_mod_log + g_mod_log

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
ax1.plot(ak, k_mod, label=r"$(\widetilde{k}/k)^{\frac{5}{4}}$", lw=2)
ax1.plot(ak, N_mod, label=r"$(\widetilde{N}/N)^{\frac{1}{2}}$", lw=2)
ax1.plot(ak, g_mod, label=r"$(\widetilde{g}/g)^{-\frac{1}{4}}$", lw=2)
ax1.plot(ak, ak_mod, "k-", label=r"$(\widetilde{ak}/(ak))$", lw=2)
ax1.legend()
ax1.set_title("Modulation Factors")
ax1.set_xlabel(r"$\varepsilon_L$")
ax1.set_ylabel("Modulation")
ax1.set_xlim(0, 0.4)
ax1.set_ylim(1, 4)
ax1.text(0, 1.02, "(a)", transform=ax1.transAxes, va="bottom", ha="left")
ax1.grid(True, linestyle="--", alpha=0.7)

# Second subplot (modified to show percentages)
k_contrib_percent = 100 * k_mod_log / ak_mod_log
N_contrib_percent = 100 * N_mod_log / ak_mod_log
g_contrib_percent = 100 * g_mod_log / ak_mod_log

ax2.plot(ak, k_contrib_percent, label="Wavenumber", lw=2)
ax2.plot(ak, N_contrib_percent, label="Wave action", lw=2)
ax2.plot(ak, g_contrib_percent, label="Effective gravity", lw=2)
ax2.legend()
ax2.set_title("Relative contribution")
ax2.set_xlabel(r"$\varepsilon_L$")
ax2.set_ylabel("Contribution (%)")
ax2.set_xlim(0, 0.4)
ax2.set_ylim(0, 100)
ax2.text(0, 1.02, "(b)", transform=ax2.transAxes, va="bottom", ha="left")
ax2.grid(True, linestyle="--", alpha=0.7)

# Adjust layout and display the plot
plt.savefig("../figures/fig_modulation_contributors.pdf")
plt.close()

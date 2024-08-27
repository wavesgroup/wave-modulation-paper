import matplotlib.pyplot as plt
from twowave import WaveModulationModel
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rc("font", size=14)


def g_modulation(psi, a, k):
    eta = a * np.cos(psi)
    res = 1 - a * k * np.exp(k * eta) * (np.cos(psi) - a * k * np.sin(psi) ** 2)
    return res


def calculate_modulation(ak_L: float):
    k_long = 1
    a_long = ak_L / k_long

    m = WaveModulationModel(a_long=a_long, k_long=k_long)
    m.run(ramp_type="groups")
    ds = m.to_xarray()

    ind = 1002  # Exactly 5 periods

    phase = ds.space * k_long

    k_mod = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
    N_mod = k_mod
    g_mod = g_modulation(phase, a_long, k_long)
    a_mod = g_mod ** (-0.25) * k_mod**0.25 * N_mod**0.5

    k_mod_numerical = ds.wavenumber[ind] / m.k_short
    a_mod_numerical = ds.amplitude[ind] / m.a_short

    return phase, k_mod, a_mod, k_mod_numerical, a_mod_numerical


ak = [0.1, 0.2, 0.4]

fig, axes = plt.subplots(3, len(ak), figsize=(12, 8))

for n, ax in enumerate(axes.flatten()):
    ax.grid()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax.set_xticklabels([])
    ax.plot([0, 2 * np.pi], [1, 1], "k--")
    ax.text(
        0,
        1.02,
        f"({chr(97 + n)})",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=18,
    )

for n in range(3):
    (
        phase,
        k_modulation,
        a_modulation,
        k_modulation_numerical,
        a_modulation_numerical,
    ) = calculate_modulation(ak[n])

    axes[0, n].plot(
        phase,
        k_modulation,
        linestyle="-",
        lw=2,
        color="tab:blue",
        label="Analytical solution",
        alpha=0.8,
    )
    axes[0, n].plot(
        phase,
        k_modulation_numerical,
        linestyle="-",
        lw=2,
        color="tab:orange",
        label="Numerical solution",
        alpha=0.8,
    )
    if n == 0:
        plt.figlegend(loc="upper center", ncol=2)

    axes[1, n].plot(
        phase, a_modulation, linestyle="-", lw=2, color="tab:blue", alpha=0.8
    )
    axes[1, n].plot(
        phase,
        a_modulation_numerical,
        linestyle="-",
        lw=2,
        color="tab:orange",
        alpha=0.8,
    )

    axes[2, n].plot(
        phase,
        a_modulation * k_modulation,
        linestyle="-",
        lw=2,
        color="tab:blue",
        alpha=0.8,
    )
    axes[2, n].plot(
        phase,
        a_modulation_numerical * k_modulation_numerical,
        linestyle="-",
        lw=2,
        color="tab:orange",
        alpha=0.8,
    )

for n in range(len(ak)):
    axes[0, n].set_title(r"$\varepsilon_L$ = " + f"{ak[n]}")

axes[0, 0].set_ylabel(r"$\widetilde{k}/k$")
axes[1, 0].set_ylabel(r"$\widetilde{a}/a$")
axes[2, 0].set_ylabel(r"$\widetilde{ak}/(ak)$")

for ax in axes[-1]:
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel("Phase")

plt.savefig("../figures/numerical_vs_analytical_solutions_with_phase.png", dpi=200)
plt.close()

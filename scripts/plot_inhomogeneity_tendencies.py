import matplotlib.pyplot as plt
from twowave import WaveModulationModel
import numpy as np
import matplotlib

matplotlib.rc("font", size=16)

m1 = WaveModulationModel(num_periods=10)
m2 = WaveModulationModel(num_periods=10)

m1.run(ramp_type=None, wave_type="linear", save_tendencies=True)
m2.run(ramp_type="linear", wave_type="linear", save_tendencies=True)

ds1 = m1.to_xarray()
ds2 = m2.to_xarray()

T_L = 2 * np.pi / m1.omega_long
periods = ds1.time / T_L

ak1 = ds1.amplitude * ds1.wavenumber
ak2 = ds2.amplitude * ds2.wavenumber
ak_init = 0.1

t = 1 * T_L
n0 = np.argmin((ds1.time.values - t) ** 2)

eta = m1.a_long * np.cos(ds1.space)

fig = plt.figure(figsize=(12, 8))
axes = [
    plt.subplot2grid((2, 1), (0, 0)),
    plt.subplot2grid((2, 1), (1, 0)),
]
axes[0].plot(ds1.space, eta, "k--", lw=3, label=r"$\eta_L$")
axes[0].plot(ds1.space, ds1.wave_action[n0] - 1, "k:", lw=3, label=r"$N - 1$")
axes[0].plot(
    ds1.space,
    ds1.N_propagation_tendency[n0],
    lw=3,
    label=r"$C_g \frac{\partial N}{\partial x}$",
)
axes[0].plot(
    ds1.space,
    ds1.N_inhomogeneity_tendency[n0],
    lw=3,
    label=r"$N \frac{\partial C_g}{\partial x}$",
)
axes[0].plot(
    ds1.space,
    ds1.N_propagation_tendency.values[n0] + ds1.N_inhomogeneity_tendency.values[n0],
    lw=3,
    label=r"$C_g \frac{\partial N}{\partial x} + N \frac{\partial C_g}{\partial x}$",
)
axes[1].plot(ds2.space, eta, "k--", lw=3, label=r"$\eta_L$")
axes[1].plot(ds1.space, ds2.wave_action[n0] - 1, "k:", lw=3, label=r"$N - 1$")
axes[1].plot(
    ds2.space,
    ds2.N_propagation_tendency[n0],
    label=r"$C_g \frac{\partial N}{\partial x}$",
    lw=3,
)
axes[1].plot(
    ds2.space,
    ds2.N_inhomogeneity_tendency[n0],
    label=r"$N \frac{\partial C_g}{\partial x}$",
    lw=3,
)
axes[1].plot(
    ds2.space,
    ds2.N_propagation_tendency.values[n0] + ds2.N_inhomogeneity_tendency.values[n0],
    label=r"$C_g \frac{\partial N}{\partial x} + N \frac{\partial C_g}{\partial x}$",
    lw=3,
)
axes[1].legend(ncol=5, loc="upper center")
axes[1].set_xlabel(r"$\Psi$")

for ax in axes:
    ax.grid()
    ax.set_ylabel("Tendency")
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([])

axes[1].set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

axes[0].set_title("Sudden onset of long waves, $t = T_L$")
axes[1].set_title("Linear ramp of long waves, $t = T_L$")

axes[0].text(
    0.0, 1.0, "(a)", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=20
)
axes[1].text(
    0.0, 1.0, "(b)", transform=axes[1].transAxes, ha="left", va="bottom", fontsize=20
)

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

plt.savefig("../figures/inhomogeneity_tendencies.pdf")
plt.close()

import matplotlib.pyplot as plt
import numpy as np
from twowave import WaveModulationModel, diff

import matplotlib

matplotlib.rcParams["font.size"] = 14


def get_homogeneity_and_stationarity(a_long):
    m = WaveModulationModel(a_long=a_long)
    m.run(ramp_type="groups")
    ds = m.to_xarray()

    x, t = ds.space.values, ds.time.values
    dx, dt = np.diff(x)[0], np.diff(t)[0]

    # Homogeneity criteria
    n, _ = np.unravel_index(np.argmax(ds.wave_action.values), ds.wave_action.shape)
    k_h = ds.wavenumber.values[n]
    g_h = ds.gravitational_acceleration.values[n]
    N_h = ds.wave_action.values[n]

    H_k = 1 - np.abs(diff(k_h)) / dx / k_h / k_h
    H_g = 1 - np.abs(diff(g_h)) / dx / g_h / k_h
    H_N = 1 - np.abs(diff(N_h)) / dx / N_h / k_h

    # Stationarity criteria
    k_s = ds.wavenumber.values[:, 0]
    g_s = ds.gravitational_acceleration.values[:, 0]
    N_s = ds.wave_action.values[:, 0]
    sigma = np.sqrt(g_s * k_s)

    S_k = 1 - np.abs(diff(k_s)) / dt / k_s / sigma
    S_g = 1 - np.abs(diff(g_s)) / dt / g_s / sigma
    S_N = 1 - np.abs(diff(N_s)) / dt / N_s / sigma

    return (
        np.min(H_k),
        np.min(H_g),
        np.min(H_N),
        np.min(S_k),
        np.min(S_g),
        np.min(S_N),
    )


ak_values = np.arange(0, 0.41, 0.01)
H_k_values = np.zeros_like(ak_values)
H_g_values = np.zeros_like(ak_values)
H_N_values = np.zeros_like(ak_values)

S_k_values = np.zeros_like(ak_values)
S_g_values = np.zeros_like(ak_values)
S_N_values = np.zeros_like(ak_values)

for i, ak in enumerate(ak_values):
    (
        H_k_values[i],
        H_g_values[i],
        H_N_values[i],
        S_k_values[i],
        S_g_values[i],
        S_N_values[i],
    ) = get_homogeneity_and_stationarity(ak)


fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(ak_values, H_k_values, color="tab:blue", label=r"$H_k$", lw=2)
ax.plot(ak_values, H_N_values, color="tab:orange", label=r"$H_N$", lw=2)
ax.plot(ak_values, H_g_values, color="tab:green", label=r"$H_g$", lw=2)
ax.plot(ak_values, S_k_values, "--", color="tab:blue", label=r"$S_k$", lw=2)
ax.plot(ak_values, S_N_values, "--", color="tab:orange", label=r"$S_N$", lw=2)
ax.plot(ak_values, S_g_values, "--", color="tab:green", label=r"$S_g$", lw=2)

ax.plot(ak_values, 0.9 * np.ones_like(ak_values), "k:", lw=3)

ax.set_xlabel(r"$\varepsilon_L$")
ax.set_ylabel("Homogeneity & Stationarity")
ax.legend(ncol=2)
ax.grid()
ax.set_xlim(0, 0.4)
ax.set_ylim(0.5, 1)

fig.savefig("../figures/fig_action_conservation_criteria.pdf")
plt.close()

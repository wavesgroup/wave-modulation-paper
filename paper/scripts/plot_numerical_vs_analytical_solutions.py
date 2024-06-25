import matplotlib.pyplot as plt
from model import WaveModulationModel
import numpy as np
import matplotlib

matplotlib.rc("font", size=16)


def g_modulation(psi, a, k):
    eta = a * np.cos(psi)
    res = 1 - a * k * np.exp(k * eta) * (np.cos(psi) - a * k * np.sin(psi) ** 2)
    return res


k_long = 1

ak_range = np.arange(0, 0.45, 0.05)
solutions1 = []
solutions2 = []
for ak in ak_range:
    a_long = ak / k_long

    m = WaveModulationModel(a_long=a_long, k_long=k_long)
    m.run(ramp_type="groups")
    solutions1.append(m)

    m = WaveModulationModel(a_long=a_long, k_long=k_long)
    m.run(wave_type="stokes", ramp_type="groups")
    solutions2.append(m)


k_modulation_numerical_linear = np.array(
    [np.max(m.to_xarray().wavenumber) / m.k_short for m in solutions1]
)
a_modulation_numerical_linear = np.array(
    [np.max(m.to_xarray().amplitude) / m.a_short for m in solutions1]
)
k_modulation_numerical_stokes = np.array(
    [np.max(m.to_xarray().wavenumber) / m.k_short for m in solutions2]
)
a_modulation_numerical_stokes = np.array(
    [np.max(m.to_xarray().amplitude) / m.a_short for m in solutions2]
)
ak_modulation_numerical_linear = np.array(
    [
        np.max(m.to_xarray().amplitude * m.to_xarray().wavenumber)
        / m.a_short
        / m.k_short
        for m in solutions1
    ]
)

ak_modulation_numerical_stokes = np.array(
    [
        np.max(m.to_xarray().amplitude * m.to_xarray().wavenumber)
        / m.a_short
        / m.k_short
        for m in solutions2
    ]
)

phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
a_modulation_analytical = []
k_modulation_analytical = []
ak_modulation_analytical = []
for ak in ak_range:
    a_long = ak / k_long
    kmod = np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
    Nmod = kmod
    gmod = g_modulation(phase, a_long, k_long)
    amod = gmod ** (-0.25) * kmod**0.25 * Nmod**0.5
    a_modulation_analytical.append(np.max(amod))
    k_modulation_analytical.append(np.max(kmod))
    ak_modulation_analytical.append(np.max(amod * kmod))
a_modulation_analytical = np.array(a_modulation_analytical)
k_modulation_analytical = np.array(k_modulation_analytical)
ak_modulation_analytical = np.array(ak_modulation_analytical)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ax.plot(ak_range, (1 + ak_range), "k-", lw=2, label="Amplitude, L-HS 1960")
ax.plot(
    ak_range,
    a_modulation_analytical,
    linestyle="-",
    lw=2,
    color="tab:blue",
    label="Amplitude, analytical",
)
ax.plot(
    ak_range,
    a_modulation_numerical_linear,
    linestyle="-",
    lw=2,
    color="tab:orange",
    label="Amplitude, numerical",
)
ax.plot(
    ak_range,
    a_modulation_numerical_stokes,
    linestyle="-",
    lw=2,
    color="tab:green",
    label="Amplitude, numerical",
)

ax.plot(ak_range, (1 + ak_range) ** 2, "k--", lw=2, label="Steepness, L-HS 1960")
ax.plot(
    ak_range,
    ak_modulation_analytical,
    linestyle="--",
    lw=2,
    color="tab:blue",
    label="Steepness, analytical",
)
ax.plot(
    ak_range,
    ak_modulation_numerical_linear,
    linestyle="--",
    lw=2,
    color="tab:orange",
    label="Steepness, numerical",
)
ax.plot(
    ak_range,
    ak_modulation_numerical_stokes,
    linestyle="--",
    lw=2,
    color="tab:green",
    label="Steepness, numerical",
)


ax.legend()
ax.set_xlabel(r"$\varepsilon_L$")
ax.set_ylabel(r"Amplitude and steepness modulation")
ax.grid()
ax.set_xlim(0, 0.4)
ax.set_ylim(1, 7)

plt.savefig("../figures/numerical_vs_analytical_solutions.png", dpi=200)
plt.close()
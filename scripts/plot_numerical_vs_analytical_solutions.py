import matplotlib.pyplot as plt
from twowave import WaveModulationModel, diff
import numpy as np
import pandas as pd
import matplotlib
import multiprocessing as mp
import xarray as xr
import os

matplotlib.rc("font", size=14)


def process_ak(a_long: float, k_long: float, wave_type: str):
    filename = f"data/model_{wave_type}_ak_{a_long:.2f}.nc"
    if os.path.exists(filename):
        return xr.open_dataset(filename)
    m = WaveModulationModel(a_long=a_long, k_long=k_long)
    m.run(wave_type=wave_type, ramp_type="groups")
    ds = m.to_xarray()
    ds["a_long"] = a_long
    ds["k_long"] = k_long
    ds["a_short"] = m.a_short
    ds["k_short"] = m.k_short
    ds.to_netcdf(filename)
    return ds


def g_modulation(psi, a, k, Cg_short):
    eta = a * np.cos(psi)
    g0 = 9.8
    omega = np.sqrt(g0 * k)
    Cp = omega / k
    U = a * omega * np.cos(psi) * np.exp(k * eta)
    W = a * omega * np.sin(psi) * np.exp(k * eta)
    dx = np.diff(psi / k)[0]
    a_z = -diff(W) / dx * (Cp - U - Cg_short)
    gmod = (g0 + a_z) / g0
    return gmod


k_long = 1

if not os.path.exists("data"):
    os.makedirs("data")

# Run in parallel using multiprocessing
ak_values = np.arange(0.01, 0.41, 0.01)
print("Running linear solutions")
with mp.Pool(processes=mp.cpu_count()) as pool:
    solutions1 = pool.starmap(process_ak, [(ak, k_long, "linear") for ak in ak_values])

print("Running Stokes solutions")
with mp.Pool(processes=mp.cpu_count()) as pool:
    solutions2 = pool.starmap(process_ak, [(ak, k_long, "stokes") for ak in ak_values])

print("Running nonlinear solutions")
with mp.Pool(processes=mp.cpu_count()) as pool:
    solutions3 = pool.starmap(
        process_ak, [(ak, k_long, "nonlinear") for ak in ak_values]
    )


k_modulation_numerical_linear = np.array(
    [np.max(ds.wavenumber) / ds.k_short for ds in solutions1]
)
a_modulation_numerical_linear = np.array(
    [np.max(ds.amplitude) / ds.a_short for ds in solutions1]
)
k_modulation_numerical_stokes = np.array(
    [np.max(ds.wavenumber) / ds.k_short for ds in solutions2]
)
a_modulation_numerical_stokes = np.array(
    [np.max(ds.amplitude) / ds.a_short for ds in solutions2]
)
ak_modulation_numerical_linear = np.array(
    [
        np.max(ds.amplitude * ds.wavenumber) / ds.a_short / ds.k_short
        for ds in solutions1
    ]
)

ak_modulation_numerical_stokes = np.array(
    [
        np.max(ds.amplitude * ds.wavenumber) / ds.a_short / ds.k_short
        for ds in solutions2
    ]
)

k_modulation_numerical_nonlinear = np.array(
    [np.max(ds.wavenumber) / ds.k_short for ds in solutions3]
)
a_modulation_numerical_nonlinear = np.array(
    [np.max(ds.amplitude) / ds.a_short for ds in solutions3]
)
ak_modulation_numerical_nonlinear = np.array(
    [
        np.max(ds.amplitude * ds.wavenumber) / ds.a_short / ds.k_short
        for ds in solutions3
    ]
)

k_short = 10
g0 = 9.8
omega_short = np.sqrt(g0 * k_short)
Cg_short = omega_short / k_short / 2

phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
a_modulation_analytical = []
k_modulation_analytical = []
ak_modulation_analytical = []
for ak in ak_values:
    a_long = ak / k_long
    kmod = np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
    Nmod = kmod
    gmod = g_modulation(phase, a_long, k_long, Cg_short)
    amod = gmod ** (-0.25) * kmod**0.25 * Nmod**0.5
    a_modulation_analytical.append(np.max(amod))
    k_modulation_analytical.append(np.max(kmod))
    ak_modulation_analytical.append(np.max(amod * kmod))
a_modulation_analytical = np.array(a_modulation_analytical)
k_modulation_analytical = np.array(k_modulation_analytical)
ak_modulation_analytical = np.array(ak_modulation_analytical)


df_k_lh1987 = pd.read_csv("../data/lh1987_k_modulation_r10.csv")
df_ak_lh1987 = pd.read_csv("../data/lh1987_ak_modulation_r10.csv")


labels = [
    "Longuet-Higgins & Stewart (1960), analytical",
    "Longuet-Higgins (1987), numerical",
    "This paper, analytical",
    r"This paper, numerical, 1$^{st}$ order (linear)",
    r"This paper, numerical, 3$^{rd}$ order Stokes",
    "This paper, numerical, fully nonlinear",
]


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(ak_values, (1 + ak_values), "k-", lw=2, label=labels[0])
ax.plot(
    df_k_lh1987.epsilon,
    df_k_lh1987.k_modulation,
    "ko",
    ms=8,
    label=labels[1],
    zorder=10,
)
ax.plot(
    ak_values,
    k_modulation_analytical,
    linestyle="-",
    lw=2,
    color="tab:blue",
    label=labels[2],
)
ax.plot(
    ak_values,
    k_modulation_numerical_linear,
    linestyle="-",
    lw=2,
    color="tab:green",
    label=labels[3],
)
ax.plot(
    ak_values,
    k_modulation_numerical_stokes,
    linestyle="-",
    lw=2,
    color="tab:orange",
    label=labels[4],
)
ax.plot(
    ak_values,
    k_modulation_numerical_nonlinear,
    linestyle="-",
    lw=2,
    color="tab:red",
    label=labels[5],
)
ax.legend(loc="upper left")
ax.set_xlabel(r"$\varepsilon_L$")
ax.set_ylabel(r"$\widetilde{k}/k$")
ax.grid()
ax.set_xlim(0, 0.405)
ax.set_ylim(1, 3.5)
plt.tight_layout()
plt.savefig("../figures/numerical_vs_analytical_solutions_k.pdf", dpi=200)
plt.close()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(ak_values, (1 + ak_values), "k-", lw=2, label=labels[0])
ax.plot(
    df_ak_lh1987.epsilon,
    df_ak_lh1987.ak_modulation / df_k_lh1987.k_modulation,
    "ko",
    ms=8,
    label=labels[1],
    zorder=10,
)
ax.plot(
    ak_values,
    a_modulation_analytical,
    linestyle="-",
    lw=2,
    color="tab:blue",
    label=labels[2],
)
ax.plot(
    ak_values,
    a_modulation_numerical_linear,
    linestyle="-",
    lw=2,
    color="tab:green",
    label=labels[3],
)
ax.plot(
    ak_values,
    a_modulation_numerical_stokes,
    linestyle="-",
    lw=2,
    color="tab:orange",
    label=labels[4],
)
ax.plot(
    ak_values,
    a_modulation_numerical_nonlinear,
    linestyle="-",
    lw=2,
    color="tab:red",
    label=labels[5],
)
ax.legend(loc="upper left")
ax.set_xlabel(r"$\varepsilon_L$")
ax.set_ylabel(r"$\widetilde{a}/a$")
ax.grid()
ax.set_xlim(0, 0.405)
ax.set_ylim(1, 3)
plt.tight_layout()
plt.savefig("../figures/numerical_vs_analytical_solutions_a.pdf")
plt.close()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(ak_values, (1 + ak_values) ** 2, "k-", lw=2, label=labels[0])
ax.plot(
    df_ak_lh1987.epsilon,
    df_ak_lh1987.ak_modulation,
    "ko",
    ms=8,
    label=labels[1],
    zorder=10,
)
ax.plot(
    ak_values,
    ak_modulation_analytical,
    linestyle="-",
    lw=2,
    color="tab:blue",
    label=labels[2],
)
ax.plot(
    ak_values,
    ak_modulation_numerical_linear,
    linestyle="-",
    lw=2,
    color="tab:green",
    label=labels[3],
)
ax.plot(
    ak_values,
    ak_modulation_numerical_stokes,
    linestyle="-",
    lw=2,
    color="tab:orange",
    label=labels[4],
)
ax.plot(
    ak_values,
    ak_modulation_numerical_nonlinear,
    linestyle="-",
    lw=2,
    color="tab:red",
    label=labels[5],
)
ax.legend(loc="upper left")
ax.set_xlabel(r"$\varepsilon_L$")
ax.set_ylabel(r"$\widetilde{ak}/(ak)$")
ax.grid()
ax.set_xlim(0, 0.405)
ax.set_ylim(1, 8)
plt.tight_layout()
plt.savefig("../figures/numerical_vs_analytical_solutions_ak.pdf")
plt.close()

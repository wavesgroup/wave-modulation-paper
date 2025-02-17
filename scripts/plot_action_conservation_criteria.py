import matplotlib.pyplot as plt
import numpy as np
from twowave import WaveModulationModel, diff

import matplotlib

matplotlib.rcParams["font.size"] = 14


def get_homogeneity_and_stationarity(a_long, k_long, k_short):
    m = WaveModulationModel(a_long=a_long, k_long=k_long, k_short=k_short)
    m.run(wave_type="linear", ramp_type="groups")
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


def homogeneity_k(k0, k_L, ak, phase):
    return 1 - np.abs(k_L / k0 * ak * np.sin(phase) / (1 + ak * np.cos(phase)) ** 2)


def homogeneity_g(k0, k_L, ak, phase):
    return 1 - np.abs(k_L / k0 * ak * np.sin(phase) / (1 - ak**2 * np.cos(phase) ** 2))


def stationarity_k(sigma0, sigma_L, ak, phase):
    return 1 - np.abs(
        sigma_L
        / sigma0
        * ak
        * np.sin(phase)
        / (1 + ak * np.cos(phase))
        / np.sqrt(1 - ak**2 * np.cos(phase) ** 2)
    )


def stationarity_g(sigma0, sigma_L, ak, phase):
    return 1 - np.abs(
        sigma_L
        / sigma0
        * ak
        * np.sin(phase)
        / (1 - ak * np.cos(phase))
        / np.sqrt(1 - ak**2 * np.cos(phase) ** 2)
    )


# Numerical model results
ak_values = np.arange(0, 0.42, 0.02)
r_values = np.logspace(1, 2, 10)
k_L = 1
H_k_values = np.zeros((len(r_values), len(ak_values)))
H_g_values = np.zeros((len(r_values), len(ak_values)))
H_N_values = np.zeros((len(r_values), len(ak_values)))

S_k_values = np.zeros((len(r_values), len(ak_values)))
S_g_values = np.zeros((len(r_values), len(ak_values)))
S_N_values = np.zeros((len(r_values), len(ak_values)))

for i, r in enumerate(r_values):
    for j, ak in enumerate(ak_values):
        print(
            f"Processing r={r:.2f}, ak={ak:.2f} ({i + 1}/{len(r_values)}, {j + 1}/{len(ak_values)})"
        )
        (
            H_k_values[i, j],
            H_g_values[i, j],
            H_N_values[i, j],
            S_k_values[i, j],
            S_g_values[i, j],
            S_N_values[i, j],
        ) = get_homogeneity_and_stationarity(ak, k_long=k_L, k_short=k_L * r)

# Clip stationarity values to minimum of 0.5
S_k_values = np.clip(S_k_values, 0.5, None)
S_g_values = np.clip(S_g_values, 0.5, None)
S_N_values = np.clip(S_N_values, 0.5, None)


# Create 6-panel figure for numerical results
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# Common plotting parameters
cmap = plt.cm.viridis

# Plot homogeneity (left column)
H_levels = np.arange(0.9, 1.01, 0.01)
c1 = ax1.contourf(ak_values, r_values, H_k_values, levels=H_levels, cmap=cmap)
c3 = ax3.contourf(ak_values, r_values, H_N_values, levels=H_levels, cmap=cmap)
c5 = ax5.contourf(ak_values, r_values, H_g_values, levels=H_levels, cmap=cmap)

ax1.contour(
    ak_values, r_values, H_k_values, levels=H_levels, colors="black", linewidths=0.2
)
ax3.contour(
    ak_values, r_values, H_N_values, levels=H_levels, colors="black", linewidths=0.2
)
ax5.contour(
    ak_values, r_values, H_g_values, levels=H_levels, colors="black", linewidths=0.2
)

plt.colorbar(c1, ax=ax1)
plt.colorbar(c3, ax=ax3)
plt.colorbar(c5, ax=ax5)

# Plot stationarity (right column)
S_levels = np.arange(0.6, 1.05, 0.05)
c2 = ax2.contourf(ak_values, r_values, S_k_values, levels=S_levels, cmap=cmap)
c4 = ax4.contourf(ak_values, r_values, S_N_values, levels=S_levels, cmap=cmap)
c6 = ax6.contourf(ak_values, r_values, S_g_values, levels=S_levels, cmap=cmap)

ax2.contour(
    ak_values, r_values, S_k_values, levels=S_levels, colors="black", linewidths=0.2
)
ax4.contour(
    ak_values, r_values, S_N_values, levels=S_levels, colors="black", linewidths=0.2
)
ax6.contour(
    ak_values, r_values, S_g_values, levels=S_levels, colors="black", linewidths=0.2
)

plt.colorbar(c2, ax=ax2)
plt.colorbar(c4, ax=ax4)
plt.colorbar(c6, ax=ax6)

# Add 0.99 and 0.90 contour lines
for ax, values in [(ax1, H_k_values), (ax3, H_N_values), (ax5, H_g_values)]:
    ax.contour(
        ak_values,
        r_values,
        values,
        levels=[0.99],
        colors="black",
        linewidths=2,
        linestyles="solid",
    )
    ax.contour(
        ak_values,
        r_values,
        values,
        levels=[0.90],
        colors="black",
        linewidths=2,
        linestyles="dashed",
    )

for ax, values in [(ax2, S_k_values), (ax4, S_N_values), (ax6, S_g_values)]:
    ax.contour(
        ak_values,
        r_values,
        values,
        levels=[0.99],
        colors="black",
        linewidths=2,
        linestyles="solid",
    )
    ax.contour(
        ak_values,
        r_values,
        values,
        levels=[0.90],
        colors="black",
        linewidths=2,
        linestyles="dashed",
    )

# Common axis properties
for ax in axes:
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_yticks([10, 20, 30, 40, 60, 100])
    ax.set_yticklabels([10, 20, 30, 40, 60, 100])

# Add labels
for ax in [ax1, ax3, ax5]:
    ax.set_ylabel(r"$k/k_L$")

for ax in [ax5, ax6]:
    ax.set_xlabel(r"$\varepsilon_L$")

# Add titles
ax1.set_title(r"$H_k$")
ax2.set_title(r"$S_k$")
ax3.set_title(r"$H_N$")
ax4.set_title(r"$S_N$")
ax5.set_title(r"$H_g$")
ax6.set_title(r"$S_g$")

# Save figure
fig.savefig(
    "../figures/fig_homogeneity_stationarity_numerical.pdf", bbox_inches="tight"
)
plt.close()


# Analytical results
phase = np.linspace(0, 2 * np.pi, 1000)
ak_values = np.arange(0, 0.41, 0.01)
r_values = np.logspace(1, 2, 100)

g0 = 9.8
k_L = 0.1
sigma_L = np.sqrt(g0 * k_L)
C_p = sigma_L / k_L

# Initialize 2D arrays
H_k_values = np.zeros((len(r_values), len(ak_values)))
H_g_values = np.zeros((len(r_values), len(ak_values)))
S_k_values = np.zeros((len(r_values), len(ak_values)))
S_g_values = np.zeros((len(r_values), len(ak_values)))

for i, r in enumerate(r_values):
    k0 = k_L * r
    sigma0 = np.sqrt(g0 * k0)

    for j, ak in enumerate(ak_values):
        H_k = homogeneity_k(k0, k_L, ak, phase)
        H_g = homogeneity_g(k0, k_L, ak, phase)
        S_k = stationarity_k(sigma0, sigma_L, ak, phase)
        S_g = stationarity_g(sigma0, sigma_L, ak, phase)

        H_k_values[i, j] = np.min(H_k)
        H_g_values[i, j] = np.min(H_g)
        S_k_values[i, j] = np.min(S_k)
        S_g_values[i, j] = np.min(S_g)

# Create 4-panel figure for contour plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
axes = [ax1, ax2, ax3, ax4]

# Common plotting parameters
levels = np.linspace(0.85, 1.0, 16)
cmap = plt.cm.viridis

# Plot homogeneity and stationarity contours
c1 = ax1.contourf(ak_values, r_values, H_k_values, levels=levels, cmap=cmap)
c2 = ax2.contourf(ak_values, r_values, S_k_values, levels=levels, cmap=cmap)
c3 = ax3.contourf(ak_values, r_values, H_g_values, levels=levels, cmap=cmap)
c4 = ax4.contourf(ak_values, r_values, S_g_values, levels=levels, cmap=cmap)

# Add contour lines at each level
ax1.contour(
    ak_values, r_values, H_k_values, levels=levels, colors="black", linewidths=0.2
)
ax2.contour(
    ak_values, r_values, S_k_values, levels=levels, colors="black", linewidths=0.2
)
ax3.contour(
    ak_values, r_values, H_g_values, levels=levels, colors="black", linewidths=0.2
)
ax4.contour(
    ak_values, r_values, S_g_values, levels=levels, colors="black", linewidths=0.2
)

# Add colorbars
plt.colorbar(c1, ax=ax1)
plt.colorbar(c2, ax=ax2)
plt.colorbar(c3, ax=ax3)
plt.colorbar(c4, ax=ax4)

# Add black contours at 0.99 and 0.90 levels
cs1 = ax1.contour(
    ak_values, r_values, H_k_values, levels=[0.99], colors="black", linewidths=2
)
cs2 = ax2.contour(
    ak_values, r_values, S_k_values, levels=[0.99], colors="black", linewidths=2
)
cs3 = ax3.contour(
    ak_values, r_values, H_g_values, levels=[0.99], colors="black", linewidths=2
)
cs4 = ax4.contour(
    ak_values, r_values, S_g_values, levels=[0.99], colors="black", linewidths=2
)
cs1 = ax1.contour(
    ak_values,
    r_values,
    H_k_values,
    levels=[0.90],
    colors="black",
    linewidths=2,
    linestyles="dashed",
)
cs2 = ax2.contour(
    ak_values,
    r_values,
    S_k_values,
    levels=[0.90],
    colors="black",
    linewidths=2,
    linestyles="dashed",
)
cs3 = ax3.contour(
    ak_values,
    r_values,
    H_g_values,
    levels=[0.90],
    colors="black",
    linewidths=2,
    linestyles="dashed",
)
cs4 = ax4.contour(
    ak_values,
    r_values,
    S_g_values,
    levels=[0.90],
    colors="black",
    linewidths=2,
    linestyles="dashed",
)

# Set y-axis to log scale for all subplots
for ax in axes:
    ax.set_yscale("log")
    ax.grid(True)
    ax.set_yticks([10, 20, 30, 40, 60, 100])
    ax.set_yticklabels([10, 20, 30, 40, 60, 100])


# Labels and titles
ax1.set_title(r"$H_k$")
ax2.set_title(r"$S_k$")
ax3.set_title(r"$H_g$")
ax4.set_title(r"$S_g$")

ax1.set_ylabel(r"$k/k_L$")
ax3.set_ylabel(r"$k/k_L$")
ax3.set_xlabel(r"$\varepsilon_L$")
ax4.set_xlabel(r"$\varepsilon_L$")

# Save figure
fig.savefig(
    "../figures/fig_homogeneity_stationarity_analytical.pdf", bbox_inches="tight"
)
plt.close()

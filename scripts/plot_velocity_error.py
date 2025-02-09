import matplotlib.pyplot as plt
import numpy as np
from ssgw import SSGW
from twowave import elevation, orbital_horizontal_velocity

import matplotlib

matplotlib.rcParams.update({"font.size": 14})

g = 9.8
k = 1
a = 0.4
ak = a * k
d = np.inf
omega = np.sqrt(g * k * np.tanh(k * d))
t = 0

# Fully nonlinear solution
wave = SSGW(d, ak)

phase, eta = np.real(wave.zs), np.imag(wave.zs)
x = phase / k
U = (np.real(wave.ws) + wave.ce) * np.sqrt(g)

eta_linear = elevation(x, t, a, k, omega, wave_type="linear")
eta_stokes = elevation(x, t, a, k, omega, wave_type="stokes")

U_linear = orbital_horizontal_velocity(x, eta_linear, t, a, k, omega)
U_stokes = orbital_horizontal_velocity(x, eta_stokes, t, a, k, omega)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# Top panel: Surface elevation
ax1.plot(phase, eta_linear, label="Linear")
ax1.plot(phase, eta_stokes, label="3$^{rd}$ order Stokes")
ax1.plot(phase, eta, color="k", label="Fully nonlinear")
ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
ax1.set_ylabel("Surface elevation (m)")
ax1.legend()
ax1.grid(True)

# Middle panel: Velocities
ax2.plot(phase, U_linear, label="Linear")
ax2.plot(phase, U_stokes, label="3$^{rd}$ order Stokes")
ax2.plot(phase, U, color="k", label="Fully nonlinear")
ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
ax2.set_ylabel("Horizontal velocity (m/s)")
ax2.grid(True)

# Middle panel: Absolute errors
error_linear = U_linear - U
error_stokes = U_stokes - U

ax3.plot(phase, error_linear, label="Linear")
ax3.plot(phase, error_stokes, label="3$^{rd}$ order Stokes")
ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)
ax3.set_ylabel("Velocity Error (m/s)")
ax3.legend()
ax3.grid(True)
ax3.set_xlabel("Phase")

# Set x-axis limits and ticks for all subplots
xticks = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
xticklabels = ["0", "π/2", "π", "3π/2", "2π"]
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

ax1.set_title(r"$\varepsilon_L = {:.1f}$".format(ak))

plt.savefig("../figures/fig_velocity_error_by_phase.pdf")
plt.close()

a_omega = []

# Compute errors across steepness values
ak_values = np.arange(0, 0.41, 0.01)
max_error_linear = []
max_error_stokes = []

for ak in ak_values:
    a = ak / k
    omega = np.sqrt(g * k * np.tanh(k * d))

    a_omega.append(a * omega)

    # Compute SSGW solution
    wave = SSGW(d, ak)
    phase, eta = np.real(wave.zs), np.imag(wave.zs)
    U = (np.real(wave.ws) + wave.ce) * np.sqrt(g)
    x = phase / k

    eta_linear = elevation(x, t, a, k, omega, wave_type="linear")
    eta_stokes = elevation(x, t, a, k, omega, wave_type="stokes")

    # Compute linear and Stokes solutions
    U_linear = orbital_horizontal_velocity(x, eta_linear, t, a, k, omega)
    U_stokes = orbital_horizontal_velocity(x, eta_stokes, t, a, k, omega)

    # Compute max errors
    error_linear = np.max(np.abs(U_linear - U))
    error_stokes = np.max(np.abs(U_stokes - U))

    max_error_linear.append(error_linear)
    max_error_stokes.append(error_stokes)


# Plot errors vs steepness
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.semilogy(
    ak_values,
    np.array(max_error_linear) / a_omega,
    color="tab:blue",
    lw=2,
    label="Linear wave",
)
ax.semilogy(
    ak_values,
    np.array(max_error_stokes) / a_omega,
    color="tab:orange",
    lw=2,
    label=r"3$^{rd}$ order Stokes wave",
)
ax.set_xlabel("Wave steepness (ak)")
ax.set_ylabel("Surface velocity relative error")
ax.legend()
ax.set_xlim(0, 0.4)
ax.grid()

plt.savefig("../figures/fig_relative_velocity_error.pdf")
plt.close()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib


matplotlib.rc("font", size=16)


def plot_fig_frequency_modulation():
    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    g = 9.8

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for ak_L in [0.1, 0.2, 0.3, 0.4]:
        term1 = -ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase))
        term2 = -((ak_L * np.sin(phase)) ** 2) * np.exp(ak_L * np.cos(phase))
        g_rel = 1 + term1 + term2
        k_rel = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
        omega_rel = np.sqrt(g_rel * k_rel)
        plt.plot(
            phase, omega_rel, marker="", linestyle="-", label=r"$a_L k_L=%.1f$" % ak_L
        )

    plt.legend()
    plt.xlabel("Phase [rad]")
    plt.ylabel(r"$\widetilde \omega/\omega$")
    plt.grid()
    plt.plot([0, 2 * np.pi], [1, 1], "k--")
    plt.xlim(0, 2 * np.pi)
    plt.xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    plt.savefig("../figures/fig_frequency_modulation.png", dpi=200)
    plt.close(fig)


def plot_fig_phase_speed_modulation():
    phase = np.arange(0, 2 * np.pi + 1e-3, 1e-3)

    g0 = 9.8
    k0 = 10
    omega0 = np.sqrt(g0 * k0)
    cp0 = omega0 / k0

    k_L = 1
    omega_L = np.sqrt(g0 * k_L)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for n, ak_L in enumerate([0.1, 0.2, 0.3, 0.4]):
        a_L = ak_L / k_L
        eta = a_L * np.cos(phase)
        u_L = omega_L * eta * np.exp(k_L * eta)
        cp0_tot = cp0 + u_L

        term1 = -ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase))
        term2 = -((ak_L * np.sin(phase)) ** 2) * np.exp(ak_L * np.cos(phase))
        g_rel = 1 + term1 + term2
        g = g0 * g_rel

        k = k0 * np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
        omega = np.sqrt(g * k)
        cp = omega / k
        cp_tot = cp + u_L

        plt.plot(
            phase,
            cp_tot / cp0_tot,
            marker="",
            linestyle="-",
            label=r"$a_L k_L=%.1f$" % ak_L,
            color=colors[n],
        )
        plt.plot(phase, cp / cp0, marker="", linestyle="--")

    plt.legend()
    plt.xlabel("Phase [rad]")
    plt.ylabel(r"$\widetilde C_p/C_p$")
    plt.grid()
    plt.plot([0, 2 * np.pi], [1, 1], "k--")
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0.4, 3)
    plt.xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    plt.savefig("../figures/fig_phase_speed_modulation.png", dpi=200)
    plt.close(fig)


def plot_fig_wind_input_modulation():
    phase = np.arange(0, 2 * np.pi + 1e-3, 1e-3)

    g0 = 9.8
    k0 = np.linspace(10, 100, 91, endpoint=True)
    omega0 = np.sqrt(g0 * k0)
    cp0 = omega0 / k0

    k_L = 1
    omega_L = np.sqrt(g0 * k_L)
    U = 5
    Cd = 1e-3
    ustar = U * np.sqrt(Cd)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for n, ak_L in enumerate([0.1, 0.2, 0.3, 0.4]):
        a_L = ak_L / k_L
        eta = a_L * np.cos(phase)
        u_L = omega_L * eta * np.exp(k_L * eta)
        cp0_tot = cp0 + np.mean(u_L)

        term1 = -ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase))
        term2 = -((ak_L * np.sin(phase)) ** 2) * np.exp(ak_L * np.cos(phase))
        g_mod = 1 + term1 + term2
        g = g0 * g_mod

        k_mod = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
        omega_mod = np.sqrt(g_mod * k_mod)
        cp0_mod = omega_mod / k_mod

        cp_tot_mean = cp0 * np.mean(cp0_mod) + np.mean(u_L)
        cp_tot_crest = cp0 * np.min(cp0_mod) + np.max(u_L)

        wind_input0 = (ustar / cp0) ** 2
        wind_input_mean = (ustar / cp_tot_mean) ** 2
        wind_input_crest = (ustar / cp_tot_crest) ** 2

        plt.plot(
            omega0 / 2 / np.pi,
            wind_input_mean / wind_input0,
            marker="",
            linestyle="-",
            lw=2,
            label=r"$a_L k_L=%.1f$" % ak_L,
            color=colors[n],
        )
        plt.plot(
            omega0 / 2 / np.pi,
            wind_input_crest / wind_input0,
            lw=2,
            marker="",
            linestyle="--",
        )

    plt.legend()
    plt.xlabel("Short wave frequency [Hz]")
    plt.ylabel(r"$(u_*/C_p)^2$ modulation")
    plt.grid()
    plt.plot([k0[0], k0[-1]], [1, 1], "k--")
    plt.xlim(1.5, 5)
    plt.ylim(0, 1)
    plt.title("Solid: Phase-averaged; dashed: at the crest")
    plt.savefig("../figures/fig_wind_input_modulation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig_frequency_modulation()
    plot_fig_phase_speed_modulation()
    plot_fig_wind_input_modulation()

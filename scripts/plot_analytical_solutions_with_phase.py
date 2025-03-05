import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from twowave import gravity

matplotlib.rc("font", size=12)


def plot_analytical_solutions(a_L: float):
    phase = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
    k_L = 1
    ak_L = a_L * k_L
    x = phase / k_L
    t = 0
    g0 = 9.8
    omega = np.sqrt(g0 * k_L)
    k_short = 10
    omega_short = np.sqrt(g0 * k_short)
    cg_short = 0.5 * omega_short / k_short

    k_lhs = 1 + ak_L * np.cos(phase)
    k_modulation = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
    g_modulation = gravity(x, t, a_L, k_L, omega, g0, cg_short, "linear") / g0
    N_modulation = k_modulation
    a_modulation = g_modulation ** (-0.25) * k_modulation**0.25 * N_modulation**0.5
    a_lhs = k_lhs

    omega_lhs = np.sqrt(k_lhs)
    omega_modulation = np.sqrt(g_modulation * k_modulation)

    Cp_lhs = omega_lhs / k_lhs
    Cp_modulation = omega_modulation / k_modulation

    fig = plt.figure(figsize=(8, 10))
    axes = [
        plt.subplot2grid((3, 2), (0, 0)),
        plt.subplot2grid((3, 2), (0, 1)),
        plt.subplot2grid((3, 2), (1, 0)),
        plt.subplot2grid((3, 2), (1, 1)),
        plt.subplot2grid((3, 2), (2, 0)),
        plt.subplot2grid((3, 2), (2, 1)),
    ]

    axes[0].plot(phase, k_lhs, "k-", lw=2, label="L-HS 1960")
    axes[0].plot(
        phase,
        k_modulation,
        marker="",
        linestyle="-",
        color="tab:blue",
        lw=2,
        label="This paper",
    )
    axes[0].legend()

    axes[1].plot(phase, np.ones(phase.shape), "k-", lw=2)
    axes[1].plot(phase, g_modulation, marker="", linestyle="-", color="tab:blue", lw=2)

    axes[2].plot(phase, a_lhs, "k-", lw=2)
    axes[2].plot(phase, a_modulation, marker="", linestyle="-", color="tab:blue", lw=2)

    axes[3].plot(phase, a_lhs * k_lhs, "k-", lw=2)
    axes[3].plot(
        phase,
        a_modulation * k_modulation,
        marker="",
        linestyle="-",
        color="tab:blue",
        lw=2,
    )

    axes[4].plot(phase, omega_lhs, "k-", lw=2)
    axes[4].plot(
        phase, omega_modulation, marker="", linestyle="-", color="tab:blue", lw=2
    )

    axes[5].plot(phase, Cp_lhs, "k-", lw=2)
    axes[5].plot(phase, Cp_modulation, marker="", linestyle="-", color="tab:blue", lw=2)

    for ax in axes:
        ax.grid()
        ax.plot([0, 2 * np.pi], [1, 1], "k--")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
        ax.set_xticklabels([])

    for ax in axes[-2:]:
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlabel(r"$\psi$")

    axes[0].set_title(r"$\widetilde{k}/k$")
    axes[1].set_title(r"$\widetilde{g}/g$")
    axes[2].set_title(r"$\widetilde{a}/a$")
    axes[3].set_title(r"$\widetilde{ak}/(ak)$")
    axes[4].set_title(r"$\widetilde{\sigma}/\sigma$")
    axes[5].set_title(r"$\widetilde{C_p}/C_p$")

    for n, ax in enumerate(axes):
        ax.text(
            0,
            1.02,
            f"({chr(97 + n)})",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            fontsize=18,
        )

    fig.suptitle(r"$\varepsilon_L$" + f" = {a_L * k_L}", fontsize=18)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(f"../figures/fig_analytical_solutions_ak{a_L * k_L}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_analytical_solutions(0.1)
    # plot_analytical_solutions(0.2)
    # plot_analytical_solutions(0.3)
    # plot_analytical_solutions(0.4)

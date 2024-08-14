import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rc("font", size=12)


def elevation_linear(a: float, phase: float) -> float:
    return a * np.cos(phase)


def elevation_stokes(a: float, k: float, phase: float) -> float:
    term1 = np.cos(phase)
    term2 = 0.5 * a * k * np.cos(2 * phase)
    term3 = (a * k) ** 2 * (3 / 8 * np.cos(3 * phase) - 1 / 16 * np.cos(phase))
    return a * (term1 + term2 + term3)


def gravity_linear(
    a: float,
    k: float,
    phase: float,
    g0: float,
) -> float:
    """Gravitational acceleration at the surface of a linear wave.
    """
    eta = elevation_linear(a, phase)
    return g0 * (1 - a * k * np.exp(k * eta) * (np.cos(phase) - a * k * np.sin(phase) ** 2))


def gravity_stokes(
    a: float, k: float, phase: float, g0: float
) -> float:
    """Gravitational acceleration at the surface of a Stokes wave.
    """
    eta = elevation_stokes(a, k, phase)
    res = g0 * (1 - a * k
        * (np.cos(phase) - a * k * np.sin(phase)
            * (
                (1 - 1 / 16 * (a * k) ** 2) * np.sin(phase)
                + a * k * np.sin(2 * phase)
                + 9 / 8 * (a * k) ** 2 * np.sin(3 * phase)
            )
        ) * np.exp(k * eta)
    )
    return res



def plot_analytical_solutions(a_L: float):

    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    k_L = 1
    ak_L = a_L * k_L

    k_lhs = 1 + ak_L * np.cos(phase)
    k_modulation = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))

    g_modulation = gravity_linear(a_L, k_L, phase, 1)
    #g_modulation_stokes = gravity_stokes(a_L, k_L, phase, 1)

    N_modulation = k_modulation
    a_modulation = g_modulation**(-0.25) * k_modulation**0.25 * N_modulation**0.5
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
    axes[0].plot(phase, k_modulation, "r-", lw=2, label="This paper")
    axes[0].legend()

    axes[1].plot(phase, np.ones(phase.shape), "k-", lw=2)
    axes[1].plot(phase, g_modulation, "r-", lw=2)

    axes[2].plot(phase, a_lhs, "k-", lw=2)
    axes[2].plot(phase, a_modulation, "r-", lw=2)

    axes[3].plot(phase, a_lhs * k_lhs, "k-", lw=2)
    axes[3].plot(phase, a_modulation * k_modulation, "r-", lw=2)

    axes[4].plot(phase, omega_lhs, "k-", lw=2)
    axes[4].plot(phase, omega_modulation, "r-", lw=2)

    axes[5].plot(phase, Cp_lhs, "k-", lw=2)
    axes[5].plot(phase, Cp_modulation, "r-", lw=2)

    for ax in axes:
        ax.grid()
        ax.plot([0, 2 * np.pi], [1, 1], "k--")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
        ax.set_xticklabels([])

    for ax in axes[-2:]:
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    axes[0].set_title(r"$\widetilde{k}/k$")
    axes[1].set_title(r"$\widetilde{g}/g$")
    axes[2].set_title(r"$\widetilde{a}/a$")
    axes[3].set_title(r"$\widetilde{ak}/(ak)$")
    axes[4].set_title(r"$\widetilde{\omega}/\omega$")
    axes[5].set_title(r"$\widetilde{C_p}/C_p$")

    for n, ax in enumerate(axes):
        ax.text(
            0, 1.02, f"({chr(97 + n)})", ha="left", va="bottom", transform=ax.transAxes, fontsize=18
        )

    fig.suptitle(r"$\varepsilon_L$" + f" = {a_L * k_L}", fontsize=18)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(f"../figures/fig_analytical_solutions_ak{a_L * k_L}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_analytical_solutions(0.1)
    plot_analytical_solutions(0.2)
    #plot_analytical_solutions(0.3)
    plot_analytical_solutions(0.4)
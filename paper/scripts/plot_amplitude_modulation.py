import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rc("font", size=16)


def g_eulerian(g, psi, a, k, order=3):
    assert order in [1, 2, 3]
    eta = a * np.cos(psi)
    if order == 1:
        res = g * (1 - a * k * np.cos(psi))
    elif order == 2:
        res = g * (1 - a * k * np.cos(psi) * np.exp(k * eta))
    else:
        res = g * (
            1 - a * k * np.exp(k * eta) * (np.cos(psi) - a * k * np.sin(psi) ** 2)
        )
    return res


def plot_amplitude_modulation():
    """Plot the apparent gravity figure."""

    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    a_L = 0.1
    k_L = 1
    ak = a_L * k_L

    g = 9.8

    g_modulation = g_eulerian(g, phase, a_L, k_L, order=3) / g
    k_modulation = np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
    N_modulation = k_modulation
    a_modulation = g_modulation ** (-0.25) * k_modulation**0.25 * N_modulation**0.5

    a_modulation_lhs = 1 + ak * np.cos(phase)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.text(
        0, 1.02, "(a)", ha="left", va="bottom", transform=ax1.transAxes, fontsize=20
    )
    ax2.text(
        0, 1.02, "(b)", ha="left", va="bottom", transform=ax2.transAxes, fontsize=20
    )

    ax1.plot(phase, a_modulation_lhs, "k-", label=r"L-HS 1960", lw=2)
    ax1.plot(phase, a_modulation, "r-", label=r"This paper", lw=2)
    ax1.legend()
    ax1.set_xlabel("Phase (rad.)")
    ax1.set_ylabel(r"$a'/a$")
    ax1.grid()
    ax1.plot([0, 2 * np.pi], [1, 1], "k--")
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax1.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    a_min = []
    a_max = []
    a_min_lhs = []
    a_max_lhs = []
    ak_range = np.arange(0, 0.41, 0.01)
    for ak in ak_range:
        a_L = ak / k_L
        g_modulation = g_eulerian(g, phase, a_L, k_L, order=3) / g
        k_modulation = np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
        N_modulation = k_modulation
        a_modulation = g_modulation ** (-0.25) * k_modulation**0.25 * N_modulation**0.5
        a_min.append(np.min(a_modulation))
        a_max.append(np.max(a_modulation))
        a_min_lhs.append(np.min(1 + ak * np.cos(phase)))
        a_max_lhs.append(np.max(1 + ak * np.cos(phase)))
    a_min = np.array(a_min)
    a_max = np.array(a_max)
    a_min_lhs = np.array(a_min_lhs)
    a_max_lhs = np.array(a_max_lhs)

    ax2.plot(ak_range, a_max_lhs, "k-", label=r"L-HS 1960, max.", lw=2)
    ax2.plot(ak_range, a_min_lhs, "k:", label=r"L-HS 1960, min.", lw=2)
    ax2.plot(ak_range, a_max, "r-", label=r"This paper, max.", lw=2)
    ax2.plot(ak_range, a_min, "r:", label=r"This paper, min.", lw=2)
    ax2.legend()
    ax2.plot([0, 0.4], [1, 1], "k--")
    ax2.grid()
    ax2.set_xlabel(r"$\varepsilon_L$")
    ax2.set_ylabel(r"$a'/a$")
    ax2.set_xlim(0, 0.4)
    ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4])

    fig.subplots_adjust(left=0.06, right=0.96)
    plt.savefig("../figures/fig_amplitude_modulation_2panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_amplitude_modulation()

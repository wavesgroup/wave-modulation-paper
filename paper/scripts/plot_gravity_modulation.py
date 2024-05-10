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
            1 - a * k * np.exp(k * eta) * (np.cos(psi) + a * k * np.sin(psi) ** 2)
        )
    return res


def g_lagrangian(g, psi, a, k, order=4):
    assert order in [3, 4]
    eta = a * np.cos(psi)
    g_E = g_eulerian(g, psi, a, k, order=3)
    if order == 3:
        res = g_E - a**2 * k**2 * np.exp(2 * k * eta)
    if order == 4:
        res = (
            g_E
            - a**2 * k**2 * np.exp(2 * k * eta)
            - a**3
            * k**3
            * np.exp(2 * k * eta)
            * (np.sin(psi) ** 2 * np.cos(psi) - np.sin(2 * psi) * np.sin(psi))
        )
    return res


def plot_fig_apparent_gravity():
    """Plot the apparent gravity figure."""

    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    a_L = 0.2
    k_L = 1

    g = 9.8

    g_eulerian_1 = g_eulerian(g, phase, a_L, k_L, order=1)
    g_eulerian_2 = g_eulerian(g, phase, a_L, k_L, order=2)
    g_eulerian_3 = g_eulerian(g, phase, a_L, k_L, order=3)
    g_lagrangian_3 = g_lagrangian(g, phase, a_L, k_L, order=3)
    # g_lagrangian_4 = g_lagrangian(g, phase, a_L, k_L, order=4) # Essentially the same as 3rd order; omit

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.text(
        0, 1.02, "(a)", ha="left", va="bottom", transform=ax1.transAxes, fontsize=20
    )
    ax2.text(
        0, 1.02, "(b)", ha="left", va="bottom", transform=ax2.transAxes, fontsize=20
    )

    ax1.plot(phase, g_eulerian_1 / g, label=r"Eulerian, 1$^{st}$ order", lw=2)
    ax1.plot(phase, g_eulerian_2 / g, label=r"Eulerian, 2$^{nd}$ order", lw=2)
    ax1.plot(phase, g_eulerian_3 / g, label=r"Eulerian, 3$^{rd}$ order", lw=2)
    ax1.plot(phase, g_lagrangian_3 / g, label=r"Lagrangian, 3$^{rd}$ order", lw=2)
    ax1.legend()
    ax1.set_xlabel("Phase (rad.)")
    ax1.set_ylabel(r"$\widetilde g/g$")
    ax1.grid()
    ax1.plot([0, 2 * np.pi], [1, 1], "k--")
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax1.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    # Evaluate Eulerian and Lagrangian gravity vs. ak
    g_E1 = []
    g_E2 = []
    g_E3 = []
    g_L3 = []
    ak_range = np.arange(0, 0.41, 0.01)
    for ak in ak_range:
        a_L = ak / k_L
        g_E1.append(np.mean(g_eulerian(g, phase, a_L, k_L, order=1)))
        g_E2.append(np.mean(g_eulerian(g, phase, a_L, k_L, order=2)))
        g_E3.append(np.mean(g_eulerian(g, phase, a_L, k_L, order=3)))
        g_L3.append(np.mean(g_lagrangian(g, phase, a_L, k_L, order=3)))
    g_E1 = np.array(g_E1) / g
    g_E2 = np.array(g_E2) / g
    g_E3 = np.array(g_E3) / g
    g_L3 = np.array(g_L3) / g

    ax2.plot(ak_range, g_E1, label=r"Eulerian, 1$^{st}$ order", lw=2)
    ax2.plot(ak_range, g_E2, label=r"Eulerian, 2$^{nd}$ order", lw=2)
    ax2.plot(ak_range, g_E3, label=r"Eulerian, 3$^{rd}$ order", lw=2)
    ax2.plot(ak_range, g_L3, label=r"Lagrangian, 3$^{rd}$ order", lw=2)
    ax2.plot([0, 0.4], [1, 1], "k--")
    ax2.grid()
    ax2.set_xlabel(r"$\varepsilon_L$")
    ax2.set_ylabel(r"$\overline{\widetilde g}/g$")
    ax2.set_xlim(0, 0.4)
    ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_ylim(0.8, 1.05)

    fig.subplots_adjust(left=0.06, right=0.96)
    plt.savefig("../figures/fig_gravity_modulation_2panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig_apparent_gravity()

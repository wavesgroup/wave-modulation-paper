import matplotlib.pyplot as plt
import numpy as np
from modulation import wavenumber
import matplotlib

matplotlib.rc("font", size=16)


def plot_fig_wavenumber_modulation():
    """Plot the wavenumber modulation comparison figure."""

    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    k_L = 1
    a_L = 0.1
    ak_L = a_L * k_L

    solution1 = 1 + ak_L * np.cos(phase)
    solution2 = np.exp(ak_L * np.cos(phase))
    solution3 = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(phase, solution1, "k-", lw=2, label="L-H&S")
    ax.plot(phase, solution3, "r-", lw=2, label="This paper")
    ax.legend()
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel(r"$\widetilde k/k$")
    ax.grid()
    ax.plot([0, 2 * np.pi], [1, 1], "k--")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_title(r"$\varepsilon_L = 0.1$")
    plt.savefig("../figures/fig_wavenumber_modulation_by_phase.png", dpi=200)
    plt.close(fig)

    # Evaluate wa enumber modulation as function of ak
    ak_range = np.arange(0, 0.41, 0.01)
    k1_mean, k1_max = [], []
    k2_mean, k2_max = [], []
    for ak in ak_range:
        k1 = 1 + ak * np.cos(phase)
        k2 = np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
        k1_mean.append(np.mean(k1))
        k1_max.append(np.max(k1))
        k2_mean.append(np.mean(k2))
        k2_max.append(np.max(k2))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ak_range, k1_max, "k-", label="L-H&S, max", lw=2)
    ax.plot(ak_range, k1_mean, "k--", label="L-H&S, mean", lw=2)
    ax.plot(ak_range, k2_max, "r-", label="This paper, max", lw=2)
    ax.plot(ak_range, k2_mean, "r--", label="This paper, mean", lw=2)
    ax.legend(loc="upper left")
    ax.grid()
    ax.set_xlabel(r"$\varepsilon_L$")
    ax.set_ylabel(r"$\~ k/k$")
    ax.set_xlim(0, 0.4)
    plt.savefig("../figures/fig_wavenumber_modulation_by_ak.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig_wavenumber_modulation()

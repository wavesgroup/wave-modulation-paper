import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rc("font", size=16)


def plot_fig_wavenumber_modulation():
    """Plot the wavenumber modulation comparison figure."""

    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    k_L = 1
    a_L = 0.2
    ak_L = a_L * k_L

    solution1 = 1 + ak_L * np.cos(phase)
    solution3 = np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.text(
        0, 1.02, "(a)", ha="left", va="bottom", transform=ax1.transAxes, fontsize=20
    )
    ax2.text(
        0, 1.02, "(b)", ha="left", va="bottom", transform=ax2.transAxes, fontsize=20
    )

    ax1.plot(phase, solution1, "k-", lw=2, label="L-H&S")
    ax1.plot(phase, solution3, "r-", lw=2, label="This paper")
    ax1.legend()
    ax1.set_xlabel("Phase (rad.)")
    ax1.set_ylabel(r"$\widetilde k/k$")
    ax1.grid()
    ax1.plot([0, 2 * np.pi], [1, 1], "k--")
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax1.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    # Evaluate wa enumber modulation as function of ak
    ak_range = np.arange(0, 0.41, 0.01)
    k1_mean, k1_max, k1_min = [], [], []
    k2_mean, k2_max, k2_min = [], [], []
    for ak in ak_range:
        k1 = 1 + ak * np.cos(phase)
        k2 = np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
        k1_mean.append(np.mean(k1))
        k1_max.append(np.max(k1))
        k1_min.append(np.min(k1))
        k2_mean.append(np.mean(k2))
        k2_max.append(np.max(k2))
        k2_min.append(np.min(k2))

    ax2.plot(ak_range, k1_max, "k-", label="L-H&S, max.", lw=2)
    ax2.plot(ak_range, k1_mean, "k--", label="L-H&S, mean", lw=2)
    ax2.plot(ak_range, k1_min, "k:", label="L-H&S, min.", lw=2)
    ax2.plot(ak_range, k2_max, "r-", label="This paper, max.", lw=2)
    ax2.plot(ak_range, k2_mean, "r--", label="This paper, mean", lw=2)
    ax2.plot(ak_range, k2_min, "r:", label="This paper, min.", lw=2)
    ax2.legend(loc="upper left")
    ax2.grid()
    ax2.set_xlabel(r"$\varepsilon_L$")
    ax2.set_ylabel(r"$\widetilde k/k$")
    ax2.set_xlim(0, 0.4)
    ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4])

    fig.subplots_adjust(left=0.06, right=0.96)
    plt.savefig("../figures/fig_wavenumber_modulation_2panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig_wavenumber_modulation()

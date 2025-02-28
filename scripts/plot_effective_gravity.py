import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from twowave import gravity

matplotlib.rc("font", size=16)


def plot_effective_gravities(a: float):
    a = 0.2
    phase = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
    k = 1
    ak = a * k
    x = phase / k
    g0 = 9.8
    omega = np.sqrt(g0 * k)
    t = 0

    g1 = g0 * (1 - ak * np.cos(phase))
    g2 = gravity(x, t, a, k, omega, g0, "linear", curvilinear=False)
    g3 = gravity(x, t, a, k, omega, g0, "stokes", curvilinear=False)
    g4 = gravity(x, t, a, k, omega, g0, "linear", curvilinear=True)
    g5 = gravity(x, t, a, k, omega, g0, "stokes", curvilinear=True)

    fig = plt.figure(figsize=(8, 12))
    axes = fig.subplots(2, 1)
    axes[0].plot(
        phase, g1 / g0, label=r"Linear wave, $z=0$", color="black", lw=2, alpha=0.8
    )
    axes[0].plot(
        phase,
        g2 / g0,
        label=r"Linear wave, $z=\eta$, $\alpha=0$",
        color="tab:blue",
        lw=2,
        alpha=0.8,
    )
    axes[0].plot(
        phase,
        g3 / g0,
        label=r"Stokes wave, $z=\eta$, $\alpha=0$",
        color="tab:green",
        lw=2,
        alpha=0.8,
    )
    axes[0].plot(
        phase,
        g4 / g0,
        label=r"Linear wave, $z=\eta$, $\alpha=\frac{\partial \eta}{\partial x}$",
        color="tab:orange",
        lw=2,
        alpha=0.8,
    )
    axes[0].plot(
        phase,
        g5 / g0,
        label=r"Stokes wave, $z=\eta$, $\alpha=\frac{\partial \eta}{\partial x}$",
        color="tab:red",
        lw=2,
        alpha=0.8,
    )
    axes[0].set_ylabel(r"$\widetilde{g}/g$")
    axes[0].legend(ncol=1, prop={"size": 13})
    axes[0].set_title(r"$\varepsilon_L=$" + f"{ak}")

    axes[1].plot(
        phase,
        g2 / g1,
        label=r"Linear wave, $z=\eta$, $\alpha=0$",
        color="tab:blue",
        lw=2,
        alpha=0.8,
    )
    axes[1].plot(
        phase,
        g3 / g1,
        label=r"Stokes wave, $z=\eta$, $\alpha=0$",
        color="tab:green",
        lw=2,
        alpha=0.8,
    )
    axes[1].plot(
        phase,
        g4 / g1,
        label=r"Linear wave, $z=\eta$, $\alpha=\frac{\partial \eta}{\partial x}$",
        color="tab:orange",
        lw=2,
        alpha=0.8,
    )
    axes[1].plot(
        phase,
        g5 / g1,
        label=r"Stokes wave, $z=\eta$, $\alpha=\frac{\partial \eta}{\partial x}$",
        color="tab:red",
        lw=2,
        alpha=0.8,
    )
    axes[1].set_xlabel(r"$\psi$")
    axes[1].set_ylabel(r"$\widetilde{g}/\widetilde{g_1}$")
    axes[1].legend(ncol=2, prop={"size": 13})
    axes[1].set_ylim(0.9, 1.06)

    for ax in axes:
        ax.grid()
        ax.plot([0, 2 * np.pi], [1, 1], color="black", lw=1, ls="--")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    axes[0].text(
        0.0,
        1.05,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=16,
        va="top",
    )
    axes[1].text(
        0.0,
        1.05,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=16,
        va="top",
    )

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.05, top=0.95)
    plt.savefig("../figures/fig_effective_gravities.pdf")
    plt.close()

    g1_mean = []
    g2_mean = []
    g3_mean = []
    g4_mean = []
    g5_mean = []
    g1_max = []
    g2_max = []
    g3_max = []
    g4_max = []
    g5_max = []
    g1_min = []
    g2_min = []
    g3_min = []
    g4_min = []
    g5_min = []
    for a in np.arange(0, 0.41, 0.01):
        ak = a * k
        g1 = g0 * (1 - ak * np.cos(phase))
        g2 = gravity(x, t, a, k, omega, g0, "linear", curvilinear=False)
        g3 = gravity(x, t, a, k, omega, g0, "stokes", curvilinear=False)
        g4 = gravity(x, t, a, k, omega, g0, "linear", curvilinear=True)
        g5 = gravity(x, t, a, k, omega, g0, "stokes", curvilinear=True)
        g1_max.append(np.max(g1))
        g2_max.append(np.max(g2))
        g3_max.append(np.max(g3))
        g4_max.append(np.max(g4))
        g5_max.append(np.max(g5))
        g1_mean.append(np.mean(g1))
        g2_mean.append(np.mean(g2))
        g3_mean.append(np.mean(g3))
        g4_mean.append(np.mean(g4))
        g5_mean.append(np.mean(g5))
        g1_min.append(np.min(g1))
        g2_min.append(np.min(g2))
        g3_min.append(np.min(g3))
        g4_min.append(np.min(g4))
        g5_min.append(np.min(g5))
    g1_mean = np.array(g1_mean)
    g2_mean = np.array(g2_mean)
    g3_mean = np.array(g3_mean)
    g4_mean = np.array(g4_mean)
    g5_mean = np.array(g5_mean)
    g1_max = np.array(g1_max)
    g2_max = np.array(g2_max)
    g3_max = np.array(g3_max)
    g4_max = np.array(g4_max)
    g5_max = np.array(g5_max)
    g1_min = np.array(g1_min)
    g2_min = np.array(g2_min)
    g3_min = np.array(g3_min)
    g4_min = np.array(g4_min)
    g5_min = np.array(g5_min)

    ak_range = np.arange(0, 0.41, 0.01)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(
        ak_range,
        g1_mean / g0,
        label=r"Linear wave, $z=0$",
        color="black",
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        ak_range,
        g2_mean / g0,
        label=r"Linear wave, $z=\eta$, $\alpha=0$",
        color="tab:blue",
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        ak_range,
        g3_mean / g0,
        label=r"Stokes wave, $z=\eta$, $\alpha=0$",
        color="tab:green",
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        ak_range,
        g4_mean / g0,
        label=r"Linear wave, $z=\eta$, $\alpha=\frac{\partial \eta}{\partial x}$",
        color="tab:orange",
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        ak_range,
        g5_mean / g0,
        label=r"Stokes wave, $z=\eta$, $\alpha=\frac{\partial \eta}{\partial x}$",
        color="tab:red",
        lw=2,
        alpha=0.8,
    )
    ax.legend(ncol=1, prop={"size": 12})

    ax.plot(ak_range, g1_max / g0, color="black", lw=2, ls="--")
    ax.plot(ak_range, g2_max / g0, color="tab:blue", lw=2, ls="--")
    ax.plot(ak_range, g3_max / g0, color="tab:green", lw=2, ls="--")
    ax.plot(ak_range, g4_max / g0, color="tab:orange", lw=2, ls="--")
    ax.plot(ak_range, g5_max / g0, color="tab:red", lw=2, ls="--")

    ax.plot(ak_range, g1_min / g0, color="black", lw=2, ls=":")
    ax.plot(ak_range, g2_min / g0, color="tab:blue", lw=2, ls=":")
    ax.plot(ak_range, g3_min / g0, color="tab:green", lw=2, ls=":")
    ax.plot(ak_range, g4_min / g0, color="tab:orange", lw=2, ls=":")
    ax.plot(ak_range, g5_min / g0, color="tab:red", lw=2, ls=":")

    ax.grid()

    ax.set_xlabel(r"$\varepsilon_L$")
    ax.set_ylabel(r"$\widetilde{g}/g$")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.3, 1.4)

    ax.text(0.35, 1.02, "Mean", color="black", fontsize=16)
    ax.text(0.15, 1.3, "Max. (troughs)", color="black", fontsize=16)
    ax.text(0.25, 0.4, "Min. (crests)", color="black", fontsize=16)

    plt.savefig("../figures/fig_effective_gravities_mean_max_min.pdf")
    plt.close()


if __name__ == "__main__":
    plot_effective_gravities(0.1)

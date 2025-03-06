import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from twowave import (
    diff,
    elevation,
    orbital_horizontal_velocity,
    orbital_vertical_velocity,
    surface_slope,
    nonlinear_wave_properties,
)


def gravity(
    x: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    g0: float = 9.8,
    cg: float = 0.0,
    wave_type: str = "linear",
    nonlinear_props: tuple = None,
    curvilinear: bool = True,
    lagrangian: bool = True,
) -> float:
    z = elevation(x, t, a, k, omega, wave_type, nonlinear_props)
    U = orbital_horizontal_velocity(x, z, t, a, k, omega, wave_type, nonlinear_props)
    W = orbital_vertical_velocity(x, z, t, a, k, omega, wave_type, nonlinear_props)
    Cp = omega / k
    dx = np.diff(x)[0]
    vel = Cp
    if lagrangian:
        vel = vel - U - cg
    a_z = -diff(W) / dx * vel
    if curvilinear:
        a_x = -diff(U) / dx * vel
        slope = surface_slope(x, t, a, k, omega, wave_type, nonlinear_props)
        g = g0 * np.cos(slope) + a_z * np.cos(slope) + a_x * np.sin(slope)
    else:
        g = g0 + a_z
    return g


def plot_effective_gravities(a: float):
    phase = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    k = 1
    ak = a * k
    x = phase / k
    g0 = 9.8
    omega = np.sqrt(g0 * k)
    t = 0

    nonlinear_props = nonlinear_wave_properties(a, k, g0, num_points=x.size)

    g1 = g0 * (1 - ak * np.cos(phase))
    g2 = gravity(
        x, t, a, k, omega, g0, wave_type="linear", curvilinear=False, lagrangian=False
    )
    g3 = gravity(
        x, t, a, k, omega, g0, wave_type="linear", curvilinear=False, lagrangian=True
    )
    g4 = gravity(
        x, t, a, k, omega, g0, wave_type="linear", curvilinear=True, lagrangian=True
    )
    g5 = gravity(
        x, t, a, k, omega, g0, wave_type="stokes", curvilinear=True, lagrangian=True
    )
    g6 = gravity(
        x,
        t,
        a,
        k,
        omega,
        g0,
        wave_type="nonlinear",
        curvilinear=True,
        lagrangian=True,
        nonlinear_props=nonlinear_props,
    )

    matplotlib.rc("font", size=16)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(phase, g1 / g0, label=r"1$^{st}$ order, $z=0$", color="black", lw=3)
    ax.plot(phase, g2 / g0, label=r"1$^{st}$ order, $z=\eta$", color="tab:blue", lw=3)
    ax.plot(
        phase,
        g3 / g0,
        label=r"1$^{st}$ order, $z=\eta$, Lagrangian",
        color="tab:orange",
        lw=3,
    )
    ax.plot(
        phase,
        g4 / g0,
        label=r"1$^{st}$ order, $z=\eta$, Lagrangian, curvilinear",
        color="tab:green",
        lw=3,
    )
    ax.plot(
        phase,
        g5 / g0,
        label=r"3$^{rd}$ order, $z=\eta$, Lagrangian, curvilinear",
        color="tab:red",
        lw=3,
    )
    ax.plot(
        phase,
        g6 / g0,
        label=r"Nonlinear, $z=\eta$, Lagrangian, curvilinear",
        color="tab:purple",
        lw=3,
    )
    ax.legend(ncol=1, prop={"size": 16})
    ax.set_title(r"$\varepsilon_L=$" + f"{ak}")
    ax.grid()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels(
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    )
    ax.set_xlabel(r"$\Psi$")
    ax.set_ylabel(r"$\widetilde{g}/g$")

    plt.savefig("../figures/fig_effective_gravities.pdf")
    plt.close()

    g1_min = []
    g2_min = []
    g3_min = []
    g4_min = []
    g5_min = []
    g6_min = []
    for a in np.arange(0, 0.41, 0.01):
        ak = a * k
        g1 = g0 * (1 - ak * np.cos(phase))
        nonlinear_props = nonlinear_wave_properties(a + 1e-6, k, g0, num_points=x.size)
        g1 = g0 * (1 - ak * np.cos(phase))
        g2 = gravity(
            x,
            t,
            a,
            k,
            omega,
            g0,
            wave_type="linear",
            curvilinear=False,
            lagrangian=False,
        )
        g3 = gravity(
            x,
            t,
            a,
            k,
            omega,
            g0,
            wave_type="linear",
            curvilinear=False,
            lagrangian=True,
        )
        g4 = gravity(
            x, t, a, k, omega, g0, wave_type="linear", curvilinear=True, lagrangian=True
        )
        g5 = gravity(
            x, t, a, k, omega, g0, wave_type="stokes", curvilinear=True, lagrangian=True
        )
        g6 = gravity(
            x,
            t,
            a,
            k,
            omega,
            g0,
            wave_type="nonlinear",
            curvilinear=True,
            lagrangian=True,
            nonlinear_props=nonlinear_props,
        )
        g1_min.append(np.min(g1))
        g2_min.append(np.min(g2))
        g3_min.append(np.min(g3))
        g4_min.append(np.min(g4))
        g5_min.append(np.min(g5))
        g6_min.append(np.min(g6))
    g1_min = np.array(g1_min)
    g2_min = np.array(g2_min)
    g3_min = np.array(g3_min)
    g4_min = np.array(g4_min)
    g5_min = np.array(g5_min)
    g6_min = np.array(g6_min)

    ak_range = np.arange(0, 0.41, 0.01)

    matplotlib.rc("font", size=14)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ak_range, g1_min / g0, color="black", lw=3, label=r"1$^{st}$ order, $z=0$")
    ax.plot(
        ak_range, g2_min / g0, color="tab:blue", lw=3, label=r"1$^{st}$ order, $z=\eta$"
    )
    ax.plot(
        ak_range,
        g3_min / g0,
        color="tab:orange",
        lw=3,
        label=r"1$^{st}$ order, $z=\eta$, Lagrangian",
    )
    ax.plot(
        ak_range,
        g4_min / g0,
        color="tab:green",
        lw=3,
        label=r"1$^{st}$ order, $z=\eta$, Lagrangian, curvilinear",
    )
    ax.plot(
        ak_range,
        g5_min / g0,
        color="tab:red",
        lw=3,
        label=r"3$^{rd}$ order, $z=\eta$, Lagrangian, curvilinear",
    )
    ax.plot(
        ak_range,
        g6_min / g0,
        color="tab:purple",
        lw=3,
        label=r"Nonlinear, $z=\eta$, Lagrangian, curvilinear",
    )
    ax.legend(ncol=1, prop={"size": 12}, loc="lower left")
    ax.grid()
    ax.set_xlabel(r"$\varepsilon_L$")
    ax.set_ylabel(r"$min(\widetilde{g}/g)$")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.4, 1.0)
    plt.savefig("../figures/fig_effective_gravities_min.pdf")
    plt.close()


if __name__ == "__main__":
    plot_effective_gravities(0.3)

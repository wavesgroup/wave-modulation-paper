import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import twowave

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
    """Gravitational acceleration at the surface of a linear wave."""
    eta = elevation_linear(a, phase)
    return g0 * (
        1 - a * k * np.exp(k * eta) * (np.cos(phase) - a * k * np.sin(phase) ** 2)
    )


def gravity_stokes(a: float, k: float, phase: float, g0: float) -> float:
    """Gravitational acceleration at the surface of a Stokes wave."""
    eta = elevation_stokes(a, k, phase)
    res = g0 * (
        1
        - a
        * k
        * (
            np.cos(phase)
            - a
            * k
            * np.sin(phase)
            * (
                (1 - 1 / 16 * (a * k) ** 2) * np.sin(phase)
                + a * k * np.sin(2 * phase)
                + 9 / 8 * (a * k) ** 2 * np.sin(3 * phase)
            )
        )
        * np.exp(k * eta)
    )
    return res


def calculate_tendencies(a_L: float):
    phase = np.arange(0, 2 * np.pi + 1e-4, 1e-4)
    g0 = 9.8
    k_L = 1
    ak_L = a_L * k_L
    omega_L = np.sqrt(g0 * k_L)

    L = 2 * np.pi / k_L
    dx = L / phase.size

    k0 = 10
    N0 = 1

    k = k0 * np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
    N = N0 * np.exp(ak_L * np.cos(phase) * np.exp(ak_L * np.cos(phase)))
    g = gravity_stokes(a_L, k_L, phase, g0)
    eta = elevation_stokes(a_L, k_L, phase)
    U = a_L * omega_L * np.exp(k_L * eta) * np.cos(phase)
    omega = np.sqrt(g * k)
    Cp = omega / k
    Cg = Cp / 2

    dkdx = twowave.diff(k) / dx
    dgdx = twowave.diff(g) / dx
    dUdx = twowave.diff(U) / dx
    dCgdx = twowave.diff(Cg) / dx
    dNdx = twowave.diff(N) / dx

    k_propagation = -Cg * dkdx
    k_advection = -U * dkdx
    k_convergence = -k * dUdx
    k_inhomogeneity = -0.5 * np.sqrt(k / g) * dgdx
    dkdt = k_propagation + k_advection + k_convergence + k_inhomogeneity

    N_propagation = -Cg * dNdx
    N_advection = -U * dNdx
    N_convergence = -N * dUdx
    N_inhomogeneity = -N * dCgdx
    N_inhomogeneity = -N * dCgdx
    dNdt = N_propagation + N_advection + N_convergence + N_inhomogeneity

    return (
        phase,
        k_propagation,
        k_advection,
        k_convergence,
        k_inhomogeneity,
        dkdt,
        N_propagation,
        N_advection,
        N_convergence,
        N_inhomogeneity,
        dNdt,
    )


def plot_scale_analysis(ak_L: float):
    (
        phase,
        k_propagation,
        k_advection,
        k_convergence,
        k_inhomogeneity,
        dkdt,
        N_propagation,
        N_advection,
        N_convergence,
        N_inhomogeneity,
        dNdt,
    ) = calculate_tendencies(ak_L)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(phase, k_propagation, label=r"$C_g \dfrac{\partial k}{\partial x}$")
    ax.plot(phase, k_advection, label=r"$U \dfrac{\partial k}{\partial x}$")
    ax.plot(phase, k_convergence, label=r"$k \dfrac{\partial U}{\partial x}$")
    ax.plot(
        phase,
        k_inhomogeneity,
        label=r"$\dfrac{1}{2} \sqrt{\dfrac{k}{g}} \dfrac{\partial g}{\partial x}$",
    )
    ax.plot(phase, dkdt, "k--", label=r"$\dfrac{\partial k}{\partial t}$")
    ax.legend(ncol=5)
    ax.grid()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_ylabel("Wavenumber tendency [rad/m/s]")
    plt.savefig(f"../figures/fig_scale_analysis_k_ak{ak_L}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_scale_analysis(0.1)
    plot_scale_analysis(0.2)
    plot_scale_analysis(0.3)
    plot_scale_analysis(0.4)

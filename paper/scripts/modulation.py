import numpy as np


def gravity(
    x: float, t: float, a: float, k: float, omega: float, g0: float, order: int = 3
) -> float:
    assert order in [0, 1, 2, 3]
    phi = phase(x, t, k, omega)
    term1 = -a * k * np.cos(phi)
    term2 = -a * k * np.cos(phi) * np.exp(a * k * np.cos(phi))
    term3 = ((a * k * np.sin(phi)) ** 2) * np.exp(a * k * np.cos(phi))
    if order == 0:
        return g0 * np.ones_like(x)
    elif order == 1:
        return g0 * (1 + term1)
    elif order == 2:
        return g0 * (1 + term2)
    elif order == 3:
        return g0 * (1 + term2 + term3)
    else:
        raise ValueError("Invalid order")


def phase(
    x: float | np.ndarray,
    t: float | np.ndarray,
    k: float | np.ndarray,
    omega: float | np.ndarray,
) -> float | np.ndarray:
    return k * x - omega * t


def wavenumber(
    frequency: float | np.ndarray,
    water_depth: float,
    grav: float = 9.8,
    surface_tension: float = 0.074,
    water_density: float = 1e3,
    num_iterations: int = 100,
) -> float | np.ndarray:
    """Solve the dispersion relationship for wavenumber using the
    Newton-Raphson method."""
    frequency_nondim = 2 * np.pi * np.sqrt(water_depth / grav) * frequency
    k = frequency_nondim**2
    surface_tension_nondim = surface_tension / (grav * water_density * water_depth**2)
    count = 0
    while count < num_iterations:
        t = np.tanh(k)
        dk = -(frequency_nondim**2 - k * t * (1 + surface_tension_nondim * k**2)) / (
            3 * surface_tension_nondim * k**2 * t
            + t
            + k * (1 + surface_tension_nondim * k**2) * (1 - t**2)
        )
        k -= dk
        count += 1
    return k / water_depth

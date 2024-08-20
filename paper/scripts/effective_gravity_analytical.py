from sympy import (
    diff,
    exp,
    cos,
    sin,
    atan,
    symbols,
    simplify,
    init_printing,
)

init_printing()

x, t, a, k, ω, ψ, U, W, g, g0, η, α = symbols(
    "x t a k ω ψ U W g g_0 η α", real=True, positive=True
)

# Long-wave phase
ψ = k * x - ω * t

# Long-wave elevation
η = a * cos(ψ)

# Local elevation slope
α = atan(diff(η, x))

# Orbital velocities
U = a * ω * exp(k * η) * cos(ψ)
W = a * ω * exp(k * η) * sin(ψ)

# Orbital accelerations
dU_dt = simplify(diff(U, t))
dW_dt = simplify(diff(W, t))

g = simplify(g0 * cos(α) + dW_dt * cos(α) + dU_dt * sin(α))

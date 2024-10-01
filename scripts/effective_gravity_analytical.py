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

x, t, a, k, ak, ω, ψ, U, W, U_St, W_St, g, g0, η, η_St, α, α_St = symbols(
    "x t a k ak ω ψ U W U_St W_St g g_0 η η_St α α_St", real=True, positive=True
)

# Long-wave phase
ψ = k * x - ω * t

ak = a * k

# Long-wave elevation (linear wave)
η = a * cos(ψ)
η_St = a * (
    cos(ψ) + 0.5 * ak * cos(2 * ψ) + ak**2 * (3 / 8.0 * cos(3 * ψ) - 1 / 8.0 * cos(ψ))
)

# Local elevation slope
α = atan(diff(η, x))
α_St = atan(diff(η_St, x))

# Orbital velocities
U = a * ω * exp(k * η) * cos(ψ)
W = a * ω * exp(k * η) * sin(ψ)

U_St = a * ω * exp(k * η_St) * cos(ψ)
W_St = a * ω * exp(k * η_St) * sin(ψ)

# Orbital accelerations
dU_dt = simplify(diff(U, t))
dW_dt = simplify(diff(W, t))

dU_dt_St = simplify(diff(U_St, t))
dW_dt_St = simplify(diff(W_St, t))

# Effective gravities
g = simplify(g0 * cos(α) + dW_dt * cos(α) + dU_dt * sin(α))
g_St = simplify(g0 * cos(α_St) + dW_dt_St * cos(α_St) + dU_dt_St * sin(α_St))

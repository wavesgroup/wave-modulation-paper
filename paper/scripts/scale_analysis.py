import matplotlib.pyplot as plt
from model import diff, WaveModulationModel
import numpy as np

phase = np.linspace(0, 2 * np.pi, 100, endpoint=False)
a_long = 0.1
k_long = 1
ak = a_long * k_long
x = phase / k_long
dx = x[1] - x[0]

k0 = 10
g0 = 9.8

omega_long = np.sqrt(g0 * k_long)
U = a_long * omega_long * np.cos(phase) * np.exp(ak * np.cos(phase))

k = k0 * np.exp(ak * np.cos(phase) * np.exp(ak * np.cos(phase)))
g = g0 * (
    1 - ak * np.exp(ak * np.cos(phase)) * (np.cos(phase) + ak * np.sin(phase) ** 2)
)
omega = np.sqrt(g * k)
Cg = 0.5 * omega / k

propagation = -Cg * diff(k) / dx
advection = -U * diff(k) / dx
convergence = -k * diff(U) / dx
inhomogeneity = -0.5 * np.sqrt(k / g) * diff(g) / dx


m = WaveModulationModel(a_long=a_long, k_long=k_long)
m.run(ramp_type="groups")
ds = m.to_xarray()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(phase, k0 * (1 + ak * np.cos(phase)))
ax.plot(phase, k)
ax.plot(phase, ds.wavenumber[1004, :])
ax.grid()

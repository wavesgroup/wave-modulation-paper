import numpy as np
from twowave import elevation, diff

phase = np.linspace(0, 2 * np.pi, 100, endpoint=False)
k = 1
a = 0.4
omega = np.sqrt(9.8 * k)

x = phase / k
t = 0

eta = elevation(x, t, a, k, omega)
deta_dx_exact = (-a * k * np.sin(phase))[1:-1]
deta_dx = (diff(eta) / diff(x))[1:-1]

max_abs_error = np.max(np.abs(deta_dx - deta_dx_exact))
rel_error = np.abs(deta_dx - deta_dx_exact) / np.abs(deta_dx_exact)
max_rel_error = np.max(rel_error[rel_error < 1])

print("Max. abs. error: ", max_abs_error)
print("Max. rel. error: ", max_rel_error * 100, "%")

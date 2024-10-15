import matplotlib.pyplot as plt
from twowave import WaveModulationModel
import numpy as np
import matplotlib

matplotlib.rc("font", size=16)

m1 = WaveModulationModel(num_periods=30)
m2 = WaveModulationModel(num_periods=30)

m1.run(ramp_type=None, wave_type="linear")
m2.run(ramp_type="linear", wave_type="linear")

ds1 = m1.to_xarray()
ds2 = m2.to_xarray()

T_L = 2 * np.pi / m1.omega_long
periods = ds1.time / T_L

ak1 = ds1.amplitude * ds1.wavenumber
ak2 = ds2.amplitude * ds2.wavenumber
ak_init = 0.1

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(
    periods,
    np.max(ak1, 1) / ak_init,
    color="tab:blue",
    lw=2,
    label="Sudden onset of long waves",
)
ax.plot(
    periods, np.max(ak2, 1) / ak_init, color="tab:orange", lw=2, label="Linear ramp"
)
ax.legend()
ax.set_xlabel("Time (periods)")
ax.set_ylabel(r"$\widetilde{ak}/(ak)$")
ax.grid()
ax.set_xlim(0, 30)
ax.set_ylim(1, 4.5)
ax.plot([0, 30], [4, 4], "k--")
plt.savefig("../figures/fig_modulation_ramp_timeseries.pdf")
plt.close()

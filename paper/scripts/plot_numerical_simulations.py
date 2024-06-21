import matplotlib.pyplot as plt
from model import WaveModulationModel
import numpy as np


def plot_modulation_3panel(model, contour_levels, title):
    ds = model.to_xarray()

    num_periods = 10
    time = ds.time / model.T_long
    ak = ds.amplitude * ds.wavenumber

    fig = plt.figure(figsize=(8, 6))
    axes = fig.subplots(1, 3, sharey=True)

    axes[0].contourf(ds.space, time, ds.wave_action, contour_levels)
    axes[0].contour(ds.space, time, ds.wave_action, [1], colors="k", linewidths=0.5)
    axes[0].set_ylabel("Time (long-wave periods)")
    axes[0].set_title(r"$N$")

    axes[1].contourf(ds.space, time, ds.wavenumber / model.k_short, contour_levels)
    axes[1].contour(
        ds.space, time, ds.wavenumber / model.k_short, [1], colors="k", linewidths=0.5
    )
    axes[1].set_title(r"$k$")

    cm = axes[2].contourf(
        ds.space, time, ak / model.k_short / model.a_short, contour_levels
    )
    axes[2].contour(
        ds.space,
        time,
        ak / model.k_short / model.a_short,
        [1],
        colors="k",
        linewidths=0.5,
    )
    axes[2].set_title(r"$ak$")

    for ax in axes:
        ax.set_yticks(range(0, num_periods + 1, 1))
        ax.set_xticks(np.arange(0, 2.5 * np.pi, 0.5 * np.pi))
        ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlabel("Phase")

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.0, top=0.9)
    cbar = fig.colorbar(
        cm, ax=axes, orientation="horizontal", pad=0.15, shrink=0.7, label="Modulation"
    )

    plt.savefig(
        f"../figures/fig_modulation_3panel_{title.lower().replace(' ', '_')}.png",
        dpi=200,
    )
    plt.close(fig)


m = WaveModulationModel()
m.run(ramp_type=None)
plot_modulation_3panel(m, np.arange(0.5, 2.3, 0.1), "Infinite wave train")

m.run(ramp_type="linear")
plot_modulation_3panel(m, np.arange(0.75, 1.3, 0.05), "Linear ramp")

m.run(ramp_type="groups")
plot_modulation_3panel(m, np.arange(0.75, 1.3, 0.05), "Wave group")

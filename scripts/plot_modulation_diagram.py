import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 4))  # Increased width to accommodate two periods

# Generate x values
x = np.linspace(0, 3.5 * np.pi, 2000)  # Extended to 4π, increased number of points

# Generate long wave
long_wave = np.sin(x)

# Generate short wave (shorter and more subtle)
short_wave_base = 0.05 * np.sin(20 * x)

# Modulate short wave amplitude based on long wave phase
modulation = 0.5 * (1 + long_wave)  # This will be 1 at peaks and 0 at troughs
short_wave = short_wave_base * modulation

# Combine waves
combined_wave = long_wave + short_wave

# Plot the combined wave
ax.plot(x, combined_wave, "k-", lw=3, zorder=1)

# Add surface velocity arrows
arrow_positions = np.arange(np.pi, 4 * np.pi, np.pi)
for i, pos in enumerate(arrow_positions):
    # Calculate wave slope at this position
    slope = np.cos(pos)  # Derivative of sin(x)
    angle = np.arctan(slope)

    # Calculate arrow components
    arrow_length = 0.6
    dx = arrow_length * np.cos(angle)
    dy = arrow_length * np.sin(angle)

    # Position arrow slightly below the wave
    y_offset = -0.25

    if i % 2 == 0:  # Even positions (arrows pointing inward)
        ax.arrow(
            pos - 1.6 * dx,
            np.sin(pos - 1.6 * dx) + y_offset,
            dx,
            dy * 0.9,
            head_width=0.2,
            head_length=0.2,
            fc="tab:blue",
            ec="tab:blue",
            width=0.1,
        )
        ax.arrow(
            pos + 1.6 * dx,
            np.sin(pos + 1.6 * dx) + y_offset,
            -dx,
            -dy,
            head_width=0.2,
            head_length=0.2,
            fc="tab:blue",
            ec="tab:blue",
            width=0.1,
        )
    else:  # Odd positions (arrows pointing outward)
        ax.arrow(
            pos - 0.4 * dx,
            np.sin(pos - 0.4 * dx) + y_offset,
            -dx,
            -dy * 0.9,
            head_width=0.2,
            head_length=0.2,
            fc="tab:blue",
            ec="tab:blue",
            width=0.1,
        )
        ax.arrow(
            pos + 0.4 * dx,
            np.sin(pos + 0.4 * dx) + y_offset,
            dx,
            dy,
            head_width=0.2,
            head_length=0.2,
            fc="tab:blue",
            ec="tab:blue",
            width=0.1,
        )

    ax.arrow(
        pos - 0.25 * np.pi,
        np.sin(pos - 0.25 * np.pi) - y_offset,
        1.5 * dx,
        1.5 * dy,
        head_width=0.12,
        head_length=0.2,
        fc="k",
        width=0.04,
    )

pos = arrow_positions[0]
ax.text(
    pos, np.sin(pos) - 0.3, r"$U$", fontsize=24, ha="right", va="top", color="tab:blue"
)

ax.text(
    pos - 0.05,
    np.sin(pos) + 1.1,
    r"$C_g$",
    fontsize=24,
    ha="right",
    va="top",
    color="k",
)

arrow_positions = np.arange(0.5 * np.pi, 3.5 * np.pi, np.pi)
for pos in arrow_positions:
    ax.arrow(
        pos,
        np.sin(pos) - 0.1,
        0,
        -0.5 * np.sin(pos),
        head_width=0.4,
        head_length=0.2,
        fc="tab:orange",
        ec="tab:orange",
        width=0.2,
        zorder=2,
    )

pos = arrow_positions[0]
ax.text(
    pos - 0.2,
    np.sin(pos) - 0.4,
    r"$\frac{\partial W}{\partial t}$",
    fontsize=28,
    ha="right",
    va="top",
    color="tab:orange",
)

# Add propagation arrow
ax.arrow(
    1.25 * np.pi,
    1.5,
    0.5 * np.pi,
    0,
    head_width=0.2,
    head_length=0.2,
    fc="k",
    ec="k",
    width=0.05,
)
ax.text(
    1.5 * np.pi, 1.75, r"$C_{pL}$", fontsize=24, ha="center", va="center", color="k"
)

# Keep axes visible but remove ticks
ax.tick_params(axis="both", which="both", length=0)
ax.set_xticks([])
ax.set_yticks([])

# Set equal aspect ratio
ax.set_aspect("equal", adjustable="box")

# Adjust plot limits
ax.set_xlim(0, 3.5 * np.pi)
ax.set_ylim(-1.5, 2)

plt.tight_layout()
plt.savefig("../figures/fig_hydrodynamic_modulation_diagram.pdf")
plt.savefig("../figures/fig_hydrodynamic_modulation_diagram.png", dpi=300)
plt.close()

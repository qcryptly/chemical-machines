# %% Cell 1 - Monte Carlo Pi Estimation with Animation
# Demonstrates Monte Carlo method to estimate Pi using random sampling

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from cm.views import html, image, log

log("Monte Carlo Pi Estimation", level="info")

html("""
<h2>Monte Carlo Estimation of Pi</h2>
<p>We estimate π by randomly throwing darts at a unit square containing a quarter circle.</p>
<p>The ratio of points inside the circle to total points approximates π/4.</p>
""")

# %% Cell 2 - Run the Simulation and Create Animation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tempfile
import os
from cm.views import html, image, log

log("Generating animation... (this may take a few seconds)", level="info")

# Simulation parameters
np.random.seed(42)
total_points = 1000
frames = 50
points_per_frame = total_points // frames

# Pre-generate all random points
all_x = np.random.uniform(0, 1, total_points)
all_y = np.random.uniform(0, 1, total_points)
all_inside = (all_x**2 + all_y**2) <= 1

# Create figure with dark theme
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#1e1e2e')

# Left plot: scatter of points
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_aspect('equal')
ax1.set_facecolor('#1e1e2e')
ax1.set_title('Random Points', fontsize=14, color='#cdd6f4')
ax1.set_xlabel('x', color='#cdd6f4')
ax1.set_ylabel('y', color='#cdd6f4')
ax1.tick_params(colors='#6c7086')

# Draw quarter circle
theta = np.linspace(0, np.pi/2, 100)
ax1.plot(np.cos(theta), np.sin(theta), color='#89b4fa', linewidth=2, label='Quarter circle')
ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='#6c7086', linewidth=1, linestyle='--')
ax1.legend(loc='upper right', facecolor='#313244', edgecolor='#45475a')

# Initialize scatter plots (empty)
inside_scatter = ax1.scatter([], [], c='#a6e3a1', s=8, alpha=0.7, label='Inside')
outside_scatter = ax1.scatter([], [], c='#f38ba8', s=8, alpha=0.7, label='Outside')

# Right plot: Pi estimate over time
ax2.set_xlim(0, total_points)
ax2.set_ylim(2.5, 4.0)
ax2.set_facecolor('#1e1e2e')
ax2.set_title('Pi Estimate Convergence', fontsize=14, color='#cdd6f4')
ax2.set_xlabel('Number of Points', color='#cdd6f4')
ax2.set_ylabel('Estimated π', color='#cdd6f4')
ax2.tick_params(colors='#6c7086')
ax2.axhline(y=np.pi, color='#f9e2af', linestyle='--', linewidth=2, label=f'Actual π = {np.pi:.6f}')
ax2.legend(loc='upper right', facecolor='#313244', edgecolor='#45475a')

# Line for pi estimates
line, = ax2.plot([], [], color='#89b4fa', linewidth=2)

# Text annotations
pi_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12,
                   color='#cdd6f4', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#313244', edgecolor='#45475a'))

# Storage for animation data
x_inside, y_inside = [], []
x_outside, y_outside = [], []
n_points_list = []
pi_estimates = []

def init():
    """Initialize animation."""
    inside_scatter.set_offsets(np.empty((0, 2)))
    outside_scatter.set_offsets(np.empty((0, 2)))
    line.set_data([], [])
    pi_text.set_text('')
    return inside_scatter, outside_scatter, line, pi_text

def animate(frame):
    """Update animation for each frame."""
    # Add new points for this frame
    start_idx = frame * points_per_frame
    end_idx = (frame + 1) * points_per_frame

    for i in range(start_idx, end_idx):
        if all_inside[i]:
            x_inside.append(all_x[i])
            y_inside.append(all_y[i])
        else:
            x_outside.append(all_x[i])
            y_outside.append(all_y[i])

    # Update scatter plots
    if x_inside:
        inside_scatter.set_offsets(np.column_stack([x_inside, y_inside]))
    if x_outside:
        outside_scatter.set_offsets(np.column_stack([x_outside, y_outside]))

    # Calculate and store pi estimate
    n_total = end_idx
    n_inside_total = len(x_inside)
    pi_estimate = 4 * n_inside_total / n_total

    n_points_list.append(n_total)
    pi_estimates.append(pi_estimate)

    # Update line plot
    line.set_data(n_points_list, pi_estimates)

    # Update text
    error = abs(pi_estimate - np.pi)
    pi_text.set_text(f'Points: {n_total}\n'
                     f'Inside: {n_inside_total}\n'
                     f'π ≈ {pi_estimate:.6f}\n'
                     f'Error: {error:.6f}')

    return inside_scatter, outside_scatter, line, pi_text

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
                     interval=100, blit=True)

# Save to temporary file, then read bytes
with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
    tmp_path = tmp.name

writer = PillowWriter(fps=10)
anim.save(tmp_path, writer=writer, dpi=100)
plt.close(fig)

# Read the GIF and display
with open(tmp_path, 'rb') as f:
    gif_bytes = f.read()
os.unlink(tmp_path)  # Clean up temp file

image(gif_bytes, mime_type='image/gif', alt='Monte Carlo Pi Estimation Animation')

log("Animation complete!", level="success")

# %% Cell 3 - Final Results

import numpy as np
from cm.views import html, log

# Calculate final statistics
np.random.seed(42)
n = 1000
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
inside = (x**2 + y**2) <= 1
n_inside = np.sum(inside)
pi_estimate = 4 * n_inside / n
error = abs(pi_estimate - np.pi)
error_percent = 100 * error / np.pi

html(f"""
<h3>Results Summary</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Points</td><td>{n:,}</td></tr>
<tr><td>Points Inside Circle</td><td>{n_inside:,}</td></tr>
<tr><td>Estimated π</td><td>{pi_estimate:.6f}</td></tr>
<tr><td>Actual π</td><td>{np.pi:.6f}</td></tr>
<tr><td>Absolute Error</td><td>{error:.6f}</td></tr>
<tr><td>Relative Error</td><td>{error_percent:.3f}%</td></tr>
</table>

<h3>The Math</h3>
<p>For a quarter circle of radius 1 inscribed in a unit square:</p>
<ul>
<li>Area of quarter circle = π/4</li>
<li>Area of unit square = 1</li>
<li>Ratio = π/4</li>
</ul>
<p>So: <strong>π ≈ 4 × (points inside) / (total points)</strong></p>

<p style="color: #a6e3a1;">As the number of points increases, the estimate converges to π by the Law of Large Numbers.</p>
""")

log(f"Final estimate: π ≈ {pi_estimate:.6f} (error: {error_percent:.3f}%)", level="success")

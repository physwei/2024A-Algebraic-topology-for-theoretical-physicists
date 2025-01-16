import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation


# Stereographic projection from S3 to R3
def proj_S3(theta, phi):
    U1_dummy_psi = np.linspace(0, 2 * np.pi, 100)
    U1_bundle_x = np.cos(theta / 2) * np.sin(U1_dummy_psi) / (1 - np.cos(theta / 2) * np.cos(U1_dummy_psi))
    U1_bundle_y = np.sin(theta / 2) * np.cos(phi + U1_dummy_psi) / (1 - np.cos(theta / 2) * np.cos(U1_dummy_psi))
    U1_bundle_z = np.sin(theta / 2) * np.sin(phi + U1_dummy_psi) / (1 - np.cos(theta / 2) * np.cos(U1_dummy_psi))
    return np.array(U1_bundle_x), np.array(U1_bundle_y), np.array(U1_bundle_z)


# Define and draw the S2 sphere
phi, theta = np.mgrid[0.0 : 2.0 * np.pi : 100j, 0.0 : np.pi : 50j]
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Define points on the reference sphere
n_points = 20
ref_phi = np.concatenate(
    (
        np.linspace(0, 2 * np.pi, n_points),
        np.linspace(0, 2 * np.pi, n_points),
        np.linspace(0, 2 * np.pi, n_points),
        np.linspace(0, 2 * np.pi, n_points),
        np.linspace(0, 2 * np.pi, n_points),
        np.ones(n_points) * 0,
        np.ones(n_points) * np.pi / 4,
        np.ones(n_points) * np.pi / 4 * 2,
        np.ones(n_points) * np.pi / 4 * 3,
    )
)
ref_theta = np.concatenate(
    (
        (np.pi / 3 + np.pi / 2) * np.ones(n_points),
        (np.pi / 4 + np.pi / 2) * np.ones(n_points),
        np.pi / 2 * np.ones(n_points),
        (-np.pi / 4 + np.pi / 2) * np.ones(n_points),
        (-np.pi / 3 + np.pi / 2) * np.ones(n_points),
        np.linspace(0 + np.pi / 10, 2 * np.pi - np.pi / 10, n_points),
        np.linspace(0 + np.pi / 10, 2 * np.pi - np.pi / 10, n_points),
        np.linspace(0 + np.pi / 10, 2 * np.pi - np.pi / 10, n_points),
        np.linspace(0 + np.pi / 10, 2 * np.pi - np.pi / 10, n_points),
    )
)
color_map = (
    ["tab:blue"] * n_points
    + ["tab:orange"] * n_points
    + ["tab:green"] * n_points
    + ["tab:red"] * n_points
    + ["tab:purple"] * n_points
    + ["tab:brown"] * n_points
    + ["tab:pink"] * n_points
    + ["tab:gray"] * n_points
    + ["tab:olive"] * n_points
)


total_frames = len(ref_phi)

ref_x = np.sin(ref_theta) * np.cos(ref_phi)
ref_y = np.sin(ref_theta) * np.sin(ref_phi)
ref_z = np.cos(ref_theta)


# Create a figure with two subplots
fig = plt.figure(figsize=(14, 7))

# Left subplot: original 3D sphere
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(x, y, z, color="tab:gray", alpha=0.2)
ax1.plot_wireframe(x, y, z, color="k", alpha=0.2, linewidth=0.4)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_zlim([-1.5, 1.5])


# Initialize the point on the equator for the left subplot
(point1,) = ax1.plot([ref_x[0]], [ref_y[0]], [ref_z[0]], "ro")

# Right subplot: 3D coordinates
ax2 = fig.add_subplot(122, projection="3d")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax2.set_zlim([-1.5, 1.5])


S3_x, S3_y, S3_z = proj_S3(ref_theta[0], ref_phi[0])


S3_updated_lines = []

# List to store positions of point1
point1_trace = []


ax1.view_init(elev=10, azim=90 + 10)
ax2.view_init(elev=0, azim=90)

clean_tag = True


# Update function for animation
def update(frame):
    global clean_tag
    point1.set_data([ref_x[frame]], [ref_y[frame]])
    point1.set_3d_properties([ref_z[frame]])
    point1.set_color(color_map[frame])
    if frame > 0:
        if color_map[frame] == color_map[frame - 1]:
            (trace,) = ax1.plot(
                [ref_x[frame - 1], ref_x[frame]],
                [ref_y[frame - 1], ref_y[frame]],
                [ref_z[frame - 1], ref_z[frame]],
                "-",
                markersize=2,
                color=color_map[frame],
            )
            point1_trace.append(trace)

    S3_x, S3_y, S3_z = proj_S3(ref_theta[frame], ref_phi[frame])
    (line,) = ax2.plot(S3_x, S3_y, S3_z, color=color_map[frame], alpha=0.6)
    S3_updated_lines.append(line)

    if frame > 0:
        if color_map[frame] != color_map[frame - 1]:
            for line in S3_updated_lines:
                line.set_alpha(0.2)
    if color_map[frame] == "tab:brown":
        if clean_tag:
            for line in S3_updated_lines:
                line.set_linewidth(0)
            clean_tag = False

    return [point1] + point1_trace + S3_updated_lines


# Create animation
ani = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)

# Save the animation
ani.save("hopf_fibration.mp4", writer="ffmpeg")
# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Parameters for the cube and cylinders
cube_size = 10
cylinder_radius = 1
cylinder_height = cube_size

# Create cube data
x = np.linspace(0, cube_size, 2)
y = np.linspace(0, cube_size, 2)
z = np.linspace(0, cube_size, 2)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Plot each face of the cube
for i in range(2):
    ax.plot_surface(X, Y, Z + i*cube_size, alpha=0.3, color='blue') # Top and bottom
    ax.plot_surface(X, Z + i*cube_size, Y, alpha=0.3, color='blue') # Front and back
    ax.plot_surface(Z + i*cube_size, X, Y, alpha=0.3, color='blue') # Sides

# Function to plot a cylinder
def plot_cylinder(position, radius, height):
    # Parameters
    z = np.linspace(0, height, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + position[0]
    y_grid = radius * np.sin(theta_grid) + position[1]
    z_grid = z_grid + position[2]

    # Plot
    ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.6)

# Coordinates for cylinders (assumed to be non-overlapping and evenly spaced)
num_cylinders = 11
# x_positions = np.linspace(cylinder_radius, cube_size - cylinder_radius, int(np.sqrt(num_cylinders)))
# y_positions = np.linspace(cylinder_radius, cube_size - cylinder_radius, int(np.sqrt(num_cylinders)))
# Calculate positions for 11 cylinders
x_positions = np.linspace(1.5, cube_size - 1.5, 4)
y_positions = np.linspace(1.5, cube_size - 1.5, 3)

# # Plot each cylinder
# for x_pos in x_positions:
#     for y_pos in y_positions:
#         if len(x_positions) * len(y_positions) > num_cylinders:
#             continue
#         plot_cylinder((x_pos, y_pos, 0), cylinder_radius, cylinder_height)

count = 0
for y_pos in y_positions:
    for x_pos in x_positions:
        if count < 11:
            plot_cylinder((x_pos, y_pos, 0), cylinder_radius, cylinder_height)
            count += 1

# Set limits and labels
ax.set_xlim(0, cube_size)
ax.set_ylim(0, cube_size)
ax.set_zlim(0, cube_size)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

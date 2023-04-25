import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import csv
import sys

# Get the base filename of the CSV files from command line arguments
if len(sys.argv) != 2:
    print("Usage: python script.py <csv_base_filename>")
    sys.exit(1)
csv_base_filename = sys.argv[1]

data = np.loadtxt('{}_0.csv'.format(csv_base_filename), delimiter=',')
rows, cols = data.shape

# Create a meshgrid for the x and y coordinates
dx = 1.0 / cols
x = np.arange(cols)*dx + 0.5*dx
y = np.arange(rows)*dx + 0.5*dx
X, Y = np.meshgrid(x, y)

# Range of data
vmin = -0.1
vmax = 0.5

# Create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a function to update the plot for each frame
def update(frame):
    # Load the height field data from the CSV file
    csv_filename = '{}_{:d}.csv'.format(csv_base_filename, frame)
    with open(csv_filename) as f:
        reader = csv.reader(f)
        data = np.array([list(map(float, row)) for row in reader])

    # Clear the previous plot
    ax.clear()

    # Create a new plot of the height field
    surf = ax.plot_surface(X, Y, data, cmap='terrain', vmin=vmin, vmax=vmax)

    # Set the x, y, and z axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    
    # Set the z-axis limits
    ax.set_zlim(vmin, vmax)

    # Set equal aspect ratio for all the axes
    ax.set_box_aspect((1, 1, vmax - vmin))

    return surf,


# Create an animation using the update function and the range of frames
anim = FuncAnimation(fig, update, frames=range(24), interval=500)

# Show the animation
plt.show()

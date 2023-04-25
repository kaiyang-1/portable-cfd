import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys

# Get the filename of the CSV file from command line arguments
if len(sys.argv) != 2:
    print("Usage: python script.py <csv_filename>")
    sys.exit(1)
csv_filename = sys.argv[1]

# Load the CSV file into a NumPy array
with open(csv_filename) as f:
    reader = csv.reader(f)
    data = np.array([list(map(float, row)) for row in reader])\
    

# Create a meshgrid for the x and y coordinates
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

# Create a 3D plot of the height field data as a mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, data, cmap='terrain')

# Set the x, y, and z axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
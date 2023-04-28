import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV file
data = np.loadtxt('cavity_flow.csv', delimiter=',')

# Get the number of rows and columns
rows, cols = data.shape

# Set the grid size
dh = 1 / cols
x = np.arange(0, cols, 1) * dh + 0.5*dh
y = np.arange(0, rows, 1) * dh + 0.5*dh

# Create the grid
X, Y = np.meshgrid(x, y)

# Plot the heatmap
plt.pcolormesh(X, Y, data, clim=[0, 1])

# Add a colorbar
plt.colorbar()

# Set the axis labels
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()
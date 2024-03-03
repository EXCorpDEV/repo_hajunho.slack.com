import matplotlib.pyplot as plt
import numpy as np

# Define the equation y = 2x over a range of x values
x = np.linspace(-10, 10, 400)
y = 2 * x

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = 2x')
plt.title('Graph of y = 2x')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()

# Show the plot
plt.show()


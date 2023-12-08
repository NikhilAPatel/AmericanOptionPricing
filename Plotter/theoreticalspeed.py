import matplotlib.pyplot as plt
import numpy as np

# Define the range of x values
x_values = np.arange(1, 17)

# Define the functions
y1 = 2 * x_values
n = 100
y2 = n * (1 / (2*x_values))

# # Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x_values, y1, label='2P_s')
plt.plot(x_values, y2, label='100 * (1/2P_s)')

# Adding labels and title
plt.xlabel('P_s')
plt.ylabel('Training Samples', color='orange')  # Orange y-axis label on the left
plt.tick_params(axis='y', labelcolor='orange')
plt.title('Plot of Training Samples and Speedup Factor for Dynamic Approach Thread Split with P=16')
plt.legend()

ax1 = plt.gca()  # Get the current axes
ax2 = ax1.twinx()  # Create a twin of the first y-axis
ax2.set_ylim(ax1.get_ylim())  # Synchronize the y-axis limits with the first y-axis
ax2.set_ylabel('Speedup Factor', color='blue')  # Blue y-axis label on the right
ax2.tick_params(axis='y', labelcolor='blue')

# Show the plot
plt.show()


# Creating two subplots to display the functions side by side

# plt.figure(figsize=(12, 6))
#
# # First plot for 2x
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
# plt.plot(x_values, y1, 'b-', label='2x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of 2x')
# plt.legend()
#
# # Second plot for 10000 * (1/2)x
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
# plt.plot(x_values, y2, 'r-', label='10000 * (1/2)x')
# plt.xlabel('x')
# plt.title('Plot of 10000 * (1/2)x')
# plt.legend()
#
# # Adjust the layout
# plt.tight_layout()
#
# # Show the plots
# plt.show()

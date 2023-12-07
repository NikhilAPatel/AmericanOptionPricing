import matplotlib.pyplot as plt

# Data
# Figure 3
data = [
    (1, -0.20308, 26913),
    (2, -0.25318, 7183),
    (3, -0.26354, 4333),
    (5, -0.298868, 2103),
    (6, -0.32605, 1867)

]




# Unpacking the data
num_threads, percent_error, execution_time = zip(*data)
absolute_error = [abs(error)*10 for error in percent_error]

# Plotting
plt.figure(figsize=(12, 6))

# Error vs. Number of Threads
plt.subplot(1, 2, 1)
plt.plot(num_threads, absolute_error, marker='o')
plt.xlabel('Target Ratio of Completed Simulations to Completed Regressions')
plt.ylabel('Percent Error')
plt.title('Error vs. Ratio')

# Time vs. Number of Threads
plt.subplot(1, 2, 2)
plt.plot(num_threads, execution_time, marker='o', color='r')
plt.xlabel('Target Ratio of Completed Simulations to Completed Regressions')
plt.ylabel('Execution Time (ms)')
plt.title('Time vs. Ratio')

plt.tight_layout()
plt.show()


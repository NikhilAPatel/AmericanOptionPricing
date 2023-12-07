import matplotlib.pyplot as plt

# Data
# This data is actually used in Fig 1
# data = [
#     (1, -2.21431, 5048),
#     (2, -2.83119, 2638),
#     (3, -3.12073, 1779),
#     (4, -3.21656, 1283),
#     (5, -3.20717, 1031),
#     (6, -3.25029, 803),
#     (7, -3.28994, 671),
#     (8, -3.7242, 535)
# ]

# Used in Fig 2
# data = [
# (1, -2.32229, 30178),
# (2, -2.36586, 15987),
# (3, -2.38218, 10372),
# (4, -2.38684, 6933),
# (5, -2.56316, 4592),
# (6, -2.81656, 3061),
# (7, -3.00717, 2437),
# (8, -3.08994, 2023),
# (9, -3.2242, 1510),
# (10, -3.25832, 1112),
# (11, -3.20289, 919),
# (12, -3.26432, 809),
# (13, -3.21526, 709),
# (14, -3.26377, 635),
# (15, -3.25198, 568),
# (16, -3.72391, 498)]

data = [
(1, 1.60996, 48170),
(2, 1.65319, 22849),
(3, 1.60046, 12782),
(4, 1.69382, 8990),
(5, 1.71183, 5980),
(6, 2.74806, 4420),
(7, 3.09324, 3061),
(8, 3.08395, 2566),
(9, 3.32647, 1974),
(10, 3.11019, 1779),
(11, 3.28921, 1189),
(12, 2.98394, 1047),
(13, 2.93351, 810),
(14, 3.2181, 674),
(15, 3.5844, 556),
(16, -3.99382, 474)]



# Unpacking the data
num_threads, percent_error, execution_time = zip(*data)
absolute_error = [abs(error) for error in percent_error]

# Plotting
plt.figure(figsize=(12, 6))

# Error vs. Number of Threads
plt.subplot(1, 2, 1)
plt.plot(num_threads, absolute_error, marker='o')
plt.xlabel('Number of Simulator Threads')
plt.ylabel('Percent Error')
plt.title('Error vs. Number of Threads')

# Time vs. Number of Threads
plt.subplot(1, 2, 2)
plt.plot(num_threads, execution_time, marker='o', color='r')
plt.xlabel('Number of Simulator Threads')
plt.ylabel('Execution Time (ms)')
plt.title('Time vs. Number of Threads')

plt.tight_layout()
plt.show()


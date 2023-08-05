import numpy as np
import matplotlib.pyplot as plt

# Create a random array with 50 elements between 0 and 1
arr1 = np.random.rand(50)

# Create another random array with 50 elements between -1 and 1
arr2 = np.random.uniform(-1, 1, 50)

# Plot the arrays using a scatter plot
plt.scatter(range(len(arr1)), arr1, color='red', label='Array 1')
plt.scatter(range(len(arr2)), arr2, color='blue', label='Array 2')

# Add title and labels to the plot
plt.title('Random Arrays')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Display the plot
plt.savefig('/projects/p31961/ENIGMA/tests/testfig.png')
#git test
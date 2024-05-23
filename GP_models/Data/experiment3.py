import numpy as np
import matplotlib.pyplot as plt


# Data
y = np.array([-377431.49566605, -998659.91375769, -799017.60278674,
              4053.03061004, -6256.1055222,   -14636.69734688])

# Create a bar plot
plt.scatter(range(len(y)), y, marker='x')

# Add labels and title
plt.xlabel('Chord number')
plt.ylabel('Log Marginal Likelihood')


plt.title('(Positive) Log Marginal Likelihoods with changing chords')

# Show the plot
plt.show()

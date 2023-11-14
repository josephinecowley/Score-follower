# This analyses the data found in an experiment measuring the Log Marginal Liklihood
# The parameters used were:
# amplitude=i; f = 349; M=16; sigma_f=5; v = 2.37; T = 0.465; B=0.0005;
# Each likelihood was calculated using 1000 samples, the total time being 0:35:36 and 183.8 s/it
# THough note there were periods when the computer was off - so roughly 8s/it or a total of 2 hours

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 40, 2)
y = np.array([-6616.71985856,  3393.41419535,  3369.54016245,  3355.25895294,  3344.95663184,  3336.85563549,  3330.15888949,  3324.43873361,  3319.43849959,  3314.9917696,
             3310.98434923,  3307.33440082,  3303.98122395,  3300.87852203,  3297.99015888,  3295.28737469,  3292.7468968,   3290.34962058,  3288.07966534,  3285.92368387])

# Add grid to see spacing of samples
for i in x:
    plt.axvline(x=i, color='grey', linestyle='-', lw=0.2)
# Add grid lines at each y-axis tick
plt.grid(axis='y', linestyle='-', alpha=0.7)

plt.plot(x, y)
plt.title("Log Marginal Likelihood for amplitude of piano at 349 Hz")
plt.xlabel("Amplitude of single source")
plt.ylabel("Log Marginal Likelihood")
plt.yticks(np.arange(min(y), max(y+1000), step=500))


# Get the current figure
fig = plt.gcf()

# Save the figure using np.save()
plt.savefig(f'{"Experiment 2"}')
plt.show()

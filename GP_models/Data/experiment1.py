# This analyses the data found in an experiment measuring the Log Marginal Liklihood
# The parameters used were:
# f = i; M=16; sigma_f=5; v = 2.37; T = 0.465; (B is varying, depending on f)
# Each likelihood was calculated using 1000 samples, the total time being 3:35:36 and 183.8 s/it
# THough note there were periods when the computer was off - so roughly 8s/it or a total of 2 hours

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 700, 10)
y = np.array([-6436.99967859, -5254.72148193,  2520.03153891,  2566.02995066,  2619.85472278,  2638.32714955,  2620.04099794,  2659.45498421,  2663.23902724,  3003.87872488,  2263.20237976,  2622.98353495,  2635.33146208,  1218.21286661, -1493.68289283, -161.41726575,  2099.71775863,  3100.96314869,  2698.88537354,  1878.04338819, -589.59453926, -4371.95067632, -5202.92712621, -4443.28498284, -5007.33591832, -5120.78474143, -5470.11388316, -5374.89492009, -4275.24670211, -3132.54984221, -952.22266436,  1114.38241818,  2065.35763684,  2478.84715618,  2925.80809043,
             3298.21748723,  2786.42031529,  2414.46583146,  1835.06296321,   697.68147883, -1178.4578828, -3604.51099061, -5257.37626082, -5828.19843586, -5688.57555724, -5314.04906089, -5395.18605016, -5678.75443781, -6120.51010046, -6198.53040065, -6168.81605894, -6351.78218051, -6416.96639952, -6388.68257472, -6390.93163574, -6412.32696378, -6398.83036153, -6566.22704647, -6576.5348222, -6575.98168527, -6651.23297328, -6699.95604686, -6634.40864798, -6507.49626223, -6509.63289731, -6592.05258874, -6603.94793305, -6429.97789609, -6234.52171557, - 6123.73906197])

# Add grid to see spacing of samples
for i in x:
    plt.axvline(x=i, color='grey', linestyle='-', lw=0.2)
# Add grid lines at each y-axis tick
plt.grid(axis='y', linestyle='-', alpha=0.7)

plt.plot(x, y)
plt.title("Log Marginal Likelihood for frequency of piano at 349 Hz")
plt.xlabel("Fundamental frequency of single source")
plt.ylabel("Log Marginal Likelihood")
plt.yticks(np.arange(min(y), max(y+1000), step=500))

# Add lines to indicate the fundamental frequencies and all those which are also half
for i in range(4):
    plt.axvline(x=349/2**i, color='pink', linestyle='--')

# Get the current figure
fig = plt.gcf()

# Save the figure using np.save()
plt.savefig(f'{"Experiment 1"}')
plt.show()

# creating features for labs and grey hounds
import numpy as np
import matplotlib.pyplot as plt

#samples
greyhounds = 500
labs = 500

#feature data (height and eye color)
# statistics is necessary here 
lab_height = 24 + 4 * np.random.randn(greyhounds)
greyhound_height = 28 + 4 * np.random.randn(labs)


plt.hist([lab_height,greyhound_height],stacked = True, color = ['r','b'])
plt.show()

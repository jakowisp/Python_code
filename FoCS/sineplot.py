import math
import numpy as np
import matplotlib.pyplot as plt


x = range(0,360)
np_x = np.array(x)
np_x = 3.14 / 180 * np_x
y = []
for i in np_x: 
    y.append(math.sin(i))
np_y = np.array(y)

x= [1,2,3,4,5,6]
y= [1,1,2,3,5,8]
plt.plot(np_x,np_y)
plt.title("Sin(x)")
plt.show()

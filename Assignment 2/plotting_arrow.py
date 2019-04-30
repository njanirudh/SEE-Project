import matplotlib.pyplot as plt
import numpy as np

A = np.array([10,23])
B = np.array([20,30])
plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], head_width=3, length_includes_head=True, head_length=4.0, 
                  fc='red', ec='black')
plt.xlim(-30,50)
plt.ylim(-30,50)
plt.show()

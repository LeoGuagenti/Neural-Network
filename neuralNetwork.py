import numpy as np

def sigmoid(x, deriv=False):
    if deriv == True:
        return (x * (1 - x))
    else:
        return 1 / (1 + np.exp(-x))


np.random.seed(1)
print(2 * np.random.random((3,4)) - 1) 
print("----------------")
print(2 * np.random.random((4,1)) - 1) 
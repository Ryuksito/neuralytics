import numpy as np
class StandardScaler():
    
    def __init__(self, x)-> np.ndarray:
        return self.normalize(x)

    def normalize(x:np.ndarray)-> np.ndarray:
        x_mean = np.mean(x)
        x_std = np.std(x)
        return np.array((x-x_mean)/x_std)
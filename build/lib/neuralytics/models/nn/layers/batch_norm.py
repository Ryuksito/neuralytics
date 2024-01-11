import numpy as np

class BatchNorm: 
    
    def __init__(self, input_shape: tuple):
        self.y = None   # gamma
        self.B = None   # beta
        self.e = None   # epsilon
    
    def __str__(self):
        return f"\nDense Layer:\nWeights:\n{self.w}\nBiases:\n{self.b}"

    def __call__(self, x: np.ndarray):
        self.a = x.reshape(-1,self.flattened_shape)
        return self.a
import numpy as np

class Sigmoid: 

    Z='z'
    nType = 'act'
    layer_name = 'relu'
    
    def __init__(self):
        super().__init__()
        self.a:np.ndarray = None

    def __str__(self):

        return f"\nReLU Layer"

    def __call__(self, z: np.ndarray):
        self.a = 1 / (1 + np.exp(-z))
        return self.a
    
    def get_weights(self):
        return None

    def set_weights(self, parameters:tuple):
        pass
    
    def df(z: np.ndarray, respect: str = Z):
        if respect == Sigmoid.Z or respect == Sigmoid.Z.upper():
            a = Sigmoid(z)
            return a * (1 - a)
        else:
            raise Exception(f'Error: la función no puede derivarse con respecto a {respect}')
    
    def backward(self, lr:float, deltaL_1: np.ndarray, aL_1:np.ndarray, daL_1_dzL_1:np.ndarray, l:int):
        if l != 0:
            return (deltaL_1, None, None)
        else:
            return (None, None, None)


import numpy as np

class ReLu: 

    Z='z'
    nType = 'act'
    layer_name = 'relu'
    
    def __init__(self):
        super().__init__()
        self.a:np.ndarray = None
    
    def __str__(self):

        return f"\nReLU Layer"

    def __call__(self, z: np.ndarray):
        self.a = np.maximum(z, 0)
        return self.a
    
    def get_weights(self):
        return None

    def set_weights(self, parameters:tuple):
        pass
    
    def df(self, z:np.ndarray, respect:str=Z):
        if respect == ReLu.Z or respect == ReLu.Z.upper():
            return np.where(z <= 0, 0, 1)
        else: 
            raise Exception(f'error the funtion cant derivate respetc to {respect}')
    
    def backward(self, lr:float, deltaL_1: np.ndarray, aL_1:np.ndarray, daL_1_dzL_1:np.ndarray, l:int):
        if l != 0:
            return (deltaL_1, None, None)
        else:
            return (None, None, None)



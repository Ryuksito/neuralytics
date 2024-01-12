import numpy as np

class ReLU:
    Z='z'
    nType = 'act'
    def __new__(cls, z=None):
        if z is not None:
            return np.maximum(z, 0)
        return 'ReLU'
    
    @staticmethod
    def df(z:np.ndarray, respect:str=Z):
        if respect == ReLU.Z or respect == ReLU.Z.upper():
            return np.where(z <= 0, 0, 1)
        else: 
            raise Exception(f'error the funtion cant derivate respetc to {respect}')
        

class Sigmoid:
    Z = 'z'
    nType = 'activation'

    def __new__(cls, z=None):
        if z is not None:
            return 1 / (1 + np.exp(-z))
        return 'Sigmoid'

    @staticmethod
    def df(z: np.ndarray, respect: str = Z):
        if respect == Sigmoid.Z or respect == Sigmoid.Z.upper():
            a = Sigmoid(z)
            return a * (1 - a)
        else:
            raise Exception(f'Error: la función no puede derivarse con respecto a {respect}')
import numpy as np
import pdb

class BatchNorm: 

    nType = 'layer'
    layer_name = 'batchnorm'
    
    def __init__(self, neurons:int, e:np.float32, name:str='BatchNorm'):
        self.y = np.ones((1, neurons))   # gamma
        self.B = np.zeros((neurons,1))   # beta
        self.e = e   # epsilon
        self.name = name
        self.a: np.ndarray = None
        self.xi_hat = None
        self.std = None
        self.mean = None
    
    def __str__(self):
        return f"\BatchNorm Layer: {self.name}\nGamma:\n{self.y.shape}\nBeta:\n{self.B.shape}"

    def __call__(self, x: np.ndarray):

        self.std = np.std(x)
        self.mean = np.mean(x)
        self.xi_hat = (x - self.mean)/self.std + self.e
        self.a = self.xi_hat @ self.y.T + self.B

        return self.a
    
    def get_weights(self):
        return self.y, self.B

    def set_weights(self, parameters:tuple):
        self.y = parameters[0]
        self.B = parameters[1]
    
    def backward(self, lr:float, deltaL_1: np.ndarray, aL_1:np.ndarray, daL_1_dzL_1:np.ndarray, l:int):

        # calculo del error
        pdb.set_trace()
        ey =  deltaL_1.T @ self.xi_hat
        eB = deltaL_1
        eB = np.sum(eB, axis=0, keepdims=True)

        y = self.y - ey.T * lr
        B = self.B - eB * lr

        if l != 0: 
            d_std = (aL_1 - self.mean) * ((1/2)*((self.std + self.e)**(-3/2)))
            d_mean = (-1/self.std + self.e) + d_std * -2 * (aL_1 - self.mean)
            deltaL_1 = (deltaL_1 @ self.y.T) * (1/self.std + self.e) + d_std * (2*(aL_1-self.mean)) + d_mean
        else: 
            deltaL_1 = None

        return (deltaL_1, y, B)
    

# def fn(funct:callable):
#     if funct is callable:
#         print('is callable', funct(np.array([[1,2,3]])))
#     else:
#         print('is not callable', funct(np.array([[1,2,3]])))



    

# fn(BatchNorm)
# print('------------------------------------------------')
# fn(BatchNorm())

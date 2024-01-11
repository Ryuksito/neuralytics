import numpy as np

class MSE:
    A ='a'
    Y = 'y'
    def __new__(cls, y:np.ndarray=None, a:np.ndarray=None):
        if y is not None and a is not None:
            return np.mean(np.sum((y-a)**2))
        return 'MSE'
    
    @staticmethod
    def df(y:np.ndarray=None, a:np.ndarray=None, respect:str=A):
        if respect == MSE.A or respect == MSE.A.upper():
            return 2*(a-y)
        elif respect == MSE.Y or respect == MSE.Y.upper():
            return 2*(y-a)
        else: 
            raise Exception(f'error the funtion cant derivate respetc to {respect}')
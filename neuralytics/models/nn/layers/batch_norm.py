import numpy as np

class BatchNorm: 

    nType = 'layer'
    
    def __init__(self, input_shape:tuple, neurons:int, e:np.float32, name:str='BatchNorm'):
        self.y = np.random.randn(np.prod(input_shape), neurons) / np.sqrt(np.prod(input_shape)/2)   # gamma
        self.B = np.zeros((1,neurons))   # beta
        self.e = e   # epsilon
        self.name = name
    
    def __str__(self):
        return f"\nDense Layer:\nGamma:\n{self.y}\nBeta:\n{self.B}"

    def __call__(self, x: np.ndarray):

        xi = (x - np.mean(x))/np.std(x) + self.e
        yi = xi @ self.y + self.B

        return yi
    

def fn(funct:callable):
    if funct is callable:
        print('is callable', funct(np.array([[1,2,3]])))
    else:
        print('is not callable', funct(np.array([[1,2,3]])))



    

fn(BatchNorm)
print('------------------------------------------------')
fn(BatchNorm())

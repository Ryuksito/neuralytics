import numpy as np
from .abstract_layer import AbstractLayer


class Dense(AbstractLayer):

    nType = 'layer'
    layer_name = 'dense'

    def __init__(self, input_shape:tuple, neurons:int, name:str='Relu'):
        super().__init__(input_shape, neurons)
        self.name = name
        
    
    def __str__(self):
        return f"\nDense Layer: {self.name}\nw_shape: {self.w.shape}\nb_shape: {self.b.shape}"
        

    def __call__(self, x:np.ndarray):
        
        flattened_input = x.reshape(-1, np.prod(self.input_shape))

        # Calcular la entrada a la capa
        self.z = flattened_input @ self.w + self.b
        
        self.a = self.z
        return self.a
    
    def backward(self, lr:float, deltaL_1:np.ndarray, aL_1:np.ndarray, daL_1_dzL_1:np.ndarray, i:int) -> tuple: # (deltaL_1,w,b)
        # calculo del error
        ew =  deltaL_1.T @ aL_1
        eb = deltaL_1
        eb = np.sum(eb, axis=0, keepdims=True)

        w = self.w - ew.T * lr
        b = self.b - eb * lr

        #calculo de delta
        if i != 0:
            deltaL_1 = (deltaL_1 @ self.w.T) * daL_1_dzL_1
        else: 
            deltaL_1 = None
        
        return  (deltaL_1, w, b)
    
    
    

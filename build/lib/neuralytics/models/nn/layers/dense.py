import numpy as np
from .abstract_layer import AbstractLayer


class Dense(AbstractLayer):

    def __init__(self, input_shape:tuple, neurons:int, activation:callable=None, name:str='Relu'):
        super().__init__(input_shape, neurons, activation)
        self.name = name
        
    
    def __str__(self):
        return f"\nDense Layer: {self.name}\nw_shape: {self.w.shape}\nb_shape: {self.b.shape}"
        

    def __call__(self, x:np.ndarray):
        
        flattened_input = x.reshape(-1, np.prod(self.input_shape))

        # Calcular la entrada a la capa
        self.z = flattened_input @ self.w + self.b

        # Aplicar la función de activación
        if(self.activation != None): 
            self.a = self.activation(self.z)
            return self.a
        
        self.a = self.z
        return self.a
    
    def backward(self, lr:float, deltaL_i:np.ndarray, aL_i:np.ndarray, wL:np.ndarray, daL_i_dzL_i:np.ndarray, i:int) -> tuple: # (deltaL_1,w,b)
        # calculo del error
        ew =  deltaL_i.T @ aL_i
        eb = deltaL_i
        eb = np.sum(eb, axis=0, keepdims=True)

        w = self.w - ew.T * lr
        b = self.b - eb * lr

        #calculo de delta
        if i != 0:
            deltaL_i = (deltaL_i @ wL) * daL_i_dzL_i
        else: 
            deltaL_i = None
        
        return  (deltaL_i, w, b)
    
    

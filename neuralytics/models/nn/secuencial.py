from typing import Any
import numpy as np
from .layers import Flatten, Dense, AbstractLayer
from tqdm import tqdm
import pdb

# python -m unittest tests/test_nn.py

class Secuencial:
    def __init__(self, layers:list=[AbstractLayer]) -> None:
        self.layers:list[AbstractLayer]  = layers
        self.loss_function:callable = None

    def compile(self, loss_function:callable=None):
        if loss_function is None: 
            raise Exception(f'loss function is not valid: {loss_function}')
        self.loss_function = loss_function

    def fit(self, x:np.ndarray, y: np.ndarray, epochs: int=10, lr: int=0.01, batch_size: int=32, verbose: bool=False):

        # pdb.set_trace()

        for epoch in range(epochs):
            yp:np.ndarray
            #batch loop
            with tqdm(range(0, x.shape[0], batch_size), desc='Epoch: ' + str(epoch), disable=verbose) as progress_bar:
                for i in progress_bar:
                    temp_wb = []
                    yi = np.array(y[i:i+batch_size], dtype=np.float64)
                    xi = np.array(x[i:i+batch_size], dtype=np.float64)
                    yp   = self._predict(xi)

                    # pdb.set_trace()
                    


                    temp_wb = self._backpropagation(lr, xi, yi)
                    self._update(temp_wb)
                    loss = self.loss_function(y=yi, a=self._predict(xi))
                    progress_bar.set_postfix(loss=f'{loss}' if loss is not None else 'None')


    
    def _backpropagation(self, lr, x, y):
        #backpropagation
        temp_wb = []
        deltaL_1 = self.loss_function.df(y,self.layers[-1].a) * (self.layers[-1].activation.df(self.layers[-1].z) if hasattr(self.layers[-1], 'activation') and self.layers[-1].activation != None else np.ones(self.layers[-1].a.shape))
        
        for i,l in enumerate(reversed(self.layers)):

            i = len(self.layers)- 1 - i

            if not isinstance(l, (AbstractLayer, Dense)): 
                temp_wb.append((None,None))
                continue

            aL_1 = (self.layers[i-1].a if i != 0 else x) # aL-i
            wL = l.w.T
            daL_1_dzL_1 = (self.layers[i-1].activation.df(self.layers[i-1].z) if hasattr(self.layers[i-1], 'activation') and self.layers[i-1].activation != None else np.ones(self.layers[i-1].a.shape))

            deltaL_1, w, b = l.backward(lr, deltaL_1, aL_1, wL, daL_1_dzL_1, i) # learning rate, delta, 


            temp_wb.append((w,b))

        return temp_wb
                
    def _update(self, wb:list):
        for i,l in enumerate(reversed(self.layers)):
          if not isinstance(l, (AbstractLayer, Dense)): 
              continue
          l.set_weights(wb[i][0],wb[i][1])
        pass           

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x:np.ndarray) -> np.ndarray:
        a = x

        for layer in self.layers:
            a = layer(a)

        return a
    
    def _predict(self, x:np.ndarray) -> np.ndarray:
        a:np.ndarray = x
        for layer in self.layers:

            a = layer(a)

        return a
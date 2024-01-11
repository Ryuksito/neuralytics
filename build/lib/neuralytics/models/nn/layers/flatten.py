import numpy as np
from .abstract_layer import AbstractLayer

class Flatten():

    def __init__(self, input_shape: tuple, name:str='Flatten'):
        if not isinstance(input_shape, (tuple, list)): raise Exception('input_shape must be a tuple or list object')
        elif(len(input_shape) <= 1): raise Exception('one-dimensional arrays cannot be flattened')
        elif(len(input_shape) == 2): 
            self.flattened_shape = np.prod(input_shape)
        self.flattened_shape = np.prod([c for c in input_shape])
        self.a:np.ndarray = None
        self.name = name
    
    def __str__(self):
        return f"\nFlatten Layer: {self.name}\nshape: {self.flattened_shape}"

    def __call__(self, x: np.ndarray):
        self.a = x.reshape(-1,self.flattened_shape)
        return self.a

import numpy as np

class AbstractLayer:

    def __init__(self, input_shape: tuple, neurons: int):
        self.input_shape = input_shape
        self.neurons = neurons

        self.w = np.random.randn(np.prod(input_shape), neurons) / np.sqrt(np.prod(input_shape)/2)
        self.b = np.zeros((1,neurons))
        self.z: np.ndarray = None
        self.a:np.ndarray = None

    def __call__(self, x: np.ndarray):
        raise NotImplementedError

    def get_weights(self):
        return self.w, self.b

    def set_weights(self, parameters:tuple):
        self.w = parameters[0]
        self.b = parameters[1]
import numpy as np
import random

class LinearRegression():

    def __init__(self):
        self.w: np.ndarray
        self.b: np.ndarray
        self.x_mean: np.int64
        self.x_std: np.int64
        self.y_mean: np.int64
        self.y_std: np.int64

    def fit(self, x:np.ndarray, y: np.ndarray, epochs: int=10, lr: int=0.01, batch_size: int=32, oneline_logs: bool=True):
        #inicializar parametros aleatorios
        self.w = np.array([random.uniform(-1,1) for _ in range(x.shape[1]) ], dtype=np.float64)
        self.b = np.array(random.randint(-1,1), dtype=np.float64)


        # normalization
        self.x_mean = np.mean(x, axis=0)
        self.y_mean = np.mean(y)
        self.x_std = np.std(x, axis=0)
        self.y_std = np.std(y)

        x = self._normalize(x, self.x_mean, self.x_std)
        y = self._normalize(y, self.y_mean, self.y_std)

        count = 0
        data = np.empty((2, 0))
        mse:np.ndarray = np.array(0, dtype=np.float64)
        char = 0
        # bucle de epocas
        for epoch in range(epochs):
            yp:np.ndarray
            loadbar = 0
            chars = ['\\', '|', '/', '-']
            

            # iteracion de batches
            for i in range(0,x.shape[0],batch_size):
                yi = np.array(y[i:i+batch_size], dtype=np.float64)
                xi = np.array(x[i:i+batch_size], dtype=np.float64)

                yp = self._predict(xi)
                mse = self._mse(yi,yp, yi.shape[0])
                self.w -= np.array(self._dw(xi,yi,yp,yi.shape[0]) * lr, dtype=np.float64)
                self.b -= np.array(self._db(yi,yp,yi.shape[0]) * lr, dtype=np.float64)
                
                data = np.hstack((data, np.array([[mse], [count]])))
                count = count + 1
                

                if i >= int(x.shape[0]/50) * loadbar:  
                    loadbar += 1
            
                char += 1
                if char >= 4:
                    char = 0
                print(f'\033[F\033[K', end='')
                print(f'Epoch: {epoch}, {"="*loadbar}>{":"*(50-loadbar)}, [{chars[char]}] mse: {mse:0.4}')
            not oneline_logs and print('\n')

        return data


    def predict(self, x:np.ndarray):
        x = self._normalize(x, self.x_mean, self.x_std)
        return self._unnormalize(np.dot(x, self.w) + self.b, self.y_mean, self.y_std)
        pass

    def _predict(self,x:np.ndarray):
        return np.dot(x,self.w) + self.b

    def _mse(self, y:np.ndarray, yp:np.ndarray, n:int):
        return np.array((1/n)*np.sum((y-yp)**2), dtype=np.float64)

    def _dw(self, x:np.ndarray, y:np.ndarray, yp:np.ndarray, n:int):
        return np.array((-2/n) * np.sum((y-yp)[:, np.newaxis] * x, axis=0), dtype=np.float64)

    def _db(self, y:np.ndarray, yp:np.ndarray, n:int):
        return np.array((-2/n) * np.sum(y- yp, axis=0), dtype=np.float64)

    def _normalize(self, x:np.ndarray, mean:np.int64, std:np.int64):
        # print(f'x: {x}, mean: {mean}, std: {std}, r: {(x - mean) / std}')
        return (x - mean) / std

    def _unnormalize(self, x:np.ndarray, mean:np.int64, std:np.int64):
        return (x * std) + mean
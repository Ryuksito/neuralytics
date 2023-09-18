import numpy as np
import random

class SimpleLinearRegression():

    def __init__(self) -> None:
        self.w: np.ndarray = np.array(random.uniform(a=-1, b=1))
        self.b: np.ndarray = np.array(random.uniform(a=-1, b=1))
        self.x_mean: np.float64
        self.x_std: np.float64
        self.y_mean: np.float64
        self.y_std: np.float64
        self.x: np.ndarray
        self.y: np.ndarray
        self.yp: np.ndarray
        self.n: np.ndarray
        self.lr: np.ndarray

    def _mse(self, y:np.ndarray, yp:np.ndarray, n:np.int64):
        # print(f'y: {y.shape}, yp: {yp.shape}, n: {n}, {(np.int8(1)/n)} /{np.sum((y - yp) * (y - yp))} / {(y - yp) ** 2} / ')
        return np.array((np.int8(1)/n) * np.sum((y - yp) * (y - yp)))

    def _dw(self, x:np.ndarray, y:np.ndarray, yp:np.ndarray, n:np.int64):
        return np.array((-2/n) * np.sum(x*(y - yp)),dtype=np.float64)

    def _db(self, y:np.ndarray, yp:np.ndarray, n:np.int64):
        return np.array((-2/n) * np.sum(y- yp))
    
    def _create_batches(self, data:np.ndarray, batch_size:int=32):
        data_length = data.shape[0]
        num_batches = int(np.ceil(data_length / batch_size))
        # print(np.array_split(data, num_batches))
        return np.array(np.array_split(data, num_batches),dtype=np.float64)

    def predict(self, x:np.ndarray):
        if(type(x) != np.ndarray): raise Exception('parameter x is not an instance of ndarray')
        x = self._normalize(x,mean=self.x_mean, std=self.x_std)
        return self._unnormalize((self.w*x) + self.b,std=self.y_std,mean=self.y_mean)
    
    def _predict(self, x:np.ndarray):
        if(type(x)!= np.ndarray): raise Exception('parameter x is not an instance of ndarray')
        return (self.w*x) + self.b

    def _normalize(self, x, mean:np.float64, std: np.float64):
        return (x - mean) / std

    def _unnormalize(self,x, std:np.float64, mean:np.float64):
        print(f'mean: {mean}, std: {std}')
        print(f'unnorm{x}: ', (x * std) + mean)
        return (x * std) + mean

    def fit(self, x:np.ndarray, y:np.ndarray, epochs:int=10, lr:float=0.01, batch_size:int=32):
        if(x.shape[0] != y.shape[0]): raise Exception('length of "x" and "y" are different')
        self.n = x.shape[0]
        self.lr = np.array(lr)
        self.x = x
        self.y = y
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        x = self._normalize(x, self.x_mean, self.x_std)
        y = self._normalize(y,self.y_mean, self.y_std)

        # y:np.ndarray = self._create_batches(data=self.y,batch_size=batch_size)
        # x: np.ndarray = self._create_batches(data=self.x,batch_size=batch_size)
        # print('shape: ',x.shape)
        count = 0
        data = np.empty((2, 0))
        # Iterear las epocas
        for epoch in range(0,epochs,1):
            mse:float = 0
            yp:np.ndarray
            # Iterar los bathces de datos
            loadbar = 0
            chars = ['\\', '|', '/', '-']
            char = 0
            for i in range(0,x.shape[0],batch_size):
                yi = np.array(y[i:i+batch_size])
                xi = np.array(x[i:i+batch_size])

                yp = self._predict(xi)
                mse = self._mse(y=yi, yp=yp, n=yi.shape[0])
                self.w -= self._dw(x=xi, y=yi, yp=yp, n=xi.shape[0]) * self.lr
                self.b -= self._db(y=yi, yp=yp, n=xi.shape[0]) * self.lr
                data = np.hstack((data, np.array([[mse], [count]])))
                count = count + 1
                
                # print(i,int(x.shape[0]/50) * loadbar, i >= int(x.shape[0]/50) * loadbar)
                if i >= int(x.shape[0]/50) * loadbar:
                    char += 1
                    if char >= 4:
                        char = 0
                    loadbar += 1
                
                # print(loadbar)
                print(f'\033[F\033[K', end='')
                print(f'Epoch: {epoch}, {"="*loadbar}>{":"*(50-loadbar)}, [{chars[char]}] mse: {mse:0.4}')
            print('\n')    
            # print(f'{"-"*50}{"-"*50}\n')

        return data

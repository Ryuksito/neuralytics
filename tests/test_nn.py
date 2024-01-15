import unittest
from neuralytics.models import *
from matplotlib import pyplot as plt
import pdb

# For run tests
# python -m unittest tests/test_nn.py

class TestSimpleLinearRegression(unittest.TestCase):

    def test_nn_cf(self):
        import numpy as np
        from neuralytics.models.nn import Secuencial
        from neuralytics.models.nn.layers import Dense, ReLu, BatchNorm
        from neuralytics.models.nn import MSE


        
        #layers

        dense1 = Dense(input_shape=(1), neurons=3, name='dense1')


        dense2 = Dense(input_shape=(1,3), neurons=2, name='dense2')
        batchNorm2 = BatchNorm(neurons=2, e=1e-8, name='batchNorm2')


        dense3 = Dense(input_shape=(1,2), neurons=1, name='dense3')

        print(f'Initial state: \n{dense1}\n{dense2}\n{batchNorm2}\n{dense3}')

        

        #data
        data_len = 10
        data_step = 5
        x = np.array([((i-20)) for i in range(0,data_len*data_step,data_step)]).reshape(data_len,1)
        y = np.array([((i*9) / 5) + 32 for i in x]).reshape(data_len,1)
        

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std
        # desviacion_estandar = 0.1  
        # ruido = np.random.normal(0, desviacion_estandar, y.shape)
        # y = y + ruido



        model = Secuencial(layers=[
            dense1, 
            dense2, 
            batchNorm2,
            ReLu(),
            dense3
        ])

        model.compile(loss_function=MSE)
        model.fit(x=x, y=y,epochs=1,lr=0.01)
        print(f'final state: \n{dense1}\n{dense2}\n{batchNorm2}\n{dense3}')


        print(x)
        print('--------------------------------')
        print(y)
        while True:
            print('En farenheits: ', ((model.predict(((np.array(float(input('Introduce un numero en grados celcius: '))).reshape(1,1))-x_mean)/x_std))*y_std)+y_mean)

    # def test_nn_images(self):
    #     import tensorflow as tf

    #     import numpy as np
    #     from neuralytics.models.nn import Secuencial
    #     from neuralytics.models.nn.layers import Dense, Flatten
    #     from neuralytics.models.nn import ReLU, Sigmoid, MSE
    #     from neuralytics.preprocessing import OneHotEncoder
        

    #     # Cargar el conjunto de datos MNIST (dígitos escritos a mano)
    #     mnist = tf.keras.datasets.mnist

    #     # Cargar los datos
    #     (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #     train_labels = OneHotEncoder.transform(labels=train_labels)
    #     train_images = train_images/255
    #     test_images = test_images/255

    #     # pdb.set_trace()

    #     flatten1 = Flatten((28,28))

    #     dense1 = Dense(activation=ReLU, input_shape=(784), neurons=10)


    #     dense2 = Dense(activation=ReLU, input_shape=(1,10), neurons=15)


    #     dense3 = Dense(input_shape=(1,15), neurons=10)

    #     model = Secuencial(layers=[flatten1, dense1, dense2, dense3])

    #     model.compile(loss_function=MSE)

    #     model.fit(x=train_images, y=train_labels,epochs=250,lr=0.001, batch_size=32)

    #     val_predicted = model.predict(train_images)
    #     acurrancy = [ 1 if np.argmax(val) == np.argmax(train_labels[i]) else 0 for i, val in enumerate(val_predicted)]

    #     print(f'prediccion: {acurrancy.count(1)}')
    #     print(f'cantidad total: {len(acurrancy)}')
    #     print(f'porcentaje: {(acurrancy.count(1)/len(acurrancy))*100}')
    #     print('datos de validaion')
        
    #     val_predicted = model.predict(test_images)
    #     acurrancy = [ 1 if np.argmax(val) == np.argmax(test_labels[i]) else 0 for i, val in enumerate(val_predicted)]

    #     print(f'prediccion: {acurrancy.count(1)}')
    #     print(f'cantidad total: {len(acurrancy)}')
    #     print(f'porcentaje: {(acurrancy.count(1)/len(acurrancy))*100}')




if __name__ == '__main__':
    unittest.main()
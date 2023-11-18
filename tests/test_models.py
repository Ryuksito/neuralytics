import unittest
from neuralytics.models import *
from matplotlib import pyplot as plt

# For run tests
# python -m unittest discover tests

class TestSimpleLinearRegression(unittest.TestCase):
    # def test_predict(self):
    #     with self.assertRaises(Exception):
    #         model = SimpleLinearRegression(1)
    #     model = SimpleLinearRegression()
    #     print(model.w, model.b)

    # def test_suma(self):
    #     resultado = 3 + 2
    #     self.assertEqual(resultado, 5)

    # def test_fit(self):
    #     model = SimpleLinearRegression()
    #     x = np.array([i for i in range(100)], dtype=np.float64)
    #     y = np.array([(i*12)+19 for i in range(100)],dtype=np.float64)
    #     mse,epoch = model.fit(x,y,10000,0.0001,10)

    #     # x = np.array([i for i in range(50)], dtype=np.float64)
    #     # y = np.array([(i*2)+5 for i in range(50)],dtype=np.float64)

    #     # mse,epoch = model.fit(x,y,40,0.001,4)
    #     print(model.predict(np.array(1)))
    #     plt.plot(epoch, mse, color='red', label='Línea')
    #     plt.xlabel('epochs')
    #     plt.ylabel('mse')
    #     plt.title('Gráfico de dispersión: epochs vs mse')
    #     plt.legend()
    #     plt.show()

    def test_newfit(self):
        import numpy as np
        import random
        from matplotlib import pyplot as plt
        from neuralytics.models import SimpleLinearRegression

        x = np.array([x+random.uniform(random.uniform(-100,0),random.uniform(0,100)) for x in range(30000)])
        y = np.array([(5*xi-7)+random.uniform(random.uniform(-100,0),random.uniform(0,100)) for xi in x])

        plt.figure(figsize=(12, 6))

        # Subgráfico 1: Datos originales
        plt.subplot(1, 2, 1)
        plt.scatter(x,y, marker='o', linestyle='-', color='b')
        plt.title('Datos Originales')
        plt.xlabel('Índice de Datos')
        plt.ylabel('Valores')

        # Mostrar los gráficos
        plt.tight_layout()
        plt.show()

        model1 = SimpleLinearRegression()
        mse,epoch = model1.fit(x,y,100,0.01,32)
        plt.plot(epoch, mse, color='red', label='Línea')
        plt.xlabel('epochs')
        plt.ylabel('mse')
        plt.title('Gráfico de dispersión: epochs vs mse')
        plt.legend()
        plt.show()
        print('\n\n')

        p1 = model1.predict(np.array(4))

        print('prediction: ', p1)


    def test_linear_regresion(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import random

        x = np.array([[x+random.uniform(random.uniform(-100,0),random.uniform(0,100)) for x in range(3)] for _ in range(random.randint(2,100000))])
        print(x.shape)
        w = np.random.rand(x.shape[1])
        print(w.shape)
        y = np.array([np.dot(xi,w) - 7 for xi in x])


        exit()

        model = LinearRegression()
        mse,epoch = model.fit(batch_size=1000, epochs=100, lr=0.001, x=x, y=y, oneline_logs=not True)
        plt.plot(epoch, mse, color='red', label='Línea')
        plt.xlabel('epochs')
        plt.ylabel('mse')
        plt.title('Gráfico de dispersión: epochs vs mse')
        plt.legend()
        plt.show()

        print(x[0])
        n = int(input('introdusca la cantidad de caractreristicas: '))
        p1 = model.predict(np.array([float(input()) for _ in range(n)]))
        print('---')
        print(p1)
        print('---')
        print(y[0])

    def test_nn(self):
        from neuralytics.models.nn import Secuencial

if __name__ == '__main__':
    unittest.main()
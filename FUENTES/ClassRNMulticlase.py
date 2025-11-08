import numpy as np
import time

from matplotlib import pylab as plt
from IPython import display

from grafica import *

class RNMulticlase(object):
    """
    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    cotaE : float
        minimum error threshold
    FUN : string
        activation function: 'sigmoid', 'tanh', otherwise linear
    random_state : int
        Random number generator seed for random weight initialization.
        
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, alpha=0.01, n_iter=50, cotaE=10e-07, FUN='sigmoid', COSTO='ECM', random_state=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.cotaE = cotaE
        self.FUN = str(FUN)  # ← Asegurar que sea string
        self.COSTO = str(COSTO)  # ← Asegurar que sea string
        self.random_state = random_state

    def mostrar_progreso(self, epoca, info_adicional=""):
        """
        Muestra una barra de progreso del entrenamiento.
        
        Parameters
        ----------
        epoca : int
            Número de época actual (base 0)
        info_adicional : str, optional
            Información adicional para mostrar (ej: "Error: 0.023" o "1/1000")
        """
        porcentaje = (epoca + 1) / self.n_iter * 100
        barra_len = 30
        progreso = int(barra_len * (epoca + 1) / self.n_iter)
        barra = '█' * progreso + '░' * (barra_len - progreso)
        print(f'\rÉpoca {epoca+1}/{self.n_iter} |{barra}| {porcentaje:.1f}% - {info_adicional}', end='', flush=True)


    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples, n_class]
            Target values (instances created with one-hot-encoder)
        Returns
        -------
        self : object
        """

        # Asegurar que X e y sean arrays de tipo float
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)  

        rgen = np.random.RandomState(self.random_state)

        # self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])
        nRow = X.shape[0]  # cantidad de ejemplos
        nIn = X.shape[1]  # cantidad de atributos de entrada ---
        nOut = y.shape[1] # -- cantidad de neuronas de salida (deben ser por lo menos 2)

        self.w_ = np.random.uniform(-0.5, 0.5, [nOut, nIn])
        self.b_ = np.random.uniform(-0.5, 0.5, [nOut,1])

        self.errors_ = []
        self.accuracy_ = []
        ErrorAnt = 0
        ErrorAct = 1
        
        i = 0
        while ((i<self.n_iter) and (np.abs(ErrorAnt- ErrorAct) > self.cotaE)):
            ErrorAnt = ErrorAct
            ErrorAct = 0
            for e in range(nRow):
                
                xi = X[e:e+1,:]
                
                salida = self.predict_nOut(xi).T
                errorXi = (y[e:e+1, :].T - salida)

                # Caso especial: Softmax + EC tienen derivada simplificada
                if (self.FUN == 'softmax' and self.COSTO == 'EC'):
                    update = self.alpha * errorXi
                else:
                    update = self.alpha * errorXi * self.derivar(salida)
                    
                self.w_ += update * xi
                self.b_ += update

                ErrorAct += self.fCosto(y[e:e+1, :].T , salida)
                
            ErrorAct = ErrorAct / nRow    
            self.errors_.append(ErrorAct)
            self.accuracy_.append(self.accuracy(X,y))
            
            # Mostrar progreso
            self.mostrar_progreso(i, f"Error {self.COSTO}: {ErrorAct:.6f}")
                       
            i = i + 1
        
        print()  # Salto de línea al finalizar
        return self

    def fCosto(self,y, y_hat):
        EPS = np.finfo(float).eps
        if (self.COSTO=='ECM'):
            return(np.sum((y-y_hat)**2))
        if (self.COSTO=='EC_binaria'):
            return(np.sum(-y*np.log(y_hat+EPS)-(1-y)*np.log(1-y_hat+EPS)))
        if (self.COSTO=='EC'):
            return(np.sum(-y*np.log(y_hat+EPS)))
               
    def net_input(self, X):
        """Calculate net input"""
        netas = self.w_ @ X.T + self.b_
        return netas.T
    
    def evaluar(self, x):
        if (self.FUN=='tanh'):
            return (2.0 / (1+np.exp(-2*x)) - 1)
        elif (self.FUN=='sigmoid'):
            return (1.0/(1+np.exp(-x)))
        elif (self.FUN=='softmax'):
            return (np.exp(x)/(np.sum(np.exp(x),axis=1).reshape(-1,1)))
        else:
            return(x)
        
    def derivar(self,x):
        if (self.FUN=='tanh'):
            return (1-x**2)
        elif (self.FUN=='sigmoid'):
            return (x*(1-x))
        else:
            return(1)    

    def predict_nOut(self, X):
        """Return class label after unit step"""
        return self.evaluar(self.net_input(X))
    
    def predict(self, X):
        """Retorna un entero con el ìndice de la clase más probable """
        y_hat = self.predict_nOut(X)
        if (self.FUN=='tanh'):
            y_hat = (y_hat>0)*1
        if (self.FUN=='sigmoid'):
            y_hat = (y_hat>0.5)*1

        return(np.argmax(y_hat,axis=1))
            
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        OK = np.sum(np.argmax(y,axis=1)==y_pred)
        return (OK/X.shape[0])

    def save(self, archivo):
        np.savez(archivo, matriz1=self.w_, matriz2=self.b_)

    def load(self, archivo):
        with np.load(archivo) as data:
            claves = list(data.keys())
            if (len(claves)!=2):
                printf("ERROR --> Formato de archivo incorrecto")

            self.w_ = data['matriz1']
            self.b_ = data['matriz2']
from aim_machine_learning.base_regressor import Regressor
import numpy as np

class MultipleRegressor(Regressor):

    def __init__(self, a, b):

        self.a=a
        self.b=b
    
    def fit(self, X_train, y_train):

        self.X=X_train
        self.y=y_train
    
    def predict(self, X_test):

        a_new=np.array(self.a).reshape(-1) #occorre fare il reshape per garantire tutti i casi possibili
        return (np.dot(X_test, a_new)+self.b).reshape(-1)
    
    def __add__(self, model2): #creiamo il metodo somma, che come predizione avra come bias la somma dei bias e come a la concatenazione di a1 e a2

        a3=[[self.a], [model2.a]]
        b3=self.b+model2.b
        
        model3=MultipleRegressor(a3, b3)

        return model3
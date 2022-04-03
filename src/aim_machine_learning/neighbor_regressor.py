import numpy as np

from aim_machine_learning.base_regressor import Regressor
from sklearn.neighbors import KNeighborsRegressor

class NeighborRegressor(Regressor):
    
    def __init__(self, k=1, **params):  
        self.k=k
        self.status_fit=0
        super().__init__(**params)
        
    def fit(self, X, y):
        self.X_train=X
        self.y_train=y
        
        self.status_fit=1
    
    def predict(self, X_test): 
        
        if self.status_fit==0:
            raise NameError('Non si è allenato il modello, non è possibile chiamare predict')
        
        n_test=X_test.shape[0]
        n_train=self.X_train.shape[0]
        
        predictions=np.zeros(n_test)
        
        
        
        for i in range(n_test):
            
            distances=np.zeros(n_train)
        
            for j in range(n_train):
                
                if X.shape[1]
                distances[j]=self.distance(self.X_train[j, :], X_test[i, :])
            
           
            indices=np.argpartition(distances ,self.k)[0:self.k]
            
                
            predictions[i]=np.mean(self.y_train[indices])


        return predictions
            
            
            
        
    
    def distance(self, x1, x2):
        
        return np.sqrt(np.sum((x1-x2)**2))


class MySklearnNeighborRegressor(KNeighborsRegressor, Regressor):

    pass
        
        
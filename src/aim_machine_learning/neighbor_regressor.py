import numpy as np

from aim_machine_learning.base_regressor import Regressor
from sklearn.neighbors import KNeighborsRegressor

class NeighborRegressor(Regressor):
    
    def __init__(self, k=1, **params):  
        self.k=k
        self.status_fit=0
        super().__init__(**params)
        
    def fit(self, X, y): #il modello dei knn non ha bisogno di un vero è proprio fit, in quanto basa la sua predizione esclusivamente sui valori che si sono osservati nel training set
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
                
                
                distances[j]=self.distance(self.X_train[j, :], X_test[i, :]) #per ogni sample, calcoliamo la sua distanza da ogni training set
            
           
            indices=np.argpartition(distances ,self.k)[0:self.k] #consideriamo solo i k punti piu vicini (prendiamo gli indici)
            
                
            predictions[i]=np.mean(self.y_train[indices]) #la predict sara semplicemente la media tra i valori osservati nei k punti piu vicini


        return predictions
            
            
            
        
    
    def distance(self, x1, x2):
        
        return np.sqrt(np.sum((x1-x2)**2))


class MySklearnNeighborRegressor(KNeighborsRegressor, Regressor):

    pass
        
        
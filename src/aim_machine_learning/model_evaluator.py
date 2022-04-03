import numpy as np

class ModelEvaluator():
    
    def __init__(self, model_class, params, X, y):
        
        self.model_class=model_class(**params) #i due asterischi cercano se nel dizionario c'è la chiave k
        self.X=X
        self.y=y
    
    def train_test_split_eval(self, eval_obj, test_proportion):
        
        n_samples=self.X.shape[0]
        test_samples=int(n_samples*test_proportion)
        
        X_test=self.X[0:test_samples, :]
        Y_test=self.y[0:test_samples]
        X_train=self.X[test_samples:, :]
        Y_train=self.y[test_samples:]
        
        
        self.model_class.fit(X_train, Y_train)
        Y_pred=self.model_class.predict(X_test)
        
        return eval_obj(y_true=Y_test, y_pred=Y_pred)
    
    def kfold_cv_eval(self, eval_obj, K):
        
        test_samples=int(self.X.shape[0]/K)
        
        for i in range(K):

            if i==0:
                self.model_class.fit(self.X[test_samples:, :], self.y[test_samples:])
                Y_pred=self.model_class.predict(self.X[0:test_samples, :])
                eval_k=eval_obj(y_true=self.y[0:test_samples], y_pred=Y_pred)
            
            if i!=0 and i!=K-1:
                X_train=np.concatenate((self.X[0:i*test_samples, :], self.X[(i+1)*test_samples:, :]), axis=0)
                Y_train=np.concatenate((self.y[0:i*test_samples], self.y[(i+1)*test_samples:]), axis=0 )
                self.model_class.fit(X_train, Y_train)
                Y_pred=self.model_class.predict(self.X[i*test_samples:(i+1)*test_samples, :])
                eval_k=self.add_dict(eval_k, eval_obj(y_true=self.y[i*test_samples:(i+1)*test_samples], y_pred=Y_pred))
                
            
            if i==K-1:
                self.model_class.fit(self.X[0:(K-1)*test_samples, :], self.y[0:(K-1)*test_samples])
                Y_pred=self.model_class.predict(self.X[(K-1)*test_samples:, :])
                eval_k=self.add_dict(eval_k, eval_obj(y_true=self.y[(K-1)*test_samples:], y_pred=Y_pred))
                
        return self.divide_dict(eval_k, K)
        
    
    def add_dict(self, dict1, dict2):
    
        keys=dict1.keys()
        values1=np.array(list(dict1.values()))
        values2=np.array(list(dict2.values()))
        values=values1+values2
                     
        return dict(zip(keys, list(values)))
    
    
    def divide_dict(self, dict1, n):
    
        keys=dict1.keys()
        values=np.array(list(dict1.values()))
        
        for i in range(len(values)):
            values[i]=round(values[i]/n, 2)
                     
        return dict(zip(keys, list(values)))
            
            
            
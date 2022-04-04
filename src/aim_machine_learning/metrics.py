import numpy as np

class Evaluator():
    
    def __init__(self, supported_metrics):
        
        self.supported_metrics=supported_metrics
    
    def set_metric(self, new_metric):
        
        if new_metric not in self.supported_metrics:  #se la metrica non è tra quelle che indichiamo, alziamo un errore
            raise NameError('{} non e\' una metrica supportata'.format(new_metric))
        
        else:
            self.current_metric=new_metric #altrimenti, settiamo la metrica
        
        return self 
        
    def __repr__(self):
        return 'Current metric is {}'.format(self.current_metric)
        
    def __call__(self, y_true, y_pred): #rendiamo il nostro oggetto un oggetto callable, in ogni caso restituiamo un dizionario che contiene le statistiche di interesse per la valutazione
        
        if self.current_metric=='mse':
            
            keys=['mean', 'std']
            mean=round(np.mean((y_true-y_pred)**2), 2)
            std=round(np.std((y_true-y_pred)**2), 2)
            
            return {'mean': mean, 'std':std}
        
        if self.current_metric=='mae':
            
            keys=['mean', 'std']
            mean=round(np.mean(abs(y_true-y_pred)), 2)
            std=round(np.std(abs(y_true-y_pred)), 2)
            
            return {'mean': mean, 'std':std}
        
        if self.current_metric=='corr':
            
            corr=round(np.corrcoef(y_true, y_pred)[0][1], 2)
            
            return {'corr':corr}
            
    
    
   
    
    
    
    